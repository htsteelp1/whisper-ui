import gradio as gr
import shutil
import tempfile
import os
import wave
import struct
import numpy as np
from faster_whisper import WhisperModel

loaded_models = {}

# ─── Model loading ────────────────────────────────────────────────────────────

def get_model(model_name, device, compute_type, log):
    key = (model_name, device, compute_type)
    if key not in loaded_models:
        log(f"[INFO] Loading model '{model_name}' on {device} ({compute_type})...")
        try:
            loaded_models[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
            log(f"[OK]   Model '{model_name}' loaded and ready.")
        except Exception as e:
            if device == "cuda":
                log(f"[WARN] CUDA failed: {e}")
                log(f"[INFO] Falling back to CPU with int8...")
                loaded_models[key] = WhisperModel(model_name, device="cpu", compute_type="int8")
                log(f"[OK]   Model loaded on CPU (int8).")
            else:
                raise
    else:
        log(f"[INFO] Using cached model '{model_name}'.")
    return loaded_models[key]


# ─── Save mic audio (numpy array → wav temp file) ─────────────────────────────

def save_mic_to_tempfile(mic_data):
    """mic_data is (sample_rate, np.ndarray) from gr.Audio type='numpy'"""
    sample_rate, audio = mic_data
    if audio.ndim > 1:
        audio = audio[:, 0]  # take first channel if stereo
    audio = audio.astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return tmp.name


# ─── Language map ─────────────────────────────────────────────────────────────

LANGUAGE_MAP = {
    "Auto-detect": None, "English": "en", "Spanish": "es",
    "French": "fr", "German": "de", "Italian": "it",
    "Portuguese": "pt", "Dutch": "nl", "Russian": "ru",
    "Chinese": "zh", "Japanese": "ja", "Korean": "ko",
    "Arabic": "ar", "Hindi": "hi", "Turkish": "tr",
}


# ─── Main transcription (generator for live UI updates) ───────────────────────

def run(audio_file, mic_audio, direct_path, model_name, language_label, device, compute_type, progress=gr.Progress()):
    console_lines = []

    def log(msg):
        print(msg)
        console_lines.append(msg)

    # ── Resolve input file ──
    file_path = None
    use_temp = False

    if direct_path and direct_path.strip():
        file_path = direct_path.strip()
        log(f"[INFO] Using direct path: {file_path}")
        if not os.path.exists(file_path):
            yield "", f"File not found: {file_path}", "", "\n".join(console_lines)
            return

    elif mic_audio is not None:
        log("[INFO] Saving microphone recording to temp file...")
        file_path = save_mic_to_tempfile(mic_audio)
        use_temp = True
        log(f"[OK]   Mic audio saved.")

    elif audio_file is not None:
        raw = audio_file if isinstance(audio_file, str) else (audio_file.get("path") or audio_file.get("name", ""))
        log(f"[INFO] Copying uploaded file...")
        ext = os.path.splitext(raw)[-1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            shutil.copy2(raw, tmp.name)
            file_path = tmp.name
        use_temp = True
        log(f"[OK]   File ready.")

    else:
        yield "", "No input provided.", "", "No audio provided. Upload a file, record from mic, or paste a path."
        return

    # Yield console state so user sees progress immediately
    yield "", "", "", "\n".join(console_lines)

    # ── Load model ──
    try:
        progress(0, desc="Loading model...")
        model = get_model(model_name, device, compute_type, log)
        yield "", "", "", "\n".join(console_lines)
    except Exception as e:
        log(f"[ERROR] Failed to load model: {e}")
        yield "", "Error loading model", "", "\n".join(console_lines)
        if use_temp and file_path:
            os.unlink(file_path)
        return

    # ── Transcribe ──
    lang_code = LANGUAGE_MAP.get(language_label, None)
    log(f"[INFO] Starting transcription (lang={lang_code or 'auto'}, beam=5, VAD=on)...")
    yield "", "", "", "\n".join(console_lines)

    try:
        progress(0.05, desc="Detecting language...")

        segments_gen, info = model.transcribe(
            file_path,
            language=lang_code,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        detected_lang = info.language.upper()
        duration = info.duration or 1
        log(f"[INFO] Detected language: {detected_lang} (duration: {duration:.1f}s)")
        yield "", f"Detected: {detected_lang}", "", "\n".join(console_lines)

        full_text = ""
        segments_text = ""
        seg_count = 0
        last_end = 0.0

        for seg in segments_gen:
            seg_count += 1
            full_text += seg.text + " "
            last_end = seg.end
            m_s, s_s = divmod(seg.start, 60)
            m_e, s_e = divmod(seg.end, 60)
            seg_line = f"[{int(m_s):02d}:{s_s:04.1f} → {int(m_e):02d}:{s_e:04.1f}]  {seg.text.strip()}"
            segments_text += seg_line + "\n"

            pct = min(0.05 + 0.93 * (last_end / duration), 0.98)
            progress(pct, desc=f"Transcribing... {last_end:.0f}s / {duration:.0f}s")

            log(f"[SEG {seg_count:03d}] {seg_line}")

            # Yield live updates every segment
            yield full_text.strip(), f"Detected: {detected_lang}", segments_text.strip(), "\n".join(console_lines)

        progress(1.0, desc="Done!")
        log(f"[OK]   Transcription complete. {seg_count} segments, {len(full_text.split())} words.")
        yield full_text.strip(), f"Detected: {detected_lang}", segments_text.strip(), "\n".join(console_lines)

    except RuntimeError as e:
        if "cublas" in str(e).lower() or "cuda" in str(e).lower():
            log(f"[WARN] CUDA error: {e}")
            log(f"[INFO] Retrying on CPU with int8...")
            yield "", "", "", "\n".join(console_lines)
            try:
                cpu_model = WhisperModel(model_name, device="cpu", compute_type="int8")
                log(f"[OK]   CPU model loaded. Retrying transcription...")
                yield "", "", "", "\n".join(console_lines)

                segments_gen, info = cpu_model.transcribe(
                    file_path,
                    language=lang_code,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                detected_lang = info.language.upper() + " (CPU fallback)"
                duration = info.duration or 1
                full_text = ""
                segments_text = ""
                seg_count = 0
                for seg in segments_gen:
                    seg_count += 1
                    full_text += seg.text + " "
                    m_s, s_s = divmod(seg.start, 60)
                    m_e, s_e = divmod(seg.end, 60)
                    seg_line = f"[{int(m_s):02d}:{s_s:04.1f} → {int(m_e):02d}:{s_e:04.1f}]  {seg.text.strip()}"
                    segments_text += seg_line + "\n"
                    pct = min(0.05 + 0.93 * (seg.end / duration), 0.98)
                    progress(pct, desc=f"Transcribing (CPU)... {seg.end:.0f}s / {duration:.0f}s")
                    log(f"[SEG {seg_count:03d}] {seg_line}")
                    yield full_text.strip(), f"Detected: {detected_lang}", segments_text.strip(), "\n".join(console_lines)

                progress(1.0, desc="Done!")
                log(f"[OK]   Done on CPU. {seg_count} segments.")
                yield full_text.strip(), f"Detected: {detected_lang}", segments_text.strip(), "\n".join(console_lines)

            except Exception as e2:
                log(f"[ERROR] CPU fallback also failed: {e2}")
                yield "", "Error", "", "\n".join(console_lines)
        else:
            log(f"[ERROR] {e}")
            yield "", "Error", "", "\n".join(console_lines)

    except Exception as e:
        log(f"[ERROR] Unexpected error: {e}")
        yield "", "Error", "", "\n".join(console_lines)

    finally:
        if use_temp and file_path and os.path.exists(file_path):
            os.unlink(file_path)


# ─── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Faster Whisper Transcription") as demo:
    gr.Markdown("# ⚡ Faster Whisper — Local Transcription\n**GPU-accelerated · ~4x faster · 100% offline · Live output**")

    with gr.Row():
        # ── Left column: inputs ──
        with gr.Column(scale=1):
            with gr.Tab("Upload File"):
                audio_input = gr.File(
                    label="Audio / Video File",
                    file_types=["audio", "video", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mkv", ".mp4"],
                )
            with gr.Tab("Record Mic"):
                mic_input = gr.Audio(
                    label="Record from Microphone",
                    sources=["microphone"],
                    type="numpy",
                )
            with gr.Tab("File Path"):
                direct_path_input = gr.Textbox(
                    label="Paste full file path",
                    placeholder=r"C:\Users\Henry\Music\lecture.wav",
                    info="Fastest option — reads directly from disk, no upload.",
                )

            with gr.Row():
                model_input = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                    value="base",
                    label="Model",
                    info="Larger = more accurate, slower",
                )
                device_input = gr.Dropdown(
                    choices=["cuda", "cpu"],
                    value="cuda",
                    label="Device",
                )

            compute_input = gr.Dropdown(
                choices=["float16", "int8_float16", "int8"],
                value="float16",
                label="Compute Type",
                info="float16=fastest GPU · int8_float16=less VRAM · int8=CPU",
            )
            language_input = gr.Dropdown(
                choices=list(LANGUAGE_MAP.keys()),
                value="Auto-detect",
                label="Language",
            )
            submit_btn = gr.Button("Transcribe →", variant="primary", size="lg")

        # ── Right column: outputs ──
        with gr.Column(scale=2):
            detected_out = gr.Textbox(label="Detected Language", interactive=False, max_lines=1)
            text_out = gr.Textbox(
                label="Transcription",
                lines=10,
                placeholder="Transcript streams here as it's processed…",
            )
            segments_out = gr.Textbox(
                label="Timestamped Segments",
                lines=10,
                placeholder="Segments appear here in real time…",
            )
            console_out = gr.Textbox(
                label="Console Output",
                lines=8,
                placeholder="Logs will appear here…",
                interactive=False,
                show_label=True,
            )

    submit_btn.click(
        fn=run,
        inputs=[audio_input, mic_input, direct_path_input, model_input, language_input, device_input, compute_input],
        outputs=[text_out, detected_out, segments_out, console_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", theme=gr.themes.Monochrome())