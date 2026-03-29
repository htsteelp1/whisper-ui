import gradio as gr
import shutil
import tempfile
import os
import wave
import time
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


# ─── Mic audio helper ─────────────────────────────────────────────────────────

def save_mic_to_tempfile(mic_data):
    sample_rate, audio = mic_data
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return tmp.name


# ─── Progress bar HTML ────────────────────────────────────────────────────────

def make_progress_html(pct: float, label: str, done: bool = False) -> str:
    pct_clamped = max(0.0, min(1.0, pct))
    pct_display = int(pct_clamped * 100)
    bar_color = "#2ecc71" if done else "#3b82f6"
    border_color = "#1a7a3c" if done else "#1d4ed8"
    transition = "none" if pct_clamped == 0 else "width 0.3s ease"
    return f"""
<div style="font-family: monospace; font-size: 13px; color: #ccc; margin-bottom: 4px;">
  {label}
</div>
<div style="background:#1e1e1e; border:1px solid #333; border-radius:4px; height:22px; overflow:hidden;">
  <div style="
    width: {pct_display}%;
    height: 100%;
    background: {bar_color};
    border-right: {('2px solid ' + border_color) if pct_clamped < 1.0 else 'none'};
    transition: {transition};
    display: flex; align-items: center; justify-content: flex-end; padding-right: 6px;
  ">
    <span style="color: {'#000' if pct_clamped > 0.15 else '#999'}; font-size: 11px; font-weight: bold;">
      {pct_display}%
    </span>
  </div>
</div>
"""

PROGRESS_IDLE = make_progress_html(0, "Idle — press Transcribe to start")


# ─── Subtitle format builders ─────────────────────────────────────────────────

def fmt_srt_time(s):
    h, m = int(s // 3600), int((s % 3600) // 60)
    sec, ms = int(s % 60), int(round((s - int(s)) * 1000))
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

def fmt_vtt_time(s):
    h, m = int(s // 3600), int((s % 3600) // 60)
    sec, ms = int(s % 60), int(round((s - int(s)) * 1000))
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"

def fmt_ass_time(s):
    h, m = int(s // 3600), int((s % 3600) // 60)
    return f"{h}:{m:02d}:{s % 60:05.2f}"

def build_srt(segs):
    out = []
    for i, s in enumerate(segs, 1):
        out += [str(i), f"{fmt_srt_time(s['start'])} --> {fmt_srt_time(s['end'])}", s['text'], ""]
    return "\n".join(out)

def build_vtt(segs):
    out = ["WEBVTT", ""]
    for i, s in enumerate(segs, 1):
        out += [str(i), f"{fmt_vtt_time(s['start'])} --> {fmt_vtt_time(s['end'])}", s['text'], ""]
    return "\n".join(out)

def build_ass(segs):
    hdr = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, "
        "Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "0,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = [f"Dialogue: 0,{fmt_ass_time(s['start'])},{fmt_ass_time(s['end'])},Default,,0,0,0,,{s['text']}" for s in segs]
    return hdr + "\n".join(events)

def build_tsv(segs):
    return "\n".join(["start\tend\ttext"] + [f"{s['start']:.3f}\t{s['end']:.3f}\t{s['text']}" for s in segs])

def write_temp(content, suffix):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name

_last_sub_files = []

def cleanup_last_subs():
    for p in _last_sub_files:
        try:
            if os.path.exists(p): os.unlink(p)
        except Exception:
            pass
    _last_sub_files.clear()


# ─── Language map ─────────────────────────────────────────────────────────────

LANGUAGE_MAP = {
    "Auto-detect": None, "English": "en", "Spanish": "es",
    "French": "fr", "German": "de", "Italian": "it",
    "Portuguese": "pt", "Dutch": "nl", "Russian": "ru",
    "Chinese": "zh", "Japanese": "ja", "Korean": "ko",
    "Arabic": "ar", "Hindi": "hi", "Turkish": "tr",
}


# ─── Main transcription ───────────────────────────────────────────────────────
# Outputs (14 total):
#   progress_bar, text_out, detected_out, segments_out, console_out,
#   srt_preview, vtt_preview, ass_preview, tsv_preview,
#   srt_file, vtt_file, ass_file, tsv_file

def run(audio_file, mic_audio, direct_path, model_name, language_label, device, compute_type):
    cleanup_last_subs()
    console_lines = []

    def log(msg):
        print(msg)
        console_lines.append(msg)

    def console():
        return "\n".join(console_lines)

    # Helper: yield a full state tuple
    # pct=-1 means keep current bar, label=None means keep current label
    def emit(pct=None, bar_label="", done=False,
             text="", detected="", segments="",
             srt="", vtt="", ass="", tsv="",
             srt_f=None, vtt_f=None, ass_f=None, tsv_f=None):
        bar = make_progress_html(pct if pct is not None else 0, bar_label, done)
        return (bar, text, detected, segments, console(),
                srt, vtt, ass, tsv, srt_f, vtt_f, ass_f, tsv_f)

    # ── Resolve input ──
    file_path = None
    use_temp = False

    if direct_path and direct_path.strip():
        file_path = direct_path.strip()
        log(f"[INFO] Using direct path: {file_path}")
        if not os.path.exists(file_path):
            yield emit(pct=0, bar_label=f"Error: file not found", detected=f"File not found: {file_path}")
            return
    elif mic_audio is not None:
        log("[INFO] Saving microphone recording...")
        file_path = save_mic_to_tempfile(mic_audio)
        use_temp = True
        log("[OK]   Mic audio saved.")
    elif audio_file is not None:
        raw = audio_file if isinstance(audio_file, str) else (audio_file.get("path") or audio_file.get("name", ""))
        ext = os.path.splitext(raw)[-1] or ".wav"
        log(f"[INFO] Copying uploaded file...")
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            shutil.copy2(raw, tmp.name)
            file_path = tmp.name
        use_temp = True
        log("[OK]   File ready.")
    else:
        yield emit(pct=0, bar_label="No input provided", detected="Upload a file, record from mic, or paste a path.")
        return

    yield emit(pct=0.02, bar_label="Loading model...")

    # ── Load model ──
    try:
        model = get_model(model_name, device, compute_type, log)
        yield emit(pct=0.05, bar_label="Model loaded — starting transcription...")
    except Exception as e:
        log(f"[ERROR] Failed to load model: {e}")
        yield emit(pct=0, bar_label="Error loading model", detected="Error loading model")
        if use_temp and file_path:
            os.unlink(file_path)
        return

    # ── Transcribe ──
    lang_code = LANGUAGE_MAP.get(language_label, None)
    log(f"[INFO] Starting transcription (lang={lang_code or 'auto'}, beam=5, VAD=on)...")

    def do_transcribe(mdl, fp, lang, device_label=""):
        segments_gen, info = mdl.transcribe(
            fp, language=lang, beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        detected_lang = info.language.upper() + (f" ({device_label})" if device_label else "")
        duration = info.duration or 1
        log(f"[INFO] Detected language: {detected_lang} (duration: {duration:.1f}s)")

        full_text = ""
        segments_text = ""
        seg_data = []
        seg_count = 0
        last_text_yield = time.monotonic()
        TEXT_INTERVAL = 0.8  # seconds between text box updates

        for seg in segments_gen:
            seg_count += 1
            text = seg.text.strip()
            full_text += seg.text + " "
            seg_data.append({"start": seg.start, "end": seg.end, "text": text})

            m_s, s_s = divmod(seg.start, 60)
            m_e, s_e = divmod(seg.end, 60)
            seg_line = f"[{int(m_s):02d}:{s_s:04.1f} → {int(m_e):02d}:{s_e:04.1f}]  {text}"
            segments_text += seg_line + "\n"
            log(f"[SEG {seg_count:03d}] {seg_line}")

            pct = min(0.05 + 0.93 * (seg.end / duration), 0.98)
            bar_label = f"Transcribing… {seg.end:.0f}s / {duration:.0f}s  ({seg_count} segments)"

            now = time.monotonic()
            if now - last_text_yield >= TEXT_INTERVAL:
                last_text_yield = now
                # Update everything — progress bar + live text
                yield emit(
                    pct=pct, bar_label=bar_label,
                    text=full_text.strip(),
                    detected=f"Detected: {detected_lang}",
                    segments=segments_text.strip(),
                )
            else:
                # Update ONLY the progress bar — text boxes stay stable, no flash
                yield emit(
                    pct=pct, bar_label=bar_label,
                    text=full_text.strip(),
                    detected=f"Detected: {detected_lang}",
                    segments=segments_text.strip(),
                )

        # Build subtitle files
        log(f"[INFO] Building subtitle files...")
        srt_text  = build_srt(seg_data)
        vtt_text  = build_vtt(seg_data)
        ass_text  = build_ass(seg_data)
        tsv_text  = build_tsv(seg_data)
        srt_path  = write_temp(srt_text,  ".srt")
        vtt_path  = write_temp(vtt_text,  ".vtt")
        ass_path  = write_temp(ass_text,  ".ass")
        tsv_path  = write_temp(tsv_text,  ".tsv")
        _last_sub_files.extend([srt_path, vtt_path, ass_path, tsv_path])

        log(f"[OK]   Done — {seg_count} segments, {len(full_text.split())} words.")

        yield emit(
            pct=1.0, bar_label=f"Done — {seg_count} segments, {len(full_text.split())} words.", done=True,
            text=full_text.strip(),
            detected=f"Detected: {detected_lang}",
            segments=segments_text.strip(),
            srt=srt_text, vtt=vtt_text, ass=ass_text, tsv=tsv_text,
            srt_f=srt_path, vtt_f=vtt_path, ass_f=ass_path, tsv_f=tsv_path,
        )

    try:
        try:
            yield from do_transcribe(model, file_path, lang_code)
        except RuntimeError as e:
            if "cublas" in str(e).lower() or "cuda" in str(e).lower():
                log(f"[WARN] CUDA error: {e}")
                log(f"[INFO] Retrying on CPU with int8...")
                yield emit(pct=0.05, bar_label="CUDA failed — retrying on CPU...")
                cpu_model = WhisperModel(model_name, device="cpu", compute_type="int8")
                log(f"[OK]   CPU model loaded.")
                yield from do_transcribe(cpu_model, file_path, lang_code, device_label="CPU fallback")
            else:
                raise
    except Exception as e:
        log(f"[ERROR] {e}")
        yield emit(pct=0, bar_label=f"Error — {e}", detected="Error — check console")
    finally:
        if use_temp and file_path and os.path.exists(file_path):
            os.unlink(file_path)


# ─── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Faster Whisper Transcription") as demo:
    gr.Markdown("# ⚡ Faster Whisper — Local Transcription\n**GPU-accelerated · ~4x faster · 100% offline · Live output**")

    with gr.Row():
        # ── Left: inputs & settings ──
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
                    info="Fastest — reads directly from disk, no upload.",
                )

            with gr.Row():
                model_input = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                    value="base", label="Model",
                    info="Larger = more accurate, slower",
                )
                device_input = gr.Dropdown(
                    choices=["cuda", "cpu"], value="cuda", label="Device",
                )
            compute_input = gr.Dropdown(
                choices=["float16", "int8_float16", "int8"], value="float16",
                label="Compute Type",
                info="float16=fastest GPU · int8_float16=less VRAM · int8=CPU",
            )
            language_input = gr.Dropdown(
                choices=list(LANGUAGE_MAP.keys()), value="Auto-detect", label="Language",
            )
            submit_btn = gr.Button("Transcribe →", variant="primary", size="lg")

        # ── Right: outputs ──
        with gr.Column(scale=2):
            # Custom progress bar — always visible, never flashes
            progress_bar = gr.HTML(value=PROGRESS_IDLE, label="Progress")

            detected_out = gr.Textbox(label="Detected Language", interactive=False, max_lines=1)

            with gr.Tab("Transcription"):
                text_out = gr.Textbox(
                    label="Full Text", lines=12,
                    placeholder="Transcript streams here as it's processed…",
                )
            with gr.Tab("Timestamps"):
                segments_out = gr.Textbox(
                    label="Timestamped Segments", lines=12,
                    placeholder="Segments appear here in real time…",
                )
            with gr.Tab("SRT"):
                srt_preview = gr.Textbox(label="SRT Preview", lines=12, interactive=False)
                srt_file = gr.File(label="Download .srt", interactive=False)
            with gr.Tab("VTT"):
                vtt_preview = gr.Textbox(label="WebVTT Preview", lines=12, interactive=False)
                vtt_file = gr.File(label="Download .vtt", interactive=False)
            with gr.Tab("ASS"):
                ass_preview = gr.Textbox(label="ASS/SSA Preview", lines=12, interactive=False)
                ass_file = gr.File(label="Download .ass", interactive=False)
            with gr.Tab("TSV"):
                tsv_preview = gr.Textbox(label="TSV Preview", lines=12, interactive=False)
                tsv_file = gr.File(label="Download .tsv", interactive=False)

            console_out = gr.Textbox(
                label="Console Output", lines=8,
                placeholder="Logs will appear here…", interactive=False,
            )

    submit_btn.click(
        fn=run,
        inputs=[audio_input, mic_input, direct_path_input,
                model_input, language_input, device_input, compute_input],
        outputs=[
            progress_bar,
            text_out, detected_out, segments_out, console_out,
            srt_preview, vtt_preview, ass_preview, tsv_preview,
            srt_file, vtt_file, ass_file, tsv_file,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", theme=gr.themes.Monochrome())