import gradio as gr
import shutil
import tempfile
import os
from faster_whisper import WhisperModel

loaded_models = {}

def get_model(model_name, device, compute_type):
    key = (model_name, device, compute_type)
    if key not in loaded_models:
        print(f"Loading model '{model_name}' on {device} ({compute_type})...")
        try:
            loaded_models[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as e:
            if device == "cuda":
                print(f"CUDA failed ({e}), falling back to CPU int8...")
                loaded_models[key] = WhisperModel(model_name, device="cpu", compute_type="int8")
            else:
                raise
    return loaded_models[key]

LANGUAGE_MAP = {
    "Auto-detect": None, "English": "en", "Spanish": "es",
    "French": "fr", "German": "de", "Italian": "it",
    "Portuguese": "pt", "Dutch": "nl", "Russian": "ru",
    "Chinese": "zh", "Japanese": "ja", "Korean": "ko",
    "Arabic": "ar", "Hindi": "hi", "Turkish": "tr",
}

def run(audio, direct_path, model_name, language_label, device, compute_type):
    # Prefer direct path (instant, no upload) over uploaded file
    if direct_path and direct_path.strip():
        file_path = direct_path.strip()
        if not os.path.exists(file_path):
            return f"File not found: {file_path}", "", ""
        use_temp = False
    elif audio is not None:
        # gr.File returns a dict with a 'path' or 'name' key
        raw = audio if isinstance(audio, str) else (audio.get("path") or audio.get("name", ""))
        ext = os.path.splitext(raw)[-1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            shutil.copy2(raw, tmp.name)
            file_path = tmp.name
        use_temp = True
    else:
        return "No audio provided. Upload a file or paste a file path.", "", ""

    lang_code = LANGUAGE_MAP.get(language_label, None)

    def do_transcribe(dev, ctype):
        model = get_model(model_name, dev, ctype)
        segments_gen, info = model.transcribe(
            file_path,
            language=lang_code,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        full_text = ""
        segments_text = ""
        for seg in segments_gen:
            full_text += seg.text + " "
            m_s, s_s = divmod(seg.start, 60)
            m_e, s_e = divmod(seg.end, 60)
            segments_text += f"[{int(m_s):02d}:{s_s:04.1f} \u2192 {int(m_e):02d}:{s_e:04.1f}]  {seg.text.strip()}\n"
        return full_text.strip(), info.language.upper(), segments_text.strip()

    try:
        try:
            full_text, detected, segments_text = do_transcribe(device, compute_type)
        except RuntimeError as cuda_err:
            if "cuda" in device.lower() or "cublas" in str(cuda_err).lower() or "cuda" in str(cuda_err).lower():
                print(f"[WARNING] CUDA failed ({cuda_err}). Retrying on CPU with int8...")
                full_text, detected, segments_text = do_transcribe("cpu", "int8")
                detected += " (CPU fallback — install CUDA 12 for GPU speed)"
            else:
                raise

        return full_text, f"Detected: {detected}", segments_text

    finally:
        if use_temp:
            os.unlink(file_path)



with gr.Blocks(title="Faster Whisper Transcription") as demo:
    gr.Markdown(
        """
        # ⚡ Faster Whisper — Local Transcription
        **GPU-accelerated. ~4x faster than standard Whisper. 100% offline.**
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.File(
                label="Upload Audio / Video File",
                file_types=["audio", "video", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mkv", ".mp4"],
            )
            direct_path_input = gr.Textbox(
                label="Or paste a file path (instant — no upload needed)",
                placeholder=r"e.g. C:\Users\Henry\Music\lecture.wav",
                info="If filled, this takes priority over the upload above.",
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
                    info="Use cuda for NVIDIA GPU",
                )

            compute_input = gr.Dropdown(
                choices=["float16", "int8_float16", "int8"],
                value="float16",
                label="Compute Type",
                info="float16 = fastest on GPU · int8 = less VRAM · int8 on CPU",
            )

            language_input = gr.Dropdown(
                choices=list(LANGUAGE_MAP.keys()),
                value="Auto-detect",
                label="Language",
            )

            submit_btn = gr.Button("Transcribe →", variant="primary")

        with gr.Column(scale=2):
            detected_out = gr.Textbox(label="Language", interactive=False, max_lines=1)
            text_out = gr.Textbox(
                label="Transcription",
                lines=12,
                placeholder="Your transcript will appear here…",
            )
            segments_out = gr.Textbox(
                label="Timestamped Segments",
                lines=12,
                placeholder="Segments with timestamps will appear here…",
            )

    submit_btn.click(
        fn=run,
        inputs=[audio_input, direct_path_input, model_input, language_input, device_input, compute_input],
        outputs=[text_out, detected_out, segments_out],
    )

    gr.Markdown(
        """
        ---
        **Compute type guide:**
        - `float16` — fastest, recommended for most NVIDIA GPUs
        - `int8_float16` — slightly slower but uses less VRAM
        - `int8` — best for CPU or low VRAM GPUs

        **Tips:**
        - If you get a CUDA error, switch Device to `cpu` and Compute Type to `int8`
        - VAD filter is enabled — silent sections are automatically skipped for speed
        - Models are cached in memory after first load
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", theme=gr.themes.Monochrome())