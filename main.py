import gradio as gr
import whisper

loaded_models = {}

def get_model(model_name):
    if model_name not in loaded_models:
        loaded_models[model_name] = whisper.load_model(model_name)
    return loaded_models[model_name]

def transcribe(audio_path, model_name, language):
    if audio_path is None:
        return "No audio provided.", "", ""

    model = get_model(model_name)

    options = {}
    if language != "Auto-detect":
        options["language"] = language

    result = model.transcribe(audio_path, **options)

    full_text = result["text"].strip()
    detected = result.get("language", "unknown").upper()

    # Build timestamped transcript
    segments_text = ""
    for seg in result.get("segments", []):
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        m_s, s_s = divmod(start, 60)
        m_e, s_e = divmod(end, 60)
        segments_text += f"[{int(m_s):02d}:{s_s:04.1f} → {int(m_e):02d}:{s_e:04.1f}]  {text}\n"

    return full_text, detected, segments_text.strip()


LANGUAGES = [
    "Auto-detect", "English", "Spanish", "French", "German",
    "Italian", "Portuguese", "Dutch", "Russian", "Chinese",
    "Japanese", "Korean", "Arabic", "Hindi", "Turkish",
]

LANGUAGE_MAP = {
    "Auto-detect": None, "English": "en", "Spanish": "es",
    "French": "fr", "German": "de", "Italian": "it",
    "Portuguese": "pt", "Dutch": "nl", "Russian": "ru",
    "Chinese": "zh", "Japanese": "ja", "Korean": "ko",
    "Arabic": "ar", "Hindi": "hi", "Turkish": "tr",
}

def run(audio, model_name, language_label):
    lang_code = LANGUAGE_MAP.get(language_label, None)
    text, detected, segments = transcribe(audio, model_name, lang_code)
    return text, f"Detected: {detected}", segments

with gr.Blocks(title="Whisper Transcription", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown(
        """
        # 🎙 Whisper — Local Transcription
        **100% offline. Nothing leaves your machine.**
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Audio / Video File",
                type="filepath",
                sources=["upload", "microphone"],
            )
            model_input = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large"],
                value="base",
                label="Model",
                info="Larger = more accurate but slower",
            )
            language_input = gr.Dropdown(
                choices=LANGUAGES,
                value="Auto-detect",
                label="Language",
            )
            submit_btn = gr.Button("Transcribe →", variant="primary")

        with gr.Column(scale=2):
            detected_out = gr.Textbox(label="Language", interactive=False, max_lines=1)
            text_out = gr.Textbox(
                label="Transcription",
                lines=12,
                show_copy_button=True,
                placeholder="Your transcript will appear here…",
            )
            segments_out = gr.Textbox(
                label="Timestamped Segments",
                lines=12,
                show_copy_button=True,
                placeholder="Segments with timestamps will appear here…",
            )

    submit_btn.click(
        fn=run,
        inputs=[audio_input, model_input, language_input],
        outputs=[text_out, detected_out, segments_out],
    )

    gr.Markdown(
        """
        ---
        **Tips:**
        - First run downloads the model (~75MB for `base`) — cached after that
        - Use `tiny` or `base` for speed, `large` for best accuracy
        - You can also record directly from your microphone
        """
    )

if __name__ == "__main__":
    demo.launch()