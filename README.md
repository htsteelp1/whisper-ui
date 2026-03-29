# ⚡ Faster Whisper — Local Transcription

A local, private, GPU-accelerated audio/video transcription app built with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [Gradio](https://gradio.app).

- ~4x faster than standard OpenAI Whisper
- Runs entirely on your machine — no data sent anywhere
- Accessible from other devices on your local network
- Supports 14 languages + auto-detection
- Timestamped segment output

---

## Requirements

- Python 3.10–3.12 recommended (3.14 works but is less tested with ML libs)
- NVIDIA GPU recommended (CUDA 12.x); falls back to CPU automatically
- Install.bat supports the latest Cuda Version
- ffmpeg installed and on PATH

### Install ffmpeg (if not already)

```
winget install ffmpeg
```

Or download from https://ffmpeg.org/download.html and add to PATH manually.

---

## Installation

### Auto Install Method

This assumes that you already have CUDA installed on the latest version. Simply run the install.bat file

### Step 1 — Install PyTorch with CUDA support

**RTX 50 series / CUDA 13.x (your setup):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Older NVIDIA GPU (CUDA 11.x / 12.x):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**
```bash
pip install torch torchaudio
```

### Step 2 — Install app dependencies

```bash
pip install faster-whisper gradio
```

### Step 3 — Run

```bash
python main.py
```

Or double-click `start.bat`.

The app opens at **http://localhost:7860**

---

## Accessing from another device on your network

The app binds to `0.0.0.0` so it's already network-accessible.

1. Find your PC's local IP:
   ```
   ipconfig
   ```
   Look for **IPv4 Address** under Wi-Fi or Ethernet — e.g. `192.168.1.45`

2. On the other device, open a browser and go to:
   ```
   http://192.168.1.45:7860
   ```

If it doesn't connect, allow the port through Windows Firewall:
```
netsh advfirewall firewall add rule name="Gradio Whisper" dir=in action=allow protocol=TCP localport=7860
```

---

## Usage

| Input method | When to use |
|---|---|
| **File upload** | Uploading from another device on the network |
| **Paste file path** | Fastest option when using on the same PC — bypasses the HTTP upload entirely |

### Model guide

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| tiny | 39 MB | Fastest | Lowest |
| base | 74 MB | Fast | Decent |
| small | 244 MB | Moderate | Good |
| medium | 769 MB | Slow | Better |
| large-v2 | 1.5 GB | Slowest | Best |
| large-v3 | 1.5 GB | Slowest | Best (latest) |

Models are downloaded on first use and cached at `C:\Users\<you>\.cache\huggingface\hub\`.

### Compute type guide

| Type | Best for |
|---|---|
| `float16` | NVIDIA GPU, fastest |
| `int8_float16` | NVIDIA GPU, lower VRAM usage |
| `int8` | CPU, or low VRAM GPU |

---

## Troubleshooting

**`cublas64_12.dll not found`**
Your CUDA runtime is missing. Install CUDA 12.8 Toolkit from:
https://developer.nvidia.com/cuda-12-8-0-download-archive
Then reinstall PyTorch with `--index-url https://download.pytorch.org/whl/cu128`

**Permission denied on temp file**
Known Gradio 6 bug on Windows. This app works around it by using `gr.File` instead of `gr.Audio`. If you still see it, use the file path box instead of uploading.

**CUDA error / crashes on transcribe**
The app will automatically retry on CPU. To permanently fix, see the CUDA section above.

**Slow transcription on CPU**
Switch to `int8` compute type and use `tiny` or `base` model for fastest results.

**HuggingFace symlink warning on Windows**
Non-fatal warning. Enable Developer Mode in Windows Settings to suppress it, or set:
```
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

---

## File structure

```
whisper-gradio/
├── main.py          # Main app
├── requirements.txt # Python dependencies
├── start.bat        # Double-click to launch
└── README.md        # This file
```
