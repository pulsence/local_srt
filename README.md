# Local SRT

Local SRT generator using **faster-whisper** (offline transcription) and **ffmpeg** (media decoding).

This tool converts audio or video files into readable `.srt` subtitle files with intelligent
punctuation-aware chunking, pacing heuristics, and presets for YouTube, Shorts, and Podcasts.

## Features
- Fully local transcription (no API calls)
- Uses `faster-whisper` for high-quality speech recognition
- Intelligent subtitle chunking:
  - Punctuation-aware splitting (strong → medium → weak)
  - Reading-speed constraints (characters per second)
  - Minimum and maximum subtitle durations
- Preset modes:
  - `yt` – standard YouTube captions
  - `shorts` – fast-paced, compact captions
  - `podcast` – slower pacing, longer phrasing
- CUDA support with automatic CPU fallback
- Progress indicators with segment-based timing
- Supports batch processing of files
- Works on Windows, macOS, and Linux

## Requirements
- **Python**: 3.10 or newer
- **ffmpeg** (required)
- **ffprobe** (recommended; usually included with ffmpeg)
- Optional:
  - NVIDIA GPU + CUDA drivers for `--device cuda`

## Installation
### Option A: pipx (recommended)
```bash
python -m pip install --upgrade pipx
pipx ensurepath
pipx install .
```

From GitHub:

```bash
pipx install git+https://github.com/your-org/local-srt.git
```

---

### Option B: Virtual environment

```bash
python -m venv .venv
```

Activate:

**Windows**
```powershell
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install:

```bash
python -m pip install .
```


## Installing ffmpeg
### Windows
```powershell
winget install Gyan.FFmpeg
```

or

```powershell
choco install ffmpeg
```

---

### macOS

```bash
brew install ffmpeg
```

---

### Linux

```bash
sudo apt install ffmpeg
```

---

## Usage

Basic:

```bash
srtgen input.mp4 -o output.srt
```

Preset modes:

```bash
srtgen input.mp4 --mode yt
srtgen input.mp4 --mode shorts
srtgen input.mp4 --mode podcast
```

CUDA (with fallback):

```bash
srtgen input.mp4 --device cuda
```

---

## Common Options

```bash
--model small
--language en
--max_chars 42
--max_lines 2
--target_cps 17
--min_dur 1.0
--max_dur 6.0
--overwrite
--quiet
```

---

## Troubleshooting

**ffmpeg not found**
```bash
ffmpeg -version
```

**CUDA errors**
- Ensure NVIDIA drivers are installed
- Tool will fall back to CPU automatically

---

## License
MIT