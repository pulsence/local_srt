# Local SRT

Local SRT generator using **faster-whisper** (offline transcription) and **ffmpeg** (media decoding).

This tool converts audio or video files into readable `.srt` subtitle files with intelligent
punctuation-aware chunking, pacing heuristics, and presets for YouTube, Shorts, and Podcasts.

## Caveat Emptor
This project was primarily created for my personal use. I will not be responding to pull requests
or issues unless they directly impact my use cases. Feel free to fork and make whatever changes
you would like. I was frustrated that a "turn key" local SRT generator was not easily 
available, now there is.

I generated this tool primarily using an AI code assistant and so all the code branches are not
explored or tested, but they should be farely correct.

## Features
- Fully local transcription (no remote API calls)
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
- Output formats: SRT, VTT, ASS, TXT, JSON
- Word-level cue output (optional)

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

ASS output:

```bash
srtgen input.mp4 --format ass
```

Word-level output:

```bash
srtgen input.mp4 --word-level
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
--word-level
--no-silence-split
--overwrite
--quiet
```

## Model Management

List downloaded models:

```bash
srtgen --list-models
```

Download a model:

```bash
srtgen --download-model small
```

Delete a downloaded model:

```bash
srtgen --delete-model small
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
