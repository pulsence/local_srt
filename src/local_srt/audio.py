#!/usr/bin/env python3
"""Audio processing utilities for Local SRT.

This module handles audio conversion and silence detection using ffmpeg.
"""
from __future__ import annotations

import re
import subprocess
from typing import List, Optional, Tuple

from .system import ffmpeg_ok, probe_duration_seconds, run_cmd_text


# ============================================================
# Silence Detection
# ============================================================

_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")


def detect_silences(
    wav_path: str,
    *,
    min_silence_dur: float,
    silence_threshold_db: float,
) -> List[Tuple[float, float]]:
    """Detect silent regions in an audio file using ffmpeg's silencedetect filter.

    Args:
        wav_path: Path to WAV file to analyze
        min_silence_dur: Minimum duration of silence to detect (seconds)
        silence_threshold_db: Silence threshold in dB (e.g., -35.0)

    Returns:
        List of (start_time, end_time) tuples for detected silent regions
    """
    if not ffmpeg_ok():
        return []

    filt = f"silencedetect=noise={silence_threshold_db}dB:d={min_silence_dur}"
    cmd = ["ffmpeg", "-i", wav_path, "-af", filt, "-f", "null", "-"]
    code, _, err = run_cmd_text(cmd)
    if code != 0:
        return []

    silences: List[Tuple[float, float]] = []
    pending_start: Optional[float] = None

    for line in err.splitlines():
        m = _SILENCE_START_RE.search(line)
        if m:
            pending_start = float(m.group(1))
            continue
        m = _SILENCE_END_RE.search(line)
        if m and pending_start is not None:
            end = float(m.group(1))
            start = min(pending_start, end)
            if end - start > 0:
                silences.append((start, end))
            pending_start = None

    if pending_start is not None:
        dur = probe_duration_seconds(wav_path)
        if dur is not None and dur > pending_start:
            silences.append((pending_start, dur))

    if not silences:
        return []

    silences.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    for s, e in silences:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


# ============================================================
# Audio Conversion
# ============================================================

def to_wav_16k_mono(input_path: str, wav_path: str) -> None:
    """Convert an audio/video file to 16kHz mono WAV format.

    This is the required format for Whisper transcription.

    Args:
        input_path: Path to input audio/video file
        wav_path: Path where output WAV file should be written

    Raises:
        subprocess.CalledProcessError: If ffmpeg conversion fails
    """
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        wav_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        tail = "\n".join((p.stderr or "").splitlines()[-20:])
        raise subprocess.CalledProcessError(p.returncode, cmd, output=None, stderr=tail)
