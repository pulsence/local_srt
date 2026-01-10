#!/usr/bin/env python3
"""Output writers for various subtitle formats.

This module handles writing subtitles to different formats:
- SRT (SubRip)
- VTT (WebVTT)
- ASS (Advanced SubStation Alpha)
- TXT (plain transcript)
- JSON (complete bundle with metadata)
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .models import ResolvedConfig, SubtitleBlock
from .system import ensure_parent_dir
from .text_processing import normalize_spaces, wrap_text_lines


# ============================================================
# Time Formatters
# ============================================================

def format_srt_time(seconds: float) -> str:
    """Format time for SRT format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "00:01:23,456")
    """
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_vtt_time(seconds: float) -> str:
    """Format time for WebVTT format (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "00:01:23.456")
    """
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_ass_time(seconds: float) -> str:
    """Format time for ASS format (H:MM:SS.cc, where cc is centiseconds).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "0:01:23.45")
    """
    cs = int(round(seconds * 100))
    h = cs // 360_000
    cs %= 360_000
    m = cs // 6_000
    cs %= 6_000
    s = cs // 100
    cs %= 100
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


# ============================================================
# Atomic File Writing
# ============================================================

def atomic_write_text(path: Path, content: str) -> None:
    """Write text to a file atomically using a temporary file.

    Args:
        path: Destination file path
        content: Text content to write
    """
    ensure_parent_dir(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


# ============================================================
# Format Writers
# ============================================================

def write_srt(subs: List[SubtitleBlock], out_path: Path, *, max_chars: int, max_lines: int) -> None:
    """Write subtitles in SRT (SubRip) format.

    Args:
        subs: List of SubtitleBlock objects
        out_path: Output file path
        max_chars: Maximum characters per line
        max_lines: Maximum lines per subtitle
    """
    chunks: List[str] = []
    for i, sb in enumerate(subs, start=1):
        text = normalize_spaces(" ".join(sb.lines))
        lines = wrap_text_lines(text, max_chars)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        chunks.append(
            f"{i}\n"
            f"{format_srt_time(sb.start)} --> {format_srt_time(sb.end)}\n"
            f"{'\n'.join(lines).strip()}\n"
        )
    atomic_write_text(out_path, "\n".join(chunks).strip() + "\n")


def write_vtt(subs: List[SubtitleBlock], out_path: Path, *, max_chars: int, max_lines: int) -> None:
    """Write subtitles in WebVTT format.

    Args:
        subs: List of SubtitleBlock objects
        out_path: Output file path
        max_chars: Maximum characters per line
        max_lines: Maximum lines per subtitle
    """
    chunks: List[str] = ["WEBVTT\n"]
    for sb in subs:
        text = normalize_spaces(" ".join(sb.lines))
        lines = wrap_text_lines(text, max_chars)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        chunks.append(
            f"{format_vtt_time(sb.start)} --> {format_vtt_time(sb.end)}\n"
            f"{'\n'.join(lines).strip()}\n"
        )
    atomic_write_text(out_path, "\n".join(chunks).rstrip() + "\n")


def write_ass(subs: List[SubtitleBlock], out_path: Path, *, max_chars: int, max_lines: int) -> None:
    """Write subtitles in ASS (Advanced SubStation Alpha) format.

    Args:
        subs: List of SubtitleBlock objects
        out_path: Output file path
        max_chars: Maximum characters per line
        max_lines: Maximum lines per subtitle
    """
    header = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1920",
            "PlayResY: 1080",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,"
            "1,2,0,2,80,80,60,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )
    events: List[str] = [header]
    for sb in subs:
        text = normalize_spaces(" ".join(sb.lines))
        lines = wrap_text_lines(text, max_chars)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        ass_text = "\\N".join(lines).strip()
        events.append(
            f"Dialogue: 0,{format_ass_time(sb.start)},{format_ass_time(sb.end)},Default,,0,0,0,,{ass_text}"
        )
    atomic_write_text(out_path, "\n".join(events).rstrip() + "\n")


def write_txt(subs: List[SubtitleBlock], out_path: Path) -> None:
    """Write plain text transcript from subtitle blocks.

    Args:
        subs: List of SubtitleBlock objects
        out_path: Output file path
    """
    lines: List[str] = []
    for sb in subs:
        lines.append(normalize_spaces(" ".join(sb.lines)))
    atomic_write_text(out_path, "\n".join(lines).strip() + "\n")


# ============================================================
# JSON Utilities
# ============================================================

def segments_to_jsonable(segments: List[Any], *, include_words: bool) -> List[Dict[str, Any]]:
    """Convert transcription segments to JSON-serializable format.

    Args:
        segments: List of transcription segment objects
        include_words: If True, include word-level timing data

    Returns:
        List of dictionaries representing segments
    """
    out: List[Dict[str, Any]] = []
    for s in segments:
        d: Dict[str, Any] = {
            "start": float(s.start),
            "end": float(s.end),
            "text": getattr(s, "text", ""),
        }
        if include_words and getattr(s, "words", None):
            d["words"] = [
                {"start": float(w.start), "end": float(w.end), "word": w.word}
                for w in s.words
            ]
        out.append(d)
    return out


def write_json_bundle(
    out_path: Path,
    *,
    input_file: str,
    device_used: str,
    compute_type_used: str,
    cfg: ResolvedConfig,
    segments: List[Any],
    subs: List[SubtitleBlock],
    tool_version: str,
) -> None:
    """Write a complete JSON bundle with metadata, segments, and subtitles.

    Args:
        out_path: Output file path
        input_file: Input file name
        device_used: Device used for transcription (cpu/cuda)
        compute_type_used: Compute type used (int8/float16)
        cfg: Configuration used
        segments: List of transcription segments
        subs: List of SubtitleBlock objects
        tool_version: Tool version string
    """
    payload = {
        "tool_version": tool_version,
        "input_file": input_file,
        "device_used": device_used,
        "compute_type_used": compute_type_used,
        "config": dataclasses.asdict(cfg),
        "segments": segments_to_jsonable(segments, include_words=cfg.word_timestamps),
        "subtitles": [
            {"start": sb.start, "end": sb.end, "text": normalize_spaces(" ".join(sb.lines))}
            for sb in subs
        ],
    }
    ensure_parent_dir(out_path)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)
