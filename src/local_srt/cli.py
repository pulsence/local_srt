#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from faster_whisper import WhisperModel
from faster_whisper import utils as fw_utils


# ============================================================
# Versioning
# ============================================================

TOOL_VERSION = "0.1.0"


# ============================================================
# Logging / Progress
# ============================================================

def log(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)


def warn(msg: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"WARNING: {msg}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> int:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    return code


def progress_line(msg: str, *, enabled: bool, quiet: bool) -> None:
    if quiet or not enabled:
        return
    sys.stdout.write("\r" + msg[:160].ljust(160))
    sys.stdout.flush()


def progress_done(*, enabled: bool, quiet: bool) -> None:
    if quiet or not enabled:
        return
    sys.stdout.write("\n")
    sys.stdout.flush()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


# ============================================================
# Config / Presets
# ============================================================

@dataclass
class ResolvedConfig:
    # caption formatting
    max_chars: int = 42
    max_lines: int = 2

    # readability / timing heuristics
    target_cps: float = 17.0
    min_dur: float = 1.0
    max_dur: float = 6.0

    # punctuation splitting
    allow_commas: bool = True
    allow_medium: bool = True
    prefer_punct_splits: bool = False

    # timing polish
    min_gap: float = 0.08
    pad: float = 0.00

    # transcription options
    vad_filter: bool = True
    word_timestamps: bool = False

    # silence-aware timing
    use_silence_split: bool = True
    silence_min_dur: float = 0.2
    silence_threshold_db: float = -35.0


PRESETS: Dict[str, Dict[str, Any]] = {
    "shorts": {
        "max_chars": 18,
        "max_lines": 1,
        "target_cps": 18.0,
        "min_dur": 0.7,
        "max_dur": 3.0,
        "prefer_punct_splits": False,
        "allow_commas": True,
        "allow_medium": True,
        "min_gap": 0.08,
        "pad": 0.00,
    },
    "yt": {
        "max_chars": 42,
        "max_lines": 2,
        "target_cps": 17.0,
        "min_dur": 1.0,
        "max_dur": 6.0,
        "prefer_punct_splits": False,
        "allow_commas": True,
        "allow_medium": True,
        "min_gap": 0.08,
        "pad": 0.00,
    },
    "podcast": {
        "max_chars": 40,
        "max_lines": 2,
        "target_cps": 16.0,
        "min_dur": 0.9,
        "max_dur": 5.0,
        "prefer_punct_splits": True,
        "allow_commas": True,
        "allow_medium": True,
        "min_gap": 0.08,
        "pad": 0.05,
    },
}

MODE_ALIASES = {
    "short": "shorts",
    "shorts": "shorts",
    "yt": "yt",
    "youtube": "yt",
    "pod": "podcast",
    "podcast": "podcast",
}

def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object at top-level.")
    return data


def apply_overrides(base: ResolvedConfig, overrides: Dict[str, Any]) -> ResolvedConfig:
    d = dataclasses.asdict(base)
    for k, v in overrides.items():
        if k in d:
            d[k] = v
    return ResolvedConfig(**d)


# ============================================================
# System / Dependency checks
# ============================================================

def ensure_parent_dir(path: Path) -> None:
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def which_or_none(name: str) -> Optional[str]:
    return shutil.which(name)


def ffmpeg_ok() -> bool:
    return which_or_none("ffmpeg") is not None


def ffprobe_ok() -> bool:
    return which_or_none("ffprobe") is not None


def run_cmd_text(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def ffmpeg_version() -> Optional[str]:
    if not ffmpeg_ok():
        return None
    code, out, _ = run_cmd_text(["ffmpeg", "-version"])
    if code != 0:
        return None
    return out.splitlines()[0].strip() if out else None


def ffprobe_version() -> Optional[str]:
    if not ffprobe_ok():
        return None
    code, out, _ = run_cmd_text(["ffprobe", "-version"])
    if code != 0:
        return None
    return out.splitlines()[0].strip() if out else None


def probe_duration_seconds(path: str) -> Optional[float]:
    if not ffprobe_ok():
        return None
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    code, out, _ = run_cmd_text(cmd)
    if code != 0:
        return None
    try:
        return float(out.strip())
    except Exception:
        return None


_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")


def detect_silences(
    wav_path: str,
    *,
    min_silence_dur: float,
    silence_threshold_db: float,
) -> List[Tuple[float, float]]:
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
# Audio conversion
# ============================================================

def to_wav_16k_mono(input_path: str, wav_path: str) -> None:
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


# ============================================================
# Device init
# ============================================================

def init_whisper_model(
    model_name: str,
    device: str,               # auto|cpu|cuda
    quiet: bool,
    strict_cuda: bool,
) -> Tuple[WhisperModel, str, str]:
    """
    Returns: (model, device_used, compute_type_used)
    """
    if device == "cpu":
        compute_type = "int8"
        return WhisperModel(model_name, device="cpu", compute_type=compute_type), "cpu", compute_type

    if device == "cuda":
        try:
            compute_type = "float16"
            m = WhisperModel(model_name, device="cuda", compute_type=compute_type)
            log("   Using device=cuda compute_type=float16", quiet=quiet)
            return m, "cuda", compute_type
        except Exception as e:
            if strict_cuda:
                raise RuntimeError(f"CUDA requested but init failed: {e}") from e
            log(f"   CUDA init failed; falling back to CPU. Reason: {e}", quiet=quiet)
            compute_type = "int8"
            return WhisperModel(model_name, device="cpu", compute_type=compute_type), "cpu", compute_type

    # auto
    try:
        compute_type = "float16"
        m = WhisperModel(model_name, device="cuda", compute_type=compute_type)
        log("   CUDA available: using device=cuda compute_type=float16", quiet=quiet)
        return m, "cuda", compute_type
    except Exception as e:
        log(f"   CUDA not available; using CPU. Reason: {e}", quiet=quiet)
        compute_type = "int8"
        return WhisperModel(model_name, device="cpu", compute_type=compute_type), "cpu", compute_type


# ============================================================
# Text normalization / chunking
# ============================================================

def normalize_spaces(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wrap_text_lines(text: str, max_chars_per_line: int) -> List[str]:
    text = normalize_spaces(text)
    if not text:
        return []
    words = text.split(" ")
    lines: List[str] = []
    cur: List[str] = []

    def cur_len() -> int:
        return len(" ".join(cur)) if cur else 0

    for w in words:
        if not cur:
            cur = [w]
            continue
        if cur_len() + 1 + len(w) <= max_chars_per_line:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def block_fits(text: str, max_chars_per_line: int, max_lines: int) -> bool:
    return len(wrap_text_lines(text, max_chars_per_line)) <= max_lines


def wrap_fallback_blocks(text: str, max_chars_per_line: int, max_lines: int) -> List[str]:
    lines = wrap_text_lines(text, max_chars_per_line)
    blocks: List[str] = []
    for i in range(0, len(lines), max_lines):
        blocks.append(" ".join(lines[i:i + max_lines]))
    return blocks


def split_on_delims(text: str, delims: str) -> List[str]:
    text = normalize_spaces(text)
    if not text:
        return []
    pattern = re.compile(rf".+?(?:[{re.escape(delims)}]+)(?=\s+)")
    parts: List[str] = []
    last_end = 0
    for m in pattern.finditer(text):
        end = m.end()
        chunk = text[last_end:end].strip()
        if chunk:
            parts.append(chunk)
        ws = re.match(r"\s+", text[end:])
        last_end = end + (ws.end() if ws else 0)

    rem = text[last_end:].strip()
    if rem:
        parts.append(rem)

    parts = [p for p in parts if re.search(r"\w", p)]
    return parts


def split_text_into_blocks(
    text: str,
    max_chars_per_line: int,
    max_lines: int,
    allow_commas: bool = True,
    allow_medium: bool = True,
    prefer_punct_splits: bool = False,
) -> List[str]:
    text = normalize_spaces(text)
    if not text:
        return []

    tiers: List[str] = [".?!"]
    if allow_medium:
        tiers.append(";:")
    if allow_commas:
        tiers.append(",")

    def refine_chunk(chunk: str, tier_index: int) -> List[str]:
        chunk = normalize_spaces(chunk)
        if not chunk:
            return []

        fits = block_fits(chunk, max_chars_per_line, max_lines)
        if fits and not (prefer_punct_splits and tier_index == 0):
            return [chunk]

        if tier_index >= len(tiers):
            return wrap_fallback_blocks(chunk, max_chars_per_line, max_lines)

        parts = split_on_delims(chunk, tiers[tier_index])
        if len(parts) <= 1:
            return refine_chunk(chunk, tier_index + 1)

        out: List[str] = []
        for p in parts:
            out.extend(refine_chunk(p, tier_index + 1))
        return out

    blocks = refine_chunk(text, 0)

    safe: List[str] = []
    for b in blocks:
        if block_fits(b, max_chars_per_line, max_lines):
            safe.append(b)
        else:
            safe.extend(wrap_fallback_blocks(b, max_chars_per_line, max_lines))
    return safe


def preferred_split_index(text: str) -> int:
    punct = [". ", "? ", "! ", "; ", ": ", ", "]
    for p in punct:
        idx = text.rfind(p)
        if idx != -1 and idx > 20:
            return idx + len(p)
    sp = text.rfind(" ")
    if sp > 20:
        return sp + 1
    return -1


def distribute_time(start: float, end: float, parts: List[str]) -> List[Tuple[float, float, str]]:
    total = max(1, sum(len(p) for p in parts))
    dur = max(0.0, end - start)
    out: List[Tuple[float, float, str]] = []
    t = start
    for i, p in enumerate(parts):
        frac = len(p) / total
        seg_dur = dur * frac if i < len(parts) - 1 else (end - t)
        out.append((t, t + seg_dur, p))
        t += seg_dur
    return out


def enforce_timing(
    blocks: List[Tuple[float, float, str]],
    min_dur: float,
    max_dur: float,
    *,
    split_long: bool = True,
) -> List[Tuple[float, float, str]]:
    merged: List[Tuple[float, float, str]] = []
    i = 0
    while i < len(blocks):
        s, e, txt = blocks[i]
        dur = e - s
        if dur < min_dur and i + 1 < len(blocks):
            ns, ne, ntxt = blocks[i + 1]
            merged.append((s, ne, normalize_spaces(txt + " " + ntxt)))
            i += 2
            continue
        merged.append((s, e, txt))
        i += 1

    if not split_long:
        return merged

    final: List[Tuple[float, float, str]] = []
    for s, e, txt in merged:
        dur = e - s
        if dur > max_dur and len(txt) > 120:
            cut = preferred_split_index(txt)
            if cut == -1:
                cut = len(txt) // 2
            p1 = normalize_spaces(txt[:cut])
            p2 = normalize_spaces(txt[cut:])
            parts = [p1, p2] if p2 else [p1]
            final.extend(distribute_time(s, e, parts))
        else:
            final.append((s, e, txt))
    return final


@dataclass
class SubtitleBlock:
    start: float
    end: float
    lines: List[str]


@dataclass
class WordItem:
    start: float
    end: float
    text: str


def collect_words(segments: List[Any]) -> List[WordItem]:
    words: List[WordItem] = []
    for seg in segments:
        for w in getattr(seg, "words", []) or []:
            txt = normalize_spaces(getattr(w, "word", ""))
            if not txt:
                continue
            words.append(WordItem(start=float(w.start), end=float(w.end), text=txt))
    return words


def words_to_text(words: List[WordItem]) -> str:
    return normalize_spaces(" ".join(w.text for w in words))


def silence_between(
    start: float,
    end: float,
    silences: List[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    for s, e in silences:
        if s >= start and e <= end:
            return (s, e)
    return None


def split_words_on_silence(
    words: List[WordItem],
    silences: List[Tuple[float, float]],
) -> List[List[WordItem]]:
    if not words:
        return []
    if not silences:
        return [words]
    runs: List[List[WordItem]] = []
    cur: List[WordItem] = [words[0]]
    for w in words[1:]:
        gap = silence_between(cur[-1].end, w.start, silences)
        if gap is not None:
            runs.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        runs.append(cur)
    return runs


def map_text_blocks_to_word_spans(
    blocks: List[str],
    words: List[WordItem],
) -> List[Tuple[float, float, str]]:
    out: List[Tuple[float, float, str]] = []
    idx = 0
    total = len(words)
    for block in blocks:
        count = len(normalize_spaces(block).split())
        if count <= 0:
            continue
        if idx >= total:
            break
        take = min(count, total - idx)
        chunk = words[idx:idx + take]
        idx += take
        if not chunk:
            continue
        out.append((chunk[0].start, chunk[-1].end, block))
    return out


def chunk_segments_to_subtitles(
    segments: List[Any],
    cfg: ResolvedConfig,
) -> List[SubtitleBlock]:
    raw: List[Tuple[float, float, str]] = []
    for seg in segments:
        txt = normalize_spaces(getattr(seg, "text", ""))
        if not txt:
            continue
        raw.append((float(seg.start), float(seg.end), txt))

    split_raw: List[Tuple[float, float, str]] = []
    for s, e, txt in raw:
        parts = split_text_into_blocks(
            txt,
            cfg.max_chars,
            cfg.max_lines,
            allow_commas=cfg.allow_commas,
            allow_medium=cfg.allow_medium,
            prefer_punct_splits=cfg.prefer_punct_splits,
        )
        if len(parts) == 1:
            split_raw.append((s, e, txt))
        else:
            split_raw.extend(distribute_time(s, e, parts))

    split_raw = enforce_timing(split_raw, cfg.min_dur, cfg.max_dur)

    density_fixed: List[Tuple[float, float, str]] = []
    for s, e, txt in split_raw:
        dur = max(0.01, e - s)
        cps = len(txt) / dur
        if cps > cfg.target_cps and len(txt) > 80:
            cut = preferred_split_index(txt)
            if cut == -1:
                cut = len(txt) // 2
            p1 = normalize_spaces(txt[:cut])
            p2 = normalize_spaces(txt[cut:])
            parts = [p1, p2] if p2 else [p1]
            density_fixed.extend(distribute_time(s, e, parts))
        else:
            density_fixed.append((s, e, txt))

    subs: List[SubtitleBlock] = []
    for s, e, txt in density_fixed:
        txt = normalize_spaces(txt)
        parts = split_text_into_blocks(
            txt,
            cfg.max_chars,
            cfg.max_lines,
            allow_commas=cfg.allow_commas,
            allow_medium=cfg.allow_medium,
            prefer_punct_splits=cfg.prefer_punct_splits,
        )
        timed_parts = distribute_time(s, e, parts) if len(parts) > 1 else [(s, e, txt)]
        for ps, pe, ptxt in timed_parts:
            lines = wrap_text_lines(ptxt, cfg.max_chars)
            if len(lines) > cfg.max_lines:
                lines = lines[:cfg.max_lines]
            subs.append(SubtitleBlock(start=float(ps), end=float(pe), lines=lines))
    return subs


def chunk_words_to_subtitles(
    words: List[WordItem],
    cfg: ResolvedConfig,
    silences: List[Tuple[float, float]],
) -> List[SubtitleBlock]:
    runs = split_words_on_silence(words, silences)
    subs: List[SubtitleBlock] = []
    for run in runs:
        text = words_to_text(run)
        parts = split_text_into_blocks(
            text,
            cfg.max_chars,
            cfg.max_lines,
            allow_commas=cfg.allow_commas,
            allow_medium=cfg.allow_medium,
            prefer_punct_splits=cfg.prefer_punct_splits,
        )
        timed_parts = map_text_blocks_to_word_spans(parts, run)
        timed_parts = enforce_timing(
            timed_parts,
            cfg.min_dur,
            cfg.max_dur,
            split_long=False,
        )
        for s, e, ptxt in timed_parts:
            lines = wrap_text_lines(ptxt, cfg.max_chars)
            if len(lines) > cfg.max_lines:
                lines = lines[:cfg.max_lines]
            subs.append(SubtitleBlock(start=float(s), end=float(e), lines=lines))
    return subs


def words_to_subtitles(words: List[WordItem]) -> List[SubtitleBlock]:
    subs: List[SubtitleBlock] = []
    for w in words:
        txt = normalize_spaces(w.text)
        if not txt:
            continue
        subs.append(SubtitleBlock(start=float(w.start), end=float(w.end), lines=[txt]))
    return subs


def apply_silence_alignment(
    subs: List[SubtitleBlock],
    silences: List[Tuple[float, float]],
) -> List[SubtitleBlock]:
    if not subs or not silences:
        return subs

    aligned: List[SubtitleBlock] = []
    for i, sb in enumerate(subs):
        s = sb.start
        e = sb.end

        # Clamp if start/end sit inside detected silence.
        for ss, ee in silences:
            if s >= ss and s <= ee:
                s = ee
            if e >= ss and e <= ee:
                e = ss

        # If silence exists between this and next block, align to the gap.
        if i + 1 < len(subs):
            gap = silence_between(e, subs[i + 1].start, silences)
            if gap is not None:
                s_start, s_end = gap
                e = min(e, s_start)
                next_start = max(subs[i + 1].start, s_end)
                subs[i + 1].start = next_start

        if e < s + 0.001:
            e = s + 0.001
        aligned.append(SubtitleBlock(s, e, sb.lines))

    return aligned

# ============================================================
# Timing polish + hygiene
# ============================================================

def subs_text(sb: SubtitleBlock) -> str:
    return normalize_spaces(" ".join(sb.lines))


def hygiene_and_polish(
    subs: List[SubtitleBlock],
    *,
    min_gap: float,
    pad: float,
    silence_intervals: Optional[List[Tuple[float, float]]] = None,
) -> List[SubtitleBlock]:
    # Remove empties / normalize
    cleaned: List[SubtitleBlock] = []
    for sb in subs:
        txt = subs_text(sb)
        if not txt:
            continue
        s = max(0.0, float(sb.start))
        e = max(s + 0.001, float(sb.end))
        cleaned.append(SubtitleBlock(s, e, wrap_text_lines(txt, 10_000)))  # keep as single line temporarily

    if not cleaned:
        return []

    # Sort by time
    cleaned.sort(key=lambda x: (x.start, x.end))

    # Merge identical consecutive blocks unless a silence gap is present
    merged: List[SubtitleBlock] = []
    for sb in cleaned:
        if merged and subs_text(merged[-1]) == subs_text(sb):
            if silence_intervals and silence_between(merged[-1].end, sb.start, silence_intervals):
                merged.append(sb)
            else:
                merged[-1].end = max(merged[-1].end, sb.end)
        else:
            merged.append(sb)

    # Apply padding and enforce gaps without overlap
    out: List[SubtitleBlock] = []
    for i, sb in enumerate(merged):
        prev_end = out[-1].end if out else None
        next_start = merged[i + 1].start if i + 1 < len(merged) else None

        s = sb.start
        e = sb.end

        # Pad into silence where possible
        if pad > 0:
            if prev_end is not None:
                s = max(prev_end + min_gap, s - pad)
            else:
                s = max(0.0, s - pad)

            if next_start is not None:
                e = min(next_start - min_gap, e + pad)
            else:
                e = e + pad

        # Enforce min gap to previous unless silence gap already exists
        if prev_end is not None:
            if not silence_intervals or not silence_between(prev_end, s, silence_intervals):
                if s < prev_end + min_gap:
                    s = prev_end + min_gap
                    if e < s + 0.001:
                        e = s + 0.001

        # Enforce min gap to next unless silence gap already exists
        if next_start is not None:
            if not silence_intervals or not silence_between(e, next_start, silence_intervals):
                if e > next_start - min_gap:
                    e = max(s + 0.001, next_start - min_gap)

        out.append(SubtitleBlock(s, e, sb.lines))

    # Final monotonic clamp
    final: List[SubtitleBlock] = []
    last_end = 0.0
    for sb in out:
        s = max(last_end, sb.start)
        e = max(s + 0.001, sb.end)
        final.append(SubtitleBlock(s, e, sb.lines))
        last_end = e

    return final


# ============================================================
# Writers (atomic)
# ============================================================

def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_vtt_time(seconds: float) -> str:
    # WebVTT: HH:MM:SS.mmm
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_ass_time(seconds: float) -> str:
    cs = int(round(seconds * 100))
    h = cs // 360_000
    cs %= 360_000
    m = cs // 6_000
    cs %= 6_000
    s = cs // 100
    cs %= 100
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def atomic_write_text(path: Path, content: str) -> None:
    ensure_parent_dir(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def write_srt(subs: List[SubtitleBlock], out_path: Path, *, max_chars: int, max_lines: int) -> None:
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
    # Plain transcript derived from subtitle blocks (chronological)
    lines: List[str] = []
    for sb in subs:
        lines.append(normalize_spaces(" ".join(sb.lines)))
    atomic_write_text(out_path, "\n".join(lines).strip() + "\n")


def segments_to_jsonable(segments: List[Any], *, include_words: bool) -> List[Dict[str, Any]]:
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
) -> None:
    payload = {
        "tool_version": TOOL_VERSION,
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


# ============================================================
# Input expansion / batch planning
# ============================================================

MEDIA_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".mp4", ".mkv", ".mov", ".webm", ".m4v"}


def iter_media_files_in_dir(d: Path) -> Iterable[Path]:
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
            yield p


def expand_inputs(inputs: List[str], glob_pat: Optional[str]) -> List[Path]:
    out: List[Path] = []
    for s in inputs:
        p = Path(s)
        if p.exists() and p.is_dir():
            out.extend(list(iter_media_files_in_dir(p)))
        elif any(ch in s for ch in ["*", "?", "["]) and not p.exists():
            out.extend([Path(x) for x in glob.glob(s)])
        else:
            out.append(p)

    if glob_pat:
        out.extend([Path(x) for x in glob.glob(glob_pat)])

    # de-dupe, preserve order
    seen = set()
    uniq: List[Path] = []
    for p in out:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def default_output_for(input_file: Path, outdir: Optional[Path], fmt: str, keep_structure: bool, base_root: Optional[Path]) -> Path:
    suffix = {
        "srt": ".srt",
        "vtt": ".vtt",
        "txt": ".txt",
        "ass": ".ass",
        "json": ".json",
    }[fmt]

    if outdir is None:
        return input_file.with_suffix(suffix)

    if keep_structure and base_root and input_file.is_absolute():
        try:
            rel = input_file.relative_to(base_root)
        except Exception:
            rel = Path(input_file.name)
    elif keep_structure and base_root:
        try:
            rel = input_file.resolve().relative_to(base_root.resolve())
        except Exception:
            rel = Path(input_file.name)
    else:
        rel = Path(input_file.name)

    return (outdir / rel).with_suffix(suffix)


# ============================================================
# Run one file
# ============================================================

def preflight_one(input_path: Path, output_path: Path, overwrite: bool) -> Tuple[bool, str]:
    if not input_path.exists():
        return False, f"Input file not found: {input_path}"
    if input_path.is_dir():
        return False, f"Input path is a directory (expected media file): {input_path}"
    if output_path.exists() and output_path.is_dir():
        return False, f"Output path is a directory (expected file): {output_path}"
    if output_path.exists() and not overwrite:
        return False, f"Output already exists: {output_path} (use --overwrite)"
    return True, ""


def run_one(
    *,
    input_path: Path,
    output_path: Path,
    fmt: str,
    transcript_path: Optional[Path],
    segments_path: Optional[Path],
    json_bundle_path: Optional[Path],
    args: argparse.Namespace,
    cfg: ResolvedConfig,
    quiet: bool,
    show_progress: bool,
) -> int:
    if not ffmpeg_ok():
        return die("ffmpeg not found on PATH. Install it or add it to PATH.", 2)

    ensure_parent_dir(output_path)
    if transcript_path:
        ensure_parent_dir(transcript_path)
    if segments_path:
        ensure_parent_dir(segments_path)
    if json_bundle_path:
        ensure_parent_dir(json_bundle_path)

    tmpdir = args.tmpdir if args.tmpdir else None
    fd, tmp_wav = tempfile.mkstemp(prefix="srtgen_", suffix=".wav", dir=tmpdir)
    os.close(fd)

    started = time.time()
    try:
        log(f"Input: {input_path}", quiet=quiet)
        log(f"Output: {output_path}", quiet=quiet)

        if args.dry_run:
            log("Dry run: skipping transcription.", quiet=quiet)
            return 0

        log("1/5 Converting audio with ffmpeg...", quiet=quiet)
        to_wav_16k_mono(str(input_path), tmp_wav)

        log("2/5 Loading model...", quiet=quiet)
        model, device_used, compute_type_used = init_whisper_model(
            model_name=args.model,
            device=args.device,
            quiet=quiet,
            strict_cuda=args.strict_cuda,
        )

        log("3/5 Transcribing...", quiet=quiet)
        t0 = time.time()

        segments_iter, info = model.transcribe(
            tmp_wav,
            vad_filter=cfg.vad_filter,
            language=args.language,
            word_timestamps=cfg.word_timestamps,
        )

        seg_list: List[Any] = []
        dur_total = probe_duration_seconds(str(tmp_wav))
        last_ratio = 0.0

        # Progress uses:
        # - segment time in media (seg.end)
        # - wall elapsed
        # - realtime factor = seg.end / elapsed
        # - ETA in wall-time = (dur_total - seg.end) / realtime_factor
        for idx, seg in enumerate(segments_iter, start=1):
            seg_list.append(seg)

            pct = ""
            eta = ""
            rt = ""

            now = time.time()
            elapsed = max(0.001, now - t0)
            media_t = float(getattr(seg, "end", 0.0))
            if media_t > 0:
                rtf = media_t / elapsed  # media seconds per wall second
                rt = f"{rtf:4.2f}x"

            if dur_total and dur_total > 0:
                ratio = media_t / dur_total
                ratio = max(last_ratio, ratio)
                ratio = min(1.0, ratio)
                last_ratio = ratio
                pct = f"{ratio * 100:5.1f}%"

                if media_t > 0:
                    rtf = media_t / elapsed
                    if rtf > 0.01:
                        remaining_media = max(0.0, dur_total - media_t)
                        eta_sec = remaining_media / rtf
                        eta = f"ETA {format_duration(eta_sec)}"

            progress_line(
                f"   {pct} segs:{idx:5d} | media_t={format_duration(media_t)} | {rt} {eta}",
                enabled=show_progress,
                quiet=quiet,
            )

        progress_done(enabled=show_progress, quiet=quiet)
        log(f"   Transcription complete: {len(seg_list)} segments in {format_duration(time.time() - t0)}", quiet=quiet)

        log("4/5 Chunking + formatting...", quiet=quiet)
        t1 = time.time()
        silences: List[Tuple[float, float]] = []
        if cfg.use_silence_split:
            silences = detect_silences(
                tmp_wav,
                min_silence_dur=cfg.silence_min_dur,
                silence_threshold_db=cfg.silence_threshold_db,
            )

        words = collect_words(seg_list) if cfg.word_timestamps else []
        if args.word_level:
            if not words:
                return die("Word-level output requested but no word timestamps are available.", 2)
            subs = words_to_subtitles(words)
        else:
            if cfg.use_silence_split and words:
                subs = chunk_words_to_subtitles(words, cfg, silences)
            else:
                if cfg.use_silence_split and not words:
                    warn("Silence splitting requested but no word timestamps returned; falling back.", quiet=quiet)
                subs = chunk_segments_to_subtitles(seg_list, cfg)

        if cfg.use_silence_split and silences:
            subs = apply_silence_alignment(subs, silences)
        subs = hygiene_and_polish(
            subs,
            min_gap=cfg.min_gap,
            pad=cfg.pad,
            silence_intervals=silences if cfg.use_silence_split else None,
        )
        log(f"   Chunking complete: {len(subs)} subtitle blocks in {format_duration(time.time() - t1)}", quiet=quiet)

        log("5/5 Writing outputs...", quiet=quiet)

        if fmt == "srt":
            write_srt(subs, output_path, max_chars=cfg.max_chars, max_lines=cfg.max_lines)
        elif fmt == "vtt":
            write_vtt(subs, output_path, max_chars=cfg.max_chars, max_lines=cfg.max_lines)
        elif fmt == "ass":
            write_ass(subs, output_path, max_chars=cfg.max_chars, max_lines=cfg.max_lines)
        elif fmt == "txt":
            write_txt(subs, output_path)
        elif fmt == "json":
            write_json_bundle(
                output_path,
                input_file=str(input_path),
                device_used=device_used,
                compute_type_used=compute_type_used,
                cfg=cfg,
                segments=seg_list,
                subs=subs,
            )
        else:
            return die(f"Unknown format: {fmt}", 2)

        # Optional side outputs
        if transcript_path:
            # transcript from subtitle blocks
            write_txt(subs, transcript_path)

        if segments_path:
            ensure_parent_dir(segments_path)
            tmp = segments_path.with_suffix(segments_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "input_file": str(input_path),
                        "segments": segments_to_jsonable(seg_list, include_words=cfg.word_timestamps),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(tmp, segments_path)

        if json_bundle_path:
            write_json_bundle(
                json_bundle_path,
                input_file=str(input_path),
                device_used=device_used,
                compute_type_used=compute_type_used,
                cfg=cfg,
                segments=seg_list,
                subs=subs,
            )

        log(f"Done: {output_path} (total {format_duration(time.time() - started)})", quiet=quiet)
        return 0

    finally:
        if not args.keep_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


# ============================================================
# Diagnose / Version
# ============================================================

def diagnose() -> None:
    print(f"tool_version: {TOOL_VERSION}")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"ffmpeg: {ffmpeg_version()}")
    print(f"ffprobe: {ffprobe_version()}")
    try:
        import faster_whisper  # type: ignore
        print(f"faster_whisper: {getattr(faster_whisper, '__version__', 'unknown')}")
    except Exception:
        print("faster_whisper: (unable to read version)")
    print("PATH ffmpeg:", which_or_none("ffmpeg"))
    print("PATH ffprobe:", which_or_none("ffprobe"))


def list_downloaded_models() -> List[Tuple[str, str]]:
    downloaded: List[Tuple[str, str]] = []
    for name in fw_utils.available_models():
        try:
            path = fw_utils.download_model(name, local_files_only=True)
        except Exception:
            continue
        if path and os.path.exists(path):
            downloaded.append((name, path))
    return downloaded


def list_available_models() -> List[str]:
    return list(fw_utils.available_models())


def download_model_cli(model_name: str) -> int:
    try:
        path = fw_utils.download_model(model_name, local_files_only=False)
    except Exception as e:
        return die(f"Failed to download model '{model_name}': {e}", 2)
    print(f"Downloaded {model_name} to {path}")
    return 0


def delete_model_cli(model_name: str) -> int:
    try:
        path = fw_utils.download_model(model_name, local_files_only=True)
    except Exception:
        return die(f"Model '{model_name}' is not downloaded.", 2)
    if not path or not os.path.exists(path):
        return die(f"Model '{model_name}' is not downloaded.", 2)
    try:
        shutil.rmtree(path)
    except Exception as e:
        return die(f"Failed to delete model '{model_name}': {e}", 2)
    print(f"Deleted cached model: {model_name}")
    return 0


# ============================================================
# CLI
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Local SRT/VTT generator (faster-whisper + ffmpeg)")
    ap.add_argument("inputs", nargs="*", help="Media file(s), directory, or glob pattern(s)")
    ap.add_argument("--glob", default=None, help="Additional glob pattern to include (optional)")
    ap.add_argument("--outdir", default=None, help="Output directory (batch mode). If omitted, writes next to input.")
    ap.add_argument("--keep-structure", action="store_true", help="When using --outdir, preserve directory structure.")
    ap.add_argument("--root", default=None, help="Base root for --keep-structure (defaults to common parent when possible).")

    ap.add_argument("-o", "--output", default=None, help="Single-file output path (only valid when one input expands to one file).")

    ap.add_argument("--format", choices=["srt", "vtt", "ass", "txt", "json"], default="srt", help="Primary output format")
    ap.add_argument("--emit-transcript", default=None, help="Also write a transcript TXT to this path (or outdir-based if a directory).")
    ap.add_argument("--emit-segments", default=None, help="Also write segments JSON to this path (or outdir-based if a directory).")
    ap.add_argument("--emit-bundle", default=None, help="Also write a full JSON bundle (segments+subs+config) to this path (or outdir-based if a directory).")

    ap.add_argument("--model", default="small", help="tiny/base/small/medium/large-v3")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="auto/cpu/cuda")
    ap.add_argument("--strict-cuda", action="store_true", help="If set, fail instead of falling back when CUDA init fails.")
    ap.add_argument("--language", default=None, help="Optional language code (e.g., en). If omitted, auto-detect.")
    ap.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Request word timestamps (preserved in JSON outputs).",
    )
    ap.add_argument("--word-level", action="store_true", help="Output word-level subtitle cues (requires word timestamps).")

    ap.add_argument("--mode", default=None, help="Preset modes: shorts | yt | podcast (aliases: short, youtube, pod).")
    ap.add_argument("--config", default=None, help="JSON config file. CLI args override config.")
    ap.add_argument("--dry-run", action="store_true", help="Validate inputs and show resolved settings, but do not transcribe.")

    ap.add_argument("--max_chars", type=int, default=None)
    ap.add_argument("--max_lines", type=int, default=None)
    ap.add_argument("--target_cps", type=float, default=None)
    ap.add_argument("--min_dur", type=float, default=None)
    ap.add_argument("--max_dur", type=float, default=None)

    ap.add_argument("--no-comma-split", action="store_true")
    ap.add_argument("--no-medium-split", action="store_true")
    ap.add_argument("--prefer-punct-splits", action="store_true")

    ap.add_argument("--min-gap", type=float, default=None, help="Minimum gap between consecutive subtitle cues (seconds).")
    ap.add_argument("--pad", type=float, default=None, help="Pad cues into silence where possible (seconds).")
    ap.add_argument("--no-silence-split", action="store_true", help="Disable silence-based splitting/alignment.")
    ap.add_argument("--silence-min-dur", type=float, default=None, help="Minimum silence duration for splits (seconds).")
    ap.add_argument("--silence-threshold", type=float, default=None, help="Silence threshold in dB (e.g., -35).")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    ap.add_argument("--keep_wav", action="store_true", help="Do not delete temporary WAV file")
    ap.add_argument("--tmpdir", default=None, help="Directory for temporary WAV (defaults to system temp)")

    ap.add_argument("--list-models", action="store_true", help="List downloaded faster-whisper models and exit.")
    ap.add_argument("--list-available-models", action="store_true", help="List available faster-whisper model names and exit.")
    ap.add_argument("--download-model", default=None, help="Download a faster-whisper model and exit.")
    ap.add_argument("--delete-model", default=None, help="Delete a downloaded model from cache and exit.")

    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true", help="Batch mode: continue processing other files on error.")
    ap.add_argument("--version", action="store_true")
    ap.add_argument("--diagnose", action="store_true")

    args = ap.parse_args()

    if args.version:
        print(TOOL_VERSION)
        return 0

    if args.diagnose:
        diagnose()
        return 0

    if args.list_models or args.list_available_models or args.download_model or args.delete_model:
        if args.inputs:
            return die("Model management options must be used without input files.", 2)
        rc = 0
        if args.list_models:
            downloaded = list_downloaded_models()
            if downloaded:
                print("Downloaded models:")
                for name, path in downloaded:
                    print(f"  - {name}: {path}")
            else:
                print("No downloaded models found.")
            print("Available models:")
            print("  " + ", ".join(list_available_models()))
        if args.list_available_models and not args.list_models:
            print("Available models:")
            print("  " + ", ".join(list_available_models()))
        if args.download_model:
            rc = download_model_cli(args.download_model)
            if rc != 0:
                return rc
        if args.delete_model:
            rc = delete_model_cli(args.delete_model)
            if rc != 0:
                return rc
        return rc

    quiet = args.quiet
    show_progress = not args.no_progress

    if args.mode:
        mode_key = args.mode.lower()
        if mode_key not in MODE_ALIASES:
            return die(
                f"Invalid --mode '{args.mode}'. "
                f"Valid modes: shorts, yt, podcast",
                code=2,
            )
        args.mode = MODE_ALIASES[mode_key]

    # Expand inputs (files, dirs, globs)
    if not args.inputs:
        return die("No input files provided.", 2)
    files = expand_inputs(args.inputs, args.glob)
    files = [p for p in files if p.exists() and p.is_file()]
    if not files:
        return die("No input files found after expansion.", 2)

    # Compute outdir/root
    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        ensure_parent_dir(outdir / "dummy.txt")  # ensures outdir's parent exists
        outdir.mkdir(parents=True, exist_ok=True)

    if args.root:
        base_root = Path(args.root)
    else:
        # best-effort common parent for keep-structure
        base_root = None
        if args.keep_structure and len(files) > 1:
            try:
                base_root = Path(os.path.commonpath([str(f.resolve()) for f in files]))
                if base_root.is_file():
                    base_root = base_root.parent
            except Exception:
                base_root = None

    # Build config: defaults -> config file -> preset -> CLI overrides
    cfg = ResolvedConfig()

    try:
        cfg_file = load_config_file(args.config)
    except Exception as e:
        return die(str(e), 2)

    cfg = apply_overrides(cfg, cfg_file)

    if args.mode:
        cfg = apply_overrides(cfg, PRESETS[args.mode])

    # CLI overrides
    if args.max_chars is not None:
        cfg.max_chars = args.max_chars
    if args.max_lines is not None:
        cfg.max_lines = args.max_lines
    if args.target_cps is not None:
        cfg.target_cps = args.target_cps
    if args.min_dur is not None:
        cfg.min_dur = args.min_dur
    if args.max_dur is not None:
        cfg.max_dur = args.max_dur

    if args.no_comma_split:
        cfg.allow_commas = False
    if args.no_medium_split:
        cfg.allow_medium = False
    if args.prefer_punct_splits:
        cfg.prefer_punct_splits = True

    if args.min_gap is not None:
        cfg.min_gap = float(args.min_gap)
    if args.pad is not None:
        cfg.pad = float(args.pad)

    if args.no_silence_split:
        cfg.use_silence_split = False
    if args.silence_min_dur is not None:
        cfg.silence_min_dur = float(args.silence_min_dur)
    if args.silence_threshold is not None:
        cfg.silence_threshold_db = float(args.silence_threshold)

    if args.word_timestamps:
        cfg.word_timestamps = True
    if args.word_level or cfg.use_silence_split:
        cfg.word_timestamps = True

    # Basic dependency check
    if not ffmpeg_ok():
        return die("ffmpeg not found on PATH. Install it or add it to PATH.", 2)

    # If user provided --output, require single file expansion
    if args.output is not None and len(files) != 1:
        return die("--output may only be used when exactly one input file is provided (after expansion).", 2)

    # Show resolved config for transparency
    if args.dry_run and not quiet:
        log("Resolved config:", quiet=quiet)
        log(json.dumps(dataclasses.asdict(cfg), indent=2), quiet=quiet)

    # Run (single or batch)
    failures: List[Tuple[Path, str]] = []
    for f in files:
        # Determine output path(s)
        if args.output:
            primary_out = Path(args.output)
        else:
            primary_out = default_output_for(f, outdir, args.format, args.keep_structure, base_root)

        ok, reason = preflight_one(f, primary_out, args.overwrite)
        if not ok:
            failures.append((f, reason))
            if not args.continue_on_error:
                return die(reason, 2)
            warn(f"{f}: {reason}", quiet=quiet)
            continue

        # Side outputs
        def side_path(user_path: Optional[str], ext: str) -> Optional[Path]:
            if not user_path:
                return None
            p = Path(user_path)
            if p.exists() and p.is_dir():
                # directory: mirror naming there
                return (p / f.name).with_suffix(ext)
            if user_path.endswith(os.sep) or user_path.endswith("/"):
                # treat as directory-like
                p.mkdir(parents=True, exist_ok=True)
                return (p / f.name).with_suffix(ext)
            # explicit file path
            return p

        transcript_out = side_path(args.emit_transcript, ".txt")
        segments_out = side_path(args.emit_segments, ".segments.json")
        bundle_out = side_path(args.emit_bundle, ".bundle.json") if args.emit_bundle else None

        try:
            rc = run_one(
                input_path=f,
                output_path=primary_out,
                fmt=args.format,
                transcript_path=transcript_out,
                segments_path=segments_out,
                json_bundle_path=bundle_out,
                args=args,
                cfg=cfg,
                quiet=quiet,
                show_progress=show_progress,
            )
            if rc != 0:
                failures.append((f, f"failed with exit code {rc}"))
                if not args.continue_on_error:
                    return rc
        except KeyboardInterrupt:
            return die("Interrupted by user.", 130)
        except Exception as e:
            if args.debug:
                traceback.print_exc()
            failures.append((f, str(e)))
            if not args.continue_on_error:
                return die(f"{f}: {e}", 1)
            warn(f"{f}: {e}", quiet=quiet)

    if failures:
        if not quiet:
            log("\nSummary: failures:", quiet=quiet)
            for f, msg in failures:
                log(f"  - {f}: {msg}", quiet=quiet)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#END OF FILE
