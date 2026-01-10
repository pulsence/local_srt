#!/usr/bin/env python3
"""Subtitle generation and timing logic for Local SRT.

This module converts transcription segments and words into properly
timed and formatted subtitle blocks.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .models import ResolvedConfig, SubtitleBlock, WordItem
from .text_processing import (
    distribute_time,
    enforce_timing,
    normalize_spaces,
    preferred_split_index,
    split_text_into_blocks,
    wrap_text_lines,
)


# ============================================================
# Word Collection
# ============================================================

def collect_words(segments: List[Any]) -> List[WordItem]:
    """Extract word-level timing information from transcription segments.

    Args:
        segments: List of transcription segment objects from faster-whisper

    Returns:
        List of WordItem objects with word-level timing
    """
    words: List[WordItem] = []
    for seg in segments:
        for w in getattr(seg, "words", []) or []:
            txt = normalize_spaces(getattr(w, "word", ""))
            if not txt:
                continue
            words.append(WordItem(start=float(w.start), end=float(w.end), text=txt))
    return words


def words_to_text(words: List[WordItem]) -> str:
    """Convert a list of words to a single normalized text string.

    Args:
        words: List of WordItem objects

    Returns:
        Concatenated normalized text
    """
    return normalize_spaces(" ".join(w.text for w in words))


# ============================================================
# Silence-Based Splitting
# ============================================================

def silence_between(
    start: float,
    end: float,
    silences: List[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """Find a silence interval that falls entirely between start and end times.

    Args:
        start: Start time to search from
        end: End time to search to
        silences: List of (start, end) silence intervals

    Returns:
        First silence interval found within range, or None
    """
    for s, e in silences:
        if s >= start and e <= end:
            return (s, e)
    return None


def split_words_on_silence(
    words: List[WordItem],
    silences: List[Tuple[float, float]],
) -> List[List[WordItem]]:
    """Split a list of words into runs separated by silence intervals.

    Args:
        words: List of WordItem objects to split
        silences: List of (start, end) silence intervals

    Returns:
        List of word runs (each run is a List[WordItem])
    """
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
    """Map text blocks to word spans to get accurate timing from word timestamps.

    Args:
        blocks: List of text blocks (from split_text_into_blocks)
        words: List of WordItem objects with timing

    Returns:
        List of (start, end, text) tuples with word-level timing
    """
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


# ============================================================
# Segment-Based Subtitle Generation
# ============================================================

def chunk_segments_to_subtitles(
    segments: List[Any],
    cfg: ResolvedConfig,
) -> List[SubtitleBlock]:
    """Convert transcription segments into subtitle blocks using segment-level timing.

    Args:
        segments: List of transcription segment objects
        cfg: Configuration settings

    Returns:
        List of SubtitleBlock objects
    """
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


# ============================================================
# Word-Based Subtitle Generation
# ============================================================

def chunk_words_to_subtitles(
    words: List[WordItem],
    cfg: ResolvedConfig,
    silences: List[Tuple[float, float]],
) -> List[SubtitleBlock]:
    """Convert word-level timing into subtitle blocks using silence-aware splitting.

    Args:
        words: List of WordItem objects with word-level timing
        cfg: Configuration settings
        silences: List of (start, end) silence intervals

    Returns:
        List of SubtitleBlock objects
    """
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
    """Convert words directly to subtitle blocks (one word per subtitle).

    Args:
        words: List of WordItem objects

    Returns:
        List of SubtitleBlock objects (one per word)
    """
    subs: List[SubtitleBlock] = []
    for w in words:
        txt = normalize_spaces(w.text)
        if not txt:
            continue
        subs.append(SubtitleBlock(start=float(w.start), end=float(w.end), lines=[txt]))
    return subs


# ============================================================
# Silence Alignment
# ============================================================

def apply_silence_alignment(
    subs: List[SubtitleBlock],
    silences: List[Tuple[float, float]],
) -> List[SubtitleBlock]:
    """Align subtitle timing to avoid overlapping with detected silence.

    Args:
        subs: List of SubtitleBlock objects
        silences: List of (start, end) silence intervals

    Returns:
        List of timing-adjusted SubtitleBlock objects
    """
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
# Timing Polish + Hygiene
# ============================================================

def subs_text(sb: SubtitleBlock) -> str:
    """Extract normalized text from a subtitle block.

    Args:
        sb: SubtitleBlock object

    Returns:
        Normalized concatenated text
    """
    return normalize_spaces(" ".join(sb.lines))


def hygiene_and_polish(
    subs: List[SubtitleBlock],
    *,
    min_gap: float,
    pad: float,
    silence_intervals: Optional[List[Tuple[float, float]]] = None,
) -> List[SubtitleBlock]:
    """Apply timing hygiene and polish to subtitle blocks.

    This function:
    - Removes empty subtitles
    - Sorts by time
    - Merges identical consecutive blocks (unless separated by silence)
    - Applies padding into silence regions
    - Enforces minimum gaps between subtitles
    - Ensures monotonic timing

    Args:
        subs: List of SubtitleBlock objects
        min_gap: Minimum gap between consecutive subtitles (seconds)
        pad: Padding to extend subtitles into silence (seconds)
        silence_intervals: Optional list of (start, end) silence intervals

    Returns:
        List of polished SubtitleBlock objects
    """
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
