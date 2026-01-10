#!/usr/bin/env python3
"""Text processing utilities for Local SRT.

This module provides text normalization, wrapping, chunking, and timing
distribution algorithms used for subtitle generation.
"""
from __future__ import annotations

import re
from typing import List, Tuple


# ============================================================
# Text Normalization
# ============================================================

def normalize_spaces(text: str) -> str:
    """Normalize whitespace in text by replacing non-breaking spaces and collapsing multiple spaces.

    Args:
        text: Input text

    Returns:
        Normalized text with single spaces
    """
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Text Wrapping
# ============================================================

def wrap_text_lines(text: str, max_chars_per_line: int) -> List[str]:
    """Wrap text into lines that fit within a maximum character width.

    Args:
        text: Input text to wrap
        max_chars_per_line: Maximum characters per line

    Returns:
        List of wrapped text lines
    """
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
    """Check if a text block fits within line and character constraints.

    Args:
        text: Text to check
        max_chars_per_line: Maximum characters per line
        max_lines: Maximum number of lines

    Returns:
        True if text fits within constraints
    """
    return len(wrap_text_lines(text, max_chars_per_line)) <= max_lines


def wrap_fallback_blocks(text: str, max_chars_per_line: int, max_lines: int) -> List[str]:
    """Wrap text into blocks by grouping wrapped lines.

    Args:
        text: Text to wrap
        max_chars_per_line: Maximum characters per line
        max_lines: Maximum lines per block

    Returns:
        List of text blocks
    """
    lines = wrap_text_lines(text, max_chars_per_line)
    blocks: List[str] = []
    for i in range(0, len(lines), max_lines):
        blocks.append(" ".join(lines[i:i + max_lines]))
    return blocks


# ============================================================
# Text Splitting
# ============================================================

def split_on_delims(text: str, delims: str) -> List[str]:
    """Split text on specified delimiters while preserving punctuation.

    Args:
        text: Text to split
        delims: String of delimiter characters (e.g., ".?!")

    Returns:
        List of text parts split at delimiters
    """
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
    """Split text into subtitle-sized blocks using hierarchical punctuation splitting.

    Args:
        text: Text to split
        max_chars_per_line: Maximum characters per line
        max_lines: Maximum lines per block
        allow_commas: Allow splitting at commas
        allow_medium: Allow splitting at semicolons and colons
        prefer_punct_splits: Prefer splitting at punctuation even when text fits

    Returns:
        List of text blocks suitable for subtitles
    """
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
    """Find the preferred index to split text, preferring punctuation breaks.

    Args:
        text: Text to analyze

    Returns:
        Index to split at, or -1 if no good split point found
    """
    punct = [". ", "? ", "! ", "; ", ": ", ", "]
    for p in punct:
        idx = text.rfind(p)
        if idx != -1 and idx > 20:
            return idx + len(p)
    sp = text.rfind(" ")
    if sp > 20:
        return sp + 1
    return -1


# ============================================================
# Timing Distribution
# ============================================================

def distribute_time(start: float, end: float, parts: List[str]) -> List[Tuple[float, float, str]]:
    """Distribute time across text parts proportionally to their length.

    Args:
        start: Start time in seconds
        end: End time in seconds
        parts: List of text parts to distribute time across

    Returns:
        List of (start, end, text) tuples with distributed timing
    """
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
    """Enforce minimum and maximum duration constraints on timed text blocks.

    Args:
        blocks: List of (start, end, text) tuples
        min_dur: Minimum duration in seconds
        max_dur: Maximum duration in seconds
        split_long: If True, split blocks that exceed max_dur

    Returns:
        List of timing-adjusted blocks
    """
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
