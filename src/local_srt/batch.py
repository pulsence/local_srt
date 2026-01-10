#!/usr/bin/env python3
"""Batch processing utilities for Local SRT.

This module handles:
- Media file discovery and expansion
- Output path calculation for batch processing
- Preflight checks before processing
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


# ============================================================
# Media File Discovery
# ============================================================

MEDIA_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".mp4", ".mkv", ".mov", ".webm", ".m4v"}


def iter_media_files_in_dir(d: Path) -> Iterable[Path]:
    """Recursively iterate over all media files in a directory.

    Args:
        d: Directory path to search

    Yields:
        Path objects for each media file found
    """
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
            yield p


def expand_inputs(inputs: List[str], glob_pat: Optional[str]) -> List[Path]:
    """Expand input specifications into a list of file paths.

    Handles:
    - Individual files
    - Directories (recursively finds media files)
    - Glob patterns (e.g., "*.mp3")

    Args:
        inputs: List of input file/directory/glob specifications
        glob_pat: Optional additional glob pattern

    Returns:
        Deduplicated list of Path objects
    """
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


# ============================================================
# Output Path Calculation
# ============================================================

def default_output_for(input_file: Path, outdir: Optional[Path], fmt: str, keep_structure: bool, base_root: Optional[Path]) -> Path:
    """Calculate the default output path for an input file.

    Args:
        input_file: Input file path
        outdir: Optional output directory
        fmt: Output format (srt/vtt/ass/txt/json)
        keep_structure: If True, preserve directory structure in outdir
        base_root: Base directory for structure preservation

    Returns:
        Output file path
    """
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
# Preflight Checks
# ============================================================

def preflight_one(input_path: Path, output_path: Path, overwrite: bool) -> Tuple[bool, str]:
    """Perform preflight checks before processing a file.

    Args:
        input_path: Input file path
        output_path: Output file path
        overwrite: If True, allow overwriting existing output

    Returns:
        Tuple of (success: bool, error_message: str)
        If success is True, error_message will be empty
    """
    if not input_path.exists():
        return False, f"Input file not found: {input_path}"
    if input_path.is_dir():
        return False, f"Input path is a directory (expected media file): {input_path}"
    if output_path.exists() and output_path.is_dir():
        return False, f"Output path is a directory (expected file): {output_path}"
    if output_path.exists() and not overwrite:
        return False, f"Output already exists: {output_path} (use --overwrite)"
    return True, ""
