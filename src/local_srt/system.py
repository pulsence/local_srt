#!/usr/bin/env python3
"""System utilities and dependency checks for Local SRT.

This module provides system-level utilities including:
- File system operations
- External dependency checks (ffmpeg, ffprobe)
- Command execution helpers
- Audio file probing
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


# ============================================================
# File System Utilities
# ============================================================

def ensure_parent_dir(path: Path) -> None:
    """Ensure that a file's parent directory exists, creating it if needed.

    Args:
        path: Path to a file whose parent directory should exist
    """
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# External Dependency Checks
# ============================================================

def which_or_none(name: str) -> Optional[str]:
    """Find the path to an executable, or None if not found.

    Args:
        name: Name of executable to find

    Returns:
        Path to executable or None if not found
    """
    return shutil.which(name)


def ffmpeg_ok() -> bool:
    """Check if ffmpeg is available on the system PATH.

    Returns:
        True if ffmpeg is available, False otherwise
    """
    return which_or_none("ffmpeg") is not None


def ffprobe_ok() -> bool:
    """Check if ffprobe is available on the system PATH.

    Returns:
        True if ffprobe is available, False otherwise
    """
    return which_or_none("ffprobe") is not None


# ============================================================
# Command Execution
# ============================================================

def run_cmd_text(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a command and capture its output as text.

    Args:
        cmd: Command and arguments to execute

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def ffmpeg_version() -> Optional[str]:
    """Get the ffmpeg version string.

    Returns:
        Version string or None if ffmpeg is not available
    """
    if not ffmpeg_ok():
        return None
    code, out, _ = run_cmd_text(["ffmpeg", "-version"])
    if code != 0:
        return None
    return out.splitlines()[0].strip() if out else None


def ffprobe_version() -> Optional[str]:
    """Get the ffprobe version string.

    Returns:
        Version string or None if ffprobe is not available
    """
    if not ffprobe_ok():
        return None
    code, out, _ = run_cmd_text(["ffprobe", "-version"])
    if code != 0:
        return None
    return out.splitlines()[0].strip() if out else None


# ============================================================
# Audio Probing
# ============================================================

def probe_duration_seconds(path: str) -> Optional[float]:
    """Probe an audio/video file to get its duration in seconds.

    Args:
        path: Path to media file

    Returns:
        Duration in seconds, or None if probing fails
    """
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
