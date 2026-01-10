#!/usr/bin/env python3
"""Model management utilities for Local SRT.

This module provides functionality for:
- Listing downloaded Whisper models
- Downloading new models
- Deleting cached models
- System diagnostics
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from typing import List, Tuple

from faster_whisper import utils as fw_utils

from .logging_utils import die
from .models import TOOL_VERSION
from .system import ffmpeg_version, ffprobe_version, which_or_none


# ============================================================
# System Diagnostics
# ============================================================

def diagnose() -> None:
    """Print system diagnostic information to stdout.

    Displays:
    - Tool version
    - Python version
    - Platform information
    - ffmpeg/ffprobe versions and paths
    - faster-whisper version
    """
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


# ============================================================
# Model Listing
# ============================================================

def list_downloaded_models() -> List[Tuple[str, str]]:
    """Get a list of already-downloaded Whisper models.

    Returns:
        List of (model_name, path) tuples for downloaded models
    """
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
    """Get a list of all available Whisper model names.

    Returns:
        List of model names (e.g., ["tiny", "base", "small", ...])
    """
    return list(fw_utils.available_models())


# ============================================================
# Model Download/Delete
# ============================================================

def download_model_cli(model_name: str) -> int:
    """Download a Whisper model from the internet.

    Args:
        model_name: Name of the model to download (e.g., "small")

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        path = fw_utils.download_model(model_name, local_files_only=False)
    except Exception as e:
        return die(f"Failed to download model '{model_name}': {e}", 2)
    print(f"Downloaded {model_name} to {path}")
    return 0


def delete_model_cli(model_name: str) -> int:
    """Delete a cached Whisper model from disk.

    Args:
        model_name: Name of the model to delete (e.g., "small")

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
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
