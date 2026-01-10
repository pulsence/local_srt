#!/usr/bin/env python3
"""Whisper model initialization wrapper for Local SRT.

This module handles initialization of the faster-whisper model with
appropriate device and compute type selection.
"""
from __future__ import annotations

from typing import Tuple

from faster_whisper import WhisperModel

from .logging_utils import log


# ============================================================
# Device Initialization
# ============================================================

def init_whisper_model(
    model_name: str,
    device: str,               # auto|cpu|cuda
    quiet: bool,
    strict_cuda: bool,
) -> Tuple[WhisperModel, str, str]:
    """Initialize a Whisper model with appropriate device and compute type.

    Args:
        model_name: Name of the Whisper model (e.g., "small", "medium")
        device: Device selection: "auto", "cpu", or "cuda"
        quiet: If True, suppress log messages
        strict_cuda: If True, fail if CUDA requested but unavailable

    Returns:
        Tuple of (model, device_used, compute_type_used)

    Raises:
        RuntimeError: If strict_cuda=True and CUDA initialization fails
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
