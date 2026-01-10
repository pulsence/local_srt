#!/usr/bin/env python3
"""Data models for Local SRT.

This module contains all data classes used throughout the application:
- TOOL_VERSION: Version constant
- ResolvedConfig: Configuration settings dataclass
- SubtitleBlock: Represents a subtitle cue with timing and text
- WordItem: Represents a single transcribed word with timing
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


# ============================================================
# Versioning
# ============================================================

TOOL_VERSION = "0.1.1"


# ============================================================
# Configuration
# ============================================================

@dataclass
class ResolvedConfig:
    """Resolved configuration for subtitle generation.

    This dataclass contains all settings that control how subtitles
    are generated, formatted, and timed.
    """
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


# ============================================================
# Subtitle Data Structures
# ============================================================

@dataclass
class SubtitleBlock:
    """Represents a single subtitle cue with timing and text.

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        lines: List of text lines to display
    """
    start: float
    end: float
    lines: List[str]


@dataclass
class WordItem:
    """Represents a single transcribed word with timing.

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        text: The word text
    """
    start: float
    end: float
    text: str
