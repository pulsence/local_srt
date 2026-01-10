#!/usr/bin/env python3
"""Configuration management for Local SRT.

This module handles configuration loading, preset management, and
configuration merging/overrides.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .models import ResolvedConfig


# ============================================================
# Presets
# ============================================================

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


# ============================================================
# Configuration Loading
# ============================================================

def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        path: Path to JSON config file, or None to skip loading

    Returns:
        Dictionary of configuration values, or empty dict if path is None

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ValueError: If the config file isn't a valid JSON object
    """
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
    """Apply configuration overrides to a base configuration.

    Args:
        base: Base ResolvedConfig instance
        overrides: Dictionary of configuration values to override

    Returns:
        New ResolvedConfig instance with overrides applied
    """
    d = dataclasses.asdict(base)
    for k, v in overrides.items():
        if k in d:
            d[k] = v
    return ResolvedConfig(**d)
