#!/usr/bin/env python3
"""Logging and progress utilities for Local SRT.

This module provides logging functions, progress display utilities,
and time formatting helpers used throughout the application.
"""
from __future__ import annotations

import sys


# ============================================================
# Logging / Progress
# ============================================================

def log(msg: str, *, quiet: bool = False) -> None:
    """Print a log message to stdout unless quiet mode is enabled.

    Args:
        msg: Message to log
        quiet: If True, suppress output
    """
    if not quiet:
        print(msg, flush=True)


def warn(msg: str, *, quiet: bool = False) -> None:
    """Print a warning message to stderr unless quiet mode is enabled.

    Args:
        msg: Warning message to display
        quiet: If True, suppress output
    """
    if not quiet:
        print(f"WARNING: {msg}", file=sys.stderr, flush=True)


def die(msg: str, code: int = 1) -> int:
    """Print an error message to stderr and return an exit code.

    Args:
        msg: Error message to display
        code: Exit code to return (default: 1)

    Returns:
        The exit code provided
    """
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    return code


def progress_line(msg: str, *, enabled: bool, quiet: bool) -> None:
    """Display a progress message on the current line (overwrites previous).

    Args:
        msg: Progress message to display (truncated to 160 chars)
        enabled: If False, suppress output
        quiet: If True, suppress output
    """
    if quiet or not enabled:
        return
    sys.stdout.write("\r" + msg[:160].ljust(160))
    sys.stdout.flush()


def progress_done(*, enabled: bool, quiet: bool) -> None:
    """Finalize progress display by adding a newline.

    Args:
        enabled: If False, suppress output
        quiet: If True, suppress output
    """
    if quiet or not enabled:
        return
    sys.stdout.write("\n")
    sys.stdout.flush()


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS or MM:SS.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string (e.g., "1:23:45" or "23:45")
    """
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"
