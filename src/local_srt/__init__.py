"""Local SRT - Local subtitle generator using faster-whisper + ffmpeg.

This package provides tools for generating subtitles from audio/video files
using the Whisper speech recognition model.
"""
from .cli import main
from .models import TOOL_VERSION

__version__ = TOOL_VERSION
__all__ = ["main", "__version__"]
