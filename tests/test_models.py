"""Tests for the models module."""
import pytest
from local_srt.models import TOOL_VERSION, ResolvedConfig, SubtitleBlock, WordItem


class TestToolVersion:
    """Tests for TOOL_VERSION constant."""

    def test_tool_version_exists(self):
        """Test that TOOL_VERSION is defined."""
        assert TOOL_VERSION is not None
        assert isinstance(TOOL_VERSION, str)
        assert len(TOOL_VERSION) > 0

    def test_tool_version_format(self):
        """Test that TOOL_VERSION follows semantic versioning."""
        parts = TOOL_VERSION.split(".")
        assert len(parts) >= 2  # At least major.minor


class TestResolvedConfig:
    """Tests for ResolvedConfig dataclass."""

    def test_default_config(self):
        """Test that default config can be instantiated."""
        cfg = ResolvedConfig()
        assert cfg is not None
        assert isinstance(cfg, ResolvedConfig)

    def test_default_values(self):
        """Test default configuration values."""
        cfg = ResolvedConfig()

        # Caption formatting
        assert cfg.max_chars == 42
        assert cfg.max_lines == 2

        # Readability / timing heuristics
        assert cfg.target_cps == 17.0
        assert cfg.min_dur == 1.0
        assert cfg.max_dur == 6.0

        # Punctuation splitting
        assert cfg.allow_commas is True
        assert cfg.allow_medium is True
        assert cfg.prefer_punct_splits is False

        # Timing polish
        assert cfg.min_gap == 0.08
        assert cfg.pad == 0.00

        # Transcription options
        assert cfg.vad_filter is True
        assert cfg.word_timestamps is False

        # Silence-aware timing
        assert cfg.use_silence_split is True
        assert cfg.silence_min_dur == 0.2
        assert cfg.silence_threshold_db == -35.0

    def test_custom_config(self):
        """Test creating config with custom values."""
        cfg = ResolvedConfig(
            max_chars=50,
            max_lines=3,
            target_cps=20.0,
            min_dur=0.5,
            max_dur=10.0,
        )

        assert cfg.max_chars == 50
        assert cfg.max_lines == 3
        assert cfg.target_cps == 20.0
        assert cfg.min_dur == 0.5
        assert cfg.max_dur == 10.0

    def test_config_is_dataclass(self):
        """Test that ResolvedConfig is a proper dataclass."""
        import dataclasses
        assert dataclasses.is_dataclass(ResolvedConfig)


class TestSubtitleBlock:
    """Tests for SubtitleBlock dataclass."""

    def test_subtitle_block_creation(self):
        """Test creating a SubtitleBlock instance."""
        block = SubtitleBlock(start=0.0, end=5.0, lines=["Hello", "World"])

        assert block.start == 0.0
        assert block.end == 5.0
        assert block.lines == ["Hello", "World"]

    def test_subtitle_block_with_single_line(self):
        """Test SubtitleBlock with a single line."""
        block = SubtitleBlock(start=1.5, end=3.7, lines=["Single line"])

        assert block.start == 1.5
        assert block.end == 3.7
        assert len(block.lines) == 1
        assert block.lines[0] == "Single line"

    def test_subtitle_block_with_empty_lines(self):
        """Test SubtitleBlock with empty lines list."""
        block = SubtitleBlock(start=0.0, end=1.0, lines=[])

        assert block.start == 0.0
        assert block.end == 1.0
        assert block.lines == []

    def test_subtitle_block_timing_precision(self):
        """Test SubtitleBlock with precise float timing."""
        block = SubtitleBlock(start=1.234567, end=5.987654, lines=["Test"])

        assert block.start == 1.234567
        assert block.end == 5.987654

    def test_subtitle_block_is_dataclass(self):
        """Test that SubtitleBlock is a proper dataclass."""
        import dataclasses
        assert dataclasses.is_dataclass(SubtitleBlock)


class TestWordItem:
    """Tests for WordItem dataclass."""

    def test_word_item_creation(self):
        """Test creating a WordItem instance."""
        word = WordItem(start=1.0, end=1.5, text="hello")

        assert word.start == 1.0
        assert word.end == 1.5
        assert word.text == "hello"

    def test_word_item_with_punctuation(self):
        """Test WordItem with punctuation."""
        word = WordItem(start=0.0, end=0.8, text="Hello,")

        assert word.start == 0.0
        assert word.end == 0.8
        assert word.text == "Hello,"

    def test_word_item_empty_text(self):
        """Test WordItem with empty text."""
        word = WordItem(start=0.0, end=0.1, text="")

        assert word.start == 0.0
        assert word.end == 0.1
        assert word.text == ""

    def test_word_item_timing_precision(self):
        """Test WordItem with precise float timing."""
        word = WordItem(start=1.234567, end=1.789012, text="test")

        assert word.start == 1.234567
        assert word.end == 1.789012

    def test_word_item_is_dataclass(self):
        """Test that WordItem is a proper dataclass."""
        import dataclasses
        assert dataclasses.is_dataclass(WordItem)
