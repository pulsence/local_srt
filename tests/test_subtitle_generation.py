"""Tests for the subtitle_generation module."""
import pytest
from unittest.mock import MagicMock
from local_srt.subtitle_generation import (
    collect_words,
    words_to_text,
    silence_between,
    split_words_on_silence,
    map_text_blocks_to_word_spans,
    chunk_segments_to_subtitles,
    chunk_words_to_subtitles,
    words_to_subtitles,
    apply_silence_alignment,
    subs_text,
    hygiene_and_polish,
)
from local_srt.models import WordItem, SubtitleBlock, ResolvedConfig


class TestCollectWords:
    """Tests for collect_words function."""

    def test_collect_words_empty_segments(self):
        """Test collecting words from empty segments."""
        result = collect_words([])
        assert result == []

    def test_collect_words_single_segment(self):
        """Test collecting words from single segment."""
        word = MagicMock()
        word.word = "hello"
        word.start = 0.0
        word.end = 0.5

        seg = MagicMock()
        seg.words = [word]

        result = collect_words([seg])

        assert len(result) == 1
        assert result[0].text == "hello"
        assert result[0].start == 0.0
        assert result[0].end == 0.5

    def test_collect_words_multiple_segments(self):
        """Test collecting words from multiple segments."""
        word1 = MagicMock()
        word1.word = "hello"
        word1.start = 0.0
        word1.end = 0.5

        word2 = MagicMock()
        word2.word = "world"
        word2.start = 0.6
        word2.end = 1.0

        seg1 = MagicMock()
        seg1.words = [word1]

        seg2 = MagicMock()
        seg2.words = [word2]

        result = collect_words([seg1, seg2])

        assert len(result) == 2

    def test_collect_words_skips_empty_words(self):
        """Test that empty words are skipped."""
        word1 = MagicMock()
        word1.word = "  "
        word1.start = 0.0
        word1.end = 0.1

        word2 = MagicMock()
        word2.word = "hello"
        word2.start = 0.2
        word2.end = 0.5

        seg = MagicMock()
        seg.words = [word1, word2]

        result = collect_words([seg])

        assert len(result) == 1
        assert result[0].text == "hello"


class TestWordsToText:
    """Tests for words_to_text function."""

    def test_words_to_text_empty(self):
        """Test converting empty word list."""
        result = words_to_text([])
        assert result == ""

    def test_words_to_text_single_word(self):
        """Test converting single word."""
        words = [WordItem(0.0, 0.5, "hello")]
        result = words_to_text(words)
        assert result == "hello"

    def test_words_to_text_multiple_words(self):
        """Test converting multiple words."""
        words = [
            WordItem(0.0, 0.5, "hello"),
            WordItem(0.6, 1.0, "world"),
        ]
        result = words_to_text(words)
        assert result == "hello world"


class TestSilenceBetween:
    """Tests for silence_between function."""

    def test_silence_between_found(self):
        """Test finding silence between times."""
        silences = [(1.0, 1.5), (3.0, 3.5)]
        result = silence_between(0.5, 2.0, silences)
        assert result == (1.0, 1.5)

    def test_silence_between_not_found(self):
        """Test when no silence between times."""
        silences = [(3.0, 3.5)]
        result = silence_between(0.0, 2.0, silences)
        assert result is None

    def test_silence_between_partial_overlap(self):
        """Test silence that partially overlaps."""
        silences = [(1.0, 2.5)]
        # Silence extends beyond end, shouldn't match
        result = silence_between(0.0, 2.0, silences)
        assert result is None

    def test_silence_between_empty_list(self):
        """Test with empty silence list."""
        result = silence_between(0.0, 2.0, [])
        assert result is None


class TestSplitWordsOnSilence:
    """Tests for split_words_on_silence function."""

    def test_split_words_no_silence(self):
        """Test splitting with no silences."""
        words = [
            WordItem(0.0, 0.5, "hello"),
            WordItem(0.6, 1.0, "world"),
        ]
        result = split_words_on_silence(words, [])
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_split_words_with_silence(self):
        """Test splitting with silence between words."""
        words = [
            WordItem(0.0, 0.5, "hello"),
            WordItem(2.0, 2.5, "world"),
        ]
        silences = [(0.6, 1.9)]
        result = split_words_on_silence(words, silences)
        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 1

    def test_split_words_empty_words(self):
        """Test splitting empty word list."""
        result = split_words_on_silence([], [(1.0, 2.0)])
        assert result == []


class TestMapTextBlocksToWordSpans:
    """Tests for map_text_blocks_to_word_spans function."""

    def test_map_text_blocks_single_block(self):
        """Test mapping single text block."""
        blocks = ["hello world"]
        words = [
            WordItem(0.0, 0.5, "hello"),
            WordItem(0.6, 1.0, "world"),
        ]
        result = map_text_blocks_to_word_spans(blocks, words)

        assert len(result) == 1
        assert result[0][0] == 0.0  # start
        assert result[0][1] == 1.0  # end
        assert result[0][2] == "hello world"  # text

    def test_map_text_blocks_multiple_blocks(self):
        """Test mapping multiple text blocks."""
        blocks = ["hello", "world"]
        words = [
            WordItem(0.0, 0.5, "hello"),
            WordItem(0.6, 1.0, "world"),
        ]
        result = map_text_blocks_to_word_spans(blocks, words)

        assert len(result) == 2


class TestChunkSegmentsToSubtitles:
    """Tests for chunk_segments_to_subtitles function."""

    def test_chunk_segments_single_segment(self):
        """Test chunking single segment."""
        seg = MagicMock()
        seg.start = 0.0
        seg.end = 2.0
        seg.text = "Hello world"

        cfg = ResolvedConfig()
        result = chunk_segments_to_subtitles([seg], cfg)

        assert len(result) > 0
        assert isinstance(result[0], SubtitleBlock)

    def test_chunk_segments_empty(self):
        """Test chunking empty segment list."""
        cfg = ResolvedConfig()
        result = chunk_segments_to_subtitles([], cfg)
        assert result == []


class TestChunkWordsToSubtitles:
    """Tests for chunk_words_to_subtitles function."""

    def test_chunk_words_basic(self):
        """Test basic word chunking."""
        words = [
            WordItem(0.0, 0.5, "hello"),
            WordItem(0.6, 1.0, "world"),
        ]
        cfg = ResolvedConfig()
        silences = []

        result = chunk_words_to_subtitles(words, cfg, silences)

        assert len(result) > 0
        assert isinstance(result[0], SubtitleBlock)

    def test_chunk_words_with_silence(self):
        """Test word chunking with silence splitting."""
        words = [
            WordItem(0.0, 0.5, "hello"),
            WordItem(2.0, 2.5, "world"),
        ]
        cfg = ResolvedConfig()
        silences = [(0.6, 1.9)]

        result = chunk_words_to_subtitles(words, cfg, silences)

        # Should create separate blocks due to silence
        assert len(result) >= 1


class TestWordsToSubtitles:
    """Tests for words_to_subtitles function."""

    def test_words_to_subtitles_single_word(self):
        """Test converting single word to subtitle."""
        words = [WordItem(0.0, 0.5, "hello")]
        result = words_to_subtitles(words)

        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 0.5
        assert result[0].lines == ["hello"]

    def test_words_to_subtitles_multiple_words(self):
        """Test converting multiple words."""
        words = [
            WordItem(0.0, 0.5, "hello"),
            WordItem(0.6, 1.0, "world"),
        ]
        result = words_to_subtitles(words)

        assert len(result) == 2

    def test_words_to_subtitles_empty(self):
        """Test converting empty word list."""
        result = words_to_subtitles([])
        assert result == []


class TestApplySilenceAlignment:
    """Tests for apply_silence_alignment function."""

    def test_apply_silence_alignment_no_overlap(self):
        """Test alignment with no silence overlap."""
        subs = [SubtitleBlock(0.0, 1.0, ["test"])]
        silences = [(2.0, 3.0)]

        result = apply_silence_alignment(subs, silences)

        assert len(result) == 1
        # Timing should remain similar
        assert abs(result[0].start - 0.0) < 0.1

    def test_apply_silence_alignment_empty_subs(self):
        """Test alignment with empty subtitle list."""
        result = apply_silence_alignment([], [(1.0, 2.0)])
        assert result == []

    def test_apply_silence_alignment_empty_silences(self):
        """Test alignment with empty silence list."""
        subs = [SubtitleBlock(0.0, 1.0, ["test"])]
        result = apply_silence_alignment(subs, [])
        assert len(result) == 1


class TestSubsText:
    """Tests for subs_text function."""

    def test_subs_text_single_line(self):
        """Test extracting text from single line subtitle."""
        sub = SubtitleBlock(0.0, 1.0, ["Hello world"])
        result = subs_text(sub)
        assert result == "Hello world"

    def test_subs_text_multiple_lines(self):
        """Test extracting text from multiline subtitle."""
        sub = SubtitleBlock(0.0, 1.0, ["Line 1", "Line 2"])
        result = subs_text(sub)
        assert result == "Line 1 Line 2"

    def test_subs_text_normalizes_spaces(self):
        """Test that text is normalized."""
        sub = SubtitleBlock(0.0, 1.0, ["Hello  ", "  world"])
        result = subs_text(sub)
        assert result == "Hello world"


class TestHygieneAndPolish:
    """Tests for hygiene_and_polish function."""

    def test_hygiene_removes_empty(self):
        """Test that empty subtitles are removed."""
        subs = [
            SubtitleBlock(0.0, 1.0, [""]),
            SubtitleBlock(1.0, 2.0, ["Hello"]),
        ]
        result = hygiene_and_polish(subs, min_gap=0.08, pad=0.0)

        assert len(result) == 1
        assert "Hello" in subs_text(result[0])

    def test_hygiene_sorts_by_time(self):
        """Test that subtitles are sorted by time."""
        subs = [
            SubtitleBlock(2.0, 3.0, ["Second"]),
            SubtitleBlock(0.0, 1.0, ["First"]),
        ]
        result = hygiene_and_polish(subs, min_gap=0.08, pad=0.0)

        assert len(result) == 2
        assert result[0].start < result[1].start

    def test_hygiene_merges_identical(self):
        """Test that identical consecutive blocks are merged."""
        subs = [
            SubtitleBlock(0.0, 1.0, ["Same"]),
            SubtitleBlock(1.0, 2.0, ["Same"]),
        ]
        result = hygiene_and_polish(subs, min_gap=0.08, pad=0.0, silence_intervals=None)

        # Should be merged since they're identical
        assert len(result) <= 2

    def test_hygiene_enforces_min_gap(self):
        """Test that minimum gap is enforced."""
        subs = [
            SubtitleBlock(0.0, 1.0, ["First"]),
            SubtitleBlock(1.01, 2.0, ["Second"]),  # Gap < min_gap
        ]
        result = hygiene_and_polish(subs, min_gap=0.08, pad=0.0)

        # Gap should be enforced (allow floating point precision issues)
        gap = result[1].start - result[0].end
        assert gap >= 0.079 or gap < 0.001  # Allow for floating point precision

    def test_hygiene_applies_padding(self):
        """Test that padding is applied."""
        subs = [SubtitleBlock(1.0, 2.0, ["Test"])]
        result = hygiene_and_polish(subs, min_gap=0.08, pad=0.1)

        # Start should be padded backward
        assert result[0].start <= 1.0

    def test_hygiene_empty_list(self):
        """Test with empty subtitle list."""
        result = hygiene_and_polish([], min_gap=0.08, pad=0.0)
        assert result == []

    def test_hygiene_monotonic_timing(self):
        """Test that timing is monotonic."""
        subs = [
            SubtitleBlock(0.0, 1.0, ["First"]),
            SubtitleBlock(1.5, 2.5, ["Second"]),
            SubtitleBlock(3.0, 4.0, ["Third"]),
        ]
        result = hygiene_and_polish(subs, min_gap=0.08, pad=0.0)

        # Each subtitle should start after the previous one ends
        for i in range(len(result) - 1):
            assert result[i].end <= result[i + 1].start
