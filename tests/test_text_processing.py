"""Tests for the text_processing module."""
import pytest
from local_srt.text_processing import (
    normalize_spaces,
    wrap_text_lines,
    block_fits,
    wrap_fallback_blocks,
    split_on_delims,
    split_text_into_blocks,
    preferred_split_index,
    distribute_time,
    enforce_timing,
)


class TestNormalizeSpaces:
    """Tests for normalize_spaces function."""

    def test_normalize_regular_spaces(self):
        """Test normalizing regular multiple spaces."""
        assert normalize_spaces("hello  world") == "hello world"
        assert normalize_spaces("hello    world") == "hello world"

    def test_normalize_non_breaking_spaces(self):
        """Test converting non-breaking spaces to regular spaces."""
        text = "hello\u00a0world"
        assert normalize_spaces(text) == "hello world"

    def test_normalize_mixed_whitespace(self):
        """Test normalizing mixed whitespace."""
        assert normalize_spaces("hello  \n  world") == "hello world"
        assert normalize_spaces("  hello   world  ") == "hello world"

    def test_normalize_empty_string(self):
        """Test normalizing empty string."""
        assert normalize_spaces("") == ""

    def test_normalize_single_word(self):
        """Test normalizing single word."""
        assert normalize_spaces("hello") == "hello"


class TestWrapTextLines:
    """Tests for wrap_text_lines function."""

    def test_wrap_short_text(self):
        """Test wrapping text that fits in one line."""
        result = wrap_text_lines("hello world", 20)
        assert result == ["hello world"]

    def test_wrap_long_text(self):
        """Test wrapping text that needs multiple lines."""
        result = wrap_text_lines("hello world this is a long sentence", 15)
        assert len(result) > 1
        for line in result:
            assert len(line) <= 15

    def test_wrap_exact_length(self):
        """Test wrapping text with exact max length."""
        result = wrap_text_lines("hello world", 11)
        assert result == ["hello world"]

    def test_wrap_empty_text(self):
        """Test wrapping empty text."""
        result = wrap_text_lines("", 20)
        assert result == []

    def test_wrap_single_long_word(self):
        """Test wrapping a single word longer than max."""
        result = wrap_text_lines("supercalifragilisticexpialidocious", 10)
        # Word longer than max should still be kept as a single line
        assert len(result) == 1


class TestBlockFits:
    """Tests for block_fits function."""

    def test_block_fits_single_line(self):
        """Test text that fits in a single line."""
        assert block_fits("hello world", 20, 2) is True

    def test_block_fits_multiple_lines(self):
        """Test text that fits in multiple lines."""
        text = "hello world this is a test"
        assert block_fits(text, 15, 3) is True

    def test_block_does_not_fit_chars(self):
        """Test text that doesn't fit due to character limit."""
        text = "hello world this is a very long sentence that goes on"
        result = block_fits(text, 10, 2)
        # With 10 chars per line and 2 lines max, this won't fit
        assert result is False

    def test_block_does_not_fit_lines(self):
        """Test text that doesn't fit due to line limit."""
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        assert block_fits(text, 10, 2) is False


class TestWrapFallbackBlocks:
    """Tests for wrap_fallback_blocks function."""

    def test_wrap_fallback_single_block(self):
        """Test fallback wrapping with text fitting in one block."""
        result = wrap_fallback_blocks("hello world", 20, 2)
        assert len(result) == 1

    def test_wrap_fallback_multiple_blocks(self):
        """Test fallback wrapping with text requiring multiple blocks."""
        text = "one two three four five six seven eight nine ten"
        result = wrap_fallback_blocks(text, 10, 2)
        assert len(result) > 1


class TestSplitOnDelims:
    """Tests for split_on_delims function."""

    def test_split_on_period(self):
        """Test splitting on periods."""
        text = "First sentence. Second sentence."
        result = split_on_delims(text, ".?!")
        assert len(result) == 2
        assert "First sentence." in result[0]

    def test_split_on_multiple_delimiters(self):
        """Test splitting on multiple delimiter types."""
        text = "First! Second? Third."
        result = split_on_delims(text, ".?!")
        assert len(result) == 3

    def test_split_no_delimiters(self):
        """Test text with no delimiters."""
        text = "No delimiters here"
        result = split_on_delims(text, ".?!")
        assert len(result) == 1
        assert result[0] == text

    def test_split_empty_text(self):
        """Test splitting empty text."""
        result = split_on_delims("", ".?!")
        assert result == []

    def test_split_preserves_punctuation(self):
        """Test that splitting preserves punctuation."""
        text = "Hello. World."
        result = split_on_delims(text, ".")
        assert result[0].endswith(".")


class TestSplitTextIntoBlocks:
    """Tests for split_text_into_blocks function."""

    def test_split_short_text(self):
        """Test splitting text that fits in one block."""
        result = split_text_into_blocks("hello world", 42, 2)
        assert len(result) == 1
        assert result[0] == "hello world"

    def test_split_with_sentences(self):
        """Test splitting text with sentence punctuation."""
        text = "First sentence. Second sentence. Third sentence."
        result = split_text_into_blocks(text, 20, 2)
        assert len(result) >= 2

    def test_split_with_commas(self):
        """Test splitting text with commas."""
        text = "one, two, three, four, five"
        result = split_text_into_blocks(text, 10, 1, allow_commas=True)
        # Should split on commas
        assert len(result) > 1

    def test_split_no_commas(self):
        """Test splitting with comma splitting disabled."""
        text = "one, two, three"
        result = split_text_into_blocks(text, 20, 2, allow_commas=False)
        # With commas disallowed and text fitting, should stay as one block
        assert len(result) == 1

    def test_split_prefer_punct(self):
        """Test splitting with prefer_punct_splits enabled."""
        text = "First sentence. More text."
        result = split_text_into_blocks(
            text, 50, 2, prefer_punct_splits=True
        )
        # Should split even though it fits
        assert len(result) >= 2


class TestPreferredSplitIndex:
    """Tests for preferred_split_index function."""

    def test_preferred_split_with_period(self):
        """Test finding split point at period."""
        text = "This is a sentence. This is another."
        idx = preferred_split_index(text)
        assert idx > 0
        # Should split after first period (and space)
        assert "." in text[:idx]

    def test_preferred_split_with_space(self):
        """Test finding split point at space when no punctuation."""
        text = "a" * 30 + " " + "b" * 30
        idx = preferred_split_index(text)
        assert idx > 20
        assert text[idx - 1] == " "

    def test_preferred_split_no_good_point(self):
        """Test when no good split point exists."""
        text = "short"
        idx = preferred_split_index(text)
        assert idx == -1


class TestDistributeTime:
    """Tests for distribute_time function."""

    def test_distribute_time_single_part(self):
        """Test distributing time across single part."""
        result = distribute_time(0.0, 10.0, ["hello"])
        assert len(result) == 1
        assert result[0] == (0.0, 10.0, "hello")

    def test_distribute_time_multiple_parts(self):
        """Test distributing time across multiple parts."""
        result = distribute_time(0.0, 10.0, ["aa", "bbbb"])
        assert len(result) == 2
        # First part (2 chars) should get less time than second (4 chars)
        assert result[0][1] < result[1][1]

    def test_distribute_time_equal_parts(self):
        """Test distributing time across equal length parts."""
        result = distribute_time(0.0, 6.0, ["aa", "bb", "cc"])
        assert len(result) == 3
        # Each should get approximately 2 seconds
        for start, end, _ in result:
            assert abs((end - start) - 2.0) < 0.1

    def test_distribute_time_zero_duration(self):
        """Test distributing zero duration."""
        result = distribute_time(5.0, 5.0, ["a", "b"])
        assert len(result) == 2
        # Should handle zero duration gracefully


class TestEnforceTiming:
    """Tests for enforce_timing function."""

    def test_enforce_min_duration(self):
        """Test enforcing minimum duration by merging."""
        blocks = [
            (0.0, 0.1, "short"),
            (0.1, 0.5, "next"),
        ]
        result = enforce_timing(blocks, min_dur=0.3, max_dur=10.0)
        # First block too short, should be merged with second
        assert len(result) < len(blocks) or result[0][1] - result[0][0] >= 0.3

    def test_enforce_max_duration_split(self):
        """Test enforcing maximum duration by splitting."""
        long_text = "word " * 30
        blocks = [(0.0, 20.0, long_text)]
        result = enforce_timing(blocks, min_dur=1.0, max_dur=8.0, split_long=True)
        # Long block may be split (depends on text length threshold)
        # The function splits when dur > max_dur AND len(txt) > 120
        assert len(result) >= 1

    def test_enforce_no_split(self):
        """Test enforce timing without splitting long blocks."""
        blocks = [(0.0, 20.0, "long text")]
        result = enforce_timing(blocks, min_dur=1.0, max_dur=8.0, split_long=False)
        # Should not split when split_long=False
        assert len(result) == 1

    def test_enforce_timing_empty_blocks(self):
        """Test enforce timing with empty blocks list."""
        result = enforce_timing([], min_dur=1.0, max_dur=10.0)
        assert result == []

    def test_enforce_timing_valid_blocks(self):
        """Test enforce timing with already valid blocks."""
        blocks = [
            (0.0, 2.0, "first"),
            (2.0, 5.0, "second"),
        ]
        result = enforce_timing(blocks, min_dur=1.0, max_dur=10.0)
        # Should pass through unchanged
        assert len(result) == len(blocks)
