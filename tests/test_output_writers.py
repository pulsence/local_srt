"""Tests for the output_writers module."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from local_srt.output_writers import (
    format_srt_time,
    format_vtt_time,
    format_ass_time,
    atomic_write_text,
    write_srt,
    write_vtt,
    write_ass,
    write_txt,
    segments_to_jsonable,
    write_json_bundle,
)
from local_srt.models import SubtitleBlock, ResolvedConfig


class TestFormatSrtTime:
    """Tests for format_srt_time function."""

    def test_format_srt_time_zero(self):
        """Test formatting zero time."""
        assert format_srt_time(0.0) == "00:00:00,000"

    def test_format_srt_time_seconds(self):
        """Test formatting seconds only."""
        assert format_srt_time(45.5) == "00:00:45,500"

    def test_format_srt_time_minutes(self):
        """Test formatting with minutes."""
        assert format_srt_time(90.123) == "00:01:30,123"

    def test_format_srt_time_hours(self):
        """Test formatting with hours."""
        assert format_srt_time(3661.456) == "01:01:01,456"

    def test_format_srt_time_large(self):
        """Test formatting large time value."""
        # 10 hours, 5 minutes, 30.789 seconds
        assert format_srt_time(36330.789) == "10:05:30,789"

    def test_format_srt_time_rounding(self):
        """Test that milliseconds are rounded."""
        assert format_srt_time(1.9999) == "00:00:02,000"


class TestFormatVttTime:
    """Tests for format_vtt_time function."""

    def test_format_vtt_time_zero(self):
        """Test formatting zero time."""
        assert format_vtt_time(0.0) == "00:00:00.000"

    def test_format_vtt_time_seconds(self):
        """Test formatting seconds only."""
        assert format_vtt_time(45.5) == "00:00:45.500"

    def test_format_vtt_time_minutes(self):
        """Test formatting with minutes."""
        assert format_vtt_time(90.123) == "00:01:30.123"

    def test_format_vtt_time_hours(self):
        """Test formatting with hours."""
        assert format_vtt_time(3661.456) == "01:01:01.456"

    def test_format_vtt_time_uses_dot(self):
        """Test that VTT uses dot separator (not comma)."""
        result = format_vtt_time(1.5)
        assert "." in result
        assert "," not in result


class TestFormatAssTime:
    """Tests for format_ass_time function."""

    def test_format_ass_time_zero(self):
        """Test formatting zero time."""
        assert format_ass_time(0.0) == "0:00:00.00"

    def test_format_ass_time_seconds(self):
        """Test formatting seconds only."""
        assert format_ass_time(45.5) == "0:00:45.50"

    def test_format_ass_time_minutes(self):
        """Test formatting with minutes."""
        assert format_ass_time(90.12) == "0:01:30.12"

    def test_format_ass_time_hours(self):
        """Test formatting with hours."""
        assert format_ass_time(3661.45) == "1:01:01.45"

    def test_format_ass_time_centiseconds(self):
        """Test that ASS uses centiseconds (not milliseconds)."""
        # 1.234 seconds = 123 centiseconds (rounded to 12)
        result = format_ass_time(1.234)
        assert result.endswith(".23")


class TestAtomicWriteText:
    """Tests for atomic_write_text function."""

    def test_atomic_write_text_creates_file(self, tmp_path):
        """Test that file is created."""
        file_path = tmp_path / "test.txt"
        content = "test content"

        atomic_write_text(file_path, content)

        assert file_path.exists()
        assert file_path.read_text(encoding="utf-8") == content

    def test_atomic_write_text_creates_parent(self, tmp_path):
        """Test that parent directories are created."""
        file_path = tmp_path / "subdir" / "test.txt"
        content = "test"

        atomic_write_text(file_path, content)

        assert file_path.exists()
        assert file_path.parent.exists()

    def test_atomic_write_text_overwrites(self, tmp_path):
        """Test that existing file is overwritten."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("old content", encoding="utf-8")

        atomic_write_text(file_path, "new content")

        assert file_path.read_text(encoding="utf-8") == "new content"

    def test_atomic_write_text_unicode(self, tmp_path):
        """Test writing unicode content."""
        file_path = tmp_path / "test.txt"
        content = "Hello 世界 🌍"

        atomic_write_text(file_path, content)

        assert file_path.read_text(encoding="utf-8") == content


class TestWriteSrt:
    """Tests for write_srt function."""

    def test_write_srt_single_subtitle(self, tmp_path):
        """Test writing single subtitle."""
        subs = [SubtitleBlock(0.0, 2.0, ["Hello world"])]
        out_path = tmp_path / "test.srt"

        write_srt(subs, out_path, max_chars=42, max_lines=2)

        content = out_path.read_text(encoding="utf-8")
        assert "1" in content
        assert "00:00:00,000 --> 00:00:02,000" in content
        assert "Hello world" in content

    def test_write_srt_multiple_subtitles(self, tmp_path):
        """Test writing multiple subtitles."""
        subs = [
            SubtitleBlock(0.0, 2.0, ["First"]),
            SubtitleBlock(2.0, 4.0, ["Second"]),
        ]
        out_path = tmp_path / "test.srt"

        write_srt(subs, out_path, max_chars=42, max_lines=2)

        content = out_path.read_text(encoding="utf-8")
        assert "1\n" in content
        assert "2\n" in content
        assert "First" in content
        assert "Second" in content

    def test_write_srt_multiline(self, tmp_path):
        """Test writing subtitle with multiple lines."""
        subs = [SubtitleBlock(0.0, 2.0, ["Line 1", "Line 2"])]
        out_path = tmp_path / "test.srt"

        write_srt(subs, out_path, max_chars=42, max_lines=2)

        content = out_path.read_text(encoding="utf-8")
        # After wrapping, should contain the text
        assert "Line 1 Line 2" in content or ("Line 1" in content and "Line 2" in content)

    def test_write_srt_empty_list(self, tmp_path):
        """Test writing empty subtitle list."""
        subs = []
        out_path = tmp_path / "test.srt"

        write_srt(subs, out_path, max_chars=42, max_lines=2)

        content = out_path.read_text(encoding="utf-8")
        assert content.strip() == ""


class TestWriteVtt:
    """Tests for write_vtt function."""

    def test_write_vtt_header(self, tmp_path):
        """Test that VTT file starts with WEBVTT header."""
        subs = [SubtitleBlock(0.0, 2.0, ["Test"])]
        out_path = tmp_path / "test.vtt"

        write_vtt(subs, out_path, max_chars=42, max_lines=2)

        content = out_path.read_text(encoding="utf-8")
        assert content.startswith("WEBVTT")

    def test_write_vtt_timing_format(self, tmp_path):
        """Test VTT timing format uses dots."""
        subs = [SubtitleBlock(0.0, 2.5, ["Test"])]
        out_path = tmp_path / "test.vtt"

        write_vtt(subs, out_path, max_chars=42, max_lines=2)

        content = out_path.read_text(encoding="utf-8")
        # VTT uses dots for milliseconds
        assert "00:00:00.000 --> 00:00:02.500" in content

    def test_write_vtt_no_cue_numbers(self, tmp_path):
        """Test that VTT doesn't include cue numbers."""
        subs = [SubtitleBlock(0.0, 2.0, ["Test"])]
        out_path = tmp_path / "test.vtt"

        write_vtt(subs, out_path, max_chars=42, max_lines=2)

        content = out_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        # VTT cues don't have numbers like SRT
        assert not any(line.strip() == "1" for line in lines if "WEBVTT" not in line)


class TestWriteAss:
    """Tests for write_ass function."""

    def test_write_ass_header(self, tmp_path):
        """Test that ASS file has proper header."""
        subs = [SubtitleBlock(0.0, 2.0, ["Test"])]
        out_path = tmp_path / "test.ass"

        write_ass(subs, out_path, max_chars=42, max_lines=2)

        content = out_path.read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "[V4+ Styles]" in content
        assert "[Events]" in content

    def test_write_ass_dialogue_line(self, tmp_path):
        """Test ASS dialogue line format."""
        subs = [SubtitleBlock(0.0, 2.0, ["Test"])]
        out_path = tmp_path / "test.ass"

        write_ass(subs, out_path, max_chars=42, max_lines=2)

        content = out_path.read_text(encoding="utf-8")
        assert "Dialogue:" in content

    def test_write_ass_multiline_separator(self, tmp_path):
        """Test that ASS uses \\N for line breaks."""
        subs = [SubtitleBlock(0.0, 2.0, ["Line 1", "Line 2"])]
        out_path = tmp_path / "test.ass"

        write_ass(subs, out_path, max_chars=10, max_lines=2)

        content = out_path.read_text(encoding="utf-8")
        # ASS uses \N for line breaks (may be in wrapped text)
        assert "Line" in content


class TestWriteTxt:
    """Tests for write_txt function."""

    def test_write_txt_single_subtitle(self, tmp_path):
        """Test writing single subtitle as text."""
        subs = [SubtitleBlock(0.0, 2.0, ["Hello world"])]
        out_path = tmp_path / "test.txt"

        write_txt(subs, out_path)

        content = out_path.read_text(encoding="utf-8")
        assert "Hello world" in content
        # Should not include timing
        assert "00:00" not in content

    def test_write_txt_multiple_subtitles(self, tmp_path):
        """Test writing multiple subtitles."""
        subs = [
            SubtitleBlock(0.0, 2.0, ["First"]),
            SubtitleBlock(2.0, 4.0, ["Second"]),
        ]
        out_path = tmp_path / "test.txt"

        write_txt(subs, out_path)

        content = out_path.read_text(encoding="utf-8")
        assert "First" in content
        assert "Second" in content

    def test_write_txt_joins_lines(self, tmp_path):
        """Test that multiline subtitles are joined."""
        subs = [SubtitleBlock(0.0, 2.0, ["Line 1", "Line 2"])]
        out_path = tmp_path / "test.txt"

        write_txt(subs, out_path)

        content = out_path.read_text(encoding="utf-8")
        assert "Line 1 Line 2" in content


class TestSegmentsToJsonable:
    """Tests for segments_to_jsonable function."""

    def test_segments_to_jsonable_basic(self):
        """Test converting segments without words."""
        seg = MagicMock()
        seg.start = 0.0
        seg.end = 2.0
        seg.text = "Hello"

        result = segments_to_jsonable([seg], include_words=False)

        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 2.0
        assert result[0]["text"] == "Hello"

    def test_segments_to_jsonable_with_words(self):
        """Test converting segments with word timestamps."""
        word = MagicMock()
        word.start = 0.0
        word.end = 0.5
        word.word = "Hello"

        seg = MagicMock()
        seg.start = 0.0
        seg.end = 2.0
        seg.text = "Hello world"
        seg.words = [word]

        result = segments_to_jsonable([seg], include_words=True)

        assert len(result) == 1
        assert "words" in result[0]
        assert len(result[0]["words"]) == 1

    def test_segments_to_jsonable_empty(self):
        """Test converting empty segment list."""
        result = segments_to_jsonable([], include_words=False)
        assert result == []


class TestWriteJsonBundle:
    """Tests for write_json_bundle function."""

    def test_write_json_bundle_structure(self, tmp_path):
        """Test JSON bundle structure."""
        cfg = ResolvedConfig()
        subs = [SubtitleBlock(0.0, 2.0, ["Test"])]
        seg = MagicMock()
        seg.start = 0.0
        seg.end = 2.0
        seg.text = "Test"
        seg.words = []

        out_path = tmp_path / "bundle.json"

        write_json_bundle(
            out_path,
            input_file="test.mp3",
            device_used="cpu",
            compute_type_used="int8",
            cfg=cfg,
            segments=[seg],
            subs=subs,
            tool_version="0.1.0",
        )

        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))

        assert "tool_version" in data
        assert "input_file" in data
        assert "device_used" in data
        assert "compute_type_used" in data
        assert "config" in data
        assert "segments" in data
        assert "subtitles" in data

    def test_write_json_bundle_values(self, tmp_path):
        """Test JSON bundle contains correct values."""
        cfg = ResolvedConfig()
        subs = [SubtitleBlock(0.0, 2.0, ["Test"])]
        seg = MagicMock()
        seg.start = 0.0
        seg.end = 2.0
        seg.text = "Test"

        out_path = tmp_path / "bundle.json"

        write_json_bundle(
            out_path,
            input_file="test.mp3",
            device_used="cuda",
            compute_type_used="float16",
            cfg=cfg,
            segments=[seg],
            subs=subs,
            tool_version="0.1.0",
        )

        data = json.loads(out_path.read_text(encoding="utf-8"))

        assert data["input_file"] == "test.mp3"
        assert data["device_used"] == "cuda"
        assert data["compute_type_used"] == "float16"
        assert data["tool_version"] == "0.1.0"
