"""Tests for the batch module."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from local_srt.batch import (
    MEDIA_EXTS,
    iter_media_files_in_dir,
    expand_inputs,
    default_output_for,
    preflight_one,
)


class TestMediaExts:
    """Tests for MEDIA_EXTS constant."""

    def test_media_exts_exists(self):
        """Test that MEDIA_EXTS is defined."""
        assert MEDIA_EXTS is not None
        assert isinstance(MEDIA_EXTS, set)

    def test_media_exts_common_formats(self):
        """Test that common media formats are included."""
        expected = {".mp3", ".mp4", ".wav", ".m4a"}
        assert expected.issubset(MEDIA_EXTS)

    def test_media_exts_video_formats(self):
        """Test that video formats are included."""
        video_formats = {".mp4", ".mkv", ".mov", ".webm"}
        assert video_formats.issubset(MEDIA_EXTS)

    def test_media_exts_audio_formats(self):
        """Test that audio formats are included."""
        audio_formats = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
        assert audio_formats.issubset(MEDIA_EXTS)


class TestIterMediaFilesInDir:
    """Tests for iter_media_files_in_dir function."""

    def test_iter_empty_directory(self, tmp_path):
        """Test iterating over empty directory."""
        result = list(iter_media_files_in_dir(tmp_path))
        assert result == []

    def test_iter_media_files_single_file(self, tmp_path):
        """Test iterating directory with single media file."""
        media_file = tmp_path / "test.mp3"
        media_file.touch()

        result = list(iter_media_files_in_dir(tmp_path))

        assert len(result) == 1
        assert result[0] == media_file

    def test_iter_media_files_multiple_files(self, tmp_path):
        """Test iterating directory with multiple media files."""
        files = [
            tmp_path / "test1.mp3",
            tmp_path / "test2.mp4",
            tmp_path / "test3.wav",
        ]
        for f in files:
            f.touch()

        result = list(iter_media_files_in_dir(tmp_path))

        assert len(result) == 3
        assert set(result) == set(files)

    def test_iter_media_files_ignores_non_media(self, tmp_path):
        """Test that non-media files are ignored."""
        media = tmp_path / "media.mp3"
        media.touch()
        non_media = tmp_path / "readme.txt"
        non_media.touch()

        result = list(iter_media_files_in_dir(tmp_path))

        assert len(result) == 1
        assert result[0] == media

    def test_iter_media_files_recursive(self, tmp_path):
        """Test recursive iteration through subdirectories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        file1 = tmp_path / "test1.mp3"
        file2 = subdir / "test2.mp4"
        file1.touch()
        file2.touch()

        result = list(iter_media_files_in_dir(tmp_path))

        assert len(result) == 2
        assert file1 in result
        assert file2 in result

    def test_iter_media_files_case_insensitive(self, tmp_path):
        """Test that file extensions are case-insensitive."""
        upper_case = tmp_path / "test.MP3"
        upper_case.touch()

        result = list(iter_media_files_in_dir(tmp_path))

        assert len(result) == 1


class TestExpandInputs:
    """Tests for expand_inputs function."""

    def test_expand_inputs_single_file(self, tmp_path):
        """Test expanding single file input."""
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        result = expand_inputs([str(test_file)], None)

        assert len(result) == 1
        assert result[0] == test_file

    def test_expand_inputs_directory(self, tmp_path):
        """Test expanding directory input."""
        media1 = tmp_path / "test1.mp3"
        media2 = tmp_path / "test2.mp4"
        media1.touch()
        media2.touch()

        result = expand_inputs([str(tmp_path)], None)

        assert len(result) == 2

    def test_expand_inputs_glob_pattern(self, tmp_path):
        """Test expanding glob pattern."""
        mp3_1 = tmp_path / "test1.mp3"
        mp3_2 = tmp_path / "test2.mp3"
        mp4 = tmp_path / "test.mp4"
        mp3_1.touch()
        mp3_2.touch()
        mp4.touch()

        pattern = str(tmp_path / "*.mp3")
        result = expand_inputs([pattern], None)

        assert len(result) == 2
        assert all(p.suffix == ".mp3" for p in result)

    def test_expand_inputs_deduplication(self, tmp_path):
        """Test that duplicate paths are removed."""
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        # Pass same file twice
        result = expand_inputs([str(test_file), str(test_file)], None)

        assert len(result) == 1

    def test_expand_inputs_with_glob_param(self, tmp_path):
        """Test using additional glob parameter."""
        mp3 = tmp_path / "test.mp3"
        mp4 = tmp_path / "test.mp4"
        mp3.touch()
        mp4.touch()

        result = expand_inputs([str(mp3)], str(tmp_path / "*.mp4"))

        assert len(result) == 2

    def test_expand_inputs_empty_list(self):
        """Test expanding empty input list."""
        result = expand_inputs([], None)
        assert result == []

    def test_expand_inputs_mixed(self, tmp_path):
        """Test expanding mixed input types."""
        file1 = tmp_path / "direct.mp3"
        file1.touch()

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        file2 = subdir / "from_dir.mp3"
        file2.touch()

        result = expand_inputs([str(file1), str(subdir)], None)

        assert len(result) == 2


class TestDefaultOutputFor:
    """Tests for default_output_for function."""

    def test_default_output_no_outdir(self, tmp_path):
        """Test default output without outdir."""
        input_file = tmp_path / "test.mp3"
        result = default_output_for(input_file, None, "srt", False, None)

        assert result == tmp_path / "test.srt"

    def test_default_output_with_outdir(self, tmp_path):
        """Test default output with outdir."""
        input_file = tmp_path / "test.mp3"
        outdir = tmp_path / "output"

        result = default_output_for(input_file, outdir, "srt", False, None)

        assert result == outdir / "test.srt"

    def test_default_output_formats(self, tmp_path):
        """Test different output formats."""
        input_file = tmp_path / "test.mp3"

        formats = {
            "srt": ".srt",
            "vtt": ".vtt",
            "txt": ".txt",
            "ass": ".ass",
            "json": ".json",
        }

        for fmt, ext in formats.items():
            result = default_output_for(input_file, None, fmt, False, None)
            assert result.suffix == ext

    def test_default_output_keep_structure(self, tmp_path):
        """Test output with structure preservation."""
        base_root = tmp_path
        input_file = tmp_path / "sub1" / "sub2" / "test.mp3"
        outdir = tmp_path / "output"

        result = default_output_for(input_file, outdir, "srt", True, base_root)

        # Should preserve directory structure
        assert "sub1" in str(result) and "sub2" in str(result)

    def test_default_output_without_keep_structure(self, tmp_path):
        """Test output without structure preservation."""
        input_file = tmp_path / "sub1" / "sub2" / "test.mp3"
        outdir = tmp_path / "output"

        result = default_output_for(input_file, outdir, "srt", False, None)

        # Should only use filename
        assert result == outdir / "test.srt"


class TestPreflightOne:
    """Tests for preflight_one function."""

    def test_preflight_input_not_found(self, tmp_path):
        """Test preflight when input doesn't exist."""
        input_path = tmp_path / "nonexistent.mp3"
        output_path = tmp_path / "output.srt"

        success, msg = preflight_one(input_path, output_path, False)

        assert success is False
        assert "not found" in msg.lower()

    def test_preflight_input_is_directory(self, tmp_path):
        """Test preflight when input is a directory."""
        input_path = tmp_path / "dir"
        input_path.mkdir()
        output_path = tmp_path / "output.srt"

        success, msg = preflight_one(input_path, output_path, False)

        assert success is False
        assert "directory" in msg.lower()

    def test_preflight_output_exists_no_overwrite(self, tmp_path):
        """Test preflight when output exists and overwrite disabled."""
        input_path = tmp_path / "input.mp3"
        input_path.touch()
        output_path = tmp_path / "output.srt"
        output_path.touch()

        success, msg = preflight_one(input_path, output_path, False)

        assert success is False
        assert "exists" in msg.lower() or "overwrite" in msg.lower()

    def test_preflight_output_exists_with_overwrite(self, tmp_path):
        """Test preflight when output exists and overwrite enabled."""
        input_path = tmp_path / "input.mp3"
        input_path.touch()
        output_path = tmp_path / "output.srt"
        output_path.touch()

        success, msg = preflight_one(input_path, output_path, True)

        assert success is True
        assert msg == ""

    def test_preflight_output_is_directory(self, tmp_path):
        """Test preflight when output path is a directory."""
        input_path = tmp_path / "input.mp3"
        input_path.touch()
        output_path = tmp_path / "outdir"
        output_path.mkdir()

        success, msg = preflight_one(input_path, output_path, False)

        assert success is False
        assert "directory" in msg.lower()

    def test_preflight_success(self, tmp_path):
        """Test successful preflight."""
        input_path = tmp_path / "input.mp3"
        input_path.touch()
        output_path = tmp_path / "output.srt"

        success, msg = preflight_one(input_path, output_path, False)

        assert success is True
        assert msg == ""
