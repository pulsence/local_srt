"""Tests for the system module."""
import shutil
import subprocess
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from local_srt.system import (
    ensure_parent_dir,
    which_or_none,
    ffmpeg_ok,
    ffprobe_ok,
    run_cmd_text,
    ffmpeg_version,
    ffprobe_version,
    probe_duration_seconds,
)


class TestEnsureParentDir:
    """Tests for ensure_parent_dir function."""

    def test_ensure_parent_dir_creates_directory(self, tmp_path):
        """Test that parent directory is created if it doesn't exist."""
        file_path = tmp_path / "subdir1" / "subdir2" / "file.txt"
        ensure_parent_dir(file_path)

        assert file_path.parent.exists()
        assert file_path.parent.is_dir()

    def test_ensure_parent_dir_exists_already(self, tmp_path):
        """Test that existing parent directory is not affected."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        file_path = existing_dir / "file.txt"

        ensure_parent_dir(file_path)

        assert existing_dir.exists()
        assert existing_dir.is_dir()

    def test_ensure_parent_dir_root_path(self, tmp_path):
        """Test with file directly in tmp_path."""
        file_path = tmp_path / "file.txt"
        ensure_parent_dir(file_path)

        assert tmp_path.exists()


class TestWhichOrNone:
    """Tests for which_or_none function."""

    def test_which_or_none_existing_command(self):
        """Test finding an existing command."""
        # Python should exist on the system
        result = which_or_none("python")
        if result is not None:
            assert isinstance(result, str)
            assert len(result) > 0

    @patch('shutil.which')
    def test_which_or_none_nonexistent_command(self, mock_which):
        """Test finding a nonexistent command."""
        mock_which.return_value = None
        result = which_or_none("nonexistent_command_xyz")
        assert result is None

    @patch('shutil.which')
    def test_which_or_none_calls_shutil(self, mock_which):
        """Test that which_or_none calls shutil.which."""
        mock_which.return_value = "/usr/bin/test"
        result = which_or_none("test")
        mock_which.assert_called_once_with("test")


class TestFfmpegOk:
    """Tests for ffmpeg_ok function."""

    @patch('local_srt.system.which_or_none')
    def test_ffmpeg_ok_available(self, mock_which):
        """Test when ffmpeg is available."""
        mock_which.return_value = "/usr/bin/ffmpeg"
        assert ffmpeg_ok() is True

    @patch('local_srt.system.which_or_none')
    def test_ffmpeg_ok_not_available(self, mock_which):
        """Test when ffmpeg is not available."""
        mock_which.return_value = None
        assert ffmpeg_ok() is False


class TestFfprobeOk:
    """Tests for ffprobe_ok function."""

    @patch('local_srt.system.which_or_none')
    def test_ffprobe_ok_available(self, mock_which):
        """Test when ffprobe is available."""
        mock_which.return_value = "/usr/bin/ffprobe"
        assert ffprobe_ok() is True

    @patch('local_srt.system.which_or_none')
    def test_ffprobe_ok_not_available(self, mock_which):
        """Test when ffprobe is not available."""
        mock_which.return_value = None
        assert ffprobe_ok() is False


class TestRunCmdText:
    """Tests for run_cmd_text function."""

    @patch('subprocess.run')
    def test_run_cmd_text_success(self, mock_run):
        """Test running command successfully."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "output"
        mock_process.stderr = ""
        mock_run.return_value = mock_process

        code, out, err = run_cmd_text(["echo", "test"])

        assert code == 0
        assert out == "output"
        assert err == ""

    @patch('subprocess.run')
    def test_run_cmd_text_failure(self, mock_run):
        """Test running command with failure."""
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = ""
        mock_process.stderr = "error message"
        mock_run.return_value = mock_process

        code, out, err = run_cmd_text(["false"])

        assert code == 1
        assert err == "error message"

    @patch('subprocess.run')
    def test_run_cmd_text_captures_both(self, mock_run):
        """Test that both stdout and stderr are captured."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "stdout text"
        mock_process.stderr = "stderr text"
        mock_run.return_value = mock_process

        code, out, err = run_cmd_text(["test"])

        assert out == "stdout text"
        assert err == "stderr text"


class TestFfmpegVersion:
    """Tests for ffmpeg_version function."""

    @patch('local_srt.system.ffmpeg_ok')
    def test_ffmpeg_version_not_available(self, mock_ffmpeg_ok):
        """Test getting version when ffmpeg is not available."""
        mock_ffmpeg_ok.return_value = False
        result = ffmpeg_version()
        assert result is None

    @patch('local_srt.system.ffmpeg_ok')
    @patch('local_srt.system.run_cmd_text')
    def test_ffmpeg_version_success(self, mock_run_cmd, mock_ffmpeg_ok):
        """Test getting ffmpeg version successfully."""
        mock_ffmpeg_ok.return_value = True
        mock_run_cmd.return_value = (0, "ffmpeg version 4.4.2\nmore info", "")

        result = ffmpeg_version()

        assert result == "ffmpeg version 4.4.2"

    @patch('local_srt.system.ffmpeg_ok')
    @patch('local_srt.system.run_cmd_text')
    def test_ffmpeg_version_command_fails(self, mock_run_cmd, mock_ffmpeg_ok):
        """Test when ffmpeg version command fails."""
        mock_ffmpeg_ok.return_value = True
        mock_run_cmd.return_value = (1, "", "error")

        result = ffmpeg_version()

        assert result is None


class TestFfprobeVersion:
    """Tests for ffprobe_version function."""

    @patch('local_srt.system.ffprobe_ok')
    def test_ffprobe_version_not_available(self, mock_ffprobe_ok):
        """Test getting version when ffprobe is not available."""
        mock_ffprobe_ok.return_value = False
        result = ffprobe_version()
        assert result is None

    @patch('local_srt.system.ffprobe_ok')
    @patch('local_srt.system.run_cmd_text')
    def test_ffprobe_version_success(self, mock_run_cmd, mock_ffprobe_ok):
        """Test getting ffprobe version successfully."""
        mock_ffprobe_ok.return_value = True
        mock_run_cmd.return_value = (0, "ffprobe version 4.4.2\nmore info", "")

        result = ffprobe_version()

        assert result == "ffprobe version 4.4.2"


class TestProbeDurationSeconds:
    """Tests for probe_duration_seconds function."""

    @patch('local_srt.system.ffprobe_ok')
    def test_probe_duration_ffprobe_not_available(self, mock_ffprobe_ok):
        """Test probing when ffprobe is not available."""
        mock_ffprobe_ok.return_value = False
        result = probe_duration_seconds("test.mp3")
        assert result is None

    @patch('local_srt.system.ffprobe_ok')
    @patch('local_srt.system.run_cmd_text')
    def test_probe_duration_success(self, mock_run_cmd, mock_ffprobe_ok):
        """Test successfully probing duration."""
        mock_ffprobe_ok.return_value = True
        mock_run_cmd.return_value = (0, "123.456\n", "")

        result = probe_duration_seconds("test.mp3")

        assert result == 123.456

    @patch('local_srt.system.ffprobe_ok')
    @patch('local_srt.system.run_cmd_text')
    def test_probe_duration_command_fails(self, mock_run_cmd, mock_ffprobe_ok):
        """Test when ffprobe command fails."""
        mock_ffprobe_ok.return_value = True
        mock_run_cmd.return_value = (1, "", "error")

        result = probe_duration_seconds("test.mp3")

        assert result is None

    @patch('local_srt.system.ffprobe_ok')
    @patch('local_srt.system.run_cmd_text')
    def test_probe_duration_invalid_output(self, mock_run_cmd, mock_ffprobe_ok):
        """Test when ffprobe returns invalid output."""
        mock_ffprobe_ok.return_value = True
        mock_run_cmd.return_value = (0, "not a number\n", "")

        result = probe_duration_seconds("test.mp3")

        assert result is None

    @patch('local_srt.system.ffprobe_ok')
    @patch('local_srt.system.run_cmd_text')
    def test_probe_duration_integer(self, mock_run_cmd, mock_ffprobe_ok):
        """Test probing duration returning integer."""
        mock_ffprobe_ok.return_value = True
        mock_run_cmd.return_value = (0, "60\n", "")

        result = probe_duration_seconds("test.mp3")

        assert result == 60.0
        assert isinstance(result, float)
