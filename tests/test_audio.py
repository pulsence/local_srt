"""Tests for the audio module."""
import subprocess
import pytest
from unittest.mock import patch, MagicMock
from local_srt.audio import (
    _SILENCE_START_RE,
    _SILENCE_END_RE,
    detect_silences,
    to_wav_16k_mono,
)


class TestSilenceRegex:
    """Tests for silence detection regex patterns."""

    def test_silence_start_regex(self):
        """Test silence_start regex pattern."""
        line = "silence_start: 1.234"
        match = _SILENCE_START_RE.search(line)
        assert match is not None
        assert match.group(1) == "1.234"

    def test_silence_end_regex(self):
        """Test silence_end regex pattern."""
        line = "silence_end: 5.678"
        match = _SILENCE_END_RE.search(line)
        assert match is not None
        assert match.group(1) == "5.678"

    def test_silence_regex_no_match(self):
        """Test that regex doesn't match invalid lines."""
        line = "not a silence line"
        assert _SILENCE_START_RE.search(line) is None
        assert _SILENCE_END_RE.search(line) is None


class TestDetectSilences:
    """Tests for detect_silences function."""

    @patch('local_srt.audio.ffmpeg_ok')
    def test_detect_silences_ffmpeg_not_available(self, mock_ffmpeg_ok):
        """Test detect_silences when ffmpeg is not available."""
        mock_ffmpeg_ok.return_value = False

        result = detect_silences("test.wav", min_silence_dur=0.5, silence_threshold_db=-35.0)

        assert result == []

    @patch('local_srt.audio.ffmpeg_ok')
    @patch('local_srt.audio.run_cmd_text')
    def test_detect_silences_command_fails(self, mock_run_cmd, mock_ffmpeg_ok):
        """Test detect_silences when command fails."""
        mock_ffmpeg_ok.return_value = True
        mock_run_cmd.return_value = (1, "", "error")

        result = detect_silences("test.wav", min_silence_dur=0.5, silence_threshold_db=-35.0)

        assert result == []

    @patch('local_srt.audio.ffmpeg_ok')
    @patch('local_srt.audio.run_cmd_text')
    def test_detect_silences_single_silence(self, mock_run_cmd, mock_ffmpeg_ok):
        """Test detecting single silence period."""
        mock_ffmpeg_ok.return_value = True
        stderr = "silence_start: 1.0\nsilence_end: 2.0\n"
        mock_run_cmd.return_value = (0, "", stderr)

        result = detect_silences("test.wav", min_silence_dur=0.5, silence_threshold_db=-35.0)

        assert len(result) == 1
        assert result[0] == (1.0, 2.0)

    @patch('local_srt.audio.ffmpeg_ok')
    @patch('local_srt.audio.run_cmd_text')
    def test_detect_silences_multiple_silences(self, mock_run_cmd, mock_ffmpeg_ok):
        """Test detecting multiple silence periods."""
        mock_ffmpeg_ok.return_value = True
        stderr = (
            "silence_start: 1.0\nsilence_end: 2.0\n"
            "silence_start: 5.0\nsilence_end: 6.0\n"
        )
        mock_run_cmd.return_value = (0, "", stderr)

        result = detect_silences("test.wav", min_silence_dur=0.5, silence_threshold_db=-35.0)

        assert len(result) == 2
        assert result[0] == (1.0, 2.0)
        assert result[1] == (5.0, 6.0)

    @patch('local_srt.audio.ffmpeg_ok')
    @patch('local_srt.audio.run_cmd_text')
    @patch('local_srt.audio.probe_duration_seconds')
    def test_detect_silences_pending_start(self, mock_probe, mock_run_cmd, mock_ffmpeg_ok):
        """Test handling silence that extends to end of file."""
        mock_ffmpeg_ok.return_value = True
        mock_probe.return_value = 10.0
        stderr = "silence_start: 8.0\n"  # No end
        mock_run_cmd.return_value = (0, "", stderr)

        result = detect_silences("test.wav", min_silence_dur=0.5, silence_threshold_db=-35.0)

        # Should extend to file duration
        assert len(result) == 1
        assert result[0][0] == 8.0
        assert result[0][1] == 10.0

    @patch('local_srt.audio.ffmpeg_ok')
    @patch('local_srt.audio.run_cmd_text')
    def test_detect_silences_merges_overlapping(self, mock_run_cmd, mock_ffmpeg_ok):
        """Test merging overlapping silence periods."""
        mock_ffmpeg_ok.return_value = True
        stderr = (
            "silence_start: 1.0\nsilence_end: 3.0\n"
            "silence_start: 2.5\nsilence_end: 4.0\n"  # Overlaps with first
        )
        mock_run_cmd.return_value = (0, "", stderr)

        result = detect_silences("test.wav", min_silence_dur=0.5, silence_threshold_db=-35.0)

        # Should be merged into one silence period
        assert len(result) == 1
        assert result[0][0] == 1.0
        assert result[0][1] == 4.0

    @patch('local_srt.audio.ffmpeg_ok')
    @patch('local_srt.audio.run_cmd_text')
    def test_detect_silences_sorts_results(self, mock_run_cmd, mock_ffmpeg_ok):
        """Test that results are sorted by start time."""
        mock_ffmpeg_ok.return_value = True
        stderr = (
            "silence_start: 5.0\nsilence_end: 6.0\n"
            "silence_start: 1.0\nsilence_end: 2.0\n"
        )
        mock_run_cmd.return_value = (0, "", stderr)

        result = detect_silences("test.wav", min_silence_dur=0.5, silence_threshold_db=-35.0)

        # Should be sorted
        assert result[0][0] < result[1][0]

    @patch('local_srt.audio.ffmpeg_ok')
    @patch('local_srt.audio.run_cmd_text')
    def test_detect_silences_filters_zero_duration(self, mock_run_cmd, mock_ffmpeg_ok):
        """Test filtering out zero-duration silences."""
        mock_ffmpeg_ok.return_value = True
        stderr = "silence_start: 2.0\nsilence_end: 2.0\n"  # Zero duration
        mock_run_cmd.return_value = (0, "", stderr)

        result = detect_silences("test.wav", min_silence_dur=0.5, silence_threshold_db=-35.0)

        # Zero duration silence should be filtered out
        assert len(result) == 0

    @patch('local_srt.audio.ffmpeg_ok')
    @patch('local_srt.audio.run_cmd_text')
    def test_detect_silences_empty_output(self, mock_run_cmd, mock_ffmpeg_ok):
        """Test handling empty output (no silences detected)."""
        mock_ffmpeg_ok.return_value = True
        mock_run_cmd.return_value = (0, "", "")

        result = detect_silences("test.wav", min_silence_dur=0.5, silence_threshold_db=-35.0)

        assert result == []


class TestToWav16kMono:
    """Tests for to_wav_16k_mono function."""

    @patch('subprocess.run')
    def test_to_wav_16k_mono_success(self, mock_run):
        """Test successful audio conversion."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stderr = ""
        mock_run.return_value = mock_process

        # Should not raise exception
        to_wav_16k_mono("input.mp3", "output.wav")

        # Verify ffmpeg was called
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "ffmpeg" in args
        assert "input.mp3" in args
        assert "output.wav" in args

    @patch('subprocess.run')
    def test_to_wav_16k_mono_parameters(self, mock_run):
        """Test that correct ffmpeg parameters are used."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stderr = ""
        mock_run.return_value = mock_process

        to_wav_16k_mono("input.mp3", "output.wav")

        args = mock_run.call_args[0][0]
        # Check for mono (-ac 1) and 16kHz (-ar 16000)
        assert "-ac" in args
        assert "1" in args
        assert "-ar" in args
        assert "16000" in args
        # Check for no video (-vn)
        assert "-vn" in args

    @patch('subprocess.run')
    def test_to_wav_16k_mono_failure(self, mock_run):
        """Test handling of conversion failure."""
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stderr = "error message"
        mock_run.return_value = mock_process

        with pytest.raises(subprocess.CalledProcessError):
            to_wav_16k_mono("input.mp3", "output.wav")

    @patch('subprocess.run')
    def test_to_wav_16k_mono_error_truncation(self, mock_run):
        """Test that long error messages are truncated."""
        mock_process = MagicMock()
        mock_process.returncode = 1
        # Create error message with many lines
        error_lines = ["error line " + str(i) for i in range(100)]
        mock_process.stderr = "\n".join(error_lines)
        mock_run.return_value = mock_process

        try:
            to_wav_16k_mono("input.mp3", "output.wav")
        except subprocess.CalledProcessError as e:
            # Error should be truncated to last 20 lines
            assert e.stderr is not None
            error_line_count = len(e.stderr.split("\n"))
            assert error_line_count <= 21  # 20 lines + potential empty line

    @patch('subprocess.run')
    def test_to_wav_16k_mono_overwrite_flag(self, mock_run):
        """Test that overwrite flag (-y) is used."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stderr = ""
        mock_run.return_value = mock_process

        to_wav_16k_mono("input.mp3", "output.wav")

        args = mock_run.call_args[0][0]
        # Check for overwrite flag
        assert "-y" in args

    @patch('subprocess.run')
    def test_to_wav_16k_mono_loglevel(self, mock_run):
        """Test that error loglevel is used."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stderr = ""
        mock_run.return_value = mock_process

        to_wav_16k_mono("input.mp3", "output.wav")

        args = mock_run.call_args[0][0]
        # Check for loglevel error
        assert "-loglevel" in args
        assert "error" in args
