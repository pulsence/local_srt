"""Tests for the logging_utils module."""
import sys
import pytest
from io import StringIO
from local_srt.logging_utils import log, warn, die, progress_line, progress_done, format_duration


class TestLog:
    """Tests for log function."""

    def test_log_output(self, capsys):
        """Test that log outputs to stdout."""
        log("test message", quiet=False)
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_log_quiet_mode(self, capsys):
        """Test that log respects quiet mode."""
        log("test message", quiet=True)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_log_default_not_quiet(self, capsys):
        """Test that log defaults to not quiet."""
        log("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out


class TestWarn:
    """Tests for warn function."""

    def test_warn_output(self, capsys):
        """Test that warn outputs to stderr with WARNING prefix."""
        warn("test warning", quiet=False)
        captured = capsys.readouterr()
        assert "WARNING: test warning" in captured.err

    def test_warn_quiet_mode(self, capsys):
        """Test that warn respects quiet mode."""
        warn("test warning", quiet=True)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_warn_default_not_quiet(self, capsys):
        """Test that warn defaults to not quiet."""
        warn("test warning")
        captured = capsys.readouterr()
        assert "WARNING: test warning" in captured.err


class TestDie:
    """Tests for die function."""

    def test_die_output(self, capsys):
        """Test that die outputs to stderr with ERROR prefix."""
        code = die("test error", code=1)
        captured = capsys.readouterr()
        assert "ERROR: test error" in captured.err
        assert code == 1

    def test_die_default_code(self, capsys):
        """Test that die defaults to exit code 1."""
        code = die("test error")
        assert code == 1

    def test_die_custom_code(self, capsys):
        """Test that die accepts custom exit code."""
        code = die("test error", code=42)
        assert code == 42

    def test_die_returns_code(self):
        """Test that die returns the exit code."""
        result = die("error", code=5)
        assert result == 5


class TestProgressLine:
    """Tests for progress_line function."""

    def test_progress_line_output(self, capsys):
        """Test that progress_line outputs to stdout."""
        progress_line("processing...", enabled=True, quiet=False)
        captured = capsys.readouterr()
        assert "processing..." in captured.out

    def test_progress_line_disabled(self, capsys):
        """Test that progress_line respects enabled flag."""
        progress_line("processing...", enabled=False, quiet=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_progress_line_quiet(self, capsys):
        """Test that progress_line respects quiet flag."""
        progress_line("processing...", enabled=True, quiet=True)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_progress_line_truncation(self, capsys):
        """Test that progress_line truncates long messages."""
        long_msg = "x" * 200
        progress_line(long_msg, enabled=True, quiet=False)
        captured = capsys.readouterr()
        # Should be truncated and padded to 160 chars
        assert len(captured.out) <= 162  # 160 + \r + potential newline

    def test_progress_line_overwrites(self, capsys):
        """Test that progress_line uses carriage return."""
        progress_line("test", enabled=True, quiet=False)
        captured = capsys.readouterr()
        # Should start with carriage return
        assert captured.out.startswith("\r")


class TestProgressDone:
    """Tests for progress_done function."""

    def test_progress_done_output(self, capsys):
        """Test that progress_done outputs newline."""
        progress_done(enabled=True, quiet=False)
        captured = capsys.readouterr()
        assert captured.out == "\n"

    def test_progress_done_disabled(self, capsys):
        """Test that progress_done respects enabled flag."""
        progress_done(enabled=False, quiet=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_progress_done_quiet(self, capsys):
        """Test that progress_done respects quiet flag."""
        progress_done(enabled=True, quiet=True)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_format_duration_seconds(self):
        """Test formatting duration in seconds only."""
        assert format_duration(45) == "0:45"
        assert format_duration(5) == "0:05"

    def test_format_duration_minutes(self):
        """Test formatting duration with minutes."""
        assert format_duration(90) == "1:30"
        assert format_duration(600) == "10:00"

    def test_format_duration_hours(self):
        """Test formatting duration with hours."""
        assert format_duration(3600) == "1:00:00"
        assert format_duration(3661) == "1:01:01"
        assert format_duration(7325) == "2:02:05"

    def test_format_duration_zero(self):
        """Test formatting zero duration."""
        assert format_duration(0) == "0:00"

    def test_format_duration_negative(self):
        """Test formatting negative duration (treated as 0)."""
        assert format_duration(-10) == "0:00"

    def test_format_duration_float(self):
        """Test formatting float duration (truncates to int)."""
        assert format_duration(45.7) == "0:45"
        assert format_duration(90.9) == "1:30"

    def test_format_duration_large_hours(self):
        """Test formatting very large durations."""
        result = format_duration(36000)  # 10 hours
        assert result == "10:00:00"

    def test_format_duration_all_components(self):
        """Test formatting with all time components."""
        # 2 hours, 15 minutes, 30 seconds = 8130 seconds
        assert format_duration(8130) == "2:15:30"
