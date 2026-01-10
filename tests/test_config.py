"""Tests for the config module."""
import json
import pytest
from pathlib import Path
from local_srt.config import (
    PRESETS,
    MODE_ALIASES,
    load_config_file,
    apply_overrides,
)
from local_srt.models import ResolvedConfig


class TestPresets:
    """Tests for PRESETS constant."""

    def test_presets_exist(self):
        """Test that presets dictionary exists."""
        assert PRESETS is not None
        assert isinstance(PRESETS, dict)

    def test_preset_keys(self):
        """Test that expected preset keys exist."""
        expected_keys = {"shorts", "yt", "podcast"}
        assert set(PRESETS.keys()) == expected_keys

    def test_preset_structure(self):
        """Test that each preset has the expected structure."""
        for name, preset in PRESETS.items():
            assert isinstance(preset, dict), f"Preset {name} is not a dict"
            # Check for common keys
            assert "max_chars" in preset
            assert "max_lines" in preset
            assert "target_cps" in preset

    def test_shorts_preset(self):
        """Test shorts preset values."""
        shorts = PRESETS["shorts"]
        assert shorts["max_chars"] == 18
        assert shorts["max_lines"] == 1
        assert shorts["target_cps"] == 18.0

    def test_yt_preset(self):
        """Test yt preset values."""
        yt = PRESETS["yt"]
        assert yt["max_chars"] == 42
        assert yt["max_lines"] == 2
        assert yt["target_cps"] == 17.0

    def test_podcast_preset(self):
        """Test podcast preset values."""
        podcast = PRESETS["podcast"]
        assert podcast["max_chars"] == 40
        assert podcast["max_lines"] == 2
        assert podcast["target_cps"] == 16.0
        assert podcast["prefer_punct_splits"] is True


class TestModeAliases:
    """Tests for MODE_ALIASES constant."""

    def test_mode_aliases_exist(self):
        """Test that mode aliases dictionary exists."""
        assert MODE_ALIASES is not None
        assert isinstance(MODE_ALIASES, dict)

    def test_mode_alias_mappings(self):
        """Test that mode aliases map correctly."""
        assert MODE_ALIASES["short"] == "shorts"
        assert MODE_ALIASES["shorts"] == "shorts"
        assert MODE_ALIASES["yt"] == "yt"
        assert MODE_ALIASES["youtube"] == "yt"
        assert MODE_ALIASES["pod"] == "podcast"
        assert MODE_ALIASES["podcast"] == "podcast"

    def test_all_aliases_map_to_presets(self):
        """Test that all aliases map to valid preset names."""
        for alias, preset_name in MODE_ALIASES.items():
            assert preset_name in PRESETS


class TestLoadConfigFile:
    """Tests for load_config_file function."""

    def test_load_config_none_path(self):
        """Test loading config with None path returns empty dict."""
        result = load_config_file(None)
        assert result == {}

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config_file("/nonexistent/path/config.json")

    def test_load_valid_config_file(self, tmp_path):
        """Test loading a valid config file."""
        config_data = {
            "max_chars": 50,
            "max_lines": 3,
            "target_cps": 20.0,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        result = load_config_file(str(config_file))
        assert result == config_data

    def test_load_config_invalid_json(self, tmp_path):
        """Test loading invalid JSON raises error."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("not valid json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_config_file(str(config_file))

    def test_load_config_non_dict(self, tmp_path):
        """Test loading JSON that isn't a dict raises error."""
        config_file = tmp_path / "array.json"
        config_file.write_text(json.dumps(["array", "data"]), encoding="utf-8")

        with pytest.raises(ValueError, match="Config must be a JSON object"):
            load_config_file(str(config_file))

    def test_load_empty_config_file(self, tmp_path):
        """Test loading empty config file."""
        config_file = tmp_path / "empty.json"
        config_file.write_text(json.dumps({}), encoding="utf-8")

        result = load_config_file(str(config_file))
        assert result == {}


class TestApplyOverrides:
    """Tests for apply_overrides function."""

    def test_apply_overrides_empty(self):
        """Test applying empty overrides."""
        base = ResolvedConfig()
        result = apply_overrides(base, {})

        # Should return new instance with same values
        assert result is not base
        assert result.max_chars == base.max_chars
        assert result.max_lines == base.max_lines

    def test_apply_overrides_single_field(self):
        """Test applying override to single field."""
        base = ResolvedConfig()
        result = apply_overrides(base, {"max_chars": 50})

        assert result.max_chars == 50
        # Other fields should remain default
        assert result.max_lines == base.max_lines
        assert result.target_cps == base.target_cps

    def test_apply_overrides_multiple_fields(self):
        """Test applying overrides to multiple fields."""
        base = ResolvedConfig()
        overrides = {
            "max_chars": 50,
            "max_lines": 3,
            "target_cps": 20.0,
            "min_dur": 0.5,
        }
        result = apply_overrides(base, overrides)

        assert result.max_chars == 50
        assert result.max_lines == 3
        assert result.target_cps == 20.0
        assert result.min_dur == 0.5

    def test_apply_overrides_invalid_field_ignored(self):
        """Test that invalid field names are ignored."""
        base = ResolvedConfig()
        overrides = {
            "max_chars": 50,
            "invalid_field": "should be ignored",
        }
        result = apply_overrides(base, overrides)

        assert result.max_chars == 50
        assert not hasattr(result, "invalid_field")

    def test_apply_overrides_preserves_base(self):
        """Test that base config is not modified."""
        base = ResolvedConfig(max_chars=42)
        original_max_chars = base.max_chars

        result = apply_overrides(base, {"max_chars": 50})

        # Base should be unchanged
        assert base.max_chars == original_max_chars
        # Result should have new value
        assert result.max_chars == 50

    def test_apply_overrides_with_preset(self):
        """Test applying preset as overrides."""
        base = ResolvedConfig()
        result = apply_overrides(base, PRESETS["shorts"])

        assert result.max_chars == PRESETS["shorts"]["max_chars"]
        assert result.max_lines == PRESETS["shorts"]["max_lines"]
        assert result.target_cps == PRESETS["shorts"]["target_cps"]

    def test_apply_overrides_boolean_fields(self):
        """Test applying overrides to boolean fields."""
        base = ResolvedConfig()
        overrides = {
            "allow_commas": False,
            "prefer_punct_splits": True,
            "vad_filter": False,
        }
        result = apply_overrides(base, overrides)

        assert result.allow_commas is False
        assert result.prefer_punct_splits is True
        assert result.vad_filter is False

    def test_apply_overrides_float_fields(self):
        """Test applying overrides to float fields."""
        base = ResolvedConfig()
        overrides = {
            "min_gap": 0.1,
            "pad": 0.05,
            "silence_threshold_db": -40.0,
        }
        result = apply_overrides(base, overrides)

        assert result.min_gap == 0.1
        assert result.pad == 0.05
        assert result.silence_threshold_db == -40.0
