"""
Tests for the configuration system.

Tests cover:
  • Loading default config
  • Environment variable overrides
  • YAML file merging
  • Validation
  • Environment profiles (dev/prod)
"""

import os
import tempfile
from pathlib import Path
import pytest
from config.settings import (
    Settings,
    load_config,
    reload_settings,
    _deep_merge,
)


class TestDefaultConfig:
    """Test loading default configuration."""

    def test_load_default_config(self):
        """Test loading default.yaml."""
        # This will load from the actual config directory
        config = load_config(env="dev")
        assert config.env == "dev"
        assert config.debug is True  # dev has debug=true
        assert config.audio.sample_rate == 16000
        assert config.toxicity.threshold == 0.5

    def test_load_prod_config(self):
        """Test loading prod.yaml with stricter settings."""
        config = load_config(env="prod")
        assert config.env == "prod"
        assert config.debug is False
        assert config.pipeline.latency_budget_ms == 300.0
        assert config.toxicity.threshold == 0.6  # Higher in prod

    def test_load_test_config(self):
        """Test loading test.yaml for CI/CD."""
        config = load_config(env="test")
        assert config.env == "test"
        assert config.dashboard.enabled is False


class TestEnvironmentVariableOverrides:
    """Test environment variable override priority."""

    def test_env_var_overrides_yaml(self, monkeypatch):
        """Test that SPEECHSHIELD_* env vars override YAML."""
        monkeypatch.setenv("SPEECHSHIELD_ENV", "dev")
        monkeypatch.setenv("SPEECHSHIELD_DEBUG", "true")
        monkeypatch.setenv("SPEECHSHIELD_SAMPLE_RATE", "48000")
        monkeypatch.setenv("SPEECHSHIELD_TOXICITY_THRESHOLD", "0.7")

        # Reload to pick up env vars
        config = load_config(env="dev")

        assert config.debug is True
        assert config.audio.sample_rate == 48000
        assert config.toxicity.threshold == 0.7

    def test_threshold_validation(self, monkeypatch):
        """Test that invalid threshold values are rejected."""
        monkeypatch.setenv("SPEECHSHIELD_TOXICITY_THRESHOLD", "1.5")  # Invalid

        with pytest.raises(ValueError):
            load_config(env="dev")

    def test_latency_budget_minimum(self, monkeypatch):
        """Test that latency_budget_ms must be >= 100."""
        monkeypatch.setenv("SPEECHSHIELD_LATENCY_BUDGET", "50")  # Too low

        with pytest.raises(ValueError):
            load_config(env="dev")


class TestConfigStructure:
    """Test configuration structure and defaults."""

    def test_audio_settings(self):
        """Test audio settings structure."""
        config = load_config(env="dev")
        assert config.audio.sample_rate > 0
        assert config.audio.channels in (1, 2)
        assert config.audio.chunk_size > 0
        assert config.audio.queue_size > 0

    def test_asr_settings(self):
        """Test ASR settings structure."""
        config = load_config(env="dev")
        assert config.asr.model_size in ("tiny", "base", "small", "medium", "large")
        assert config.asr.timeout_seconds > 0

    def test_toxicity_settings(self):
        """Test toxicity settings structure."""
        config = load_config(env="dev")
        assert 0 <= config.toxicity.threshold <= 1
        assert config.toxicity.timeout_seconds > 0

    def test_rewrite_settings(self):
        """Test rewrite settings structure."""
        config = load_config(env="dev")
        assert config.rewrite.llm_provider in ("ollama", "openai", "local")
        assert config.rewrite.timeout_seconds > 0
        assert config.rewrite.max_retries >= 0

    def test_voice_settings(self):
        """Test voice settings structure."""
        config = load_config(env="dev")
        assert config.voice.provider in ("edge_tts", "espeak", "pyttsx3")
        assert config.voice.timeout_seconds > 0

    def test_pipeline_settings(self):
        """Test pipeline settings structure."""
        config = load_config(env="dev")
        assert config.pipeline.latency_budget_ms >= 100
        assert config.pipeline.default_rewrite_strictness >= 0.5
        assert config.pipeline.latency_window_size >= 10

    def test_logging_settings(self):
        """Test logging settings structure."""
        config = load_config(env="dev")
        assert config.logging.level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        assert config.logging.format in ("text", "json")

    def test_dashboard_settings(self):
        """Test dashboard settings structure."""
        config = load_config(env="dev")
        assert 1024 <= config.dashboard.port <= 65535
        assert config.dashboard.refresh_interval > 0


class TestDeepMerge:
    """Test dictionary deep merge utility."""

    def test_shallow_merge(self):
        """Test merging top-level keys."""
        base = {"a": 1, "b": 2}
        updates = {"b": 3, "c": 4}
        _deep_merge(base, updates)
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {"audio": {"sample_rate": 16000, "channels": 1}}
        updates = {"audio": {"sample_rate": 48000}}
        _deep_merge(base, updates)
        assert base["audio"]["sample_rate"] == 48000
        assert base["audio"]["channels"] == 1  # Preserved from base

    def test_deep_merge_overwrites_non_dict(self):
        """Test that non-dict values are overwritten."""
        base = {"value": {"nested": "dict"}}
        updates = {"value": "scalar"}
        _deep_merge(base, updates)
        assert base["value"] == "scalar"


class TestConfigExport:
    """Test configuration export functionality."""

    def test_to_dict(self):
        """Test exporting config to dictionary."""
        config = load_config(env="dev")
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["env"] == "dev"
        assert "audio" in config_dict
        assert "asr" in config_dict
        assert "toxicity" in config_dict
        assert "rewrite" in config_dict
        assert "voice" in config_dict
        assert "pipeline" in config_dict
        assert "logging" in config_dict
        assert "dashboard" in config_dict


class TestEnvironmentProfiles:
    """Test differences between environment profiles."""

    def test_dev_vs_prod_differences(self):
        """Test that dev and prod configs differ appropriately."""
        dev = load_config(env="dev")
        prod = load_config(env="prod")

        # Dev should be more lenient
        assert dev.pipeline.latency_budget_ms > prod.pipeline.latency_budget_ms
        assert dev.logging.level == "DEBUG"
        assert prod.logging.level == "WARNING"

        # Prod should have higher threshold to reduce false positives
        assert prod.toxicity.threshold >= dev.toxicity.threshold

    def test_test_vs_prod_differences(self):
        """Test that test config is lenient for CI/CD."""
        test = load_config(env="test")
        prod = load_config(env="prod")

        # Test should have very relaxed latency
        assert test.pipeline.latency_budget_ms > prod.pipeline.latency_budget_ms

        # Test dashboard disabled
        assert test.dashboard.enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
