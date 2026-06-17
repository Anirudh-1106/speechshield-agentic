"""
Configuration management for SpeechShield.

Supports:
  • YAML-based config files (default.yaml, dev.yaml, prod.yaml)
  • Environment variable overrides (12-factor app)
  • Pydantic v2 validation and type safety
  • Environment-specific profiles (dev, test, prod)

Usage:
    from config.settings import settings, load_config
    config = settings()  # Loads environment-specific config
"""

import os
import logging
from pathlib import Path
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field
import yaml

logger = logging.getLogger("Config")


class AudioSettings(BaseModel):
    """Audio configuration."""

    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz",
    )
    channels: int = Field(
        default=1,
        description="Number of audio channels (mono)",
    )
    chunk_size: int = Field(
        default=1024,
        description="Audio chunk size in samples",
    )
    input_device: Optional[int] = Field(
        default=None,
        description="Input device index (None = default)",
    )
    output_device: Optional[int] = Field(
        default=None,
        description="Output device index (None = default)",
    )
    queue_size: int = Field(
        default=200,
        description="Max size of audio chunk queue",
    )
    silence_threshold: float = Field(
        default=500.0,
        description="RMS threshold below which audio chunks are dropped",
    )


class ASRSettings(BaseModel):
    """Automatic Speech Recognition configuration."""

    model_size: Literal["tiny", "base", "small", "medium", "large"] = Field(  # pyrefly: ignore
        default="small",
        description="Faster-Whisper model size",
    )
    model_dir: str = Field(
        default="models/",
        description="Directory where ASR models are cached",
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="ASR inference timeout",
    )
    idle_flush_ms: float = Field(
        default=500.0,
        description="Milliseconds of silence before transcribing early",
    )
    vad_filter: bool = Field(
        default=True,
        description="Enable Whisper VAD filter",
    )
    vad_min_silence_ms: int = Field(
        default=500,
        description="Whisper VAD min silence duration",
    )


class ToxicitySettings(BaseModel):
    """Toxicity detection configuration."""

    model_name: str = Field(
        default="models/toxic-bert",
        description="HuggingFace model for toxicity detection",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Toxicity score threshold (0-1)",
    )
    timeout_seconds: float = Field(
        default=5.0,
        description="Toxicity inference timeout",
    )


class RewriteSettings(BaseModel):
    """Text rewriting configuration."""

    llm_provider: Literal["ollama", "openai", "local"] = Field(
        default="ollama",
        description="LLM provider for rewriting",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    model: str = Field(
        default="gemma:1b",
        description="Ollama model name",
    )
    timeout_seconds: float = Field(
        default=10.0,
        description="Rewrite inference timeout",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="Max rewrite retry attempts",
    )
    fallback_enabled: bool = Field(
        default=True,
        description="Enable fallback to rules if LLM fails",
    )


class VoiceSettings(BaseModel):
    """Text-to-Speech configuration."""

    provider: Literal["edge_tts", "espeak", "pyttsx3"] = Field(
        default="edge_tts",
        description="TTS provider",
    )
    voice: str = Field(
        default="en-US-GuyNeural",
        description="Edge TTS voice ID",
    )
    timeout_seconds: float = Field(
        default=15.0,
        description="TTS synthesis timeout",
    )
    play_audio: bool = Field(
        default=True,
        description="Enable audio playback",
    )
    virtual_device_support: bool = Field(
        default=False,
        description="Support virtual audio devices (VB-Cable, etc)",
    )


class PipelineSettings(BaseModel):
    """Pipeline orchestration configuration."""

    latency_budget_ms: float = Field(
        default=500.0,
        ge=100.0,
        description="Target end-to-end latency budget",
    )
    default_rewrite_strictness: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Default rewrite strictness (1.0 = normal)",
    )
    latency_window_size: int = Field(
        default=100,
        ge=10,
        description="Number of passes to track for latency averaging",
    )
    graceful_shutdown_timeout: float = Field(
        default=10.0,
        ge=1.0,
        description="Seconds to wait for graceful agent shutdown",
    )


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(  # pyrefly: ignore
        default="INFO",
        description="Logging level",
    )
    log_dir: str = Field(
        default="logs/",
        description="Directory for log files",
    )
    format: Literal["text", "json"] = Field(  # pyrefly: ignore
        default="text",
        description="Log format (text or JSON for structured logging)",
    )
    enable_file_logging: bool = Field(
        default=True,
        description="Write logs to files",
    )


class DashboardSettings(BaseModel):
    """Streamlit dashboard configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable dashboard",
    )
    port: int = Field(
        default=8501,
        ge=1024,
        le=65535,
        description="Dashboard port",
    )
    refresh_interval: float = Field(
        default=1.0,
        ge=0.1,
        description="Dashboard refresh interval (seconds)",
    )
    max_rows: int = Field(
        default=500,
        ge=10,
        description="Max dashboard feed rows to keep",
    )


class Settings(BaseModel):
    """Root configuration object."""

    env: Literal["dev", "test", "prod"] = Field(
        default="dev",
        description="Environment: dev, test, or prod",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # Sub-configs
    audio: AudioSettings = Field(default_factory=AudioSettings)
    asr: ASRSettings = Field(default_factory=ASRSettings)
    toxicity: ToxicitySettings = Field(default_factory=ToxicitySettings)
    rewrite: RewriteSettings = Field(default_factory=RewriteSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)

    def to_dict(self) -> dict:
        """Export config as dictionary (for state initialization)."""
        return self.model_dump()


def load_config(env: Optional[str] = None) -> Settings:
    """
    Load configuration from YAML file + environment variables.

    Priority (highest to lowest):
    1. Environment variables (SPEECHSHIELD_*)
    2. YAML file (config/{env}.yaml)
    3. Default YAML (config/default.yaml)
    4. Pydantic defaults

    Args:
        env: Environment name (dev, test, prod). If None, reads from SPEECHSHIELD_ENV.

    Returns:
        Settings object with merged configuration.

    Raises:
        FileNotFoundError: If required config files not found.
        ValueError: If configuration validation fails.
    """
    if env is None:
        env = os.getenv("SPEECHSHIELD_ENV", "dev")

    config_dir = Path(__file__).parent
    default_yaml = config_dir / "default.yaml"
    env_yaml = config_dir / f"{env}.yaml"

    logger.info(f"Loading config for environment: {env}")

    # Load default config
    if not default_yaml.exists():
        logger.warning(f"Default config not found at {default_yaml}, using Pydantic defaults")
        config_data: dict[str, Any] = {}
    else:
        with open(default_yaml) as f:
            config_data = yaml.safe_load(f) or {}
        logger.info(f"Loaded default config from {default_yaml}")

    # Merge environment-specific config
    if env_yaml.exists():
        with open(env_yaml) as f:
            env_data = yaml.safe_load(f) or {}
        _deep_merge(config_data, env_data)
        logger.info(f"Merged {env} config from {env_yaml}")
    else:
        logger.warning(f"Environment config not found at {env_yaml}, using defaults")

    # Apply environment variable overrides
    _apply_env_overrides(config_data)

    # Create settings object
    settings = Settings(**config_data)
    logger.info(f"Configuration loaded: env={settings.env}, debug={settings.debug}")

    return settings


def _apply_env_overrides(config_data: dict) -> None:
    """
    Apply SPEECHSHIELD_* environment variables to config_data.
    
    Maps flat env var names to nested config paths.
    E.g., SPEECHSHIELD_SAMPLE_RATE -> config_data['audio']['sample_rate']
    """
    env_mapping = {
        # Top level
        "SPEECHSHIELD_ENV": ("env",),
        "SPEECHSHIELD_DEBUG": ("debug",),
        # Audio
        "SPEECHSHIELD_SAMPLE_RATE": ("audio", "sample_rate"),
        "SPEECHSHIELD_CHANNELS": ("audio", "channels"),
        "SPEECHSHIELD_CHUNK_SIZE": ("audio", "chunk_size"),
        "SPEECHSHIELD_INPUT_DEVICE": ("audio", "input_device"),
        "SPEECHSHIELD_OUTPUT_DEVICE": ("audio", "output_device"),
        "SPEECHSHIELD_AUDIO_QUEUE_SIZE": ("audio", "queue_size"),
        # ASR
        "SPEECHSHIELD_ASR_MODEL_SIZE": ("asr", "model_size"),
        "SPEECHSHIELD_ASR_MODEL_DIR": ("asr", "model_dir"),
        "SPEECHSHIELD_ASR_TIMEOUT": ("asr", "timeout_seconds"),
        # Toxicity
        "SPEECHSHIELD_TOXICITY_MODEL": ("toxicity", "model_name"),
        "SPEECHSHIELD_TOXICITY_THRESHOLD": ("toxicity", "threshold"),
        "SPEECHSHIELD_TOXICITY_TIMEOUT": ("toxicity", "timeout_seconds"),
        # Rewrite
        "SPEECHSHIELD_REWRITE_PROVIDER": ("rewrite", "llm_provider"),
        "SPEECHSHIELD_REWRITE_MODEL": ("rewrite", "model"),
        "SPEECHSHIELD_OLLAMA_BASE_URL": ("rewrite", "ollama_base_url"),
        "SPEECHSHIELD_REWRITE_TIMEOUT": ("rewrite", "timeout_seconds"),
        "SPEECHSHIELD_REWRITE_MAX_RETRIES": ("rewrite", "max_retries"),
        "SPEECHSHIELD_REWRITE_FALLBACK": ("rewrite", "fallback_enabled"),
        # Voice
        "SPEECHSHIELD_TTS_PROVIDER": ("voice", "provider"),
        "SPEECHSHIELD_TTS_VOICE": ("voice", "voice"),
        "SPEECHSHIELD_TTS_TIMEOUT": ("voice", "timeout_seconds"),
        "SPEECHSHIELD_PLAY_AUDIO": ("voice", "play_audio"),
        "SPEECHSHIELD_VIRTUAL_DEVICE": ("voice", "virtual_device_support"),
        # Pipeline
        "SPEECHSHIELD_LATENCY_BUDGET": ("pipeline", "latency_budget_ms"),
        "SPEECHSHIELD_STRICTNESS": ("pipeline", "default_rewrite_strictness"),
        "SPEECHSHIELD_LATENCY_WINDOW": ("pipeline", "latency_window_size"),
        "SPEECHSHIELD_SHUTDOWN_TIMEOUT": ("pipeline", "graceful_shutdown_timeout"),
        # Logging
        "SPEECHSHIELD_LOG_LEVEL": ("logging", "level"),
        "SPEECHSHIELD_LOG_DIR": ("logging", "log_dir"),
        "SPEECHSHIELD_LOG_FORMAT": ("logging", "format"),
        "SPEECHSHIELD_FILE_LOGGING": ("logging", "enable_file_logging"),
        # Dashboard
        "SPEECHSHIELD_DASHBOARD_ENABLED": ("dashboard", "enabled"),
        "SPEECHSHIELD_DASHBOARD_PORT": ("dashboard", "port"),
        "SPEECHSHIELD_DASHBOARD_REFRESH": ("dashboard", "refresh_interval"),
        "SPEECHSHIELD_DASHBOARD_MAX_ROWS": ("dashboard", "max_rows"),
    }
    
    for env_var, path in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None:
            # Ensure nested dicts exist
            current = config_data
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Convert value based on context
            current[path[-1]] = _parse_env_value(value)


def _parse_env_value(value: str):
    """Parse environment variable value to appropriate type."""
    # Handle booleans
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    # Handle null
    if value.lower() in ("null", "none"):
        return None
    # Try float
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        # Return as string
        return value


def _deep_merge(base: dict, updates: dict) -> None:
    """Recursively merge updates into base dictionary."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# Singleton instance (lazy-loaded on first access)
_settings_instance: Optional[Settings] = None


def settings() -> Settings:
    """Get global settings instance (lazy-loaded)."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = load_config()
    return _settings_instance


def reload_settings() -> Settings:
    """Reload settings from disk/env (for testing)."""
    global _settings_instance
    _settings_instance = load_config()
    return _settings_instance
