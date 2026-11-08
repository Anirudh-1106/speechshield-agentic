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
        default=300.0,
        ge=0.0,
        description="Minimum RMS (int16 scale) for a frame to count as speech. "
        "The segmenter adapts above this floor to the measured noise level.",
    )
    capture_enabled: bool = Field(
        default=True,
        description="Capture audio at all. Disable for headless/API-only deployments (e.g. Docker).",
    )


class SegmenterSettings(BaseModel):
    """Voice-activity based utterance segmentation.

    Audio is cut into utterances at natural pauses instead of fixed
    windows, so ASR never sees half-words and every utterance carries a
    single correlation ID end-to-end.
    """

    min_silence_ms: int = Field(
        default=400,
        ge=50,
        description="Silence needed to close an utterance",
    )
    max_segment_seconds: float = Field(
        default=4.0,
        ge=1.0,
        le=30.0,
        description="Force-close utterances longer than this",
    )
    min_speech_ms: int = Field(
        default=250,
        ge=0,
        description="Discard utterances with less voiced audio than this (clicks, coughs)",
    )
    pre_roll_ms: int = Field(
        default=200,
        ge=0,
        description="Audio kept from before speech onset so first syllables are not clipped",
    )
    noise_multiplier: float = Field(
        default=3.0,
        ge=1.0,
        description="Speech threshold = max(silence_threshold, noise_floor * noise_multiplier)",
    )


class ASRSettings(BaseModel):
    """Automatic Speech Recognition configuration."""

    model_size: Literal[  # pyrefly: ignore
        "auto", "tiny", "tiny.en", "base", "base.en", "small", "small.en",
        "medium", "medium.en", "large", "large-v3", "distil-small.en",
        "distil-medium.en", "distil-large-v3", "turbo",
    ] = Field(
        default="auto",
        description="Faster-Whisper model (auto = small on GPU, base.en on CPU)",
    )
    model_dir: str = Field(
        default="models/",
        description="Directory where ASR models are cached",
    )
    device: Literal["cpu", "cuda", "auto"] = Field(  # pyrefly: ignore
        default="auto",
        description="Inference device (auto = cuda if available)",
    )
    compute_type: str = Field(
        default="auto",
        description="CTranslate2 compute type (auto = int8 on CPU, float16 on GPU)",
    )
    cpu_threads: int = Field(
        default=0,
        ge=0,
        description="CPU threads for CTranslate2 (0 = library default)",
    )
    beam_size: int = Field(
        default=1,
        ge=1,
        description="Beam size (1 = greedy, fastest)",
    )
    language: Optional[str] = Field(
        default="en",
        description="Spoken language (None = auto-detect, slower)",
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="ASR inference timeout",
    )
    vad_filter: bool = Field(
        default=True,
        description="Enable Whisper VAD filter",
    )
    vad_min_silence_ms: int = Field(
        default=500,
        description="Whisper VAD min silence duration",
    )
    word_timestamps: bool = Field(
        default=True,
        description="Emit per-word timings (needed to bleep individual words in interceptor mode)",
    )
    no_speech_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Drop segments Whisper thinks are not speech (hallucination guard)",
    )
    min_avg_logprob: float = Field(
        default=-1.0,
        description="Drop low-confidence segments below this average log-probability",
    )


class ToxicitySettings(BaseModel):
    """Toxicity detection configuration."""

    model_name: str = Field(
        default="unitary/toxic-bert",
        description="HuggingFace model ID or local path for toxicity detection",
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
    quantize: bool = Field(
        default=True,
        description="Apply dynamic int8 quantization on CPU (~2x faster, negligible accuracy loss)",
    )
    num_threads: int = Field(
        default=0,
        ge=0,
        description="Torch intra-op threads (0 = library default)",
    )
    keyword_safety_net: bool = Field(
        default=True,
        description="Flag known slurs/profanity even if the model under-scores them",
    )


class RewriteSettings(BaseModel):
    """Text rewriting configuration."""

    llm_provider: Literal["ollama", "openai", "local"] = Field(
        default="ollama",
        description="LLM backend: ollama, openai (any OpenAI-compatible server), or local (rules only)",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible base URL (OpenAI, LM Studio, vLLM, llama.cpp server …)",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="API key for the OpenAI-compatible backend (prefer the env var)",
    )
    model: str = Field(
        default="qwen2.5:3b",
        description="Model name. For Ollama, falls back to any installed model if missing.",
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
    context_turns: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Previous utterances from the same speaker given to the LLM as context",
    )
    keep_alive: str = Field(
        default="30m",
        description="How long Ollama keeps the model loaded between requests",
    )
    circuit_breaker_failures: int = Field(
        default=3,
        ge=1,
        description="Consecutive LLM failures before the circuit opens (rules-only)",
    )
    circuit_breaker_cooldown_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="How long the circuit stays open before retrying the LLM",
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
    speak_clean: bool = Field(
        default=True,
        description="Standard mode: also speak clean utterances (False = only speak rewrites)",
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
    subscriber_queue_size: int = Field(
        default=256,
        ge=8,
        description="Per-agent inbox size on the event bus (oldest events dropped when full)",
    )
    max_agent_restarts: int = Field(
        default=5,
        ge=0,
        description="Times the supervisor restarts a crashed agent before giving up",
    )
    restart_backoff_seconds: float = Field(
        default=1.0,
        ge=0.1,
        description="Initial restart back-off (doubles each time, capped at 30s)",
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


class ServerSettings(BaseModel):
    """HTTP server: Prometheus metrics, health probes and the moderation API."""

    enabled: bool = Field(
        default=True,
        description="Run the HTTP server",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Bind address (use 0.0.0.0 inside containers)",
    )
    port: int = Field(
        default=9501,
        ge=1024,
        le=65535,
        description="HTTP port",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="If set, /v1/* endpoints require 'Authorization: Bearer <key>'",
    )
    max_text_length: int = Field(
        default=2000,
        ge=1,
        description="Reject moderation requests longer than this",
    )
    rate_limit_per_minute: int = Field(
        default=120,
        ge=0,
        description="Per-client moderation API requests per minute (0 = unlimited)",
    )


class InterceptorSettings(BaseModel):
    """Inline audio censor configuration.

    When enabled, SpeechShield intercepts call audio via a virtual
    audio device (e.g. VB-Cable), holds each utterance until the
    pipeline has judged it, and plays it to the user's real headphones —
    with toxic words bleeped/muted, or the whole utterance replaced by a
    polite TTS rewrite.
    """

    enabled: bool = Field(
        default=False,
        description="Enable inline audio interception mode",
    )
    delay_seconds: float = Field(
        default=0.8,
        ge=0.2,
        le=5.0,
        description="Minimum delay applied to all audio (smooths playback)",
    )
    max_hold_seconds: float = Field(
        default=4.0,
        ge=0.5,
        le=30.0,
        description="How long after an utterance ends to wait for its verdict before fail_mode applies",
    )
    input_device_name: str = Field(
        default="CABLE Output",
        description="Virtual audio input device name (e.g. VB-Cable Output)",
    )
    output_device_name: str = Field(
        default="",
        description="Real audio output device name (empty = system default)",
    )
    censor_mode: Literal["bleep", "mute", "voice_clone", "tts"] = Field(  # pyrefly: ignore
        default="bleep",
        description="bleep/mute = keep the real voice and censor only toxic words; "
        "voice_clone = replace each toxic word with a polite one spoken in the speaker's own (cloned) voice; "
        "tts = replace the whole toxic utterance with the polite rewrite",
    )
    streaming: Literal["auto", "on", "off"] = Field(  # pyrefly: ignore
        default="auto",
        description="Release audio word-by-word as it is confirmed clean, while the speaker is still talking "
        "(auto = on when ASR runs on a GPU). Not used by censor_mode=tts, which needs whole sentences.",
    )
    partial_interval_seconds: float = Field(
        default=0.5,
        ge=0.2,
        le=3.0,
        description="Streaming: how much new audio triggers another transcription pass",
    )
    commit_guard_seconds: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Streaming: never commit words ending this close to the live edge (they may still change)",
    )
    clone_timeout_seconds: float = Field(
        default=2.0,
        ge=0.2,
        le=10.0,
        description="voice_clone: how long a toxic word may wait for its cloned replacement before it is bleeped",
    )
    fail_mode: Literal["closed", "open"] = Field(  # pyrefly: ignore
        default="closed",
        description="If no verdict arrives in time: closed = mute the utterance, open = let it through",
    )
    word_padding_ms: int = Field(
        default=60,
        ge=0,
        le=500,
        description="Extra audio censored either side of a toxic word",
    )
    bleep_frequency_hz: float = Field(
        default=1000.0,
        ge=100.0,
        le=4000.0,
        description="Bleep tone frequency",
    )
    bleep_volume: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Bleep loudness (0-1 of full scale)",
    )


class VoiceCloneSettings(BaseModel):
    """Voice-clone worker (isolated process, its own venv; see VOICE_CLONE.md)."""

    python: str = Field(
        default=".venv-voice/Scripts/python.exe",
        description="Interpreter of the voice-clone venv (relative to the project root)",
    )
    host: str = Field(default="127.0.0.1", description="Worker bind address")
    port: int = Field(default=9502, ge=1024, le=65535, description="Worker port")
    autostart: bool = Field(
        default=True,
        description="Launch the worker automatically when censor_mode=voice_clone",
    )
    model: Literal["turbo", "standard"] = Field(  # pyrefly: ignore
        default="turbo",
        description="Chatterbox variant: turbo (fast) or standard (slower, more expressive)",
    )
    device: str = Field(default="cuda", description="cuda or cpu")
    min_reference_seconds: float = Field(
        default=6.0,
        ge=5.0,
        description="Clean speech needed from a speaker before their voice can be cloned",
    )
    max_reference_seconds: float = Field(
        default=15.0,
        ge=5.0,
        le=60.0,
        description="Reference audio kept per speaker (most recent clean speech)",
    )
    request_timeout_seconds: float = Field(default=8.0, ge=0.5, description="Per-synthesis HTTP timeout")


class DiarizationSettings(BaseModel):
    """Streaming speaker identification."""

    enabled: bool = Field(
        default=False,
        description="Identify speakers (needs pyannote.audio and a HuggingFace token)",
    )
    embedding_model: str = Field(
        default="pyannote/wespeaker-voxceleb-resnet34-LM",
        description="Speaker-embedding model",
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="HuggingFace token for gated pyannote models (prefer HF_TOKEN env var)",
    )
    similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Cosine similarity needed to match an existing speaker",
    )
    max_speakers: int = Field(
        default=8,
        ge=1,
        description="Cap on distinct speakers tracked per session",
    )
    min_segment_seconds: float = Field(
        default=0.8,
        ge=0.1,
        description="Utterances shorter than this reuse the previous speaker",
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
    segmenter: SegmenterSettings = Field(default_factory=SegmenterSettings)
    asr: ASRSettings = Field(default_factory=ASRSettings)
    toxicity: ToxicitySettings = Field(default_factory=ToxicitySettings)
    rewrite: RewriteSettings = Field(default_factory=RewriteSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    interceptor: InterceptorSettings = Field(default_factory=InterceptorSettings)
    diarization: DiarizationSettings = Field(default_factory=DiarizationSettings)
    voice_clone: VoiceCloneSettings = Field(default_factory=VoiceCloneSettings)

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
        with open(default_yaml, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        logger.info(f"Loaded default config from {default_yaml}")

    # Merge environment-specific config
    if env_yaml.exists():
        with open(env_yaml, encoding="utf-8") as f:
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


# Flat env var name -> nested config path
ENV_MAPPING: dict[str, tuple[str, ...]] = {
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
    "SPEECHSHIELD_SILENCE_THRESHOLD": ("audio", "silence_threshold"),
    "SPEECHSHIELD_CAPTURE_ENABLED": ("audio", "capture_enabled"),
    # Segmenter
    "SPEECHSHIELD_SEGMENT_MIN_SILENCE_MS": ("segmenter", "min_silence_ms"),
    "SPEECHSHIELD_SEGMENT_MAX_SECONDS": ("segmenter", "max_segment_seconds"),
    # ASR
    "SPEECHSHIELD_ASR_MODEL_SIZE": ("asr", "model_size"),
    "SPEECHSHIELD_ASR_MODEL_DIR": ("asr", "model_dir"),
    "SPEECHSHIELD_ASR_DEVICE": ("asr", "device"),
    "SPEECHSHIELD_ASR_COMPUTE_TYPE": ("asr", "compute_type"),
    "SPEECHSHIELD_ASR_CPU_THREADS": ("asr", "cpu_threads"),
    "SPEECHSHIELD_ASR_TIMEOUT": ("asr", "timeout_seconds"),
    # Toxicity
    "SPEECHSHIELD_TOXICITY_MODEL": ("toxicity", "model_name"),
    "SPEECHSHIELD_TOXICITY_THRESHOLD": ("toxicity", "threshold"),
    "SPEECHSHIELD_TOXICITY_TIMEOUT": ("toxicity", "timeout_seconds"),
    "SPEECHSHIELD_TOXICITY_QUANTIZE": ("toxicity", "quantize"),
    # Rewrite
    "SPEECHSHIELD_REWRITE_PROVIDER": ("rewrite", "llm_provider"),
    "SPEECHSHIELD_REWRITE_MODEL": ("rewrite", "model"),
    "SPEECHSHIELD_OLLAMA_BASE_URL": ("rewrite", "ollama_base_url"),
    "SPEECHSHIELD_OPENAI_BASE_URL": ("rewrite", "openai_base_url"),
    "SPEECHSHIELD_OPENAI_API_KEY": ("rewrite", "openai_api_key"),
    "SPEECHSHIELD_REWRITE_TIMEOUT": ("rewrite", "timeout_seconds"),
    "SPEECHSHIELD_REWRITE_MAX_RETRIES": ("rewrite", "max_retries"),
    "SPEECHSHIELD_REWRITE_FALLBACK": ("rewrite", "fallback_enabled"),
    "SPEECHSHIELD_REWRITE_CONTEXT_TURNS": ("rewrite", "context_turns"),
    # Voice
    "SPEECHSHIELD_TTS_PROVIDER": ("voice", "provider"),
    "SPEECHSHIELD_TTS_VOICE": ("voice", "voice"),
    "SPEECHSHIELD_TTS_TIMEOUT": ("voice", "timeout_seconds"),
    "SPEECHSHIELD_PLAY_AUDIO": ("voice", "play_audio"),
    "SPEECHSHIELD_SPEAK_CLEAN": ("voice", "speak_clean"),
    "SPEECHSHIELD_VIRTUAL_DEVICE": ("voice", "virtual_device_support"),
    # Pipeline
    "SPEECHSHIELD_LATENCY_BUDGET": ("pipeline", "latency_budget_ms"),
    "SPEECHSHIELD_STRICTNESS": ("pipeline", "default_rewrite_strictness"),
    "SPEECHSHIELD_LATENCY_WINDOW": ("pipeline", "latency_window_size"),
    "SPEECHSHIELD_SHUTDOWN_TIMEOUT": ("pipeline", "graceful_shutdown_timeout"),
    "SPEECHSHIELD_MAX_AGENT_RESTARTS": ("pipeline", "max_agent_restarts"),
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
    # Server
    "SPEECHSHIELD_SERVER_ENABLED": ("server", "enabled"),
    "SPEECHSHIELD_SERVER_HOST": ("server", "host"),
    "SPEECHSHIELD_SERVER_PORT": ("server", "port"),
    "SPEECHSHIELD_API_KEY": ("server", "api_key"),
    "SPEECHSHIELD_RATE_LIMIT": ("server", "rate_limit_per_minute"),
    # Interceptor
    "SPEECHSHIELD_INTERCEPTOR_ENABLED": ("interceptor", "enabled"),
    "SPEECHSHIELD_INTERCEPTOR_DELAY": ("interceptor", "delay_seconds"),
    "SPEECHSHIELD_INTERCEPTOR_MAX_HOLD": ("interceptor", "max_hold_seconds"),
    "SPEECHSHIELD_INTERCEPTOR_INPUT_DEVICE": ("interceptor", "input_device_name"),
    "SPEECHSHIELD_INTERCEPTOR_OUTPUT_DEVICE": ("interceptor", "output_device_name"),
    "SPEECHSHIELD_CENSOR_MODE": ("interceptor", "censor_mode"),
    "SPEECHSHIELD_FAIL_MODE": ("interceptor", "fail_mode"),
    "SPEECHSHIELD_STREAMING": ("interceptor", "streaming"),
    "SPEECHSHIELD_PARTIAL_INTERVAL": ("interceptor", "partial_interval_seconds"),
    "SPEECHSHIELD_COMMIT_GUARD": ("interceptor", "commit_guard_seconds"),
    "SPEECHSHIELD_CLONE_TIMEOUT": ("interceptor", "clone_timeout_seconds"),
    # Voice clone worker
    "SPEECHSHIELD_VOICE_PYTHON": ("voice_clone", "python"),
    "SPEECHSHIELD_VOICE_PORT": ("voice_clone", "port"),
    "SPEECHSHIELD_VOICE_DEVICE": ("voice_clone", "device"),
    "SPEECHSHIELD_VOICE_MODEL": ("voice_clone", "model"),
    # Diarization
    "SPEECHSHIELD_DIARIZATION_ENABLED": ("diarization", "enabled"),
    "SPEECHSHIELD_DIARIZATION_THRESHOLD": ("diarization", "similarity_threshold"),
    "HF_TOKEN": ("diarization", "hf_token"),
}


def _apply_env_overrides(config_data: dict) -> None:
    """
    Apply SPEECHSHIELD_* environment variables to config_data.

    Maps flat env var names to nested config paths.
    E.g., SPEECHSHIELD_SAMPLE_RATE -> config_data['audio']['sample_rate']
    """
    for env_var, path in ENV_MAPPING.items():
        value = os.getenv(env_var)
        if value is not None:
            # Ensure nested dicts exist
            current = config_data
            for key in path[:-1]:
                if key not in current or not isinstance(current[key], dict):
                    current[key] = {}
                current = current[key]

            current[path[-1]] = _parse_env_value(value)


def _parse_env_value(value: str) -> Optional[str]:
    """Normalise an environment variable value.

    Only null-like values are converted here; everything else stays a
    string and Pydantic coerces it to the field's declared type. Guessing
    types here would turn e.g. an API key of "1234" into an int, or a
    device index of "0" into False.
    """
    if value.strip().lower() in ("null", "none", ""):
        return None
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
