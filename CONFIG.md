# Configuration System — SpeechShield v2

## Overview

SpeechShield v2 uses a **centralized, environment-aware configuration system** built on:

- **Pydantic v2** for type-safe validation
- **YAML files** for readable config management
- **Environment variables** for 12-factor app compliance
- **Environment profiles** (dev/test/prod) for workflow-specific settings

## Quick Start

### Default Behavior (Development)

```bash
python main.py
```

Loads `config/dev.yaml` (relaxed thresholds, DEBUG logging).

### Production

```bash
export SPEECHSHIELD_ENV=prod
python main.py
```

Loads `config/prod.yaml` (strict thresholds, optimized latency).

### Environment Variable Overrides

```bash
export SPEECHSHIELD_TOXICITY_THRESHOLD=0.8
export SPEECHSHIELD_LATENCY_BUDGET=250
python main.py
```

Environment variables override YAML files.

## Configuration Structure

### Audio Settings

```yaml
audio:
  sample_rate: 16000           # Hz
  channels: 1                   # Mono
  chunk_size: 1024              # Samples per chunk
  input_device: null            # null = default device
  output_device: null           # null = default device
  queue_size: 200               # Max buffered chunks
```

**Environment Variables:**
- `SPEECHSHIELD_SAMPLE_RATE=48000`
- `SPEECHSHIELD_CHANNELS=2`
- `SPEECHSHIELD_CHUNK_SIZE=2048`
- `SPEECHSHIELD_INPUT_DEVICE=1`
- `SPEECHSHIELD_AUDIO_QUEUE_SIZE=300`

### ASR (Speech Recognition) Settings

```yaml
asr:
  model_size: small             # tiny, base, small, medium, large
  model_dir: models/            # Cache directory
  timeout_seconds: 30.0         # Max inference time
```

**Environment Variables:**
- `SPEECHSHIELD_ASR_MODEL_SIZE=base`
- `SPEECHSHIELD_ASR_TIMEOUT=60`

### Toxicity Detection Settings

```yaml
toxicity:
  model_name: models/toxic-bert
  threshold: 0.5                # 0-1 scale
  timeout_seconds: 5.0
```

**Environment Variables:**
- `SPEECHSHIELD_TOXICITY_THRESHOLD=0.6`
- `SPEECHSHIELD_TOXICITY_TIMEOUT=10`

### LLM Rewriting Settings

```yaml
rewrite:
  llm_provider: ollama          # ollama, openai, local
  ollama_base_url: http://localhost:11434
  model: gemma:1b
  timeout_seconds: 10.0
  max_retries: 2                # Retry failed rewrites
  fallback_enabled: true        # Fall back to rules if LLM fails
```

**Environment Variables:**
- `SPEECHSHIELD_REWRITE_PROVIDER=openai`
- `SPEECHSHIELD_REWRITE_MODEL=gpt-4-turbo`
- `SPEECHSHIELD_OLLAMA_BASE_URL=http://localhost:8000`
- `SPEECHSHIELD_REWRITE_TIMEOUT=5`
- `SPEECHSHIELD_REWRITE_MAX_RETRIES=1`

### Text-to-Speech Settings

```yaml
voice:
  provider: edge_tts            # edge_tts, espeak, pyttsx3
  voice: en-US-GuyNeural        # Edge TTS voice ID
  timeout_seconds: 15.0
  play_audio: true              # Enable playback
  virtual_device_support: false # VB-Cable, etc (Phase 3)
```

**Environment Variables:**
- `SPEECHSHIELD_TTS_VOICE=en-US-AriaNeural`
- `SPEECHSHIELD_PLAY_AUDIO=false`
- `SPEECHSHIELD_VIRTUAL_DEVICE=true`

### Pipeline Settings

```yaml
pipeline:
  latency_budget_ms: 500.0          # Target latency
  default_rewrite_strictness: 1.0   # 1.0 = normal
  latency_window_size: 100          # Tracking window
  graceful_shutdown_timeout: 10.0
```

**Environment Variables:**
- `SPEECHSHIELD_LATENCY_BUDGET=300`
- `SPEECHSHIELD_STRICTNESS=1.2`

### Logging Settings

```yaml
logging:
  level: INFO                   # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_dir: logs/
  format: text                  # text or json (Phase 2)
  enable_file_logging: true
```

**Environment Variables:**
- `SPEECHSHIELD_LOG_LEVEL=DEBUG`
- `SPEECHSHIELD_LOG_FORMAT=json`

### Dashboard Settings

```yaml
dashboard:
  enabled: true
  port: 8501
  refresh_interval: 1.0         # Seconds
  max_rows: 500                 # Max feed items
```

**Environment Variables:**
- `SPEECHSHIELD_DASHBOARD_ENABLED=false`
- `SPEECHSHIELD_DASHBOARD_PORT=8080`

## Configuration Priority

Settings are loaded in this priority order (highest to lowest):

1. **Environment Variables** (`SPEECHSHIELD_*`)
2. **Environment-Specific YAML** (`config/{ENV}.yaml`)
3. **Default YAML** (`config/default.yaml`)
4. **Pydantic Defaults** (hard-coded in `config/settings.py`)

### Example Resolution

If you set:
```bash
export SPEECHSHIELD_TOXICITY_THRESHOLD=0.8
```

With `config/dev.yaml` inheriting from `config/default.yaml`:

```
default.yaml → toxicity.threshold = 0.5
dev.yaml → toxicity.threshold = 0.5 (inherited)
env var → SPEECHSHIELD_TOXICITY_THRESHOLD=0.8 (WINS)
```

Final value: **0.8**

## Environment Profiles

### Development (`dev`)

- **Purpose:** Local development with fast feedback
- **Key Settings:**
  - `debug: true`
  - `logging.level: DEBUG`
  - `pipeline.latency_budget_ms: 1000.0` (relaxed)
  - `dashboard.refresh_interval: 0.5` (responsive)

### Testing (`test`)

- **Purpose:** CI/CD and automated testing
- **Key Settings:**
  - `debug: false`
  - `logging.level: WARNING`
  - `dashboard.enabled: false`
  - `pipeline.latency_budget_ms: 5000.0` (very relaxed for CI)

### Production (`prod`)

- **Purpose:** Real-time voice moderation in production
- **Key Settings:**
  - `debug: false`
  - `logging.level: WARNING`
  - `logging.format: json` (structured logging)
  - `asr.model_size: base` (faster)
  - `pipeline.latency_budget_ms: 300.0` (strict)
  - `toxicity.threshold: 0.6` (higher, fewer false positives)

## Using Configuration in Code

### Loading Settings

```python
from config.settings import settings

config = settings()  # Lazy-loaded singleton

print(config.env)
print(config.audio.sample_rate)
print(config.pipeline.latency_budget_ms)
```

### Passing to GlobalState

```python
from config.settings import settings
from core.state import GlobalState

config = settings()
state = GlobalState(settings=config)

# Now state.config has all the values
print(state.config["latency_budget_ms"])
```

### Accessing in Agents

Agents receive `state` which contains settings:

```python
class MyAgent:
    def __init__(self, bus: EventBus, state: GlobalState):
        self.state = state
        self.timeout = state.settings.asr.timeout_seconds
        self.threshold = state.settings.toxicity.threshold
```

### Testing with Different Configs

```python
from config.settings import load_config

# Load test config
config = load_config(env="test")
assert config.dashboard.enabled is False

# Load prod config
config = load_config(env="prod")
assert config.pipeline.latency_budget_ms == 300.0
```

## Creating Custom Configs

### Example: Custom Staging Config

Create `config/staging.yaml`:

```yaml
env: staging
debug: false

audio:
  chunk_size: 1024

asr:
  model_size: medium
  timeout_seconds: 20.0

toxicity:
  threshold: 0.55

pipeline:
  latency_budget_ms: 400.0

logging:
  level: INFO
  format: json
```

Then run:

```bash
export SPEECHSHIELD_ENV=staging
python main.py
```

## Validation

Pydantic automatically validates all settings:

```python
# This will raise ValueError
export SPEECHSHIELD_TOXICITY_THRESHOLD=1.5  # Must be 0-1
python main.py

# This will raise ValueError
export SPEECHSHIELD_LATENCY_BUDGET=50  # Must be >= 100
python main.py
```

## Debugging Configuration

### Print Current Config

```bash
python -c "from config.settings import settings; import json; print(json.dumps(settings().to_dict(), indent=2))"
```

### Verbose Config Loading

```bash
export SPEECHSHIELD_LOG_LEVEL=DEBUG
python main.py
```

Look for log lines like:
```
Loading config for environment: dev
Loaded default config from .../config/default.yaml
Merged dev config from .../config/dev.yaml
Configuration loaded: env=dev debug=True
```

### Check Environment Variable Overrides

```bash
export SPEECHSHIELD_TOXICITY_THRESHOLD=0.8
export SPEECHSHIELD_LOG_LEVEL=DEBUG
python -c "
from config.settings import settings
cfg = settings()
print(f'Threshold: {cfg.toxicity.threshold}')
print(f'Log Level: {cfg.logging.level}')
"
```

## Migration from Hardcoded Constants

### Before (Phase 0)

```python
# Scattered hardcoded constants
TOXICITY_THRESHOLD = 0.5
LATENCY_BUDGET_MS = 500.0
SAMPLE_RATE = 16000
VOICE = "en-US-GuyNeural"
```

### After (Phase 1)

```python
from config.settings import settings

config = settings()
print(config.toxicity.threshold)
print(config.pipeline.latency_budget_ms)
print(config.audio.sample_rate)
print(config.voice.voice)
```

All values are now:
- **Configurable** via YAML or env vars
- **Environment-specific** (dev/prod differences)
- **Type-safe** (Pydantic validation)
- **Documented** (inline docstrings)

## Next Steps

**Phase 2** will add structured logging and metrics that leverage this configuration:

```python
from config.settings import settings

config = settings()
logger = setup_structured_logging(
    level=config.logging.level,
    format=config.logging.format,
)
```

The configuration system is now the foundation for all future phases.
