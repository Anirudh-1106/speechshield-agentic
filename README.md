# SpeechShield v2 — Real-Time Multi-Agent Voice Moderation

A production-grade Python system for real-time voice moderation using async agents, LLM-based rewriting, and toxicity detection.

## Quick Start

### Prerequisites

- Python 3.12+
- pip/poetry
- Ollama (with `gemma:1b` model)
- PyAudio (for audio I/O)

### Installation

```bash
git clone <repo>
cd speechshield-agentic
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Running

```bash
# Development (relaxed thresholds, DEBUG logging)
python main.py

# Production (strict thresholds, optimized latency)
SPEECHSHIELD_ENV=prod python main.py

# Custom configuration
SPEECHSHIELD_TOXICITY_THRESHOLD=0.7 SPEECHSHIELD_LOG_LEVEL=DEBUG python main.py
```

### Dashboard

Open in another terminal:

```bash
streamlit run dashboard.py
```

Navigate to http://localhost:8501

## Configuration

SpeechShield v2 is **fully configurable** via:

- **YAML files** (`config/default.yaml`, `config/dev.yaml`, `config/prod.yaml`)
- **Environment variables** (`SPEECHSHIELD_*`)
- **Type-safe validation** (Pydantic v2)

See [CONFIG.md](CONFIG.md) for complete reference.

## Architecture

### Agents

- **AudioAgent** — Captures audio from microphone
- **ASRAgent** — Speech-to-text via Faster-Whisper
- **ToxicityAgent** — Detects toxic speech with toxic-bert
- **RewriteAgent** — LLM-based rewriting via Ollama
- **VoiceAgent** — Text-to-speech via Edge TTS
- **OrchestratorAgent** — System supervision and policy enforcement

### Design Principles

✅ **Event-driven** — All communication via async pub/sub EventBus  
✅ **Modular** — Agents are independent and testable  
✅ **Async-first** — Non-blocking pipeline for low latency  
✅ **Graceful** — Handles failures with fallbacks  
✅ **Observable** — Logging, metrics, health checks (Phase 2)  

## Roadmap

- **Phase 1** ✅ Configuration System
- **Phase 2** — Structured Logging & Observability
- **Phase 3** — Real-Time Streaming Pipeline (Virtual Mics)
- **Phase 4** — Latency Optimization
- **Phase 5** — Safe LLM Rewriting Improvements
- **Phase 6** — Speaker Diarization
- **Phase 7** — CI/CD + Quality Gates
- **Phase 8** — Dockerization
- **Phase 9** — Advanced Dashboard
- **Phase 10** — Production Hardening

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_config.py

# Verbose output
pytest -v

# Coverage
pytest --cov=. tests/
```

## Documentation

- [CONFIG.md](CONFIG.md) — Complete configuration reference
- [ROADMAP.md](ROADMAP.md) — Engineering roadmap (coming in Phase 2)
- Code docstrings — All modules have comprehensive docstrings

## Development

### Adding a New Configuration Option

1. Add field to appropriate settings class in `config/settings.py`
2. Update `config/default.yaml`
3. Update `.env.example`
4. Add tests in `tests/test_config.py`
5. Document in `CONFIG.md`

### Local Development with .env

```bash
cp .env.example .env
# Edit .env with your values
python -m dotenv-cli -f .env run python main.py
```

## Performance

**Latency targets** (configurable):

- Development: 1000 ms
- Production: 300 ms
- Test/CI: 5000 ms (relaxed)

See [CONFIG.md](CONFIG.md) for tuning options.

## License

MIT (or your license)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) (coming in Phase 7)

---

**Built with Phase 1 of the SpeechShield v2 Engineering Roadmap**
