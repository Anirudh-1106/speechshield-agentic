<<<<<<< HEAD
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
=======
# 🛡️ SpeechShield

A multi-agent real-time voice moderation system built with Python.

It captures live microphone audio, transcribes speech, detects toxic language, rewrites offensive segments into polite alternatives, and synthesizes cleaned audio output — all continuously in real time.

---

## 🏗️ Architecture


The system consists of 6 independent agents communicating exclusively through an asynchronous event bus.
No agent directly calls another.
Agents emit events, subscribe to events, and maintain isolated local state.
This ensures loose coupling, scalability, and fault isolation.

🤖 Agent Overview
| Agent             | Listens For        | Emits                   | Model Used                      |
| ----------------- | ------------------ | ----------------------- | ------------------------------- |
| AudioAgent        | Hardware mic input | audio_chunk             | None                            |
| ASRAgent          | audio_chunk        | text_transcript         | Faster Whisper (small)          |
| ToxicityAgent     | text_transcript    | toxicity_result         | toxic-bert + keyword safety net |
| RewriteAgent      | toxicity_result    | rewrite_result          | Ollama LLM / rule engine        |
| VoiceAgent        | rewrite_result     | audio_output_ready      | Edge Neural TTS                 |
| OrchestratorAgent | All events         | Policy enforcement only | None                            |

🧱 Tech Stack
| Layer              | Technology                                      |
| ------------------ | ----------------------------------------------- |
| Language           | Python 3.12                                     |
| Concurrency        | asyncio (event-driven pub/sub)                  |
| Audio Capture      | PyAudio (16kHz mono, non-blocking)              |
| Speech to Text     | Faster Whisper (CTranslate2, CPU optimized)     |
| Toxicity Detection | HuggingFace Transformers – unitary/toxic-bert   |
| Inference Backend  | PyTorch (CPU)                                   |
| Text Rewrite       | Ollama (gemma3:1b) + rule-based fallback        |
| Text to Speech     | Edge TTS (Microsoft Azure Neural Voices)        |
| Dashboard          | Streamlit                                       |
| Testing            | pytest + pytest-asyncio (81 tests)              |

---

## ⚙️ Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
📦 First Run

Initial launch downloads required models:
Whisper small (~461 MB)
toxic-bert (~440 MB)
Subsequent runs use cached models.

🧠 LLM Rewrite (Optional)
Enable context-aware rewrites using a local LLM:
1️⃣ Install Ollama
https://ollama.com/download

2️⃣ Pull model:
ollama pull gemma3:1b
SpeechShield auto-detects Ollama.
If available → LLM rewriting
If unavailable → rule-based fallback (zero downtime)

▶️ Run the System
# Terminal 1 — Start pipeline
python main.py
# Terminal 2 — Start dashboard
streamlit run dashboard.py
Open:
http://localhost:8501
Dashboard displays:
Live transcripts
Toxicity scores
Rewrite history
Speaker statistics
Stop with:
Ctrl + C

🧪 Testing
# Fast tests (no model loading)
pytest -m "not slow"
# Full test suite
pytest
⚡ How It Works
1️⃣ AudioAgent
Captures microphone input in 1024-frame chunks @ 16kHz and publishes raw audio into the event bus.

2️⃣ ASRAgent
Buffers ~2 seconds of audio, runs Faster Whisper transcription on CPU, and emits text.

3️⃣ ToxicityAgent
Runs toxic-bert classification (sigmoid across 6 labels) with a keyword safety net for profanity edge cases.

4️⃣ RewriteAgent
Attempts LLM rewrite via Ollama.
Fallback: regex-driven substitutions preserving structure and grammar.

5️⃣ VoiceAgent
Synthesizes rewritten text using Edge TTS and plays the cleaned output audio.

6️⃣ OrchestratorAgent
Acts as system supervisor:
Tracks speaker toxicity trends
Adjusts rewrite strictness dynamically
Enforces a 500ms latency budget
---

## 🧩 Design Patterns
| Pattern             | Application                                   |
| ------------------- | --------------------------------------------- |
| Publish / Subscribe | EventBus decouples agent communication        |
| Pipeline            | Sequential processing from audio to output    |
| Supervisor          | Orchestrator monitors health & policy         |
| Strategy            | Swappable rewrite backends (LLM vs rules)     |
| Shared Blackboard   | GlobalState stores metrics & speaker profiles |

---
## 📂 Project Structure

```text
speechshield-agentic/
├── agents/
│   ├── audio_agent.py          # Mic capture
│   ├── asr_agent.py            # Whisper transcription
│   ├── toxicity_agent.py       # Toxicity detection
│   ├── rewrite_agent.py        # Rewrite orchestration
│   ├── llm_rewriter.py         # Ollama client
│   ├── voice_agent.py          # TTS synthesis
│   └── orchestrator.py         # System supervisor
│
├── core/
│   ├── event_bus.py            # Async pub/sub broker
│   └── state.py                # Shared state & profiles
│
├── tests/                      # 81 pytest tests
├── audio/                      # Generated TTS output
├── logs/                       # Runtime logs
├── dashboard.py                # Streamlit dashboard
├── main.py                     # Entry point
└── requirements.txt
```

## 🛡️ Ethical Constraints

SpeechShield enforces the following guardrails:

- No audio is stored permanently (TTS output is ephemeral)
- No voice cloning or impersonation
- Rewrites preserve original intent
- Only the speaker’s own words are modified

>>>>>>> 5c3532f6f784271d372b2153b9659f6c4aab0514
