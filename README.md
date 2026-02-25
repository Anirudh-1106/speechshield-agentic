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

