# 🛡️ SpeechShield

**A real-time AI filter for voice chat.** SpeechShield sits between your call
app (Discord, Zoom, Teams, games) and your headphones. It listens to
incoming voices, finds toxic words, and censors them *before you hear them*,
while the rest of what people say comes through in their own voice.

What makes it different:

- **Word-level, not sentence-level.** Only the toxic words are touched.
  "You are a f\*\*\*ing idiot, you can't even aim" keeps "You are a … you
  can't even aim" in the speaker's real voice.
- **Voice-clone replacement.** Instead of a bleep, toxic words can be
  replaced by polite ones *spoken in the speaker's own cloned voice*: "…you
  are a **goof**…". See [VOICE_CLONE.md](VOICE_CLONE.md).
- **Streaming.** With a GPU, audio is released word by word as each word is
  confirmed clean, while the person is still talking, instead of waiting for
  the whole sentence.
- **Fail-closed.** Nothing is played before it's judged. If the AI can't keep
  up, audio is muted, never leaked.

Clean speech is untouched; it just arrives about a second late.

---

## Measured performance

From the repo's own tools, on an RTX 3050 Laptop GPU (4 GB) with 7 GB RAM:
`scripts/benchmark.py` (8 labelled real-speech utterances) and
`scripts/stream_sim.py` (feeds them chunk by chunk through the real models
and the real censor engine).

| | Result |
|---|---|
| Detection on the labelled set | **8 / 8** correct; every censored span was exactly the toxic words |
| Speech recognition per utterance | **~0.25 s** (Whisper small, GPU) · ~1.0 s (base.en, CPU) |
| Toxicity check | ~60 ms (toxic-bert, int8, CPU) |
| Listener delay, streaming (GPU) | **p50 1.06 s · p95 1.70 s · max 2.8 s** |
| Listener delay, per-sentence | p50 1.54 s · p95 3.07 s · max 3.9 s |
| Voice-clone replacement word | ~0.7–1.2 s new, **0 ms cached** (vocabulary is pre-synthesized); same listener delay as bleeping |
| Voice-clone output, re-transcribed by Whisper | "You're a **goof**, you can't even aim." · "**Hold on** and stop feeding, you **silly** noob." · "This is **nonsense**. The whole team is **rough**." · "**Take a break.** Nobody wants you here." |
| LLM rewrite (qwen2.5:3b, warm) | 0.15–0.8 s, re-checked by toxic-bert before use |

Reproduce: `python scripts/benchmark.py --llm` and `python scripts/stream_sim.py`.

---

## Quick start (Windows)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-gpu.txt      # NVIDIA GPU: ~12x faster speech recognition (recommended)
python main.py --doctor                   # checks ffmpeg, devices, models, GPU, Ollama
```

Then pick a mode:

```powershell
python main.py --mode interceptor                            # filter a call (bleep toxic words)
python main.py --mode interceptor --censor-mode voice_clone  # …replace them in the speaker's voice
python main.py                                               # demo: speak into your mic, hear the cleaned version
python main.py --mode api                                    # headless text-moderation API
python main.py --text "shut up you stupid idiot"             # one-off check from the terminal
```

- **Interceptor mode** needs the free VB-Cable driver: [VBCABLE_SETUP.md](VBCABLE_SETUP.md) (5 minutes).
- **Voice-clone mode** needs a one-time setup: [VOICE_CLONE.md](VOICE_CLONE.md).
- **LLM rewrites** (optional): install [Ollama](https://ollama.com) and run `ollama pull qwen2.5:3b`.
  Without it, a deterministic rule engine is used.
- **Dashboard**: <http://localhost:8501> (opens with the app).
- **Metrics & health**: <http://127.0.0.1:9501/metrics>, `/health`, `/ready`, `/live`.

Requirements: Python 3.11+, [ffmpeg](https://ffmpeg.org) on PATH. An NVIDIA GPU is
strongly recommended for live calls; CPU works with more delay.

---

## Censor modes

| `--censor-mode` | Toxic words become | Notes |
|---|---|---|
| `bleep` *(default)* | a short tone | Original voice everywhere else |
| `mute` | silence | Original voice everywhere else |
| `voice_clone` | polite words in the speaker's own voice; intensifiers and slurs cut out | Needs a GPU and `.venv-voice`; falls back to bleep per word |
| `tts` | the whole sentence, rewritten politely in a neutral TTS voice | Uses the LLM rewriter |

Toxicity that isn't tied to a word (e.g. a threat phrased in clean words) is
detected by the model and handled by muting or bleeping the utterance.

---

## Moderation API

```bash
curl -s localhost:9501/v1/moderate -H "content-type: application/json" \
     -d '{"text": "shut up you stupid idiot", "speaker": "user42"}'
```

```json
{
  "toxic": true, "confidence": 0.99,
  "labels": {"toxic": 0.99, "insult": 0.97, "obscene": 0.41, "...": "..."},
  "masked": "s*** u* you s***** i****",
  "rewritten": "Please be quiet.",
  "rewrite_method": "llm",
  "latency_ms": 180
}
```

`speaker` makes rewrites stricter for repeat offenders. Optional bearer-token
auth (`SPEECHSHIELD_API_KEY`), per-client rate limiting, and a request-size
limit are built in. Load test: `python scripts/loadtest.py`.

---

## How it works

```
 call audio ─► AudioInterceptor ──speech_partial──► ASRAgent (Whisper, word timestamps)
                 │  segmenter       speech_segment        │  partial_transcript / text_transcript
                 │  (VAD)                                 ▼
                 │                                  ToxicityAgent (toxic-bert + lexicon)
                 │◄──── partial_verdict / toxicity_result ┤
                 │                                        ▼
                 │◄──── clone_ready ── VoiceCloneAgent ◄─ RewriteAgent (LLM + guardrails | rules)
                 │                       │ HTTP                │
                 ▼                       ▼                     ▼
            your headphones      voice worker (.venv-voice)  VoiceAgent (TTS) · Orchestrator · Dashboard
```

- **Event bus.** Every agent has its own bounded inbox, so a slow model never
  blocks the others. Events for one utterance share a correlation ID end to end.
- **Segmenter.** Adaptive energy-based voice detection cuts speech at natural
  pauses (no half-words). A 10th-percentile noise floor ignores steady hum.
- **Streaming.** Whisper re-transcribes the utterance in progress about every
  0.5 s. Words confirmed by two consecutive passes (LocalAgreement) are judged
  and released.
- **Censor engine.** Each utterance has a timeline of edits (bleep, mute, cut,
  splice). Partial verdicts, the final verdict, and late voice clones all add
  edits, which are applied as audio streams out with 5 ms crossfades.
- **Guarded LLM rewriting.** Ollama or any OpenAI-compatible server, with
  few-shot prompting and conversation context. Every rewrite is checked for
  leftover toxic words and refusals, then re-scored by toxic-bert. One
  corrective retry, then rule-based fallback. A circuit breaker skips a
  failing LLM.
- **Supervision.** Crashed agents restart with backoff. Optional components
  (LLM, diarization, voice cloning, TTS) degrade instead of failing.
  `--doctor` diagnoses setup problems.

---

## Configuration

`config/default.yaml` plus `config/{dev,test,prod}.yaml`, overridable with
`SPEECHSHIELD_*` env vars and CLI flags. The full generated reference is
[CONFIG.md](CONFIG.md). The settings that matter most:

| Setting | Default | |
|---|---|---|
| `interceptor.censor_mode` | `bleep` | `bleep` · `mute` · `voice_clone` · `tts` |
| `interceptor.streaming` | `auto` | word-by-word release (auto = on when ASR is on a GPU) |
| `interceptor.fail_mode` | `closed` | mute if no verdict in time (`open` = pass) |
| `asr.model_size` | `auto` | `small` on GPU, `base.en` on CPU |
| `toxicity.threshold` | `0.5` (prod `0.6`) | |
| `rewrite.model` | `qwen2.5:3b` | any installed Ollama model is used if missing |

---

## Docker

```bash
docker compose up -d    # moderation API + Ollama + dashboard
```

Containers can't reach audio devices on Windows or macOS, so the image runs
the text-moderation API and dashboard. Run audio modes natively.

---

## Development

```powershell
pip install -r requirements-dev.txt
pytest -m "not slow"            # 300+ fast tests, no models or GPU needed
pytest                          # + slow tests with real models
ruff check . ; mypy agents core config main.py dashboard.py
python scripts/gen_config_docs.py   # regenerate CONFIG.md after changing settings
pre-commit install
```

CI (GitHub Actions) runs lint, types, a config-docs freshness check, tests on
Python 3.11 and 3.12 with a 75 % coverage gate, and a Docker build.

```
agents/      audio_interceptor (censor engine) · asr · toxicity · rewrite + llm_rewriter
             voice (TTS) · voice_clone · diarization · orchestrator · audio (mic)
core/        event_bus · segmenter · lexicon · state · supervisor · api · metrics_server
             audio_io · gpu · doctor · health · metrics · logging · diarization
voice_service/  isolated voice-clone worker (own venv)
config/      settings (Pydantic) + YAML profiles
scripts/     benchmark · stream_sim · loadtest · gen_config_docs · setup_voice_clone
tests/       unit, agent, pipeline-flow and streaming tests
```

---

## Responsible use

- Only **incoming** audio is filtered, on **your** machine. Nothing is
  recorded or uploaded. TTS and transcripts live in memory; the dashboard
  shows recent history only.
- Voice cloning is opt-in, limited to a fixed polite vocabulary, ephemeral,
  and watermarked. See [VOICE_CLONE.md](VOICE_CLONE.md#responsible-use).
- Toxicity models have known biases (e.g. over-flagging identity terms). Tune
  `toxicity.threshold` and review the dashboard.

## Known limitations

- English only (Whisper `*.en` models and an English lexicon).
- The ~1 s delay is inherent to judging speech before you hear it. It's fine
  for games and casual calls; fast back-and-forth conversation can feel it.
- Context-dependent toxicity (sarcasm, in-jokes) is hard. Expect some false
  positives and negatives.
- On CPU-only machines, streaming is off and delay is ~1.5–3 s per sentence.
