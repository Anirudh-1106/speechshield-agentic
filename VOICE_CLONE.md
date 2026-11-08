# Voice-clone censoring

`censor_mode: voice_clone` replaces each toxic word with a polite one **spoken
in the speaker's own voice**, inside their original sentence:

> "You are a f\*\*\*ing idiot, you can't even aim."
> → *"You are a **goof**, you can't even aim."* — in the same person's voice

No bleeps, no robot voice, and no rewriting of the clean parts of what they said.

## How it works

1. **Learning the voice.** Every *clean* utterance a speaker says is added to
   their reference (the most recent ~15 s is kept, in memory only). After about
   6 s of clean speech, the voice can be cloned. Toxic speech is never used as
   a reference.
2. **Pre-synthesis.** As soon as a voice is learned, the whole replacement
   vocabulary (~27 short words and phrases) is synthesized in the background.
   Most replacements later are instant cache hits.
3. **Live replacement.** When a toxic word is confirmed (mid-sentence when
   streaming), the interceptor holds the audio at that word, asks for the
   replacement, and splices it in. The clip is loudness-matched to the word it
   replaces and crossfaded at both edges. Pure intensifiers ("f\*\*\*ing") and
   slurs are simply cut out, with no replacement.
4. **Graceful fallback.** If the voice isn't learned yet, the worker is down, or
   a replacement takes longer than `clone_timeout_seconds` (2 s), that word is
   bleeped instead. Censoring never waits on cloning for long.

| Toxic | Replacement |
|---|---|
| idiot, moron, dumbass | goof |
| stupid, dumb | silly |
| shut up | hold on |
| bullshit | nonsense |
| trash, garbage | rough |
| go kill yourself | take a break |
| f\*\*\*ing, slurs | *(cut out)* |

The full table is `_SOFT_WORDS` / `_SOFT_PHRASES` in `core/lexicon.py`.

## Architecture

The cloning model (Chatterbox Turbo, MIT licence, Resemble AI) runs in a
**separate worker process with its own virtual environment**
(`voice_service/worker.py`, `.venv-voice`):

- Chatterbox pins PyTorch 2.6, which conflicts with the main pipeline.
- A crash or GPU out-of-memory in the worker can't take the censor down.
- The main process starts and stops it automatically and talks to it over
  localhost HTTP (`/enroll`, `/synthesize`, `/prewarm`, `/health`).

### Running on small GPUs (4 GB)

The stock Chatterbox loader peaks at ~5 GB of system RAM and ~3.9 GB of VRAM,
which spills into shared memory on 4 GB laptop GPUs and becomes ~10× slower.
The worker's own loader:

| Change | Effect |
|---|---|
| Builds each module directly on the GPU and streams weights tensor by tensor (no memory-mapping) | Host RAM peak ≈ one tensor; no Windows "paging file" crashes |
| Token model (T3) in fp16, generation under autocast | −0.9 GB VRAM |
| Speech tokenizer (only used when learning a voice) kept on CPU | −0.5 GB VRAM |
| Flow decoder kept in **fp32** | fp16 made words noticeably less intelligible |

Measured on an RTX 3050 Laptop (4 GB): **peak 1.7 GB VRAM, ~1.0–1.3 s per
new word, 0 ms for cached words.** In a Whisper round-trip test, all
replacement words were transcribed back correctly.

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_voice_clone.ps1
python main.py --mode interceptor --censor-mode voice_clone
```

The first run downloads ~4 GB of model weights.

## Responsible use

Voice cloning makes people "say" words they did not say. SpeechShield limits
this:

- **Listener-side only.** It changes what *you* hear on your own machine.
  Nothing is sent back into the call or recorded.
- **Fixed vocabulary.** The clone can only produce the ~27 polite phrases in
  the replacement table. It cannot be made to say arbitrary text, so it is not
  an impersonation tool.
- **Ephemeral.** Reference audio lives only in worker memory and is discarded
  on exit. The temporary file used for conditioning is deleted immediately.
- **Watermarked.** Every clip carries Resemble AI's inaudible Perth watermark,
  so the audio remains detectable as synthetic.
- **Visible.** The dashboard marks every voice-cloned utterance.
- **Your call.** It's opt-in (`bleep` is the default). Some places regulate
  voiceprints (e.g. BIPA, GDPR) and synthetic audio disclosure (EU AI Act), so
  check before deploying it for others.

## Limitations

- Needs a few seconds of clean speech from each person before their words can
  be cloned; until then those words are bleeped.
- With diarization off, everyone on the call shares one learned voice. Enable
  `diarization.enabled` for per-speaker voices on group calls.
- On 8 GB-RAM machines, run with few other heavy apps open. If memory runs out,
  replacements fall back to bleeps.
