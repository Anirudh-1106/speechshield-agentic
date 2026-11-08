"""
Interceptor latency simulation on real speech with the real models.

Feeds the benchmark utterances through the CensorEngine chunk by chunk on a
simulated clock.  Every Whisper / toxic-bert call is *actually executed*
and timed, and its result is delivered to the engine at the moment it
would really arrive (the GPU/CPU is treated as one sequential worker).

Reports, per mode, how long after being spoken each piece of speech
reaches the listener — the number that decides whether a conversation
feels natural — and which words were censored.

Usage:
    python scripts/stream_sim.py                 # streaming vs utterance-level
    python scripts/stream_sim.py --mode voice_clone --save demo_output
        (voice_clone needs the worker: .venv-voice/Scripts/python.exe voice_service/worker.py)
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

SR = 16000


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))] if s else 0.0


class CloneClient:
    """Talks to the real voice worker exactly as VoiceCloneAgent does."""

    def __init__(self, url: str) -> None:
        import httpx
        self.http = httpx.Client(base_url=url, timeout=60)

    def ready(self) -> bool:
        try:
            return bool(self.http.get("/health").json().get("ready"))
        except Exception:
            return False

    def enroll(self, speaker: str, pcm: bytes) -> dict:
        import base64
        r = self.http.post("/enroll", json={"speaker": speaker, "pcm_b64": base64.b64encode(pcm).decode()})
        r.raise_for_status()
        return r.json()

    def prewarm_sync(self, speaker: str, texts: list[str]) -> int:
        ok = 0
        for text in texts:
            try:
                ok += self.http.post("/synthesize", json={"speaker": speaker, "text": text}).status_code == 200
            except Exception:
                pass  # the live agent treats failures as "bleep this word"
        return ok

    def reply(self, speaker: str, span_text: str):
        import base64
        from core.lexicon import soft_replacement
        rep = soft_replacement(span_text)
        if rep is None:
            return None, rep
        if rep == "":
            return b"", rep
        try:
            r = self.http.post("/synthesize", json={"speaker": speaker, "text": rep})
        except Exception:
            return None, rep
        return (base64.b64decode(r.json()["pcm_b64"]) if r.status_code == 200 else None), rep


def simulate(settings, asr, tox, pcm: bytes, streaming: bool, clone=None, out=None):
    from agents.audio_interceptor import CensorEngine
    from agents.toxicity_agent import judge_partial
    from core.lexicon import toxic_word_spans

    engine = CensorEngine(settings, streaming=streaming)
    chunk_bytes = settings.audio.chunk_size * 2
    pad = np.zeros(int(0.5 * SR), dtype=np.int16).tobytes()
    tail = np.zeros(int(1.5 * SR), dtype=np.int16).tobytes()
    audio = pad + pcm + tail
    chunks = [audio[i:i + chunk_bytes] for i in range(0, len(audio) - chunk_bytes + 1, chunk_bytes)]
    dt = settings.audio.chunk_size / SR

    events: list = []      # (deliver_at, seq, fn)
    order = itertools.count()
    busy_until = 0.0       # single sequential inference worker
    compute_ms: list[float] = []

    def run_job(now, fn):
        nonlocal busy_until
        start = max(now, busy_until)
        t0 = time.perf_counter()
        deliver = fn()
        took = time.perf_counter() - t0
        compute_ms.append(took * 1000)
        busy_until = start + took
        heapq.heappush(events, (busy_until, next(order), deliver))

    def partial_job(cid, seg_audio):
        def fn():
            _, words = asr._transcribe(seg_audio)
            committed = asr._commit(cid, words, len(seg_audio) / 2 / SR - settings.interceptor.commit_guard_seconds)
            res = tox._classify(" ".join(w["word"] for w in committed)) if committed else {"toxic": False}
            # Same decision function the live ToxicityAgent uses.
            safe, spans, meaning = judge_partial(committed, bool(res["toxic"]), committed[-1]["end"] if committed else 0.0)
            return lambda: engine.on_partial(cid, safe, spans, meaning)
        return fn

    def final_job(seg):
        def fn():
            text, words = asr._transcribe(seg.audio)
            if not text:
                return lambda: engine.on_transcript_empty(seg.segment_id)
            res = tox._classify(text)
            spans = [s.to_dict() for s in toxic_word_spans(words, utterance_toxic=res["toxic"])]
            if res["toxic"] and words and not spans:
                spans = [{"start": 0, "end": 0, "whole": True}]
            final.append((text, res["toxic"], [s.get("text") for s in spans]))
            return lambda: engine.on_toxicity(seg.segment_id, res["toxic"], spans)
        return fn

    final: list = []
    swaps: list = []
    played = 0          # samples released so far
    delays: list[float] = []
    capture_times: list[float] = []
    t = 0.0
    speech_start = len(pad) // 2
    speech_end = speech_start + len(pcm) // 2
    for seq, c in enumerate(chunks + [b""] * 400):
        if c:
            capture_times.append(t)
            seg = engine.push(c, seq, t)
            for cid, a in engine.take_partial_requests():
                run_job(t, partial_job(cid, a))
            if seg:
                run_job(t, final_job(seg))
        while events and events[0][0] <= t:
            heapq.heappop(events)[2]()
        for req in engine.take_clone_requests():
            def clone_fn(req=req):
                pcm_rep, rep = clone.reply("default", req.text)
                swaps.append((req.text, rep, "voice" if pcm_rep else ("cut" if pcm_rep == b"" else "bleep")))
                return lambda: engine.on_clone(req.key, pcm_rep)
            run_job(t, clone_fn)
        released, _ = engine.release(t, output_buffered=-1e9)
        if out is not None:
            out.extend(b"".join(released))
        for a in released:
            n = len(a) // 2
            for k in range(played, played + n, settings.audio.chunk_size):
                idx = k // settings.audio.chunk_size
                if idx < len(capture_times) and speech_start <= k < speech_end:
                    delays.append(t - capture_times[idx])
            played += n
        t += dt
        if not c and not events and engine.pending_segments == 0 and played >= len(audio) // 2 - chunk_bytes:
            break
    return delays, final, compute_ms, swaps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="bleep", choices=("bleep", "mute", "voice_clone"))
    ap.add_argument("--save", help="Write original/censored WAVs of each clip to this folder")
    ap.add_argument("--worker", default="http://127.0.0.1:9502")
    args = ap.parse_args()

    os.environ.setdefault("SPEECHSHIELD_LOG_LEVEL", "WARNING")
    from agents.asr_agent import ASRAgent
    from agents.toxicity_agent import ToxicityAgent
    from config.settings import load_config
    from core.event_bus import EventBus
    from core.gpu import register_cuda_libraries
    from core.logging import configure_logging
    from core.state import GlobalState

    register_cuda_libraries()
    configure_logging(level="WARNING", enable_file_logging=False)
    cfg = load_config()
    cfg = cfg.model_copy(update={"interceptor": cfg.interceptor.model_copy(update={"censor_mode": args.mode})})
    state = GlobalState(settings=cfg)
    asr, tox = ASRAgent(EventBus(), state), ToxicityAgent(EventBus(), state)
    asr._load_model()
    tox._load_model()
    print(f"ASR: {state.system_info['asr']} | toxicity: {state.system_info['toxicity']}")

    clips = sorted((ROOT / "models" / "bench_audio").glob("case_*_en-US-GuyNeural.pcm"))
    if not clips:
        sys.exit("Run scripts/benchmark.py once first to synthesise the test clips.")

    clone = None
    if args.mode == "voice_clone":
        from core.lexicon import SOFT_VOCABULARY
        clone = CloneClient(args.worker)
        if not clone.ready():
            sys.exit(f"Voice worker not ready at {args.worker}")
        # Enroll from the clean clips only — the same rule the live agent follows.
        for f in clips:
            text, _ = asr._transcribe(f.read_bytes())
            if not tox._classify(text)["toxic"]:
                info = clone.enroll("default", f.read_bytes())
        print(f"enrolled speaker: {info}")
        t0 = time.time()
        ok = clone.prewarm_sync("default", SOFT_VOCABULARY)
        print(f"pre-synthesised {ok}/{len(SOFT_VOCABULARY)} replacements in {time.time() - t0:.1f}s")

    save = Path(args.save) if args.save else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    for streaming in ((True,) if clone else (False, True)):
        all_delays, compute = [], []
        print(f"\n=== {'STREAMING (word-by-word)' if streaming else 'UTTERANCE-LEVEL'} — censor={args.mode} ===")
        for f in clips:
            asr._hypotheses.clear()
            rendered = bytearray()
            delays, final, ms, swaps = simulate(cfg, asr, tox, f.read_bytes(), streaming, clone, rendered)
            all_delays += delays
            compute += ms
            for text, toxic, spans in final:
                print(f"  {'TOXIC' if toxic else 'clean'}  {text!r}" + (f"  censored={spans}" if spans else ""))
            for src, rep, how in swaps:
                print(f"        {how:6} {src!r}" + (f" -> {rep!r} (in speaker's voice)" if how == "voice" else ""))
            if save and streaming and any(tx for _, tx, _ in final):
                import wave
                for name, data in ((f"{f.stem[:6]}_original.wav", f.read_bytes()), (f"{f.stem[:6]}_{args.mode}.wav", bytes(rendered))):
                    with wave.open(str(save / name), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(SR)
                        wf.writeframes(data)
        print(f"  listener delay  p50 {pct(all_delays, 50):.2f}s  p95 {pct(all_delays, 95):.2f}s  max {max(all_delays):.2f}s")
        print(f"  inference calls {len(compute)}, mean {np.mean(compute):.0f} ms")


if __name__ == "__main__":
    main()
