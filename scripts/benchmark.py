"""
End-to-end latency benchmark on real speech.

Synthesises a fixed set of clean and toxic utterances with Edge TTS (cached
after the first run), streams them through the real pipeline components —
segmenter → Faster-Whisper → toxic-bert → CensorEngine — and reports:

  • ASR / toxicity / verdict latency (p50, p95) per ASR model
  • detection accuracy on the labelled utterances
  • which words the interceptor would bleep

Usage:
    python scripts/benchmark.py                       # configured ASR model
    python scripts/benchmark.py --asr tiny.en base.en small
    python scripts/benchmark.py --no-quantize         # compare fp32 toxicity
    python scripts/benchmark.py --llm                 # also time LLM rewrites
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

CASES: list[tuple[str, bool]] = [
    ("Hey, can you cover the left side while I reload?", False),
    ("Nice shot, I think we can win this round.", False),
    ("I killed the boss on my second try.", False),
    ("You are a fucking idiot, you can't even aim.", True),
    ("Shut up and stop feeding, you stupid noob.", True),
    ("This is bullshit, the whole team is trash.", True),
    ("Go kill yourself, nobody wants you here.", True),
    ("Thanks for the help, good game everyone.", False),
]
SR = 16000
CACHE = ROOT / "models" / "bench_audio"


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


async def synth(text: str, voice: str) -> bytes:
    from agents.voice_agent import VoiceAgent
    import edge_tts

    mp3 = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            mp3.extend(chunk["data"])
    return VoiceAgent._mp3_to_pcm(bytes(mp3), SR)


async def load_audio(voice: str) -> list[bytes]:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = []
    for i, (text, _) in enumerate(CASES):
        f = CACHE / f"case_{i}_{voice}.pcm"
        if not f.exists():
            print(f"  synthesising: {text}")
            f.write_bytes(await synth(text, voice))
        out.append(f.read_bytes())
    return out


def stream_segments(cfg, pcm: bytes) -> list:
    """Run one utterance (padded with silence) through the real segmenter."""
    from core.segmenter import UtteranceSegmenter

    seg = UtteranceSegmenter(cfg.segmenter, sample_rate=SR, silence_threshold=cfg.audio.silence_threshold)
    pad = np.zeros(int(0.6 * SR), dtype=np.int16).tobytes()
    audio = pad + pcm + pad
    n = cfg.audio.chunk_size * 2
    closed = []
    for i in range(0, len(audio) - n + 1, n):
        r = seg.process(audio[i:i + n], i // n, i / 2 / SR)
        if r.closed:
            closed.append(r.closed)
    r = seg.flush()
    if r.closed:
        closed.append(r.closed)
    return closed


async def run_for_model(cfg, audio: list[bytes], asr_model: str, use_llm: bool) -> dict:
    from agents.asr_agent import ASRAgent
    from agents.llm_rewriter import LLMRewriter
    from agents.toxicity_agent import ToxicityAgent
    from core.event_bus import EventBus
    from core.lexicon import toxic_word_spans
    from core.state import GlobalState

    cfg = cfg.model_copy(update={"asr": cfg.asr.model_copy(update={"model_size": asr_model})})
    state = GlobalState(settings=cfg)
    asr = ASRAgent(EventBus(), state)
    tox = ToxicityAgent(EventBus(), state)
    t0 = time.time()
    asr._load_model()
    tox._load_model()
    load_s = time.time() - t0

    llm = LLMRewriter(cfg.rewrite) if use_llm else None
    if llm:
        llm.verifier = tox.score
        await llm.warmup()

    rows, asr_ms, tox_ms, verdict_ms, llm_ms = [], [], [], [], []
    correct = 0
    for (_text, label), pcm in zip(CASES, audio, strict=True):
        segments = stream_segments(cfg, pcm)
        transcript, words, toxic, spans = [], [], False, []
        for seg in segments:
            a0 = time.perf_counter()
            seg_text, seg_words = asr._transcribe(seg.audio)
            a1 = time.perf_counter()
            res = tox._classify(seg_text) if seg_text else {"toxic": False, "confidence": 0.0}
            a2 = time.perf_counter()
            asr_ms.append((a1 - a0) * 1000)
            tox_ms.append((a2 - a1) * 1000)
            verdict_ms.append((a2 - a0) * 1000)
            transcript.append(seg_text)
            words += seg_words
            if res["toxic"]:
                toxic = True
                spans += [s.text for s in toxic_word_spans(seg_words, utterance_toxic=True)]
        heard = " ".join(transcript)
        correct += toxic == label
        rewrite = ""
        if llm and toxic:
            l0 = time.perf_counter()
            outcome = await llm.rewrite(heard)
            llm_ms.append((time.perf_counter() - l0) * 1000)
            rewrite = outcome.text or f"[rules: {outcome.reason}]"
        rows.append((label, toxic, heard, spans, rewrite, len(segments)))
    if llm:
        await llm.close()
    return {
        "model": asr_model, "load_s": load_s, "rows": rows, "accuracy": correct / len(CASES),
        "asr": asr_ms, "tox": tox_ms, "verdict": verdict_ms, "llm": llm_ms,
        "asr_info": state.system_info.get("asr"), "tox_info": state.system_info.get("toxicity"),
    }


def report(r: dict) -> None:
    print(f"\n=== ASR {r['asr_info']} | toxicity {r['tox_info']} | load {r['load_s']:.1f}s ===")
    for label, toxic, heard, spans, rewrite, nseg in r["rows"]:
        mark = "OK " if label == toxic else "ERR"
        tag = "TOXIC" if toxic else "clean"
        print(f"  [{mark}] {tag:5} ({nseg} seg) heard: {heard!r}")
        if spans:
            print(f"               bleep: {spans}")
        if rewrite:
            print(f"               rewrite: {rewrite!r}")
    def line(name: str, xs: list[float]) -> str:
        return f"  {name:<22} p50 {pct(xs, 50):7.0f} ms   p95 {pct(xs, 95):7.0f} ms   mean {statistics.mean(xs):7.0f} ms" if xs else ""
    print(f"\n  detection accuracy: {r['accuracy']:.0%}")
    print(line("ASR", r["asr"]))
    print(line("toxicity", r["tox"]))
    print(line("verdict (ASR+tox)", r["verdict"]))
    if r["llm"]:
        print(line("LLM rewrite", r["llm"]))


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr", nargs="+", help="ASR model sizes to compare")
    ap.add_argument("--no-quantize", action="store_true", help="Run toxicity in fp32")
    ap.add_argument("--llm", action="store_true", help="Also benchmark LLM rewrites")
    ap.add_argument("--voice", default="en-US-GuyNeural")
    args = ap.parse_args()

    os.environ.setdefault("SPEECHSHIELD_LOG_LEVEL", "WARNING")
    from config.settings import load_config
    from core.logging import configure_logging

    cfg = load_config()
    configure_logging(level="WARNING", enable_file_logging=False)
    if args.no_quantize:
        cfg = cfg.model_copy(update={"toxicity": cfg.toxicity.model_copy(update={"quantize": False})})

    print("Preparing benchmark audio …")
    audio = await load_audio(args.voice)
    total = sum(len(a) for a in audio) / 2 / SR
    print(f"{len(audio)} utterances, {total:.1f}s of speech, chunk={cfg.audio.chunk_size}")
    for model in args.asr or [cfg.asr.model_size]:
        report(await run_for_model(cfg, audio, model, args.llm))


if __name__ == "__main__":
    asyncio.run(main())
