"""
Load test for the moderation API.

Fires a mix of clean and toxic messages at POST /v1/moderate with bounded
concurrency and reports throughput, latency percentiles and status codes.

Usage (with SpeechShield running, e.g. `python main.py --mode api`):
    python scripts/loadtest.py --requests 200 --concurrency 8
    python scripts/loadtest.py --url http://host:9501 --api-key KEY
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time
from collections import Counter

import httpx

MESSAGES = [
    "gg everyone, that was close",
    "can someone cover mid please",
    "nice shot, I killed the boss",
    "you are a fucking idiot",
    "shut up and stop feeding noob",
    "this game is complete bullshit",
    "go kill yourself loser",
    "thanks for the carry",
    "why would you push alone, you moron",
    "let's regroup at the bridge",
]


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))] if s else 0.0


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:9501")
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--api-key")
    ap.add_argument("--speakers", type=int, default=5, help="Distinct speaker IDs to spread requests over")
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    latencies: list[float] = []
    statuses: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    sem = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=60) as client:
        async def one(i: int) -> None:
            body = {"text": random.choice(MESSAGES), "speaker": f"user{i % args.speakers}"}
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post("/v1/moderate", json=body)
                    statuses[str(r.status_code)] += 1
                    if r.status_code == 200:
                        latencies.append((time.perf_counter() - t0) * 1000)
                        methods[r.json().get("rewrite_method", "?")] += 1
                except httpx.HTTPError as e:
                    statuses[type(e).__name__] += 1

        start = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(args.requests)))
        wall = time.perf_counter() - start

    print(f"\n{args.requests} requests, concurrency {args.concurrency}, {wall:.1f}s wall")
    print(f"  throughput   {args.requests / wall:6.1f} req/s")
    if latencies:
        print(f"  latency      p50 {pct(latencies, 50):.0f} ms   p95 {pct(latencies, 95):.0f} ms   "
              f"p99 {pct(latencies, 99):.0f} ms   mean {statistics.mean(latencies):.0f} ms")
    print(f"  status codes {dict(statuses)}")
    print(f"  rewrites     {dict(methods)}\n")


if __name__ == "__main__":
    asyncio.run(main())
