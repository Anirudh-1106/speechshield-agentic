"""Tests for the bus delivery model: per-subscriber inboxes, no head-of-line
blocking, per-handler ordering, bounded queues, request/response waits."""

import asyncio
import threading

import pytest

from core.event_bus import Event, EventBus


async def test_slow_handler_does_not_block_others() -> None:
    bus = EventBus()
    fast: list[int] = []
    release = asyncio.Event()

    async def slow(event: Event) -> None:
        await release.wait()

    async def quick(event: Event) -> None:
        fast.append(event.data["i"])

    bus.subscribe("x", slow)
    bus.subscribe("x", quick)
    task = asyncio.create_task(bus.start())
    for i in range(5):
        await bus.emit(Event("x", {"i": i}))
    await asyncio.sleep(0.3)
    assert fast == [0, 1, 2, 3, 4]  # delivered while `slow` is still stuck
    release.set()
    bus.stop()
    await task


async def test_per_handler_order_preserved() -> None:
    bus = EventBus()
    seen: list[int] = []

    async def handler(event: Event) -> None:
        await asyncio.sleep(0.001 * (5 - event.data["i"]))  # later events finish faster
        seen.append(event.data["i"])

    bus.subscribe("x", handler)
    task = asyncio.create_task(bus.start())
    for i in range(5):
        await bus.emit(Event("x", {"i": i}))
    await asyncio.sleep(0.3)
    bus.stop()
    await task
    assert seen == [0, 1, 2, 3, 4]


async def test_full_inbox_drops_oldest() -> None:
    bus = EventBus(subscriber_queue_size=8)
    seen: list[int] = []
    gate = asyncio.Event()

    async def handler(event: Event) -> None:
        await gate.wait()
        seen.append(event.data["i"])

    bus.subscribe("x", handler)
    task = asyncio.create_task(bus.start())
    for i in range(30):
        await bus.emit(Event("x", {"i": i}))
    await asyncio.sleep(0.3)
    gate.set()
    await asyncio.sleep(0.2)
    bus.stop()
    await task
    assert seen[-1] == 29       # newest kept
    assert len(seen) <= 10      # bounded: oldest shed
    assert 5 not in seen


async def test_wildcard_receives_everything() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event: Event) -> None:
        seen.append(event.type)

    bus.subscribe("*", handler)
    task = asyncio.create_task(bus.start())
    await bus.emit(Event("a"))
    await bus.emit(Event("b"))
    await asyncio.sleep(0.3)
    bus.stop()
    await task
    assert seen == ["a", "b"]


async def test_wait_for_correlation_id() -> None:
    bus = EventBus()
    task = asyncio.create_task(bus.start())
    waiter = asyncio.create_task(bus.wait_for("done", "abc", timeout=2.0))
    await asyncio.sleep(0.05)
    await bus.emit(Event("done", {"v": 0}, correlation_id="other"))
    await bus.emit(Event("done", {"v": 1}, correlation_id="abc"))
    result = await waiter
    bus.stop()
    await task
    assert result.data["v"] == 1
    assert bus._waiters == {}


async def test_wait_for_timeout_cleans_up() -> None:
    bus = EventBus()
    task = asyncio.create_task(bus.start())
    with pytest.raises(asyncio.TimeoutError):
        await bus.wait_for("never", "x", timeout=0.1)
    bus.stop()
    await task
    assert bus._waiters == {}


async def test_unsubscribe_owner() -> None:
    class Agent:
        def __init__(self) -> None:
            self.seen = 0

        async def on(self, event: Event) -> None:
            self.seen += 1

    bus = EventBus()
    a = Agent()
    bus.subscribe("x", a.on)
    bus.subscribe("y", a.on)
    assert bus.unsubscribe_owner(a) == 2
    task = asyncio.create_task(bus.start())
    await bus.emit(Event("x"))
    await asyncio.sleep(0.2)
    bus.stop()
    await task
    assert a.seen == 0


async def test_emit_threadsafe() -> None:
    bus = EventBus()
    got = asyncio.Event()

    async def handler(event: Event) -> None:
        got.set()

    bus.subscribe("t", handler)
    task = asyncio.create_task(bus.start())
    await asyncio.sleep(0.05)
    threading.Thread(target=bus.emit_threadsafe, args=(Event("t"),)).start()
    await asyncio.wait_for(got.wait(), timeout=2.0)
    bus.stop()
    await task
