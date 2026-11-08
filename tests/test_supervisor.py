"""Tests for agent supervision: restart on crash, give-up policy."""

import asyncio

from core.event_bus import EventBus
from core.supervisor import Supervisor


class Flaky:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.starts = 0
        self.stopped = False

    async def start(self) -> None:
        self.starts += 1
        self.stopped = False
        if self.starts <= self.failures:
            raise RuntimeError("boom")
        while not self.stopped:
            await asyncio.sleep(0.01)

    def stop(self) -> None:
        self.stopped = True


class Quits:
    def __init__(self) -> None:
        self.starts = 0

    async def start(self) -> None:
        self.starts += 1

    def stop(self) -> None:
        pass


async def test_crashed_agent_is_restarted() -> None:
    sup = Supervisor(EventBus(), max_restarts=5, backoff_seconds=0.01)
    agent = Flaky(failures=2)
    sup.add("Flaky", agent)
    sup.start_all()
    await asyncio.sleep(0.3)
    assert agent.starts == 3
    await sup.stop_all(timeout=1)
    assert not sup.fatal.is_set()


async def test_critical_agent_exhausting_restarts_is_fatal() -> None:
    sup = Supervisor(EventBus(), max_restarts=2, backoff_seconds=0.01)
    sup.add("ASR", Flaky(failures=99), critical=True)
    sup.start_all()
    await asyncio.wait_for(sup.fatal.wait(), timeout=2)
    assert "ASR" in sup.fatal_reason
    await sup.stop_all(timeout=1)


async def test_noncritical_agent_gives_up_quietly() -> None:
    sup = Supervisor(EventBus(), max_restarts=1, backoff_seconds=0.01)
    agent = Flaky(failures=99)
    sup.add("Diarization", agent)
    await asyncio.wait(sup.start_all(), timeout=2)
    assert agent.starts == 2
    assert not sup.fatal.is_set()


async def test_agent_that_returns_is_not_restarted() -> None:
    sup = Supervisor(EventBus(), backoff_seconds=0.01)
    agent = Quits()
    sup.add("Quits", agent)
    await asyncio.wait(sup.start_all(), timeout=1)
    assert agent.starts == 1
