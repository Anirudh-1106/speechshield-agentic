"""
Supervisor — keeps agents running.

Each agent's ``start()`` coroutine runs inside a supervised task.  If it
raises, the supervisor removes the agent's bus subscriptions, waits with
exponential back-off, and starts it again (the agent object is reused, so
already-loaded models are not reloaded).

  • An agent whose ``start()`` *returns* is treated as deliberately finished
    (e.g. diarization disabling itself when its model is unavailable).
  • A *critical* agent that exhausts its restarts triggers a full shutdown:
    a pipeline without ASR or capture should stop loudly, not limp on.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from core.health import health_checks
from core.logging import get_logger
from core.metrics import metrics

logger = get_logger("core.supervisor")


@dataclass
class AgentSpec:
    name: str
    agent: Any
    critical: bool = False
    restarts: int = 0


class Supervisor:
    """Runs agents with restart-on-crash and back-off."""

    def __init__(self, bus: Any, *, max_restarts: int = 5, backoff_seconds: float = 1.0) -> None:
        self.bus = bus
        self.max_restarts = max_restarts
        self.backoff_seconds = backoff_seconds
        self.specs: list[AgentSpec] = []
        self.fatal = asyncio.Event()
        self.fatal_reason: Optional[str] = None
        self._tasks: list[asyncio.Task] = []
        self._stopping = False

    def add(self, name: str, agent: Any, *, critical: bool = False) -> None:
        self.specs.append(AgentSpec(name, agent, critical))
        health_checks().mark_starting(name)

    def start_all(self) -> list[asyncio.Task]:
        self._tasks = [asyncio.create_task(self._run(spec), name=spec.name) for spec in self.specs]
        return self._tasks

    async def _run(self, spec: AgentSpec) -> None:
        while not self._stopping:
            try:
                await spec.agent.start()
                if not self._stopping:
                    logger.info("agent_exited", agent=spec.name)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stopping:
                    return
                spec.restarts += 1
                metrics().agent_restarts_total.labels(agent=spec.name).inc()
                health_checks().mark_error(spec.name, f"{type(exc).__name__}: {exc}")
                logger.error(
                    "agent_crashed",
                    agent=spec.name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    restart=spec.restarts,
                    max_restarts=self.max_restarts,
                )
                try:
                    spec.agent.stop()
                except Exception:
                    pass
                self.bus.unsubscribe_owner(spec.agent)

                if spec.restarts > self.max_restarts:
                    if spec.critical:
                        self.fatal_reason = f"{spec.name} failed {spec.restarts} times: {exc}"
                        self.fatal.set()
                    return
                delay = min(30.0, self.backoff_seconds * 2 ** (spec.restarts - 1))
                await asyncio.sleep(delay)
                health_checks().mark_starting(spec.name)

    async def stop_all(self, timeout: float = 10.0) -> None:
        """Ask every agent to stop, then cancel whatever is still running."""
        self._stopping = True
        for spec in self.specs:
            try:
                spec.agent.stop()
            except Exception as exc:
                logger.warning("agent_stop_error", agent=spec.name, error=str(exc))
        if not self._tasks:
            return
        _, pending = await asyncio.wait(self._tasks, timeout=timeout)
        for t in pending:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
