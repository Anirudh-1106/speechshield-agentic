"""
EventBus — Asyncio-based global event bus for inter-agent communication.

Agents emit events and listen for events without direct coupling.
Each event is a dict with at least a "type" key.
Includes correlation ID support for distributed tracing.

Delivery model
--------------
Every subscription owns a bounded inbox and a worker task, so:

  • a slow handler (e.g. Whisper transcription) never blocks delivery to
    other agents — there is no head-of-line blocking;
  • each handler still sees its events strictly in order;
  • when a handler falls behind, the *oldest* queued event is dropped and
    counted, keeping the system real-time instead of building unbounded lag.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger("EventBus")


class Event:
    """Typed wrapper around an event payload with correlation tracking."""

    __slots__ = ("type", "data", "timestamp", "correlation_id")

    def __init__(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        self.type: str = event_type
        self.data: dict[str, Any] = data or {}
        self.timestamp: float = time.time()
        # Generate correlation ID if not provided, for tracing through the pipeline
        self.correlation_id: str = correlation_id or str(uuid.uuid4())

    def __repr__(self) -> str:
        return f"Event(type={self.type!r}, correlation_id={self.correlation_id[:8]}..., keys={list(self.data)})"


# Type alias for handler coroutines
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]

# Subscribe to this type to receive every event (monitoring, recording).
WILDCARD = "*"


@dataclass
class _Subscription:
    event_type: str
    handler: EventHandler
    queue: asyncio.Queue
    worker: Optional[asyncio.Task] = None
    dropped: int = 0
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = getattr(self.handler, "__qualname__", repr(self.handler))


class EventBus:
    """
    Async event bus with per-subscriber inboxes.

    Agents register handlers for specific event types (or ``"*"``).
    The dispatcher copies each event into every matching inbox; one worker
    per subscription drains its inbox sequentially.
    """

    def __init__(self, maxsize: int = 0, subscriber_queue_size: int = 256) -> None:
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self._subscriptions: dict[str, list[_Subscription]] = {}
        self._subscriber_queue_size = subscriber_queue_size
        self._waiters: dict[tuple[str, str], list[asyncio.Future]] = {}
        self._running: bool = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(self, event_type: str, handler: EventHandler, *, queue_size: Optional[int] = None) -> None:
        """Register *handler* to be called whenever *event_type* is emitted."""
        sub = _Subscription(
            event_type=event_type,
            handler=handler,
            queue=asyncio.Queue(maxsize=queue_size or self._subscriber_queue_size),
        )
        self._subscriptions.setdefault(event_type, []).append(sub)
        if self._running:
            self._start_worker(sub)
        logger.debug("Subscribed handler %s to '%s'", sub.name, event_type)

    def unsubscribe_owner(self, owner: object) -> int:
        """Remove every subscription whose handler is a bound method of *owner*.

        Used by the supervisor before restarting a crashed agent, so the
        restarted agent does not end up subscribed twice.
        """
        removed = 0
        for event_type, subs in list(self._subscriptions.items()):
            keep = []
            for sub in subs:
                if getattr(sub.handler, "__self__", None) is owner:
                    if sub.worker is not None:
                        sub.worker.cancel()
                    removed += 1
                else:
                    keep.append(sub)
            if keep:
                self._subscriptions[event_type] = keep
            else:
                del self._subscriptions[event_type]
        return removed

    @property
    def _handlers(self) -> dict[str, list[EventHandler]]:
        """Handlers by event type (read-only view, kept for introspection/tests)."""
        return {t: [s.handler for s in subs] for t, subs in self._subscriptions.items()}

    def _start_worker(self, sub: _Subscription) -> None:
        if sub.worker is None or sub.worker.done():
            sub.worker = asyncio.create_task(self._worker(sub), name=f"bus:{sub.name}")

    async def _worker(self, sub: _Subscription) -> None:
        while True:
            event = await sub.queue.get()
            try:
                await sub.handler(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # error boundary: one bad event never kills the agent
                logger.error(
                    "Handler %s raised %s on %s: %s",
                    sub.name,
                    type(exc).__name__,
                    event,
                    exc,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
                _count_error(sub.name)
            finally:
                sub.queue.task_done()

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    async def emit(self, event: Event) -> None:
        """Put an event onto the bus queue."""
        await self._queue.put(event)
        logger.debug("Emitted %s", event)

    def emit_threadsafe(self, event: Event) -> None:
        """Emit from a non-asyncio thread (audio callbacks, HTTP server)."""
        if self._loop is None:
            raise RuntimeError("EventBus is not running")
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    async def wait_for(self, event_type: str, correlation_id: str, timeout: float) -> Event:
        """Wait for the next *event_type* carrying *correlation_id*.

        Register the waiter *before* emitting the triggering event to avoid
        missing a fast response.
        """
        fut = self.expect(event_type, correlation_id)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self.discard_expectation(event_type, correlation_id, fut)

    def expect(self, event_type: str, correlation_id: str) -> asyncio.Future:
        """Return a future resolved by the next matching event.

        Always pair with :meth:`discard_expectation` (e.g. in ``finally``) so
        abandoned waits do not accumulate.
        """
        fut = asyncio.get_running_loop().create_future()
        self._waiters.setdefault((event_type, correlation_id), []).append(fut)
        return fut

    def discard_expectation(self, event_type: str, correlation_id: str, fut: asyncio.Future) -> None:
        key = (event_type, correlation_id)
        futs = self._waiters.get(key)
        if futs and fut in futs:
            futs.remove(fut)
            if not futs:
                del self._waiters[key]

    # ------------------------------------------------------------------
    # Dispatch loop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Main dispatch loop — runs until ``stop()`` is called.

        Pulls events from the queue and fans them out to subscriber inboxes.
        """
        self._running = True
        self._loop = asyncio.get_running_loop()
        for subs in self._subscriptions.values():
            for sub in subs:
                self._start_worker(sub)
        logger.info("EventBus started.")
        try:
            while self._running:
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                try:
                    self._dispatch(event)
                finally:
                    self._queue.task_done()
        finally:
            await self._stop_workers()
            logger.info("EventBus stopped.")

    def _dispatch(self, event: Event) -> None:
        waiters = self._waiters.pop((event.type, event.correlation_id), None)
        for fut in waiters or ():
            if not fut.done():
                fut.set_result(event)

        targets = self._subscriptions.get(event.type, []) + self._subscriptions.get(WILDCARD, [])
        if not targets:
            logger.debug("No handlers for event type '%s'", event.type)
            return

        for sub in targets:
            if sub.queue.full():
                # Real-time policy: shed the oldest work, never block the bus.
                try:
                    sub.queue.get_nowait()
                    sub.queue.task_done()
                except asyncio.QueueEmpty:
                    pass
                sub.dropped += 1
                _count_drop(event.type, sub.name)
                logger.warning("Inbox full for %s — dropped oldest event (%d total)", sub.name, sub.dropped)
            sub.queue.put_nowait(event)

    async def _stop_workers(self) -> None:
        workers = [s.worker for subs in self._subscriptions.values() for s in subs if s.worker]
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        for subs in self._subscriptions.values():
            for s in subs:
                s.worker = None

    def stop(self) -> None:
        """Signal the dispatch loop to exit."""
        self._running = False
        logger.info("EventBus stop requested.")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def queue_depths(self) -> dict[str, int]:
        """Current inbox depth per subscriber (for metrics and the dashboard)."""
        return {
            s.name: s.queue.qsize()
            for subs in self._subscriptions.values()
            for s in subs
        }


def _count_drop(event_type: str, subscriber: str) -> None:
    try:
        from core.metrics import metrics
        metrics().event_bus_dropped_total.labels(event_type=event_type, subscriber=subscriber).inc()
    except Exception:
        pass


def _count_error(subscriber: str) -> None:
    try:
        from core.metrics import metrics
        metrics().handler_errors_total.labels(subscriber=subscriber).inc()
    except Exception:
        pass
