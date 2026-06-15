"""
EventBus — Asyncio-based global event bus for inter-agent communication.

Agents emit events and listen for events without direct coupling.
Each event is a dict with at least a "type" key.
Includes correlation ID support for distributed tracing.
"""

import asyncio
import logging
import uuid
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
        import time
        self.type: str = event_type
        self.data: dict[str, Any] = data or {}
        self.timestamp: float = time.time()
        # Generate correlation ID if not provided, for tracing through the pipeline
        self.correlation_id: str = correlation_id or str(uuid.uuid4())

    def __repr__(self) -> str:
        return f"Event(type={self.type!r}, correlation_id={self.correlation_id[:8]}...)"


# Type alias for handler coroutines
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Async event bus using asyncio.Queue.

    Agents register handlers for specific event types.
    The bus dispatches events to matching handlers concurrently.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self._handlers: dict[str, list[EventHandler]] = {}
        self._running: bool = False

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register *handler* to be called whenever *event_type* is emitted."""
        self._handlers.setdefault(event_type, []).append(handler)
        logger.debug("Subscribed handler %s to '%s'", handler.__qualname__, event_type)

    async def emit(self, event: Event) -> None:
        """Put an event onto the bus queue."""
        await self._queue.put(event)
        logger.debug("Emitted %s", event)

    async def start(self) -> None:
        """
        Main dispatch loop — runs until ``stop()`` is called.

        Pulls events from the queue and fans them out to subscribed handlers.
        """
        self._running = True
        logger.info("EventBus started.")
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            handlers = self._handlers.get(event.type, [])
            if not handlers:
                logger.warning("No handlers for event type '%s'", event.type)
                continue

            # Fan out to all handlers concurrently
            tasks = [asyncio.create_task(h(event)) for h in handlers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        "Handler %s raised %s: %s",
                        handlers[i].__qualname__,
                        type(result).__name__,
                        result,
                    )

            self._queue.task_done()

        logger.info("EventBus stopped.")

    def stop(self) -> None:
        """Signal the dispatch loop to exit."""
        self._running = False
        logger.info("EventBus stop requested.")
