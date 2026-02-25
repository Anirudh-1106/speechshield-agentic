"""
Tests for core.event_bus — EventBus and Event classes.
"""

import asyncio
import pytest
from core.event_bus import Event, EventBus


class TestEvent:
    """Unit tests for the Event data class."""

    def test_event_creation(self) -> None:
        e = Event("test_type", {"key": "value"})
        assert e.type == "test_type"
        assert e.data == {"key": "value"}
        assert e.timestamp > 0

    def test_event_default_data(self) -> None:
        e = Event("empty")
        assert e.data == {}

    def test_event_repr(self) -> None:
        e = Event("test", {"a": 1, "b": 2})
        r = repr(e)
        assert "test" in r
        assert "a" in r


class TestEventBus:
    """Unit and integration tests for the EventBus."""

    @pytest.mark.asyncio
    async def test_subscribe_and_emit(self) -> None:
        """Handlers receive emitted events of their subscribed type."""
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("ping", handler)

        # Start bus in background, emit, then stop
        task = asyncio.create_task(bus.start())
        await bus.emit(Event("ping", {"msg": "hello"}))
        await asyncio.sleep(0.3)
        bus.stop()
        await task

        assert len(received) == 1
        assert received[0].data["msg"] == "hello"

    @pytest.mark.asyncio
    async def test_multiple_handlers(self) -> None:
        """Multiple handlers for the same event type all get called."""
        bus = EventBus()
        results: list[str] = []

        async def handler_a(event: Event) -> None:
            results.append("a")

        async def handler_b(event: Event) -> None:
            results.append("b")

        bus.subscribe("multi", handler_a)
        bus.subscribe("multi", handler_b)

        task = asyncio.create_task(bus.start())
        await bus.emit(Event("multi"))
        await asyncio.sleep(0.3)
        bus.stop()
        await task

        assert sorted(results) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_unsubscribed_events_ignored(self) -> None:
        """Events with no subscribers don't crash the bus."""
        bus = EventBus()

        task = asyncio.create_task(bus.start())
        await bus.emit(Event("nobody_listens"))
        await asyncio.sleep(0.3)
        bus.stop()
        await task
        # No exception = pass

    @pytest.mark.asyncio
    async def test_handler_exception_handled(self) -> None:
        """A failing handler doesn't crash the bus."""
        bus = EventBus()
        good_results: list[str] = []

        async def bad_handler(event: Event) -> None:
            raise ValueError("boom")

        async def good_handler(event: Event) -> None:
            good_results.append("ok")

        bus.subscribe("err_test", bad_handler)
        bus.subscribe("err_test", good_handler)

        task = asyncio.create_task(bus.start())
        await bus.emit(Event("err_test"))
        await asyncio.sleep(0.3)
        bus.stop()
        await task

        # Good handler still ran despite bad handler crashing
        assert good_results == ["ok"]

    @pytest.mark.asyncio
    async def test_stop_halts_loop(self) -> None:
        """Calling stop() terminates the dispatch loop."""
        bus = EventBus()
        task = asyncio.create_task(bus.start())
        await asyncio.sleep(0.2)
        bus.stop()
        await asyncio.wait_for(task, timeout=2.0)
        # Task completed = pass
