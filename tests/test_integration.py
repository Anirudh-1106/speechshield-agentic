"""
Integration test — full pipeline simulation (no mic, no TTS).

Verifies that events flow correctly through the bus from
text_transcript → toxicity_result → rewrite_result.

This test DOES load the toxic-bert model (slow, ~30s first time).
Mark with @pytest.mark.slow so CI can skip if needed.
"""

import asyncio
import pytest
from core.event_bus import EventBus, Event
from core.state import GlobalState
from agents.toxicity_agent import ToxicityAgent
from agents.rewrite_agent import RewriteAgent


@pytest.mark.slow
class TestPipelineIntegration:
    """End-to-end pipeline tests (requires models)."""

    @pytest.fixture
    def setup_pipeline(self):
        """Create bus, state, and agents."""
        bus = EventBus()
        state = GlobalState()
        toxicity_agent = ToxicityAgent(bus, state)
        rewrite_agent = RewriteAgent(bus, state)
        return bus, state, toxicity_agent, rewrite_agent

    @pytest.mark.asyncio
    async def test_toxic_text_gets_rewritten(self, setup_pipeline) -> None:
        bus, state, toxicity_agent, rewrite_agent = setup_pipeline

        results: list[Event] = []
        done = asyncio.Event()

        async def capture_rewrite(event: Event) -> None:
            results.append(event)
            done.set()

        bus.subscribe("rewrite_result", capture_rewrite)

        # Start agents and bus
        tasks = [
            asyncio.create_task(bus.start()),
            asyncio.create_task(toxicity_agent.start()),
            asyncio.create_task(rewrite_agent.start()),
        ]

        await asyncio.sleep(0.5)

        # Inject a toxic transcript directly
        await bus.emit(Event("text_transcript", {
            "text": "You are a fucking idiot",
            "speaker": "test_user",
        }))

        # Wait for pipeline to process
        try:
            await asyncio.wait_for(done.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            pass

        # Cleanup
        bus.stop()
        toxicity_agent.stop()
        rewrite_agent.stop()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # Verify
        assert len(results) >= 1
        r = results[0]
        assert r.data["was_toxic"] is True
        assert "fucking" not in r.data["rewritten_text"].lower()
        assert "idiot" not in r.data["rewritten_text"].lower()

    @pytest.mark.asyncio
    async def test_clean_text_passes_through(self, setup_pipeline) -> None:
        bus, state, toxicity_agent, rewrite_agent = setup_pipeline

        results: list[Event] = []
        done = asyncio.Event()

        async def capture_rewrite(event: Event) -> None:
            results.append(event)
            done.set()

        bus.subscribe("rewrite_result", capture_rewrite)

        tasks = [
            asyncio.create_task(bus.start()),
            asyncio.create_task(toxicity_agent.start()),
            asyncio.create_task(rewrite_agent.start()),
        ]

        await asyncio.sleep(0.5)

        await bus.emit(Event("text_transcript", {
            "text": "I really enjoy working with you",
            "speaker": "test_user",
        }))

        try:
            await asyncio.wait_for(done.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            pass

        bus.stop()
        toxicity_agent.stop()
        rewrite_agent.stop()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) >= 1
        r = results[0]
        assert r.data["was_toxic"] is False
        assert r.data["rewritten_text"] == "I really enjoy working with you"
