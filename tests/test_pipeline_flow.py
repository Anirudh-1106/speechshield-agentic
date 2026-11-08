"""
Fast end-to-end flow through the real agents and bus, without ML models:
the toxicity agent runs keyword-only and the rewriter rules-only.

Covers the event contract (text_transcript → toxicity_result →
rewrite_result), the orchestrator's dashboard rows and latency, and the
moderation gateway used by the HTTP API.
"""

import asyncio

import pytest

from agents.orchestrator import OrchestratorAgent
from agents.rewrite_agent import RewriteAgent
from agents.toxicity_agent import ToxicityAgent
from config.settings import Settings
from core.api import ModerationGateway
from core.event_bus import Event, EventBus
from core.state import GlobalState


@pytest.fixture
async def pipeline():
    s = Settings()
    s = s.model_copy(update={"rewrite": s.rewrite.model_copy(update={"llm_provider": "local"})})
    bus, state = EventBus(), GlobalState(settings=s)
    tox = ToxicityAgent(bus, state)
    tox._keyword_only = True  # no model download in fast tests
    agents = [tox, RewriteAgent(bus, state, verifier=tox.score), OrchestratorAgent(bus, state)]
    tasks = [asyncio.create_task(bus.start())] + [asyncio.create_task(a.start()) for a in agents]
    await asyncio.sleep(0.2)
    yield bus, state
    for a in agents:
        a.stop()
    bus.stop()
    await asyncio.wait(tasks, timeout=2)


async def test_toxic_transcript_flows_to_rewrite(pipeline) -> None:
    bus, state = pipeline
    words = [{"word": w, "start": i * 0.3, "end": i * 0.3 + 0.25} for i, w in enumerate(["you", "fucking", "idiot"])]
    fut = bus.expect("rewrite_result", "seg_1")
    tox_fut = bus.expect("toxicity_result", "seg_1")
    await bus.emit(Event("text_transcript", {"text": "you fucking idiot", "speaker": "alice", "words": words}, correlation_id="seg_1"))
    rw = await asyncio.wait_for(fut, 3)
    tox = await asyncio.wait_for(tox_fut, 3)

    assert tox.data["toxic"] is True
    assert tox.data["toxic_spans"] == [{"start": 0.3, "end": 0.85, "text": "fucking idiot"}]
    assert rw.data["was_toxic"] is True
    assert rw.data["rewrite_method"] == "rules"
    assert "fuck" not in rw.data["rewritten_text"].lower()

    await asyncio.sleep(0.1)
    row = state.dashboard_log[-1]
    assert row.correlation_id == "seg_1" and row.speaker == "alice" and row.toxic
    assert state.pass_counter == 1
    assert state.get_speaker("alice").toxic_count == 1


async def test_clean_transcript_passes_through(pipeline) -> None:
    bus, state = pipeline
    fut = bus.expect("rewrite_result", "seg_2")
    await bus.emit(Event("text_transcript", {"text": "nice shot, I killed the boss", "speaker": "bob"}, correlation_id="seg_2"))
    rw = await asyncio.wait_for(fut, 3)
    assert rw.data["was_toxic"] is False
    assert rw.data["rewritten_text"] == "nice shot, I killed the boss"


async def test_gateway_moderates_text(pipeline) -> None:
    bus, state = pipeline
    gw = ModerationGateway(bus, state, asyncio.get_running_loop(), timeout=3)
    result = await gw.moderate("shut up you stupid bitch", speaker="user9")
    assert result["toxic"] is True
    assert result["masked"] == "s*** u* you s***** b****"
    assert "bitch" not in result["rewritten"]
    assert result["rewrite_method"] == "rules"
    assert state.trace(result["id"]).source == "api"
    assert bus._waiters == {}
