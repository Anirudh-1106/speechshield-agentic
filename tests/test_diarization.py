import pytest
import asyncio
from unittest.mock import MagicMock, patch

from core.event_bus import EventBus, Event
from core.state import GlobalState
from agents.diarization_agent import DiarizationAgent

@pytest.mark.asyncio
async def test_diarization_agent_event_handling():
    bus = EventBus()
    state = GlobalState()
    agent = DiarizationAgent(bus, state)
    
    # Mock the internal diarize method so we don't need pyannote installed for unit tests
    agent._diarize_buffer = MagicMock(return_value=[
        {"start": 0.0, "end": 1.0, "speaker": "speaker_1", "confidence": 0.9, "duration": 1.0}
    ])
    
    # We won't actually start the agent loop, just call the handler directly
    event = Event("audio_chunk", {"audio": b"x" * 64000}) # 2 seconds of audio
    
    # No need to subscribe since bus is not started
    
    # Instead of running the full loop which is blocking in the test, 
    # we just invoke the handler directly
    await agent._handle_audio_chunk(event)
    
    # Check that it extracted the speaker and updated state
    assert not bus._queue.empty()
    emitted = await bus._queue.get()
    assert emitted.type == "speaker_identified"
    assert emitted.data["speaker_id"] == "speaker_1"
    
    assert state.current_speaker == "speaker_1"
