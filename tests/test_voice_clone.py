"""VoiceCloneAgent (with a fake worker) and the worker's pure helpers."""

import base64
import json

import httpx
import numpy as np
import pytest

from agents.voice_clone_agent import VoiceCloneAgent
from core.event_bus import Event, EventBus
from core.state import GlobalState


class FakeWorker:
    def __init__(self) -> None:
        self.enrolled: dict[str, float] = {}
        self.prewarmed: list[str] = []
        self.synth: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        if request.url.path == "/enroll":
            secs = self.enrolled.get(body["speaker"], 0.0) + len(base64.b64decode(body["pcm_b64"])) / 32000
            self.enrolled[body["speaker"]] = secs
            return httpx.Response(200, json={"speaker": body["speaker"], "seconds": secs, "ready": secs >= 6})
        if request.url.path == "/prewarm":
            self.prewarmed.append(body["speaker"])
            return httpx.Response(200, json={"queued": len(body["texts"])})
        if request.url.path == "/synthesize":
            self.synth.append((body["speaker"], body["text"]))
            return httpx.Response(200, json={"pcm_b64": base64.b64encode(b"\x05\x00" * 800).decode(), "cached": False})
        return httpx.Response(404)


def make_agent():
    worker = FakeWorker()
    agent = VoiceCloneAgent(EventBus(), GlobalState())
    agent._client = httpx.AsyncClient(base_url="http://w", transport=httpx.MockTransport(worker))
    agent._worker_ready = True
    return agent, worker


async def enroll(agent, speaker, seconds, cid, toxic=False):
    audio = b"\x01\x00" * int(16000 * seconds)
    await agent._on_segment(Event("speech_segment", {"audio": audio, "sample_rate": 16000}, correlation_id=cid))
    await agent._on_toxicity(Event("toxicity_result", {"toxic": toxic, "speaker": speaker}, correlation_id=cid))


async def test_enrolls_clean_speech_then_prewarms() -> None:
    agent, worker = make_agent()
    await enroll(agent, "alice", 4, "a1")
    assert "alice" not in agent._ready_speakers
    await enroll(agent, "alice", 3, "a2")
    assert "alice" in agent._ready_speakers
    assert worker.prewarmed == ["alice"]


async def test_toxic_speech_is_never_used_as_reference() -> None:
    agent, worker = make_agent()
    await enroll(agent, "bob", 8, "b1", toxic=True)
    assert worker.enrolled == {}


async def reply(agent, text, speaker="alice"):
    await agent._on_clone_request(Event("clone_request", {"key": "k1", "text": text, "speaker": speaker}, correlation_id="c"))
    return agent.bus._queue.get_nowait().data


async def test_replacement_synthesised_in_speakers_voice() -> None:
    agent, worker = make_agent()
    await enroll(agent, "alice", 7, "a1")
    data = await reply(agent, "idiot")
    assert data["pcm"] == b"\x05\x00" * 800
    assert worker.synth == [("alice", "goof")]


async def test_intensifier_is_cut_without_synthesis() -> None:
    agent, worker = make_agent()
    await enroll(agent, "alice", 7, "a1")
    assert (await reply(agent, "fucking"))["pcm"] == b""
    assert worker.synth == []


async def test_unenrolled_speaker_falls_back_to_bleep() -> None:
    agent, _ = make_agent()
    assert (await reply(agent, "idiot", speaker="stranger"))["pcm"] is None


async def test_worker_down_falls_back_to_bleep() -> None:
    agent, _ = make_agent()
    agent._ready_speakers.add("alice")

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    agent._client = httpx.AsyncClient(base_url="http://w", transport=httpx.MockTransport(down))
    assert (await reply(agent, "idiot"))["pcm"] is None


# ---------------------------------------------------------------------------
# Worker helpers (importable without Chatterbox)
# ---------------------------------------------------------------------------

def test_stream_safetensors_matches_reference_loader(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    from voice_service.worker import stream_safetensors

    tensors = {
        "a": torch.randn(3, 4),
        "b": torch.arange(10, dtype=torch.int64),
        "c": torch.randn(2, 2).half(),
        "empty": torch.zeros(0),
    }
    path = tmp_path / "m.safetensors"
    st.save_file(tensors, str(path))
    loaded = stream_safetensors(path, "cpu")
    for k, v in tensors.items():
        assert torch.equal(loaded[k], v), k
        assert loaded[k].dtype == v.dtype


def test_trim_silence_keeps_soft_edges() -> None:
    from voice_service.worker import _trim_silence

    word = np.concatenate([np.zeros(8000), np.linspace(0, 0.05, 400), np.ones(3000) * 0.5, np.zeros(8000)]).astype(np.float32)
    out = _trim_silence(word)
    assert 3400 <= len(out) <= 3400 + 2 * 640 + 400  # word + 40 ms either side
