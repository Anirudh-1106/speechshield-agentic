"""
ModerationGateway — lets the HTTP server (its own thread and event loop)
push text through the live agent pipeline and await the verdict.

The request is injected as a ``text_transcript`` event with
``source="api"``, so it gets exactly the same toxicity model, keyword
safety net, per-speaker strictness and guarded LLM rewriting as speech —
but no TTS and no audio side effects.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

from core.event_bus import Event
from core.lexicon import mask_text

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState


class ModerationGateway:
    def __init__(self, bus: "EventBus", state: "GlobalState", loop: asyncio.AbstractEventLoop, timeout: float = 20.0) -> None:
        self.bus = bus
        self.state = state
        self.loop = loop
        self.timeout = timeout

    async def moderate(self, text: str, speaker: str = "api") -> dict[str, Any]:
        """Call from any event loop; runs the pipeline on the main loop."""
        cf = asyncio.run_coroutine_threadsafe(self._moderate(text, speaker), self.loop)
        return await asyncio.wrap_future(cf)

    def snapshot(self) -> dict[str, Any]:
        cf = asyncio.run_coroutine_threadsafe(self._snapshot(), self.loop)
        return cf.result(timeout=5.0)

    async def _snapshot(self) -> dict[str, Any]:
        return self.state.get_dashboard_snapshot()

    async def _moderate(self, text: str, speaker: str) -> dict[str, Any]:
        cid = f"api_{uuid.uuid4().hex[:12]}"
        trace = self.state.trace(cid)
        trace.speech_end = time.time()
        trace.source = "api"

        tox_fut = self.bus.expect("toxicity_result", cid)
        rw_fut = self.bus.expect("rewrite_result", cid)
        try:
            await self.bus.emit(Event("text_transcript", {
                "text": text, "speaker": speaker, "words": [], "source": "api", "timestamp": time.time(),
            }, correlation_id=cid))
            rw = await asyncio.wait_for(rw_fut, timeout=self.timeout)
            tox = tox_fut.result() if tox_fut.done() else None
        finally:
            self.bus.discard_expectation("toxicity_result", cid, tox_fut)
            self.bus.discard_expectation("rewrite_result", cid, rw_fut)

        toxic = bool(rw.data.get("was_toxic"))
        labels = {k: v for k, v in (tox.data.get("labels", {}) if tox else {}).items() if not k.startswith("_")}
        return {
            "id": cid,
            "text": text,
            "toxic": toxic,
            "confidence": round(float(tox.data.get("confidence", 0.0)) if tox else 0.0, 4),
            "labels": labels,
            "masked": mask_text(text, include_mild=toxic) if toxic else text,
            "rewritten": rw.data.get("rewritten_text", text),
            "rewrite_method": rw.data.get("rewrite_method", "passthrough"),
            "speaker": speaker,
            "latency_ms": round((time.time() - trace.speech_end) * 1000.0, 1),
        }
