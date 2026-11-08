"""
OrchestratorAgent — Meta-agent / system supervisor.

Responsibilities:
  • Measure end-to-end latency per utterance (end of speech → verdict)
    and enforce the latency budget
  • Assemble the dashboard feed from per-utterance pipeline traces
  • Publish queue depths and rolling latency to Prometheus
  • Log all agent decisions
"""

import asyncio
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

from core.event_bus import Event
from core.health import health_checks
from core.metrics import metrics
from core.state import DashboardRow

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = logging.getLogger("Orchestrator")


class OrchestratorAgent:
    """System-level supervisor that observes pipeline events and enforces policies."""

    def __init__(self, event_bus: "EventBus", state: "GlobalState") -> None:
        self.bus = event_bus
        self.state = state
        self._running = False
        # Rows still receiving updates (TTS / interceptor release), by correlation id
        self._open_rows: "OrderedDict[str, DashboardRow]" = OrderedDict()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self.bus.subscribe("text_transcript", self._on_transcript)
        self.bus.subscribe("toxicity_result", self._on_toxicity_result)
        self.bus.subscribe("rewrite_result", self._on_rewrite_result)
        self.bus.subscribe("segment_released", self._on_segment_released)

        health_checks().mark_ready("OrchestratorAgent")
        logger.info("OrchestratorAgent started — monitoring pipeline events.")

        while self._running:
            await asyncio.sleep(1.0)
            self._periodic_report()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Event handlers (observational — no side effects on pipeline data)
    # ------------------------------------------------------------------

    async def _on_transcript(self, event: Event) -> None:
        logger.info("[Orchestrator] Transcript from '%s': %s", event.data.get("speaker", "default"), event.data.get("text", "")[:100])

    async def _on_toxicity_result(self, event: Event) -> None:
        """The verdict is what the censor waits for, so latency is measured here."""
        trace = self.state.trace(event.correlation_id)
        latency_ms = trace.stages.get("toxicity", 0.0)
        self.state.record_pass(latency_ms)

        m = metrics()
        m.pipeline_latency_seconds.observe(latency_ms / 1000.0)
        m.pipeline_latency_quantiles.observe(latency_ms / 1000.0)
        m.pipeline_pass_counter.set(self.state.pass_counter)
        m.avg_latency_ms.set(self.state.avg_latency_ms)

        budget = self.state.settings.pipeline.latency_budget_ms
        if latency_ms > budget and event.data.get("source", "audio") == "audio":
            logger.warning(
                "[Orchestrator] Latency over budget: %.0f ms > %.0f ms (asr=%.0f toxicity=%.0f)",
                latency_ms, budget, trace.stages.get("asr", 0.0), latency_ms,
            )

        profile = self.state.get_speaker(event.data.get("speaker", "default"))
        logger.info(
            "[Orchestrator] Verdict speaker='%s' toxic=%s conf=%.3f latency=%.0fms strictness=%.2f",
            event.data.get("speaker", "default"),
            event.data.get("toxic", False),
            event.data.get("confidence", 0.0),
            latency_ms,
            profile.rewrite_strictness,
        )

    async def _on_rewrite_result(self, event: Event) -> None:
        cid = event.correlation_id
        trace = self.state.trace(cid)
        labels = {k: v for k, v in trace.labels.items() if not k.startswith("_")}
        row = DashboardRow(
            timestamp=event.data.get("timestamp", trace.speech_end),
            transcript=event.data.get("original_text", ""),
            toxic=event.data.get("was_toxic", False),
            confidence=trace.confidence,
            rewritten=event.data.get("rewritten_text", ""),
            speaker=event.data.get("speaker", "default"),
            latency_ms=trace.stages.get("toxicity", 0.0),
            correlation_id=cid,
            rewrite_method=event.data.get("rewrite_method", ""),
            action=trace.action,
            source=event.data.get("source", "audio"),
            stages=trace.stages,  # shared dict: later stages (tts, released) show up live
            top_label=max(labels, key=lambda k: labels[k]) if labels else "",
        )
        self.state.add_dashboard_row(row)
        self._open_rows[cid] = row
        while len(self._open_rows) > 200:
            self._open_rows.popitem(last=False)

        if row.toxic:
            logger.info("[Orchestrator] Rewrite (%s): '%s' → '%s'", row.rewrite_method, row.transcript[:50], row.rewritten[:50])

    async def _on_segment_released(self, event: Event) -> None:
        row = self._open_rows.get(event.correlation_id)
        if row is not None:
            row.action = event.data.get("action", "")

    # ------------------------------------------------------------------
    # Periodic health report
    # ------------------------------------------------------------------

    def _periodic_report(self) -> None:
        depths = self.bus.queue_depths()
        m = metrics()
        m.asr_queue_depth_samples.set(sum(v for k, v in depths.items() if k.startswith("ASRAgent")))
        m.toxicity_queue_depth_samples.set(sum(v for k, v in depths.items() if k.startswith("ToxicityAgent")))
        m.rewrite_queue_depth_samples.set(sum(v for k, v in depths.items() if k.startswith("RewriteAgent")))
        self.state.system_info["queue_depths"] = {k: v for k, v in depths.items() if v}

        speaker_summary = ", ".join(
            f"{sid}(toxic={p.toxic_count}/{p.total_utterances} strict={p.rewrite_strictness:.1f})"
            for sid, p in self.state.speakers.items()
        )
        if speaker_summary:
            logger.debug("[Orchestrator] Health — avg_latency=%.1f ms speakers=[%s]", self.state.avg_latency_ms, speaker_summary)
