"""
OrchestratorAgent — Meta-agent / system supervisor.

Responsibilities:
  • Enforce global policies (latency budget, rewrite escalation)
  • Monitor pipeline latency
  • Adjust rewrite strictness on repeated toxicity
  • Log all agent decisions
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from core.event_bus import Event

if TYPE_CHECKING:
    from core.event_bus import EventBus
    from core.state import GlobalState

logger = logging.getLogger("Orchestrator")


class OrchestratorAgent:
    """System-level supervisor that observes all events and enforces policies."""

    def __init__(self, event_bus: "EventBus", state: "GlobalState") -> None:
        self.bus = event_bus
        self.state = state
        self._running = False
        # Track per-pass timing
        self._pass_starts: dict[str, float] = {}
        # Temporary store for toxicity data (needed when rewrite_result arrives)
        self._pending_toxicity: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True

        # Subscribe to every pipeline event for monitoring
        self.bus.subscribe("audio_chunk", self._on_audio_chunk)
        self.bus.subscribe("text_transcript", self._on_transcript)
        self.bus.subscribe("toxicity_result", self._on_toxicity_result)
        self.bus.subscribe("rewrite_result", self._on_rewrite_result)
        self.bus.subscribe("audio_output_ready", self._on_audio_output)

        logger.info("OrchestratorAgent started — monitoring all pipeline events.")

        while self._running:
            await asyncio.sleep(1.0)
            self._periodic_report()

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Event handlers (observational — no side effects on pipeline data)
    # ------------------------------------------------------------------

    async def _on_audio_chunk(self, event: Event) -> None:
        # Mark pipeline pass start time using a rolling key
        pass_key = f"pass_{self.state.pass_counter + 1}"
        if pass_key not in self._pass_starts:
            self._pass_starts[pass_key] = time.time()

    async def _on_transcript(self, event: Event) -> None:
        text = event.data.get("text", "")
        speaker = event.data.get("speaker", "default")
        logger.info("[Orchestrator] Transcript from '%s': %s", speaker, text[:100])

    async def _on_toxicity_result(self, event: Event) -> None:
        toxic = event.data.get("toxic", False)
        confidence = event.data.get("confidence", 0.0)
        speaker = event.data.get("speaker", "default")
        text = event.data.get("text", "")

        # Stash for dashboard row creation when rewrite arrives
        self._pending_toxicity[text] = {
            "toxic": toxic,
            "confidence": confidence,
            "speaker": speaker,
            "timestamp": event.data.get("timestamp", time.time()),
        }

        profile = self.state.get_speaker(speaker)
        logger.info(
            "[Orchestrator] Toxicity — speaker='%s' toxic=%s conf=%.3f "
            "consecutive=%d strictness=%.2f text='%s'",
            speaker,
            toxic,
            confidence,
            profile.consecutive_toxic,
            profile.rewrite_strictness,
            text[:60],
        )

    async def _on_rewrite_result(self, event: Event) -> None:
        was_toxic = event.data.get("was_toxic", False)
        original = event.data.get("original_text", "")
        rewritten = event.data.get("rewritten_text", "")
        speaker = event.data.get("speaker", "default")
        ts = event.data.get("timestamp", time.time())

        # Build dashboard row
        tox_info = self._pending_toxicity.pop(original, {})
        from core.state import DashboardRow
        self.state.add_dashboard_row(DashboardRow(
            timestamp=ts,
            transcript=original,
            toxic=tox_info.get("toxic", was_toxic),
            confidence=tox_info.get("confidence", 0.0),
            rewritten=rewritten,
            speaker=speaker,
        ))

        if was_toxic:
            logger.info(
                "[Orchestrator] Rewrite applied: '%s' → '%s'",
                original[:50],
                rewritten[:50],
            )

    async def _on_audio_output(self, event: Event) -> None:
        wav_path = event.data.get("wav_path", "")
        ts = event.data.get("timestamp", time.time())

        # Calculate end-to-end latency for the current logical pass
        pass_key = f"pass_{self.state.pass_counter}"
        start = self._pass_starts.pop(pass_key, None)
        if start:
            latency_ms = (ts - start) * 1000
            logger.info(
                "[Orchestrator] End-to-end latency for pass #%d: %.1f ms (budget: %.0f ms)",
                self.state.pass_counter,
                latency_ms,
                self.state.config["latency_budget_ms"],
            )
            if latency_ms > self.state.config["latency_budget_ms"]:
                logger.warning(
                    "[Orchestrator] Latency EXCEEDED budget! %.1f ms > %.0f ms",
                    latency_ms,
                    self.state.config["latency_budget_ms"],
                )

        logger.info("[Orchestrator] Audio output ready: %s", wav_path)

    # ------------------------------------------------------------------
    # Periodic health report
    # ------------------------------------------------------------------

    def _periodic_report(self) -> None:
        avg_lat = self.state.avg_latency_ms
        speakers = self.state.speakers
        speaker_summary = ", ".join(
            f"{sid}(toxic={p.toxic_count}/{p.total_utterances} strict={p.rewrite_strictness:.1f})"
            for sid, p in speakers.items()
        )
        if speaker_summary:
            logger.debug(
                "[Orchestrator] Health — avg_latency=%.1f ms  speakers=[%s]",
                avg_lat,
                speaker_summary,
            )
