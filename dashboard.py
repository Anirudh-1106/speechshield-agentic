"""
SpeechShield — Streamlit Dashboard

Live web UI showing:
  • Real-time transcript feed with toxicity highlighting
  • Rewrite comparison (original vs. polite version)
  • Speaker toxicity trend charts
  • System health metrics (latency, pass count)

Usage:
    streamlit run dashboard.py

The dashboard reads state that the running pipeline writes to a
shared JSON file (dashboard_state.json) — this decouples Streamlit's
own event loop from the asyncio pipeline.
"""

import json
import os
import time
from pathlib import Path

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STATE_FILE = Path(__file__).parent / "logs" / "dashboard_state.json"
REFRESH_INTERVAL = 2  # seconds

st.set_page_config(
    page_title="SpeechShield Dashboard",
    page_icon="🛡️",
    layout="wide",
)


def load_state() -> dict:
    """Load the latest pipeline state from the shared JSON file."""
    if not STATE_FILE.exists():
        return {}
    try:
        text = STATE_FILE.read_text(encoding="utf-8")
        return json.loads(text) if text.strip() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def render_header(state: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pipeline Passes", state.get("pass_counter", 0))
    col2.metric("Avg Latency", f"{state.get('avg_latency_ms', 0):.0f} ms")

    speakers = state.get("speakers", {})
    total_toxic = sum(s.get("toxic_count", 0) for s in speakers.values())
    total_utterances = sum(s.get("total_utterances", 0) for s in speakers.values())
    col3.metric("Toxic / Total", f"{total_toxic} / {total_utterances}")

    ratio = (total_toxic / total_utterances * 100) if total_utterances > 0 else 0
    col4.metric("Toxicity Rate", f"{ratio:.1f}%")


def render_live_feed(state: dict) -> None:
    st.subheader("📝 Live Transcript Feed")
    log = state.get("log", [])
    if not log:
        st.info("No transcripts yet. Speak into the microphone with the pipeline running.")
        return

    # Show newest first
    for row in reversed(log):
        is_toxic = row.get("toxic", False)
        conf = row.get("confidence", 0.0)
        ts = row.get("time", "")

        if is_toxic:
            with st.container():
                st.markdown(
                    f"🔴 **{ts}** &nbsp;|&nbsp; Confidence: `{conf:.1%}` &nbsp;|&nbsp; "
                    f"Speaker: `{row.get('speaker', 'default')}`"
                )
                col_orig, col_arrow, col_rewrite = st.columns([5, 1, 5])
                with col_orig:
                    st.error(f"**Original:** {row.get('transcript', '')}")
                with col_arrow:
                    st.markdown("<h2 style='text-align:center'>→</h2>", unsafe_allow_html=True)
                with col_rewrite:
                    st.success(f"**Rewritten:** {row.get('rewritten', '')}")
                st.divider()
        else:
            st.markdown(
                f"🟢 **{ts}** &nbsp;|&nbsp; `{row.get('transcript', '')}` "
                f"&nbsp;|&nbsp; Speaker: `{row.get('speaker', 'default')}`"
            )


def render_speaker_stats(state: dict) -> None:
    st.subheader("👤 Speaker Toxicity Profiles")
    speakers = state.get("speakers", {})
    if not speakers:
        st.info("No speaker data yet.")
        return

    rows = []
    for sid, data in speakers.items():
        rows.append({
            "Speaker": sid,
            "Utterances": data.get("total_utterances", 0),
            "Toxic Count": data.get("toxic_count", 0),
            "Toxicity %": f"{data.get('toxicity_ratio', 0) * 100:.1f}%",
            "Consecutive Toxic": data.get("consecutive_toxic", 0),
            "Strictness": data.get("rewrite_strictness", 1.0),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_toxicity_chart(state: dict) -> None:
    st.subheader("📊 Toxicity Over Time")
    log = state.get("log", [])
    if len(log) < 2:
        st.info("Need at least 2 data points to chart.")
        return

    chart_data = pd.DataFrame([
        {
            "time": row.get("time", ""),
            "confidence": row.get("confidence", 0.0),
            "toxic": 1 if row.get("toxic", False) else 0,
        }
        for row in log
    ])

    st.line_chart(chart_data, x="time", y=["confidence", "toxic"], height=300)


def render_rewrite_history(state: dict) -> None:
    st.subheader("🔄 Rewrite History")
    log = state.get("log", [])
    toxic_rows = [r for r in log if r.get("toxic", False)]
    if not toxic_rows:
        st.info("No toxic rewrites recorded yet.")
        return

    rows = []
    for r in reversed(toxic_rows[-50:]):
        rows.append({
            "Time": r.get("time", ""),
            "Original": r.get("transcript", ""),
            "Rewritten": r.get("rewritten", ""),
            "Confidence": f"{r.get('confidence', 0):.1%}",
            "Speaker": r.get("speaker", "default"),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("🛡️ SpeechShield — Real-Time Voice Moderation")
    st.caption("Multi-Agent Agentic AI Pipeline • Live Dashboard")

    state = load_state()

    if not state:
        st.warning(
            "⏳ Waiting for pipeline data…\n\n"
            "Make sure the pipeline is running (`python main.py`) and has processed some audio."
        )
        time.sleep(REFRESH_INTERVAL)
        st.rerun()
        return

    render_header(state)
    st.divider()

    tab_feed, tab_chart, tab_history, tab_speakers = st.tabs([
        "📝 Live Feed", "📊 Toxicity Chart", "🔄 Rewrite History", "👤 Speakers"
    ])

    with tab_feed:
        render_live_feed(state)
    with tab_chart:
        render_toxicity_chart(state)
    with tab_history:
        render_rewrite_history(state)
    with tab_speakers:
        render_speaker_stats(state)

    # Auto-refresh
    time.sleep(REFRESH_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()
