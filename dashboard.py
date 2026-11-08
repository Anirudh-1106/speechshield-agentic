"""
SpeechShield — Streamlit Dashboard

Live web UI showing:
  • KPIs: utterances, toxicity rate, verdict latency vs budget, system health
  • Live feed with filters (toxic only, speaker, search) and censor actions
  • Latency breakdown per pipeline stage (p50 / p95) and over time
  • Per-speaker toxicity profiles and trends
  • Component health, and CSV export of the session

Usage:
    streamlit run dashboard.py          (main.py launches it automatically)

State comes from the running pipeline's HTTP API (GET /v1/state); if the
server is disabled it falls back to logs/dashboard_state.json.
"""

import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from config.settings import load_config

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CFG = load_config()
STATE_FILE = Path(__file__).parent / CFG.logging.log_dir / "dashboard_state.json"
API_HOST = "127.0.0.1" if CFG.server.host in ("0.0.0.0", "") else CFG.server.host
STATE_URL = os.getenv("SPEECHSHIELD_STATE_URL", f"http://{API_HOST}:{CFG.server.port}/v1/state")
API_KEY = CFG.server.api_key
REFRESH_INTERVAL = max(1.0, CFG.dashboard.refresh_interval)

ACTION_BADGE = {
    "passed": "🟢 passed",
    "bleeped": "🔊 bleeped",
    "muted": "🔇 muted",
    "replaced": "🗣️ replaced (TTS)",
    "voice_cloned": "🎭 voice-cloned",
    "timeout_muted": "⏱️ muted (timeout)",
    "timeout_passed": "⏱️ passed (timeout)",
    "": "",
}

st.set_page_config(
    page_title="SpeechShield Dashboard",
    page_icon="🛡️",
    layout="wide",
)


def load_state() -> tuple[dict, str]:
    """Latest pipeline state and where it came from."""
    try:
        headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else None
        resp = httpx.get(STATE_URL, timeout=2.0, headers=headers)
        if resp.status_code == 200:
            return resp.json(), "live"
    except httpx.HTTPError:
        pass
    if STATE_FILE.exists():
        try:
            text = STATE_FILE.read_text(encoding="utf-8")
            if text.strip():
                return json.loads(text), "file"
        except (json.JSONDecodeError, OSError):
            pass
    return {}, "none"


def feed_frame(state: dict) -> pd.DataFrame:
    log = state.get("log", [])
    if not log:
        return pd.DataFrame()
    df = pd.DataFrame(log)
    df["when"] = pd.to_datetime(df["ts"], unit="s") if "ts" in df else df["time"]
    return df


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def render_header(state: dict, source: str) -> None:
    system = state.get("system", {})
    health = state.get("health", {})
    speakers = state.get("speakers", {})
    total_toxic = sum(s.get("toxic_count", 0) for s in speakers.values())
    total = sum(s.get("total_utterances", 0) for s in speakers.values())
    budget = state.get("latency_budget_ms", 0) or 0
    verdict = state.get("stage_latency", {}).get("toxicity", {})

    c = st.columns(5)
    c[0].metric("Utterances", total)
    rate = total_toxic / total * 100 if total else 0
    c[1].metric("Toxic", f"{total_toxic}  ({rate:.0f}%)")
    p50 = verdict.get("p50", state.get("avg_latency_ms", 0))
    over = bool(budget) and p50 > budget
    c[2].metric("Verdict latency p50", f"{p50:.0f} ms" + ("  ⚠️" if over else ""), help=f"Latency budget: {budget:.0f} ms")
    c[3].metric("Verdict latency p95", f"{verdict.get('p95', 0):.0f} ms")
    status = "🟢 OK" if health.get("ready") and health.get("alive") else ("🟡 Degraded" if health.get("alive", True) else "🔴 Error")
    c[4].metric("System", status if source == "live" else "⚪ Offline")

    started = system.get("started_at")
    uptime = f"{(time.time() - started) / 60:.0f} min" if started else "—"
    st.caption(
        f"Mode **{system.get('mode', '?')}**"
        + (f" ({system.get('censor_mode')})" if system.get("censor_mode") else "")
        + f" · ASR {system.get('asr', '—')} · Toxicity {system.get('toxicity', '—')}"
        + f" · Rewrite {system.get('rewrite', '—')} · Uptime {uptime}"
        + ("" if source == "live" else " · ⚠️ showing last saved state")
    )


def render_live_feed(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No utterances yet. Speak (or play call audio) with the pipeline running, "
                "or POST text to /v1/moderate.")
        return

    f1, f2, f3 = st.columns([1, 2, 3])
    toxic_only = f1.toggle("Toxic only", value=False)
    speakers = sorted(df["speaker"].unique())
    chosen = f2.multiselect("Speakers", speakers, default=speakers)
    query = f3.text_input("Search", placeholder="Filter transcripts…")

    view = df[df["speaker"].isin(chosen)]
    if toxic_only:
        view = view[view["toxic"]]
    if query:
        q = query.lower()
        view = view[view["transcript"].str.lower().str.contains(q, regex=False)
                    | view["rewritten"].str.lower().str.contains(q, regex=False)]

    for row in view.iloc[::-1].head(60).itertuples():
        meta = f"`{row.time}` · **{row.speaker}** · {row.latency_ms:.0f} ms"
        if getattr(row, "action", ""):
            meta += f" · {ACTION_BADGE.get(row.action, row.action)}"
        if getattr(row, "source", "audio") == "api":
            meta += " · via API"
        if row.toxic:
            label = f" · {row.top_label}" if getattr(row, "top_label", "") else ""
            st.markdown(f"🔴 {meta} · confidence `{row.confidence:.0%}`{label} · rewrite: *{row.rewrite_method}*")
            a, b = st.columns(2)
            a.error(row.transcript)
            b.success(row.rewritten)
        else:
            st.markdown(f"🟢 {meta} — {row.transcript}")


def render_latency(state: dict, df: pd.DataFrame) -> None:
    stages = state.get("stage_latency", {})
    if not stages:
        st.info("No latency data yet.")
        return
    order = ["diarization", "asr", "toxicity", "rewrite", "tts", "released"]
    rows = [{"stage": s, "p50 (ms)": stages[s]["p50"], "p95 (ms)": stages[s]["p95"], "samples": stages[s]["count"]}
            for s in order if s in stages]
    st.markdown("**Time from end of speech until each stage finished**")
    st.bar_chart(pd.DataFrame(rows).set_index("stage")[["p50 (ms)", "p95 (ms)"]], height=260, stack=False)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    audio = df[df["source"] == "audio"] if "source" in df else df
    if not audio.empty:
        st.markdown("**Verdict latency per utterance**")
        chart = audio[["when", "latency_ms"]].set_index("when")
        chart["budget"] = state.get("latency_budget_ms", 0)
        st.line_chart(chart, height=240)


def render_speakers(state: dict, df: pd.DataFrame) -> None:
    speakers = state.get("speakers", {})
    if not speakers:
        st.info("No speaker data yet.")
        return
    table = pd.DataFrame([
        {
            "Speaker": sid,
            "Utterances": d.get("total_utterances", 0),
            "Toxic": d.get("toxic_count", 0),
            "Toxicity %": round(d.get("toxicity_ratio", 0) * 100, 1),
            "Streak": d.get("consecutive_toxic", 0),
            "Strictness": d.get("rewrite_strictness", 1.0),
        }
        for sid, d in speakers.items()
    ]).sort_values("Toxic", ascending=False)
    st.dataframe(table, hide_index=True, use_container_width=True)

    if not df.empty and len(df) >= 2:
        st.markdown("**Rolling toxicity rate per speaker (last 10 utterances)**")
        trend = (
            df.assign(toxic_i=df["toxic"].astype(int))
            .groupby("speaker", group_keys=False)
            .apply(lambda g: g.assign(rate=g["toxic_i"].rolling(10, min_periods=1).mean() * 100))
        )
        pivot = trend.pivot_table(index="when", columns="speaker", values="rate").ffill()
        st.line_chart(pivot, height=260)


def render_health(state: dict) -> None:
    comps = state.get("health", {}).get("components", {})
    if not comps:
        st.info("Health data is available when the dashboard is connected to a running pipeline.")
        return
    icon = {"ready": "🟢", "degraded": "🟡", "error": "🔴", "starting": "🔵"}
    st.dataframe(pd.DataFrame([
        {
            "Component": name,
            "Status": f"{icon.get(c['status'], '⚪')} {c['status']}",
            "Detail": c.get("last_error") if c["status"] == "error" else (c.get("message") or ""),
            "Errors": c.get("error_count", 0),
        }
        for name, c in comps.items()
    ]), hide_index=True, use_container_width=True)
    depths = state.get("system", {}).get("queue_depths") or {}
    if depths:
        st.markdown("**Agent inbox backlog**")
        st.json(depths)


def render_export(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Nothing to export yet.")
        return
    cols = [c for c in ("time", "speaker", "source", "toxic", "confidence", "top_label", "transcript",
                        "rewritten", "rewrite_method", "action", "latency_ms", "correlation_id") if c in df]
    st.download_button(
        "Download session as CSV",
        df[cols].to_csv(index=False).encode("utf-8"),
        file_name=f"speechshield_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
    st.dataframe(df[cols].iloc[::-1], hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
st.title("🛡️ SpeechShield")
st.caption("Real-time voice moderation · live dashboard")


@st.fragment(run_every=REFRESH_INTERVAL)
def live() -> None:
    state, source = load_state()
    if not state:
        st.warning("⏳ Waiting for the pipeline… start it with `python main.py` "
                   f"(looking for {STATE_URL}).")
        return
    df = feed_frame(state)
    render_header(state, source)
    tab_feed, tab_latency, tab_speakers, tab_health, tab_export = st.tabs(
        ["📝 Live feed", "⏱️ Latency", "👤 Speakers", "🩺 Health", "⬇️ Export"]
    )
    with tab_feed:
        render_live_feed(df)
    with tab_latency:
        render_latency(state, df)
    with tab_speakers:
        render_speakers(state, df)
    with tab_health:
        render_health(state)
    with tab_export:
        render_export(df)


live()
