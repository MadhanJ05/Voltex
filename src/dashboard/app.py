"""VOLTEX — Fidelity SRE Early-Warning Dashboard."""

from __future__ import annotations

import json
import html
from pathlib import Path

import pandas as pd
import streamlit as st

# Stable import is intentionally eager: it makes the saved artifact's class
# resolvable before dashboard live mode asks the loader to unpickle it.
from src.models.anomaly import VoltexAnomalyDetector  # noqa: F401

from src.dashboard.data import ROOT, available_dates, feature_snapshot, forecast_chart_data, live_day_result, load_audit, load_backtest, load_validation, replay_day


PALETTE = {"navy": "#0D1B2A", "card": "#1E3044", "teal": "#00B4D8", "gold": "#F0A500", "red": "#E74C3C", "green": "#2ECC71"}
TIER_COLOR = {"Normal": PALETTE["teal"], "Moderate": PALETTE["gold"], "High": "#E67E22", "Critical": PALETTE["red"]}


st.set_page_config(page_title="VOLTEX — Fidelity SRE", page_icon="⚡", layout="wide")
st.markdown("""<style>
.stApp {background:#F7F8F3; color:#263238}
[data-testid="stHeader"], [data-testid="stToolbar"] {background:transparent}
.v-title {font-size:1.8rem;font-weight:800;color:#263238;margin:5px 0}.v-sub{color:#6C747D;font-size:.9rem;margin-bottom:15px}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:14px 0 20px}.card{background:#fff;border:1px solid #E2E6E8;border-radius:12px;padding:17px 18px;box-shadow:0 1px 2px rgba(20,35,45,.04)}.label{font-size:.77rem;color:#747D86;font-weight:700;margin-bottom:7px}.value{font-size:1.65rem;font-weight:800;color:#27323A;white-space:nowrap}
.critical{background:#FBEEEE;border-color:#F0D0D0}.critical .label,.critical .value{color:#8B2020}.high{background:#FFF4E7;border-color:#F4D8B1}.high .label,.high .value{color:#9A5412}.moderate{background:#FFF8E8;border-color:#F1E0AE}.moderate .label,.moderate .value{color:#886614}.normal{background:#EDF5E8;border-color:#D2E6C8}.normal .label,.normal .value{color:#2E6B2E}
.section-card{background:#fff;border:1px solid #E2E6E8;border-radius:12px;padding:18px;margin:12px 0}.driver{display:grid;grid-template-columns:190px 1fr 35px;gap:10px;align-items:center;margin:10px 0;color:#39444C;font-size:.9rem}.track{height:9px;background:#EDF0F2;border-radius:99px;overflow:hidden}.fill{height:100%;border-radius:99px;background:#E24B4A}.fill.amber{background:#EF9F27}.fill.green{background:#5DCAA5}
.agent{background:#fff;border:1px solid #E2E6E8;border-left:5px solid #E24B4A;border-radius:12px;padding:20px;margin:12px 0}.agent.normal{border-left-color:#5DCAA5}.agent.high{border-left-color:#EF9F27}.agent h3{margin:0 0 8px;color:#8B2020}.agent.normal h3{color:#2E6B2E}.agent.high h3{color:#9A5412}.agent p{color:#44505A;line-height:1.55;margin:0}
.actions{display:grid;grid-template-columns:1fr 290px;gap:14px;margin:12px 0}.action{background:#E8F0FB;border-radius:10px;padding:16px;color:#29496C}.action.normal{background:#F0F1F2;color:#4B555E}.meta{background:#fff;border:1px solid #E2E6E8;border-radius:10px;padding:14px;color:#68737C;font-size:.85rem;line-height:1.65}.advisory{background:#FFF7DE;border:1px solid #F2D998;color:#765A11;border-radius:9px;padding:10px 14px;font-size:.86rem;margin-top:15px}
@media(max-width:800px){.kpis{grid-template-columns:repeat(2,1fr)}.actions{grid-template-columns:1fr}.driver{grid-template-columns:120px 1fr 30px}}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def cached_validation(): return load_validation()


@st.cache_data(ttl=300)
def cached_replay(date: str): return replay_day(date)

@st.cache_data(ttl=300)
def cached_live_result(): return live_day_result()


def _alert_row(row: dict) -> dict:
    alert = row.get("alert", {})
    return {"risk_tier": row["market_risk_tier"], "p_critical": row["p_critical"], "p_high": 0.0,
            "vol_zscore": 0.0, "vix_level": 0.0, "stress_breadth": row["stress_breadth"], "alert": alert,
            "guardrails": row.get("guardrails", {}), "agent_path": row.get("agent_path", "fallback"),
            "latency_ms": row.get("latency_ms", 0.0), "forecast_surprise_zscore": row.get("forecast_surprise_zscore", 0.0)}


def render_replay(row: dict, mode: str = "replay") -> None:
    alert = row.get("alert", {}); tier = row["market_risk_tier"]
    features = feature_snapshot(row["date"])
    vol_z = float(features.get('volume_zscore_20d', row.get('vol_zscore', 0)))
    vix = float(features.get('vix_level', row.get('vix_level', 0)))
    style = tier.lower()
    st.markdown(f"""<div class='v-title'>{html.escape(tier)} {html.escape(mode)} — {html.escape(row['date'])}</div><div class='v-sub'>Fidelity SRE early-warning surface · advisory only</div><div class='kpis'>
      <div class='card {style}'><div class='label'>CURRENT RISK</div><div class='value'>{html.escape(tier)}</div></div>
      <div class='card'><div class='label'>P(CRITICAL)</div><div class='value'>{row['p_critical']:.1%}</div></div>
      <div class='card'><div class='label'>VOLUME Z-SCORE</div><div class='value'>{vol_z:+.2f}</div></div>
      <div class='card'><div class='label'>INTRADAY VOLATILITY (VIX PROXY)</div><div class='value'>{vix:.1f}</div></div></div>""", unsafe_allow_html=True)
    left,right=st.columns(2)
    with left:
        st.markdown("<div class='section-card'><b>Forecast vs actual volume</b>", unsafe_allow_html=True)
        chart = pd.DataFrame(row.get("forecast_chart", [])) if row.get("forecast_chart") else forecast_chart_data(row["date"])
        if not chart.empty: st.line_chart(chart)
        else: st.bar_chart(pd.DataFrame({"forecast_surprise_zscore": [row["forecast_surprise_zscore"]]}, index=[row["date"]]))
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='section-card'><b>Top risk drivers (SHAP)</b>", unsafe_allow_html=True)
        drivers = alert.get("top_drivers", [])
        color = "green" if tier == "Normal" else ""
        rows = "".join(f"<div class='driver'><span>{html.escape(name)}</span><div class='track'><div class='fill {color if i != 1 else 'amber'}' style='width:{max(35,100-i*24)}%'></div></div><span>{len(drivers)-i}</span></div>" for i,name in enumerate(drivers))
        st.markdown(rows or "No drivers available.", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    astyle = "normal" if tier == "Normal" else ("high" if tier in {"High", "Moderate"} else "")
    st.markdown(f"<div class='agent {astyle}'><h3>Agent message</h3><p>{html.escape(alert.get('plain_english_brief', 'No saved alert text.'))}</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='actions'><div class='action {'normal' if tier == 'Normal' else ''}'><b>Recommended action</b><br>{html.escape(alert.get('recommended_action', 'No action available.'))}</div><div class='meta'><b>Path</b> / {html.escape(row.get('agent_path','fallback').upper())}<br><b>Latency</b> / {row.get('latency_ms',0)/1000:.1f}s<br><b>Lead time</b> / {row.get('lead_time_minutes',15)} min</div></div>", unsafe_allow_html=True)
    checks = row.get("guardrails", {}).get("checks", {})
    if checks:
        st.caption(" · ".join(f"{name}: {'PASS' if detail['passed'] else 'FAIL'}" for name, detail in checks.items()))
    else:
        st.caption("Guardrails: template fallback or archived result.")
    cited = alert.get("cited_precedents", [])
    if cited: st.caption("Cited precedents: " + "; ".join(cited))
    st.markdown("<div class='advisory'>Advisory only — SRE retains full discretion. VOLTEX takes no autonomous action.</div>", unsafe_allow_html=True)


def main() -> None:
    validation, backtest = cached_validation(), load_backtest()
    st.title("VOLTEX — Fidelity SRE Early-Warning Dashboard")
    st.caption("Guardrails are enforced before display.")
    tab_ops, tab_validation, tab_audit = st.tabs(["SRE Alert", "Model Validation", "Audit Log"])
    with tab_ops:
        live_mode = st.toggle("Live mode", value=False, help="Uses live SPY/^VIX when available; replay remains the safe default.")
        if live_mode:
            if st.button("Refresh live feed"):
                cached_live_result.clear()
            try:
                row = cached_live_result()
                if row.get("feed_status") == "live":
                    st.success(f"LIVE — {row['date']} · SPY/^VIX feed connected")
                else:
                    st.warning(f"LIVE (cached {row['date']}) — yfinance unavailable; using warmed cache")
                st.caption(f"LIVE — {row['date']}")
                render_replay(row, mode="live")
            except Exception as exc:
                print(f"[VOLTEX LIVE] branch=failure reason={type(exc).__name__}: {exc}", flush=True)
                st.warning("LIVE FEED FAILED — using cached/replay artifacts.")
                render_replay(cached_replay("2015-08-24"))
        else:
            options = available_dates(); default = "2015-08-24" if "2015-08-24" in options else options[0]
            date = st.selectbox("Selected-day replay", options, index=options.index(default))
            if date not in set(backtest["date"].astype(str)):
                st.info("Full replay artifacts exist for the six validated events. Select one below for an end-to-end alert.")
                date = st.selectbox("Validated event", backtest["date"].astype(str).tolist())
            st.caption(f"REPLAY — {date}")
            render_replay(cached_replay(date))
    with tab_validation:
        st.subheader("Six-event backtest (artifact-backed)"); st.dataframe(backtest, use_container_width=True)
        st.subheader("Acceptance gates"); st.dataframe(pd.DataFrame(validation["gates"]), use_container_width=True)
        cols=st.columns(2)
        for col,name in zip(cols,["roc_curve.png","pr_curve.png"]):
            with col:
                path=ROOT/"models/eval"/name
                if path.exists(): st.image(str(path), use_column_width=True)
        st.image(str(ROOT/"models/eval/forecast_surprise_crises.png"), use_column_width=True)
        details=st.columns(2)
        for col,name in zip(details,["confusion_matrix.png","calibration_curve.png"]):
            with col:
                path=ROOT/"models/eval"/name
                if path.exists(): st.image(str(path), use_column_width=True)
    with tab_audit:
        st.subheader("Last 50 audited alerts"); st.dataframe(load_audit(), use_container_width=True)


if __name__ == "__main__": main()
