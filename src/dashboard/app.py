"""VOLTEX — Fidelity SRE Early-Warning Dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.data import ROOT, available_dates, feature_snapshot, forecast_chart_data, load_audit, load_backtest, load_validation, replay_day


PALETTE = {"navy": "#0D1B2A", "card": "#1E3044", "teal": "#00B4D8", "gold": "#F0A500", "red": "#E74C3C", "green": "#2ECC71"}
TIER_COLOR = {"Normal": PALETTE["teal"], "Moderate": PALETTE["gold"], "High": "#E67E22", "Critical": PALETTE["red"]}


st.set_page_config(page_title="VOLTEX — Fidelity SRE", page_icon="⚡", layout="wide")
st.markdown(f"""<style>
.stApp {{background:{PALETTE['navy']}; color:white}} [data-testid="stMetric"] {{background:{PALETTE['card']}; padding:14px; border-radius:9px}}
.alert-panel {{background:{PALETTE['card']}; border-left:5px solid {PALETTE['teal']}; padding:22px; border-radius:9px}}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def cached_validation(): return load_validation()


@st.cache_data(ttl=300)
def cached_replay(date: str): return replay_day(date)


def _alert_row(row: dict) -> dict:
    alert = row.get("alert", {})
    return {"risk_tier": row["market_risk_tier"], "p_critical": row["p_critical"], "p_high": 0.0,
            "vol_zscore": 0.0, "vix_level": 0.0, "stress_breadth": row["stress_breadth"], "alert": alert,
            "guardrails": row.get("guardrails", {}), "agent_path": row.get("agent_path", "fallback"),
            "latency_ms": row.get("latency_ms", 0.0), "forecast_surprise_zscore": row.get("forecast_surprise_zscore", 0.0)}


def render_replay(row: dict) -> None:
    alert = row.get("alert", {}); tier = row["market_risk_tier"]
    features = feature_snapshot(row["date"])
    st.markdown(f"## <span style='color:{TIER_COLOR.get(tier, PALETTE['teal'])}'>{tier}</span> replay — {row['date']}", unsafe_allow_html=True)
    k1,k2,k3,k4=st.columns(4)
    k1.metric("Risk Tier", tier); k2.metric("P(Critical)", f"{row['p_critical']:.1%}")
    k3.metric("Volume Z-Score", f"{float(features.get('volume_zscore_20d', 0)):+.2f}"); k4.metric("VIX Level", f"{float(features.get('vix_level', 0)):.1f}")
    left,right=st.columns(2)
    with left:
        st.subheader("Forecast vs actual volume")
        chart = forecast_chart_data(row["date"])
        if not chart.empty: st.bar_chart(chart)
        else: st.bar_chart(pd.DataFrame({"forecast_surprise_zscore": [row["forecast_surprise_zscore"]]}, index=[row["date"]]))
    with right:
        st.subheader("Top drivers")
        drivers = alert.get("top_drivers", [])
        st.bar_chart(pd.DataFrame({"driver": drivers, "importance": list(range(len(drivers),0,-1))}).set_index("driver") if drivers else pd.DataFrame())
    st.markdown("<div class='alert-panel'>", unsafe_allow_html=True)
    st.subheader("SRE alert")
    st.write(alert.get("plain_english_brief", "No saved alert text."))
    st.info(alert.get("recommended_action", "No action available."))
    st.caption(f"Path: {row.get('agent_path','fallback').upper()} · latency: {row.get('latency_ms',0):.0f} ms")
    checks = row.get("guardrails", {}).get("checks", {})
    if checks:
        st.write({name: ("PASS" if detail["passed"] else f"FAIL — {detail['reason']}") for name, detail in checks.items()})
    else:
        st.write("Guardrails: template fallback or archived result.")
    cited = alert.get("cited_precedents", [])
    if cited: st.caption("Cited precedents: " + "; ".join(cited))
    st.markdown("</div>", unsafe_allow_html=True)
    st.warning("Advisory only — SRE retains full discretion. VOLTEX takes no autonomous action.")


def main() -> None:
    validation, backtest = cached_validation(), load_backtest()
    status = "REPLAY / ARTIFACTS"; latest = backtest.iloc[-1]["date"]
    st.title("VOLTEX — Fidelity SRE Early-Warning Dashboard")
    st.caption(f"{status} · last validated alert: {latest} · guardrails are enforced before display")
    tab_ops, tab_validation, tab_audit = st.tabs(["SRE Alert", "Model Validation", "Audit Log"])
    with tab_ops:
        live_mode = st.toggle("Live mode", value=False, help="Uses live SPY/^VIX when available; replay remains the safe default.")
        if live_mode:
            try:
                from src.data.loader import LiveMarketLoader
                live = LiveMarketLoader().load("2025-12-01")
                st.success(f"Live feed available ({live['date'].max().date()}); select replay mode for the validated alert view.")
            except Exception:
                st.warning("Live feed unavailable — using cached/replay artifacts. The dashboard remains operational.")
        options = available_dates(); default = "2015-08-24" if "2015-08-24" in options else options[0]
        date = st.selectbox("Selected-day replay", options, index=options.index(default))
        if date not in set(backtest["date"].astype(str)):
            st.info("Full replay artifacts exist for the six validated events. Select one below for an end-to-end alert.")
            date = st.selectbox("Validated event", backtest["date"].astype(str).tolist())
        render_replay(cached_replay(date))
    with tab_validation:
        st.subheader("Six-event backtest (artifact-backed)"); st.dataframe(backtest, use_container_width=True)
        st.subheader("Acceptance gates"); st.dataframe(pd.DataFrame(validation["gates"]), use_container_width=True)
        cols=st.columns(2)
        for col,name in zip(cols,["roc_curve.png","pr_curve.png"]):
            with col:
                path=ROOT/"models/eval"/name
                if path.exists(): st.image(str(path), use_container_width=True)
        st.image(str(ROOT/"models/eval/forecast_surprise_crises.png"), use_container_width=True)
    with tab_audit:
        st.subheader("Last 50 audited alerts"); st.dataframe(load_audit(), use_container_width=True)


if __name__ == "__main__": main()
