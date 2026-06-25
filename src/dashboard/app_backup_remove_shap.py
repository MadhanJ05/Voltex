"""VOLTEX — Fidelity SRE Early-Warning Dashboard."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
/* === VOLTEX DARK MODE — TRUE BLACK === */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"]{background:#0a0a0a!important;color:#e5e5e5!important}
[data-testid="stHeader"], [data-testid="stToolbar"]{background:transparent!important}
[data-testid="stSidebar"], [data-testid="stSidebar"]>div{background:#111111!important}
.stApp p,.stApp label{color:#e5e5e5!important}.stApp h1,.stApp h2,.stApp h3,.stApp h4{color:#f5f5f5!important}
.stCaption,[data-testid="stCaptionContainer"]{color:#737373!important}
[data-testid="stMetric"],[data-testid="stMetricValue"]{background:#171717!important;color:#e5e5e5!important}[data-testid="stMetricLabel"]{color:#a3a3a3!important}
.stSelectbox>div>div,.stRadio>div,.stTextInput>div>div>input{background:#171717!important;color:#e5e5e5!important;border-color:rgba(255,255,255,.1)!important}
.stTabs [data-baseweb="tab-list"]{background:#0a0a0a!important;border-bottom:1px solid rgba(255,255,255,.08)!important}.stTabs [data-baseweb="tab"]{color:#a3a3a3!important}.stTabs [aria-selected="true"]{color:#22c55e!important;border-bottom-color:#22c55e!important}
[data-testid="stExpander"]{background:#171717!important;border-color:rgba(255,255,255,.08)!important}.stDataFrame,[data-testid="stDataFrame"]{background:#171717!important}.stDataFrame th{background:#262626!important;color:#e5e5e5!important}.stDataFrame td{background:#171717!important;color:#d4d4d4!important}
.stImage img{border-radius:8px;border:.5px solid rgba(255,255,255,.08)}hr,.stApp hr{border-color:rgba(255,255,255,.08)!important}
.stButton>button{background:#171717!important;color:#e5e5e5!important;border:.5px solid rgba(255,255,255,.15)!important}.stButton>button:hover{background:#262626!important;border-color:rgba(255,255,255,.25)!important}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0a0a0a}::-webkit-scrollbar-thumb{background:#333;border-radius:3px}.stApp{transition:none!important}
.v-title{font-size:1.8rem;font-weight:800;color:#f5f5f5;margin:5px 0}.v-sub{color:#a3a3a3;font-size:.9rem;margin-bottom:15px}
.kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:14px 0 20px}.card{background:#171717;border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:16px 20px;box-shadow:0 1px 3px rgba(0,0,0,.4);min-width:0}.label{font-size:12px;color:#a3a3a3;font-weight:600;letter-spacing:.04em;text-transform:uppercase;margin-bottom:6px}.value{font-size:28px;font-weight:700;color:#e5e5e5;white-space:nowrap;overflow:visible}.sub{font-size:11px;color:#737373;margin-top:6px;line-height:1.3}
.critical{background:rgba(239,68,68,.15);border-color:rgba(239,68,68,.3)}.critical .label{color:#fca5a5}.critical .value{color:#fecaca}.high{background:rgba(249,115,22,.14);border-color:rgba(249,115,22,.3)}.high .label{color:#fdba74}.high .value{color:#fed7aa}.moderate{background:rgba(245,158,11,.13);border-color:rgba(245,158,11,.28)}.moderate .label{color:#fcd34d}.moderate .value{color:#fde68a}.normal{background:rgba(34,197,94,.12);border-color:rgba(34,197,94,.28)}.normal .label{color:#86efac}.normal .value{color:#bbf7d0}
.section-card{background:#171717;border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:18px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,.35)}.driver{display:grid;grid-template-columns:190px 1fr 35px;gap:10px;align-items:center;margin:10px 0;color:#d4d4d4;font-size:.9rem}.track{height:9px;background:#2a2a2a;border-radius:99px;overflow:hidden}.fill{height:100%;border-radius:99px;background:#ef4444}.fill.amber{background:#f59e0b}.fill.green{background:#22c55e}.driver-legend{font-size:.72rem;color:#737373;margin:12px 0 0}
.agent{background:#171717;border:1px solid rgba(255,255,255,.08);border-left:5px solid #ef4444;border-radius:12px;padding:20px;margin:12px 0}.agent.normal{border-left-color:#22c55e}.agent.moderate{border-left-color:#f59e0b}.agent.high{border-left-color:#f97316}.agent h3{margin:0 0 8px;color:#ef4444}.agent.normal h3{color:#22c55e}.agent.moderate h3{color:#f59e0b}.agent.high h3{color:#f97316}.agent p{color:#d4d4d4;line-height:1.55;margin:0}.message-id{font-size:.75rem;color:#737373;font-weight:700;margin-bottom:6px}.agent-meta{font-size:.8rem;color:#737373;margin-top:14px;line-height:1.6}.precedents{margin-top:12px;font-size:.84rem;color:#a3a3a3}.precedents ul{margin:5px 0 0;padding-left:20px}
.impact-card{background:#171717;border:.5px solid rgba(255,255,255,.08);border-radius:12px;padding:18px 20px;margin:2px 0 12px;border-left-width:4px;box-shadow:0 1px 3px rgba(0,0,0,.35)}.impact-header{display:flex;gap:9px;align-items:center;margin-bottom:14px}.impact-badge{font-size:.74rem;font-weight:800;padding:4px 8px;border-radius:999px}.impact-event{font-size:.95rem;font-weight:800;color:#e5e5e5}.impact-date{margin-left:auto;color:#737373;font-size:.78rem}.impact-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:9px}.impact-tile{background:#1f1f1f;border:.5px solid rgba(255,255,255,.06);border-radius:8px;padding:10px}.impact-tile-label{font-size:.69rem;color:#a3a3a3;font-weight:700;text-transform:uppercase}.impact-tile-value{font-size:1.25rem;font-weight:800;color:#e5e5e5;margin-top:3px}.impact-bottom{display:grid;grid-template-columns:1.25fr .75fr;gap:20px;margin-top:16px}.impact-section-title{font-size:.72rem;font-weight:800;color:#a3a3a3;text-transform:uppercase;margin-bottom:8px}.impact-driver{display:grid;grid-template-columns:145px 1fr 36px;gap:7px;align-items:center;margin:7px 0;font-size:.74rem;color:#d4d4d4}.impact-track{height:8px;background:#2a2a2a;border-radius:99px;overflow:hidden}.impact-fill{height:100%;border-radius:99px}.impact-signal{display:flex;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,.08);padding:5px 0;font-size:.78rem;color:#a3a3a3}.impact-signal b{color:#e5e5e5}@media(max-width:800px){.impact-bottom{grid-template-columns:1fr}.impact-driver{grid-template-columns:110px 1fr 32px}}
.signal-gauge{background:#111111;border:.5px solid rgba(255,255,255,.08);border-radius:12px;padding:24px 28px;margin:0 0 16px;box-shadow:0 1px 3px rgba(0,0,0,.4)}.signal-head{display:flex;align-items:center;gap:10px;margin-bottom:14px}.signal-pill{font-size:13px;font-weight:700;padding:4px 14px;border-radius:20px}.signal-title{font-size:1.02rem;font-weight:800;color:#e5e5e5}.signal-date{margin-left:auto;color:#737373;font-size:.82rem}.gauge-row{display:grid;grid-template-columns:140px 1fr 70px;gap:10px;align-items:center;margin:9px 0}.gauge-name{font-size:.76rem;color:#a3a3a3;text-align:right}.gauge-value{font-family:'SF Mono','Menlo','Consolas',monospace;font-size:13px;font-weight:600;color:#e5e5e5;text-align:right;min-width:70px}.gauge-track{position:relative;height:22px;background:#2a2a2a;border-radius:999px;overflow:hidden;border:1px solid rgba(255,255,255,.08)}.gauge-danger{position:absolute;top:0;height:100%;background:rgba(239,68,68,.3)}.gauge-dot{position:absolute;top:50%;width:16px;height:16px;border-radius:999px;border:2.5px solid #0a0a0a;transform:translate(-50%,-50%);box-shadow:0 1px 4px rgba(0,0,0,.5)}.verdict{margin-top:14px;border:1px solid rgba(255,255,255,.08);border-radius:6px;padding:12px 16px;background:#171717;color:#d4d4d4;font-size:.9rem}.verdict-arrow{font-size:1.05rem;margin-right:8px}.signal-foot{font-size:.72rem;color:#737373;margin-top:10px}.signal-legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:10px;color:#737373;font-size:.72rem}.legend-dot{display:inline-block;width:10px;height:10px;border-radius:999px;margin-right:5px;vertical-align:-1px}.legend-danger{display:inline-block;width:20px;height:12px;border-radius:3px;background:rgba(239,68,68,.3);margin-right:4px;vertical-align:middle}.meter-metrics{display:flex;gap:10px;flex-wrap:wrap;margin-top:4px}.meter-box{background:#171717;border:.5px solid rgba(255,255,255,.08);border-radius:8px;padding:8px 12px;min-width:120px;flex:1}.meter-label{font-size:10px;color:#737373;text-transform:uppercase;letter-spacing:.04em}.meter-value{font-size:14px;font-weight:500;font-family:'SF Mono','Menlo','Consolas',monospace;margin-top:2px}.waterfall-legend{display:flex;gap:15px;flex-wrap:wrap;font-size:11px;color:#737373;margin:2px 0 5px}.legend-square{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
.actions{display:grid;grid-template-columns:1fr 290px;gap:14px;margin:12px 0}.action{background:rgba(239,68,68,.08);border:.5px solid rgba(239,68,68,.2);border-radius:10px;padding:16px;color:#e5e5e5}.action.normal{background:rgba(34,197,94,.08);border:.5px solid rgba(34,197,94,.2);color:#e5e5e5}.meta{background:#171717;border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:14px;color:#a3a3a3;font-size:.85rem;line-height:1.65}.advisory{background:rgba(234,179,8,.08);border:.5px solid rgba(234,179,8,.2);color:#d4d4d4;border-radius:9px;padding:10px 14px;font-size:.86rem;margin-top:15px}
@media(max-width:800px){.kpis{grid-template-columns:repeat(2,1fr)}.actions{grid-template-columns:1fr}.driver{grid-template-columns:120px 1fr 30px}}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def cached_validation(): return load_validation()


@st.cache_data(ttl=300)
def cached_replay(date: str): return replay_day(date)

@st.cache_data(ttl=300)
def cached_live_result(): return live_day_result()


@st.cache_resource
def dashboard_classifier():
    """Load the saved classifier once for display-only SHAP magnitudes."""
    from src.models.classifier import load_classifier

    return load_classifier(ROOT / "models")


@st.cache_data(ttl=600)
def replay_shap_values(date: str) -> list[tuple[str, float]]:
    """Return actual SHAP contributions for the day’s most stressed ticker.

    The saved evaluation alert retains driver names but not their numeric SHAP
    values. This reads the same saved ticker feature artifact and classifier to
    render the existing chart faithfully; it does not affect inference.
    """
    try:
        feature_path = ROOT / "data/processed/ticker_features.csv"
        frame = pd.read_csv(feature_path)
        same_day = frame.loc[frame["date"].astype(str).eq(date)]
        if same_day.empty:
            return []
        classifier = dashboard_classifier()
        probabilities = classifier.predict_tickers(same_day, include_shap=False)
        stressed_index = (probabilities["p_critical"] + probabilities["p_high"]).idxmax()
        return classifier.explain_prediction(same_day.loc[[stressed_index]])
    except Exception as exc:  # Dashboard remains displayable if an optional artifact is absent.
        print(f"[VOLTEX DASHBOARD] SHAP display unavailable for {date}: {type(exc).__name__}: {exc}", flush=True)
        return []


@st.cache_data(ttl=600)
def replay_feature_driver_values(date: str) -> dict:
    """Return replay classifier features plus all Critical-class SHAP values."""
    from src.data.features import FEATURE_COLUMNS

    feature_path = ROOT / "data/processed/ticker_features.csv"
    frame = pd.read_csv(feature_path)
    same_day = frame.loc[frame["date"].astype(str).eq(date)]
    if same_day.empty:
        snapshot = feature_snapshot(date)
        return {
            "date": date,
            "features": {column: float(snapshot.get(column, 0.0)) for column in FEATURE_COLUMNS},
            "critical_shap": [(column, 0.0) for column in FEATURE_COLUMNS],
        }
    classifier = dashboard_classifier()
    probabilities = classifier.predict_tickers(same_day, include_shap=False)
    stressed_index = (probabilities["p_critical"] + probabilities["p_high"]).idxmax()
    ticker_row = same_day.loc[[stressed_index]]
    values = np.asarray(classifier._shap_values(ticker_row))
    if values.ndim == 3:
        class_values = values[0, :, 3]
    elif values.ndim == 2:
        class_values = values[0]
    else:
        class_values = np.zeros(len(FEATURE_COLUMNS))
    feature_columns = list(classifier.feature_columns)
    return {
        "date": date,
        "features": {column: float(ticker_row.iloc[0][column]) for column in feature_columns},
        "critical_shap": [(column, float(value)) for column, value in zip(feature_columns, class_values)],
    }


@st.cache_data(ttl=300)
def live_shap_values() -> list[tuple[str, float]]:
    """Rehydrate current cached-live SHAP values for the display bar chart only."""
    try:
        payload = live_feature_driver_values()
        values = payload.get("critical_shap", [])
        return sorted(values, key=lambda item: abs(float(item[1])), reverse=True)[:3]
    except Exception as exc:  # A Normal alert should remain visible if this optional display value is unavailable.
        print(f"[VOLTEX DASHBOARD] live SHAP display unavailable: {type(exc).__name__}: {exc}", flush=True)
        return []


@st.cache_data(ttl=300)
def live_feature_driver_values() -> dict:
    """Return the current live classifier features plus all Critical-class SHAP values."""
    from src.data.features import FEATURE_COLUMNS, engineer_features
    from src.data.loader import LiveMarketLoader

    daily, _ = LiveMarketLoader(ROOT / "data/cache/live_market.csv").load_with_status(
        period="5d", prefer_fresh_cache=True
    )
    feature = engineer_features(daily).iloc[-1]
    ticker_row = pd.DataFrame([{
        **{column: feature[column] for column in FEATURE_COLUMNS},
        "ticker": "SPY", "date": feature["date"],
    }])
    classifier = dashboard_classifier()
    values = np.asarray(classifier._shap_values(ticker_row))
    if values.ndim == 3:
        class_values = values[0, :, 3]
    elif values.ndim == 2:
        class_values = values[0]
    else:
        class_values = np.zeros(len(FEATURE_COLUMNS))
    return {
        "date": pd.Timestamp(feature["date"]).strftime("%Y-%m-%d"),
        "features": {column: float(feature[column]) for column in FEATURE_COLUMNS},
        "critical_shap": [(column, float(value)) for column, value in zip(FEATURE_COLUMNS, class_values)],
    }


@st.cache_data(ttl=300)
def live_forecast_components(current_date: str) -> dict:
    """Return display-only Prophet/ARIMA/ensemble forecast components for the live meter."""
    from src.models.forecaster import VolumeForecaster

    cache = pd.read_csv(ROOT / "data/cache/live_market.csv", parse_dates=["date"])
    current = pd.Timestamp(current_date)
    cache = cache.loc[cache["date"] <= current].copy()
    if cache.empty:
        return {}
    for flag in ("fomc_flag", "cpi_flag", "nfp_flag"):
        cache[flag] = 0
    target = cache.iloc[[-1]].copy()
    history = cache.loc[cache["date"] < target["date"].iloc[0], ["date", "total_volume", "fomc_flag", "cpi_flag", "nfp_flag"]].tail(90)
    if len(history) < 30:
        return {}
    weights = json.loads((ROOT / "models/metrics.json").read_text())["forecaster"]["ensemble_weights"]
    forecaster = VolumeForecaster(ensemble_weights=weights, arima_order=(2, 0, 0))
    prophet, arima, order = forecaster._fit_models(history, select_order=False)
    forecaster.arima_order = order
    raw = forecaster._predict_with_models(prophet, arima, target[["date", "total_volume", "fomc_flag", "cpi_flag", "nfp_flag"]])
    combined = forecaster._ensemble(raw, weights).iloc[0]
    return {
        "forecast_volume": float(combined["forecast"]),
        "prophet_forecast": float(combined["prophet"]),
        "arima_forecast": float(combined["arima"]),
    }


def _volume_billions(value: float) -> str:
    return f"{value / 1e9:.2f}B"


def format_volume(vol: float | None) -> str:
    """Format volume in compact units."""
    if vol is None or pd.isna(vol):
        return "N/A"
    value = float(vol)
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.0f}M"
    return f"{value:,.0f}"


def render_event_impact_card(row: dict, chart: pd.DataFrame, shap_values: list[tuple[str, float]], vol_z: float, vix: float) -> None:
    """Render the replay-only event summary using the saved pipeline outputs."""
    tier = row["market_risk_tier"]
    presentation = {
        "Normal": ("#22c55e", "#dcfce7", "#15803d"),
        "Moderate": ("#f59e0b", "#fef3c7", "#b45309"),
        "High": ("#f97316", "#ffedd5", "#c2410c"),
        "Critical": ("#ef4444", "#fee2e2", "#b91c1c"),
    }[tier]
    border, badge_background, badge_text = presentation
    event_name = str(row.get("event") or "Historical replay")
    event_day = chart.iloc[-1]
    surprise = float(row.get("forecast_surprise_zscore", 0.0))
    anomaly = float(row.get("anomaly_score", 0.0))
    maximum = max((abs(float(value)) for _, value in shap_values), default=0.0)
    driver_html = "".join(
        f"<div class='impact-driver'><span>{html.escape(str(name))}</span>"
        f"<div class='impact-track'><div class='impact-fill' style='width:{(abs(float(value)) / maximum * 100 if maximum else 0):.1f}%;background:{'#22c55e' if float(value) < 0 else '#ef4444'}'></div></div>"
        f"<span style='text-align:right'>{abs(float(value)):.2f}</span></div>"
        for name, value in shap_values
    ) or "<span style='color:#737373;font-size:.78rem'>No SHAP drivers available.</span>"
    surprise_style = "color:#ef4444" if abs(surprise) >= 2 else ""
    anomaly_style = "color:#ef4444" if anomaly >= .5 else ""
    card = f"""
    <div class='impact-card' style='border-left-color:{border}'>
      <div class='impact-header'><span class='impact-badge' style='background:{badge_background};color:{badge_text}'>{html.escape(tier)}</span><span class='impact-event'>{html.escape(event_name)}</span><span class='impact-date'>{html.escape(str(row['date']))}</span></div>
      <div class='impact-metrics'>
        <div class='impact-tile'><div class='impact-tile-label'>Actual volume</div><div class='impact-tile-value'>{_volume_billions(float(event_day['actual']))}</div></div>
        <div class='impact-tile'><div class='impact-tile-label'>Forecast volume</div><div class='impact-tile-value'>{_volume_billions(float(event_day['forecast']))}</div></div>
        <div class='impact-tile'><div class='impact-tile-label'>Forecast surprise</div><div class='impact-tile-value' style='{surprise_style}'>{surprise:+.2f}σ</div></div>
        <div class='impact-tile'><div class='impact-tile-label'>Anomaly score</div><div class='impact-tile-value' style='{anomaly_style}'>{anomaly:.2f}</div></div>
      </div>
      <div class='impact-bottom'>
        <div><div class='impact-section-title'>Top risk drivers (SHAP)</div>{driver_html}</div>
        <div><div class='impact-section-title'>System signals</div>
          <div class='impact-signal'><span>P(Critical)</span><b>{float(row['p_critical']):.1%}</b></div>
          <div class='impact-signal'><span>Volume z-score</span><b>{vol_z:+.2f}</b></div>
          <div class='impact-signal'><span>Intraday volatility</span><b>{vix:.1f}</b></div>
        </div>
      </div>
    </div>"""
    st.markdown(card, unsafe_allow_html=True)


def driver_fill_class(value: float, maximum: float) -> str:
    """Map saved Critical-class SHAP contribution to the existing risk palette."""
    if value < 0:
        return "green"
    if maximum and abs(value) < maximum * .5:
        return "amber"
    return ""


FEATURE_DISPLAY_NAMES = {
    "intraday_vol_pct": "Intraday volatility",
    "vix_level": "VIX level",
    "return_zscore_20d": "Return z-score (20d)",
    "return_std_20d": "Return std dev (20d)",
    "volume_zscore_20d": "Volume z-score (20d)",
    "ma_ratio_5_20": "MA ratio (5/20)",
    "volume_acceleration": "Volume acceleration",
    "market_breadth": "Market breadth",
    "cpi_flag": "CPI release",
    "fomc_flag": "FOMC meeting",
    "nfp_flag": "NFP release",
    "day_of_week": "Day of week",
}

BINARY_FLAGS = {"cpi_flag", "fomc_flag", "nfp_flag"}
DAY_NAMES = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}


def _feature_specs(features: dict) -> dict:
    """Presentation ranges for the actual 12 classifier features."""
    breadth_max = 1.0 if float(features.get("market_breadth", 0.0)) <= 1.0 else 100.0
    return {
        "intraday_vol_pct": {"label": "intraday_vol_pct", "min": 0.0, "max": 10.0, "danger": [(70, 100)], "fmt": "{:.2f}%"},
        "return_zscore_20d": {"label": "return_zscore_20d", "min": -5.0, "max": 5.0, "danger": [(0, 30), (70, 100)], "fmt": "{:+.2f}"},
        "nfp_flag": {"label": "nfp_flag", "min": 0.0, "max": 1.0, "danger": [(50, 100)], "fmt": "{:.0f}"},
        "volume_zscore_20d": {"label": "volume_zscore_20d", "min": -3.0, "max": 6.0, "danger": [(56, 100)], "fmt": "{:+.2f}"},
        "market_breadth": {"label": "market_breadth", "min": 0.0, "max": breadth_max, "danger": [(0, 35)], "fmt": "{:.0%}" if breadth_max == 1.0 else "{:.1f}%"},
        "vix_level": {"label": "vix_level", "min": 10.0, "max": 80.0, "danger": [(29, 100)], "fmt": "{:.1f}"},
        "volume_acceleration": {"label": "volume_acceleration", "min": -0.5, "max": 1.5, "danger": [(50, 100)], "fmt": "{:+.2f}"},
        "ma_ratio_5_20": {"label": "ma_ratio_5_20", "min": 0.90, "max": 1.10, "danger": [(0, 25), (75, 100)], "fmt": "{:.3f}"},
        "return_std_20d": {"label": "return_std_20d", "min": 0.0, "max": 0.05, "danger": [(60, 100)], "fmt": "{:.2%}"},
        "fomc_flag": {"label": "fomc_flag", "min": 0.0, "max": 1.0, "danger": [(50, 100)], "fmt": "{:.0f}"},
        "cpi_flag": {"label": "cpi_flag", "min": 0.0, "max": 1.0, "danger": [(50, 100)], "fmt": "{:.0f}"},
        "day_of_week": {"label": "day_of_week", "min": 0.0, "max": 4.0, "danger": [], "fmt": "{:.0f}"},
    }


def _format_feature_value(name: str, value: float, spec: dict) -> str:
    if name in BINARY_FLAGS:
        return "Active" if int(round(float(value))) == 1 else "Inactive"
    if name == "day_of_week":
        return DAY_NAMES.get(int(round(float(value))), str(int(round(float(value)))))
    if name == "market_breadth" and spec["max"] == 1.0:
        return spec["fmt"].format(value)
    return spec["fmt"].format(value)


def _format_feature_value_html(name: str, value: float, spec: dict) -> str:
    formatted = _format_feature_value(name, value, spec)
    if name in BINARY_FLAGS:
        if int(round(float(value))) == 1:
            return '<span style="color:#f59e0b;font-weight:600;">Active</span>'
        return '<span style="color:#737373;">Inactive</span>'
    if name == "day_of_week":
        return f'<span style="color:#a3a3a3;">{html.escape(formatted)}</span>'
    return html.escape(formatted)


def _dot_color(name: str, value: float, shap_value: float) -> str:
    if name == "day_of_week" or (name.endswith("_flag") and value <= 0.0):
        return "#B4B2A9"
    magnitude = abs(float(shap_value))
    if shap_value > 0 and magnitude > 1.0:
        return "#E24B4A"
    if shap_value > 0 and magnitude > 0.3:
        return "#EF9F27"
    return "#5DCAA5"


def _bar_color(shap_value: float) -> str:
    magnitude = abs(float(shap_value))
    if magnitude < 0.1:
        return "#9ca3af"
    if shap_value < 0:
        return "#22c55e"
    if magnitude <= 1.0:
        return "#f59e0b"
    return "#ef4444"


def _anomaly_subtitle(score: float) -> tuple[str, str]:
    if score < 0.3:
        return "Isolation Forest outlier score · well within normal (< 0.3)", "#737373"
    if score <= 0.5:
        return "Isolation Forest outlier score · elevated (0.3–0.5 range)", "#f59e0b"
    return "Isolation Forest outlier score · unusual (> 0.5)", "#ef4444"


def _surprise_zone(zscore: float) -> tuple[str, str]:
    """Return (color, verdict) based on forecast surprise z-score.

    Positive = volume surge (dangerous) → green/amber/red.
    Negative = volume shortfall (calm) → green/blue.
    """
    value = float(zscore)
    if value >= 0:
        if value < 1:
            return "#22c55e", "Volume expected within normal range — no pre-scaling needed"
        if value < 2:
            return "#f59e0b", "Mild volume surge — increased monitoring recommended"
        if value < 3:
            return "#f59e0b", "Significant volume surge — consider pre-scaling infrastructure"
        return "#ef4444", "Extreme volume surge — pre-scale and activate incident readiness"
    if value > -1:
        return "#22c55e", "Volume expected within normal range — no pre-scaling needed"
    if value > -2:
        return "#3b82f6", "Volume below forecast — quieter than expected session"
    return "#3b82f6", "Volume well below forecast — unusually calm session"


def _meter_angle(zscore: float) -> float:
    """Map z-score to angle on a semicircular arc (180° to 360°).

    180° = left end (-3σ), 270° = top center (0σ), 360° = right end (+6σ).
    """
    value = float(zscore)
    if value <= 0:
        angle = 270 + (value * 30.0)
    else:
        angle = 270 + (value * 15.0)
    return max(180.0, min(360.0, angle))


def _polar_point(angle: float, radius: float, cx: float = 160.0, cy: float = 160.0) -> tuple[float, float]:
    """Convert angle in degrees to SVG coordinates for an upward-opening semicircle."""
    radians = math.radians(angle)
    return cx + radius * math.cos(radians), cy + radius * math.sin(radians)


def _arc_path(start_angle: float, end_angle: float, radius: float = 112.0) -> str:
    """Build an SVG arc path from start_angle to end_angle."""
    start = _polar_point(start_angle, radius)
    end = _polar_point(end_angle, radius)
    large = 1 if abs(end_angle - start_angle) > 180 else 0
    return f"M {start[0]:.2f} {start[1]:.2f} A {radius:.2f} {radius:.2f} 0 {large} 1 {end[0]:.2f} {end[1]:.2f}"


def build_deviation_meter(
    surprise_zscore: float,
    forecast_volume: float | None = None,
    actual_volume: float | None = None,
    prophet_forecast: float | None = None,
    arima_forecast: float | None = None,
    mode: str = "live",
) -> str:
    """Return the full deviation meter HTML string."""
    zscore = float(surprise_zscore or 0.0)
    zone_color, verdict = _surprise_zone(zscore)
    angle = _meter_angle(zscore)
    needle_x, needle_y = _polar_point(angle, 100.0)
    tick_positions = {-3: 180, -1: 240, 0: 270, 1: 285, 3: 315, 6: 360}
    tick_colors = {-3: "#3b82f6", -1: "#3b82f6", 0: "#737373", 1: "#737373", 3: "#f59e0b", 6: "#ef4444"}
    ticks = "".join(
        f"<text x='{_polar_point(angle_value, 134)[0]:.1f}' y='{_polar_point(angle_value, 134)[1]:.1f}' "
        f"text-anchor='middle' dominant-baseline='middle' "
        f"style='font-size:10px;font-family:monospace;fill:{tick_colors.get(label, '#737373')}'>{label:+d}σ</text>"
        for label, angle_value in tick_positions.items()
    )
    zones = [
        (180.0, 210.0, "rgba(59,130,246,0.35)"),
        (210.0, 240.0, "rgba(59,130,246,0.20)"),
        (240.0, 285.0, "rgba(34,197,94,0.35)"),
        (285.0, 315.0, "rgba(245,158,11,0.35)"),
        (315.0, 360.0, "rgba(239,68,68,0.35)"),
    ]
    zone_paths = "".join(
        f"<path d='{_arc_path(start, end)}' fill='none' stroke='{color}' stroke-width='18' stroke-linecap='round'/>"
        for start, end, color in zones
    )
    if mode == "replay":
        actual_color = zone_color if abs(zscore) >= 1 else "#22c55e"
        metrics = (
            "<div class='meter-metrics'>"
            f"<div class='meter-box'><div class='meter-label'>Forecast</div><div class='meter-value' style='color:#3b82f6'>{format_volume(forecast_volume)}</div></div>"
            f"<div class='meter-box'><div class='meter-label'>Actual</div><div class='meter-value' style='color:{actual_color}'>{format_volume(actual_volume)}</div></div>"
            f"<div class='meter-box'><div class='meter-label'>Surprise</div><div class='meter-value' style='color:{zone_color}'>{zscore:+.2f}σ</div></div>"
            "</div>"
        )
        footer = "Ensemble forecast vs actual outcome for the event day."
    else:
        metrics = (
            "<div class='meter-metrics'>"
            f"<div class='meter-box'><div class='meter-label'>Prophet</div><div class='meter-value' style='color:#8b5cf6'>{format_volume(prophet_forecast)}</div></div>"
            f"<div class='meter-box'><div class='meter-label'>ARIMA</div><div class='meter-value' style='color:#06b6d4'>{format_volume(arima_forecast)}</div></div>"
            f"<div class='meter-box'><div class='meter-label'>Ensemble</div><div class='meter-value' style='color:#3b82f6'>{format_volume(forecast_volume)}</div></div>"
            "</div>"
        )
        footer = "Ensemble forecast from Prophet + ARIMA models. Surprise computed from yesterday's close."
    return f"""
    <style>
      .meter-metrics {{
        display: flex;
        gap: 12px;
        margin-top: 12px;
      }}
      .meter-box {{
        flex: 1;
        background: #171717;
        border: .5px solid rgba(255,255,255,.08);
        border-radius: 8px;
        padding: 10px 12px;
      }}
      .meter-label {{
        font-size: 10px;
        color: #737373;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: .04em;
      }}
      .meter-value {{
        font-size: 14px;
        font-weight: 500;
        font-family: monospace;
        color: #e5e5e5;
      }}
    </style>
    <div style='background:#111111;border:.5px solid rgba(255,255,255,.08);border-radius:12px;padding:24px 28px;box-shadow:0 1px 3px rgba(0,0,0,.4);margin:16px 0'>
      <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:6px'>
        <div style='font-size:13px;font-weight:500;color:#e5e5e5'>Volume forecast ensemble</div>
        <div style='font-size:11px;color:#737373'>Prophet + ARIMA</div>
      </div>
      <div style='display:flex;justify-content:center'>
        <svg viewBox='0 0 320 200' width='100%' style='max-width:520px;height:auto'>
          <path d='{_arc_path(180, 360)}' fill='none' stroke='#2a2a2a' stroke-width='18' stroke-linecap='round'/>
          {zone_paths}
          {ticks}
          <line x1='160' y1='160' x2='{needle_x:.1f}' y2='{needle_y:.1f}' stroke='{zone_color}' stroke-width='3' stroke-linecap='round'/>
          <circle cx='{needle_x:.1f}' cy='{needle_y:.1f}' r='5' fill='{zone_color}'/>
          <circle cx='160' cy='160' r='7' fill='{zone_color}'/>
          <text x='160' y='140' text-anchor='middle' style='font-size:22px;font-weight:500;font-family:monospace;fill:{zone_color}'>{zscore:+.2f}σ</text>
          <text x='160' y='156' text-anchor='middle' style='font-size:10px;fill:#737373'>forecast surprise</text>
        </svg>
      </div>
      {metrics}
      <div style='background:#171717;border-left:3px solid {zone_color};padding:8px 12px;margin-top:10px;font-size:11px;color:#a3a3a3'>{html.escape(verdict)}</div>
      <div style='font-size:11px;color:#737373;margin-top:10px'>{html.escape(footer)}</div>
    </div>"""


def build_signal_gauge_panel(
    feature_values: dict,
    shap_values: dict,
    tier: str,
    p_critical: float,
    date_label: str,
    event_name: str | None = None,
) -> str:
    """Return the full signal gauge panel HTML string."""
    presentation = {
        "Normal": ("#22c55e", "#dcfce7", "#15803d", "All signals within safe zones"),
        "Moderate": ("#f59e0b", "#fef3c7", "#b45309", "Some signals elevated — increased monitoring"),
        "High": ("#f97316", "#ffedd5", "#c2410c", "Multiple signals in stress zone — pre-scale recommended"),
        "Critical": ("#ef4444", "#fee2e2", "#b91c1c", "Critical stress detected — full incident protocol"),
    }[tier]
    tier_color, badge_background, badge_text, summary = presentation
    shap_map = {name: float(value) for name, value in shap_values.items()}
    specs = _feature_specs(feature_values)
    dot_colors = {"#5DCAA5": "#22c55e", "#EF9F27": "#f59e0b", "#E24B4A": "#ef4444", "#B4B2A9": "#9ca3af"}
    groups = [
        ("VOLATILITY", ["intraday_vol_pct", "return_zscore_20d", "return_std_20d", "vix_level"]),
        ("VOLUME", ["volume_zscore_20d", "volume_acceleration", "ma_ratio_5_20"]),
        ("BREADTH & MACRO", ["market_breadth", "nfp_flag", "fomc_flag", "cpi_flag"]),
        ("CALENDAR", ["day_of_week"]),
    ]
    rows = []
    for group_index, (label, names) in enumerate(groups):
        if group_index:
            rows.append("<div style='border-top:1px solid rgba(255,255,255,0.08);margin:12px 0 8px'></div>")
        rows.append(
            "<div style='font-size:11px;font-weight:700;color:#a3a3a3;letter-spacing:0.05em;"
            "text-transform:uppercase;margin:14px 0 6px 0;padding-left:4px'>"
            f"{html.escape(label)}</div>"
        )
        ordered_names = sorted(
            [name for name in names if name in specs],
            key=lambda item: abs(shap_map.get(item, 0.0)),
            reverse=True,
        )
        for name in ordered_names:
            value = float(feature_values.get(name, 0.0))
            spec = specs[name]
            span = max(float(spec["max"]) - float(spec["min"]), 1e-9)
            position = min(95.0, max(5.0, (value - float(spec["min"])) / span * 100.0))
            danger = "".join(
                f"<span class='gauge-danger' style='left:{left:.1f}%;width:{right-left:.1f}%;background:rgba(239,68,68,0.30)'></span>"
                for left, right in spec["danger"]
            )
            dot = dot_colors.get(_dot_color(name, value, shap_map.get(name, 0.0)), "#9ca3af")
            display_name = FEATURE_DISPLAY_NAMES.get(name, name)
            rows.append(
                "<div class='gauge-row' style='grid-template-columns:140px 1fr 70px;margin:9px 0'>"
                f"<div class='gauge-name'>{html.escape(display_name)}</div>"
                "<div class='gauge-track' style='height:22px;background:#2a2a2a;border-color:rgba(255,255,255,0.08)'>"
                f"{danger}<span class='gauge-dot' style='left:{position:.1f}%;background:{dot};"
                "width:16px;height:16px;border:2.5px solid #0a0a0a;box-shadow:0 1px 4px rgba(0,0,0,0.5)'></span></div>"
                "<div class='gauge-value' style=\"font-family:'SF Mono','Menlo','Consolas',monospace;"
                "font-size:13px;font-weight:600;text-align:right;min-width:70px\">"
                f"{_format_feature_value_html(name, value, spec)}</div></div>"
            )
    title = event_name if event_name else "Pre-market signal assessment"
    footer = (
        "Features computed from the trading day prior to the event. The classifier decided the tier based on these 12 signals."
        if event_name else
        "Features computed from yesterday's closed data. The classifier decides the tier from these 12 signals."
    )
    return f"""
    <div class='signal-gauge' style='background:#111111;border:0.5px solid rgba(255,255,255,0.08);border-radius:12px;padding:24px 28px;box-shadow:0 1px 3px rgba(0,0,0,0.4);margin-bottom:16px'>
      <div class='signal-head'>
        <span class='signal-pill' style='background:{badge_background};color:{badge_text};font-weight:700;padding:4px 14px;border-radius:20px;font-size:13px'>{html.escape(tier)}</span>
        <span class='signal-title'>{html.escape(title)}</span>
        <span class='signal-date'>{html.escape(date_label)}</span>
      </div>
      {''.join(rows)}
      <div class='verdict' style='border-left:4px solid {tier_color};background:#171717;border-radius:6px;padding:12px 16px'>
        <span class='verdict-arrow'>→</span>XGBoost classifier verdict:
        <b style='color:{badge_text}'>{html.escape(tier)}</b>
        <span> · P(Critical) = {float(p_critical):.1%} · {html.escape(summary)}</span>
      </div>
      <div class='signal-foot'>{html.escape(footer)}</div>
      <div class='signal-legend'>
        <span><i class='legend-dot' style='background:#22c55e'></i>Safe zone</span>
        <span><i class='legend-dot' style='background:#f59e0b'></i>Elevated</span>
        <span><i style='display:inline-block;width:20px;height:12px;background:rgba(239,68,68,0.30);border-radius:3px;vertical-align:middle;margin-right:4px'></i>Danger zone</span>
        <span><i class='legend-dot' style='background:#9ca3af'></i>Neutral/inactive</span>
      </div>
    </div>"""


def render_live_shap_waterfall(features: dict, shap_values: list[tuple[str, float]]) -> None:
    """Render a live-only 12-feature Critical-class SHAP waterfall-style bar chart."""
    if not shap_values:
        st.info("Live SHAP values are unavailable for this run.")
        return
    specs = _feature_specs(features)
    rows = sorted(shap_values, key=lambda item: abs(float(item[1])), reverse=True)
    labels = [
        f"{FEATURE_DISPLAY_NAMES.get(name, name)} "
        f"({_format_feature_value(name, float(features.get(name, 0.0)), specs.get(name, {'fmt': '{:.2f}', 'max': 1}))})"
        for name, _ in rows
    ]
    magnitudes = [abs(float(value)) for _, value in rows]
    colors = [_bar_color(float(value)) for _, value in rows]
    figure, axis = plt.subplots(figsize=(12.5, 5.2))
    figure.patch.set_facecolor("#0a0a0a")
    axis.set_facecolor("#0a0a0a")
    positions = np.arange(len(rows))
    axis.barh(positions, magnitudes, color=colors, height=.62)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8, color="#a3a3a3")
    axis.invert_yaxis()
    axis.set_xlabel("SHAP magnitude (impact on risk tier)", fontsize=9, color="#a3a3a3")
    axis.set_title("All risk drivers (SHAP)", loc="left", fontsize=11, fontweight="bold", color="#e5e5e5")
    axis.tick_params(axis="x", labelsize=8, colors="#a3a3a3")
    axis.tick_params(axis="y", labelsize=8, colors="#a3a3a3")
    axis.grid(axis="x", color="#262626", linewidth=.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#333333")
    for position, magnitude in zip(positions, magnitudes):
        axis.text(magnitude + max(magnitudes, default=1.0) * .02, position, f"{magnitude:.2f}", va="center", fontsize=8, color="#d4d4d4")
    axis.set_xlim(0, max(max(magnitudes, default=1.0) * 1.18, 0.2))
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)
    st.markdown(
        "<div class='waterfall-legend'>"
        "<span><i class='legend-square' style='background:#ef4444'></i>Pushes toward Critical</span>"
        "<span><i class='legend-square' style='background:#f59e0b'></i>Mild stress</span>"
        "<span><i class='legend-square' style='background:#22c55e'></i>Pushes toward Normal</span>"
        "<span><i class='legend-square' style='background:#9ca3af'></i>Negligible</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_replay(row: dict, mode: str = "replay") -> None:
    alert = row.get("alert", {}); tier = row["market_risk_tier"]
    features = feature_snapshot(row["date"])
    vol_z = float(features.get('volume_zscore_20d', row.get('vol_zscore', 0)))
    vix = float(features.get('vix_level', row.get('vix_level', 0)))
    style = tier.lower()
    p_critical = float(row["p_critical"])
    anomaly_score = float(row.get("anomaly_score", 0.0))
    anomaly_subtitle, anomaly_subtitle_color = _anomaly_subtitle(anomaly_score)
    st.markdown(f"""<div class='v-title'>{html.escape(tier)} {html.escape(mode)} — {html.escape(row['date'])}</div><div class='v-sub'>Fidelity SRE early-warning surface · advisory only</div><div class='kpis'>
      <div class='card {style}'><div class='label'>CURRENT RISK</div><div class='value'>{html.escape(tier)}</div></div>
      <div class='card'><div class='label'>P(CRITICAL)</div><div class='value'>{p_critical:.1%}</div><div class='sub'>Model-estimated Critical probability</div></div>
      <div class='card'><div class='label'>ANOMALY SCORE</div><div class='value'>{anomaly_score:.2f}</div><div class='sub' style='color:{anomaly_subtitle_color}'>{html.escape(anomaly_subtitle)}</div></div></div>""", unsafe_allow_html=True)
    chart = (
        pd.DataFrame(row.get("forecast_chart", [])) if row.get("forecast_chart")
        else forecast_chart_data(row["date"])
    ) if mode == "replay" else pd.DataFrame()
    replay_shaps = replay_shap_values(row["date"]) if mode == "replay" else []
    if mode == "replay" and not chart.empty:
        render_event_impact_card(row, chart, replay_shaps, vol_z, vix)
    if mode == "live":
        try:
            live_payload = live_feature_driver_values()
            live_features = live_payload["features"]
            shap_values = live_payload["critical_shap"]
        except Exception as exc:
            print(f"[VOLTEX DASHBOARD] live signal panel unavailable: {type(exc).__name__}: {exc}", flush=True)
            live_features = {
                "volume_zscore_20d": vol_z, "intraday_vol_pct": 0.0, "volume_acceleration": 0.0,
                "return_zscore_20d": 0.0, "market_breadth": float(row.get("stress_breadth", 0.0)),
                "vix_level": vix, "ma_ratio_5_20": 1.0, "return_std_20d": 0.0,
                "day_of_week": pd.Timestamp(row["date"]).dayofweek, "fomc_flag": 0.0, "cpi_flag": 0.0, "nfp_flag": 0.0,
            }
            shap_values = [(name, 0.0) for name in live_features]
        st.markdown(
            build_signal_gauge_panel(
                feature_values=live_features,
                shap_values={name: float(value) for name, value in shap_values},
                tier=tier,
                p_critical=p_critical,
                date_label=f"{row['date']} · before 9:30 AM ET",
            ),
            unsafe_allow_html=True,
        )
        try:
            forecast_components = live_forecast_components(str(row["date"]))
        except Exception as exc:
            print(f"[VOLTEX DASHBOARD] live deviation meter unavailable: {type(exc).__name__}: {exc}", flush=True)
            forecast_components = {}
        st.markdown(
            build_deviation_meter(
                surprise_zscore=float(row.get("forecast_surprise_zscore", 0.0)),
                forecast_volume=forecast_components.get("forecast_volume"),
                prophet_forecast=forecast_components.get("prophet_forecast"),
                arima_forecast=forecast_components.get("arima_forecast"),
                mode="live",
            ),
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        render_live_shap_waterfall(live_features, shap_values)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        try:
            replay_payload = replay_feature_driver_values(row["date"])
            replay_features = replay_payload["features"]
            replay_all_shaps = {name: float(value) for name, value in replay_payload["critical_shap"]}
        except Exception as exc:
            print(f"[VOLTEX DASHBOARD] replay signal panel unavailable for {row['date']}: {type(exc).__name__}: {exc}", flush=True)
            replay_features = {
                "volume_zscore_20d": vol_z, "intraday_vol_pct": float(features.get("intraday_vol_pct", 0.0)),
                "volume_acceleration": float(features.get("volume_acceleration", 0.0)),
                "return_zscore_20d": float(features.get("return_zscore_20d", 0.0)),
                "market_breadth": float(features.get("market_breadth", 0.0)), "vix_level": vix,
                "ma_ratio_5_20": float(features.get("ma_ratio_5_20", 1.0)),
                "return_std_20d": float(features.get("return_std_20d", 0.0)),
                "day_of_week": float(features.get("day_of_week", pd.Timestamp(row["date"]).dayofweek)),
                "fomc_flag": float(features.get("fomc_flag", 0.0)), "cpi_flag": float(features.get("cpi_flag", 0.0)),
                "nfp_flag": float(features.get("nfp_flag", 0.0)),
            }
            replay_all_shaps = {name: 0.0 for name in replay_features}
        st.markdown(
            build_signal_gauge_panel(
                feature_values=replay_features,
                shap_values=replay_all_shaps,
                tier=tier,
                p_critical=p_critical,
                date_label=str(row["date"]),
                event_name=str(row.get("event") or "Historical replay"),
            ),
            unsafe_allow_html=True,
        )
        event_day = chart.iloc[-1] if not chart.empty else pd.Series(dtype=float)
        st.markdown(
            build_deviation_meter(
                surprise_zscore=float(row.get("forecast_surprise_zscore", 0.0)),
                forecast_volume=event_day.get("forecast"),
                actual_volume=event_day.get("actual"),
                mode="replay",
            ),
            unsafe_allow_html=True,
        )
    astyle = tier.lower()
    precedents = alert.get("cited_precedents", [])
    precedent_html = ("<ul>" + "".join(f"<li>{html.escape(str(name))} <span style='color:#737373'>(cosine score not retained in archived alert)</span></li>" for name in precedents) + "</ul>"
                      if precedents else "No similar precedent retrieved above threshold.")
    checks = row.get("guardrails", {}).get("checks", {})
    guardrail_html = (" · ".join(f"{html.escape(str(name))}: {'PASS' if detail['passed'] else 'FAIL'}" for name, detail in checks.items())
                      if checks else "Guardrails: template fallback or archived result.")
    path = "LLM" if str(row.get("agent_path", "fallback")).lower() == "llm" else "Fallback"
    st.markdown(f"""<div class='agent {astyle}'><div class='message-id'>VTX-{html.escape(str(row['date']))}</div><h3>Agent message</h3><p>{html.escape(alert.get('plain_english_brief', 'No saved alert text.'))}</p><div class='precedents'><b>Cited precedents</b><br>{precedent_html}</div><div class='agent-meta'><b>Path</b> / {path} · {row.get('latency_ms', 0) / 1000:.1f}s<br><b>Guardrails</b> / {guardrail_html}</div></div>""", unsafe_allow_html=True)
    st.markdown(f"<div class='actions'><div class='action {'normal' if tier == 'Normal' else ''}'><b>Recommended action</b><br>{html.escape(alert.get('recommended_action', 'No action available.'))}</div><div class='meta'><b>Path</b> / {html.escape(row.get('agent_path','fallback').upper())}<br><b>Latency</b> / {row.get('latency_ms',0)/1000:.1f}s<br><b>Lead time</b> / {row.get('lead_time_minutes',15)} min</div></div>", unsafe_allow_html=True)
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
