"""VOLTEX — Fidelity SRE Early-Warning Dashboard."""

from __future__ import annotations

import json
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.ticker import FuncFormatter

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
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:14px 0 20px}.card{background:#fff;border:1px solid #E2E6E8;border-radius:12px;padding:17px 18px;box-shadow:0 1px 2px rgba(20,35,45,.04);min-width:0}.label{font-size:.77rem;color:#747D86;font-weight:700;margin-bottom:7px}.value{font-size:1.65rem;font-weight:800;color:#27323A;white-space:nowrap;overflow:visible}.sub{font-size:.72rem;color:#6B7280;margin-top:7px;line-height:1.3}
.critical{background:#FBEEEE;border-color:#F0D0D0}.critical .label,.critical .value{color:#8B2020}.high{background:#FFF1E8;border-color:#F4D1B3}.high .label,.high .value{color:#9A4D12}.moderate{background:#FFF8E8;border-color:#F1E0AE}.moderate .label,.moderate .value{color:#886614}.normal{background:#EDF5E8;border-color:#D2E6C8}.normal .label,.normal .value{color:#2E6B2E}
.section-card{background:#fff;border:1px solid #E2E6E8;border-radius:12px;padding:18px;margin:12px 0}.driver{display:grid;grid-template-columns:190px 1fr 35px;gap:10px;align-items:center;margin:10px 0;color:#39444C;font-size:.9rem}.track{height:9px;background:#EDF0F2;border-radius:99px;overflow:hidden}.fill{height:100%;border-radius:99px;background:#E24B4A}.fill.amber{background:#EF9F27}.fill.green{background:#5DCAA5}
.driver-legend{font-size:.72rem;color:#6B7280;margin:12px 0 0}.agent{background:#fff;border:1px solid #E2E6E8;border-left:5px solid #E24B4A;border-radius:12px;padding:20px;margin:12px 0}.agent.normal{border-left-color:#5DCAA5}.agent.moderate{border-left-color:#EF9F27}.agent.high{border-left-color:#E67E22}.agent h3{margin:0 0 8px;color:#8B2020}.agent.normal h3{color:#2E6B2E}.agent.moderate h3{color:#886614}.agent.high h3{color:#9A4D12}.agent p{color:#44505A;line-height:1.55;margin:0}.message-id{font-size:.75rem;color:#6B7280;font-weight:700;margin-bottom:6px}.agent-meta{font-size:.8rem;color:#6B7280;margin-top:14px;line-height:1.6}.precedents{margin-top:12px;font-size:.84rem;color:#46515A}.precedents ul{margin:5px 0 0;padding-left:20px}
.impact-card{background:#fff;border:.5px solid #E2E6E8;border-radius:12px;padding:18px 20px;margin:2px 0 12px;border-left-width:4px}.impact-header{display:flex;gap:9px;align-items:center;margin-bottom:14px}.impact-badge{font-size:.74rem;font-weight:800;padding:4px 8px;border-radius:999px}.impact-event{font-size:.95rem;font-weight:800;color:#2B3640}.impact-date{margin-left:auto;color:#6B7280;font-size:.78rem}.impact-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:9px}.impact-tile{background:#F9FAFB;border-radius:8px;padding:10px}.impact-tile-label{font-size:.69rem;color:#6B7280;font-weight:700;text-transform:uppercase}.impact-tile-value{font-size:1.25rem;font-weight:800;color:#27323A;margin-top:3px}.impact-bottom{display:grid;grid-template-columns:1.25fr .75fr;gap:20px;margin-top:16px}.impact-section-title{font-size:.72rem;font-weight:800;color:#6B7280;text-transform:uppercase;margin-bottom:8px}.impact-driver{display:grid;grid-template-columns:145px 1fr 36px;gap:7px;align-items:center;margin:7px 0;font-size:.74rem;color:#66717A}.impact-track{height:8px;background:#EDF0F2;border-radius:99px;overflow:hidden}.impact-fill{height:100%;border-radius:99px}.impact-signal{display:flex;justify-content:space-between;border-bottom:1px solid #EEF0F2;padding:5px 0;font-size:.78rem;color:#66717A}.impact-signal b{color:#27323A}@media(max-width:800px){.impact-bottom{grid-template-columns:1fr}.impact-driver{grid-template-columns:110px 1fr 32px}}
.signal-gauge{background:#fff;border:1px solid #E2E6E8;border-radius:12px;padding:18px 20px;margin:0 0 14px;box-shadow:0 1px 2px rgba(20,35,45,.04)}.signal-head{display:flex;align-items:center;gap:10px;margin-bottom:14px}.signal-pill{font-size:.74rem;font-weight:800;padding:5px 10px;border-radius:999px}.signal-title{font-size:1.02rem;font-weight:800;color:#27323A}.signal-date{margin-left:auto;color:#6B7280;font-size:.82rem}.gauge-row{display:grid;grid-template-columns:140px 1fr 60px;gap:10px;align-items:center;margin:8px 0}.gauge-name{font-size:.76rem;color:#747D86;text-align:right}.gauge-value{font-size:.82rem;font-weight:800;color:#27323A;text-align:right}.gauge-track{position:relative;height:20px;background:#F0F0EC;border-radius:999px;overflow:hidden;border:1px solid #EBEDE8}.gauge-danger{position:absolute;top:0;height:100%;background:rgba(226,75,74,.12)}.gauge-dot{position:absolute;top:50%;width:12px;height:12px;border-radius:999px;border:2px solid #fff;transform:translate(-50%,-50%);box-shadow:0 0 0 1px rgba(38,50,56,.12)}.verdict{margin-top:14px;border:1px solid #E2E6E8;border-radius:10px;padding:12px 14px;background:#FBFCFC;color:#3D4852;font-size:.9rem}.verdict-arrow{font-size:1.05rem;margin-right:8px}.signal-foot{font-size:.72rem;color:#6B7280;margin-top:10px}.signal-legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:10px;color:#6B7280;font-size:.72rem}.legend-dot{display:inline-block;width:10px;height:10px;border-radius:999px;margin-right:5px;vertical-align:-1px}.legend-danger{display:inline-block;width:18px;height:10px;border-radius:3px;background:rgba(226,75,74,.16);margin-right:5px;vertical-align:-1px}.waterfall-legend{display:flex;gap:15px;flex-wrap:wrap;font-size:11px;color:#68737C;margin:2px 0 5px}.legend-square{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
.actions{display:grid;grid-template-columns:1fr 290px;gap:14px;margin:12px 0}.action{background:#E8F0FB;border-radius:10px;padding:16px;color:#29496C}.action.normal{background:#F0F1F2;color:#4B555E}.meta{background:#fff;border:1px solid #E2E6E8;border-radius:10px;padding:14px;color:#68737C;font-size:.85rem;line-height:1.65}.advisory{background:#FFF7DE;border:1px solid #F2D998;color:#765A11;border-radius:9px;padding:10px 14px;font-size:.86rem;margin-top:15px}
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
def live_forecast_chart_data(current_date: str) -> pd.DataFrame:
    """Build the live trailing forecast chart without claiming a pre-open actual.

    The warmed cache supplies closed-session actual volumes.  Its final row is
    the live advisory date, so its ``actual`` value is deliberately replaced
    with NaN: only the pre-open forecast and threshold extend to that point.
    """
    from src.models.forecaster import VolumeForecaster

    cache = pd.read_csv(ROOT / "data/cache/live_market.csv", parse_dates=["date"])
    current = pd.Timestamp(current_date)
    cache = cache.loc[cache["date"] <= current].copy()
    window = cache.tail(10).copy()
    if len(window) < 10:
        return pd.DataFrame()
    for flag in ("fomc_flag", "cpi_flag", "nfp_flag"):
        window[flag] = 0
        cache[flag] = 0
    weights = json.loads((ROOT / "models/metrics.json").read_text())["forecaster"]["ensemble_weights"]
    forecaster = VolumeForecaster(ensemble_weights=weights)
    records: list[dict] = []
    for _, target in window.iterrows():
        history = cache.loc[cache["date"] < target["date"], ["date", "total_volume", "fomc_flag", "cpi_flag", "nfp_flag"]].tail(90)
        prophet, arima, order = forecaster._fit_models(history, select_order=forecaster.arima_order is None)
        forecaster.arima_order = order
        prediction = forecaster._predict_with_models(prophet, arima, pd.DataFrame([target]))
        forecast = float(forecaster._ensemble(prediction, forecaster.ensemble_weights).iloc[0]["forecast"])
        is_current = target["date"].normalize() == current.normalize()
        records.append({
            "date": target["date"],
            "actual": np.nan if is_current else float(target["total_volume"]),
            "forecast": forecast,
            "alert_threshold": forecast * 1.5,
        })
    return pd.DataFrame(records).set_index("date")


def render_volume_chart(chart: pd.DataFrame) -> None:
    """Render the saved trailing-window volume series with compact B-axis labels."""
    if chart.empty:
        return
    display = chart.copy()
    display.index = pd.to_datetime(display.index)
    all_values = pd.concat([display[column] for column in ("actual", "forecast", "alert_threshold") if column in display], ignore_index=True)
    low, high = float(all_values.min()), float(all_values.max())
    tick_low, tick_high = np.floor(low / 1e9) * 1e9, np.ceil(high / 1e9) * 1e9
    tick_values = np.linspace(tick_low, tick_high, num=5) if tick_high > tick_low else np.array([tick_low])
    dates = display.index.to_list()
    # Streamlit 1.30 does not reliably decode Plotly 6's binary ``bdata``
    # serialization for float Series. Plain lists preserve the raw shares
    # values and keep all three traces in the same coordinate space.
    actual_values = display["actual"].astype(float).tolist()
    forecast_values = display["forecast"].astype(float).tolist()
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=dates, y=actual_values, mode="lines", name="Actual volume",
                                line={"color": "#173B66", "width": 3}))
    figure.add_trace(go.Scatter(x=dates, y=forecast_values, mode="lines", name="Forecast volume",
                                line={"color": "#E24B4A", "width": 3}))
    if "alert_threshold" in display:
        threshold_values = display["alert_threshold"].astype(float).tolist()
        figure.add_trace(go.Scatter(x=dates, y=threshold_values, mode="lines",
                                    name="Surge threshold (1.5× forecast)", opacity=.55,
                                    line={"color": "#6B7280", "width": 1, "dash": "dash"}))
    figure.update_layout(
        template="plotly_white", height=325, margin={"l": 10, "r": 10, "t": 25, "b": 10},
        legend={"orientation": "h", "y": 1.15, "x": 0},
        yaxis={"title": "Volume (shares)", "tickvals": tick_values,
               "ticktext": [f"{value / 1e9:.1f}B" for value in tick_values], "gridcolor": "#EDF0F2"},
        xaxis={"gridcolor": "#EDF0F2"},
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def _volume_billions(value: float) -> str:
    return f"{value / 1e9:.2f}B"


def render_event_impact_card(row: dict, chart: pd.DataFrame, shap_values: list[tuple[str, float]], vol_z: float, vix: float) -> None:
    """Render the replay-only event summary using the saved pipeline outputs."""
    tier = row["market_risk_tier"]
    presentation = {
        "Normal": ("#5DCAA5", "#EDF5E8", "#2E6B2E"),
        "Moderate": ("#EF9F27", "#FFF8E8", "#886614"),
        "High": ("#E67E22", "#FFF1E8", "#9A4D12"),
        "Critical": ("#E24B4A", "#FBEEEE", "#8B2020"),
    }[tier]
    border, badge_background, badge_text = presentation
    event_name = str(row.get("event") or "Historical replay")
    event_day = chart.iloc[-1]
    surprise = float(row.get("forecast_surprise_zscore", 0.0))
    anomaly = float(row.get("anomaly_score", 0.0))
    maximum = max((abs(float(value)) for _, value in shap_values), default=0.0)
    driver_html = "".join(
        f"<div class='impact-driver'><span>{html.escape(str(name))}</span>"
        f"<div class='impact-track'><div class='impact-fill' style='width:{(abs(float(value)) / maximum * 100 if maximum else 0):.1f}%;background:{'#5DCAA5' if float(value) < 0 else '#E24B4A'}'></div></div>"
        f"<span style='text-align:right'>{abs(float(value)):.2f}</span></div>"
        for name, value in shap_values
    ) or "<span style='color:#6B7280;font-size:.78rem'>No SHAP drivers available.</span>"
    surprise_style = "color:#E24B4A" if abs(surprise) >= 2 else ""
    anomaly_style = "color:#E24B4A" if anomaly >= .5 else ""
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


def render_replay_evidence_chart(chart: pd.DataFrame, tier: str, surprise: float) -> None:
    """Render the replay-only lead-up/event-day volume evidence chart."""
    display = chart.copy()
    dates = pd.to_datetime(display.index)
    positions = np.arange(len(display))
    actual = display["actual"].to_numpy(dtype=float)
    forecast = display["forecast"].to_numpy(dtype=float)
    threshold = display["alert_threshold"].to_numpy(dtype=float)
    event_position = positions[-1]
    boundary = event_position - .5
    y_max = float(np.nanmax(np.concatenate([actual, forecast, threshold])))
    y_min = float(np.nanmin(np.concatenate([actual, forecast, threshold])))
    padding = max((y_max - y_min) * .15, 1e8)

    figure, axis = plt.subplots(figsize=(8.8, 4.5))
    figure.patch.set_facecolor("#FFFFFF")
    axis.set_facecolor("#FFFFFF")
    axis.axvspan(-.5, boundary, color=(136 / 255, 135 / 255, 128 / 255, .04), zorder=0)
    axis.axvspan(boundary, event_position + .5, color=(226 / 255, 75 / 255, 74 / 255, .08), zorder=0)
    axis.axvline(boundary, color="#E24B4A", linewidth=1, linestyle=(0, (4, 4)), alpha=.8)
    axis.text((boundary - .5) / 2, y_max + padding * .62, "LEAD-UP", ha="center", va="center", color="#8A9097", fontsize=8, fontweight="bold")
    axis.text((boundary + event_position + .5) / 2, y_max + padding * .62, "EVENT DAY", ha="center", va="center", color="#C43E3D", fontsize=8, fontweight="bold")
    axis.fill_between(positions, actual, forecast, where=actual > forecast, interpolate=True, color="#E24B4A", alpha=.10, zorder=1)
    axis.plot(positions, actual, color="#185FA5", linewidth=2.5, marker="o", markersize=4.5, zorder=3)
    axis.plot(positions, forecast, color="#A32D2D", linewidth=2, marker="o", markersize=3.5, zorder=3)
    axis.plot(positions, threshold, color="#888780", linewidth=1.5, linestyle=(0, (4, 4)), alpha=.55, zorder=2)
    axis.scatter([event_position], [actual[-1]], color="#E24B4A", s=95, zorder=4, edgecolors="#FFFFFF", linewidths=1.3)
    if tier in {"Critical", "High"}:
        bracket_x = event_position + .18
        lower, upper = sorted((forecast[-1], actual[-1]))
        axis.vlines(bracket_x, lower, upper, color="#E24B4A", linewidth=1.25, linestyles=(0, (3, 3)), zorder=4)
        axis.hlines([lower, upper], bracket_x - .04, bracket_x + .04, color="#E24B4A", linewidth=1.25, zorder=4)
        axis.annotate(f"{surprise:+.2f}σ", xy=(bracket_x, (lower + upper) / 2), xytext=(8, 0), textcoords="offset points", va="center", color="#C43E3D", fontsize=9, fontweight="bold")
    axis.set_xlim(-.5, event_position + .6)
    axis.set_ylim(y_min - padding * .15, y_max + padding)
    axis.set_ylabel("Volume (billions of shares)", color="#4E5962", fontsize=9)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e9:.1f}B"))
    axis.set_xticks(positions)
    axis.set_xticklabels([value.strftime("%b %d") for value in dates], fontsize=8, color="#68737C")
    axis.tick_params(axis="y", labelsize=8, colors="#68737C")
    axis.grid(axis="y", color="#E9ECEF", linewidth=.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#DDE2E5")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)
    st.markdown("<div style='display:flex;gap:15px;flex-wrap:wrap;font-size:11px;color:#68737C;margin:2px 0 5px'><span><b style='color:#185FA5'>■</b> Actual volume</span><span><b style='color:#A32D2D'>■</b> Forecast volume</span><span><b style='color:#888780'>╌</b> Surge threshold</span><span><b style='color:#E24B4A;opacity:.45'>■</b> Forecast surprise zone</span></div><div style='font-size:11px;color:#6B7280'>Forecast = expected normal volume. Gap from actual = stress signal.</div>", unsafe_allow_html=True)


def driver_fill_class(value: float, maximum: float) -> str:
    """Map saved Critical-class SHAP contribution to the existing risk palette."""
    if value < 0:
        return "green"
    if maximum and abs(value) < maximum * .5:
        return "amber"
    return ""


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
    if name == "market_breadth" and spec["max"] == 1.0:
        return spec["fmt"].format(value)
    return spec["fmt"].format(value)


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
        return "#B4B2A9"
    if shap_value < 0:
        return "#5DCAA5"
    if magnitude <= 1.0:
        return "#EF9F27"
    return "#E24B4A"


def render_live_signal_gauge(row: dict, features: dict, shap_values: list[tuple[str, float]]) -> None:
    """Render the live-only pre-market signal assessment panel."""
    tier = row["market_risk_tier"]
    presentation = {
        "Normal": ("#5DCAA5", "#EDF5E8", "#2E6B2E", "All signals within safe zones"),
        "Moderate": ("#EF9F27", "#FFF8E8", "#886614", "Some signals elevated — increased monitoring"),
        "High": ("#E67E22", "#FFF1E8", "#9A4D12", "Multiple signals in stress zone — pre-scale recommended"),
        "Critical": ("#E24B4A", "#FBEEEE", "#8B2020", "Critical stress detected — full incident protocol"),
    }[tier]
    border, badge_background, badge_text, summary = presentation
    shap_map = {name: float(value) for name, value in shap_values}
    specs = _feature_specs(features)
    rows = []
    for name in specs:
        value = float(features.get(name, 0.0))
        spec = specs[name]
        span = max(float(spec["max"]) - float(spec["min"]), 1e-9)
        position = min(95.0, max(5.0, (value - float(spec["min"])) / span * 100.0))
        danger = "".join(
            f"<span class='gauge-danger' style='left:{left:.1f}%;width:{right-left:.1f}%'></span>"
            for left, right in spec["danger"]
        )
        dot = _dot_color(name, value, shap_map.get(name, 0.0))
        rows.append(
            f"<div class='gauge-row'><div class='gauge-name'>{html.escape(spec['label'])}</div>"
            f"<div class='gauge-track'>{danger}<span class='gauge-dot' style='left:{position:.1f}%;background:{dot}'></span></div>"
            f"<div class='gauge-value'>{html.escape(_format_feature_value(name, value, spec))}</div></div>"
        )
    p_critical = float(row.get("p_critical", 0.0))
    panel = f"""
    <div class='signal-gauge' style='border-left:4px solid {border}'>
      <div class='signal-head'>
        <span class='signal-pill' style='background:{badge_background};color:{badge_text}'>{html.escape(tier)}</span>
        <span class='signal-title'>Pre-market signal assessment</span>
        <span class='signal-date'>{html.escape(str(row['date']))} · before 9:30 AM ET</span>
      </div>
      {''.join(rows)}
      <div class='verdict'><span class='verdict-arrow'>→</span>XGBoost classifier verdict:
        <b style='color:{badge_text}'>{html.escape(tier)}</b>
        <span> · P(Critical) = {p_critical:.1%} · {html.escape(summary)}</span>
      </div>
      <div class='signal-foot'>Features computed from yesterday's closed data. The classifier decides the tier — the forecast chart validates after market close.</div>
      <div class='signal-legend'>
        <span><i class='legend-dot' style='background:#5DCAA5'></i>Safe zone</span>
        <span><i class='legend-dot' style='background:#EF9F27'></i>Elevated</span>
        <span><i class='legend-danger'></i>Danger zone</span>
        <span><i class='legend-dot' style='background:#B4B2A9'></i>Neutral/inactive</span>
      </div>
    </div>"""
    st.markdown(panel, unsafe_allow_html=True)


def render_live_shap_waterfall(features: dict, shap_values: list[tuple[str, float]]) -> None:
    """Render a live-only 12-feature Critical-class SHAP waterfall-style bar chart."""
    if not shap_values:
        st.info("Live SHAP values are unavailable for this run.")
        return
    specs = _feature_specs(features)
    rows = sorted(shap_values, key=lambda item: abs(float(item[1])), reverse=True)
    labels = [
        f"{specs.get(name, {'label': name, 'fmt': '{:.2f}', 'min': 0, 'max': 1})['label']} "
        f"({_format_feature_value(name, float(features.get(name, 0.0)), specs.get(name, {'fmt': '{:.2f}', 'max': 1}))})"
        for name, _ in rows
    ]
    magnitudes = [abs(float(value)) for _, value in rows]
    colors = [_bar_color(float(value)) for _, value in rows]
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    figure.patch.set_facecolor("#FFFFFF")
    axis.set_facecolor("#FFFFFF")
    positions = np.arange(len(rows))
    axis.barh(positions, magnitudes, color=colors, height=.62)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8, color="#39444C")
    axis.invert_yaxis()
    axis.set_xlabel("SHAP magnitude (impact on risk tier)", fontsize=9, color="#4E5962")
    axis.set_title("All risk drivers (SHAP)", loc="left", fontsize=11, fontweight="bold", color="#27323A")
    axis.tick_params(axis="x", labelsize=8, colors="#68737C")
    axis.grid(axis="x", color="#E9ECEF", linewidth=.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#DDE2E5")
    for position, magnitude in zip(positions, magnitudes):
        axis.text(magnitude + max(magnitudes, default=1.0) * .02, position, f"{magnitude:.2f}", va="center", fontsize=8, color="#4E5962")
    axis.set_xlim(0, max(max(magnitudes, default=1.0) * 1.18, 0.2))
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)
    st.markdown(
        "<div class='waterfall-legend'>"
        "<span><i class='legend-square' style='background:#E24B4A'></i>Pushes toward Critical</span>"
        "<span><i class='legend-square' style='background:#EF9F27'></i>Mild stress</span>"
        "<span><i class='legend-square' style='background:#5DCAA5'></i>Pushes toward Normal</span>"
        "<span><i class='legend-square' style='background:#B4B2A9'></i>Negligible</span>"
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
    p_context = f"{p_critical:.1%} — {'below' if p_critical < .5 else 'above'} 50% alert level"
    z_context = f"{vol_z:+.2f} — {'within ±2σ normal range' if abs(vol_z) < 2 else 'exceeds ±2σ'}"
    vix_context = (f"{vix:.1f} — calm zone (&lt;20)" if vix < 20 else
                   f"{vix:.1f} — elevated (20–30)" if vix < 30 else
                   f"{vix:.1f} — stressed (≥30)")
    st.markdown(f"""<div class='v-title'>{html.escape(tier)} {html.escape(mode)} — {html.escape(row['date'])}</div><div class='v-sub'>Fidelity SRE early-warning surface · advisory only</div><div class='kpis'>
      <div class='card {style}'><div class='label'>CURRENT RISK</div><div class='value'>{html.escape(tier)}</div></div>
      <div class='card'><div class='label'>P(CRITICAL)</div><div class='value'>{p_critical:.1%}</div><div class='sub'>{p_context}</div></div>
      <div class='card'><div class='label'>VOLUME Z-SCORE</div><div class='value'>{vol_z:+.2f}</div><div class='sub'>{z_context}</div></div>
      <div class='card'><div class='label'>INTRADAY VOLATILITY (VIX PROXY)</div><div class='value'>{vix:.1f}</div><div class='sub'>{vix_context}</div></div></div>""", unsafe_allow_html=True)
    chart = (live_forecast_chart_data(row["date"]) if mode == "live" else
             pd.DataFrame(row.get("forecast_chart", [])) if row.get("forecast_chart") else
             forecast_chart_data(row["date"]))
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
        render_live_signal_gauge(row, live_features, shap_values)
        left, right = st.columns([1, 1])
        with left:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            render_live_shap_waterfall(live_features, shap_values)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='section-card'><b>Forecast vs actual volume</b>", unsafe_allow_html=True)
            if not chart.empty: render_volume_chart(chart)
            else: st.bar_chart(pd.DataFrame({"forecast_surprise_zscore": [row["forecast_surprise_zscore"]]}, index=[row["date"]]))
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("<div class='section-card'><b>Forecast vs actual volume</b>", unsafe_allow_html=True)
            if not chart.empty: render_replay_evidence_chart(chart, tier, float(row.get("forecast_surprise_zscore", 0.0)))
            else: st.bar_chart(pd.DataFrame({"forecast_surprise_zscore": [row["forecast_surprise_zscore"]]}, index=[row["date"]]))
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='section-card'><b>Top risk drivers (SHAP)</b>", unsafe_allow_html=True)
            shap_values = replay_shaps
            if not shap_values:
                shap_values = [(name, 0.0) for name in alert.get("top_drivers", [])]
            maximum = max((abs(float(value)) for _, value in shap_values), default=0.0)
            rows = "".join(
                f"<div class='driver'><span>{html.escape(str(name))}</span><div class='track'><div class='fill {driver_fill_class(float(value), maximum)}' style='width:{(max(4, abs(float(value)) / maximum * 100) if maximum else 4):.1f}%'></div></div><span>{abs(float(value)):.2f}</span></div>"
                for name, value in shap_values
            )
            st.markdown(rows or "No drivers available.", unsafe_allow_html=True)
            st.markdown("<div class='driver-legend'>green = pushes toward Normal · amber = mild stress · red = strong stress</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    astyle = tier.lower()
    precedents = alert.get("cited_precedents", [])
    precedent_html = ("<ul>" + "".join(f"<li>{html.escape(str(name))} <span style='color:#6B7280'>(cosine score not retained in archived alert)</span></li>" for name in precedents) + "</ul>"
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
