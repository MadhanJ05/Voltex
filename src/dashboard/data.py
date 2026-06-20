"""Artifact-first dashboard data access, isolated from Streamlit rendering."""

from __future__ import annotations

import json
import time
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def load_validation() -> dict:
    return json.loads((ROOT / "models/eval/validation_report.json").read_text())


def load_backtest() -> pd.DataFrame:
    return pd.read_csv(ROOT / "models/eval/six_event_backtest.csv")


def available_dates() -> list[str]:
    historical = pd.read_csv(ROOT / "data/processed/historical_features.csv", usecols=["date"])
    backtest = load_backtest()["date"].astype(str).tolist()
    return sorted(set(historical["date"].astype(str)).union(backtest), reverse=True)


def replay_day(date: str) -> dict:
    """Return saved end-to-end output for a validated event date."""
    validation = load_validation()
    for row in validation["backtest"]:
        if row["date"] == date:
            return row
    raise KeyError(f"No saved full-pipeline replay for {date}")


def feature_snapshot(date: str) -> dict:
    """Read values from the exact saved feature artifacts; never hardcode KPIs."""
    for path in (ROOT / "data/processed/historical_features.csv", ROOT / "data/cache/covid_2020.csv"):
        if path.exists():
            frame = pd.read_csv(path)
            match = frame.loc[frame["date"].astype(str).eq(date)]
            if not match.empty:
                return match.iloc[0].to_dict()
    return {}


def forecast_chart_data(date: str) -> pd.DataFrame:
    """Return a ten-session actual/forecast/threshold view ending on ``date``."""
    path = ROOT / "models/forecast_backtest_2018.csv"
    if path.exists():
        frame = pd.read_csv(path, parse_dates=["date"])
        selected = frame.loc[frame["date"].astype(str).eq(date)]
        if not selected.empty:
            trailing = frame.loc[frame["date"] <= pd.Timestamp(date)].tail(10).set_index("date")[["actual", "forecast"]]
            trailing["alert_threshold"] = trailing["forecast"] * 1.5
            return trailing
    # Historical crisis replays pre-date the saved 2018 backtest. Recreate a
    # strictly one-day-ahead trailing forecast from the saved daily series.
    history = pd.read_csv(ROOT / "data/processed/historical_features.csv", parse_dates=["date"])
    selected_day = pd.Timestamp(date)
    window = history.loc[history["date"] <= selected_day].tail(10)
    if len(window) < 10: return pd.DataFrame()
    from src.models.forecaster import VolumeForecaster
    metrics = json.loads((ROOT / "models/metrics.json").read_text())
    forecaster = VolumeForecaster(ensemble_weights=metrics["forecaster"]["ensemble_weights"])
    records = []
    for _, target in window.iterrows():
        prior = history.loc[history["date"] < target["date"], ["date", "total_volume", "fomc_flag", "cpi_flag", "nfp_flag"]]
        prophet, arima, order = forecaster._fit_models(prior, select_order=forecaster.arima_order is None)
        forecaster.arima_order = order
        raw = forecaster._predict_with_models(prophet, arima, pd.DataFrame([target]))
        forecast = forecaster._ensemble(raw, forecaster.ensemble_weights).iloc[0]["forecast"]
        records.append({"date": target["date"], "actual": target["total_volume"], "forecast": forecast, "alert_threshold": forecast * 1.5})
    return pd.DataFrame(records).set_index("date")


def load_audit(limit: int = 50) -> pd.DataFrame:
    path = ROOT / "data/voltex_audit.sqlite"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "date", "tier", "path", "guardrails", "latency_ms"])
    with sqlite3.connect(path) as connection:
        frame = pd.read_sql_query("SELECT timestamp,date,llm_used,guardrail_json,latency_ms,alert_json FROM alerts ORDER BY id DESC LIMIT ?", connection, params=(limit,))
    if frame.empty: return frame
    frame["tier"] = frame["alert_json"].map(lambda text: json.loads(text).get("risk_tier"))
    frame["path"] = frame["llm_used"].map({1: "LLM", 0: "Fallback"})
    frame["guardrails"] = frame["guardrail_json"].map(lambda text: "PASS" if json.loads(text).get("passed") else "FALLBACK")
    return frame[["timestamp", "date", "tier", "path", "guardrails", "latency_ms"]]


def live_day_result() -> dict:
    """Run the saved model stack over the latest SPY/^VIX session.

    SPY is explicitly a single-index proxy in live mode; the output carries
    that caveat rather than inventing constituent breadth.
    """
    from src.agent.agent import VoltexAgent
    from src.data.features import FEATURE_COLUMNS, engineer_features
    from src.data.loader import LiveMarketLoader
    from src.models.anomaly import load_anomaly_detector
    from src.models.classifier import aggregate_to_market, load_classifier
    from src.models.forecaster import VolumeForecaster

    # Fetch only the newest trading sessions; the warmed cache contributes the
    # longer feature/forecast history needed for this live inference.
    cache_path = ROOT / "data/cache/live_market.csv"
    exists = cache_path.exists()
    age_hours = (time.time() - cache_path.stat().st_mtime) / 3600 if exists else None
    print(
        f"[VOLTEX LIVE] cache_path={cache_path.resolve()} exists={exists} "
        f"age_hours={age_hours:.3f}" if age_hours is not None else
        f"[VOLTEX LIVE] cache_path={cache_path.resolve()} exists=False age_hours=n/a",
        flush=True,
    )
    loader = LiveMarketLoader(cache_path)
    daily, feed_status = loader.load_with_status(period="5d", prefer_fresh_cache=True)
    print(f"[VOLTEX LIVE] branch={feed_status} latest_date={daily['date'].max()}", flush=True)
    features = engineer_features(daily)
    feature = features.iloc[-1]
    classifier = load_classifier(ROOT / "models")
    ticker_row = pd.DataFrame([{**{column: feature[column] for column in FEATURE_COLUMNS}, "ticker": "SPY", "date": feature["date"]}])
    ticker_prediction = classifier.predict_tickers(ticker_row, include_shap=False)
    market = aggregate_to_market(ticker_prediction, feature["date"])
    market.update({"p_critical": float(ticker_prediction.p_critical.iloc[0]), "p_high": float(ticker_prediction.p_high.iloc[0]),
                   "shap_top3": classifier.explain_prediction(ticker_row), "date": pd.Timestamp(feature["date"]).strftime("%Y-%m-%d"),
                   "vol_zscore": float(feature.volume_zscore_20d), "vix_level": float(feature.vix_level),
                   "event_flags": {"fomc": bool(feature.fomc_flag), "cpi": bool(feature.cpi_flag), "nfp": bool(feature.nfp_flag)}})
    anomaly = load_anomaly_detector(ROOT / "models")
    anomaly_value = float(anomaly.anomaly_score(feature))
    # Use the same Prophet+ARIMA interface on the available SPY-volume history.
    history = features.iloc[:-1][["date", "total_volume", "fomc_flag", "cpi_flag", "nfp_flag"]].tail(90)
    # Reuse the production-selected ARIMA order for responsive live inference;
    # order selection belongs to offline training/backtesting, not the SRE UI.
    forecaster = VolumeForecaster(arima_order=(2, 0, 0))
    next_day = forecaster.forecast_next_day(history)
    surprise = forecaster.forecast_surprise(float(feature.total_volume), float(next_day["forecast"]))
    result = VoltexAgent().generate_alert(market, {"forecast_surprise_zscore": surprise},
                                          {"anomaly_score": anomaly_value, "anomaly_elevated": anomaly_value >= .9})
    return {"event": "Live SPY/^VIX proxy", "date": market["date"], "market_risk_tier": market["market_risk_tier"],
            "stress_breadth": market["stress_breadth"], "p_critical": market["p_critical"], "p_high": market["p_high"],
            "vol_zscore": market["vol_zscore"], "vix_level": market["vix_level"], "forecast_surprise_zscore": surprise, "anomaly_score": anomaly_value, "lead_time_minutes": 15,
            "agent_path": "llm" if result["llm_used"] else "fallback", "guardrails": result["guardrail_results"],
            "latency_ms": result["latency_ms"], "alert": result["alert"], "feed_status": feed_status,
            "source_note": "Live SPY-only proxy; constituent stress breadth unavailable."}
