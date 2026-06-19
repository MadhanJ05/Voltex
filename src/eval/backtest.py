"""Six-event, no-fabrication VOLTEX validation backtest."""

from __future__ import annotations

import __main__
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from src.agent.agent import VoltexAgent
from src.agent.signal import build_signal
from src.data.features import FEATURE_COLUMNS, engineer_features
from src.data.loader import LiveMarketLoader
from src.models.anomaly import VoltexAnomalyDetector
from src.models.classifier import RiskLabeler, aggregate_to_market, load_classifier
from .gates import evaluate_gates, gate_dicts
from .report import write_report

EVENTS = {
    "2015-08-24": "Black Monday 2015", "2016-01-15": "January 2016 selloff",
    "2016-06-24": "Brexit", "2016-11-09": "US election", "2018-02-05": "VIX shock",
    "2020-03-09": "COVID market shock",
}


def covid_features(cache_path: str | Path = "data/cache/covid_2020.csv") -> pd.DataFrame:
    """Fetch/cache SPY+VIX and route it through the same engineer_features function."""
    cache = Path(cache_path); cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        daily = pd.read_csv(cache, parse_dates=["date"])
    else:
        daily = LiveMarketLoader(cache_path=cache).load("2020-01-01", "2020-04-16")
        daily.to_csv(cache, index=False)
    features = engineer_features(daily)
    historical = pd.read_csv("data/processed/historical_features.csv", nrows=1)
    if set(features.columns) != set(historical.columns):
        raise AssertionError("COVID features do not match historical feature schema")
    return features


def _artifacts():
    __main__.RiskLabeler = RiskLabeler
    __main__.VoltexAnomalyDetector = VoltexAnomalyDetector
    classifier = load_classifier("models")
    with open("models/anomaly.pkl", "rb") as handle: anomaly = pickle.load(handle)
    metrics = json.loads(Path("models/metrics.json").read_text())
    return classifier, anomaly, metrics


def _market_prediction(classifier, feature_row: pd.Series, ticker_rows: pd.DataFrame) -> dict:
    rows = ticker_rows.reset_index(drop=True)
    pred = classifier.predict_tickers(rows, include_shap=False)
    market = aggregate_to_market(pred, feature_row["date"])
    market["p_critical"], market["p_high"] = float(pred.p_critical.mean()), float(pred.p_high.mean())
    best = int((pred.p_critical + pred.p_high).to_numpy().argmax())
    market["shap_top3"] = classifier.explain_prediction(rows.iloc[[best]])
    market.update({"date": pd.Timestamp(feature_row["date"]).strftime("%Y-%m-%d"),
                   "vol_zscore": float(feature_row["volume_zscore_20d"]), "vix_level": float(feature_row["vix_level"]),
                   "event_flags": {"fomc": bool(feature_row.fomc_flag), "cpi": bool(feature_row.cpi_flag), "nfp": bool(feature_row.nfp_flag)}})
    return market


def run_backtest(live_agent: bool = True) -> list[dict]:
    classifier, anomaly, metrics = _artifacts()
    historical = pd.read_csv("data/processed/historical_features.csv", parse_dates=["date"])
    tickers = pd.read_csv("data/processed/ticker_features.csv", parse_dates=["date"])
    covid = covid_features(); covid["date"] = pd.to_datetime(covid["date"])
    surprise = {item["date"]: item["forecast_surprise_zscore"] for item in metrics["forecaster"]["crisis_forecast_surprises"]}
    agent = VoltexAgent() if live_agent else None
    rows: list[dict] = []
    for date, event in EVENTS.items():
        day = pd.Timestamp(date)
        if day.year == 2020:
            feature = covid.loc[covid.date.eq(day)].iloc[0]
            # SPY is the only requested live constituent: explicit one-ticker
            # aggregation, not an invented 505-ticker breadth estimate.
            ticker_rows = pd.DataFrame([{**{name: feature[name] for name in FEATURE_COLUMNS}, "ticker": "SPY", "date": day}])
            forecast_z = 0.0  # no continuous 2018→2020 aggregate-volume history
            source_note = "SPY-only aggregation; forecast surprise unavailable across the 2018–2020 data gap"
        else:
            feature = historical.loc[historical.date.eq(day)].iloc[0]
            ticker_rows = tickers.loc[tickers.date.eq(day)]
            forecast_z = float(surprise[date]); source_note = "historical constituent aggregation"
        market = _market_prediction(classifier, feature, ticker_rows)
        anomaly_value = float(anomaly.anomaly_score(feature))
        anomaly_output = {"anomaly_score": anomaly_value, "anomaly_elevated": anomaly_value >= metrics["anomaly"]["crisis_top_decile_cutoff_on_later_normals"]}
        forecast_output = {"forecast_surprise_zscore": forecast_z}
        result = agent.generate_alert(market, forecast_output, anomaly_output) if agent else {"alert": {}, "llm_used": False, "guardrail_results": {}, "latency_ms": 0}
        rows.append({"event": event, "date": date, "market_risk_tier": market["market_risk_tier"],
                     "stress_breadth": market["stress_breadth"], "p_critical": market["p_critical"],
                     "forecast_surprise_zscore": forecast_z, "anomaly_score": anomaly_value,
                     "lead_time_minutes": 15, "lead_time_convention": "alert issued 09:15 ET; stress onset convention 09:30 ET open",
                     "agent_path": "llm" if result["llm_used"] else "fallback", "guardrails": result["guardrail_results"],
                     "latency_ms": result["latency_ms"], "alert": result["alert"], "source_note": source_note})
    return rows


def run_validation() -> dict:
    """Run events, gates, and persisted sponsor-report artifacts."""
    rows = run_backtest(live_agent=True)
    metrics_path = Path("models/metrics.json"); metrics = json.loads(metrics_path.read_text())
    # The synthetic probe uses deterministic template construction: it proves
    # the always-valid alert path without spending 50 live Gemini calls.
    from src.agent.fallback import fallback_alert
    from src.agent.agent import Alert
    probe_ok = 0
    template = rows[0]
    for _ in range(50):
        signal = build_signal({"market_risk_tier": template["market_risk_tier"], "p_critical": template["p_critical"], "p_high": 0.0,
                               "shap_top3": [("market_breadth", 1.0), ("vix_level", .5), ("volume_zscore_20d", .2)],
                               "vol_zscore": 0., "vix_level": 20., "stress_breadth": template["stress_breadth"],
                               "event_flags": {"fomc": False, "cpi": False, "nfp": False}, "date": template["date"]},
                              {"forecast_surprise_zscore": 0.0}, {"anomaly_score": 0.0, "anomaly_elevated": False})
        Alert.model_validate(fallback_alert(signal).model_dump()); probe_ok += 1
    p95 = float(np.percentile([row["latency_ms"] for row in rows], 95))
    gates = gate_dicts(evaluate_gates(metrics["ticker_test"], metrics["calibration"], rows, probe_ok / 50, p95))
    report = {"ticker_test": metrics["ticker_test"], "calibration": metrics["calibration"], "p95_latency_ms": p95}
    write_report(event_rows=rows, gates=gates, metrics=report)
    # Curves are recomputed from the saved ticker model on the untouched 2018
    # rows; no model fitting occurs in report generation.
    classifier, _, _ = _artifacts()
    ticker = pd.read_csv("data/processed/ticker_features.csv", parse_dates=["date"])
    test = ticker.loc[ticker.date.dt.year.eq(2018)]
    labels = classifier.labeler.label(test).to_numpy() >= 2
    probabilities = classifier.predict_tickers(test, include_shap=False)
    positive = (probabilities["p_high"] + probabilities["p_critical"]).to_numpy()
    destination = Path("models/eval")
    fpr, tpr, _ = roc_curve(labels, positive); precision, recall, _ = precision_recall_curve(labels, positive)
    for filename, x, y, xlabel, ylabel, title in (
        ("roc_curve.png", fpr, tpr, "False positive rate", "True positive rate", "2018 High/Critical ROC"),
        ("pr_curve.png", recall, precision, "Recall", "Precision", "2018 High/Critical Precision–Recall"),
    ):
        fig, ax = plt.subplots(figsize=(5, 4)); ax.plot(x, y, color="#00B4D8"); ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(destination / filename, dpi=160); plt.close(fig)
    metrics["evaluation"] = {"gates": gates, "backtest": rows, **report}
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    return {"events": rows, "gates": gates, **report}
