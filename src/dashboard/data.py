"""Artifact-first dashboard data access, isolated from Streamlit rendering."""

from __future__ import annotations

import json
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
    """Use saved forecaster outputs; crisis points retain their true forecast."""
    metrics = json.loads((ROOT / "models/metrics.json").read_text())
    for item in metrics.get("forecaster", {}).get("crisis_forecast_surprises", []):
        if item["date"] == date:
            return pd.DataFrame({"actual": [item["actual_volume"]], "forecast": [item["forecast"]]}, index=[date])
    path = ROOT / "models/forecast_backtest_2018.csv"
    if path.exists():
        frame = pd.read_csv(path, parse_dates=["date"])
        selected = frame.loc[frame["date"].astype(str).eq(date)]
        if not selected.empty:
            return selected.set_index("date")[["actual", "forecast"]]
    return pd.DataFrame()


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
