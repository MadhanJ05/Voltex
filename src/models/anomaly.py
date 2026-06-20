"""Isolation Forest second-opinion detector for market-level VOLTEX signals."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


SEED = 42
ANOMALY_FEATURES = [
    "volume_zscore_20d",
    "intraday_vol_pct",
    "volume_acceleration",
    "return_zscore_20d",
    "market_breadth",
    "vix_level",
    "ma_ratio_5_20",
    "return_std_20d",
]
CRISIS_DATES = pd.to_datetime(["2015-08-24", "2016-01-15", "2016-06-24", "2016-11-09", "2018-02-05"])
ANOMALY_CONFIG = {
    "random_state": SEED,
    "contamination": 0.027,
    "n_estimators": 300,
    "critical_percentile": 0.01,
    "features": ANOMALY_FEATURES,
}


@dataclass
class VoltexAnomalyDetector:
    model: IsolationForest | None = None
    scaler: StandardScaler = field(default_factory=StandardScaler)
    critical_raw_threshold: float | None = None
    training_raw_min: float | None = None
    training_raw_max: float | None = None

    def fit(self, normal_features_df: pd.DataFrame) -> "VoltexAnomalyDetector":
        missing = set(ANOMALY_FEATURES).difference(normal_features_df.columns)
        if missing:
            raise ValueError(f"Anomaly features missing: {sorted(missing)}")
        matrix = normal_features_df[ANOMALY_FEATURES].astype(float)
        scaled = self.scaler.fit_transform(matrix)
        self.model = IsolationForest(
            n_estimators=ANOMALY_CONFIG["n_estimators"],
            contamination=ANOMALY_CONFIG["contamination"],
            random_state=SEED,
            n_jobs=-1,
        ).fit(scaled)
        raw_scores = self.model.decision_function(scaled)
        self.critical_raw_threshold = float(np.quantile(raw_scores, ANOMALY_CONFIG["critical_percentile"]))
        self.training_raw_min = float(raw_scores.min())
        self.training_raw_max = float(raw_scores.max())
        return self

    def raw_decision_score(self, features_row: pd.Series | pd.DataFrame) -> float | np.ndarray:
        """Isolation Forest decision score: lower values mean more anomalous."""

        if self.model is None:
            raise RuntimeError("Detector must be fit before scoring")
        frame = features_row.to_frame().T if isinstance(features_row, pd.Series) else features_row
        raw = self.model.decision_function(self.scaler.transform(frame[ANOMALY_FEATURES].astype(float)))
        return float(raw[0]) if len(raw) == 1 else raw

    def anomaly_score(self, features_row: pd.Series | pd.DataFrame) -> float | np.ndarray:
        """Dashboard severity in [0, 1], where 1 is most structurally abnormal."""

        raw = self.raw_decision_score(features_row)
        denominator = max(float(self.training_raw_max - self.training_raw_min), np.finfo(float).eps)
        normalized = np.clip((float(self.training_raw_max) - np.asarray(raw)) / denominator, 0.0, 1.0)
        return float(normalized) if np.ndim(normalized) == 0 else normalized

    def anomaly_override(self, raw_decision_score: float) -> bool:
        """True when the raw decision score is in the training set's lowest 1%."""

        if self.critical_raw_threshold is None:
            raise RuntimeError("Detector must be fit before override checks")
        return bool(raw_decision_score < self.critical_raw_threshold)

    def save(self, artifact_dir: str | Path = "models") -> None:
        destination = Path(artifact_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "anomaly.pkl").open("wb") as handle:
            pickle.dump(self, handle)


def load_anomaly_detector(artifact_dir: str | Path = "models") -> VoltexAnomalyDetector:
    """Load the stable module-qualified anomaly artifact."""
    with (Path(artifact_dir) / "anomaly.pkl").open("rb") as handle:
        detector = pickle.load(handle)
    if not isinstance(detector, VoltexAnomalyDetector):
        raise TypeError("models/anomaly.pkl is not a VoltexAnomalyDetector")
    return detector


def _normal_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    dated = frame.copy()
    dated["date"] = pd.to_datetime(dated["date"])
    is_calm_period = dated["date"].dt.year <= 2016
    is_crisis = dated["date"].dt.normalize().isin(CRISIS_DATES)
    return dated.loc[is_calm_period & ~dated["surge_label"].astype(bool) & ~is_crisis].copy()


def train_anomaly_detector(frame: pd.DataFrame, artifact_dir: str | Path = "models") -> tuple[VoltexAnomalyDetector, dict[str, Any]]:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    normal_train = _normal_training_rows(data)
    detector = VoltexAnomalyDetector().fit(normal_train)

    crisis_records = []
    for day in CRISIS_DATES:
        row = data.loc[data["date"].dt.normalize() == day]
        if row.empty:
            continue
        raw = detector.raw_decision_score(row.iloc[0])
        severity = detector.anomaly_score(row.iloc[0])
        crisis_records.append({
            "date": day.strftime("%Y-%m-%d"), "raw_decision_score": raw,
            "anomaly_score": severity, "override": detector.anomaly_override(raw),
        })

    # Measure over-fire rate on later rows that are not labelled surges/crises.
    later_normal = data.loc[
        (data["date"].dt.year >= 2017)
        & ~data["surge_label"].astype(bool)
        & ~data["date"].dt.normalize().isin(CRISIS_DATES)
    ].copy()
    later_raw = detector.raw_decision_score(later_normal)
    normal_flag_rate = float(np.mean(later_raw < detector.critical_raw_threshold))
    scores = detector.anomaly_score(later_normal)
    top_decile_cutoff = float(np.quantile(scores, 0.90))
    metrics: dict[str, Any] = {
        "config": ANOMALY_CONFIG,
        "normal_training_rows": int(len(normal_train)),
        "critical_raw_threshold": detector.critical_raw_threshold,
        "normal_2017_2018_flag_rate": normal_flag_rate,
        "crisis_scores": crisis_records,
        "crisis_override_catch_count": int(sum(record["override"] for record in crisis_records)),
        "crisis_top_decile_cutoff_on_later_normals": top_decile_cutoff,
        "crisis_top_decile_count": int(sum(record["anomaly_score"] >= top_decile_cutoff for record in crisis_records)),
    }
    if metrics["crisis_override_catch_count"] == 0:
        metrics["limitations"] = [
            "The strict 1st-percentile override catches none of the five crisis rows using leakage-safe pre-open features. "
            "Overnight shocks require calendar/forecast-surprise corroboration rather than a fabricated anomaly override."
        ]
    detector.save(artifact_dir)
    destination = Path(artifact_dir)
    metric_file = destination / "metrics.json"
    existing = json.loads(metric_file.read_text(encoding="utf-8")) if metric_file.exists() else {}
    existing["anomaly"] = metrics
    metric_file.write_text(json.dumps(existing, indent=2, allow_nan=True), encoding="utf-8")
    return detector, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VOLTEX's market anomaly detector")
    parser.add_argument("--input", default="data/processed/historical_features.csv")
    parser.add_argument("--artifacts", default="models")
    args = parser.parse_args()
    _, metrics = train_anomaly_detector(pd.read_csv(args.input), args.artifacts)
    print("VOLTEX MODULE 4 — ANOMALY SUMMARY")
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
