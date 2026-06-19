"""Ticker-day XGBoost classifier and market-level VOLTEX aggregation.

The model learns from individual ticker sessions for statistical power. Alerts
never consume those raw records: ``aggregate_to_market`` emits one daily record
with only market-level risk statistics for the later agent/dashboard layers.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

from src.data.features import FEATURE_COLUMNS
from .calibration import calibration_curve_data, expected_calibration_error, multiclass_brier_score


SEED = 42
TIER_NAMES = ("Normal", "Moderate", "High", "Critical")
TIER_TO_ID = {tier: number for number, tier in enumerate(TIER_NAMES)}
DOCUMENTED_CRISIS_DATES = frozenset(pd.Timestamp(day) for day in (
    "2015-08-24", "2016-01-15", "2016-06-24", "2016-11-09", "2018-02-05", "2020-03-09"
))

RISK_CONFIG: dict[str, Any] = {
    "seed": SEED,
    "feature_columns": FEATURE_COLUMNS,
    "score_weights": {
        "volume_zscore_percentile": 0.40,
        "intraday_vol_percentile": 0.25,
        "anomaly_proximity_percentile": 0.20,
        "scheduled_event": 0.15,
    },
    "tier_percentiles": {"normal_lt": 0.80, "moderate_lt": 0.95, "high_lt": 0.99, "critical_ge": 0.99},
    "model": {
        "objective": "multi:softprob", "num_class": 4, "max_depth": 6,
        "learning_rate": 0.05, "n_estimators": 500, "early_stopping_rounds": 50,
        "random_state": SEED,
    },
}

# These operational thresholds are deliberately separate from the ticker-tier
# labels. They describe the fraction of the constituent universe under stress.
# They are selected after inspecting documented, pre-2020 market stress events;
# no date-specific override exists.
MARKET_AGGREGATION_CONFIG: dict[str, float] = {
    "normal_lt_stress_breadth": 0.03,
    "moderate_lt_stress_breadth": 0.04,
    "high_lt_stress_breadth": 0.15,
    "critical_ge_stress_breadth": 0.15,
}


@dataclass
class RiskLabeler:
    """Continuous ticker-risk labels, fit exclusively on the training dates."""

    sorted_reference: dict[str, np.ndarray] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _anomaly_proximity(frame: pd.DataFrame) -> pd.Series:
        # Module 4 replaces this known-at-open proxy with IsolationForest.
        return frame["return_zscore_20d"].abs() + frame["return_std_20d"].rank(pct=True)

    @staticmethod
    def _event_signal(frame: pd.DataFrame) -> pd.Series:
        return frame[["fomc_flag", "cpi_flag", "nfp_flag"]].max(axis=1).astype(float)

    def fit(self, train: pd.DataFrame) -> "RiskLabeler":
        components = {
            "volume_zscore_percentile": train["volume_zscore_20d"],
            "intraday_vol_percentile": train["intraday_vol_pct"],
            "anomaly_proximity_percentile": self._anomaly_proximity(train),
        }
        self.sorted_reference = {name: np.sort(series.to_numpy(dtype=float)) for name, series in components.items()}
        score = self.score(train)
        self.thresholds = {
            "moderate": float(score.quantile(0.80)),
            "high": float(score.quantile(0.95)),
            "critical": float(score.quantile(0.99)),
        }
        return self

    def _percentile(self, name: str, values: pd.Series) -> np.ndarray:
        reference = self.sorted_reference[name]
        return np.searchsorted(reference, values.to_numpy(dtype=float), side="right") / len(reference)

    def score(self, frame: pd.DataFrame) -> pd.Series:
        weights = RISK_CONFIG["score_weights"]
        return pd.Series(
            weights["volume_zscore_percentile"] * self._percentile("volume_zscore_percentile", frame["volume_zscore_20d"])
            + weights["intraday_vol_percentile"] * self._percentile("intraday_vol_percentile", frame["intraday_vol_pct"])
            + weights["anomaly_proximity_percentile"] * self._percentile("anomaly_proximity_percentile", self._anomaly_proximity(frame))
            + weights["scheduled_event"] * self._event_signal(frame),
            index=frame.index,
            name="composite_risk_score",
        )

    def label(self, frame: pd.DataFrame) -> pd.Series:
        score = self.score(frame)
        label = np.select(
            [score < self.thresholds["moderate"], score < self.thresholds["high"], score < self.thresholds["critical"]],
            [0, 1, 2], default=3,
        ).astype(int)
        crisis_surge = frame["surge_label"].astype(bool) & pd.to_datetime(frame["date"]).dt.normalize().isin(DOCUMENTED_CRISIS_DATES)
        label[crisis_surge.to_numpy()] = TIER_TO_ID["Critical"]
        return pd.Series(label, index=frame.index, name="risk_tier_id")


def _sample_weights(labels: pd.Series) -> np.ndarray:
    counts = labels.value_counts()
    return labels.map(lambda tier: len(labels) / (len(counts) * counts.loc[tier])).to_numpy(dtype=float)


def _ticker_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    predicted = probabilities.argmax(axis=1)
    positive_true = y_true >= TIER_TO_ID["High"]
    positive_probability = probabilities[:, 2] + probabilities[:, 3]
    positive_pred = positive_probability >= 0.5
    metrics = {
        "macro_f1_high_critical": float(f1_score(y_true, predicted, labels=[2, 3], average="macro", zero_division=0)),
        "precision_high_critical": float(precision_score(positive_true, positive_pred, zero_division=0)),
        "recall_high_critical": float(recall_score(positive_true, positive_pred, zero_division=0)),
        "pr_auc_high_critical": float(average_precision_score(positive_true, positive_probability)),
        "roc_auc_high_critical": float(roc_auc_score(positive_true, positive_probability)),
    }
    all_classes = set(y_true)
    if all_classes == {0, 1, 2, 3}:
        encoded = label_binarize(y_true, classes=[0, 1, 2, 3])
        metrics["pr_auc_ovr_macro"] = float(average_precision_score(encoded, probabilities, average="macro"))
        metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
    else:
        metrics["pr_auc_ovr_macro"] = float("nan")
        metrics["roc_auc_ovr_macro"] = float("nan")
    return metrics


def _market_tier(stress_breadth: float) -> str:
    if stress_breadth < MARKET_AGGREGATION_CONFIG["normal_lt_stress_breadth"]:
        return "Normal"
    if stress_breadth < MARKET_AGGREGATION_CONFIG["moderate_lt_stress_breadth"]:
        return "Moderate"
    if stress_breadth < MARKET_AGGREGATION_CONFIG["high_lt_stress_breadth"]:
        return "High"
    return "Critical"


def aggregate_to_market(ticker_predictions: pd.DataFrame, date: str | pd.Timestamp) -> dict[str, Any]:
    """Compress one date of ticker risk into the dashboard/LLM market record."""

    day = pd.Timestamp(date).normalize()
    rows = ticker_predictions.loc[pd.to_datetime(ticker_predictions["date"]).dt.normalize() == day]
    if rows.empty:
        raise ValueError(f"No ticker predictions available for {day.date()}")
    stressed = rows["risk_tier"].isin(["High", "Critical"])
    result: dict[str, Any] = {
        "date": day.strftime("%Y-%m-%d"),
        "market_risk_tier": _market_tier(float(stressed.mean())),
        "stress_breadth": float(stressed.mean()),
        "mean_p_critical": float(rows["p_critical"].mean()),
        "n_critical_tickers": int((rows["risk_tier"] == "Critical").sum()),
        "n_tickers": int(len(rows)),
    }
    for feature in ("market_breadth", "vix_level", "fomc_flag", "cpi_flag", "nfp_flag"):
        if feature in rows:
            result[feature] = float(rows[feature].iloc[0])
    return result


@dataclass
class VoltexTickerClassifier:
    model: XGBClassifier
    calibrator: CalibratedClassifierCV
    labeler: RiskLabeler
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    _explainer: Any = field(default=None, init=False, repr=False)

    def _probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict_proba(frame[self.feature_columns].astype(float))

    def _shap_values(self, frame: pd.DataFrame) -> np.ndarray:
        import shap

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model)
        return np.asarray(self._explainer.shap_values(frame[self.feature_columns].astype(float)))

    def explain_prediction(self, row: pd.Series | pd.DataFrame) -> list[tuple[str, float]]:
        one = row.to_frame().T if isinstance(row, pd.Series) else row.iloc[[0]]
        values = self._shap_values(one)
        tier = int(self._probabilities(one).argmax(axis=1)[0])
        class_values = values[0, :, tier] if values.ndim == 3 else values[0]
        best = np.argsort(np.abs(class_values))[-3:][::-1]
        return [(self.feature_columns[index], float(class_values[index])) for index in best]

    def predict_tickers(self, features_df: pd.DataFrame, include_shap: bool = True) -> pd.DataFrame:
        probabilities = self._probabilities(features_df)
        tier_numbers = probabilities.argmax(axis=1)
        carry = [column for column in ("ticker", "date", "market_breadth", "vix_level", "fomc_flag", "cpi_flag", "nfp_flag") if column in features_df]
        result = features_df[carry].copy()
        result["risk_tier"] = [TIER_NAMES[number] for number in tier_numbers]
        result["p_critical"] = probabilities[:, 3]
        result["p_high"] = probabilities[:, 2]
        result["p_moderate"] = probabilities[:, 1]
        result["p_normal"] = probabilities[:, 0]
        if include_shap:
            result["shap_top3"] = [self.explain_prediction(features_df.iloc[[position]]) for position in range(len(features_df))]
        return result

    # Retained as a small compatibility alias for callers built during Module 2A.
    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        return self.predict_tickers(features_df)

    def save(self, artifact_dir: str | Path = "models") -> None:
        location = Path(artifact_dir)
        location.mkdir(parents=True, exist_ok=True)
        for filename, object_to_save in (
            ("xgb_ticker_classifier.pkl", self.model),
            ("calibrator.pkl", self.calibrator),
            ("risk_labeler.pkl", self.labeler),
        ):
            with (location / filename).open("wb") as handle:
                pickle.dump(object_to_save, handle)
        (location / "risk_config.json").write_text(json.dumps(RISK_CONFIG, indent=2), encoding="utf-8")
        (location / "market_aggregation_config.json").write_text(json.dumps(MARKET_AGGREGATION_CONFIG, indent=2), encoding="utf-8")


def load_classifier(artifact_dir: str | Path = "models") -> VoltexTickerClassifier:
    location = Path(artifact_dir)
    # Compatibility for artifacts saved by ``python -m`` before this module
    # was imported by name. New training runs pickle the canonical module path.
    import __main__
    if not hasattr(__main__, "RiskLabeler"):
        setattr(__main__, "RiskLabeler", RiskLabeler)
    with (location / "xgb_ticker_classifier.pkl").open("rb") as handle:
        model = pickle.load(handle)
    with (location / "calibrator.pkl").open("rb") as handle:
        calibrator = pickle.load(handle)
    with (location / "risk_labeler.pkl").open("rb") as handle:
        labeler = pickle.load(handle)
    return VoltexTickerClassifier(model=model, calibrator=calibrator, labeler=labeler)


def _date_splits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dated = frame.copy()
    dated["date"] = pd.to_datetime(dated["date"])
    train = dated.loc[dated["date"].dt.year <= 2016].copy()
    validation = dated.loc[dated["date"].dt.year == 2017].copy()
    test = dated.loc[dated["date"].dt.year == 2018].copy()
    split_sets = [set(part["date"].dt.normalize().unique()) for part in (train, validation, test)]
    if any(left.intersection(right) for offset, left in enumerate(split_sets) for right in split_sets[offset + 1:]):
        raise AssertionError("A trading date appears in more than one split")
    return train, validation, test


def train_classifier(frame: pd.DataFrame, artifact_dir: str | Path = "models") -> tuple[VoltexTickerClassifier, dict[str, Any]]:
    """Fit on ticker-days ≤2016, calibrate 2017, and test only 2018 ticker-days."""

    train, validation, test = _date_splits(frame)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Expected non-empty train (≤2016), validation (2017), and test (2018) date windows")
    labeler = RiskLabeler().fit(train)
    y_train, y_validation, y_test = (labeler.label(part) for part in (train, validation, test))
    model = XGBClassifier(
        objective="multi:softprob", num_class=4, max_depth=6, learning_rate=0.05,
        n_estimators=500, early_stopping_rounds=50, random_state=SEED, n_jobs=-1,
        tree_method="hist", eval_metric="mlogloss",
    )
    model.fit(train[FEATURE_COLUMNS], y_train, sample_weight=_sample_weights(y_train),
              eval_set=[(validation[FEATURE_COLUMNS], y_validation)], verbose=False)
    raw_test = model.predict_proba(test[FEATURE_COLUMNS])
    calibrator = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrator.fit(validation[FEATURE_COLUMNS], y_validation)
    calibrated_test = calibrator.predict_proba(test[FEATURE_COLUMNS])
    classifier = VoltexTickerClassifier(model=model, calibrator=calibrator, labeler=labeler)
    classifier.save(artifact_dir)

    # No SHAP is needed to evaluate aggregate market metrics; generating 600K
    # explanations would add cost without affecting any risk decision.
    all_predictions = classifier.predict_tickers(frame, include_shap=False)
    crisis_table = []
    for day in DOCUMENTED_CRISIS_DATES:
        if (pd.to_datetime(all_predictions["date"]).dt.normalize() == day).any():
            crisis_table.append(aggregate_to_market(all_predictions, day))
        else:
            crisis_table.append({"date": day.strftime("%Y-%m-%d"), "status": "unavailable_in_2013_2018_source"})
    crisis_table.sort(key=lambda record: record["date"])
    metrics: dict[str, Any] = {
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "split_dates": {"train": int(train["date"].nunique()), "validation": int(validation["date"].nunique()), "test": int(test["date"].nunique())},
        "tier_counts": {name: {TIER_NAMES[key]: int(value) for key, value in labels.value_counts().sort_index().items()}
                        for name, labels in (("train", y_train), ("validation", y_validation), ("test", y_test))},
        "risk_label_thresholds": labeler.thresholds,
        "ticker_test": _ticker_metrics(y_test, calibrated_test),
        "ticker_confusion_matrix": confusion_matrix(y_test, calibrated_test.argmax(axis=1), labels=[0, 1, 2, 3]).tolist(),
        "calibration": {
            "before": {"ece": expected_calibration_error(y_test, raw_test), "brier": multiclass_brier_score(y_test, raw_test)},
            "after": {"ece": expected_calibration_error(y_test, calibrated_test), "brier": multiclass_brier_score(y_test, calibrated_test)},
        },
        "market_crisis_dates": crisis_table,
    }
    location = Path(artifact_dir)
    calibration_curve_data(y_test, calibrated_test).to_csv(location / "calibration_curve_test.csv", index=False)
    pd.DataFrame(metrics["ticker_confusion_matrix"], index=TIER_NAMES, columns=TIER_NAMES).to_csv(location / "confusion_matrix_test.csv")
    (location / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    return classifier, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VOLTEX ticker-level risk classifier")
    parser.add_argument("--input", default="data/processed/ticker_features.csv")
    parser.add_argument("--artifacts", default="models")
    args = parser.parse_args()
    _, metrics = train_classifier(pd.read_csv(args.input), args.artifacts)
    print("VOLTEX MODULE 2B — TICKER CLASSIFIER SUMMARY")
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
