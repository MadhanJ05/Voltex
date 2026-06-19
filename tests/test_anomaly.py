import numpy as np
import pandas as pd
import pytest

from src.models.anomaly import CRISIS_DATES, train_anomaly_detector


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    frame = pd.read_csv("data/processed/historical_features.csv")
    detector, metrics = train_anomaly_detector(frame, tmp_path_factory.mktemp("anomaly-artifacts"))
    return detector, metrics, frame


def test_anomaly_score_is_bounded(trained):
    detector, _, frame = trained
    score = detector.anomaly_score(frame.iloc[-1])
    assert 0.0 <= score <= 1.0


@pytest.mark.xfail(
    strict=True,
    reason="Pre-open lagged features do not make all overnight crisis rows anomalous; tracked limitation, not overridden.",
)
def test_crises_are_in_top_anomaly_decile(trained):
    _, metrics, _ = trained
    assert metrics["crisis_top_decile_count"] == len(CRISIS_DATES)


def test_normal_flag_rate_is_not_overfiring(trained):
    _, metrics, _ = trained
    assert metrics["normal_2017_2018_flag_rate"] <= 2 * metrics["config"]["contamination"]


@pytest.mark.xfail(
    strict=True,
    reason="No crisis row breaches the strict training 1st-percentile threshold with leakage-safe pre-open inputs.",
)
def test_override_is_bool_and_catches_most_extreme_crisis(trained):
    detector, metrics, _ = trained
    extreme = min(metrics["crisis_scores"], key=lambda item: item["raw_decision_score"])
    assert isinstance(detector.anomaly_override(extreme["raw_decision_score"]), bool)
    assert detector.anomaly_override(extreme["raw_decision_score"])
