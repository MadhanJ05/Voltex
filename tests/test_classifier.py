import pandas as pd
import pytest

from src.models.classifier import (
    DOCUMENTED_CRISIS_DATES,
    TIER_NAMES,
    _date_splits,
    aggregate_to_market,
    train_classifier,
)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    frame = pd.read_csv("data/processed/ticker_features.csv")
    classifier, metrics = train_classifier(frame, tmp_path_factory.mktemp("ticker-artifacts"))
    return classifier, metrics, frame


def test_predict_tickers_probabilities_tiers_and_shap(trained):
    classifier, _, frame = trained
    prediction = classifier.predict_tickers(frame.tail(3))
    assert prediction[["p_critical", "p_high", "p_moderate", "p_normal"]].sum(axis=1).round(10).eq(1.0).all()
    assert prediction["risk_tier"].isin(TIER_NAMES).all()
    assert prediction["shap_top3"].map(len).eq(3).all()


def test_market_aggregation_valid_range(trained):
    classifier, _, frame = trained
    date = frame["date"].iloc[-1]
    market = aggregate_to_market(
        classifier.predict_tickers(frame.loc[frame["date"] == date], include_shap=False), date
    )
    assert market["market_risk_tier"] in TIER_NAMES
    assert 0.0 <= market["stress_breadth"] <= 1.0


def test_date_splits_do_not_overlap(trained):
    _, _, frame = trained
    train, validation, test = _date_splits(frame)
    dates = [set(pd.to_datetime(part["date"]).dt.normalize()) for part in (train, validation, test)]
    assert not dates[0] & dates[1]
    assert not dates[0] & dates[2]
    assert not dates[1] & dates[2]


def test_documented_crisis_dates_high_or_critical(trained):
    _, metrics, _ = trained
    available = {
        row["date"]: row["market_risk_tier"]
        for row in metrics["market_crisis_dates"]
        if "market_risk_tier" in row
    }
    expected = {day.strftime("%Y-%m-%d") for day in DOCUMENTED_CRISIS_DATES if day.year <= 2018}
    assert expected.issubset(available)
    assert all(available[day] in {"High", "Critical"} for day in expected)
