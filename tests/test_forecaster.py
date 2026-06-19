import numpy as np
import pandas as pd

from src.models.forecaster import VolumeForecaster, train_forecaster


def test_forecast_interfaces_are_positive_and_leakage_safe():
    source = pd.read_csv("data/processed/historical_features.csv", parse_dates=["date"])
    history = source.loc[source["date"].dt.year <= 2016].tail(120)
    forecaster = VolumeForecaster()
    result = forecaster.forecast_next_day(history)
    assert result["lower"] <= result["forecast"] <= result["upper"]
    assert result["forecast"] > 0
    assert np.isclose(sum(result["ensemble_weights"].values()), 1.0)
    assert forecaster.training_end == history["date"].max()


def test_forecast_surprise_is_finite():
    forecaster = VolumeForecaster(residuals=[10.0, -20.0, 15.0, -5.0])
    assert np.isfinite(forecaster.forecast_surprise(1_200.0, 1_000.0))


def test_train_appends_metrics(tmp_path):
    source = pd.read_csv("data/processed/historical_features.csv")
    _, metrics = train_forecaster(source, tmp_path)
    assert np.isclose(sum(metrics["ensemble_weights"].values()), 1.0)
    assert (tmp_path / "forecaster.pkl").exists()
