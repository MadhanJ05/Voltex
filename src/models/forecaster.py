"""Market-volume Prophet + auto-ARIMA ensemble for VOLTEX Module 3."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import holidays
import numpy as np
import pandas as pd
import cmdstanpy
from pmdarima import ARIMA, auto_arima
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from src.data.features import FOMC_DATES, _cpi_proxy_dates, _first_fridays
from src.data.loader import flags_within_one_day


EVENT_COLUMNS = ("fomc_flag", "cpi_flag", "nfp_flag")
SEED = 42
# Refit for every target session: every scored backtest observation is a true
# one-trading-day-ahead forecast made before that session's volume is known.
REFIT_INTERVAL = 1
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
cmdstanpy.disable_logging()


def _market_holidays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Prophet holiday table from the NYSE holiday calendar."""

    nyse = holidays.financial_holidays("NYSE", years=range(start.year, end.year + 2))
    return pd.DataFrame({"holiday": "nyse_closed", "ds": pd.to_datetime(list(nyse.keys()))})


def _as_daily_frame(data: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        frame = data.rename("total_volume").reset_index()
        frame.columns = ["date", "total_volume"]
    else:
        frame = data.copy()
    required = {"date", "total_volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Volume history missing required columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["total_volume"] = pd.to_numeric(frame["total_volume"], errors="raise")
    for event in EVENT_COLUMNS:
        if event not in frame:
            frame[event] = 0
        frame[event] = pd.to_numeric(frame[event], errors="raise").astype(int)
    return frame[["date", "total_volume", *EVENT_COLUMNS]].dropna().sort_values("date").reset_index(drop=True)


def _prophet_input(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={"date": "ds", "total_volume": "y"})[["ds", "y", *EVENT_COLUMNS]]


def _forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": float(mean_absolute_percentage_error(actual, predicted)),
    }


def _known_future_event_flags(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar regressors for a future horizon; all are known before open."""

    frame = pd.DataFrame({"date": pd.to_datetime(dates)})
    frame["fomc_flag"] = flags_within_one_day(frame["date"], FOMC_DATES)
    frame["cpi_flag"] = flags_within_one_day(frame["date"], _cpi_proxy_dates(frame["date"].min(), frame["date"].max()))
    frame["nfp_flag"] = flags_within_one_day(frame["date"], _first_fridays(frame["date"].min(), frame["date"].max()))
    return frame


@dataclass
class VolumeForecaster:
    """Inverse-validation-error ensemble with pre-open event regressors."""

    prophet_model: Prophet | None = None
    arima_model: Any = None
    arima_order: tuple[int, int, int] | None = None
    ensemble_weights: dict[str, float] = field(default_factory=lambda: {"prophet": 0.5, "arima": 0.5})
    residuals: list[float] = field(default_factory=list)
    training_end: pd.Timestamp | None = None

    def _fit_models(self, history: pd.DataFrame, select_order: bool) -> tuple[Prophet, Any, tuple[int, int, int]]:
        prophet = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            holidays=_market_holidays(history["date"].min(), history["date"].max()),
            interval_width=0.80,
        )
        for event in EVENT_COLUMNS:
            prophet.add_regressor(event, mode="additive")
        prophet.fit(_prophet_input(history))
        if select_order or self.arima_order is None:
            selected = auto_arima(
                history["total_volume"].to_numpy(), seasonal=False, stepwise=True,
                max_p=3, max_q=3, suppress_warnings=True, error_action="ignore",
            )
            order = tuple(int(item) for item in selected.order)
        else:
            order = self.arima_order
        arima = ARIMA(order=order, suppress_warnings=True).fit(history["total_volume"].to_numpy())
        return prophet, arima, order

    @staticmethod
    def _predict_with_models(prophet: Prophet, arima: Any, future: pd.DataFrame) -> pd.DataFrame:
        prophet_fc = prophet.predict(future.rename(columns={"date": "ds"})[["ds", *EVENT_COLUMNS]])
        arima_fc, arima_ci = arima.predict(n_periods=len(future), return_conf_int=True, alpha=0.20)
        return pd.DataFrame(
            {
                "date": future["date"].to_numpy(),
                "prophet": prophet_fc["yhat"].to_numpy(),
                "prophet_lower": prophet_fc["yhat_lower"].to_numpy(),
                "prophet_upper": prophet_fc["yhat_upper"].to_numpy(),
                "arima": np.asarray(arima_fc),
                "arima_lower": np.asarray(arima_ci)[:, 0],
                "arima_upper": np.asarray(arima_ci)[:, 1],
            }
        )

    @staticmethod
    def _ensemble(predictions: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
        result = predictions.copy()
        result["forecast"] = weights["prophet"] * result["prophet"] + weights["arima"] * result["arima"]
        result["lower"] = weights["prophet"] * result["prophet_lower"] + weights["arima"] * result["arima_lower"]
        result["upper"] = weights["prophet"] * result["prophet_upper"] + weights["arima"] * result["arima_upper"]
        return result

    def _expanding_predictions(self, history: pd.DataFrame, targets: pd.DataFrame, refit_interval: int = REFIT_INTERVAL) -> pd.DataFrame:
        """Walk forward through target days with a one-session forecast horizon.

        Each checkpoint is fit only on observations strictly before its target.
        The event regressors are calendar-known, so no target volume leaks into
        either model.
        """

        output: list[pd.DataFrame] = []
        expanding = history.copy()
        for start in range(0, len(targets), refit_interval):
            block = targets.iloc[start:start + refit_interval].copy()
            prophet, arima, order = self._fit_models(expanding, select_order=self.arima_order is None)
            self.arima_order = order
            forecast = self._predict_with_models(prophet, arima, block)
            forecast["actual"] = block["total_volume"].to_numpy()
            output.append(forecast)
            # Expand only after every forecast in this block has been issued.
            expanding = pd.concat([expanding, block], ignore_index=True)
        return pd.concat(output, ignore_index=True)

    def fit(self, daily_volume_series: pd.Series | pd.DataFrame) -> None:
        history = _as_daily_frame(daily_volume_series)
        self.prophet_model, self.arima_model, self.arima_order = self._fit_models(history, select_order=True)
        self.training_end = history["date"].max()

    def calibrate_weights(self, train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
        """Estimate inverse-MAE weights from an expanding validation backtest."""

        self.arima_order = None
        validation_predictions = self._expanding_predictions(train, validation)
        mae_prophet = mean_absolute_error(validation_predictions["actual"], validation_predictions["prophet"])
        mae_arima = mean_absolute_error(validation_predictions["actual"], validation_predictions["arima"])
        inverse = np.array([1.0 / mae_prophet, 1.0 / mae_arima])
        inverse /= inverse.sum()
        self.ensemble_weights = {"prophet": float(inverse[0]), "arima": float(inverse[1])}
        return self._ensemble(validation_predictions, self.ensemble_weights)

    def forecast_next_day(self, history: pd.Series | pd.DataFrame) -> dict[str, float | dict[str, float]]:
        data = _as_daily_frame(history)
        self.fit(data)
        next_day = pd.Timestamp(data["date"].max()) + pd.offsets.BDay(1)
        future = _known_future_event_flags(pd.DatetimeIndex([next_day]))
        base = self._predict_with_models(self.prophet_model, self.arima_model, future)
        row = self._ensemble(base, self.ensemble_weights).iloc[0]
        return {"forecast": float(max(row["forecast"], 1.0)), "lower": float(max(row["lower"], 1.0)),
                "upper": float(max(row["upper"], 1.0)), "ensemble_weights": self.ensemble_weights.copy()}

    def forecast_horizon(self, history: pd.Series | pd.DataFrame, days: int = 1) -> pd.DataFrame:
        data = _as_daily_frame(history)
        self.fit(data)
        dates = pd.bdate_range(data["date"].max() + pd.offsets.BDay(1), periods=days)
        future = _known_future_event_flags(dates)
        return self._ensemble(self._predict_with_models(self.prophet_model, self.arima_model, future), self.ensemble_weights)

    def forecast_surprise(self, actual: float, forecast: float) -> float:
        residual = float(actual - forecast)
        scale = float(np.std(self.residuals, ddof=1)) if len(self.residuals) > 1 else max(abs(forecast) * 0.05, 1.0)
        return float(residual / max(scale, 1.0))


def _diagnostics(residuals: pd.Series) -> dict[str, Any]:
    adf_p = float(adfuller(residuals, autolag="AIC")[1])
    ljung_p = float(acorr_ljungbox(residuals, lags=[min(10, max(1, len(residuals) // 3))], return_df=True)["lb_pvalue"].iloc[0])
    return {
        "adf_pvalue": adf_p,
        "stationary_pass": adf_p < 0.05,
        "ljung_box_pvalue": ljung_p,
        "no_autocorrelation_pass": ljung_p > 0.05,
    }


def train_forecaster(frame: pd.DataFrame, artifact_dir: str | Path = "models") -> tuple[VolumeForecaster, dict[str, Any]]:
    data = _as_daily_frame(frame)
    train = data.loc[data["date"].dt.year <= 2016].copy()
    validation = data.loc[data["date"].dt.year == 2017].copy()
    test = data.loc[data["date"].dt.year == 2018].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Expected non-empty train, validation, and test periods")
    forecaster = VolumeForecaster()
    forecaster.calibrate_weights(train, validation)
    # Fit remains restricted to history up to 2017 before the 2018 walk-forward.
    test_predictions = forecaster._expanding_predictions(pd.concat([train, validation], ignore_index=True), test)
    test_predictions = forecaster._ensemble(test_predictions, forecaster.ensemble_weights)
    test_predictions["residual"] = test_predictions["actual"] - test_predictions["forecast"]
    rolling_std = test_predictions["residual"].rolling(20, min_periods=5).std(ddof=0).replace(0, np.nan)
    test_predictions["forecast_surprise_zscore"] = test_predictions["residual"] / rolling_std
    forecaster.residuals = test_predictions["residual"].dropna().tolist()
    forecaster.fit(pd.concat([train, validation, test], ignore_index=True))

    metrics = {
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "ensemble_weights": forecaster.ensemble_weights,
        "test_metrics": {
            "prophet": _forecast_metrics(test_predictions["actual"], test_predictions["prophet"]),
            "arima": _forecast_metrics(test_predictions["actual"], test_predictions["arima"]),
            "ensemble": _forecast_metrics(test_predictions["actual"], test_predictions["forecast"]),
        },
        "residual_diagnostics": _diagnostics(test_predictions["residual"]),
    }
    # Sanity-check historical crisis surprises with models fit only up to each
    # prior target date. The scale is the held-out residual dispersion.
    crisis_rows = []
    for day in pd.to_datetime(["2015-08-24", "2016-01-15", "2016-06-24", "2016-11-09", "2018-02-05"]):
        target = data.loc[data["date"] == day]
        prior = data.loc[data["date"] < day]
        if target.empty or len(prior) < 100:
            continue
        prophet, arima, _ = forecaster._fit_models(prior, select_order=False)
        raw = forecaster._predict_with_models(prophet, arima, target)
        combined = forecaster._ensemble(raw, forecaster.ensemble_weights).iloc[0]
        surprise = forecaster.forecast_surprise(float(target["total_volume"].iloc[0]), float(combined["forecast"]))
        crisis_rows.append({"date": day.strftime("%Y-%m-%d"), "forecast_surprise_zscore": surprise,
                            "actual_volume": float(target["total_volume"].iloc[0]), "forecast": float(combined["forecast"])})
    metrics["crisis_forecast_surprises"] = crisis_rows

    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "forecaster.pkl").open("wb") as handle:
        pickle.dump(forecaster, handle)
    test_predictions.to_csv(destination / "forecast_backtest_2018.csv", index=False)
    existing = {}
    metric_file = destination / "metrics.json"
    if metric_file.exists():
        existing = json.loads(metric_file.read_text(encoding="utf-8"))
    existing["forecaster"] = metrics
    metric_file.write_text(json.dumps(existing, indent=2, allow_nan=True), encoding="utf-8")
    return forecaster, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VOLTEX market-volume forecast ensemble")
    parser.add_argument("--input", default="data/processed/historical_features.csv")
    parser.add_argument("--artifacts", default="models")
    args = parser.parse_args()
    _, metrics = train_forecaster(pd.read_csv(args.input), args.artifacts)
    print("VOLTEX MODULE 3 — FORECASTER SUMMARY")
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
