"""Leakage-safe daily feature engineering for VOLTEX."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .loader import KaggleStockLoader, flags_within_one_day


FEATURE_COLUMNS = [
    "volume_zscore_20d",
    "intraday_vol_pct",
    "volume_acceleration",
    "return_zscore_20d",
    "market_breadth",
    "vix_level",
    "ma_ratio_5_20",
    "return_std_20d",
    "day_of_week",
    "fomc_flag",
    "cpi_flag",
    "nfp_flag",
]

# Scheduled FOMC meetings for the Kaggle training period. The latest event
# calendars are supplied by FRED in the live path; this compact historical
# calendar keeps the offline capstone dataset fully reproducible.
FOMC_DATES = pd.to_datetime(
    [
        "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31", "2013-09-18", "2013-10-30", "2013-12-18",
        "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30", "2014-09-17", "2014-10-29", "2014-12-17",
        "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
        "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
        "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
        "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    ]
)


def _first_fridays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """NFP releases normally occur on the first Friday; known ahead of time."""

    months = pd.date_range(start.to_period("M").start_time, end, freq="MS")
    return pd.DatetimeIndex([month + pd.offsets.Week(weekday=4) for month in months])


def _cpi_proxy_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """CPI is generally released mid-month; use the 15th for an offline proxy."""

    return pd.date_range(start.to_period("M").start_time, end, freq="MS") + pd.Timedelta(days=14)


def _safe_zscore(value: pd.Series, mean: pd.Series, std: pd.Series) -> pd.Series:
    return (value - mean) / std.replace(0, np.nan)


def engineer_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Create a next-day prediction table without same-day market leakage.

    Market-derived values are calculated at close of *t-1* and shifted onto
    target date *t*. The three calendar/event features are known before t's
    open and therefore intentionally are not lagged.
    """

    required = {"date", "market_return", "total_volume", "intraday_vol_pct", "market_breadth", "market_index"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Daily data is missing required columns: {sorted(missing)}")
    df = daily.sort_values("date").copy().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    min_periods = 20
    volume_mean = df["total_volume"].rolling(20, min_periods=min_periods).mean().shift(1)
    volume_std = df["total_volume"].rolling(20, min_periods=min_periods).std(ddof=0).shift(1)
    return_mean = df["market_return"].rolling(20, min_periods=min_periods).mean().shift(1)
    return_std = df["market_return"].rolling(20, min_periods=min_periods).std(ddof=0).shift(1)

    # Label reflects the realised target-day surge against prior 20 sessions.
    df["surge_label"] = (df["total_volume"] > 1.5 * volume_mean).astype("int8")
    df["volume_zscore_20d"] = _safe_zscore(df["total_volume"], volume_mean, volume_std).shift(1)
    # Change in the short-vs-long volume regime.  This is intentionally not a
    # restatement of the single-day volume z-score: it captures sustained
    # acceleration across the preceding week.
    short_volume_mean = df["total_volume"].rolling(5, min_periods=5).mean().shift(1)
    df["volume_acceleration"] = short_volume_mean / volume_mean - 1.0
    df["return_zscore_20d"] = _safe_zscore(df["market_return"], return_mean, return_std).shift(1)
    df["return_std_20d"] = return_std.shift(1)
    df["ma_ratio_5_20"] = (
        df["market_index"].rolling(5, min_periods=5).mean()
        / df["market_index"].rolling(20, min_periods=20).mean()
    ).shift(1)

    # Historical CSV has no VIX. This is a deliberately named proxy based on
    # constituent intraday ranges; live loader replaces it with actual ^VIX.
    if "vix_level" not in df:
        df["vix_level"] = (
            df["intraday_vol_pct"].rolling(10, min_periods=10).mean() * np.sqrt(252)
        )

    market_derived = [
        "intraday_vol_pct", "market_breadth", "vix_level",
    ]
    df[market_derived] = df[market_derived].shift(1)
    df["feature_source_date"] = df["date"].shift(1)
    df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")
    df["fomc_flag"] = flags_within_one_day(df["date"], FOMC_DATES)
    df["cpi_flag"] = flags_within_one_day(df["date"], _cpi_proxy_dates(df["date"].min(), df["date"].max()))
    df["nfp_flag"] = flags_within_one_day(df["date"], _first_fridays(df["date"].min(), df["date"].max()))

    # ``total_volume`` is the market-level forecasting target for Module 3;
    # it is not part of FEATURE_COLUMNS and therefore is never a classifier
    # input or an LLM-facing field.
    output = df[["date", "feature_source_date", "total_volume", *FEATURE_COLUMNS, "surge_label"]].dropna().reset_index(drop=True)
    run_leakage_gate(output)
    return output


def run_leakage_gate(frame: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Fail loudly for same-day leakage or nearly duplicate feature pairs."""

    required = {"date", "feature_source_date", *FEATURE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise AssertionError(f"Leakage gate cannot run; missing columns: {sorted(missing)}")
    dates = pd.to_datetime(frame["date"])
    sources = pd.to_datetime(frame["feature_source_date"])
    if (sources >= dates).any():
        bad = frame.loc[sources >= dates, ["date", "feature_source_date"]].head(3).to_dict("records")
        raise AssertionError(f"Same-day/future feature leakage detected: {bad}")

    numeric = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    corr = numeric.corr(method="spearman")
    offenders = [
        (left, right, float(corr.loc[left, right]))
        for left, right in combinations(FEATURE_COLUMNS, 2)
        if pd.notna(corr.loc[left, right]) and abs(corr.loc[left, right]) > threshold
    ]
    if offenders:
        formatted = ", ".join(f"{a}/{b}={rho:.3f}" for a, b, rho in offenders)
        raise AssertionError(f"Feature collinearity gate failed (|Spearman rho| > {threshold}): {formatted}")
    return corr


def build_historical_features(csv_path: str | Path) -> pd.DataFrame:
    return engineer_features(KaggleStockLoader(csv_path).aggregate_index())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe VOLTEX historical features")
    parser.add_argument("--input", required=True, help="Path to Kaggle all_stocks_5yr CSV")
    parser.add_argument("--output", default="data/processed/historical_features.csv")
    args = parser.parse_args()
    features = build_historical_features(args.input)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(destination, index=False)
    corr = run_leakage_gate(features)
    print(f"Wrote {len(features):,} prediction rows to {destination}")
    print(f"Surge prevalence: {features['surge_label'].mean():.2%}")
    print(f"Largest |Spearman rho|: {corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max():.3f}")


if __name__ == "__main__":
    main()
