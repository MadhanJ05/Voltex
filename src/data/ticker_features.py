"""Leakage-safe ticker-day features for the statistical VOLTEX classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, FOMC_DATES, _cpi_proxy_dates, _first_fridays, run_leakage_gate
from .loader import KaggleStockLoader, flags_within_one_day


def build_ticker_features(csv_path: str | Path) -> pd.DataFrame:
    """Build one pre-open feature record per ticker per day.

    Per-ticker rolling features are calculated after chronological sorting and
    shifted one session. Shared market-derived values are likewise lagged one
    session before being broadcast to every ticker for the target date.
    """

    raw = KaggleStockLoader(csv_path).load_raw()
    raw = raw.sort_values(["ticker", "date"]).reset_index(drop=True)
    ticker_groups = raw.groupby("ticker", observed=True, sort=False)
    raw["ticker_return"] = raw["close"] / raw["open"] - 1.0
    raw["raw_intraday_vol_pct"] = (raw["high"] - raw["low"]) / raw["open"] * 100.0

    # groupby-transform keeps every rolling operation within a ticker. There
    # is no Python loop over the 505 constituent histories.
    prior_volume_mean = ticker_groups["volume"].transform(
        lambda values: values.rolling(20, min_periods=20).mean().shift(1)
    )
    prior_volume_std = ticker_groups["volume"].transform(
        lambda values: values.rolling(20, min_periods=20).std(ddof=0).shift(1)
    )
    prior_return_mean = ticker_groups["ticker_return"].transform(
        lambda values: values.rolling(20, min_periods=20).mean().shift(1)
    )
    prior_return_std = ticker_groups["ticker_return"].transform(
        lambda values: values.rolling(20, min_periods=20).std(ddof=0).shift(1)
    )
    prior_short_volume = ticker_groups["volume"].transform(
        lambda values: values.rolling(5, min_periods=5).mean().shift(1)
    )
    ma_ratio = ticker_groups["close"].transform(
        lambda values: (values.rolling(5, min_periods=5).mean() / values.rolling(20, min_periods=20).mean()).shift(1)
    )

    raw["surge_label"] = (raw["volume"] > 1.5 * prior_volume_mean).astype("int8")
    raw["volume_zscore_20d"] = ((raw["volume"] - prior_volume_mean) / prior_volume_std.replace(0, np.nan)).groupby(raw["ticker"], observed=True).shift(1)
    raw["volume_acceleration"] = prior_short_volume / prior_volume_mean - 1.0
    raw["return_zscore_20d"] = ((raw["ticker_return"] - prior_return_mean) / prior_return_std.replace(0, np.nan)).groupby(raw["ticker"], observed=True).shift(1)
    raw["return_std_20d"] = prior_return_std.groupby(raw["ticker"], observed=True).shift(1)
    raw["ma_ratio_5_20"] = ma_ratio
    raw["intraday_vol_pct"] = raw.groupby("ticker", observed=True)["raw_intraday_vol_pct"].shift(1)
    raw["feature_source_date"] = raw.groupby("ticker", observed=True)["date"].shift(1)

    # Market features are created once per date, shifted once, and mapped back
    # to ticker rows. This is a market view known at the prior close.
    daily = (
        raw.groupby("date", as_index=True, observed=True)
        .agg(
            market_breadth=("ticker_return", lambda values: float((values > 0).mean())),
            market_intraday_range=("raw_intraday_vol_pct", "mean"),
        )
        .sort_index()
    )
    daily["vix_level"] = daily["market_intraday_range"].rolling(10, min_periods=10).mean() * np.sqrt(252)
    daily[["market_breadth", "vix_level"]] = daily[["market_breadth", "vix_level"]].shift(1)
    raw = raw.join(daily[["market_breadth", "vix_level"]], on="date")

    raw["day_of_week"] = raw["date"].dt.dayofweek.astype("int8")
    raw["fomc_flag"] = flags_within_one_day(raw["date"], FOMC_DATES)
    raw["cpi_flag"] = flags_within_one_day(raw["date"], _cpi_proxy_dates(raw["date"].min(), raw["date"].max()))
    raw["nfp_flag"] = flags_within_one_day(raw["date"], _first_fridays(raw["date"].min(), raw["date"].max()))
    result = raw[["ticker", "date", "feature_source_date", *FEATURE_COLUMNS, "surge_label"]].dropna().reset_index(drop=True)
    run_ticker_leakage_gate(result)
    return result


def run_ticker_leakage_gate(frame: pd.DataFrame) -> None:
    """Enforce chronology per ticker, then reuse the feature-correlation gate."""

    required = {"ticker", "date", "feature_source_date"}
    missing = required.difference(frame.columns)
    if missing:
        raise AssertionError(f"Ticker leakage gate missing columns: {sorted(missing)}")
    if (pd.to_datetime(frame["feature_source_date"]) >= pd.to_datetime(frame["date"])).any():
        raise AssertionError("Same-day/future ticker feature leakage detected")
    # The correlation requirement is global: a duplicate feature is harmful at
    # either inference granularity.
    run_leakage_gate(frame[["date", "feature_source_date", *FEATURE_COLUMNS]].assign(surge_label=frame["surge_label"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe VOLTEX ticker-day features")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/ticker_features.csv")
    args = parser.parse_args()
    result = build_ticker_features(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result):,} ticker-day rows ({result['ticker'].nunique()} tickers) to {output}")
    print(f"Per-ticker surge prevalence: {result['surge_label'].mean():.2%}")


if __name__ == "__main__":
    main()
