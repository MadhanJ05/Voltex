"""Historical and live data loaders used by VOLTEX Module 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


REQUIRED_KAGGLE_COLUMNS = {"date", "open", "high", "low", "close", "volume", "Name"}


@dataclass(frozen=True)
class KaggleStockLoader:
    """Load and aggregate the Kaggle 5-year constituent-level OHLCV dataset.

    The aggregate is deliberately an equal-weighted market proxy for prices
    plus total constituent volume. It avoids pretending that a simple average
    of stock prices is a tradable index.
    """

    path: str | Path

    def load_raw(self) -> pd.DataFrame:
        frame = pd.read_csv(self.path, parse_dates=["date"])
        missing = REQUIRED_KAGGLE_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"Kaggle CSV is missing required columns: {sorted(missing)}")
        frame = frame.rename(columns={"Name": "ticker"}).copy()
        numeric = ["open", "high", "low", "close", "volume"]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=["date", "ticker", *numeric])
        frame = frame[(frame["open"] > 0) & (frame["close"] > 0) & (frame["volume"] >= 0)]
        return frame.sort_values(["date", "ticker"]).reset_index(drop=True)

    def aggregate_index(self) -> pd.DataFrame:
        raw = self.load_raw()
        raw["daily_return"] = raw["close"] / raw["open"] - 1.0
        raw["intraday_range_pct"] = (raw["high"] - raw["low"]) / raw["open"] * 100.0
        raw["advance"] = (raw["close"] > raw["open"]).astype(float)

        daily = (
            raw.groupby("date", as_index=False)
            .agg(
                market_return=("daily_return", "mean"),
                total_volume=("volume", "sum"),
                intraday_vol_pct=("intraday_range_pct", "mean"),
                market_breadth=("advance", "mean"),
                constituents=("ticker", "nunique"),
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        daily["market_index"] = 100.0 * (1.0 + daily["market_return"]).cumprod()
        return daily


class LiveMarketLoader:
    """Fetch SPY/^VIX and degrade safely to a cached, schema-compatible CSV."""

    def __init__(self, cache_path: str | Path = "data/cache/live_market.csv") -> None:
        self.cache_path = Path(cache_path)

    def load(self, start: str | None = None, end: str | None = None, period: str = "5d", prefer_fresh_cache: bool = False) -> pd.DataFrame:
        return self.load_with_status(start=start, end=end, period=period, prefer_fresh_cache=prefer_fresh_cache)[0]

    def load_with_status(self, start: str | None = None, end: str | None = None, period: str = "5d", prefer_fresh_cache: bool = False) -> tuple[pd.DataFrame, str]:
        """Fetch a recent trading-day window with retry, then use cache safely."""
        if prefer_fresh_cache and self.cache_path.exists():
            age = datetime.now().timestamp() - self.cache_path.stat().st_mtime
            if age <= timedelta(hours=24).total_seconds():
                cached = pd.read_csv(self.cache_path, parse_dates=["date"])
                return cached.sort_values("date").reset_index(drop=True), "cached"
        error = None
        for attempt in range(3):
            try:
                live = self._fetch_yfinance(start=start, end=end, period=period)
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                if self.cache_path.exists():
                    previous = pd.read_csv(self.cache_path, parse_dates=["date"])
                    live = pd.concat([previous, live], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
                live.to_csv(self.cache_path, index=False)
                return live.reset_index(drop=True), "live"
            except Exception as exc:  # Slow external APIs get exponential backoff.
                error = exc
                if attempt < 2:
                    time.sleep(2 ** attempt)
        # External APIs must not take the agent down.
        if error is not None:
            if self.cache_path.exists():
                cached = pd.read_csv(self.cache_path, parse_dates=["date"])
                return cached.sort_values("date").reset_index(drop=True), "cached"
            raise RuntimeError(
                "Live market fetch failed and no cached CSV is available at "
                f"{self.cache_path}"
            ) from error

    def load_event_calendar(
        self, start: str, end: str, fred_api_key: str | None
    ) -> Mapping[str, pd.DatetimeIndex]:
        """Return scheduled macro-event dates, with an offline-safe fallback.

        FRED observations establish which CPI/NFP release months are available;
        their known monthly release conventions identify the pre-open event
        dates. FOMC dates are scheduled directly by the Federal Reserve and
        retained in the reproducible local calendar.
        """

        try:
            if not fred_api_key:
                raise ValueError("FRED_API_KEY is not set")
            return FredEventCalendar(fred_api_key).for_period(start, end)
        except Exception:
            # The historical schedule is a valid, known-in-advance fallback;
            # no network failure may prevent a pre-market prediction.
            from .features import FOMC_DATES, _cpi_proxy_dates, _first_fridays

            begin, finish = pd.Timestamp(start), pd.Timestamp(end)
            return {
                "fomc": FOMC_DATES[(FOMC_DATES >= begin) & (FOMC_DATES <= finish)],
                "cpi": _cpi_proxy_dates(begin, finish),
                "nfp": _first_fridays(begin, finish),
            }

    @staticmethod
    def _fetch_yfinance(start: str | None = None, end: str | None = None, period: str = "5d") -> pd.DataFrame:
        import yfinance as yf

        request = {"start": start, "end": end} if start else {"period": period}
        prices = yf.download(["SPY", "^VIX"], progress=False, auto_adjust=True, timeout=30, **request)
        if prices.empty:
            raise RuntimeError("yfinance returned no rows")
        spy = prices.xs("SPY", level=1, axis=1).dropna()
        vix = prices.xs("^VIX", level=1, axis=1)["Close"].reindex(spy.index).ffill()
        result = pd.DataFrame(
            {
                "date": spy.index.tz_localize(None),
                "market_return": spy["Close"].pct_change(),
                "total_volume": spy["Volume"],
                "intraday_vol_pct": (spy["High"] - spy["Low"]) / spy["Open"] * 100.0,
                # SPY-only fallback cannot observe constituent breadth. Its
                # signed-session proxy preserves the identical numeric schema
                # and is explicitly reported as a single-index limitation.
                "market_breadth": (spy["Close"].pct_change() > 0).astype(float),
                "market_index": spy["Close"],
                "vix_level": vix.to_numpy(),
            }
        )
        return result.dropna(subset=["market_return"]).reset_index(drop=True)


@dataclass(frozen=True)
class FredEventCalendar:
    """FRED-backed CPI/NFP calendar source with deterministic FOMC schedule."""

    api_key: str

    def for_period(self, start: str, end: str) -> Mapping[str, pd.DatetimeIndex]:
        # Imported lazily so historical/offline operation has no fredapi import
        # requirement. CPIAUCSL and PAYEMS are authoritative FRED series for
        # the two scheduled releases.
        from fredapi import Fred
        from .features import FOMC_DATES

        begin, finish = pd.Timestamp(start), pd.Timestamp(end)
        fred = Fred(api_key=self.api_key)
        cpi_months = pd.DatetimeIndex(fred.get_series("CPIAUCSL").index)
        nfp_months = pd.DatetimeIndex(fred.get_series("PAYEMS").index)
        cpi_months = cpi_months[(cpi_months >= begin - pd.Timedelta(days=45)) & (cpi_months <= finish)]
        nfp_months = nfp_months[(nfp_months >= begin - pd.Timedelta(days=45)) & (nfp_months <= finish)]
        cpi = pd.DatetimeIndex([month.to_period("M").start_time + pd.Timedelta(days=14) for month in cpi_months])
        nfp = pd.DatetimeIndex(
            [month.to_period("M").start_time + pd.offsets.Week(weekday=4) for month in nfp_months]
        )
        return {
            "fomc": FOMC_DATES[(FOMC_DATES >= begin) & (FOMC_DATES <= finish)],
            "cpi": cpi[(cpi >= begin) & (cpi <= finish)],
            "nfp": nfp[(nfp >= begin) & (nfp <= finish)],
        }


def flags_within_one_day(dates: pd.Series, event_dates: Iterable[pd.Timestamp]) -> pd.Series:
    """Return a known-in-advance event flag for the +/- one calendar-day window."""

    events = pd.DatetimeIndex(pd.to_datetime(list(event_dates))).normalize()
    values = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if events.empty:
        return pd.Series(0, index=dates.index, dtype="int8")
    hits = np.zeros(len(values), dtype=bool)
    for event in events:
        hits |= np.abs((values - event).days) <= 1
    return pd.Series(hits.astype("int8"), index=dates.index)
