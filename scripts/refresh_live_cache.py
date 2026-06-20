"""Warm VOLTEX's live SPY/^VIX cache before a demo or scheduled run."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.features import engineer_features
from src.data.loader import LiveMarketLoader


def main() -> None:
    loader = LiveMarketLoader("data/cache/live_market.csv")
    # Six months gives feature engineering and the live forecast sufficient
    # pre-open history while still selecting the most recent trading session.
    daily, source = loader.load_with_status(period="6mo", prefer_fresh_cache=False)
    features = engineer_features(daily)
    latest = features.iloc[-1]
    print(f"{source.upper()} cache ready through {latest['date'].date()}")
    print(f"Feature row: vol_z={latest['volume_zscore_20d']:.2f}, vix={latest['vix_level']:.2f}")


if __name__ == "__main__":
    main()
