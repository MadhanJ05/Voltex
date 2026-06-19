import pandas as pd
import pytest
import numpy as np

from src.data.features import FEATURE_COLUMNS, engineer_features, run_leakage_gate


def _daily_rows(n: int = 60) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(6080)
    return pd.DataFrame(
        {
            "date": dates,
            "market_return": rng.normal(0, 0.01, n),
            "total_volume": rng.lognormal(14, 0.2, n),
            "intraday_vol_pct": rng.uniform(0.4, 2.5, n),
            "market_breadth": rng.uniform(0.35, 0.65, n),
            "market_index": 100 * np.cumprod(1 + rng.normal(0, 0.01, n)),
        }
    )


def test_engineered_rows_are_strictly_lagged():
    result = engineer_features(_daily_rows())
    assert set(FEATURE_COLUMNS).issubset(result.columns)
    assert (pd.to_datetime(result["feature_source_date"]) < pd.to_datetime(result["date"])).all()


def test_leakage_gate_rejects_same_day_source():
    result = engineer_features(_daily_rows())
    result.loc[result.index[0], "feature_source_date"] = result.loc[result.index[0], "date"]
    with pytest.raises(AssertionError, match="Same-day"):
        run_leakage_gate(result)
