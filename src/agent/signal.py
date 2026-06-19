"""The hard ML-to-LLM compression boundary for VOLTEX."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


RAW_OHLCV_COLUMNS = frozenset({"open", "high", "low", "close", "volume", "total_volume", "ticker"})


class Signal(BaseModel):
    """The complete and only model-derived object an LLM may receive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    risk_tier: Literal["Normal", "Moderate", "High", "Critical"]
    p_critical: float = Field(ge=0, le=1)
    p_high: float = Field(ge=0, le=1)
    shap_top3: list[tuple[str, float]] = Field(min_length=3, max_length=3)
    anomaly_score: float = Field(ge=0, le=1)
    anomaly_elevated: bool
    forecast_surprise_zscore: float
    vol_zscore: float
    vix_level: float = Field(ge=0)
    stress_breadth: float = Field(ge=0, le=1)
    event_flags: dict[str, bool]
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("shap_top3")
    @classmethod
    def shap_names_are_nonempty_and_unique(cls, values: list[tuple[str, float]]) -> list[tuple[str, float]]:
        names = [name for name, _ in values]
        if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != 3:
            raise ValueError("shap_top3 must contain three distinct, non-empty driver names")
        return values

    @field_validator("event_flags")
    @classmethod
    def event_flags_are_exact(cls, values: dict[str, bool]) -> dict[str, bool]:
        required = {"fomc", "cpi", "nfp"}
        if set(values) != required or not all(isinstance(value, bool) for value in values.values()):
            raise ValueError("event_flags must contain exactly fomc, cpi, and nfp booleans")
        return values


def build_signal(market_prediction: dict, forecast_output: dict, anomaly_output: dict) -> Signal:
    """Create the fixed 11-field LLM input from trusted numerical outputs.

    Inputs are dictionaries rather than dataframes by design. Raw OHLCV keys
    are never copied to the result, and the Pydantic model forbids extras.
    """

    raw_keys = set(market_prediction) | set(forecast_output) | set(anomaly_output)
    # Presence in caller inputs is fine (dashboard data may be richer); only
    # the explicit allowlist below crosses the Layer 1/Layer 2 boundary.
    _ = raw_keys.intersection(RAW_OHLCV_COLUMNS)
    return Signal(
        risk_tier=market_prediction["market_risk_tier"],
        p_critical=market_prediction["p_critical"],
        p_high=market_prediction["p_high"],
        shap_top3=[tuple(item) for item in market_prediction["shap_top3"]],
        anomaly_score=anomaly_output["anomaly_score"],
        anomaly_elevated=anomaly_output["anomaly_elevated"],
        forecast_surprise_zscore=forecast_output["forecast_surprise_zscore"],
        vol_zscore=market_prediction["vol_zscore"],
        vix_level=market_prediction["vix_level"],
        stress_breadth=market_prediction["stress_breadth"],
        event_flags={
            "fomc": bool(market_prediction["event_flags"]["fomc"]),
            "cpi": bool(market_prediction["event_flags"]["cpi"]),
            "nfp": bool(market_prediction["event_flags"]["nfp"]),
        },
        date=str(market_prediction["date"]),
    )


assert RAW_OHLCV_COLUMNS.isdisjoint(Signal.model_fields), "Raw OHLCV must never be an LLM signal field"
