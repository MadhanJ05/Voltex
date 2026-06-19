"""Always-available deterministic alert generation."""

from __future__ import annotations

from .signal import Signal


ACTIONS = {
    "Normal": "Continue standard monitoring; no action beyond routine readiness.",
    "Moderate": "Review capacity dashboards and validate customer-access health before market open.",
    "High": "Place the trading-platform SRE on heightened watch; verify capacity, order-entry latency, and vendor health.",
    "Critical": "Activate pre-market incident readiness: assign an SRE owner, verify failover and capacity, and monitor order-entry and login paths continuously.",
}


def fallback_alert(signal: Signal, reason: str = "LLM unavailable or guardrail failure"):
    from .agent import Alert

    drivers = [name for name, _ in signal.shap_top3]
    elevated = " The anomaly signal is elevated and is treated as supporting evidence." if signal.anomaly_elevated else ""
    return Alert(
        risk_tier=signal.risk_tier, p_critical=signal.p_critical, p_high=signal.p_high,
        recommended_action=ACTIONS[signal.risk_tier], top_drivers=drivers, cited_precedents=[],
        plain_english_brief=(
            f"{signal.risk_tier} market-platform risk for {signal.date}. "
            f"Primary model drivers are {', '.join(drivers)}. "
            f"Forecast surprise is {signal.forecast_surprise_zscore:.1f} standard deviations."
            f"{elevated}"
        ),
        confidence_note=f"Deterministic template used: {reason[:430]}.",
    )
