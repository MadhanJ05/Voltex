"""Acceptance-gate calculation with explicit pass/fail values."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GateResult:
    name: str
    value: float
    target: str
    passed: bool


def evaluate_gates(ticker_metrics: dict, calibration: dict, event_rows: list[dict], guardrail_compliance: float, p95_latency_ms: float) -> list[GateResult]:
    test = ticker_metrics
    gates = [
        ("macro_f1_high_critical", test["macro_f1_high_critical"], ">= 0.77", lambda x: x >= .77),
        ("precision_high_critical", test["precision_high_critical"], ">= 0.75", lambda x: x >= .75),
        ("recall_high_critical", test["recall_high_critical"], ">= 0.80", lambda x: x >= .80),
        ("pr_auc_ovr_macro", test["pr_auc_ovr_macro"], ">= 0.78", lambda x: x >= .78),
        ("roc_auc_ovr_macro", test["roc_auc_ovr_macro"], ">= 0.85", lambda x: x >= .85),
        ("brier", calibration["after"]["brier"], "<= 0.10", lambda x: x <= .10),
        ("ece", calibration["after"]["ece"], "<= 0.05", lambda x: x <= .05),
        ("crisis_event_recall", sum(row["market_risk_tier"] in {"High", "Critical"} for row in event_rows) / 6, "= 1.00", lambda x: x == 1),
        ("guardrail_compliance", guardrail_compliance, "= 1.00", lambda x: x == 1),
        ("p95_agent_latency_ms", p95_latency_ms, "< 5000", lambda x: x < 5000),
    ]
    return [GateResult(name, float(value), target, bool(check(float(value)))) for name, value, target, check in gates]


def gate_dicts(results: list[GateResult]) -> list[dict]:
    return [asdict(result) for result in results]
