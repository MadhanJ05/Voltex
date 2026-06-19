"""Pure, independently testable guardrails for Layer 2 alerts."""

from __future__ import annotations

import re
from typing import Iterable

from .retriever import RetrievedPrecedent
from .signal import Signal


BRIEF_CAP = 700
_BANNED = re.compile(r"\b(guarantee[sd]?|risk[- ]free|buy|sell|hold|price target|will rise|will fall)\b", re.IGNORECASE)
_UNSUPPORTED_STAT = re.compile(r"(?:\$\s?\d|\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*(?:basis points|bps))", re.IGNORECASE)


def g1_tier_immutability(alert, signal: Signal) -> tuple[bool, str]:
    passed = (
        alert.risk_tier == signal.risk_tier
        and alert.p_critical == signal.p_critical
        and alert.p_high == signal.p_high
    )
    return passed, "tier and probabilities immutable" if passed else "tier or ML probability was altered"


def g2_driver_containment(alert, signal: Signal) -> tuple[bool, str]:
    allowed = {name for name, _ in signal.shap_top3}
    invalid = set(alert.top_drivers).difference(allowed)
    return not invalid, "drivers contained" if not invalid else f"unapproved drivers: {sorted(invalid)}"


def g3_precedent_containment(alert, retrieved: Iterable[RetrievedPrecedent]) -> tuple[bool, str]:
    allowed = {item.identifier for item in retrieved}
    invalid = set(alert.cited_precedents).difference(allowed)
    return not invalid, "precedents contained" if not invalid else f"unretrieved precedents: {sorted(invalid)}"


def g4_reasoning_filter(alert, signal: Signal) -> tuple[bool, str]:
    text = f"{alert.plain_english_brief} {alert.confidence_note}"
    if len(alert.plain_english_brief) > BRIEF_CAP:
        return False, f"brief exceeds {BRIEF_CAP} characters"
    if _BANNED.search(text):
        return False, "banned guarantee, financial-advice, or price-prediction claim"
    if _UNSUPPORTED_STAT.search(text):
        return False, "unsupported fabricated statistic pattern"
    return True, "reasoning filter passed"


def run_all_guardrails(alert, signal: Signal, retrieved: Iterable[RetrievedPrecedent]) -> tuple[bool, list[str]]:
    checks = tuple(guardrail_details(alert, signal, retrieved).values())
    failures = [reason for passed, reason in checks if not passed]
    return not failures, failures


def guardrail_details(alert, signal: Signal, retrieved: Iterable[RetrievedPrecedent]) -> dict[str, tuple[bool, str]]:
    return {
        "G1_tier_immutability": g1_tier_immutability(alert, signal),
        "G2_driver_containment": g2_driver_containment(alert, signal),
        "G3_precedent_containment": g3_precedent_containment(alert, retrieved),
        "G4_reasoning_filter": g4_reasoning_filter(alert, signal),
    }
