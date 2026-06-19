"""Gemini orchestration with schema validation, guardrails, fallback, and audit."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

from .audit import AuditLogger
from .guardrails import guardrail_details, run_all_guardrails
from .retriever import PrecedentRetriever, RetrievedPrecedent
from .signal import Signal, build_signal


class Alert(BaseModel):
    # p_critical/p_high are included so G1 can prove ML confidence was passed
    # through unchanged; they are not generated analytics.
    model_config = ConfigDict(extra="forbid")
    risk_tier: Literal["Normal", "Moderate", "High", "Critical"]
    p_critical: float = Field(ge=0, le=1)
    p_high: float = Field(ge=0, le=1)
    recommended_action: str = Field(min_length=1, max_length=500)
    plain_english_brief: str = Field(min_length=1, max_length=700)
    top_drivers: list[str] = Field(max_length=3)
    cited_precedents: list[str] = Field(max_length=3)
    confidence_note: str = Field(min_length=1, max_length=500)


def _prompt(signal: Signal, precedents: list[RetrievedPrecedent]) -> str:
    return """You are VOLTEX, an advisory-only trading-platform SRE alert assistant.
Return JSON only. Do not add fields. The risk tier and probabilities must exactly match the SIGNAL.
Use only SIGNAL values. top_drivers must be a subset of SIGNAL.shap_top3 feature names.
cited_precedents must be a subset of the retrieved precedent identifiers. Do not give financial advice,
guarantees, price predictions, or unsupported statistics. Anomaly is supporting evidence only.
The JSON schema is exactly: {"risk_tier": string, "p_critical": number, "p_high": number,
"recommended_action": string, "plain_english_brief": string, "top_drivers": [string],
"cited_precedents": [string], "confidence_note": string}.

SIGNAL:
""" + signal.model_dump_json() + "\nRETRIEVED_PRECEDENTS:\n" + json.dumps([
        {"identifier": item.identifier, "similarity": round(item.cosine_similarity, 3), "document": item.document}
        for item in precedents
    ])


def _gemini_call(prompt: str) -> str:
    load_dotenv()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not configured")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2, max_output_tokens=512, response_mime_type="application/json",
            response_json_schema=Alert.model_json_schema(),
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response")
    return response.text


class VoltexAgent:
    def __init__(self, *, retriever: PrecedentRetriever | None = None, audit_logger: AuditLogger | None = None,
                 llm_callable: Callable[[str], str] | None = None) -> None:
        self.retriever = retriever or PrecedentRetriever()
        self.audit_logger = audit_logger or AuditLogger()
        self.llm_callable = llm_callable or _gemini_call

    def generate_alert(self, market_prediction: dict, forecast_output: dict, anomaly_output: dict) -> dict:
        started = time.perf_counter()
        signal = build_signal(market_prediction, forecast_output, anomaly_output)
        retrieved = self.retriever.retrieve(signal)
        guardrails = {"passed": False, "failures": [], "path": "fallback"}
        llm_used = False
        alert = None
        validation_error = None
        for _attempt in range(2):
            try:
                candidate = Alert.model_validate_json(self.llm_callable(_prompt(signal, retrieved)))
            except Exception as exc:
                validation_error = str(exc)
                continue
            details = guardrail_details(candidate, signal, retrieved)
            passed, failures = run_all_guardrails(candidate, signal, retrieved)
            serialized_details = {name: {"passed": okay, "reason": reason} for name, (okay, reason) in details.items()}
            if passed:
                alert, llm_used = candidate, True
                guardrails = {"passed": True, "failures": [], "checks": serialized_details, "path": "llm"}
            else:
                guardrails = {"passed": False, "failures": failures, "checks": serialized_details, "path": "fallback"}
            break
        if alert is None:
            from .fallback import fallback_alert
            reason = "; ".join(guardrails["failures"]) or validation_error or "LLM unavailable"
            alert = fallback_alert(signal, reason)
        latency_ms = (time.perf_counter() - started) * 1000
        result = {
            "alert": alert.model_dump(), "llm_used": llm_used, "guardrail_results": guardrails,
            "latency_ms": latency_ms, "signal": signal.model_dump(),
            "retrieved_precedents": [item.__dict__ for item in retrieved],
        }
        self.audit_logger.log(date=signal.date, signal=result["signal"], llm_used=llm_used,
                              guardrails=guardrails, latency_ms=latency_ms, alert=result["alert"])
        self._append_metrics(result)
        return result

    @staticmethod
    def _append_metrics(result: dict) -> None:
        """Record aggregate agent-path health without storing another alert copy."""

        path = Path("models/metrics.json")
        metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        agent_metrics = metrics.setdefault("agent", {"alerts_generated": 0, "llm_path_count": 0, "fallback_count": 0, "latencies_ms": []})
        agent_metrics["alerts_generated"] += 1
        agent_metrics["llm_path_count" if result["llm_used"] else "fallback_count"] += 1
        agent_metrics["latencies_ms"].append(round(result["latency_ms"], 3))
        # Retain a bounded summary rather than growing metrics indefinitely.
        agent_metrics["latencies_ms"] = agent_metrics["latencies_ms"][-100:]
        agent_metrics["latest_guardrail_results"] = result["guardrail_results"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")


_DEFAULT_AGENT: VoltexAgent | None = None


def generate_alert(market_prediction: dict, forecast_output: dict, anomaly_output: dict) -> dict:
    global _DEFAULT_AGENT
    if _DEFAULT_AGENT is None:
        _DEFAULT_AGENT = VoltexAgent()
    return _DEFAULT_AGENT.generate_alert(market_prediction, forecast_output, anomaly_output)
