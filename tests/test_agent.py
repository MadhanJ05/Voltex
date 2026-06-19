import json

import pytest
from pydantic import ValidationError

from src.agent.agent import VoltexAgent
from src.agent.audit import AuditLogger
from src.agent.guardrails import g1_tier_immutability, g2_driver_containment, g3_precedent_containment, g4_reasoning_filter
from src.agent.retriever import RetrievedPrecedent
from src.agent.signal import RAW_OHLCV_COLUMNS, Signal, build_signal


def _inputs(tier="Critical"):
    market = {
        "market_risk_tier": tier, "p_critical": 0.82, "p_high": 0.14,
        "shap_top3": [("volume_zscore_20d", 1.2), ("vix_level", 0.8), ("market_breadth", -0.5)],
        "vol_zscore": 4.7, "vix_level": 32.8, "stress_breadth": 0.94,
        "event_flags": {"fomc": False, "cpi": False, "nfp": False}, "date": "2015-08-24",
        # Rich caller data must not cross into Signal.
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 999,
    }
    return market, {"forecast_surprise_zscore": 6.3}, {"anomaly_score": 0.64, "anomaly_elevated": True}


class StubRetriever:
    def retrieve(self, signal):
        return [RetrievedPrecedent("Black Monday 2015 (2015-08-24)", "Black Monday 2015", "2015-08-24", "volume-driven", 0.91, "precedent")]


def _valid_response(signal):
    return json.dumps({
        "risk_tier": signal.risk_tier, "p_critical": signal.p_critical, "p_high": signal.p_high,
        "recommended_action": "Verify capacity and order-entry health before market open.",
        "plain_english_brief": "Market conditions warrant pre-market SRE readiness.",
        "top_drivers": [signal.shap_top3[0][0], signal.shap_top3[1][0]],
        "cited_precedents": ["Black Monday 2015 (2015-08-24)"], "confidence_note": "Model signal is advisory only.",
    })


def test_signal_rejects_malformed_shap_and_raw_columns_never_cross():
    market, forecast, anomaly = _inputs()
    market["shap_top3"] = [("one", 1.0), ("two", 2.0)]
    with pytest.raises(ValidationError):
        build_signal(market, forecast, anomaly)
    assert RAW_OHLCV_COLUMNS.isdisjoint(Signal.model_fields)


def test_each_guardrail_catches_tampering():
    market, forecast, anomaly = _inputs()
    signal = build_signal(market, forecast, anomaly)
    valid = json.loads(_valid_response(signal))
    from src.agent.agent import Alert
    retrieved = StubRetriever().retrieve(signal)
    tampered_tier = Alert(**(valid | {"risk_tier": "Normal"}))
    assert not g1_tier_immutability(tampered_tier, signal)[0]
    hallucinated_driver = Alert(**(valid | {"top_drivers": ["made_up_driver"]}))
    assert not g2_driver_containment(hallucinated_driver, signal)[0]
    fake_precedent = Alert(**(valid | {"cited_precedents": ["Not Retrieved (2024-01-01)"]}))
    assert not g3_precedent_containment(fake_precedent, retrieved)[0]
    fake_stat = Alert(**(valid | {"plain_english_brief": "Conditions are 73% certain."}))
    assert not g4_reasoning_filter(fake_stat, signal)[0]


def test_guardrail_failure_fires_valid_fallback(tmp_path):
    market, forecast, anomaly = _inputs()
    agent = VoltexAgent(
        retriever=StubRetriever(), audit_logger=AuditLogger(tmp_path / "audit.sqlite"),
        llm_callable=lambda _prompt: json.dumps(json.loads(_valid_response(build_signal(market, forecast, anomaly))) | {"risk_tier": "Normal"}),
    )
    result = agent.generate_alert(market, forecast, anomaly)
    assert not result["llm_used"]
    assert result["guardrail_results"]["failures"]
    assert result["alert"]["risk_tier"] == "Critical"


def test_generate_alert_always_schema_valid_with_mock_llm(tmp_path):
    market, forecast, anomaly = _inputs("Normal")
    signal = build_signal(market, forecast, anomaly)
    agent = VoltexAgent(retriever=StubRetriever(), audit_logger=AuditLogger(tmp_path / "audit.sqlite"),
                        llm_callable=lambda _prompt: _valid_response(signal))
    result = agent.generate_alert(market, forecast, anomaly)
    assert result["llm_used"]
    assert result["guardrail_results"]["passed"]
    assert result["alert"]["risk_tier"] == "Normal"
