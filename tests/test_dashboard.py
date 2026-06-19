from src.dashboard.data import available_dates, load_audit, load_backtest, load_validation, replay_day


def test_dashboard_artifact_loaders():
    assert len(load_backtest()) == 6
    assert load_validation()["gates"]
    assert "2015-08-24" in available_dates()


def test_black_monday_replay_is_complete():
    row = replay_day("2015-08-24")
    assert row["market_risk_tier"] == "Critical"
    assert row["alert"] and row["guardrails"]


def test_audit_log_loader_graceful():
    assert set(["timestamp", "date", "tier", "path", "guardrails", "latency_ms"]).issubset(load_audit().columns)
