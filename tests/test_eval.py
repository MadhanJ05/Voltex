import pandas as pd

from src.eval.backtest import covid_features, run_backtest
from src.eval.gates import evaluate_gates


def test_gate_evaluator_known_inputs():
    results = evaluate_gates(
        {"macro_f1_high_critical": .8, "precision_high_critical": .8, "recall_high_critical": .8,
         "pr_auc_ovr_macro": .8, "roc_auc_ovr_macro": .9},
        {"after": {"brier": .05, "ece": .02}},
        [{"market_risk_tier": "High"}] * 6, 1.0, 4000,
    )
    assert all(item.passed for item in results)


def test_covid_schema_matches_historical_pipeline():
    covid = covid_features()
    historical = pd.read_csv("data/processed/historical_features.csv", nrows=1)
    assert set(covid.columns) == set(historical.columns)
    assert not covid.loc[covid["date"].astype(str).eq("2020-03-09")].empty


def test_backtest_covers_six_events_without_live_dependency():
    rows = run_backtest(live_agent=False)
    assert len(rows) == 6
