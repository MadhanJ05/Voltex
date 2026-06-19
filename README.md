# VOLTEX

VOLTEX is an advisory-only reliability agent for trading-platform SREs. It
separates market-data modelling from the LLM alert layer: raw OHLCV data is
never passed to the LLM.

## Module 1: data and features

The historical pipeline expects Kaggle's `all_stocks_5yr.csv` in `data/`
(the supplied file may have a different filename). It forms an equal-weighted
market proxy from the constituent universe, creates a next-trading-day
prediction table, and applies a strict leakage/correlation gate.

```zsh
/Users/madhan/miniforge3/bin/python -m pip install -r requirements.txt
/Users/madhan/miniforge3/bin/python -m src.data.features \
  --input "data/all_stocks_5yr (1).csv" \
  --output data/processed/historical_features.csv
```

Feature values that depend on market activity are lagged one trading day; the
row date is the day being predicted. `surge_label` is that row's realised
volume-surge outcome. Known-at-scheduling-time day-of-week and macro-event
flags remain aligned to the prediction date.

Run `notebooks/module_1_eda.ipynb` after producing the processed file for the
surge prevalence and crisis volume-z-score plots.

## Module 5 model configuration

The alert agent uses `gemini-2.5-flash` (temperature 0.2, 512-token limit) and
the RAG index uses `gemini-embedding-001` when `GOOGLE_API_KEY` is present.
Without a key, retrieval falls back to a local embedding implementation and
alert generation always uses the deterministic template. Rebuild
`data/chroma/` whenever the embedding model changes: stored document vectors
and query vectors must share the same embedding space.
