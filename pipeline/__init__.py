"""
MscDN Pipeline — End-to-end spatio-temporal forecasting pipeline.

Flow:
  raw TIFFs → validation → spatio-temporal cube → feature/target building
  → splits → scaling → training → evaluation → export + logging

Modules
-------
config      : Central configuration dataclasses and constants
ingestion   : parse_timestamp, load_cube, validate_alignment
features    : build_features (lag generation, rolling stats, multi-band)
targets     : build_target (fuzzy p95, sigmoid), p95 computation
splitting   : make_splits, fit_transformers, leakage checks
models/     : Model registry (LSTM, sklearn families, baselines)
evaluation  : Metrics (Brier, RMSE, MAE, R², …) and reporting
experiment  : ExperimentRunner — (P, Q) grid search with caching & logging
export      : Parquet/CSV export, metadata, reproducibility
checks      : Alignment assertions, leakage unit tests
"""

__version__ = "0.2.0"
