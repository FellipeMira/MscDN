"""
pipeline.checks
===============
Alignment assertions, leakage unit tests, and data-quality guards.

Can be run standalone (``python -m pipeline.checks``) or imported.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd


# ======================================================================== #
#  Alignment checks                                                         #
# ======================================================================== #

def assert_alignment(ref_meta: dict, file_paths: List[str]) -> None:
    """
    Re-validate alignment using cached reference metadata.
    Lightweight (no rasterio open) — just checks meta dict equality.
    """
    from .ingestion import _extract_meta

    for path in file_paths:
        meta = _extract_meta(path)
        for key in ("crs", "width", "height"):
            assert meta[key] == ref_meta[key], (
                f"Alignment mismatch on {os.path.basename(path)}: "
                f"{key} = {meta[key]} != {ref_meta[key]}"
            )


# ======================================================================== #
#  Temporal leakage checks                                                  #
# ======================================================================== #

def assert_no_future_in_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str = "time",
    target_col: Optional[str] = None,
) -> None:
    """
    Verify that no feature column is correlated with future information.

    Heuristic checks:
    1. Feature column names must not contain 'target' or 'future'.
    2. If a target column is provided, no feature should have correlation > 0.99
       with the target (which would suggest it *is* the target leaked in).
    """
    # Name-based check
    suspicious = [
        c for c in feature_cols
        if any(kw in c.lower() for kw in ("target", "future", "label"))
    ]
    assert not suspicious, (
        f"Suspicious feature names (possible leakage): {suspicious}"
    )

    # Correlation-based check
    if target_col is not None and target_col in df.columns:
        numeric_feats = [c for c in feature_cols if c in df.select_dtypes("number").columns]
        if numeric_feats:
            corrs = df[numeric_feats].corrwith(df[target_col]).abs()
            high = corrs[corrs > 0.99]
            if not high.empty:
                warnings.warn(
                    f"Features with suspiciously high correlation to target: "
                    f"{high.to_dict()}"
                )


def assert_lag_integrity(
    df: pd.DataFrame,
    P: int,
    time_col: str = "time",
    pixel_col: str = "pixel_id",
) -> None:
    """
    For a random subsample of pixel groups, verify that lag columns
    correspond to actual past values.
    """
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    if not lag_cols:
        return

    # Sample a few pixel groups
    pixels = df[pixel_col].unique()
    rng = np.random.default_rng(42)
    sample_px = rng.choice(pixels, size=min(5, len(pixels)), replace=False)

    for px in sample_px:
        sub = df[df[pixel_col] == px].sort_values(time_col).reset_index(drop=True)
        if len(sub) < P + 1:
            continue
        # Check that lag_1 at row i equals value at row i-1
        for col in lag_cols:
            parts = col.split("_")
            # Expected format: lag_{k}_{band}
            if len(parts) >= 3:
                lag_k = int(parts[1])
                band = "_".join(parts[2:])
                if band in sub.columns:
                    for i in range(lag_k, min(lag_k + 3, len(sub))):
                        expected = sub.loc[i - lag_k, band]
                        actual = sub.loc[i, col]
                        if pd.notna(actual) and pd.notna(expected):
                            assert np.isclose(actual, expected, rtol=1e-5), (
                                f"Lag integrity failure: pixel={px}, row={i}, "
                                f"col={col}: expected {expected}, got {actual}"
                            )


def assert_temporal_ordering(
    train_times: pd.Series,
    val_times: pd.Series,
    test_times: pd.Series,
) -> None:
    """Strict temporal ordering: max(train) < min(val) < max(val) < min(test)."""
    t_tr = pd.to_datetime(train_times)
    t_va = pd.to_datetime(val_times)
    t_te = pd.to_datetime(test_times)

    assert t_tr.max() < t_va.min(), (
        f"Train/val overlap: train max={t_tr.max()}, val min={t_va.min()}"
    )
    assert t_va.max() < t_te.min(), (
        f"Val/test overlap: val max={t_va.max()}, test min={t_te.min()}"
    )
    print("  ✓ Temporal ordering verified.")


# ======================================================================== #
#  Data quality                                                             #
# ======================================================================== #

def assert_no_nan_in_features(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """Verify that the final feature matrix has no NaN values."""
    nan_counts = df[feature_cols].isna().sum()
    has_nan = nan_counts[nan_counts > 0]
    assert has_nan.empty, f"NaN values in features:\n{has_nan}"


def assert_target_range(y: pd.Series, lo: float = 0.0, hi: float = 1.0) -> None:
    """Target values must be in [lo, hi]."""
    valid = y.dropna()
    assert valid.min() >= lo, f"Target below {lo}: min={valid.min()}"
    assert valid.max() <= hi, f"Target above {hi}: max={valid.max()}"


# ======================================================================== #
#  Run all checks                                                           #
# ======================================================================== #

def run_all_checks(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_mask: pd.Series,
    val_mask: pd.Series,
    test_mask: pd.Series,
    P: int,
) -> None:
    """Run the full battery of data-quality and leakage checks."""
    print("Running data quality and leakage checks...")

    assert_no_future_in_features(df, feature_cols, target_col=target_col)
    assert_lag_integrity(df, P)
    assert_temporal_ordering(
        df.loc[train_mask, "time"],
        df.loc[val_mask, "time"],
        df.loc[test_mask, "time"],
    )
    assert_no_nan_in_features(df, feature_cols)
    assert_target_range(df[target_col])

    print("  ✓ All data quality checks passed.")


if __name__ == "__main__":
    print("pipeline.checks — import and call run_all_checks() from your pipeline.")
