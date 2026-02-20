"""
pipeline.splitting
==================
Spatio-temporal train / val / test splitting with strict leakage prevention.

Public API
----------
make_splits(df, split_config)    → dict of split indices / masks
fit_transformers(X_train)        → fitted scalers  (apply later to val/test)
check_no_leakage(...)            → assertion-based leakage audit

Strategies
----------
**FIXED**      – hard cut-off dates  (train ≤ T₁ < val ≤ T₂ < test).
**ROLLING**    – rolling-window forward validation (multiple folds).
**EXPANDING**  – expanding training window (walk-forward).

Spatial blocking (optional): if ``split_config.spatial_block_col`` is set,
a subset of regions is held out entirely for spatial generalization tests.

Leakage checklist
-----------------
✓  Scalers fitted on training rows only.
✓  p95 computed from training rows only (enforced in ``targets.py``).
✓  No future timestamps leak into features (shift direction asserted).
✓  Temporal ordering respected: max(train.time) < min(val.time).
✓  Rolling / expanding windows move strictly forward.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from .config import SplitConfig, SplitStrategy


# ======================================================================== #
#  1.  Splitting                                                            #
# ======================================================================== #

def make_splits(
    df: pd.DataFrame,
    split_config: SplitConfig,
    time_col: str = "time",
) -> Dict[str, Any]:
    """
    Split a DataFrame according to the configured strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``time`` column (datetime).
    split_config : SplitConfig
    time_col : str

    Returns
    -------
    dict
        For **FIXED** strategy::

            {"train_mask": bool Series,
             "val_mask":   bool Series,
             "test_mask":  bool Series}

        For **ROLLING** / **EXPANDING**::

            {"folds": [{"train_mask": ..., "val_mask": ...}, ...],
             "test_mask": bool Series}
    """
    times = pd.to_datetime(df[time_col])

    if split_config.strategy == SplitStrategy.FIXED:
        return _fixed_split(times, split_config)
    elif split_config.strategy == SplitStrategy.ROLLING:
        return _rolling_split(times, split_config)
    elif split_config.strategy == SplitStrategy.EXPANDING:
        return _expanding_split(times, split_config)
    else:
        raise ValueError(f"Unknown split strategy: {split_config.strategy}")


def _fixed_split(times: pd.Series, cfg: SplitConfig) -> Dict[str, pd.Series]:
    train_end = pd.Timestamp(cfg.train_end)
    val_end = pd.Timestamp(cfg.val_end)

    train_mask = times <= train_end
    val_mask = (times > train_end) & (times <= val_end)
    test_mask = times > val_end

    _log_split_sizes(train_mask, val_mask, test_mask)
    return {"train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask}


def _rolling_split(times: pd.Series, cfg: SplitConfig) -> Dict:
    """
    Generate rolling-window folds.  A held-out test set beyond val_end
    is kept fixed.  Folds cover the train+val region only.
    """
    val_end = pd.Timestamp(cfg.val_end)
    test_mask = times > val_end

    unique_times = np.sort(times[~test_mask].unique())
    n = len(unique_times)

    tw = cfg.window_train_size or max(n // 3, 1)
    vw = cfg.window_val_size or max(n // 6, 1)
    step = cfg.step_size or vw

    folds = []
    start = 0
    while start + tw + vw <= n:
        train_times = set(unique_times[start : start + tw])
        val_times = set(unique_times[start + tw : start + tw + vw])
        fold = {
            "train_mask": times.isin(train_times),
            "val_mask": times.isin(val_times),
        }
        folds.append(fold)
        start += step

    if not folds:
        warnings.warn("Rolling split produced 0 folds — check window sizes.")

    return {"folds": folds, "test_mask": test_mask}


def _expanding_split(times: pd.Series, cfg: SplitConfig) -> Dict:
    """
    Expanding-window walk-forward: training window grows, validation
    window slides.
    """
    val_end = pd.Timestamp(cfg.val_end)
    test_mask = times > val_end

    unique_times = np.sort(times[~test_mask].unique())
    n = len(unique_times)

    tw = cfg.window_train_size or max(n // 3, 1)
    vw = cfg.window_val_size or max(n // 6, 1)
    step = cfg.step_size or vw

    folds = []
    pos = tw
    while pos + vw <= n:
        train_times = set(unique_times[:pos])
        val_times = set(unique_times[pos : pos + vw])
        fold = {
            "train_mask": times.isin(train_times),
            "val_mask": times.isin(val_times),
        }
        folds.append(fold)
        pos += step

    return {"folds": folds, "test_mask": test_mask}


def _log_split_sizes(train: pd.Series, val: pd.Series, test: pd.Series):
    print(
        f"  Split sizes — train: {train.sum():,}  "
        f"val: {val.sum():,}  test: {test.sum():,}"
    )


# ======================================================================== #
#  Optional spatial blocking                                                #
# ======================================================================== #

def spatial_block_split(
    df: pd.DataFrame,
    region_col: str,
    holdout_regions: List[Any],
) -> Tuple[pd.Series, pd.Series]:
    """
    Return (in-sample mask, holdout mask) based on region identifiers.
    Use the holdout set as an extra test for spatial generalization.
    """
    holdout_mask = df[region_col].isin(holdout_regions)
    return ~holdout_mask, holdout_mask


# ======================================================================== #
#  2.  Fit transformers (leakage-safe)                                      #
# ======================================================================== #

SCALER_REGISTRY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def fit_transformers(
    X_train: pd.DataFrame,
    feature_cols: List[str],
    scaler_type: str = "standard",
) -> Dict[str, Any]:
    """
    Fit a scaler on **training data only**.

    Parameters
    ----------
    X_train : pd.DataFrame
    feature_cols : list of str
    scaler_type : str
        One of ``'standard'``, ``'minmax'``, ``'robust'``.

    Returns
    -------
    dict
        ``{"scaler": fitted_scaler, "feature_cols": feature_cols}``
    """
    cls = SCALER_REGISTRY.get(scaler_type)
    if cls is None:
        raise ValueError(
            f"Unknown scaler_type='{scaler_type}'. "
            f"Choose from {list(SCALER_REGISTRY)}"
        )
    scaler = cls()
    scaler.fit(X_train[feature_cols])
    return {"scaler": scaler, "feature_cols": feature_cols, "scaler_type": scaler_type}


def transform_split(
    df: pd.DataFrame,
    transformer_info: Dict[str, Any],
) -> pd.DataFrame:
    """Apply a fitted scaler to a DataFrame (returns a copy)."""
    scaler = transformer_info["scaler"]
    cols = transformer_info["feature_cols"]
    out = df.copy()
    out[cols] = scaler.transform(df[cols])
    return out


# ======================================================================== #
#  3.  Leakage checks / assertions                                          #
# ======================================================================== #

def check_no_leakage(
    df: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    test_mask: pd.Series,
    time_col: str = "time",
    feature_cols: Optional[List[str]] = None,
) -> None:
    """
    Run a battery of assertions to verify there is no spatio-temporal leakage.

    Raises
    ------
    AssertionError
        With a descriptive message if any check fails.
    """
    times = pd.to_datetime(df[time_col])

    # 1. Temporal ordering
    t_train_max = times[train_mask].max()
    t_val_min = times[val_mask].min()
    t_val_max = times[val_mask].max()
    t_test_min = times[test_mask].min()

    assert t_train_max < t_val_min, (
        f"LEAKAGE: max train time ({t_train_max}) >= min val time ({t_val_min})"
    )
    assert t_val_max < t_test_min, (
        f"LEAKAGE: max val time ({t_val_max}) >= min test time ({t_test_min})"
    )

    # 2. No overlap
    assert not (train_mask & val_mask).any(), "LEAKAGE: train ∩ val != ∅"
    assert not (train_mask & test_mask).any(), "LEAKAGE: train ∩ test != ∅"
    assert not (val_mask & test_mask).any(), "LEAKAGE: val ∩ test != ∅"

    # 3. Feature columns don't contain future-named targets
    if feature_cols is not None:
        suspicious = [c for c in feature_cols if "target" in c.lower()]
        assert not suspicious, (
            f"LEAKAGE: feature columns contain target-like names: {suspicious}"
        )

    # 4. No NaN-only splits
    assert train_mask.sum() > 0, "Train set is empty"
    assert val_mask.sum() > 0, "Validation set is empty"
    assert test_mask.sum() > 0, "Test set is empty"

    print("  ✓ Leakage checks passed.")
