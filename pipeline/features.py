"""
pipeline.features
=================
Feature engineering for ML: lag generation, rolling statistics,
multi-band support, and multiple sample modes.

Public API
----------
build_features(df, P, feature_config, sample_mode) → (X_df, feature_meta)

Sample Modes
------------
**Pixel (Mode A)** — one row per (pixel, time).
  • Schema: ``[lag_1_tp, lag_2_tp, ..., roll_mean3_tp, ..., y_idx, x_idx]``
  • Pros : simple, huge N, works with any tabular model.
  • Cons : ignores spatial structure, dataset can be enormous.

**Region (Mode B)** — one row per (region, time) with aggregated stats.
  • Schema: ``[lag_1_tp_mean, lag_1_tp_std, ..., region_id]``
  • Pros : smaller dataset, captures intra-region variability.
  • Cons : loses per-pixel detail, requires polygon GeoJSON.

**Patch (Mode C)** — one sample per (centre-pixel, time) with a spatial
  window tensor ``(P, 2*hw+1, 2*hw+1, n_bands)`` → for CNN / ConvLSTM.
  • Schema: 4-D tensor stored as numpy memmap or zarr.
  • Pros : preserves spatial context for DL.
  • Cons : high memory, only usable with DL architectures.

Convention
----------
Lags are **inclusive of t** (the reference time):
  ``lag_0 = value at t,  lag_1 = value at t-1, … , lag_{P-1} = value at t-(P-1)``

Rolling windows are **strictly past** (no peeking at t):
  ``roll_mean_W`` uses ``[t-W, …, t-1]`` exclusive of t.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import FeatureConfig, SampleMode, SHORT_BANDS


# ======================================================================== #
#  Public entry point                                                       #
# ======================================================================== #

def build_features(
    df: pd.DataFrame,
    P: int,
    feature_config: FeatureConfig,
    sample_mode: SampleMode = SampleMode.PIXEL,
    region_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate feature columns from a long-format pixel timeseries DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``time, pixel_id`` and at least one band
        column (e.g. ``tp, t2m, …``).
    P : int
        Number of lookback time steps (including t).  Features span
        ``[t, t-1, …, t-(P-1)]``.
    feature_config : FeatureConfig
        Detailed feature-engineering settings.
    sample_mode : SampleMode
        PIXEL, REGION, or PATCH.
    region_col : str, optional
        Column that identifies regions (for REGION mode).

    Returns
    -------
    X_df : pd.DataFrame
        Feature matrix.  Retains ``time`` and ``pixel_id`` as index helpers.
    feature_meta : dict
        ``{"feature_names": [...], "P": P, ...}``
    """
    bands = [b for b in feature_config.bands if b in df.columns]
    if not bands:
        raise ValueError(
            f"None of the requested bands {feature_config.bands} "
            f"found in DataFrame columns {list(df.columns)}"
        )

    if sample_mode == SampleMode.REGION:
        return _build_features_region(df, P, feature_config, bands, region_col)
    elif sample_mode == SampleMode.PATCH:
        raise NotImplementedError(
            "Patch mode returns tensors — use build_patch_dataset() instead."
        )
    else:
        return _build_features_pixel(df, P, feature_config, bands)


# ======================================================================== #
#  Mode A — Pixel-level features                                            #
# ======================================================================== #

def _build_features_pixel(
    df: pd.DataFrame,
    P: int,
    cfg: FeatureConfig,
    bands: List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """Create lag + rolling features grouped by pixel_id."""

    df = df.sort_values(["pixel_id", "time"]).copy()
    feature_cols: List[str] = []

    grouped = df.groupby("pixel_id", group_keys=False)

    # --- Lagged values ---
    if cfg.include_lags:
        for lag in range(P):
            for band in bands:
                col = f"lag_{lag}_{band}"
                df[col] = grouped[band].shift(lag)
                feature_cols.append(col)

    # --- Rolling statistics (strictly past: shift(1) then rolling) ---
    if cfg.include_rolling:
        for window in cfg.rolling_windows:
            for band in bands:
                shifted = grouped[band].shift(1)  # exclude t
                for stat in cfg.rolling_stats:
                    col = f"roll_{stat}{window}_{band}"
                    if stat == "mean":
                        df[col] = shifted.rolling(window, min_periods=1).mean()
                    elif stat == "std":
                        df[col] = shifted.rolling(window, min_periods=1).std()
                    elif stat == "min":
                        df[col] = shifted.rolling(window, min_periods=1).min()
                    elif stat == "max":
                        df[col] = shifted.rolling(window, min_periods=1).max()
                    elif stat == "trend":
                        # Linear slope over window (approximated)
                        df[col] = (
                            shifted.rolling(window, min_periods=2)
                            .apply(_linear_slope, raw=True)
                        )
                    else:
                        raise ValueError(f"Unknown rolling stat: {stat}")
                    feature_cols.append(col)

    # --- Spatial coordinates ---
    if cfg.include_spatial_coords:
        for c in ("y_idx", "x_idx"):
            if c in df.columns:
                feature_cols.append(c)

    # --- Drop rows with NaN in feature columns ---
    out = df[["time", "pixel_id"] + feature_cols].dropna(subset=feature_cols)

    meta = {
        "feature_names": feature_cols,
        "P": P,
        "bands": bands,
        "sample_mode": "pixel",
        "n_features": len(feature_cols),
    }
    return out, meta


# ======================================================================== #
#  Mode B — Region-level features                                           #
# ======================================================================== #

def _build_features_region(
    df: pd.DataFrame,
    P: int,
    cfg: FeatureConfig,
    bands: List[str],
    region_col: Optional[str],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Aggregate pixel values per (region, time), then create lag + rolling
    features at the region level.
    """
    if region_col is None:
        raise ValueError("region_col is required for REGION sample mode.")

    agg_dict = {}
    for band in bands:
        for stat in cfg.region_stats:
            agg_dict[f"{band}_{stat}"] = pd.NamedAgg(column=band, aggfunc=stat)

    region_df = df.groupby([region_col, "time"]).agg(**agg_dict).reset_index()
    region_df = region_df.sort_values([region_col, "time"])

    # Now build lags / rolling on the aggregated columns
    agg_bands = [c for c in region_df.columns if c not in (region_col, "time")]
    feature_cols: List[str] = []

    grouped = region_df.groupby(region_col, group_keys=False)

    if cfg.include_lags:
        for lag in range(P):
            for col in agg_bands:
                fname = f"lag_{lag}_{col}"
                region_df[fname] = grouped[col].shift(lag)
                feature_cols.append(fname)

    if cfg.include_rolling:
        for window in cfg.rolling_windows:
            for col in agg_bands:
                shifted = grouped[col].shift(1)
                for stat in cfg.rolling_stats:
                    fname = f"roll_{stat}{window}_{col}"
                    if stat == "mean":
                        region_df[fname] = shifted.rolling(window, min_periods=1).mean()
                    elif stat == "std":
                        region_df[fname] = shifted.rolling(window, min_periods=1).std()
                    elif stat == "min":
                        region_df[fname] = shifted.rolling(window, min_periods=1).min()
                    elif stat == "max":
                        region_df[fname] = shifted.rolling(window, min_periods=1).max()
                    feature_cols.append(fname)

    out = region_df[["time", region_col] + feature_cols].dropna(subset=feature_cols)

    meta = {
        "feature_names": feature_cols,
        "P": P,
        "bands": bands,
        "sample_mode": "region",
        "n_features": len(feature_cols),
    }
    return out, meta


# ======================================================================== #
#  Mode C — Patch / tensor dataset builder (for DL)                         #
# ======================================================================== #

def build_patch_dataset(
    cube: "xr.Dataset",
    P: int,
    feature_config: FeatureConfig,
    times: "pd.DatetimeIndex",
    output_dir: str,
) -> Dict[str, Any]:
    """
    Build 4-D tensors ``(P, H, W, C)`` for every valid (centre-pixel, time).

    Saves arrays to disk as ``.npy`` or zarr to avoid OOM.

    Returns
    -------
    dict  with keys ``tensor_path``, ``index`` (DataFrame mapping sample →
    pixel + time), ``shape``.

    .. note:: This is for CNN / ConvLSTM architectures.
    """
    import os

    hw = feature_config.patch_size
    bands = [b for b in feature_config.bands if b in cube.data_vars]
    n_bands = len(bands)

    arr_dict = {b: cube[b].values for b in bands}  # (T, Y, X)
    T, Y, X = list(arr_dict.values())[0].shape
    time_arr = pd.DatetimeIndex(cube.time.values)

    records = []
    patches = []

    for t_idx, t in enumerate(time_arr):
        if t not in times.values:
            continue
        if t_idx < P - 1:
            continue
        for cy in range(hw, Y - hw):
            for cx in range(hw, X - hw):
                patch = np.empty((P, 2 * hw + 1, 2 * hw + 1, n_bands), dtype=np.float32)
                for lag in range(P):
                    ti = t_idx - lag
                    for bi, bname in enumerate(bands):
                        patch[P - 1 - lag, :, :, bi] = arr_dict[bname][
                            ti, cy - hw : cy + hw + 1, cx - hw : cx + hw + 1
                        ]
                patches.append(patch)
                records.append({"time": t, "cy": cy, "cx": cx, "sample_idx": len(records)})

    os.makedirs(output_dir, exist_ok=True)
    tensor_path = os.path.join(output_dir, "patches.npy")
    tensor = np.stack(patches, axis=0)  # (N, P, H, W, C)
    np.save(tensor_path, tensor)

    index_df = pd.DataFrame(records)
    index_df.to_parquet(os.path.join(output_dir, "patch_index.parquet"), index=False)

    return {
        "tensor_path": tensor_path,
        "index": index_df,
        "shape": tensor.shape,
        "bands": bands,
        "P": P,
        "patch_half_width": hw,
    }


# ======================================================================== #
#  Helpers                                                                  #
# ======================================================================== #

def _linear_slope(arr: np.ndarray) -> float:
    """OLS slope for a 1-D window (used in rolling trend)."""
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    return float(np.polyfit(x, arr, 1)[0])
