"""
pipeline.targets
================
Target construction for spatio-temporal precipitation forecasting.

Two modes
---------
**Option A — Fuzzy threshold (``fuzzy``):**
  Triangular / trapezoidal fuzzy membership around the per-pixel p95
  reference value:

  .. math::
      y = \\min\\!\\Big(1,\\; \\max\\!\\Big(0,\\;
          \\frac{x - (p_{95} - w)}{w}\\Big)\\Big)

  where *w* controls the fuzzy transition width.

**Option B — Sigmoid above p95 (``sigmoid``):**
  Smooth logistic mapping centred on p95:

  .. math::
      y = \\sigma\\bigl(k \\cdot (x - (p_{95} + \\text{offset}))\\bigr)
        = \\frac{1}{1 + e^{-k(x - p_{95} - \\text{offset})}}

  Parameters *k* (slope / steepness) and *offset* are configurable.

Interpretation
--------------
Both options produce a **continuous score in [0, 1]** which can serve as:

* **Probabilistic regression** target  (use MSE / Brier loss).
* **Soft classification** after thresholding at 0.5  (use binary cross-entropy).
* **Multi-head** output where one head predicts the score and another the
  binary label (optional advanced usage).

p95 computation rule
--------------------
The **recommended** approach is to compute p95 **once** from a long-term
climatological record (e.g. the 30-year ERA5 NetCDF cube) and save it as a
fixed GeoTIFF artefact (``p95_precipitacao.tif``).  Every subsequent
experiment loads this raster — no recomputation, perfect consistency.

Alternatively p95 can be computed from the training split only or from an
already-existing external raster.

Public API
----------
ensure_p95_raster(cube_nc, p95_tif, ...)     → Path  (compute once, skip if exists)
compute_p95(df, train_mask, method)          → p95_series
build_target(df, Q, p95, fuzzy_config)       → (y_series, target_meta)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import FuzzyConfig, TargetMode


# ======================================================================== #
#  p95 — one-time climatological computation from NetCDF cube               #
# ======================================================================== #

def ensure_p95_raster(
    cube_nc: str | Path,
    p95_tif: str | Path,
    variable: str = "precipitation",
    force: bool = False,
) -> Path:
    """
    Compute the per-pixel P95 from the 30-year NetCDF cube and save it as
    a GeoTIFF.  **If the GeoTIFF already exists the computation is skipped
    entirely** (unless *force=True*).

    This function is meant to be called **once** during project setup;
    all subsequent pipeline runs simply load the resulting raster.

    Parameters
    ----------
    cube_nc : path
        Path to the climatological NetCDF cube
        (e.g. ``data/raster/cubo_precipitacao.nc``).
    p95_tif : path
        Destination GeoTIFF for the per-pixel P95 grid
        (e.g. ``data/raster/p95_precipitacao.tif``).
    variable : str
        Name of the precipitation variable inside the NetCDF.
        Autodetected if there is only one data-var.
    force : bool
        Recompute even if the GeoTIFF already exists.

    Returns
    -------
    Path
        The (existing or newly-created) GeoTIFF path.
    """
    import xarray as xr
    import rioxarray  # noqa: F401

    p95_tif = Path(p95_tif)

    # ---- Skip if already exists ----
    if p95_tif.exists() and not force:
        print(f"  ✓ P95 raster already exists — skipping computation: {p95_tif}")
        return p95_tif

    cube_nc = Path(cube_nc)
    if not cube_nc.exists():
        raise FileNotFoundError(
            f"Climatological cube not found: {cube_nc}\n"
            "Run utils/proc_resposta.py first to build the 30-year cube."
        )

    print(f"  Computing P95 from climatological cube: {cube_nc}")

    # Open with Dask for memory safety
    ds = xr.open_dataset(str(cube_nc), chunks={"time": -1})

    # Auto-detect precipitation variable
    if variable in ds.data_vars:
        da = ds[variable]
    else:
        da = ds[list(ds.data_vars)[0]]
        print(f"    (auto-detected variable: '{da.name}')")

    # Rechunk: time contiguous per spatial block for quantile efficiency
    lat_dim = "latitude" if "latitude" in da.dims else "y"
    lon_dim = "longitude" if "longitude" in da.dims else "x"
    da = da.chunk({"time": -1, lat_dim: 50, lon_dim: 50})

    print("    Calculating quantile(0.95) over time dimension …")
    p95_map = da.quantile(0.95, dim="time", skipna=True).compute()

    # Ensure CRS
    p95_map.rio.write_crs("EPSG:4326", inplace=True)
    p95_map.attrs["units"] = "mm"
    p95_map.attrs["long_name"] = "95th percentile of precipitation (30-year)"

    # rioxarray prefers x/y dimension names for export
    rename_map = {}
    if "longitude" in p95_map.dims:
        rename_map["longitude"] = "x"
    if "latitude" in p95_map.dims:
        rename_map["latitude"] = "y"
    if rename_map:
        p95_map = p95_map.rename(rename_map)

    p95_tif.parent.mkdir(parents=True, exist_ok=True)
    if p95_tif.exists():
        p95_tif.unlink()

    p95_map.rio.to_raster(str(p95_tif))
    print(f"  ✓ P95 raster saved: {p95_tif}")
    ds.close()
    return p95_tif


# ======================================================================== #
#  p95 — row-level lookup for the tabular pipeline                          #
# ======================================================================== #

def compute_p95(
    df: pd.DataFrame,
    train_mask: pd.Series,
    band: str = "tp",
    method: str = "external",
    external_p95: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Return the per-row P95 reference value aligned with *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format with ``pixel_id`` and a precipitation column.
    train_mask : pd.Series[bool]
        True for rows belonging to the training set.  Used only when
        *method='train'*.
    band : str
        Column name for precipitation values.
    method : str
        ``'external'`` — **recommended** — use a pre-loaded p95 DataFrame
            derived from the 30-year climatological raster.
        ``'train'``    — compute from training rows only (legacy).
    external_p95 : pd.DataFrame, optional
        Must have columns ``pixel_id, p95``.  Required when
        *method='external'*.

    Returns
    -------
    pd.Series
        Per-row p95 value aligned with *df*.
    """
    if method == "external":
        if external_p95 is None:
            raise ValueError(
                "external_p95 DataFrame must be provided when method='external'.\n"
                "Load it with  ingestion.load_p95_grid(cfg.p95_file, stride)."
            )
        merged = df[["pixel_id"]].merge(
            external_p95[["pixel_id", "p95"]], on="pixel_id", how="left"
        )
        return merged["p95"]

    if method == "train":
        print("    ⚠  Computing P95 from training split (legacy mode).")
        train_data = df.loc[train_mask, ["pixel_id", band]]
        pixel_p95 = (
            train_data.groupby("pixel_id")[band]
            .quantile(0.95)
            .rename("p95")
            .reset_index()
        )
        merged = df[["pixel_id"]].merge(pixel_p95, on="pixel_id", how="left")
        return merged["p95"]

    raise ValueError(
        f"Unknown p95 method='{method}'.  Use 'external' or 'train'."
    )


# ======================================================================== #
#  Target construction                                                      #
# ======================================================================== #

def build_target(
    df: pd.DataFrame,
    Q: int,
    p95: pd.Series,
    fuzzy_config: FuzzyConfig,
    band: str = "tp",
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Build the response variable from the precipitation band.

    Steps:
    1. Shift precipitation by ``-Q`` (future value at t+Q).
    2. Apply the configured transformation relative to p95.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by ``[pixel_id, time]`` and contain the precipitation
        band column.
    Q : int
        Prediction horizon (steps ahead).
    p95 : pd.Series
        Per-row p95 reference (aligned with *df*).
    fuzzy_config : FuzzyConfig
        Slope, offset, mode selection.
    band : str
        Column name for raw precipitation.

    Returns
    -------
    y : pd.Series
        Target values in [0, 1].  Contains NaN for rows where the future
        value is unavailable (last Q rows per pixel).
    target_meta : dict
        Metadata about the target construction.
    """
    grouped = df.groupby("pixel_id", group_keys=False)
    future_precip = grouped[band].shift(-Q)

    if fuzzy_config.mode == TargetMode.FUZZY:
        y = _fuzzy_membership(future_precip, p95, fuzzy_config)
    elif fuzzy_config.mode == TargetMode.SIGMOID:
        y = _sigmoid_target(future_precip, p95, fuzzy_config)
    else:
        raise ValueError(f"Unknown target mode: {fuzzy_config.mode}")

    meta = {
        "Q": Q,
        "target_mode": fuzzy_config.mode.value,
        "slope": fuzzy_config.slope,
        "offset": fuzzy_config.offset,
        "band": band,
        "p95_source": fuzzy_config.p95_source,
        "n_valid": int(y.notna().sum()),
        "n_positive_05": int((y > 0.5).sum()) if y.notna().any() else 0,
    }
    return y, meta


# ======================================================================== #
#  Transformation functions                                                 #
# ======================================================================== #

def _sigmoid_target(
    precip: pd.Series,
    p95: pd.Series,
    cfg: FuzzyConfig,
) -> pd.Series:
    """
    Logistic sigmoid centred on p95 + offset.

    .. math::
        y = 1 / (1 + exp(-slope * (precip - (p95 + offset))))
    """
    z = cfg.slope * (precip - (p95 + cfg.offset))
    return 1.0 / (1.0 + np.exp(-z))


def _fuzzy_membership(
    precip: pd.Series,
    p95: pd.Series,
    cfg: FuzzyConfig,
) -> pd.Series:
    """
    Trapezoidal fuzzy membership.

    Parameters
    ----------
    slope : float
        Controls the width of the ramp (larger = sharper transition).
    offset : float
        Shift of the ramp centre relative to p95.

    Returns a score in [0, 1].
    """
    # Ramp width (inverse of slope, clamped)
    w = max(1.0 / cfg.slope, 1e-6)
    centre = p95 + cfg.offset
    score = (precip - (centre - w)) / (2 * w)
    return score.clip(0.0, 1.0)


# ======================================================================== #
#  Convenience: build target column in-place on a DataFrame                 #
# ======================================================================== #

def add_target_columns(
    df: pd.DataFrame,
    q_values: list[int],
    p95: pd.Series,
    fuzzy_config: FuzzyConfig,
    band: str = "tp",
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Add ``target_q{Q}`` columns for every Q in *q_values*.

    Modifies *df* in-place and returns it together with per-Q metadata.
    """
    all_meta = {}
    for q in q_values:
        col = f"target_q{q}"
        y, meta = build_target(df, q, p95, fuzzy_config, band=band)
        df[col] = y
        all_meta[col] = meta
    return df, all_meta
