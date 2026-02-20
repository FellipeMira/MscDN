"""
pipeline.targets
================
Target construction for spatio-temporal extreme-precipitation forecasting.

The model predicts a **Valor de Probabilidade (VP)** in [0, 1] that encodes
"how extreme" the precipitation event will be Q steps ahead, anchored on
three fixed climatological percentiles computed **once** from the 30-year
ERA5 cube:

  +-----------+--------+
  | r_future  |   VP   |
  +===========+========+
  | ≤ P90     |  0.0   |
  | = P95     |  0.5   |
  | ≥ P100    |  1.0   |
  +-----------+--------+

Three mapping functions are provided
--------------------------------------

**piecewise** — two-segment linear interpolation with hard saturation.

  .. math::
      VP =
      \\begin{cases}
         0     & x \\le P90 \\\\
         0.5\\,\\dfrac{x - P90}{P95 - P90} & P90 < x < P95 \\\\
         0.5 + 0.5\\,\\dfrac{x - P95}{P100 - P95} & P95 \\le x < P100 \\\\
         1     & x \\ge P100
      \\end{cases}

**sigmoid** — logistic curve centred on P95, steepness *k* calibrated so
  that VP(P90) ≈ ε (configurable, default ε = 0.02) and clipped to
  exact 0/1 at P90/P100.

  .. math::
      VP_{raw} = \\frac{1}{1 + e^{-k(x - P95)}} \\quad\\text{where }
      k = \\frac{\\ln(1/\\varepsilon - 1)}{P95 - P90}

  Followed by saturation:  VP = clip(VP_raw, lo=VP_raw(P90)→0, hi=VP_raw(P100)→1)

**tanh** — hyperbolic-tangent mapping scaled to pass through the three
  anchor points:

  .. math::
      VP = 0.5 + 0.5 \\cdot \\tanh\\!\\Bigl(
          \\alpha \\;\\frac{x - P95}{P100 - P90}
      \\Bigr)

  with α calibrated so that VP(P90) ≈ ε.

Percentile computation
----------------------
``ensure_percentile_rasters()`` computes P90, P95, P100 **once** from the
30-year NetCDF cube, saves them as GeoTIFFs + a JSON metadata sidecar, and
skips on subsequent runs (unless ``force=True``).

Public API
----------
ensure_percentile_rasters(cube_nc, out_dir, ...)  → dict[str, Path]
load_percentile_grids(out_dir, stride)             → pd.DataFrame
compute_vp(precip, p90, p95, p100, method, eps)    → np.ndarray
build_target(df, Q, thresholds, target_config, band) → (y, meta)
add_target_columns(df, q_values, thresholds, target_config, band) → (df, meta)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import FuzzyConfig, TargetMode


# ======================================================================== #
#  Percentile computation — one-time from 30-year climatological cube       #
# ======================================================================== #

_PERCENTILE_LEVELS = {"p90": 0.90, "p95": 0.95, "p100": 1.00}


def ensure_percentile_rasters(
    cube_nc: str | Path,
    out_dir: str | Path,
    variable: str = "precipitation",
    force: bool = False,
) -> Dict[str, Path]:
    """
    Compute per-pixel P90, P95 and P100 from the 30-year NetCDF cube and
    save each as a GeoTIFF + a JSON metadata sidecar.

    **Skips entirely** if all three rasters already exist (unless
    *force=True*).

    Parameters
    ----------
    cube_nc : path
        Path to the climatological NetCDF cube
        (e.g. ``data/raster/cubo_precipitacao.nc``).
    out_dir : path
        Directory where the GeoTIFFs will be written.
    variable : str
        Name of the precipitation variable inside the NetCDF.
        Auto-detected if the dataset has a single data variable.
    force : bool
        Recompute even if the rasters already exist.

    Returns
    -------
    dict
        ``{'p90': Path, 'p95': Path, 'p100': Path, 'metadata': Path}``
    """
    import xarray as xr
    import rioxarray  # noqa: F401

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cube_nc = Path(cube_nc)

    paths: Dict[str, Path] = {
        tag: out_dir / f"{tag}_precipitacao.tif" for tag in _PERCENTILE_LEVELS
    }
    meta_path = out_dir / "thresholds_extremos.json"
    paths["metadata"] = meta_path

    # ---- Skip if all exist ----
    if not force and all(p.exists() for p in paths.values()):
        print("  ✓ Percentile rasters already exist — skipping computation.")
        return paths

    if not cube_nc.exists():
        raise FileNotFoundError(
            f"Climatological cube not found: {cube_nc}\n"
            "Run utils/proc_resposta.py first to build the 30-year cube."
        )

    print(f"  Computing P90 / P95 / P100 from: {cube_nc}")
    ds = xr.open_dataset(str(cube_nc), chunks={"time": -1})

    # Auto-detect precipitation variable
    if variable in ds.data_vars:
        da = ds[variable]
    else:
        da = ds[list(ds.data_vars)[0]]
        print(f"    (auto-detected variable: '{da.name}')")

    lat_dim = "latitude" if "latitude" in da.dims else "y"
    lon_dim = "longitude" if "longitude" in da.dims else "x"
    da = da.chunk({"time": -1, lat_dim: 50, lon_dim: 50})

    var_name = da.name if hasattr(da, "name") else variable

    for tag, q in _PERCENTILE_LEVELS.items():
        print(f"    → {tag} (quantile {q}) …")
        if q >= 1.0:
            # P100 = max over time
            grid = da.max(dim="time", skipna=True).compute()
        else:
            grid = da.quantile(q, dim="time", skipna=True).compute()

        # Drop quantile coordinate if present
        if "quantile" in grid.coords:
            grid = grid.drop_vars("quantile")

        grid.rio.write_crs("EPSG:4326", inplace=True)
        grid.attrs["units"] = "mm"
        grid.attrs["long_name"] = f"{tag} of precipitation (30-year)"

        # rioxarray expects x / y dimension names
        rename_map = {}
        if "longitude" in grid.dims:
            rename_map["longitude"] = "x"
        if "latitude" in grid.dims:
            rename_map["latitude"] = "y"
        if rename_map:
            grid = grid.rename(rename_map)

        tif_path = paths[tag]
        if tif_path.exists():
            tif_path.unlink()
        grid.rio.to_raster(str(tif_path))
        print(f"      saved → {tif_path}")

    ds.close()

    # ---- JSON metadata sidecar ----
    meta = {
        "description": (
            "Fixed climatological precipitation thresholds (P90/P95/P100)"
        ),
        "source_cube": str(cube_nc),
        "variable": var_name,
        "period": "30-year ERA5-Land",
        "created_at": datetime.now().isoformat(),
        "files": {tag: str(paths[tag]) for tag in _PERCENTILE_LEVELS},
        "quantiles": {tag: q for tag, q in _PERCENTILE_LEVELS.items()},
        "crs": "EPSG:4326",
        "spatial_resolution": "same as source cube",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"  ✓ Metadata saved → {meta_path}")

    return paths


# ---------------------------------------------------------------------- #
# Legacy compatibility shim                                                #
# ---------------------------------------------------------------------- #

def ensure_p95_raster(
    cube_nc: str | Path,
    p95_tif: str | Path,
    variable: str = "precipitation",
    force: bool = False,
) -> Path:
    """
    **Legacy wrapper** — calls :func:`ensure_percentile_rasters` and
    returns the P95 path for backward compatibility.
    """
    out_dir = Path(p95_tif).parent
    paths = ensure_percentile_rasters(
        cube_nc, out_dir, variable=variable, force=force,
    )
    return paths["p95"]


# ======================================================================== #
#  Percentile grid loading                                                  #
# ======================================================================== #

def load_percentile_grids(
    out_dir: str | Path,
    stride: int = 1,
) -> pd.DataFrame:
    """
    Load the P90 / P95 / P100 GeoTIFFs and return a flat DataFrame with
    columns ``pixel_id, y_idx, x_idx, p90, p95, p100``.

    Parameters
    ----------
    out_dir : path
        Directory containing ``p90_precipitacao.tif``, etc.
    stride : int
        Spatial sub-sampling factor (1 = keep every pixel).

    Returns
    -------
    pd.DataFrame
    """
    import xarray as xr
    import rioxarray  # noqa: F401

    out_dir = Path(out_dir)
    result_parts: Dict[str, np.ndarray] = {}
    shape = None

    for tag in _PERCENTILE_LEVELS:
        tif_path = out_dir / f"{tag}_precipitacao.tif"
        if not tif_path.exists():
            raise FileNotFoundError(
                f"Percentile raster not found: {tif_path}\n"
                "Run  ensure_percentile_rasters()  or  "
                "python utils/compute_thresholds.py  first."
            )
        ds = xr.open_dataset(str(tif_path), engine="rasterio")
        var = list(ds.data_vars)[0]
        da = ds[var].squeeze(drop=True)
        vals = da.values[::stride, ::stride]
        result_parts[tag] = vals.ravel().astype(np.float32)
        shape = vals.shape
        ds.close()

    assert shape is not None
    ny, nx = shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    pixel_id = (yy * nx + xx).ravel()

    return pd.DataFrame({
        "pixel_id": pixel_id,
        "y_idx": yy.ravel(),
        "x_idx": xx.ravel(),
        "p90": result_parts["p90"],
        "p95": result_parts["p95"],
        "p100": result_parts["p100"],
    })


# ======================================================================== #
#  VP mapping functions — the core maths                                    #
# ======================================================================== #

def _vp_piecewise(
    precip: np.ndarray,
    p90: np.ndarray,
    p95: np.ndarray,
    p100: np.ndarray,
) -> np.ndarray:
    """
    Two-segment piecewise-linear mapping with hard saturation.

    VP(P90) = 0    VP(P95) = 0.5    VP(P100) = 1
    Below P90 → 0 ;  Above P100 → 1.
    """
    x = np.asarray(precip, dtype=np.float64)
    lo = np.asarray(p90, dtype=np.float64)
    mid = np.asarray(p95, dtype=np.float64)
    hi = np.asarray(p100, dtype=np.float64)

    vp = np.where(
        x <= lo,
        0.0,
        np.where(
            x < mid,
            0.5 * (x - lo) / np.maximum(mid - lo, 1e-12),
            np.where(
                x < hi,
                0.5 + 0.5 * (x - mid) / np.maximum(hi - mid, 1e-12),
                1.0,
            ),
        ),
    )
    return np.clip(vp, 0.0, 1.0)


def _vp_sigmoid(
    precip: np.ndarray,
    p90: np.ndarray,
    p95: np.ndarray,
    p100: np.ndarray,
    eps: float = 0.02,
) -> np.ndarray:
    """
    Logistic sigmoid centred on P95, steepness calibrated so that
    σ(k·(P90 − P95)) ≈ ε.

    k = ln(1/ε − 1) / (P95 − P90)

    Then hard-saturate:  x ≤ P90 → 0  ;  x ≥ P100 → 1.
    """
    x = np.asarray(precip, dtype=np.float64)
    lo = np.asarray(p90, dtype=np.float64)
    mid = np.asarray(p95, dtype=np.float64)
    hi = np.asarray(p100, dtype=np.float64)

    # Per-pixel steepness
    delta = np.maximum(mid - lo, 1e-12)
    k = np.log(1.0 / eps - 1.0) / delta

    z = k * (x - mid)
    # Clamp z to prevent overflow in exp
    z = np.clip(z, -50.0, 50.0)
    vp_raw = 1.0 / (1.0 + np.exp(-z))

    # Hard saturation at the anchor boundaries
    vp = np.where(x <= lo, 0.0, np.where(x >= hi, 1.0, vp_raw))
    return np.clip(vp, 0.0, 1.0)


def _vp_tanh(
    precip: np.ndarray,
    p90: np.ndarray,
    p95: np.ndarray,
    p100: np.ndarray,
    eps: float = 0.02,
) -> np.ndarray:
    """
    Hyperbolic tangent mapping:

        VP = 0.5 + 0.5 · tanh(α · (x − P95) / (P100 − P90))

    α calibrated so VP(P90) ≈ ε  →  α = arctanh(1−2ε) · (P100−P90) / (P95−P90).
    Hard saturation at P90 / P100.
    """
    x = np.asarray(precip, dtype=np.float64)
    lo = np.asarray(p90, dtype=np.float64)
    mid = np.asarray(p95, dtype=np.float64)
    hi = np.asarray(p100, dtype=np.float64)

    span = np.maximum(hi - lo, 1e-12)
    delta = np.maximum(mid - lo, 1e-12)

    # arctanh(1 - 2ε) — scalar
    atanh_val = np.arctanh(1.0 - 2.0 * eps)
    alpha = atanh_val * span / delta

    z = alpha * (x - mid) / span
    vp_raw = 0.5 + 0.5 * np.tanh(z)

    vp = np.where(x <= lo, 0.0, np.where(x >= hi, 1.0, vp_raw))
    return np.clip(vp, 0.0, 1.0)


# Dispatcher
_VP_METHODS = {
    TargetMode.PIECEWISE: _vp_piecewise,
    TargetMode.SIGMOID: _vp_sigmoid,
    TargetMode.TANH: _vp_tanh,
}


def compute_vp(
    precip: pd.Series | np.ndarray,
    p90: pd.Series | np.ndarray,
    p95: pd.Series | np.ndarray,
    p100: pd.Series | np.ndarray,
    method: TargetMode = TargetMode.SIGMOID,
    eps: float = 0.02,
) -> np.ndarray:
    """
    Map raw precipitation values to VP ∈ [0, 1] using climatological
    thresholds P90 / P95 / P100.

    Parameters
    ----------
    precip : array-like
        Raw precipitation values (e.g. at horizon Q).
    p90, p95, p100 : array-like
        Per-row climatological thresholds (same length as *precip*).
    method : TargetMode
        ``PIECEWISE``, ``SIGMOID`` or ``TANH``.
    eps : float
        Small value controlling how close to 0 the curve gets at P90
        (only used by sigmoid / tanh).

    Returns
    -------
    np.ndarray  — VP values clipped to [0, 1].
    """
    fn = _VP_METHODS.get(method)
    if fn is None:
        raise ValueError(
            f"Unknown VP method '{method}'. "
            f"Choose from {list(_VP_METHODS.keys())}."
        )
    if method == TargetMode.PIECEWISE:
        return fn(precip, p90, p95, p100)
    return fn(precip, p90, p95, p100, eps=eps)


# ======================================================================== #
#  Row-level threshold lookup (align per-pixel grids with tabular df)       #
# ======================================================================== #

def align_thresholds(
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the per-pixel threshold grid with the main DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``pixel_id``.
    thresholds : pd.DataFrame
        As returned by :func:`load_percentile_grids` — must contain
        ``pixel_id, p90, p95, p100``.

    Returns
    -------
    pd.DataFrame with columns ``p90, p95, p100`` aligned to *df*.
    """
    merged = df[["pixel_id"]].merge(
        thresholds[["pixel_id", "p90", "p95", "p100"]],
        on="pixel_id",
        how="left",
    )
    return merged[["p90", "p95", "p100"]]


# ======================================================================== #
#  Target construction                                                      #
# ======================================================================== #

def build_target(
    df: pd.DataFrame,
    Q: int,
    thresholds_aligned: pd.DataFrame,
    target_config: FuzzyConfig,
    band: str = "tp",
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Build the VP response variable from the precipitation band.

    Steps
    -----
    1. Shift the precipitation column by ``-Q`` (future value at t + Q).
    2. Apply the configured VP mapping using the aligned P90/P95/P100.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted by ``[pixel_id, time]``, containing the precipitation column.
    Q : int
        Prediction horizon (time-steps ahead).
    thresholds_aligned : pd.DataFrame
        Columns ``p90, p95, p100`` aligned row-by-row with *df*
        (use :func:`align_thresholds`).
    target_config : FuzzyConfig
        VP method selection and epsilon.
    band : str
        Column name for raw precipitation.

    Returns
    -------
    y : pd.Series
        VP values in [0, 1].  NaN where the future is unavailable.
    target_meta : dict
    """
    grouped = df.groupby("pixel_id", group_keys=False)
    future_precip = grouped[band].shift(-Q)

    vp = compute_vp(
        future_precip.values,
        thresholds_aligned["p90"].values,
        thresholds_aligned["p95"].values,
        thresholds_aligned["p100"].values,
        method=target_config.mode,
        eps=target_config.eps,
    )

    y = pd.Series(vp, index=df.index, name=f"target_q{Q}")

    # Re-inject NaN where future_precip was NaN (last Q rows per pixel)
    y[future_precip.isna()] = np.nan

    meta = {
        "Q": Q,
        "target_mode": target_config.mode.value,
        "eps": target_config.eps,
        "band": band,
        "percentile_source": "external_fixed",
        "n_valid": int(y.notna().sum()),
        "n_above_05": int((y > 0.5).sum()) if y.notna().any() else 0,
        "vp_mean": float(y.dropna().mean()) if y.notna().any() else float("nan"),
    }
    return y, meta


# ======================================================================== #
#  Convenience: add target columns for all Q values                         #
# ======================================================================== #

def add_target_columns(
    df: pd.DataFrame,
    q_values: List[int],
    thresholds_aligned: pd.DataFrame,
    target_config: FuzzyConfig,
    band: str = "tp",
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Add ``target_q{Q}`` columns for every Q in *q_values*.

    Modifies *df* in-place and returns it together with per-Q metadata.
    """
    all_meta: Dict[str, Dict] = {}
    for q in q_values:
        col = f"target_q{q}"
        y, meta = build_target(
            df, q, thresholds_aligned, target_config, band=band,
        )
        df[col] = y
        all_meta[col] = meta
    return df, all_meta


# ======================================================================== #
#  Legacy compatibility — compute_p95                                       #
# ======================================================================== #

def compute_p95(
    df: pd.DataFrame,
    train_mask: pd.Series,
    band: str = "tp",
    method: str = "external",
    external_p95: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    **Legacy shim** — kept for backward compatibility.

    Prefer :func:`load_percentile_grids` + :func:`align_thresholds` instead.
    """
    if method == "external":
        if external_p95 is None:
            raise ValueError(
                "external_p95 DataFrame required for method='external'."
            )
        merged = df[["pixel_id"]].merge(
            external_p95[["pixel_id", "p95"]], on="pixel_id", how="left",
        )
        return merged["p95"]

    if method == "train":
        train_data = df.loc[train_mask, ["pixel_id", band]]
        pixel_p95 = (
            train_data.groupby("pixel_id")[band]
            .quantile(0.95)
            .rename("p95")
            .reset_index()
        )
        merged = df[["pixel_id"]].merge(pixel_p95, on="pixel_id", how="left")
        return merged["p95"]

    raise ValueError(f"Unknown method='{method}'.")
