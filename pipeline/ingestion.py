"""
pipeline.ingestion
==================
Ingest, validate, and concatenate multi-band spatio-temporal GeoTIFF cubes
into a single xarray DataArray / Dataset.

Public API
----------
parse_timestamp(filename)        → datetime
list_sorted_tiffs(input_dir)     → List[(datetime, path)]
validate_alignment(files)        → None  (raises on mismatch)
load_cube(files, ...)            → xr.Dataset
extract_pixel_timeseries(...)    → pd.DataFrame   (pixel-level tabular)
"""

from __future__ import annotations

import glob
import os
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import rioxarray  # noqa: F401  (registers .rio accessor)

from .config import BAND_PATTERN, SHORT_BANDS, VARIABLES


# ======================================================================== #
#  1.  Timestamp parsing                                                    #
# ======================================================================== #

def parse_timestamp(filename: str) -> datetime:
    """
    Extract a timestamp from a GeoTIFF filename.

    Supported patterns (tested in order):
      • GEE band-style : ``M{MM}_D{DD}_{YYYY}_H{HH}``
      • ISO-ish A      : ``era5_cube_YYYYMMDD.tif``  (hour defaults to 00)
      • ISO-ish B      : ``era5_cube_YYYYMMDD_HH.tif``

    Parameters
    ----------
    filename : str
        Full path **or** basename of the GeoTIFF file.

    Returns
    -------
    datetime

    Raises
    ------
    ValueError
        If no timestamp pattern is found.
    """
    base = os.path.basename(filename)

    # Pattern 1 — GEE-style prefix inside band names
    m_gee = re.search(r"M(\d{2})_D(\d{2})_(\d{4})_H(\d{2})", base)
    if m_gee:
        mm, dd, yyyy, hh = m_gee.groups()
        return datetime(int(yyyy), int(mm), int(dd), int(hh))

    # Pattern 2 / 3 — date (+ optional hour) in filename
    m_iso = re.search(r"(\d{8})(?:_(\d{2}))?", base)
    if m_iso:
        ymd = m_iso.group(1)
        hh = m_iso.group(2) or "00"
        return datetime.strptime(f"{ymd}{hh}", "%Y%m%d%H")

    raise ValueError(f"Cannot parse timestamp from filename: {filename}")


# ======================================================================== #
#  2.  File listing                                                         #
# ======================================================================== #

def list_sorted_tiffs(
    input_dir: str | Path,
    max_files: Optional[int] = None,
) -> List[Tuple[datetime, str]]:
    """
    List all ``*.tif`` files in *input_dir*, parse timestamps, sort
    chronologically, and optionally cap the count.

    Returns
    -------
    list of (datetime, filepath) sorted ascending by time.
    """
    files = glob.glob(os.path.join(str(input_dir), "*.tif"))
    parsed: List[Tuple[datetime, str]] = []
    for f in files:
        try:
            parsed.append((parse_timestamp(f), f))
        except ValueError:
            warnings.warn(f"Skipping file with unparseable timestamp: {f}")
    parsed.sort(key=lambda x: x[0])
    if max_files is not None:
        parsed = parsed[:max_files]
    return parsed


# ======================================================================== #
#  3.  Alignment validation                                                 #
# ======================================================================== #

class AlignmentError(Exception):
    """Raised when cubes are spatially misaligned."""


def validate_alignment(files: List[str]) -> Dict:
    """
    Check that all GeoTIFF files share the same CRS, resolution,
    spatial extent, and nodata value.

    Parameters
    ----------
    files : list of str
        Paths to GeoTIFF files.

    Returns
    -------
    dict
        Reference metadata extracted from the first file (crs, transform,
        width, height, nodata, dtype).

    Raises
    ------
    AlignmentError
        With a descriptive message listing every mismatch.
    """
    if not files:
        raise AlignmentError("No files to validate.")

    ref = _extract_meta(files[0])
    errors: List[str] = []

    for path in files[1:]:
        meta = _extract_meta(path)
        fname = os.path.basename(path)
        if meta["crs"] != ref["crs"]:
            errors.append(
                f"{fname}: CRS {meta['crs']} != reference {ref['crs']}"
            )
        if meta["transform"] != ref["transform"]:
            errors.append(
                f"{fname}: transform {meta['transform']} != reference"
            )
        if (meta["width"], meta["height"]) != (ref["width"], ref["height"]):
            errors.append(
                f"{fname}: shape ({meta['height']},{meta['width']}) "
                f"!= reference ({ref['height']},{ref['width']})"
            )
        if meta["nodata"] != ref["nodata"]:
            errors.append(
                f"{fname}: nodata {meta['nodata']} != reference {ref['nodata']}"
            )

    if errors:
        msg = "Spatial alignment check failed:\n  • " + "\n  • ".join(errors)
        raise AlignmentError(msg)

    return ref


def _extract_meta(path: str) -> Dict:
    """Read lightweight rasterio metadata for one file."""
    with rasterio.open(path) as src:
        return {
            "crs": str(src.crs),
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "dtype": str(src.dtypes[0]),
            "count": src.count,
        }


# ======================================================================== #
#  4.  Load cube  (xarray + Dask lazy)                                      #
# ======================================================================== #

def load_cube(
    sorted_files: List[Tuple[datetime, str]],
    bands: Optional[List[str]] = None,
    chunk_xy: int = 128,
    engine: str = "rasterio",
) -> xr.Dataset:
    """
    Lazily load and concatenate multi-band GeoTIFF cubes along the time axis.

    Each file may contain N bands where each band corresponds to one
    (hour, variable) combination exported by GEE.  The function:

    1.  Opens each file with Dask-backed chunks.
    2.  Parses the band description / name to recover (time, variable).
    3.  Reshapes into ``(time, y, x)`` per variable.
    4.  Concatenates all files along ``time``.

    Parameters
    ----------
    sorted_files : list of (datetime, path)
        As returned by :func:`list_sorted_tiffs`.
    bands : list of str, optional
        Subset of short band names to keep (e.g. ``['tp','t2m']``).
        Defaults to all five ERA5 bands.
    chunk_xy : int
        Spatial chunk size (pixels) for Dask.
    engine : str
        xarray backend engine (``'rasterio'`` recommended).

    Returns
    -------
    xr.Dataset
        Dimensions: ``(time, y, x)``
        Data variables: one per short band name.
    """
    if bands is None:
        bands = list(SHORT_BANDS)

    datasets: List[xr.Dataset] = []

    for base_dt, path in sorted_files:
        ds_file = _load_single_cube(
            path, base_dt, bands, chunk_xy=chunk_xy, engine=engine
        )
        if ds_file is not None:
            datasets.append(ds_file)

    if not datasets:
        raise RuntimeError("No valid datasets were loaded from the provided files.")

    cube = xr.concat(datasets, dim="time")
    cube = cube.sortby("time")
    return cube


def _load_single_cube(
    path: str,
    base_dt: datetime,
    bands: List[str],
    chunk_xy: int = 128,
    engine: str = "rasterio",
) -> Optional[xr.Dataset]:
    """
    Load one GeoTIFF and reshape its bands into (time, y, x) per variable.

    Band names follow the GEE convention:
        ``M{MM}_D{DD}_{YYYY}_H{HH}_{band_name}``

    If the band names don't match, fall back to treating the file as
    single-variable (precipitation only) with bands = hourly time steps.
    """
    ds_raw = xr.open_dataset(path, engine=engine, chunks={"x": chunk_xy, "y": chunk_xy})
    var_name = list(ds_raw.data_vars)[0]
    da = ds_raw[var_name]

    # --- Try to parse structured band descriptions ---
    band_descs = _get_band_descriptions(path)

    if band_descs and _has_structured_names(band_descs):
        return _parse_structured_bands(da, band_descs, bands, chunk_xy)

    # --- Fallback: single-variable file (bands = hourly time offsets) ---
    return _parse_simple_bands(da, base_dt, bands, chunk_xy)


def _get_band_descriptions(path: str) -> List[str]:
    """Return band descriptions or band names from a rasterio file."""
    with rasterio.open(path) as src:
        descs = list(src.descriptions)
        if descs and descs[0]:
            return descs
        # Fall back to checking tags
        names = []
        for i in range(1, src.count + 1):
            tags = src.tags(i)
            name = tags.get("DESCRIPTION", tags.get("name", ""))
            names.append(name)
        return names


def _has_structured_names(descs: List[str]) -> bool:
    """Check if at least one description matches the GEE band pattern."""
    pattern = re.compile(BAND_PATTERN)
    for d in descs:
        if d and pattern.search(d):
            return True
    return False


def _parse_structured_bands(
    da: xr.DataArray,
    band_descs: List[str],
    keep_bands: List[str],
    chunk_xy: int,
) -> Optional[xr.Dataset]:
    """Parse GEE-style band descriptions into a (time, y, x) Dataset."""
    pattern = re.compile(BAND_PATTERN)
    records: Dict[datetime, Dict[str, int]] = {}  # dt → {band_name: band_idx}

    for idx, desc in enumerate(band_descs):
        m = pattern.search(desc or "")
        if not m:
            continue
        mm, dd, yyyy, hh, bname = m.groups()
        dt = datetime(int(yyyy), int(mm), int(dd), int(hh))
        records.setdefault(dt, {})[bname] = idx

    if not records:
        return None

    times = sorted(records.keys())
    arr = da.values  # (bands, y, x)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    ny, nx = arr.shape[1], arr.shape[2]
    data_vars = {}

    for bname in keep_bands:
        slices = []
        valid_times = []
        for dt in times:
            bidx = records[dt].get(bname)
            if bidx is not None and bidx < arr.shape[0]:
                slices.append(arr[bidx])
                valid_times.append(dt)
        if slices:
            data_vars[bname] = xr.DataArray(
                np.stack(slices, axis=0),
                dims=["time", "y", "x"],
                coords={"time": valid_times},
            )

    if not data_vars:
        return None
    return xr.Dataset(data_vars)


def _parse_simple_bands(
    da: xr.DataArray,
    base_dt: datetime,
    keep_bands: List[str],
    chunk_xy: int,
) -> Optional[xr.Dataset]:
    """
    Fallback: treat each band index as an hourly offset from base_dt.
    All bands are assumed to be the same variable (precipitation).
    """
    arr = da.values
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]

    nb, ny, nx = arr.shape
    times = [base_dt + timedelta(hours=h) for h in range(nb)]

    # The primary band in single-var files is 'tp'
    target_name = "tp" if "tp" in keep_bands else keep_bands[0]
    ds = xr.Dataset(
        {
            target_name: xr.DataArray(
                arr.astype(np.float32),
                dims=["time", "y", "x"],
                coords={"time": times},
            )
        }
    )
    return ds


# ======================================================================== #
#  5.  Pixel-level timeseries extraction (tabular, memory-friendly)         #
# ======================================================================== #

def extract_pixel_timeseries(
    sorted_files: List[Tuple[datetime, str]],
    stride: int = 1,
    bands: Optional[List[str]] = None,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    """
    Incrementally extract pixel-level timeseries from GeoTIFF cubes.

    This avoids materialising the full cube in memory.  Each file is read,
    spatially subsampled by *stride*, and flattened into long-format rows
    ``(time, pixel_id, y_idx, x_idx, <band_columns>)``.

    Parameters
    ----------
    sorted_files : list of (datetime, path)
    stride : int
        Spatial sub-sampling stride (1 = keep all pixels).
    bands : list of str, optional
        Short band names to extract.  Defaults to ``['tp']``.
    max_files : int, optional
        Cap on number of files to process.

    Returns
    -------
    pd.DataFrame
        Columns: ``time, pixel_id, y_idx, x_idx, <band_1>, <band_2>, ...``
    """
    if bands is None:
        bands = ["tp"]
    if max_files is not None:
        sorted_files = sorted_files[:max_files]

    rows: List[pd.DataFrame] = []
    base_shape = None

    for dt0, path in sorted_files:
        ds = xr.open_dataset(path, engine="rasterio")
        var = list(ds.data_vars)[0]
        da = ds[var]

        y_dim = "y" if "y" in da.dims else "latitude"
        x_dim = "x" if "x" in da.dims else "longitude"
        shape = (da.sizes[y_dim], da.sizes[x_dim])

        if base_shape is None:
            base_shape = shape
        elif shape != base_shape:
            warnings.warn(
                f"Skipping {os.path.basename(path)}: "
                f"shape {shape} != reference {base_shape}"
            )
            continue

        arr = da.values  # (bands, y, x) or (y, x)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        arr = arr[:, ::stride, ::stride]

        nodata = da.attrs.get("_FillValue", None)
        nb, ny, nx = arr.shape
        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        pixel_id = (yy * nx + xx).ravel()

        # Attempt to match bands to arr band indices via band descriptions
        band_descs = _get_band_descriptions(path)
        pattern = re.compile(BAND_PATTERN)

        # Group band indices by time step
        time_groups: Dict[datetime, Dict[str, int]] = {}
        fallback = True

        if band_descs and _has_structured_names(band_descs):
            fallback = False
            for idx, desc in enumerate(band_descs):
                m = pattern.search(desc or "")
                if not m:
                    continue
                mm, dd, yyyy, hh, bname = m.groups()
                dt = datetime(int(yyyy), int(mm), int(dd), int(hh))
                time_groups.setdefault(dt, {})[bname] = idx

        if fallback:
            # Single-band-per-timestep fallback
            for b in range(nb):
                t = dt0 + timedelta(hours=b)
                vals = arr[b, :, :].ravel()
                frame = pd.DataFrame({
                    "time": t,
                    "pixel_id": pixel_id,
                    "y_idx": yy.ravel(),
                    "x_idx": xx.ravel(),
                    "tp": vals.astype(np.float32),
                })
                if nodata is not None:
                    frame = frame[frame["tp"] != nodata]
                rows.append(frame)
        else:
            for dt, band_map in sorted(time_groups.items()):
                frame_data = {
                    "time": dt,
                    "pixel_id": pixel_id,
                    "y_idx": yy.ravel(),
                    "x_idx": xx.ravel(),
                }
                valid_mask = np.ones(pixel_id.shape[0], dtype=bool)
                for bname in bands:
                    bidx = band_map.get(bname)
                    if bidx is not None and bidx < arr.shape[0]:
                        vals = arr[bidx, :, :].ravel().astype(np.float32)
                        frame_data[bname] = vals
                        if nodata is not None:
                            valid_mask &= (vals != nodata)
                    else:
                        frame_data[bname] = np.float32(np.nan)

                frame = pd.DataFrame(frame_data)
                frame = frame[valid_mask]
                rows.append(frame)

        ds.close()

    if not rows:
        raise RuntimeError("No valid data extracted from the provided TIFFs.")

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna()
    return out


# ======================================================================== #
#  6.  p95 grid loader                                                      #
# ======================================================================== #

def load_p95_grid(p95_file: str | Path, stride: int = 1) -> pd.DataFrame:
    """
    Load a per-pixel p95 reference raster and return a flat DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: ``pixel_id, y_idx, x_idx, p95``
    """
    ds = xr.open_dataset(str(p95_file), engine="rasterio")
    var = list(ds.data_vars)[0]
    da = ds[var].squeeze(drop=True)
    p95 = da.values[::stride, ::stride]

    ny, nx = p95.shape
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    pixel_id = (yy * nx + xx).ravel()

    return pd.DataFrame({
        "pixel_id": pixel_id,
        "y_idx": yy.ravel(),
        "x_idx": xx.ravel(),
        "p95": p95.ravel().astype(np.float32),
    })
