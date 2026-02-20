"""
pipeline.export
===============
Dataset export, metadata logging, and reproducibility utilities.

Responsibilities
----------------
* Save DataFrames as Parquet (primary) and optionally CSV.
* Write experiment metadata JSON (files used, hashes, P, Q, config, …).
* Propose folder structure and naming conventions.
* Generate the summary results table (one row per run).

Folder layout
-------------
::

    experiments/
        pipeline_config.json          ← master config snapshot
        summary_results.csv           ← aggregated metrics across all runs
        cache/
            base_table_P{maxP}.parquet
            p95_train.parquet
        exp_P{P}_Q{Q}/
            metadata.json
            test_set.parquet
            train_set.parquet          (optional, toggled by flag)
            predictions.parquet
            model.joblib               (optional)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig, file_hash, get_environment_info


# ======================================================================== #
#  Save datasets                                                            #
# ======================================================================== #

def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    fmt: str = "parquet",
) -> str:
    """
    Save a DataFrame to disk.

    Parameters
    ----------
    df : pd.DataFrame
    path : str or Path
        Target file path (extension will be corrected).
    fmt : str
        ``'parquet'`` (default) or ``'csv'``.

    Returns
    -------
    str  – actual path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        path = path.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        path = path.with_suffix(".csv")
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return str(path)


# ======================================================================== #
#  Experiment metadata                                                      #
# ======================================================================== #

def save_experiment_metadata(
    exp_dir: str | Path,
    P: int,
    Q: int,
    config: PipelineConfig,
    feature_meta: Dict,
    target_meta: Dict,
    metrics: Dict[str, Dict],
    files_used: Optional[List[str]] = None,
    extra: Optional[Dict] = None,
) -> str:
    """
    Write a comprehensive ``metadata.json`` for one (P, Q, model) run.

    Returns the path of the written file.
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "P": P,
        "Q": Q,
        "pipeline_config": config.to_dict(),
        "feature_meta": _make_serialisable(feature_meta),
        "target_meta": _make_serialisable(target_meta),
        "metrics": _make_serialisable(metrics),
        "environment": get_environment_info(),
    }

    if files_used:
        meta["files_used"] = {
            os.path.basename(f): file_hash(f)
            for f in files_used
            if os.path.exists(f)
        }

    if extra:
        meta.update(_make_serialisable(extra))

    out_path = exp_dir / "metadata.json"
    out_path.write_text(json.dumps(meta, indent=2, default=str))
    return str(out_path)


# ======================================================================== #
#  Summary table                                                            #
# ======================================================================== #

def update_summary_table(
    output_dir: str | Path,
    row: Dict[str, Any],
    filename: str = "summary_results.csv",
) -> pd.DataFrame:
    """
    Append a result row to the summary CSV.  Creates the file if needed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / filename

    new_row = pd.DataFrame([row])

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(csv_path, index=False)
    return combined


# ======================================================================== #
#  Cache helpers                                                            #
# ======================================================================== #

def get_cache_path(output_dir: str | Path, name: str) -> Path:
    """Return path inside the experiment cache directory."""
    p = Path(output_dir) / "cache" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cache_exists(output_dir: str | Path, name: str) -> bool:
    return get_cache_path(output_dir, name).exists()


def save_cache(df: pd.DataFrame, output_dir: str | Path, name: str) -> str:
    path = get_cache_path(output_dir, name)
    df.to_parquet(path, index=False)
    return str(path)


def load_cache(output_dir: str | Path, name: str) -> pd.DataFrame:
    path = get_cache_path(output_dir, name)
    return pd.read_parquet(path)


# ======================================================================== #
#  Internal helpers                                                         #
# ======================================================================== #

def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
