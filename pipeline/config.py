"""
pipeline.config
===============
Central configuration: constants, dataclasses, and sane defaults.

Every experiment is fully described by a `PipelineConfig` dataclass that is
serialised alongside results for reproducibility.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np


# ---------------------------------------------------------------------------
# Band / variable catalogue  (GEE ERA5-Land naming)
# ---------------------------------------------------------------------------
VARIABLES: Dict[str, str] = {
    "total_precipitation_hourly": "tp",
    "temperature_2m": "t2m",
    "u_component_of_wind_10m": "u10",
    "v_component_of_wind_10m": "v10",
    "surface_pressure": "sp",
}

SHORT_BANDS: List[str] = list(VARIABLES.values())  # ['tp','t2m','u10','v10','sp']

# The target variable (precipitation)
TARGET_BAND: str = "tp"

# Band name pattern inside a GeoTIFF exported from GEE:
#   M{MM}_D{DD}_{YYYY}_H{HH}_{band_name}
BAND_PATTERN: str = r"M(\d{2})_D(\d{2})_(\d{4})_H(\d{2})_(\w+)"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class SampleMode(str, Enum):
    """How to build one ML sample from the cube."""
    PIXEL = "pixel"        # Mode A — one pixel-time = one row
    REGION = "region"      # Mode B — aggregate over polygon per time step
    PATCH = "patch"        # Mode C — spatial window per time step (for CNN/ConvLSTM)


class TargetMode(str, Enum):
    """VP mapping function for extreme-event target construction."""
    PIECEWISE = "piecewise"  # Two-segment linear with saturation at P90/P100
    SIGMOID = "sigmoid"      # Calibrated logistic centred on P95
    TANH = "tanh"            # Calibrated hyperbolic tangent centred on P95


class SplitStrategy(str, Enum):
    """Temporal split strategy."""
    FIXED = "fixed"                  # fixed cut-off dates
    ROLLING = "rolling"              # rolling-window forward validation
    EXPANDING = "expanding"          # expanding-window forward validation


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FuzzyConfig:
    """
    Parameters for VP target construction.

    The target is a continuous value in [0, 1] anchored on three
    climatological thresholds (P90 → 0, P95 → 0.5, P100 → 1).
    """
    mode: TargetMode = TargetMode.SIGMOID
    eps: float = 0.02                 # how close VP gets to 0 at P90 (sigmoid/tanh)
    thresholds_dir: str = "data/raster"   # dir with p90/p95/p100 rasters

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "eps": self.eps,
            "thresholds_dir": self.thresholds_dir,
        }


@dataclass
class FeatureConfig:
    """Parameters for feature engineering."""
    bands: List[str] = field(default_factory=lambda: list(SHORT_BANDS))
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6])
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std"])
    include_lags: bool = True
    include_rolling: bool = True
    include_spatial_coords: bool = True
    # Patch mode settings
    patch_size: int = 5        # spatial window half-size for PATCH mode
    # Region mode settings
    region_stats: List[str] = field(
        default_factory=lambda: ["mean", "std", "min", "max", "median"]
    )

    def to_dict(self) -> dict:
        return {
            "bands": self.bands,
            "rolling_windows": self.rolling_windows,
            "rolling_stats": self.rolling_stats,
            "include_lags": self.include_lags,
            "include_rolling": self.include_rolling,
            "include_spatial_coords": self.include_spatial_coords,
            "patch_size": self.patch_size,
            "region_stats": self.region_stats,
        }


@dataclass
class SplitConfig:
    """Temporal splitting parameters."""
    strategy: SplitStrategy = SplitStrategy.FIXED
    train_end: str = "2022-12-31"       # inclusive
    val_end: str = "2023-02-28"         # inclusive; test = everything after
    # Rolling / Expanding window parameters
    window_train_size: Optional[int] = None   # number of time steps
    window_val_size: Optional[int] = None
    step_size: Optional[int] = None
    # Optional spatial blocking
    spatial_block_col: Optional[str] = None   # e.g. "region_id"

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "train_end": self.train_end,
            "val_end": self.val_end,
            "window_train_size": self.window_train_size,
            "window_val_size": self.window_val_size,
            "step_size": self.step_size,
            "spatial_block_col": self.spatial_block_col,
        }


@dataclass
class PipelineConfig:
    """Master configuration for one experiment or a full grid search."""

    # -- Paths --
    cube_dir: str = "data/raster/cube"
    cube_nc: str = "data/raster/cubo_precipitacao.nc"   # 30-year climatological cube
    thresholds_dir: str = "data/raster"                  # P90/P95/P100 GeoTIFF dir
    roi_file: Optional[str] = "data/vector/ValeDoParaiba.geojson"
    output_dir: str = "experiments"

    # -- Forecasting grid --
    p_values: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    q_values: List[int] = field(default_factory=lambda: [1, 3, 6])

    # -- Sub-configs --
    sample_mode: SampleMode = SampleMode.PIXEL
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    fuzzy_config: FuzzyConfig = field(default_factory=FuzzyConfig)
    split_config: SplitConfig = field(default_factory=SplitConfig)

    # -- Performance --
    spatial_stride: int = 2
    max_files: Optional[int] = None
    sample_frac: float = 1.0
    chunk_xy: int = 128
    random_seed: int = 42
    n_jobs: int = -1

    # -- Reproducibility --
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # ----- helpers -----
    def to_dict(self) -> dict:
        d = {
            "cube_dir": self.cube_dir,
            "cube_nc": self.cube_nc,
            "thresholds_dir": self.thresholds_dir,
            "roi_file": self.roi_file,
            "output_dir": self.output_dir,
            "p_values": self.p_values,
            "q_values": self.q_values,
            "sample_mode": self.sample_mode.value,
            "feature_config": self.feature_config.to_dict(),
            "fuzzy_config": self.fuzzy_config.to_dict(),
            "split_config": self.split_config.to_dict(),
            "spatial_stride": self.spatial_stride,
            "max_files": self.max_files,
            "sample_frac": self.sample_frac,
            "chunk_xy": self.chunk_xy,
            "random_seed": self.random_seed,
            "n_jobs": self.n_jobs,
            "created_at": self.created_at,
        }
        return d

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        raw = json.loads(Path(path).read_text())
        raw["sample_mode"] = SampleMode(raw["sample_mode"])
        raw["feature_config"] = FeatureConfig(**raw["feature_config"])
        fc = raw["fuzzy_config"]
        fc["mode"] = TargetMode(fc["mode"])
        # Backward compat: old configs may have slope/offset/p95_source
        for old_key in ("slope", "offset", "p95_source"):
            fc.pop(old_key, None)
        raw["fuzzy_config"] = FuzzyConfig(**fc)
        # Backward compat: old configs may have p95_file instead of thresholds_dir
        if "p95_file" in raw:
            raw.setdefault("thresholds_dir", str(Path(raw.pop("p95_file")).parent))
        sc = raw["split_config"]
        sc["strategy"] = SplitStrategy(sc["strategy"])
        raw["split_config"] = SplitConfig(**sc)
        return cls(**raw)


# ---------------------------------------------------------------------------
# Environment / reproducibility snapshot
# ---------------------------------------------------------------------------
def get_environment_info() -> dict:
    """Capture runtime environment for metadata."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
    }
    try:
        info["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        info["git_commit"] = None
    return info


def file_hash(filepath: str | Path, algo: str = "sha256") -> str:
    """Compute hash of a file for provenance tracking."""
    h = hashlib.new(algo)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
