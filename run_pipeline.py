#!/usr/bin/env python3
"""
run_pipeline.py
===============
Main entry point for the MscDN spatio-temporal forecasting pipeline.

Usage
-----
  # Full grid search with defaults
  python run_pipeline.py

  # Quick smoke-test (few files, sub-sampled, fast models only)
  python run_pipeline.py --quick

  # Custom config from JSON
  python run_pipeline.py --config experiments/my_config.json

Pipeline Flow
-------------
::

  raw TIFFs  ──→  validate alignment
       │
       ▼
  parse timestamps & sort chronologically
       │
       ▼
  extract pixel timeseries  (incremental, stride-subsampled)
       │
       ▼
  build features  (lags 0..P-1, rolling stats, spatial coords)
       │
       ▼
  compute p95  (train-only, per-pixel)
       │
       ▼
  build targets  (sigmoid / fuzzy around p95, for each Q)
       │
       ▼
  split  (temporal: train ≤ T₁ < val ≤ T₂ < test)
       │
       ▼
  scale  (fit on train, transform val/test)
       │
       ▼
  for (P, Q) in grid:
      for model in registry:
          train → evaluate → log metrics → save artefacts
       │
       ▼
  summary_results.csv  +  per-run metadata.json  +  predictions.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import (
    PipelineConfig,
    FeatureConfig,
    FuzzyConfig,
    SplitConfig,
    SampleMode,
    SplitStrategy,
    TargetMode,
)
from pipeline.experiment import ExperimentRunner


# ======================================================================== #
#  Preset configurations                                                    #
# ======================================================================== #

def default_config() -> PipelineConfig:
    """Full production configuration."""
    return PipelineConfig(
        cube_dir="data/raster/cube",
        roi_file="data/vector/ValeDoParaiba.geojson",
        output_dir="experiments",
        p_values=[3, 6, 12, 24],
        q_values=[1, 3, 6],
        sample_mode=SampleMode.PIXEL,
        feature_config=FeatureConfig(
            bands=["tp", "t2m", "u10", "v10", "sp"],
            rolling_windows=[3, 6],
            rolling_stats=["mean", "std"],
            include_lags=True,
            include_rolling=True,
            include_spatial_coords=True,
        ),
        fuzzy_config=FuzzyConfig(
            mode=TargetMode.SIGMOID,
            eps=0.02,
            thresholds_dir="data/raster",
        ),
        split_config=SplitConfig(
            strategy=SplitStrategy.FIXED,
            train_end="2022-12-31",
            val_end="2023-01-31",
        ),
        spatial_stride=2,
        max_files=None,
        sample_frac=1.0,
        chunk_xy=128,
        random_seed=42,
        n_jobs=-1,
    )


def quick_config() -> PipelineConfig:
    """Fast smoke-test configuration (small data, few models)."""
    cfg = default_config()
    cfg.output_dir = "experiments_quick"
    cfg.p_values = [3, 6]
    cfg.q_values = [1, 3]
    cfg.max_files = None          # use all 14 files
    cfg.sample_frac = 0.10
    cfg.spatial_stride = 4
    cfg.feature_config.rolling_windows = [3]
    cfg.feature_config.rolling_stats = ["mean"]
    cfg.feature_config.bands = ["tp"]
    return cfg


# ======================================================================== #
#  CLI                                                                      #
# ======================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(
        description="MscDN Spatio-temporal Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a pipeline_config.json file.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast smoke-test with reduced data and model set.",
    )
    parser.add_argument(
        "--families",
        type=str,
        nargs="+",
        default=None,
        help="Model family codes to evaluate (e.g. E L DL). Default = all.",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="standard",
        choices=["standard", "minmax", "robust"],
        help="Scaler type (default: standard).",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip leakage/quality checks for faster iteration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    if args.config:
        cfg = PipelineConfig.load(args.config)
        print(f"Loaded config from {args.config}")
    elif args.quick:
        cfg = quick_config()
        print("Using QUICK config (smoke-test mode)")
    else:
        cfg = default_config()
        print("Using DEFAULT config")

    # Override families if specified
    families = args.families

    # If quick mode and no families specified, use fast subset
    if args.quick and families is None:
        families = ["D", "L", "T", "E"]

    print(f"\nPipeline Configuration:")
    print(f"  Cube dir       : {cfg.cube_dir}")
    print(f"  Cube NC (hist) : {cfg.cube_nc}")
    print(f"  Thresholds dir : {cfg.fuzzy_config.thresholds_dir}")
    print(f"  VP mode        : {cfg.fuzzy_config.mode.value}")
    print(f"  VP eps         : {cfg.fuzzy_config.eps}")
    print(f"  Output dir     : {cfg.output_dir}")
    print(f"  P values     : {cfg.p_values}")
    print(f"  Q values     : {cfg.q_values}")
    print(f"  Sample mode  : {cfg.sample_mode.value}")
    print(f"  Target mode  : {cfg.fuzzy_config.mode.value}")
    print(f"  Split        : {cfg.split_config.strategy.value} "
          f"(train≤{cfg.split_config.train_end}, val≤{cfg.split_config.val_end})")
    print(f"  Stride       : {cfg.spatial_stride}")
    print(f"  Sample frac  : {cfg.sample_frac}")
    print(f"  Max files    : {cfg.max_files or 'all'}")
    print(f"  Model families: {families or 'all'}")
    print()

    runner = ExperimentRunner(
        config=cfg,
        model_families=families,
        scaler_type=args.scaler,
        skip_checks=args.skip_checks,
    )
    summary = runner.run()

    if not summary.empty:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        display_cols = [c for c in summary.columns
                        if c in ("experiment", "model", "test_brier", "test_rmse",
                                 "test_r2", "test_auc_roc", "elapsed_s")]
        if display_cols:
            print(summary[display_cols].to_string(index=False))
        else:
            print(summary.to_string(index=False))
        print(f"\nResults saved to: {cfg.output_dir}/summary_results.csv")


if __name__ == "__main__":
    main()
