"""
pipeline.experiment
===================
Experiment runner: orchestrates the (P, Q) × model grid search with
caching, logging, and reproducibility.

Strategy to avoid recomputing expensive steps
----------------------------------------------
1. **Raw timeseries extraction** is done once and cached as
   ``base_table_P{max_P}.parquet``.  All (P, Q) combos reuse this table.
2. **P95 raster** is computed once from the 30-year NetCDF climatological
   cube and saved as a fixed GeoTIFF artefact.  All experiments load
   the same raster — no recomputation.
3. **Feature columns** for a given P are a subset of columns already
   present in the base table (which contains lags up to ``max(P)``).
4. **Target columns** for each Q are added once to the base table.
5. **Scaling** is fitted per-split (train-only) and cached per experiment.

The runner iterates:

.. code-block:: text

    for P in p_values:
        for Q in q_values:
            select features for P
            select target for Q
            split → scale → for model in models: train_evaluate → log

Public API
----------
ExperimentRunner(config) – instantiate once, call .run()
"""

from __future__ import annotations

import gc
import os
import time
import traceback
import warnings
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig, SampleMode
from .ingestion import (
    extract_pixel_timeseries,
    list_sorted_tiffs,
    load_p95_grid,
    validate_alignment,
)
from .features import build_features
from .targets import add_target_columns, compute_p95, ensure_p95_raster
from .splitting import (
    check_no_leakage,
    fit_transformers,
    make_splits,
    transform_split,
)
from .evaluation import compute_metrics, train_evaluate
from .models.registry import get_model_registry, instantiate_model
from .export import (
    cache_exists,
    get_cache_path,
    load_cache,
    save_cache,
    save_dataframe,
    save_experiment_metadata,
    update_summary_table,
)
from .checks import run_all_checks


class ExperimentRunner:
    """
    End-to-end experiment orchestrator.

    Parameters
    ----------
    config : PipelineConfig
        Master configuration.
    model_families : list of str, optional
        Model family codes to evaluate (e.g. ``['E', 'L', 'DL']``).
        None = all families.
    scaler_type : str
        Scaler to fit on training data.  ``'standard'`` | ``'robust'`` | ``'minmax'``.
    save_train : bool
        Whether to also save the training set to disk (large!).
    skip_checks : bool
        If True, skip the full leakage/quality check battery (faster).
    """

    def __init__(
        self,
        config: PipelineConfig,
        model_families: Optional[List[str]] = None,
        scaler_type: str = "standard",
        save_train: bool = False,
        skip_checks: bool = False,
    ):
        self.cfg = config
        self.model_entries = get_model_registry(model_families)
        self.scaler_type = scaler_type
        self.save_train = save_train
        self.skip_checks = skip_checks
        self._base_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def run(self) -> pd.DataFrame:
        """
        Execute the full (P, Q) × model grid.

        Returns
        -------
        pd.DataFrame
            Summary table with one row per (P, Q, model) combination.
        """
        np.random.seed(self.cfg.random_seed)
        t0 = time.time()

        # 1. Save master config
        self.cfg.save(Path(self.cfg.output_dir) / "pipeline_config.json")

        # 2. Build / load base table
        base_df = self._get_base_table()

        # 3. Iterate over (P, Q) × models
        all_results: List[Dict] = []

        for P, Q in product(self.cfg.p_values, self.cfg.q_values):
            exp_tag = f"P{P}_Q{Q}"
            print(f"\n{'='*60}")
            print(f"  Experiment: {exp_tag}")
            print(f"{'='*60}")

            try:
                results = self._run_single_pq(base_df, P, Q)
                all_results.extend(results)
            except Exception as e:
                warnings.warn(f"Experiment {exp_tag} failed: {e}")
                traceback.print_exc()

            gc.collect()

        elapsed = time.time() - t0
        print(f"\n✓ All experiments completed in {elapsed:.1f}s")

        # 4. Final summary
        if all_results:
            summary = pd.DataFrame(all_results)
            summary.to_csv(
                Path(self.cfg.output_dir) / "summary_results.csv", index=False
            )
            return summary
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Base table construction (cached)                                    #
    # ------------------------------------------------------------------ #

    def _get_base_table(self) -> pd.DataFrame:
        """Build or load the shared base table with all lags + targets."""
        max_p = max(self.cfg.p_values)
        cache_name = f"base_table_P{max_p}.parquet"

        if cache_exists(self.cfg.output_dir, cache_name):
            print(f"Loading cached base table: {cache_name}")
            return load_cache(self.cfg.output_dir, cache_name)

        print("Building base table from TIFFs...")

        # 1. List and validate files
        sorted_files = list_sorted_tiffs(
            self.cfg.cube_dir, max_files=self.cfg.max_files
        )
        if not sorted_files:
            raise RuntimeError(f"No TIFF files found in {self.cfg.cube_dir}")

        paths = [f for _, f in sorted_files]
        ref_meta = validate_alignment(paths)
        print(f"  ✓ {len(sorted_files)} files validated (shape: "
              f"{ref_meta['height']}×{ref_meta['width']}, CRS: {ref_meta['crs']})")

        # 2. Extract pixel timeseries
        bands = self.cfg.feature_config.bands
        raw_df = extract_pixel_timeseries(
            sorted_files,
            stride=self.cfg.spatial_stride,
            bands=bands,
        )
        print(f"  Raw timeseries: {len(raw_df):,} rows, "
              f"columns: {list(raw_df.columns)}")

        # 3. Build features (lags + rolling for max_p)
        feat_df, feat_meta = build_features(
            raw_df,
            P=max_p,
            feature_config=self.cfg.feature_config,
            sample_mode=self.cfg.sample_mode,
        )
        print(f"  Features built: {feat_meta['n_features']} columns")

        # 4. Merge features back with raw data for target construction
        merge_cols = ["time", "pixel_id"]
        base_df = feat_df.merge(
            raw_df[merge_cols + bands].drop_duplicates(),
            on=merge_cols,
            how="left",
        )

        # 5. Load fixed P95 raster (computed once from 30-year cube)
        #    ensure_p95_raster will skip if file already exists.
        ensure_p95_raster(
            cube_nc=self.cfg.cube_nc,
            p95_tif=self.cfg.p95_file,
        )
        p95_grid = load_p95_grid(self.cfg.p95_file, stride=self.cfg.spatial_stride)
        print(f"  P95 loaded from raster: {self.cfg.p95_file} "
              f"({len(p95_grid)} pixels)")

        # Determine p95 method and supply the external grid
        p95_method = self.cfg.fuzzy_config.p95_source  # 'external' (default)
        splits = make_splits(base_df, self.cfg.split_config)
        train_mask = splits["train_mask"]

        p95 = compute_p95(
            base_df,
            train_mask,
            band="tp",
            method=p95_method,
            external_p95=p95_grid,
        )

        # 6. Add target columns for all Q values
        base_df, target_metas = add_target_columns(
            base_df,
            self.cfg.q_values,
            p95,
            self.cfg.fuzzy_config,
        )

        # 7. Optional sub-sampling
        if self.cfg.sample_frac < 1.0:
            n_before = len(base_df)
            base_df = base_df.sample(
                frac=self.cfg.sample_frac, random_state=self.cfg.random_seed
            )
            print(f"  Sub-sampled: {n_before:,} → {len(base_df):,}")

        # 8. Cache
        save_cache(base_df, self.cfg.output_dir, cache_name)
        print(f"  ✓ Base table cached ({len(base_df):,} rows)")

        return base_df

    # ------------------------------------------------------------------ #
    #  Single (P, Q) experiment                                            #
    # ------------------------------------------------------------------ #

    def _run_single_pq(
        self, base_df: pd.DataFrame, P: int, Q: int
    ) -> List[Dict]:
        """Run all models for one (P, Q) combination."""
        target_col = f"target_q{Q}"
        if target_col not in base_df.columns:
            warnings.warn(f"Target column {target_col} not found — skipping.")
            return []

        # Select feature columns for this P
        all_feat_cols = [c for c in base_df.columns if c.startswith(("lag_", "roll_"))]
        feat_cols = self._select_features_for_P(all_feat_cols, P)

        # Add spatial coords if present
        for c in ("y_idx", "x_idx"):
            if c in base_df.columns and c not in feat_cols:
                feat_cols.append(c)

        cols_needed = ["time", "pixel_id"] + feat_cols + [target_col]
        cols_present = [c for c in cols_needed if c in base_df.columns]
        df = base_df[cols_present].dropna(subset=feat_cols + [target_col])

        if len(df) == 0:
            warnings.warn(f"P{P}_Q{Q}: empty after dropna — skipping.")
            return []

        # Split
        splits = make_splits(df, self.cfg.split_config)
        train_mask = splits["train_mask"]
        val_mask = splits["val_mask"]
        test_mask = splits["test_mask"]

        if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
            warnings.warn(f"P{P}_Q{Q}: one or more splits empty — skipping.")
            return []

        # Leakage checks
        if not self.skip_checks:
            try:
                run_all_checks(df, feat_cols, target_col, train_mask, val_mask, test_mask, P)
            except AssertionError as e:
                warnings.warn(f"Check failed for P{P}_Q{Q}: {e}")
                return []

        # Scale features (fit on train only)
        transformer = fit_transformers(
            df[train_mask], feat_cols, scaler_type=self.scaler_type
        )
        df_scaled = transform_split(df, transformer)

        X_train = df_scaled.loc[train_mask, feat_cols].values
        y_train = df_scaled.loc[train_mask, target_col].values
        X_val = df_scaled.loc[val_mask, feat_cols].values
        y_val = df_scaled.loc[val_mask, target_col].values
        X_test = df_scaled.loc[test_mask, feat_cols].values
        y_test = df_scaled.loc[test_mask, target_col].values

        print(f"  Split — train: {len(X_train):,}  val: {len(X_val):,}  "
              f"test: {len(X_test):,}  features: {len(feat_cols)}")

        # Number of features per timestep (for LSTM reshape)
        n_bands = len(self.cfg.feature_config.bands)

        # Run models
        results = []
        for entry in self.model_entries:
            model_name = entry["name"]
            print(f"    → {model_name}...", end=" ", flush=True)
            t0 = time.time()

            try:
                model = instantiate_model(entry)
                res = train_evaluate(
                    model,
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test,
                    model_name=model_name,
                    input_type=entry["input_type"],
                    P=P,
                    n_features=len(feat_cols) // P if P > 0 else len(feat_cols),
                )
            except Exception as e:
                res = {
                    "model_name": model_name,
                    "error": str(e),
                    "val_metrics": {},
                    "test_metrics": {},
                }

            elapsed = time.time() - t0

            # Build result row
            row = {
                "experiment": f"exp_P{P}_Q{Q}",
                "P": P,
                "Q": Q,
                "model": model_name,
                "family": entry["code"],
                "input_type": entry["input_type"],
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "test_rows": len(X_test),
                "n_features": len(feat_cols),
                "elapsed_s": round(elapsed, 2),
            }

            if "error" in res and res.get("error"):
                row["error"] = res["error"]
                print(f"ERROR: {res['error']}")
            else:
                for split_name in ("val_metrics", "test_metrics"):
                    prefix = split_name.split("_")[0]
                    for metric_name, value in res.get(split_name, {}).items():
                        row[f"{prefix}_{metric_name}"] = value
                test_brier = res.get("test_metrics", {}).get("brier", "?")
                print(f"Brier={test_brier:.4f}  ({elapsed:.1f}s)"
                      if isinstance(test_brier, float) else f"done ({elapsed:.1f}s)")

            results.append(row)
            update_summary_table(self.cfg.output_dir, row)

            # Save experiment artefacts
            exp_dir = Path(self.cfg.output_dir) / f"exp_P{P}_Q{Q}" / model_name
            exp_dir.mkdir(parents=True, exist_ok=True)

            if "test_predictions" in res and res["test_predictions"] is not None:
                pred_df = pd.DataFrame({
                    "y_true": y_test,
                    "y_pred": res["test_predictions"],
                })
                save_dataframe(pred_df, exp_dir / "predictions")

            # Save model if joblib-compatible
            if res.get("model") is not None:
                try:
                    import joblib
                    joblib.dump(res["model"], exp_dir / "model.joblib")
                except Exception:
                    pass

            save_experiment_metadata(
                exp_dir,
                P=P,
                Q=Q,
                config=self.cfg,
                feature_meta={"feature_names": feat_cols, "n_features": len(feat_cols)},
                target_meta={"target_col": target_col, "Q": Q},
                metrics={
                    "val": res.get("val_metrics", {}),
                    "test": res.get("test_metrics", {}),
                },
            )

        return results

    # ------------------------------------------------------------------ #
    #  Feature selection for a given P                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _select_features_for_P(all_feature_cols: List[str], P: int) -> List[str]:
        """
        From the full set of feature columns (built for max_P), select
        only those corresponding to lags 0..P-1 and rolling windows ≤ P.
        """
        selected = []
        for col in all_feature_cols:
            if col.startswith("lag_"):
                # lag_{k}_{band}
                parts = col.split("_")
                if len(parts) >= 3:
                    try:
                        k = int(parts[1])
                        if k < P:
                            selected.append(col)
                    except ValueError:
                        selected.append(col)
                else:
                    selected.append(col)
            elif col.startswith("roll_"):
                # roll_{stat}{window}_{band}  — keep if window ≤ P
                import re
                m = re.search(r"roll_\w+?(\d+)_", col)
                if m:
                    w = int(m.group(1))
                    if w <= P:
                        selected.append(col)
                else:
                    selected.append(col)
            else:
                selected.append(col)
        return selected
