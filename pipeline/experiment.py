"""
pipeline.experiment
===================
Experiment runner: orchestrates the (P, Q) × model grid search with
caching, logging, HP tuning, anomaly detection, and reproducibility.

Architecture
------------
1. **Raw timeseries extraction** — done once, cached as parquet.
2. **Percentile rasters** (P90/P95/P100) — computed once from 30-year
   NetCDF and saved as fixed GeoTIFF artefacts.
3. **Feature columns** for a given P are a subset of the max-P base table.
4. **Target columns** for each Q are added once.
5. **Scaling** fitted per-split on train only.  Tree-based models skip scaling.
6. **HP tuning** (optional) via Optuna with purged temporal CV.
7. **Anomaly threshold** calibrated on val (never test).

Naming Convention
-----------------
* **P** (``lookback_steps``)  — number of past time steps in features.
  **Not** a statistical p-value.
* **Q** (``forecast_horizon``) — time steps ahead for the target.
  **Not** an ARIMA MA order.

The runner iterates::

    for P in lookback_steps:
        for Q in forecast_horizons:
            select features → select target → split → scale →
            for model in models:
                [optional: Optuna HP tune] →
                train_evaluate →
                calibrate anomaly threshold (val) →
                compute anomaly report (test) →
                log

Public API
----------
ExperimentRunner(config) – instantiate once, call .run()
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
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
    validate_alignment,
)
from .features import build_features
from .targets import (
    add_target_columns,
    align_thresholds,
    ensure_percentile_rasters,
    load_percentile_grids,
)
from .splitting import (
    check_no_leakage,
    fit_transformers,
    make_splits,
    transform_split,
)
from .evaluation import compute_metrics, train_evaluate
from .anomaly import calibrate_threshold, compute_anomaly_report
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

logger = logging.getLogger(__name__)


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
        Scaler to fit on training data.  ``'standard'`` | ``'robust'``
        | ``'minmax'``.
    save_train : bool
        Whether to also save the training set to disk.
    skip_checks : bool
        If True, skip the full leakage/quality check battery.
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
        pd.DataFrame — summary table with one row per (P, Q, model).
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
            print(f"  Experiment: {exp_tag}  "
                  f"(lookback={P}, horizon={Q})")
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
                Path(self.cfg.output_dir) / "summary_results.csv",
                index=False,
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
            self.cfg.cube_dir, max_files=self.cfg.max_files,
        )
        if not sorted_files:
            raise RuntimeError(f"No TIFF files found in {self.cfg.cube_dir}")

        paths = [f for _, f in sorted_files]
        ref_meta = validate_alignment(paths)
        print(
            f"  ✓ {len(sorted_files)} files validated (shape: "
            f"{ref_meta['height']}×{ref_meta['width']}, "
            f"CRS: {ref_meta['crs']})"
        )

        # 2. Extract pixel timeseries
        bands = self.cfg.feature_config.bands
        raw_df = extract_pixel_timeseries(
            sorted_files, stride=self.cfg.spatial_stride, bands=bands,
        )
        print(
            f"  Raw timeseries: {len(raw_df):,} rows, "
            f"columns: {list(raw_df.columns)}"
        )

        # 3. Build features (lags + rolling for max_p)
        feat_df, feat_meta = build_features(
            raw_df, P=max_p,
            feature_config=self.cfg.feature_config,
            sample_mode=self.cfg.sample_mode,
        )
        print(f"  Features built: {feat_meta['n_features']} columns")

        # 4. Merge features back with raw data for target construction
        merge_cols = ["time", "pixel_id"]
        base_df = feat_df.merge(
            raw_df[merge_cols + bands].drop_duplicates(),
            on=merge_cols, how="left",
        )

        # 5. Load fixed percentile rasters (computed once)
        thresholds_dir = self.cfg.fuzzy_config.thresholds_dir
        ensure_percentile_rasters(
            cube_nc=self.cfg.cube_nc, out_dir=thresholds_dir,
        )
        pct_grid = load_percentile_grids(
            thresholds_dir, stride=self.cfg.spatial_stride,
        )
        print(
            f"  Thresholds loaded from: {thresholds_dir} "
            f"({len(pct_grid)} pixels)"
        )

        # Align thresholds with base_df rows
        thresholds_aligned = align_thresholds(base_df, pct_grid)

        # 6. Add target columns for all Q values
        base_df, target_metas = add_target_columns(
            base_df, self.cfg.q_values, thresholds_aligned,
            self.cfg.fuzzy_config,
        )

        # 7. Optional sub-sampling
        if self.cfg.sample_frac < 1.0:
            n_before = len(base_df)
            base_df = base_df.sample(
                frac=self.cfg.sample_frac, random_state=self.cfg.random_seed,
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
        self, base_df: pd.DataFrame, P: int, Q: int,
    ) -> List[Dict]:
        """Run all models for one (P, Q) combination."""
        target_col = f"target_q{Q}"
        if target_col not in base_df.columns:
            warnings.warn(f"Target column {target_col} not found — skipping.")
            return []

        # Select feature columns for this P
        all_feat_cols = [
            c for c in base_df.columns
            if c.startswith(("lag_", "roll_"))
        ]
        feat_cols = self._select_features_for_P(all_feat_cols, P)

        # Add non-lag features (spatial coords, temporal features)
        for c in ("y_idx", "x_idx", "hour_sin", "hour_cos",
                   "month_sin", "month_cos", "doy_sin", "doy_cos"):
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

        if (train_mask.sum() == 0 or val_mask.sum() == 0
                or test_mask.sum() == 0):
            warnings.warn(
                f"P{P}_Q{Q}: one or more splits empty — skipping."
            )
            return []

        # Leakage checks
        if not self.skip_checks:
            try:
                run_all_checks(
                    df, feat_cols, target_col,
                    train_mask, val_mask, test_mask, P,
                )
            except AssertionError as e:
                warnings.warn(f"Check failed for P{P}_Q{Q}: {e}")
                return []

        # Scale features (fit on train only)
        transformer = fit_transformers(
            df[train_mask], feat_cols, scaler_type=self.scaler_type,
        )
        df_scaled = transform_split(df, transformer)

        # Identify which features are lag-only (for LSTM reshape)
        lag_feat_cols = self._get_lag_only_features(feat_cols, P)
        bands_in_data = self.cfg.feature_config.bands
        n_bands_per_step = len([
            c for c in lag_feat_cols if c.startswith("lag_0_")
        ])

        print(
            f"  Split — train: {train_mask.sum():,}  "
            f"val: {val_mask.sum():,}  test: {test_mask.sum():,}  "
            f"features: {len(feat_cols)}"
        )

        # Extract train times for temporal CV in tuning
        train_times = df.loc[train_mask, "time"].values

        # Run models
        results = []
        for entry in self.model_entries:
            model_name = entry["name"]
            print(f"    → {model_name}...", end=" ", flush=True)
            t0m = time.time()

            try:
                result_row = self._train_single_model(
                    entry=entry,
                    df_scaled=df_scaled,
                    df_raw=df,
                    feat_cols=feat_cols,
                    lag_feat_cols=lag_feat_cols,
                    target_col=target_col,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask,
                    train_times=train_times,
                    P=P, Q=Q,
                    n_bands_per_step=n_bands_per_step,
                )
            except Exception as e:
                result_row = {
                    "model_name": model_name,
                    "error": str(e),
                    "val_metrics": {},
                    "test_metrics": {},
                }
                logger.warning("Model %s failed: %s", model_name, e)

            elapsed = time.time() - t0m

            # Build result row
            row: Dict[str, Any] = {
                "experiment": f"exp_P{P}_Q{Q}",
                "P": P,
                "Q": Q,
                "model": model_name,
                "family": entry["code"],
                "input_type": entry["input_type"],
                "train_rows": int(train_mask.sum()),
                "val_rows": int(val_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "n_features": len(feat_cols),
                "elapsed_s": round(elapsed, 2),
            }

            if "error" in result_row and result_row.get("error"):
                row["error"] = result_row["error"]
                print(f"ERROR: {result_row['error']}")
            else:
                # Flatten metrics into columns
                for split_name in ("val_metrics", "test_metrics"):
                    prefix = split_name.split("_")[0]
                    for mname, mval in result_row.get(split_name, {}).items():
                        row[f"{prefix}_{mname}"] = mval

                # Anomaly-specific columns
                for akey in ("threshold", "anomaly_precision",
                             "anomaly_recall", "anomaly_f1",
                             "anomaly_pr_auc", "anomaly_fpr"):
                    if akey in result_row:
                        row[akey] = result_row[akey]

                # Tuning info
                if "best_params" in result_row:
                    row["tuned"] = True
                    row["best_params"] = json.dumps(
                        result_row["best_params"], default=str,
                    )
                else:
                    row["tuned"] = False

                tb = result_row.get("test_metrics", {}).get("brier", "?")
                pr_auc = result_row.get("anomaly_pr_auc", "?")
                if isinstance(tb, float):
                    print(
                        f"Brier={tb:.4f}  PR-AUC="
                        f"{pr_auc:.4f}" if isinstance(pr_auc, float)
                        else f"Brier={tb:.4f}",
                        end="",
                    )
                print(f"  ({elapsed:.1f}s)")

            results.append(row)
            update_summary_table(self.cfg.output_dir, row)

            # Save experiment artefacts
            exp_dir = (
                Path(self.cfg.output_dir) / f"exp_P{P}_Q{Q}" / model_name
            )
            exp_dir.mkdir(parents=True, exist_ok=True)

            if "test_predictions" in result_row:
                pred_df = pd.DataFrame({
                    "y_true": result_row.get("y_test", []),
                    "y_pred": result_row["test_predictions"],
                })
                save_dataframe(pred_df, exp_dir / "predictions")

            # Save anomaly report
            if "anomaly_report" in result_row:
                (exp_dir / "anomaly_report.json").write_text(
                    json.dumps(result_row["anomaly_report"], indent=2,
                               default=str)
                )

            # Save model
            if result_row.get("model") is not None:
                try:
                    import joblib
                    joblib.dump(result_row["model"], exp_dir / "model.joblib")
                except Exception:
                    pass

            save_experiment_metadata(
                exp_dir, P=P, Q=Q, config=self.cfg,
                feature_meta={
                    "feature_names": feat_cols,
                    "n_features": len(feat_cols),
                },
                target_meta={"target_col": target_col, "Q": Q},
                metrics={
                    "val": result_row.get("val_metrics", {}),
                    "test": result_row.get("test_metrics", {}),
                },
            )

        return results

    # ------------------------------------------------------------------ #
    #  Single model: tune → train → evaluate → anomaly                     #
    # ------------------------------------------------------------------ #

    def _train_single_model(
        self,
        entry: Dict,
        df_scaled: pd.DataFrame,
        df_raw: pd.DataFrame,
        feat_cols: List[str],
        lag_feat_cols: List[str],
        target_col: str,
        train_mask: pd.Series,
        val_mask: pd.Series,
        test_mask: pd.Series,
        train_times: np.ndarray,
        P: int,
        Q: int,
        n_bands_per_step: int,
    ) -> Dict[str, Any]:
        """
        Full lifecycle for one model: optional tune → train → eval →
        anomaly detection.
        """
        model_name = entry["name"]
        input_type = entry["input_type"]
        needs_scaling = entry.get("needs_scaling", True)

        # Choose scaled or raw features based on model type
        src_df = df_scaled if needs_scaling else df_raw

        # For LSTM: use only lag features and reshape properly
        if input_type == "sequence":
            use_cols = lag_feat_cols
            n_f_per_step = n_bands_per_step
        else:
            use_cols = feat_cols
            n_f_per_step = None

        X_train = src_df.loc[train_mask, use_cols].values
        y_train = src_df.loc[train_mask, target_col].values
        X_val = src_df.loc[val_mask, use_cols].values
        y_val = src_df.loc[val_mask, target_col].values
        X_test = src_df.loc[test_mask, use_cols].values
        y_test = src_df.loc[test_mask, target_col].values

        # ---- Optional HP tuning ----
        best_params: Dict[str, Any] = {}
        tune_cfg = self.cfg.tuning_config

        if tune_cfg.enabled:
            try:
                from .tuning import OptunaHPTuner, SEARCH_SPACES

                if model_name in SEARCH_SPACES and SEARCH_SPACES[model_name]:
                    tuner = OptunaHPTuner(
                        model_name=model_name,
                        X_train=X_train,
                        y_train=y_train,
                        times_train=train_times,
                        metric=tune_cfg.metric,
                        direction=tune_cfg.direction,
                        n_folds=tune_cfg.n_cv_folds,
                        gap=tune_cfg.cv_gap,
                        random_seed=self.cfg.random_seed,
                        input_type=input_type,
                        P=P,
                        n_features=n_f_per_step,
                    )
                    best_params = tuner.tune(
                        n_trials=tune_cfg.n_trials,
                        timeout=tune_cfg.timeout_per_model,
                    )
                    print(f"[tuned] ", end="")
            except ImportError:
                logger.warning(
                    "Optuna not available — using defaults for %s",
                    model_name,
                )
            except Exception as exc:
                logger.warning(
                    "Tuning failed for %s: %s — using defaults",
                    model_name, exc,
                )

        # ---- Instantiate (with tuned or default params) ----
        model = instantiate_model(entry, **best_params)

        # ---- Train + evaluate ----
        res = train_evaluate(
            model,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            model_name=model_name,
            input_type=input_type,
            P=P,
            n_features=n_f_per_step,
        )

        if res.get("error"):
            return res

        # ---- Anomaly threshold calibration (on val, never test) ----
        val_pred = res.get("val_predictions")
        test_pred = res.get("test_predictions")

        threshold = 0.5  # default
        anomaly_report: Dict[str, Any] = {}

        if val_pred is not None and test_pred is not None:
            try:
                threshold, val_anomaly = calibrate_threshold(
                    y_val, val_pred,
                    method=tune_cfg.calibration_method,
                )
                # Apply to test set
                test_times = df_raw.loc[test_mask, "time"].values
                anomaly_report = compute_anomaly_report(
                    y_test, test_pred, threshold, times=test_times,
                )
            except Exception as exc:
                logger.warning("Anomaly calibration failed: %s", exc)

        # Build output
        output: Dict[str, Any] = {
            "model_name": model_name,
            "val_metrics": res.get("val_metrics", {}),
            "test_metrics": res.get("test_metrics", {}),
            "model": res.get("model"),
            "test_predictions": test_pred,
            "y_test": y_test,
            "threshold": threshold,
            "anomaly_report": anomaly_report,
        }

        # Extract key anomaly metrics for summary
        overall_anom = anomaly_report.get("overall", {})
        output["anomaly_precision"] = overall_anom.get("precision", float("nan"))
        output["anomaly_recall"] = overall_anom.get("recall", float("nan"))
        output["anomaly_f1"] = overall_anom.get("f1", float("nan"))
        output["anomaly_pr_auc"] = overall_anom.get("pr_auc", float("nan"))
        output["anomaly_fpr"] = overall_anom.get("fpr", float("nan"))

        if best_params:
            output["best_params"] = best_params

        return output

    # ------------------------------------------------------------------ #
    #  Feature selection helpers                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _select_features_for_P(
        all_feature_cols: List[str], P: int,
    ) -> List[str]:
        """
        From the full set of feature columns (built for max_P), select
        only those corresponding to lags 0..P-1 and rolling windows ≤ P.
        """
        selected = []
        for col in all_feature_cols:
            if col.startswith("lag_"):
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

    @staticmethod
    def _get_lag_only_features(
        feat_cols: List[str], P: int,
    ) -> List[str]:
        """
        Extract lag-only features sorted by (lag_index, band) for correct
        LSTM sequence construction: ``[lag_0_tp, lag_0_t2m, lag_1_tp, ...]``
        → reshape to ``(N, P, n_bands)``.
        """
        lag_cols = [c for c in feat_cols if c.startswith("lag_")]

        def _sort_key(col: str):
            parts = col.split("_")
            try:
                lag_idx = int(parts[1])
            except (ValueError, IndexError):
                lag_idx = 999
            band = "_".join(parts[2:])
            return (lag_idx, band)

        return sorted(lag_cols, key=_sort_key)
