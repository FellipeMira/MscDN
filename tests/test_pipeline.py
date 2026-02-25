"""
tests/test_pipeline.py
======================
Unit tests for the pipeline modules.
Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Optional dependency flags
try:
    import optuna  # noqa: F401
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import catboost  # noqa: F401
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# ======================================================================== #
#  Config tests                                                             #
# ======================================================================== #

class TestConfig:
    def test_pipeline_config_roundtrip(self, tmp_path):
        from pipeline.config import PipelineConfig
        cfg = PipelineConfig(p_values=[3, 6], q_values=[1])
        path = tmp_path / "cfg.json"
        cfg.save(path)
        loaded = PipelineConfig.load(path)
        assert loaded.p_values == [3, 6]
        assert loaded.q_values == [1]

    def test_fuzzy_config_defaults(self):
        from pipeline.config import FuzzyConfig, TargetMode
        cfg = FuzzyConfig()
        assert cfg.mode == TargetMode.SIGMOID
        assert cfg.eps == 0.02
        assert cfg.thresholds_dir == "data/raster"

    def test_target_mode_enum(self):
        from pipeline.config import TargetMode
        assert TargetMode.PIECEWISE.value == "piecewise"
        assert TargetMode.SIGMOID.value == "sigmoid"
        assert TargetMode.TANH.value == "tanh"

    def test_file_hash(self, tmp_path):
        from pipeline.config import file_hash
        f = tmp_path / "dummy.txt"
        f.write_text("hello")
        h = file_hash(f)
        assert isinstance(h, str) and len(h) == 64  # sha256 hex digest


# ======================================================================== #
#  Ingestion tests                                                          #
# ======================================================================== #

class TestIngestion:
    def test_parse_timestamp_iso(self):
        from pipeline.ingestion import parse_timestamp
        dt = parse_timestamp("era5_cube_20221201.tif")
        assert dt == datetime(2022, 12, 1, 0, 0)

    def test_parse_timestamp_iso_with_hour(self):
        from pipeline.ingestion import parse_timestamp
        dt = parse_timestamp("era5_cube_20221201_14.tif")
        assert dt == datetime(2022, 12, 1, 14, 0)

    def test_parse_timestamp_gee_style(self):
        from pipeline.ingestion import parse_timestamp
        dt = parse_timestamp("M12_D10_2024_H01_tp.tif")
        assert dt == datetime(2024, 12, 10, 1, 0)

    def test_parse_timestamp_bad_raises(self):
        from pipeline.ingestion import parse_timestamp
        with pytest.raises(ValueError):
            parse_timestamp("random_file.tif")


# ======================================================================== #
#  Features tests                                                           #
# ======================================================================== #

class TestFeatures:
    @pytest.fixture
    def sample_df(self):
        """Create a small pixel timeseries for testing."""
        rng = np.random.default_rng(42)
        n_times = 50
        n_pixels = 3
        rows = []
        for px in range(n_pixels):
            for t in range(n_times):
                rows.append({
                    "time": datetime(2022, 1, 1) + timedelta(hours=t),
                    "pixel_id": px,
                    "y_idx": px // 2,
                    "x_idx": px % 2,
                    "tp": rng.exponential(0.5),
                    "t2m": rng.normal(20, 5),
                })
        return pd.DataFrame(rows)

    def test_build_features_pixel(self, sample_df):
        from pipeline.config import FeatureConfig, SampleMode
        from pipeline.features import build_features

        cfg = FeatureConfig(
            bands=["tp", "t2m"],
            rolling_windows=[3],
            rolling_stats=["mean"],
            include_lags=True,
            include_rolling=True,
            include_spatial_coords=True,
        )
        X, meta = build_features(sample_df, P=3, feature_config=cfg, sample_mode=SampleMode.PIXEL)

        assert meta["P"] == 3
        assert meta["sample_mode"] == "pixel"
        assert len(meta["feature_names"]) == meta["n_features"]
        # Should have lag columns
        lag_cols = [c for c in meta["feature_names"] if c.startswith("lag_")]
        assert len(lag_cols) == 6  # 3 lags × 2 bands

    def test_no_nan_after_dropna(self, sample_df):
        from pipeline.config import FeatureConfig, SampleMode
        from pipeline.features import build_features

        cfg = FeatureConfig(bands=["tp"], rolling_windows=[3], rolling_stats=["mean"])
        X, meta = build_features(sample_df, P=5, feature_config=cfg, sample_mode=SampleMode.PIXEL)
        feat_cols = meta["feature_names"]
        assert X[feat_cols].isna().sum().sum() == 0


# ======================================================================== #
#  VP Target function tests                                                 #
# ======================================================================== #

class TestVPFunctions:
    """Test the three VP mapping functions with known anchor points."""

    @pytest.fixture
    def thresholds(self):
        """Per-pixel thresholds: P90=10, P95=15, P100=25."""
        n = 200
        return {
            "p90": np.full(n, 10.0),
            "p95": np.full(n, 15.0),
            "p100": np.full(n, 25.0),
        }

    @pytest.fixture
    def precip_samples(self):
        """Test precipitation values spanning below-P90 to above-P100."""
        return np.array([5.0, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0])

    def test_piecewise_anchors(self, thresholds):
        """Piecewise must hit exact anchors: P90→0, P95→0.5, P100→1."""
        from pipeline.targets import _vp_piecewise

        n = 3
        p90 = np.array([10.0, 10.0, 10.0])
        p95 = np.array([15.0, 15.0, 15.0])
        p100 = np.array([25.0, 25.0, 25.0])
        precip = np.array([10.0, 15.0, 25.0])

        vp = _vp_piecewise(precip, p90, p95, p100)
        np.testing.assert_allclose(vp, [0.0, 0.5, 1.0], atol=1e-10)

    def test_piecewise_saturation(self, thresholds):
        """Values below P90 → 0, above P100 → 1."""
        from pipeline.targets import _vp_piecewise

        p90 = np.array([10.0, 10.0])
        p95 = np.array([15.0, 15.0])
        p100 = np.array([25.0, 25.0])

        vp = _vp_piecewise(np.array([5.0, 30.0]), p90, p95, p100)
        assert vp[0] == 0.0
        assert vp[1] == 1.0

    def test_piecewise_monotonic(self, precip_samples, thresholds):
        """VP must be monotonically non-decreasing."""
        from pipeline.targets import _vp_piecewise

        n = len(precip_samples)
        vp = _vp_piecewise(
            precip_samples,
            thresholds["p90"][:n],
            thresholds["p95"][:n],
            thresholds["p100"][:n],
        )
        assert np.all(np.diff(vp) >= 0)

    def test_sigmoid_range_01(self, precip_samples, thresholds):
        """Sigmoid must produce values in [0, 1]."""
        from pipeline.targets import _vp_sigmoid

        n = len(precip_samples)
        vp = _vp_sigmoid(
            precip_samples,
            thresholds["p90"][:n],
            thresholds["p95"][:n],
            thresholds["p100"][:n],
            eps=0.02,
        )
        assert vp.min() >= 0.0
        assert vp.max() <= 1.0

    def test_sigmoid_anchor_p95(self):
        """At P95, sigmoid should be ≈ 0.5."""
        from pipeline.targets import _vp_sigmoid

        precip = np.array([15.0])
        vp = _vp_sigmoid(precip, np.array([10.0]), np.array([15.0]), np.array([25.0]))
        assert abs(vp[0] - 0.5) < 1e-6

    def test_sigmoid_saturation(self):
        """Hard saturation: below P90 → 0, above P100 → 1."""
        from pipeline.targets import _vp_sigmoid

        vp = _vp_sigmoid(
            np.array([5.0, 30.0]),
            np.array([10.0, 10.0]),
            np.array([15.0, 15.0]),
            np.array([25.0, 25.0]),
        )
        assert vp[0] == 0.0
        assert vp[1] == 1.0

    def test_tanh_range_01(self, precip_samples, thresholds):
        """Tanh must produce values in [0, 1]."""
        from pipeline.targets import _vp_tanh

        n = len(precip_samples)
        vp = _vp_tanh(
            precip_samples,
            thresholds["p90"][:n],
            thresholds["p95"][:n],
            thresholds["p100"][:n],
            eps=0.02,
        )
        assert vp.min() >= 0.0
        assert vp.max() <= 1.0

    def test_tanh_anchor_p95(self):
        """At P95, tanh should be ≈ 0.5."""
        from pipeline.targets import _vp_tanh

        vp = _vp_tanh(
            np.array([15.0]),
            np.array([10.0]), np.array([15.0]), np.array([25.0]),
        )
        assert abs(vp[0] - 0.5) < 1e-6

    def test_tanh_saturation(self):
        """Hard saturation at P90/P100."""
        from pipeline.targets import _vp_tanh

        vp = _vp_tanh(
            np.array([5.0, 30.0]),
            np.array([10.0, 10.0]),
            np.array([15.0, 15.0]),
            np.array([25.0, 25.0]),
        )
        assert vp[0] == 0.0
        assert vp[1] == 1.0

    def test_compute_vp_dispatcher(self):
        """compute_vp dispatches correctly to all methods."""
        from pipeline.config import TargetMode
        from pipeline.targets import compute_vp

        precip = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
        p90 = np.full(6, 10.0)
        p95 = np.full(6, 15.0)
        p100 = np.full(6, 25.0)

        for mode in [TargetMode.PIECEWISE, TargetMode.SIGMOID, TargetMode.TANH]:
            vp = compute_vp(precip, p90, p95, p100, method=mode, eps=0.02)
            assert vp.shape == (6,)
            assert vp.min() >= 0.0
            assert vp.max() <= 1.0
            # At P95 → ~0.5 for all methods
            assert abs(vp[2] - 0.5) < 0.01


# ======================================================================== #
#  Build target integration test                                            #
# ======================================================================== #

class TestBuildTarget:
    def test_build_target_produces_valid_vp(self):
        """build_target should produce VP in [0, 1] with correct shape."""
        from pipeline.config import FuzzyConfig, TargetMode
        from pipeline.targets import build_target

        n = 100
        df = pd.DataFrame({
            "pixel_id": np.repeat([0, 1], n // 2),
            "time": list(range(n)),
            "tp": np.random.exponential(1.0, n),
        })
        df = df.sort_values(["pixel_id", "time"])

        thresholds_aligned = pd.DataFrame({
            "p90": np.full(n, 0.5),
            "p95": np.full(n, 1.0),
            "p100": np.full(n, 3.0),
        })

        for mode in [TargetMode.PIECEWISE, TargetMode.SIGMOID, TargetMode.TANH]:
            cfg = FuzzyConfig(mode=mode, eps=0.02)
            y, meta = build_target(
                df, Q=1, thresholds_aligned=thresholds_aligned, target_config=cfg,
            )
            valid = y.dropna()
            assert valid.min() >= 0.0
            assert valid.max() <= 1.0
            assert meta["Q"] == 1
            assert meta["target_mode"] == mode.value


# ======================================================================== #
#  Splitting tests                                                          #
# ======================================================================== #

class TestSplitting:
    @pytest.fixture
    def time_df(self):
        dates = pd.date_range("2022-01-01", "2023-06-30", freq="D")
        return pd.DataFrame({"time": dates, "val": np.random.randn(len(dates))})

    def test_fixed_split_no_overlap(self, time_df):
        from pipeline.config import SplitConfig, SplitStrategy
        from pipeline.splitting import make_splits

        cfg = SplitConfig(
            strategy=SplitStrategy.FIXED,
            train_end="2022-12-31",
            val_end="2023-02-28",
        )
        splits = make_splits(time_df, cfg)
        tr, va, te = splits["train_mask"], splits["val_mask"], splits["test_mask"]
        assert not (tr & va).any()
        assert not (tr & te).any()
        assert not (va & te).any()
        assert (tr | va | te).all()

    def test_temporal_ordering(self, time_df):
        from pipeline.config import SplitConfig, SplitStrategy
        from pipeline.splitting import make_splits

        cfg = SplitConfig(
            strategy=SplitStrategy.FIXED,
            train_end="2022-12-31",
            val_end="2023-02-28",
        )
        splits = make_splits(time_df, cfg)
        tr, va, te = splits["train_mask"], splits["val_mask"], splits["test_mask"]
        assert time_df.loc[tr, "time"].max() < time_df.loc[va, "time"].min()
        assert time_df.loc[va, "time"].max() < time_df.loc[te, "time"].min()

    def test_scaler_fit_train_only(self):
        from pipeline.splitting import fit_transformers, transform_split

        train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        test = pd.DataFrame({"a": [4.0, 5.0], "b": [40.0, 50.0]})

        info = fit_transformers(train, ["a", "b"], "standard")
        test_scaled = transform_split(test, info)
        assert test_scaled["a"].iloc[0] > 1.0
        assert test_scaled["a"].iloc[1] > test_scaled["a"].iloc[0]


# ======================================================================== #
#  Leakage checks tests                                                     #
# ======================================================================== #

class TestLeakageChecks:
    def test_check_no_leakage_passes(self):
        from pipeline.splitting import check_no_leakage

        n = 30
        df = pd.DataFrame({
            "time": pd.date_range("2022-01-01", periods=n, freq="D"),
            "feat_1": np.random.randn(n),
        })
        tr = df.index < 10
        va = (df.index >= 10) & (df.index < 20)
        te = df.index >= 20
        check_no_leakage(df, tr, va, te, feature_cols=["feat_1"])

    def test_check_no_leakage_fails_on_overlap(self):
        from pipeline.splitting import check_no_leakage

        n = 30
        df = pd.DataFrame({
            "time": pd.date_range("2022-01-01", periods=n, freq="D"),
            "feat_1": np.random.randn(n),
        })
        tr = df.index < 15
        va = (df.index >= 10) & (df.index < 20)
        te = df.index >= 20
        with pytest.raises(AssertionError, match="LEAKAGE"):
            check_no_leakage(df, tr, va, te)


# ======================================================================== #
#  Model registry tests                                                     #
# ======================================================================== #

class TestModelRegistry:
    def test_get_all_models(self):
        from pipeline.models.registry import get_model_registry
        entries = get_model_registry()
        assert len(entries) > 10

    def test_instantiate_ridge(self):
        from pipeline.models.registry import get_model, instantiate_model
        entry = get_model("Ridge")
        model = instantiate_model(entry)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_instantiate_lightgbm(self):
        from pipeline.models.registry import get_model, instantiate_model
        entry = get_model("LightGBM")
        model = instantiate_model(entry)
        assert hasattr(model, "fit")


# ======================================================================== #
#  Evaluation tests                                                         #
# ======================================================================== #

class TestEvaluation:
    def test_compute_metrics_perfect(self):
        from pipeline.evaluation import compute_metrics
        y = np.array([0.0, 0.0, 1.0, 1.0])
        m = compute_metrics(y, y)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert m["accuracy"] == 1.0

    def test_compute_metrics_random(self):
        from pipeline.evaluation import compute_metrics
        rng = np.random.default_rng(42)
        y_true = rng.random(100)
        y_pred = rng.random(100)
        m = compute_metrics(y_true, y_pred)
        assert 0 <= m["brier"] <= 1
        assert "auc_roc" in m

    def test_compute_metrics_new_keys(self):
        """New metrics: smape, precision, recall, fpr, pr_auc."""
        from pipeline.evaluation import compute_metrics
        rng = np.random.default_rng(7)
        y_true = rng.random(200)
        y_pred = np.clip(y_true + rng.normal(0, 0.1, 200), 0, 1)
        m = compute_metrics(y_true, y_pred)
        assert "smape" in m
        assert "precision" in m
        assert "recall" in m
        assert "fpr" in m
        assert "pr_auc" in m
        assert 0 <= m["fpr"] <= 1

    def test_safe_smape_zero(self):
        """SMAPE should be 0 for identical arrays."""
        from pipeline.evaluation import _safe_smape
        y = np.array([0.3, 0.5, 0.8])
        assert _safe_smape(y, y) == pytest.approx(0.0, abs=1e-6)


# ======================================================================== #
#  TuningConfig tests                                                       #
# ======================================================================== #

class TestTuningConfig:
    def test_defaults(self):
        from pipeline.config import TuningConfig
        tc = TuningConfig()
        assert tc.enabled is False
        assert tc.n_trials == 30
        assert tc.metric == "brier"
        assert tc.direction == "minimize"
        assert tc.calibration_method == "f1"

    def test_to_dict(self):
        from pipeline.config import TuningConfig
        tc = TuningConfig(enabled=True, n_trials=50)
        d = tc.to_dict()
        assert d["enabled"] is True
        assert d["n_trials"] == 50
        assert "calibration_method" in d

    def test_pipeline_config_roundtrip_with_tuning(self, tmp_path):
        from pipeline.config import PipelineConfig, TuningConfig
        tc = TuningConfig(enabled=True, n_trials=20, cv_gap=2)
        cfg = PipelineConfig(p_values=[3], q_values=[1], tuning_config=tc)
        path = tmp_path / "cfg.json"
        cfg.save(path)
        loaded = PipelineConfig.load(path)
        assert loaded.tuning_config.enabled is True
        assert loaded.tuning_config.n_trials == 20
        assert loaded.tuning_config.cv_gap == 2

    def test_backward_compat_no_tuning(self, tmp_path):
        """Old config files without tuning_config should load fine."""
        import json
        from pipeline.config import PipelineConfig
        cfg = PipelineConfig()
        d = cfg.to_dict()
        del d["tuning_config"]  # simulate old config
        path = tmp_path / "old_cfg.json"
        path.write_text(json.dumps(d, indent=2))
        loaded = PipelineConfig.load(path)
        assert loaded.tuning_config.enabled is False


# ======================================================================== #
#  Temporal features tests                                                  #
# ======================================================================== #

class TestTemporalFeatures:
    def test_cyclical_features_generated(self):
        """Cyclical time features should appear when include_temporal=True."""
        from pipeline.config import FeatureConfig, SampleMode
        from pipeline.features import build_features

        rng = np.random.default_rng(42)
        n_times = 50
        rows = []
        for t in range(n_times):
            rows.append({
                "time": datetime(2022, 6, 1) + timedelta(hours=t),
                "pixel_id": 0,
                "y_idx": 0,
                "x_idx": 0,
                "tp": rng.exponential(0.5),
                "t2m": rng.normal(20, 5),
            })
        df = pd.DataFrame(rows)

        cfg = FeatureConfig(
            bands=["tp", "t2m"],
            rolling_windows=[3],
            rolling_stats=["mean"],
            include_temporal=True,
        )
        X, meta = build_features(df, P=3, feature_config=cfg, sample_mode=SampleMode.PIXEL)
        feat_names = meta["feature_names"]
        # Should contain cyclical features
        assert "hour_sin" in feat_names
        assert "hour_cos" in feat_names
        assert "month_sin" in feat_names
        assert "doy_sin" in feat_names

    def test_cyclical_features_not_generated_when_disabled(self):
        from pipeline.config import FeatureConfig, SampleMode
        from pipeline.features import build_features

        rng = np.random.default_rng(42)
        rows = []
        for t in range(50):
            rows.append({
                "time": datetime(2022, 6, 1) + timedelta(hours=t),
                "pixel_id": 0, "y_idx": 0, "x_idx": 0,
                "tp": rng.exponential(0.5),
            })
        df = pd.DataFrame(rows)
        cfg = FeatureConfig(bands=["tp"], include_temporal=False)
        X, meta = build_features(df, P=3, feature_config=cfg, sample_mode=SampleMode.PIXEL)
        assert "hour_sin" not in meta["feature_names"]

    def test_cyclical_values_bounded(self):
        """Sin/cos values must be in [-1, 1]."""
        from pipeline.config import FeatureConfig, SampleMode
        from pipeline.features import build_features

        rng = np.random.default_rng(42)
        rows = []
        for t in range(100):
            rows.append({
                "time": datetime(2022, 1, 1) + timedelta(hours=t),
                "pixel_id": 0, "y_idx": 0, "x_idx": 0,
                "tp": rng.exponential(0.5),
            })
        df = pd.DataFrame(rows)
        cfg = FeatureConfig(bands=["tp"], include_temporal=True)
        X, meta = build_features(df, P=3, feature_config=cfg, sample_mode=SampleMode.PIXEL)
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]:
            if col in X.columns:
                assert X[col].min() >= -1.0 - 1e-10
                assert X[col].max() <= 1.0 + 1e-10


# ======================================================================== #
#  needs_scaling registry tests                                             #
# ======================================================================== #

class TestNeedsScaling:
    def test_tree_models_no_scaling(self):
        from pipeline.models.registry import get_model
        for name in ["DecisionTree", "RandomForest", "GradientBoosting", "LightGBM"]:
            entry = get_model(name)
            assert entry["needs_scaling"] is False, f"{name} should not need scaling"

    def test_linear_models_need_scaling(self):
        from pipeline.models.registry import get_model
        for name in ["Ridge", "Lasso", "ElasticNet"]:
            entry = get_model(name)
            assert entry["needs_scaling"] is True, f"{name} should need scaling"

    @pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")
    def test_catboost_registered(self):
        from pipeline.models.registry import get_model, instantiate_model
        entry = get_model("CatBoost")
        assert entry["needs_scaling"] is False
        model = instantiate_model(entry)
        assert hasattr(model, "fit")


# ======================================================================== #
#  Anomaly module tests                                                     #
# ======================================================================== #

class TestAnomaly:
    def test_calibrate_threshold_f1(self):
        """calibrate_threshold should find a reasonable threshold."""
        from pipeline.anomaly import calibrate_threshold
        rng = np.random.default_rng(42)
        # Create a scenario where high VP = anomaly
        y_true = np.concatenate([rng.uniform(0, 0.3, 80), rng.uniform(0.6, 1.0, 20)])
        y_pred = y_true + rng.normal(0, 0.05, 100)
        y_pred = np.clip(y_pred, 0, 1)
        threshold, metrics = calibrate_threshold(y_true, y_pred, method="f1")
        assert 0.0 < threshold < 1.0
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_calibrate_threshold_youden(self):
        from pipeline.anomaly import calibrate_threshold
        rng = np.random.default_rng(123)
        y_true = np.concatenate([rng.uniform(0, 0.3, 70), rng.uniform(0.6, 1.0, 30)])
        y_pred = np.clip(y_true + rng.normal(0, 0.1, 100), 0, 1)
        threshold, metrics = calibrate_threshold(y_true, y_pred, method="youden")
        assert 0.0 < threshold < 1.0

    def test_classify_anomalies(self):
        from pipeline.anomaly import classify_anomalies
        y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        labels = classify_anomalies(y_pred, threshold=0.5)
        np.testing.assert_array_equal(labels, [0, 0, 1, 1])

    def test_compute_anomaly_report_structure(self):
        from pipeline.anomaly import compute_anomaly_report
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.random(n)
        y_pred = np.clip(y_true + rng.normal(0, 0.1, n), 0, 1)
        times = pd.date_range("2023-01-01", periods=n, freq="h")

        report = compute_anomaly_report(y_true, y_pred, threshold=0.5, times=times.values)
        assert "overall" in report
        assert "confusion_matrix" in report
        assert "by_intensity" in report
        assert "residuals" in report
        # Confusion matrix sums to n
        cm = report["confusion_matrix"]
        assert cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"] == n

    def test_report_residuals(self):
        from pipeline.anomaly import compute_anomaly_report
        y_true = np.array([0.5, 0.5, 0.5])
        y_pred = np.array([0.4, 0.5, 0.6])
        report = compute_anomaly_report(y_true, y_pred, threshold=0.5)
        assert abs(report["residuals"]["mean"]) < 0.01
        assert report["residuals"]["std"] > 0

    def test_degenerate_single_class(self):
        """When all values are one class, calibration should not crash."""
        from pipeline.anomaly import calibrate_threshold
        y_true = np.zeros(50)  # all below cutoff
        y_pred = np.random.default_rng(42).random(50) * 0.3
        threshold, metrics = calibrate_threshold(y_true, y_pred)
        assert threshold == 0.5  # default fallback


# ======================================================================== #
#  Purged temporal CV tests                                                 #
# ======================================================================== #

class TestPurgedTemporalCV:
    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_basic_folds(self):
        from pipeline.tuning import purged_temporal_cv
        times = pd.date_range("2022-01-01", periods=400, freq="D").values
        folds = purged_temporal_cv(times, n_folds=3, gap=0)
        assert len(folds) > 0
        for train_idx, val_idx in folds:
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            # No overlap
            assert len(set(train_idx) & set(val_idx)) == 0

    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_gap_enforced(self):
        from pipeline.tuning import purged_temporal_cv
        times = pd.date_range("2022-01-01", periods=200, freq="D").values
        gap = 5
        folds = purged_temporal_cv(times, n_folds=3, gap=gap)
        unique_times = np.sort(np.unique(times))
        for train_idx, val_idx in folds:
            # Max train time must be < min val time - gap
            max_train_time = times[train_idx].max()
            min_val_time = times[val_idx].min()
            train_pos = np.searchsorted(unique_times, max_train_time)
            val_pos = np.searchsorted(unique_times, min_val_time)
            assert val_pos - train_pos >= gap, "Gap not enforced"

    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_expanding_window(self):
        """Each fold's training set should be larger than the previous."""
        from pipeline.tuning import purged_temporal_cv
        times = pd.date_range("2022-01-01", periods=300, freq="D").values
        folds = purged_temporal_cv(times, n_folds=3, gap=0)
        if len(folds) >= 2:
            for i in range(1, len(folds)):
                assert len(folds[i][0]) > len(folds[i - 1][0])


# ======================================================================== #
#  Optuna HP Tuner smoke test                                               #
# ======================================================================== #

class TestOptunaHPTuner:
    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_tune_ridge(self):
        """Tuner should run and return params for Ridge."""
        from pipeline.tuning import OptunaHPTuner
        rng = np.random.default_rng(42)
        X = rng.random((200, 5))
        y = X @ rng.random(5) + rng.normal(0, 0.1, 200)
        y = np.clip(y, 0, 1)
        times = pd.date_range("2022-01-01", periods=200, freq="D").values

        tuner = OptunaHPTuner(
            model_name="Ridge",
            X_train=X,
            y_train=y,
            times_train=times,
            n_folds=2,
        )
        best = tuner.tune(n_trials=5, timeout=30)
        assert isinstance(best, dict)
        assert "alpha" in best

    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_tune_with_no_space(self):
        """Model with empty search space should return empty dict."""
        from pipeline.tuning import OptunaHPTuner
        rng = np.random.default_rng(42)
        X = rng.random((200, 5))
        y = rng.random(200)
        times = pd.date_range("2022-01-01", periods=200, freq="D").values
        tuner = OptunaHPTuner(
            model_name="BayesianRidge",
            X_train=X, y_train=y, times_train=times, n_folds=2,
        )
        best = tuner.tune(n_trials=3, timeout=10)
        assert isinstance(best, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
