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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
