"""
pipeline.models.registry
========================
Centralised catalogue of every model family the pipeline supports.

Each entry describes:
  • class / constructor
  • expected input format  (``tabular`` vs ``sequence``)
  • default hyperparameters
  • recommended preprocessing notes

Model Families
--------------
=====  =====================  ==============  ==============================
Code   Family                 Input shape     Notes
=====  =====================  ==============  ==============================
C      TransformedTarget      tabular         Wraps any regressor
CD     CrossDecomposition     tabular         PLS regression
D      Dummy                  tabular         Baseline (mean / median)
E      Ensemble               tabular         RF, GBR, XGBoost, LightGBM
G      GaussianProcess        tabular         GP with RBF + WhiteKernel
L      LinearModel            tabular         Ridge, Lasso, ElasticNet, …
N      Neighbors              tabular         k-NN, radius neighbours
NN     NeuralNetwork          tabular         MLPRegressor
S      SVM                    tabular         LinearSVR, NuSVR
T      Tree                   tabular         DecisionTree, ExtraTree
DL     LSTM                   3-D sequence    Plain LSTM
DL     LSTMPre               3-D sequence    Dense → LSTM hybrid
=====  =====================  ==============  ==============================
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ======================================================================== #
#  Registry structure                                                       #
# ======================================================================== #

def _entry(
    cls_path: str,
    code: str,
    name: str,
    input_type: str = "tabular",
    default_params: Optional[Dict] = None,
    preprocessing: str = "standard",
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "cls_path": cls_path,
        "code": code,
        "name": name,
        "input_type": input_type,  # "tabular" | "sequence"
        "default_params": default_params or {},
        "preprocessing": preprocessing,
        "notes": notes,
    }


# ======================================================================== #
#  The registry                                                             #
# ======================================================================== #

MODEL_FAMILIES: Dict[str, List[Dict[str, Any]]] = {
    # ---- D: Dummy / baseline ----
    "D": [
        _entry(
            "sklearn.dummy.DummyRegressor", "D", "DummyMean",
            default_params={"strategy": "mean"},
            notes="Always predicts the training mean — lower bound baseline.",
        ),
        _entry(
            "sklearn.dummy.DummyRegressor", "D", "DummyMedian",
            default_params={"strategy": "median"},
        ),
    ],

    # ---- L: Linear models ----
    "L": [
        _entry(
            "sklearn.linear_model.Ridge", "L", "Ridge",
            default_params={"alpha": 1.0},
            notes="L2-regularised OLS.  Fast, good baseline.",
        ),
        _entry(
            "sklearn.linear_model.Lasso", "L", "Lasso",
            default_params={"alpha": 0.01, "max_iter": 5000},
            notes="L1-regularised — produces sparse coefficients.",
        ),
        _entry(
            "sklearn.linear_model.ElasticNet", "L", "ElasticNet",
            default_params={"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 5000},
        ),
        _entry(
            "sklearn.linear_model.BayesianRidge", "L", "BayesianRidge",
            default_params={},
            notes="Bayesian ARD-style ridge — returns uncertainty.",
        ),
        _entry(
            "sklearn.linear_model.SGDRegressor", "L", "SGDRegressor",
            default_params={"max_iter": 1000, "tol": 1e-4, "random_state": 42},
            preprocessing="standard",
            notes="Online SGD — scales to very large datasets.",
        ),
    ],

    # ---- T: Tree ----
    "T": [
        _entry(
            "sklearn.tree.DecisionTreeRegressor", "T", "DecisionTree",
            default_params={"max_depth": 12, "random_state": 42},
            preprocessing="none",
            notes="No scaling needed.  Overfits easily — use as sanity check.",
        ),
        _entry(
            "sklearn.tree.ExtraTreeRegressor", "T", "ExtraTree",
            default_params={"max_depth": 12, "random_state": 42},
            preprocessing="none",
        ),
    ],

    # ---- E: Ensemble ----
    "E": [
        _entry(
            "sklearn.ensemble.RandomForestRegressor", "E", "RandomForest",
            default_params={
                "n_estimators": 300,
                "max_depth": 16,
                "min_samples_leaf": 5,
                "n_jobs": -1,
                "random_state": 42,
            },
            preprocessing="none",
        ),
        _entry(
            "sklearn.ensemble.GradientBoostingRegressor", "E", "GradientBoosting",
            default_params={
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "random_state": 42,
            },
            preprocessing="none",
        ),
        _entry(
            "xgboost.XGBRegressor", "E", "XGBoost",
            default_params={
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
            },
            preprocessing="none",
            notes="Requires xgboost package.",
        ),
        _entry(
            "lightgbm.LGBMRegressor", "E", "LightGBM",
            default_params={
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 64,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": -1,
            },
            preprocessing="none",
            notes="Requires lightgbm package.",
        ),
    ],

    # ---- N: Neighbors ----
    "N": [
        _entry(
            "sklearn.neighbors.KNeighborsRegressor", "N", "KNN",
            default_params={"n_neighbors": 10, "weights": "distance", "n_jobs": -1},
            preprocessing="standard",
            notes="Sensitive to scaling.  Slow on large datasets.",
        ),
    ],

    # ---- S: SVM ----
    "S": [
        _entry(
            "sklearn.svm.LinearSVR", "S", "LinearSVR",
            default_params={"C": 1.0, "max_iter": 5000},
            preprocessing="standard",
            notes="Scales better than kernel SVM.  Needs scaled inputs.",
        ),
        _entry(
            "sklearn.svm.NuSVR", "S", "NuSVR",
            default_params={"nu": 0.5, "C": 1.0, "kernel": "rbf"},
            preprocessing="standard",
            notes="Kernel SVM — very slow on >50k samples.",
        ),
    ],

    # ---- NN: Neural network (sklearn) ----
    "NN": [
        _entry(
            "sklearn.neural_network.MLPRegressor", "NN", "MLPRegressor",
            default_params={
                "hidden_layer_sizes": (128, 64),
                "activation": "relu",
                "solver": "adam",
                "max_iter": 500,
                "early_stopping": True,
                "random_state": 42,
            },
            preprocessing="standard",
        ),
    ],

    # ---- G: Gaussian Process ----
    "G": [
        _entry(
            "sklearn.gaussian_process.GaussianProcessRegressor", "G",
            "GaussianProcess",
            default_params={},  # kernel set in instantiate_model
            preprocessing="standard",
            notes=(
                "MLE kernel fitting (RBF + WhiteKernel).  Very slow > 5k rows — "
                "subsample or use sparse approximation."
            ),
        ),
    ],

    # ---- C: Transformed-target wrapper ----
    "C": [
        _entry(
            "sklearn.compose.TransformedTargetRegressor", "C",
            "TransformedTarget_Ridge",
            default_params={"regressor": "Ridge"},
            preprocessing="standard",
            notes="Wraps a base regressor; transforms y, trains, inverse-transforms preds.",
        ),
    ],

    # ---- CD: Cross-decomposition ----
    "CD": [
        _entry(
            "sklearn.cross_decomposition.PLSRegression", "CD", "PLSRegression",
            default_params={"n_components": 5, "max_iter": 500},
            preprocessing="standard",
        ),
    ],

    # ---- DL: Deep Learning (PyTorch LSTM) ----
    "DL": [
        _entry(
            "pipeline.models.lstm.LSTMModel", "DL", "LSTM",
            input_type="sequence",
            default_params={
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "lr": 1e-3,
                "epochs": 50,
                "batch_size": 256,
                "patience": 10,
            },
            preprocessing="standard",
            notes="Plain LSTM for sequence input (P, n_features).",
        ),
        _entry(
            "pipeline.models.lstm.LSTMPreModel", "DL", "LSTMPre",
            input_type="sequence",
            default_params={
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "pre_dense_sizes": [128, 64],
                "lr": 1e-3,
                "epochs": 50,
                "batch_size": 256,
                "patience": 10,
            },
            preprocessing="standard",
            notes="Dense layers before LSTM — learns per-timestep features first.",
        ),
    ],
}


# ======================================================================== #
#  Public helpers                                                           #
# ======================================================================== #

def get_model_registry(
    families: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Return a flat list of model entries, optionally filtered by family codes.

    Parameters
    ----------
    families : list of str, optional
        E.g. ``['E', 'L', 'DL']``.  None = all families.
    """
    if families is None:
        families = list(MODEL_FAMILIES.keys())

    entries = []
    for code in families:
        if code in MODEL_FAMILIES:
            entries.extend(MODEL_FAMILIES[code])
        else:
            warnings.warn(f"Unknown model family code: {code}")
    return entries


def get_model(name: str) -> Dict[str, Any]:
    """Look up a model entry by its unique name."""
    for entries in MODEL_FAMILIES.values():
        for e in entries:
            if e["name"] == name:
                return e
    raise KeyError(f"Model '{name}' not found in registry.")


def instantiate_model(entry: Dict[str, Any], **overrides) -> Any:
    """
    Dynamically import and instantiate a model from its registry entry.

    Parameters
    ----------
    entry : dict
        As returned by :func:`get_model`.
    **overrides
        Override any default hyperparameter.

    Returns
    -------
    model instance
    """
    cls = _import_class(entry["cls_path"])
    params = {**entry["default_params"], **overrides}

    # Special handling for certain models
    name = entry["name"]

    # Gaussian Process — construct kernel
    if name == "GaussianProcess":
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        params["kernel"] = kernel
        params.setdefault("n_restarts_optimizer", 3)
        params.setdefault("random_state", 42)

    # TransformedTarget — wrap a base regressor
    if name.startswith("TransformedTarget"):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import QuantileTransformer
        base_reg = params.pop("regressor", "Ridge")
        if base_reg == "Ridge":
            base_reg = Ridge(alpha=1.0)
        params["regressor"] = base_reg
        params["transformer"] = QuantileTransformer(
            output_distribution="normal", random_state=42
        )

    return cls(**params)


def _import_class(dotted_path: str):
    """Import a class from a dotted module path like 'sklearn.linear_model.Ridge'."""
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ImportError(f"Invalid class path: {dotted_path}")
    module_path, class_name = parts
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_path}' — "
            f"is the package installed?  ({e})"
        )
    return getattr(module, class_name)
