"""
pipeline.tuning
===============
Optuna-based hyperparameter optimisation with **purged temporal
cross-validation**.

Design Principles
-----------------
1. **No future leakage** — HP search uses expanding-window temporal CV
   *within the training set only*.  A configurable gap between train/val
   folds prevents information bleeding from rolling/lag features.

2. **Metric alignment** — the objective optimises the *same* metric used
   for final comparison (Brier score by default, lower = better).

3. **Parallelism** — Optuna trials run sequentially to avoid
   over-subscription when tree-based models also parallelise internally.

4. **Search spaces** — defined per model name in ``SEARCH_SPACES``.
   Any entry can be overridden at runtime.

Public API
----------
``OptunaHPTuner(model_name, X_train, y_train, ...)``
  ``.tune(n_trials, timeout)  → best_params``
  ``.study                    → optuna.Study``

``purged_temporal_cv(times, n_folds, gap) → [(train_idx, val_idx), ...]``
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .evaluation import compute_metrics
from .models.registry import get_model, instantiate_model

logger = logging.getLogger(__name__)


# ======================================================================== #
#  Per-model search spaces                                                  #
# ======================================================================== #
# Format:  param_name → (type, *args)
#   type ∈ {"int", "float", "log_float", "categorical"}

SEARCH_SPACES: Dict[str, Dict[str, tuple]] = {
    # ---- Linear ----
    "Ridge": {
        "alpha": ("log_float", 1e-4, 100.0),
    },
    "Lasso": {
        "alpha": ("log_float", 1e-5, 1.0),
    },
    "ElasticNet": {
        "alpha": ("log_float", 1e-5, 1.0),
        "l1_ratio": ("float", 0.05, 0.95),
    },
    "BayesianRidge": {},  # no free HPs worth tuning
    "SGDRegressor": {
        "alpha": ("log_float", 1e-6, 1e-1),
        "l1_ratio": ("float", 0.0, 1.0),
        "penalty": ("categorical", ["l2", "l1", "elasticnet"]),
    },

    # ---- Tree ----
    "DecisionTree": {
        "max_depth": ("int", 4, 24),
        "min_samples_leaf": ("int", 2, 50),
        "min_samples_split": ("int", 2, 20),
    },
    "ExtraTree": {
        "max_depth": ("int", 4, 24),
        "min_samples_leaf": ("int", 2, 50),
    },

    # ---- Ensemble ----
    "RandomForest": {
        "n_estimators": ("int", 100, 500),
        "max_depth": ("int", 6, 24),
        "min_samples_leaf": ("int", 2, 30),
        "min_samples_split": ("int", 2, 20),
        "max_features": ("categorical", ["sqrt", "log2", 0.5, 0.8]),
    },
    "GradientBoosting": {
        "n_estimators": ("int", 100, 500),
        "learning_rate": ("log_float", 0.01, 0.3),
        "max_depth": ("int", 3, 10),
        "subsample": ("float", 0.6, 1.0),
        "min_samples_leaf": ("int", 2, 30),
    },
    "XGBoost": {
        "n_estimators": ("int", 100, 500),
        "learning_rate": ("log_float", 0.01, 0.3),
        "max_depth": ("int", 3, 10),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "reg_alpha": ("log_float", 1e-8, 10.0),
        "reg_lambda": ("log_float", 1e-8, 10.0),
        "min_child_weight": ("int", 1, 10),
    },
    "LightGBM": {
        "n_estimators": ("int", 100, 500),
        "learning_rate": ("log_float", 0.01, 0.3),
        "num_leaves": ("int", 16, 128),
        "subsample": ("float", 0.6, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "min_child_samples": ("int", 5, 50),
        "reg_alpha": ("log_float", 1e-8, 10.0),
        "reg_lambda": ("log_float", 1e-8, 10.0),
    },
    "CatBoost": {
        "iterations": ("int", 100, 500),
        "learning_rate": ("log_float", 0.01, 0.3),
        "depth": ("int", 3, 10),
        "l2_leaf_reg": ("log_float", 1e-3, 10.0),
        "subsample": ("float", 0.6, 1.0),
    },

    # ---- Neighbours ----
    "KNN": {
        "n_neighbors": ("int", 3, 50),
        "weights": ("categorical", ["uniform", "distance"]),
        "p": ("int", 1, 2),
    },

    # ---- SVM ----
    "LinearSVR": {
        "C": ("log_float", 1e-3, 100.0),
        "epsilon": ("log_float", 1e-4, 1.0),
    },

    # ---- Neural (sklearn) ----
    "MLPRegressor": {
        "hidden_layer_sizes": ("categorical", [(64,), (128, 64), (256, 128), (128, 64, 32)]),
        "alpha": ("log_float", 1e-5, 1e-1),
        "learning_rate_init": ("log_float", 1e-4, 1e-2),
    },

    # ---- Deep Learning ----
    "LSTM": {
        "hidden_size": ("categorical", [32, 64, 128]),
        "num_layers": ("int", 1, 3),
        "dropout": ("float", 0.1, 0.5),
        "lr": ("log_float", 1e-4, 1e-2),
        "batch_size": ("categorical", [128, 256, 512]),
    },
    "LSTMPre": {
        "hidden_size": ("categorical", [32, 64, 128]),
        "num_layers": ("int", 1, 3),
        "dropout": ("float", 0.1, 0.5),
        "lr": ("log_float", 1e-4, 1e-2),
        "batch_size": ("categorical", [128, 256, 512]),
    },
}


# ======================================================================== #
#  Purged temporal cross-validation                                         #
# ======================================================================== #

def purged_temporal_cv(
    times: np.ndarray,
    n_folds: int = 3,
    gap: int = 0,
    min_train_size: int = 100,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window temporal CV with an optional purge gap.

    Parameters
    ----------
    times : np.ndarray
        Array of timestamps (datetime-like) for every row.
    n_folds : int
        Number of validation folds.
    gap : int
        Number of unique time steps to skip between train and val
        to prevent leakage from rolling/lag features.
    min_train_size : int
        Minimum number of training rows per fold.

    Returns
    -------
    List of ``(train_indices, val_indices)`` tuples.

    Illustration (n_folds=3, gap=g)::

        ──────────────────────────────────────────────→ time
        [===TRAIN_1===]  g  [=VAL_1=]
        [======TRAIN_2======]  g  [=VAL_2=]
        [=========TRAIN_3=========]  g  [=VAL_3=]
    """
    unique_times = np.sort(np.unique(times))
    n = len(unique_times)
    fold_size = max(1, n // (n_folds + 1))

    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    for k in range(1, n_folds + 1):
        train_end = k * fold_size
        val_start = train_end + gap
        val_end = min(val_start + fold_size, n)

        if val_start >= n or val_end <= val_start:
            break

        train_times_set = set(unique_times[:train_end])
        val_times_set = set(unique_times[val_start:val_end])

        train_idx = np.where(np.isin(times, list(train_times_set)))[0]
        val_idx = np.where(np.isin(times, list(val_times_set)))[0]

        if len(train_idx) >= min_train_size and len(val_idx) > 0:
            folds.append((train_idx, val_idx))

    return folds


# ======================================================================== #
#  Optuna HP Tuner                                                          #
# ======================================================================== #

class OptunaHPTuner:
    """
    Hyperparameter tuner using Optuna + temporal CV.

    Parameters
    ----------
    model_name : str
        Name from the model registry (e.g. ``"LightGBM"``).
    X_train : np.ndarray
        Training features (2-D for tabular, 3-D for sequence).
    y_train : np.ndarray
        Training targets.
    times_train : np.ndarray, optional
        Timestamps for temporal CV. If ``None``, falls back to sequential
        index-based splits.
    metric : str
        Metric to optimise (default ``"brier"``).
    direction : str
        ``"minimize"`` or ``"maximize"``.
    n_folds : int
        Number of temporal CV folds.
    gap : int
        Purge gap in unique time steps.
    random_seed : int
    input_type : str
        ``"tabular"`` or ``"sequence"``.
    P, n_features : int, optional
        For LSTM-family models (sequence reshape).
    custom_space : dict, optional
        Override default search space for this model.
    """

    def __init__(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        times_train: Optional[np.ndarray] = None,
        metric: str = "brier",
        direction: str = "minimize",
        n_folds: int = 3,
        gap: int = 0,
        random_seed: int = 42,
        input_type: str = "tabular",
        P: Optional[int] = None,
        n_features: Optional[int] = None,
        custom_space: Optional[Dict[str, tuple]] = None,
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for HP tuning.  "
                "Install with:  pip install optuna"
            )

        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.metric = metric
        self.direction = direction
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.input_type = input_type
        self.P = P
        self.n_features = n_features
        self.custom_space = custom_space
        self.study: Optional["optuna.Study"] = None

        # Build CV folds
        if times_train is not None:
            self.cv_folds = purged_temporal_cv(
                times_train, n_folds=n_folds, gap=gap,
            )
        else:
            # Fallback: sequential index-based splits
            n = len(X_train)
            fold_size = max(1, n // (n_folds + 1))
            self.cv_folds = []
            for k in range(1, n_folds + 1):
                ti = np.arange(k * fold_size)
                vi = np.arange(k * fold_size, min((k + 1) * fold_size, n))
                if len(vi) > 0:
                    self.cv_folds.append((ti, vi))

        if not self.cv_folds:
            warnings.warn(
                f"No valid CV folds for {model_name}. "
                f"Tuning will fall back to defaults."
            )

    # ------------------------------------------------------------------ #

    def _sample_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """Sample hyperparameters from the search space."""
        space = self.custom_space or SEARCH_SPACES.get(self.model_name, {})
        params: Dict[str, Any] = {}

        for name, spec in space.items():
            ptype = spec[0]
            if ptype == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif ptype == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif ptype == "log_float":
                params[name] = trial.suggest_float(
                    name, spec[1], spec[2], log=True,
                )
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, spec[1])

        return params

    # ------------------------------------------------------------------ #

    def _objective(self, trial: "optuna.Trial") -> float:
        """Optuna objective: mean CV metric across temporal folds."""
        params = self._sample_params(trial)

        scores: List[float] = []
        for train_idx, val_idx in self.cv_folds:
            X_tr = self.X_train[train_idx]
            y_tr = self.y_train[train_idx]
            X_va = self.X_train[val_idx]
            y_va = self.y_train[val_idx]

            try:
                entry = get_model(self.model_name)
                model = instantiate_model(entry, **params)

                # Sequence model setup
                if self.input_type == "sequence" and hasattr(model, "P"):
                    model.P = self.P
                    model.n_features = self.n_features

                # Fit
                if self.input_type == "sequence":
                    model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
                elif _supports_eval_set(model):
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
                else:
                    model.fit(X_tr, y_tr)

                # Evaluate
                preds = np.clip(model.predict(X_va), 0.0, 1.0)
                m = compute_metrics(y_va, preds)
                scores.append(m[self.metric])

            except Exception as exc:
                logger.warning("Trial %d fold failed: %s", trial.number, exc)
                return (
                    float("inf")
                    if self.direction == "minimize"
                    else float("-inf")
                )

        if not scores:
            return (
                float("inf")
                if self.direction == "minimize"
                else float("-inf")
            )
        return float(np.mean(scores))

    # ------------------------------------------------------------------ #

    def tune(
        self,
        n_trials: int = 30,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
    ) -> Dict[str, Any]:
        """
        Run the Optuna study.

        Parameters
        ----------
        n_trials : int
            Number of HP combinations to evaluate.
        timeout : int, optional
            Maximum seconds for the study.
        n_jobs : int
            Parallel trials.  Keep at 1 for tree-based models that
            already parallelise internally.

        Returns
        -------
        dict — best hyperparameters found.
        """
        if not self.cv_folds:
            logger.warning(
                "No CV folds available — returning empty best_params."
            )
            return {}

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = TPESampler(seed=self.random_seed)
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            study_name=f"tune_{self.model_name}",
        )

        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=False,
        )

        best = self.study.best_params
        logger.info(
            "Best params for %s (Brier=%.4f): %s",
            self.model_name,
            self.study.best_value,
            best,
        )
        return best


# ======================================================================== #
#  Internal helpers                                                         #
# ======================================================================== #

def _supports_eval_set(model: Any) -> bool:
    """Check if the model's fit() accepts eval_set."""
    cls_name = type(model).__name__
    return cls_name in (
        "LGBMRegressor", "XGBRegressor", "CatBoostRegressor",
    )
