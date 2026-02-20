"""
pipeline.evaluation
===================
Metrics computation and reporting for VP (Valor de Probabilidade) targets.

Metrics
-------
Since the target is a continuous VP score in [0, 1] anchored on P90/P95/P100:

* **Brier score**  — proper scoring rule for probabilistic forecasts.
* **RMSE**         — root mean squared error on the continuous VP score.
* **MAE**          — mean absolute error on the continuous VP score.
* **R²**           — coefficient of determination.
* **Binary accuracy / F1** — after thresholding VP at 0.5 (i.e. P95).
* **AUC-ROC**      — treating (VP > 0.5) as the positive class.
* **Log-loss**     — binary cross-entropy (clipped predictions).
* **Focal loss**   — weighted BCE that down-weights easy examples.

Loss Recommendations for VP Targets
------------------------------------
=============  ======  ===========  ======================================
Loss           Type    Best when    Notes
=============  ======  ===========  ======================================
MSE            cont.   balanced     Simple, smooth gradient, good default.
BCE            prob.   imbalanced   Treats VP as soft probability.
Focal          prob.   rare events  γ=2 focuses on hard examples.
Huber          cont.   outliers     L1-like for large errors.
=============  ======  ===========  ======================================

For VP targets where extreme events (VP > 0.5) are rare, **Focal loss**
or **BCE** tend to outperform MSE.  Use MSE / Huber if the VP distribution
is relatively balanced.

Public API
----------
compute_metrics(y_true, y_pred)  → dict
train_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, ...)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


# ======================================================================== #
#  Metrics                                                                  #
# ======================================================================== #

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of metrics for fuzzy / probabilistic targets.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth values in [0, 1].
    y_pred : array-like, shape (n,)
        Predicted values (clipped to [0, 1] internally).
    threshold : float
        Cut-off for converting scores to binary labels.

    Returns
    -------
    dict
        Keys: ``brier, rmse, mae, r2, accuracy, f1, log_loss, auc_roc``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)

    y_bin_true = (y_true > threshold).astype(int)
    y_pred_clip = np.clip(y_pred, 1e-7, 1.0 - 1e-7)  # for log-loss

    metrics: Dict[str, float] = {}
    metrics["brier"] = float(brier_score_loss(y_bin_true, y_pred))
    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["r2"] = float(r2_score(y_true, y_pred))

    y_bin_pred = (y_pred > threshold).astype(int)
    metrics["accuracy"] = float(accuracy_score(y_bin_true, y_bin_pred))
    metrics["f1"] = float(f1_score(y_bin_true, y_bin_pred, zero_division=0))
    metrics["log_loss"] = float(log_loss(y_bin_true, y_pred_clip))

    # AUC — requires both classes present
    if len(np.unique(y_bin_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_bin_true, y_pred))
    else:
        metrics["auc_roc"] = float("nan")

    return metrics


# ======================================================================== #
#  Train + evaluate                                                         #
# ======================================================================== #

def train_evaluate(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "",
    input_type: str = "tabular",
    P: Optional[int] = None,
    n_features: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fit a model on train, evaluate on val + test.

    Parameters
    ----------
    model : estimator
        Sklearn-compatible (has ``.fit()`` / ``.predict()``).
    X_train, y_train, X_val, y_val, X_test, y_test :
        Feature / target arrays.
    model_name : str
    input_type : str
        ``"tabular"`` or ``"sequence"``.  For sequence models the arrays
        are reshaped to 3-D ``(N, P, n_features)`` before fitting.
    P, n_features : int, optional
        Required for sequence models to reshape flat features.

    Returns
    -------
    dict
        ``{"model_name": ..., "val_metrics": {...}, "test_metrics": {...},
           "model": fitted_model}``
    """
    # Reshape for LSTM-family models
    if input_type == "sequence" and P is not None and n_features is not None:
        if hasattr(model, "P"):
            model.P = P
            model.n_features = n_features

    # Fit
    try:
        # Some models support eval_set for early stopping (LightGBM, XGBoost)
        if _supports_eval_set(model):
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )
        elif input_type == "sequence" and hasattr(model, "fit"):
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train, y_train)
    except Exception as e:
        return {
            "model_name": model_name,
            "error": str(e),
            "val_metrics": {},
            "test_metrics": {},
            "model": None,
        }

    # Predict + evaluate
    val_pred = np.clip(model.predict(X_val), 0.0, 1.0)
    test_pred = np.clip(model.predict(X_test), 0.0, 1.0)

    return {
        "model_name": model_name,
        "val_metrics": compute_metrics(y_val, val_pred),
        "test_metrics": compute_metrics(y_test, test_pred),
        "model": model,
        "test_predictions": test_pred,
    }


def _supports_eval_set(model) -> bool:
    """Check if the model's fit() accepts eval_set (boosting models)."""
    cls_name = type(model).__name__
    return cls_name in ("LGBMRegressor", "XGBRegressor")
