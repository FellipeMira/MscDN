"""
pipeline.evaluation
===================
Metrics computation and reporting for VP (Valor de Probabilidade) targets.

Two Metric Families
-------------------
**Regression metrics** — evaluate the continuous VP score ∈ [0, 1]:
  Brier, RMSE, MAE, R², SMAPE

**Anomaly-detection metrics** — after thresholding VP (default 0.5 ↔ P95):
  Precision, Recall, F1, PR-AUC, ROC-AUC, Log-loss, FPR

Both sets are computed in ``compute_metrics()`` and returned in a single dict
prefixed by their family for clarity.

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

Public API
----------
compute_metrics(y_true, y_pred)  → dict
train_evaluate(model, ...)       → dict with model, val/test metrics
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ======================================================================== #
#  Metrics                                                                  #
# ======================================================================== #

def _safe_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (bounded 0-200%)."""
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-12
    if mask.sum() == 0:
        return 0.0
    return float(
        200.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask])
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of metrics for VP targets.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth VP values in [0, 1].
    y_pred : array-like, shape (n,)
        Predicted VP values (clipped to [0, 1] internally).
    threshold : float
        Cut-off for binary anomaly classification.

    Returns
    -------
    dict  — keys include:
        *Regression*: ``brier, rmse, mae, r2, smape``
        *Anomaly*: ``accuracy, precision, recall, f1, fpr, log_loss,
        auc_roc, pr_auc``
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)

    # ---- Binary labels (VP > threshold ↔ extreme event) ----
    y_bin_true = (y_true > threshold).astype(int)
    y_pred_clip = np.clip(y_pred, 1e-7, 1.0 - 1e-7)

    metrics: Dict[str, float] = {}

    # ---- Regression ----
    metrics["brier"] = float(brier_score_loss(y_bin_true, y_pred))
    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["r2"] = float(r2_score(y_true, y_pred))
    metrics["smape"] = _safe_smape(y_true, y_pred)

    # ---- Anomaly detection (binary) ----
    y_bin_pred = (y_pred > threshold).astype(int)
    metrics["accuracy"] = float(accuracy_score(y_bin_true, y_bin_pred))
    metrics["precision"] = float(
        precision_score(y_bin_true, y_bin_pred, zero_division=0)
    )
    metrics["recall"] = float(
        recall_score(y_bin_true, y_bin_pred, zero_division=0)
    )
    metrics["f1"] = float(
        f1_score(y_bin_true, y_bin_pred, zero_division=0)
    )
    metrics["log_loss"] = float(log_loss(y_bin_true, y_pred_clip))

    # FPR — false positive rate
    fp = int(((y_bin_pred == 1) & (y_bin_true == 0)).sum())
    tn = int(((y_bin_pred == 0) & (y_bin_true == 0)).sum())
    metrics["fpr"] = float(fp / max(fp + tn, 1))

    # AUC — requires both classes present
    if len(np.unique(y_bin_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_bin_true, y_pred))
        metrics["pr_auc"] = float(average_precision_score(y_bin_true, y_pred))
    else:
        metrics["auc_roc"] = float("nan")
        metrics["pr_auc"] = float("nan")

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
        ``"tabular"`` or ``"sequence"``.  For sequence models the model's
        P / n_features attributes are set for reshape.
    P, n_features : int, optional
        Required for sequence models to reshape flat features.

    Returns
    -------
    dict
        ``{"model_name": ..., "val_metrics": {...}, "test_metrics": {...},
           "model": fitted_model, "test_predictions": ndarray}``
    """
    # Prepare sequence models
    if input_type == "sequence" and P is not None and n_features is not None:
        if hasattr(model, "P"):
            model.P = P
            model.n_features = n_features

    # Fit
    try:
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
        "val_predictions": val_pred,
        "test_predictions": test_pred,
    }


def _supports_eval_set(model) -> bool:
    """Check if the model's fit() accepts eval_set (boosting models)."""
    cls_name = type(model).__name__
    return cls_name in ("LGBMRegressor", "XGBRegressor")
