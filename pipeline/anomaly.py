"""
pipeline.anomaly
================
Threshold calibration and anomaly-detection evaluation for VP scores.

The pipeline predicts a continuous VP ∈ [0, 1].  To *operationally* flag
anomalies (extreme precipitation events) we need:

1. **Calibrate** an optimal threshold on the **validation** set.
2. **Apply** the threshold to the **test** set.
3. **Evaluate** with anomaly-specific metrics (Precision, Recall, F1,
   PR-AUC, FPR, confusion matrix, seasonal breakdown).

Critical rule
-------------
The threshold is **never** calibrated on the test set.  This module
enforces that contract by requiring explicit val/test arrays.

Public API
----------
``calibrate_threshold(y_val_true, y_val_pred, ...)  → (threshold, val_metrics)``
``classify_anomalies(y_pred, threshold)              → binary array``
``compute_anomaly_report(y_true, y_pred, threshold, ...)  → report dict``
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ======================================================================== #
#  Threshold calibration (validation set only)                              #
# ======================================================================== #

def calibrate_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "f1",
    grid_resolution: int = 200,
    anomaly_vp_cutoff: float = 0.5,
) -> tuple[float, Dict[str, float]]:
    """
    Find the optimal prediction threshold for anomaly detection.

    **Must be called with validation data only — never test.**

    Parameters
    ----------
    y_true : array
        Ground-truth VP in [0, 1].
    y_pred : array
        Predicted VP in [0, 1].
    method : str
        ``"f1"``    — maximise F1-score for the anomaly class.
        ``"youden"`` — maximise Youden's J (sensitivity + specificity − 1).
        ``"pr_auc"`` — choose the threshold that maximises F1 on the
                       precision–recall curve.
    grid_resolution : int
        Number of candidate thresholds to evaluate.
    anomaly_vp_cutoff : float
        VP value above which the ground truth is considered anomalous
        (default 0.5 ↔ precipitation ≥ P95).

    Returns
    -------
    best_threshold : float
    metrics_at_threshold : dict
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)
    y_bin = (y_true > anomaly_vp_cutoff).astype(int)

    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        # Degenerate — only one class present
        return 0.5, _compute_anomaly_metrics(
            y_bin, (y_pred > 0.5).astype(int), y_pred, 0.5,
        )

    best_t = 0.5

    if method == "f1":
        thresholds = np.linspace(0.01, 0.99, grid_resolution)
        best_score = -1.0
        for t in thresholds:
            score = f1_score(y_bin, (y_pred > t).astype(int), zero_division=0)
            if score > best_score:
                best_score = score
                best_t = float(t)

    elif method == "youden":
        thresholds = np.linspace(0.01, 0.99, grid_resolution)
        best_j = -2.0
        for t in thresholds:
            cm = confusion_matrix(y_bin, (y_pred > t).astype(int), labels=[0, 1])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                sens = tp / max(tp + fn, 1)
                spec = tn / max(tn + fp, 1)
                j = sens + spec - 1
                if j > best_j:
                    best_j = j
                    best_t = float(t)

    elif method == "pr_auc":
        prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_bin, y_pred)
        f1_arr = (
            2 * prec_arr[:-1] * rec_arr[:-1]
            / np.maximum(prec_arr[:-1] + rec_arr[:-1], 1e-8)
        )
        best_idx = int(np.argmax(f1_arr))
        best_t = float(pr_thresh[best_idx])

    else:
        raise ValueError(
            f"Unknown calibration method '{method}'. "
            f"Choose from: f1, youden, pr_auc."
        )

    y_pred_bin = (y_pred > best_t).astype(int)
    metrics = _compute_anomaly_metrics(y_bin, y_pred_bin, y_pred, best_t)
    return best_t, metrics


# ======================================================================== #
#  Classification helper                                                    #
# ======================================================================== #

def classify_anomalies(
    y_pred: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Binary anomaly labels from continuous VP predictions."""
    return (np.asarray(y_pred) > threshold).astype(int)


# ======================================================================== #
#  Comprehensive anomaly report                                             #
# ======================================================================== #

def compute_anomaly_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    times: Optional[np.ndarray] = None,
    anomaly_vp_cutoff: float = 0.5,
) -> Dict[str, Any]:
    """
    Full anomaly-detection evaluation report.

    Parameters
    ----------
    y_true, y_pred : arrays
        Ground-truth and predicted VP scores ∈ [0, 1].
    threshold : float
        Calibrated decision threshold.
    times : array, optional
        Timestamps — enables seasonal breakdown.
    anomaly_vp_cutoff : float
        VP above which ground truth counts as anomalous (default 0.5).

    Returns
    -------
    dict with sections:
        ``"overall"``          — precision, recall, F1, FPR, PR-AUC, …
        ``"confusion_matrix"`` — TP / FP / TN / FN
        ``"by_intensity"``     — error by VP intensity band
        ``"by_season"``        — metrics per meteorological season (if *times*)
        ``"residuals"``        — mean, std, skew, kurtosis, quantiles
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)
    y_bin = (y_true > anomaly_vp_cutoff).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)

    report: Dict[str, Any] = {}

    # ---- Overall metrics ----
    report["overall"] = _compute_anomaly_metrics(
        y_bin, y_pred_bin, y_pred, threshold,
    )

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_bin, y_pred_bin, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    report["confusion_matrix"] = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    # ---- Error by VP intensity band ----
    intensity_bands = [
        ("below_p90",  y_true <= 0.0),
        ("p90_to_p95", (y_true > 0.0) & (y_true <= 0.5)),
        ("p95_to_p100", (y_true > 0.5) & (y_true < 1.0)),
        ("above_p100", y_true >= 1.0),
    ]
    report["by_intensity"] = {}
    for name, mask in intensity_bands:
        n_in_band = int(mask.sum())
        if n_in_band > 0:
            errs = y_true[mask] - y_pred[mask]
            report["by_intensity"][name] = {
                "count": n_in_band,
                "mean_pred": float(y_pred[mask].mean()),
                "mae": float(np.abs(errs).mean()),
                "rmse": float(np.sqrt(np.mean(errs ** 2))),
                "bias": float(errs.mean()),
            }

    # ---- Seasonal breakdown ----
    if times is not None:
        times_dt = pd.to_datetime(times)
        months = times_dt.month
        # Southern-hemisphere meteorological seasons
        seasons = {
            "DJF_wet":  np.isin(months, [12, 1, 2]),
            "MAM_trans": np.isin(months, [3, 4, 5]),
            "JJA_dry":  np.isin(months, [6, 7, 8]),
            "SON_trans": np.isin(months, [9, 10, 11]),
        }
        report["by_season"] = {}
        for sname, smask in seasons.items():
            n_s = int(smask.sum())
            if n_s > 0:
                s_bin = y_bin[smask]
                s_pred = y_pred[smask]
                s_pred_bin = (s_pred > threshold).astype(int)
                report["by_season"][sname] = {
                    "count": n_s,
                    "anomaly_rate": float(s_bin.mean()),
                    "precision": float(precision_score(s_bin, s_pred_bin, zero_division=0)),
                    "recall": float(recall_score(s_bin, s_pred_bin, zero_division=0)),
                    "f1": float(f1_score(s_bin, s_pred_bin, zero_division=0)),
                }

    # ---- Residual analysis ----
    residuals = y_true - y_pred
    report["residuals"] = {
        "mean": float(residuals.mean()),
        "std": float(residuals.std()),
        "skewness": float(pd.Series(residuals).skew()),
        "kurtosis": float(pd.Series(residuals).kurtosis()),
        "q05": float(np.percentile(residuals, 5)),
        "q25": float(np.percentile(residuals, 25)),
        "q50": float(np.percentile(residuals, 50)),
        "q75": float(np.percentile(residuals, 75)),
        "q95": float(np.percentile(residuals, 95)),
    }

    return report


# ======================================================================== #
#  Internal helpers                                                         #
# ======================================================================== #

def _compute_anomaly_metrics(
    y_bin: np.ndarray,
    y_pred_bin: np.ndarray,
    y_pred_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Full anomaly metrics dict."""
    metrics: Dict[str, float] = {"threshold": threshold}

    cm = confusion_matrix(y_bin, y_pred_bin, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    metrics["tp"] = float(tp)
    metrics["fp"] = float(fp)
    metrics["tn"] = float(tn)
    metrics["fn"] = float(fn)

    metrics["precision"] = float(
        precision_score(y_bin, y_pred_bin, zero_division=0)
    )
    metrics["recall"] = float(
        recall_score(y_bin, y_pred_bin, zero_division=0)
    )
    metrics["f1"] = float(
        f1_score(y_bin, y_pred_bin, zero_division=0)
    )
    metrics["fpr"] = float(fp / max(fp + tn, 1))
    metrics["fnr"] = float(fn / max(fn + tp, 1))

    # AUC-based metrics (need both classes)
    if len(np.unique(y_bin)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_bin, y_pred_score))
        metrics["pr_auc"] = float(average_precision_score(y_bin, y_pred_score))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics
