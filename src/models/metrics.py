"""Regression metrics used by model training scripts."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and R2 for regression predictions."""
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if truth.size == 0:
        raise ValueError("Cannot compute metrics for empty arrays.")

    errors = truth - pred
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    denom = float(np.sum((truth - np.mean(truth)) ** 2))
    r2 = 0.0 if denom == 0 else float(1 - np.sum(errors**2) / denom)
    return {"rmse": rmse, "mae": mae, "r2": r2}
