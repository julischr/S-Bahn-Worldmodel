"""Evaluation metrics and simple stability checks."""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def is_stable(eigenvalues, threshold: float = 1.0) -> bool:
    eigenvalues_arr = np.asarray(eigenvalues)
    return bool(np.all(np.abs(eigenvalues_arr) < threshold))
