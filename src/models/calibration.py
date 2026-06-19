"""Calibration diagnostics kept independent of the classifier implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int = 4) -> float:
    """Mean squared probability error, averaged across the four classes."""

    one_hot = np.eye(n_classes, dtype=float)[np.asarray(y_true, dtype=int)]
    return float(np.mean((one_hot - probabilities) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    """Top-label ECE: confidence calibration of the class the model selected."""

    y_true = np.asarray(y_true, dtype=int)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lower) & ((confidence < upper) if upper < 1 else (confidence <= upper))
        if mask.any():
            accuracy = (predicted[mask] == y_true[mask]).mean()
            ece += mask.mean() * abs(accuracy - confidence[mask].mean())
    return float(ece)


def calibration_curve_data(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """One-vs-rest calibration data for each risk tier."""

    records: list[dict[str, float | int]] = []
    for class_id in range(probabilities.shape[1]):
        observed, predicted = calibration_curve((np.asarray(y_true) == class_id).astype(int), probabilities[:, class_id], n_bins=n_bins)
        records.extend(
            {"class_id": class_id, "mean_predicted_probability": float(p), "fraction_positive": float(o)}
            for p, o in zip(predicted, observed)
        )
    return pd.DataFrame(records)
