"""Section V -- Evaluation metrics for the Random Forest Regressor."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)


@dataclass
class Metrics:
    """The four metrics named in the paper, plus MAE/RMSE for context."""

    explained_variance: float
    r2: float
    mean_squared_log_error: float
    regressor_score: float
    mae: float
    rmse: float
    directional_accuracy: float

    def to_dict(self) -> dict:
        return {k: round(float(v), 6) for k, v in asdict(self).items()}

    def as_table(self) -> str:
        labels = {
            "explained_variance": "Explained Variance Score",
            "r2": "R^2 Score",
            "mean_squared_log_error": "Mean Squared Log Error",
            "regressor_score": "Random Forest Regressor Score",
            "mae": "Mean Absolute Error",
            "rmse": "Root Mean Squared Error",
            "directional_accuracy": "Directional Accuracy",
        }
        rows = [f"{labels[k]:<32} {v:>12.6f}" for k, v in self.to_dict().items()]
        head = f"{'METRIC':<32} {'VALUE':>12}"
        return "\n".join([head, "-" * len(head), *rows])


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, last_known: np.ndarray | None = None) -> float:
    """Share of steps where predicted and actual moves share a sign.

    For multi-output targets the move is measured step-to-step inside each
    predicted block; `last_known` (the final input close) anchors the first step.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y_true.ndim == 1:
        if len(y_true) < 2:
            return float("nan")
        return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))

    if last_known is not None:
        anchor = np.asarray(last_known, float).reshape(-1, 1)
        y_true = np.hstack([anchor, y_true])
        y_pred = np.hstack([anchor, y_pred])
    if y_true.shape[1] < 2:
        return float("nan")
    return float(np.mean(np.sign(np.diff(y_true, axis=1)) == np.sign(np.diff(y_pred, axis=1))))


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> Metrics:
    """Compute every Section V metric for a fitted regressor."""
    y_pred = model.predict(X_test)
    y_true = np.asarray(y_test, float)
    y_pred = np.asarray(y_pred, float)

    # MSLE is only defined for non-negative values; prices qualify, but guard
    # against a pathological negative prediction rather than crashing a run.
    if (y_true < 0).any() or (y_pred < 0).any():
        msle = float("nan")
    else:
        msle = float(mean_squared_log_error(y_true, y_pred))

    last_known = X_test[:, -1] if X_test.ndim == 2 and y_true.ndim == 2 else None

    return Metrics(
        explained_variance=float(explained_variance_score(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
        mean_squared_log_error=msle,
        regressor_score=float(model.score(X_test, y_test)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        directional_accuracy=directional_accuracy(y_true, y_pred, last_known),
    )
