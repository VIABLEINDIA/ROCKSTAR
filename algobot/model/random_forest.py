"""Sections IV.C-IV.E -- train, inspect, persist and apply the Random Forest.

The trained regressor is saved as a joblib bundle so the live bot can load it
exactly as described in Section IV.F ("The Random Forest model is integrated as
a joblib file with the bot").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..config import MODEL_DIR, ModelConfig, PreprocessConfig
from ..data.preprocess import Dataset, denormalize, forecast_horizon, latest_features
from .evaluate import Metrics, evaluate

log = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    """A fitted regressor plus everything needed to reuse it later."""

    regressor: RandomForestRegressor
    metrics: Metrics
    feature_names: list[str]
    target_names: list[str]
    symbol: str
    preprocess: PreprocessConfig
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data_summary: dict = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return forecast_horizon(self.preprocess)

    def feature_importances(self) -> pd.Series:
        """Section IV.C: `regressor.feature_importances_`, most important first."""
        return pd.Series(
            self.regressor.feature_importances_, index=self.feature_names, name="importance"
        ).sort_values(ascending=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.regressor.predict(X)

    def predict_next(self, prices: pd.DataFrame) -> np.ndarray:
        """Forecast the next `horizon` closes, in price units."""
        row, anchor = latest_features(prices, self.preprocess)
        pred = np.atleast_2d(self.regressor.predict(row))
        if self.preprocess.normalize:
            pred = denormalize(pred, np.array([anchor]))
        return pred[0]

    def expected_return(self, prices: pd.DataFrame, steps: int = 1) -> float:
        """Fractional return the model expects over the next `steps` bars."""
        last_close = float(prices["Close"].iloc[-1])
        if last_close <= 0:
            return 0.0
        pred = self.predict_next(prices)
        steps = max(1, min(steps, len(pred)))
        return float(pred[steps - 1] / last_close - 1.0)

    def predict_prices(self, X: np.ndarray, anchors: np.ndarray | None = None) -> np.ndarray:
        """Predict and convert back to price units when the model is normalised."""
        pred = np.atleast_2d(self.regressor.predict(X))
        if self.preprocess.normalize:
            if anchors is None:
                raise ValueError("A normalised model needs anchors to return prices")
            pred = denormalize(pred, np.asarray(anchors, dtype=float))
        return pred

    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path else MODEL_DIR / f"{self.symbol.upper()}_rf.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "regressor": self.regressor,
                "metrics": self.metrics.to_dict(),
                "feature_names": self.feature_names,
                "target_names": self.target_names,
                "symbol": self.symbol,
                "preprocess": self.preprocess,
                "trained_at": self.trained_at,
                "data_summary": self.data_summary,
            },
            path,
        )
        log.info("Saved model bundle to %s", path)
        return path

    @staticmethod
    def load(path: str | Path) -> "TrainedModel":
        bundle = joblib.load(Path(path))
        return TrainedModel(
            regressor=bundle["regressor"],
            metrics=Metrics(**bundle["metrics"]),
            feature_names=bundle["feature_names"],
            target_names=bundle["target_names"],
            symbol=bundle["symbol"],
            preprocess=bundle.get("preprocess", PreprocessConfig()),
            trained_at=bundle.get("trained_at", ""),
            data_summary=bundle.get("data_summary", {}),
        )


def default_model_path(symbol: str) -> Path:
    return MODEL_DIR / f"{symbol.upper()}_rf.joblib"


def train(dataset: Dataset, symbol: str, cfg: ModelConfig | None = None,
          preprocess: PreprocessConfig | None = None) -> TrainedModel:
    """Fit a RandomForestRegressor on the training split and score the test split."""
    cfg = cfg or ModelConfig()
    preprocess = preprocess or PreprocessConfig()

    regressor = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    log.info("Fitting RandomForestRegressor on %d rows x %d features",
             *dataset.X_train.shape)
    regressor.fit(dataset.X_train, dataset.y_train)

    metrics = evaluate(regressor, dataset.X_test, dataset.y_test)
    return TrainedModel(
        regressor=regressor,
        metrics=metrics,
        feature_names=dataset.feature_names,
        target_names=dataset.target_names,
        symbol=symbol.upper(),
        preprocess=preprocess,
        data_summary=dataset.summary(),
    )


def predict_frame(model: TrainedModel, dataset: Dataset) -> pd.DataFrame:
    """Actual vs predicted on the test split, in price units, for Figure 16.

    With multi-output targets the last column (the most distant close in each
    predicted block) is the series that lines up with the plot's date axis.
    Normalised models are converted back to prices via the stored anchors, so
    the chart always shows rupees rather than ratios.
    """
    y_pred = np.atleast_2d(model.predict(dataset.X_test))
    y_true = np.atleast_2d(dataset.y_test)
    if y_true.shape[0] == 1 and y_true.shape[1] == len(dataset.dates_test):
        y_true, y_pred = y_true.T, y_pred.T

    if dataset.normalized and dataset.anchors_test is not None:
        y_true = denormalize(y_true, dataset.anchors_test)
        y_pred = denormalize(y_pred, dataset.anchors_test)

    return pd.DataFrame(
        {"actual": y_true[:, -1], "predicted": y_pred[:, -1]}, index=dataset.dates_test
    )
