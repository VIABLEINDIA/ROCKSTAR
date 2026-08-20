"""Section IV.F -- integration of the financial strategy with the RF model.

"the Bot is made to take its decision on the basis of prediction from the model
as well as the financial strategy."

`ModelFilteredStrategy` wraps any base strategy and lets the Random Forest
veto or confirm its entries: a long is taken only when the model's forecast
return over the next `steps` bars clears `min_expected_return`, and an open
long is closed early when the forecast turns down by `exit_threshold`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.preprocess import (PreprocessConfig, anchor_column, build_lag_frame,
                               denormalize)
from .base import EXIT, FLAT, LONG, Strategy

log = logging.getLogger(__name__)


@dataclass
class ModelFilteredStrategy(Strategy):
    name: str = "ml"
    base: Strategy | None = None
    model: object | None = None            # a TrainedModel
    steps: int = 1                         # bars ahead to read from the forecast
    min_expected_return: float = 0.0
    exit_threshold: float = -0.01
    veto_only: bool = True                 # True: filter entries; False: model may also enter
    require_forecast: bool = True
    """Suppress entries on bars where the model cannot produce a forecast.

    The first `n_input_lags - 1` bars of any frame have too little history for
    a prediction. Letting the base strategy trade them unchecked would mean the
    'model-filtered' strategy silently trades unfiltered exactly where the
    model is blind, so by default those entries are dropped instead.
    """

    def __post_init__(self):
        if self.base is None:
            raise ValueError("ModelFilteredStrategy needs a base strategy")

    @property
    def warmup(self) -> int:
        pre = getattr(self.model, "preprocess", None) or PreprocessConfig()
        return max(self.base.warmup, pre.n_input_lags)

    def params(self) -> dict:
        return {
            "base": self.base.describe(),
            "steps": self.steps,
            "min_expected_return": self.min_expected_return,
            "exit_threshold": self.exit_threshold,
            "veto_only": self.veto_only,
            "require_forecast": self.require_forecast,
        }

    def expected_returns(self, df: pd.DataFrame) -> pd.Series:
        """Vectorised forecast return for every bar in `df`.

        Each row of the lag matrix is one 33-close window, so the whole test
        period is predicted in a single `predict` call rather than bar by bar.
        """
        if self.model is None:
            return pd.Series(np.nan, index=df.index)

        pre = getattr(self.model, "preprocess", None) or PreprocessConfig()

        # Windows of n_input_lags closes ending on each bar -> forecast of the
        # following bars. build_lag_frame gives windows of n_lags; take the
        # trailing n_input_lags columns so the window ends on the current bar.
        lagged = build_lag_frame(df[["Close"]], pre.n_input_lags)
        if lagged.empty:
            return pd.Series(np.nan, index=df.index)

        # The anchor is the last close in each window, i.e. the bar the
        # forecast is made from -- identical to the training-time anchor.
        anchors = lagged.iloc[:, anchor_column(pre.n_input_lags)].to_numpy(dtype=float)

        features = lagged
        if pre.normalize:
            if (anchors <= 0).any():
                return pd.Series(np.nan, index=df.index)
            features = lagged.div(pd.Series(anchors, index=lagged.index), axis=0)

        preds = np.atleast_2d(self.model.predict(features.to_numpy()))
        if preds.shape[0] != len(lagged):
            preds = preds.reshape(len(lagged), -1)
        if pre.normalize:
            preds = denormalize(preds, anchors)

        step = max(1, min(self.steps, preds.shape[1]))
        target = preds[:, step - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            exp_ret = np.where(anchors > 0, target / anchors - 1.0, np.nan)
        return pd.Series(exp_ret, index=lagged.index).reindex(df.index)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        base_signals = self.base.generate_signals(df)
        exp_ret = self.expected_returns(df)

        sig = base_signals.copy()
        known = exp_ret.notna()

        # Veto longs the model does not back.
        blocked = known & (base_signals == LONG) & (exp_ret < self.min_expected_return)
        if self.require_forecast:
            blocked = blocked | (~known & (base_signals == LONG))
        sig[blocked] = FLAT

        # Model-driven early exit.
        bail = known & (exp_ret <= self.exit_threshold)
        sig[bail] = EXIT

        if not self.veto_only:
            # Model may also open a long when the base strategy is merely quiet.
            confirm = known & (base_signals == FLAT) & (exp_ret > self.min_expected_return)
            sig[confirm] = LONG

        n_blocked = int(blocked.sum())
        if n_blocked:
            log.info("Model vetoed %d of %d base entries",
                     n_blocked, int((base_signals == LONG).sum()))
        return sig.astype(int)
