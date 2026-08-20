"""Section IV.A / IV.B -- Data pre-processing and the 60:40 split.

The paper's recipe:
  a. drop every column except Date and Close;
  b. lag the data by one day and build 41 lag columns;
  c. drop NULL rows;
  d. "Dataset is split as [0:33] data into X (inputs) and the rest into Y".

The paper does not say which end of the lag block column 0 sits at. We order
the lag matrix oldest -> newest, so columns [0:33] are the 33 oldest closes of
each 41-day window (the input history) and the remaining 8 are the most recent
closes (the values to predict). That is the only reading of "[0:33] -> X, rest
-> Y" which is causally valid -- the reverse would train the model to predict
the past from the future. `n_input_lags` and `n_lags` stay configurable so the
literal column split can still be reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import PreprocessConfig


@dataclass
class Dataset:
    """A time-ordered supervised dataset plus the metadata plots/reports need."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    dates_train: pd.DatetimeIndex
    dates_test: pd.DatetimeIndex
    feature_names: list[str]
    target_names: list[str]
    frame: pd.DataFrame
    anchors_train: np.ndarray | None = None
    anchors_test: np.ndarray | None = None
    normalized: bool = False

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_targets(self) -> int:
        return 1 if self.y_train.ndim == 1 else self.y_train.shape[1]

    def summary(self) -> dict:
        return {
            "rows": len(self.frame),
            "train_rows": len(self.X_train),
            "test_rows": len(self.X_test),
            "n_features": self.n_features,
            "n_targets": self.n_targets,
            "train_start": str(self.dates_train[0].date()) if len(self.dates_train) else None,
            "train_end": str(self.dates_train[-1].date()) if len(self.dates_train) else None,
            "test_start": str(self.dates_test[0].date()) if len(self.dates_test) else None,
            "test_end": str(self.dates_test[-1].date()) if len(self.dates_test) else None,
            "normalized": self.normalized,
        }


def close_only(df: pd.DataFrame) -> pd.DataFrame:
    """Step (a): keep Date (the index) and Close."""
    if "Close" not in df.columns:
        raise ValueError("Price frame has no Close column")
    out = df[["Close"]].copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    return out.sort_index()


def build_lag_frame(df: pd.DataFrame, n_lags: int = 41) -> pd.DataFrame:
    """Steps (b) and (c): 41 lag columns, oldest first, NULL rows dropped.

    Row t holds ``[C(t-n_lags+1) ... C(t-1), C(t)]``; naming keeps the lag
    distance explicit so feature importances stay readable.
    """
    if n_lags < 2:
        raise ValueError("n_lags must be >= 2")

    close = close_only(df)["Close"]
    cols = {}
    for i in range(n_lags - 1, -1, -1):          # oldest lag first
        cols[f"lag_{i}"] = close.shift(i)
    lagged = pd.DataFrame(cols, index=close.index)
    return lagged.dropna()


def split_xy(lagged: pd.DataFrame, n_input_lags: int = 33):
    """Step (d): columns [0:n_input_lags] -> X, the remainder -> Y."""
    if not 0 < n_input_lags < lagged.shape[1]:
        raise ValueError(
            f"n_input_lags must be in (0, {lagged.shape[1]}), got {n_input_lags}"
        )
    X = lagged.iloc[:, :n_input_lags]
    Y = lagged.iloc[:, n_input_lags:]
    return X, Y


def anchor_column(n_input_lags: int) -> int:
    """Index of the window's anchor: the last input close (lag 0 of the input)."""
    return n_input_lags - 1


def normalize_windows(lagged: pd.DataFrame, n_input_lags: int):
    """Divide every window by its anchor close, returning (ratios, anchors).

    Each row becomes a scale-free shape: inputs are ratios to the most recent
    known close (the anchor is exactly 1.0), and targets are the future closes
    expressed as multiples of it. This is what lets the forest generalise to
    price levels it never saw in training.
    """
    anchors = lagged.iloc[:, anchor_column(n_input_lags)].to_numpy(dtype=float)
    if (anchors <= 0).any():
        raise ValueError("Anchor closes must be positive to normalise windows")
    ratios = lagged.div(pd.Series(anchors, index=lagged.index), axis=0)
    return ratios, anchors


def denormalize(pred: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Turn predicted ratios back into prices."""
    pred = np.asarray(pred, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    if pred.ndim == 1:
        return pred * anchors
    return pred * anchors.reshape(-1, 1)


def train_test_split_ordered(X: pd.DataFrame, Y: pd.DataFrame, train_ratio: float = 0.60):
    """Chronological 60:40 split -- never shuffled, to avoid look-ahead leakage."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    cut = int(len(X) * train_ratio)
    if cut < 1 or cut >= len(X):
        raise ValueError(f"Not enough rows ({len(X)}) for a {train_ratio:.0%} split")
    return X.iloc[:cut], X.iloc[cut:], Y.iloc[:cut], Y.iloc[cut:]


def build_dataset(df: pd.DataFrame, cfg: PreprocessConfig | None = None) -> Dataset:
    """Run the full Section IV.A/IV.B pipeline on an OHLCV frame."""
    cfg = cfg or PreprocessConfig()

    lagged = build_lag_frame(df, cfg.n_lags)

    anchors = None
    working = lagged
    if cfg.normalize:
        working, anchors = normalize_windows(lagged, cfg.n_input_lags)

    X, Y = split_xy(working, cfg.n_input_lags)
    X_train, X_test, y_train, y_test = train_test_split_ordered(X, Y, cfg.train_ratio)

    cut = len(X_train)
    anchors_train = anchors[:cut] if anchors is not None else None
    anchors_test = anchors[cut:] if anchors is not None else None

    y_tr = y_train.to_numpy()
    y_te = y_test.to_numpy()
    if y_tr.shape[1] == 1:                      # single-output convenience
        y_tr, y_te = y_tr.ravel(), y_te.ravel()

    return Dataset(
        X_train=X_train.to_numpy(),
        X_test=X_test.to_numpy(),
        y_train=y_tr,
        y_test=y_te,
        dates_train=X_train.index,
        dates_test=X_test.index,
        feature_names=list(X.columns),
        target_names=list(Y.columns),
        frame=lagged,
        anchors_train=anchors_train,
        anchors_test=anchors_test,
        normalized=cfg.normalize,
    )


def latest_features(df: pd.DataFrame, cfg: PreprocessConfig | None = None):
    """The live feature row plus its anchor close.

    Returns ``(row, anchor)`` where `row` is shaped ``(1, n_input_lags)`` and
    already normalised when `cfg.normalize` is set.

    The window is the most recent `n_input_lags` closes, oldest first.

    Training pairs a window of `n_input_lags` consecutive closes with the
    `n_lags - n_input_lags` closes that immediately follow it, so feeding the
    window that ends on today's bar makes the model's outputs a genuine
    forward forecast (index 0 = next bar, index -1 = `horizon` bars ahead).
    Reusing the last row of the lag frame instead would predict closes that
    are already known.
    """
    cfg = cfg or PreprocessConfig()
    close = close_only(df)["Close"]
    if len(close) < cfg.n_input_lags:
        raise ValueError(
            f"Need at least {cfg.n_input_lags} closes to build a feature row, got {len(close)}"
        )
    row = close.iloc[-cfg.n_input_lags:].to_numpy(dtype=float).reshape(1, -1)
    anchor = float(row[0, anchor_column(cfg.n_input_lags)])
    if cfg.normalize:
        if anchor <= 0:
            raise ValueError("Anchor close must be positive to normalise the feature row")
        row = row / anchor
    return row, anchor


def forecast_horizon(cfg: PreprocessConfig | None = None) -> int:
    """How many bars ahead the model predicts (41 - 33 = 8 in the paper)."""
    cfg = cfg or PreprocessConfig()
    return cfg.n_lags - cfg.n_input_lags
