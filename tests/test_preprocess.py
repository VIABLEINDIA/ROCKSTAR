"""Section IV.A/IV.B -- lag construction, the [0:33] split and normalisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.config import PreprocessConfig
from algobot.data.loader import load_synthetic
from algobot.data.preprocess import (
    anchor_column,
    build_dataset,
    build_lag_frame,
    close_only,
    denormalize,
    forecast_horizon,
    latest_features,
    normalize_windows,
    split_xy,
    train_test_split_ordered,
)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return load_synthetic("TEST", days=600)


def test_close_only_keeps_one_column(prices):
    out = close_only(prices)
    assert list(out.columns) == ["Close"]
    assert out.index.name == "Date"
    assert out.index.is_monotonic_increasing


def test_lag_frame_has_41_columns_and_no_nulls(prices):
    lagged = build_lag_frame(prices, n_lags=41)
    assert lagged.shape[1] == 41
    assert not lagged.isna().any().any()
    # Dropping the warm-up rows costs exactly n_lags - 1 observations.
    assert len(lagged) == len(prices) - 40


def test_lag_columns_are_ordered_oldest_first(prices):
    lagged = build_lag_frame(prices, n_lags=5)
    row = lagged.iloc[0]
    # lag_4 is the oldest close, lag_0 the newest -- and lag_0 is that bar.
    assert list(lagged.columns) == ["lag_4", "lag_3", "lag_2", "lag_1", "lag_0"]
    assert row["lag_0"] == pytest.approx(prices["Close"].iloc[4])
    assert row["lag_4"] == pytest.approx(prices["Close"].iloc[0])


def test_split_xy_matches_the_papers_boundary(prices):
    lagged = build_lag_frame(prices, 41)
    X, Y = split_xy(lagged, 33)
    assert X.shape[1] == 33
    assert Y.shape[1] == 8
    assert list(X.columns)[-1] == "lag_8"
    assert list(Y.columns)[0] == "lag_7"


def test_split_xy_rejects_out_of_range(prices):
    lagged = build_lag_frame(prices, 41)
    with pytest.raises(ValueError):
        split_xy(lagged, 41)
    with pytest.raises(ValueError):
        split_xy(lagged, 0)


def test_train_test_split_is_chronological_and_60_40(prices):
    lagged = build_lag_frame(prices, 41)
    X, Y = split_xy(lagged, 33)
    X_tr, X_te, y_tr, y_te = train_test_split_ordered(X, Y, 0.60)

    assert len(X_tr) == int(len(X) * 0.60)
    assert len(X_tr) + len(X_te) == len(X)
    # No shuffling: every training date precedes every test date.
    assert X_tr.index.max() < X_te.index.min()


def test_normalisation_makes_windows_scale_free(prices):
    lagged = build_lag_frame(prices, 41)
    ratios, anchors = normalize_windows(lagged, 33)

    assert np.allclose(ratios.iloc[:, anchor_column(33)], 1.0)
    assert np.allclose(denormalize(ratios.to_numpy(), anchors), lagged.to_numpy())


def test_normalisation_survives_a_price_regime_shift():
    """The whole point: a 10x price level must not move the feature values."""
    base = load_synthetic("SHIFT", days=200)
    scaled = base * 10

    a, _ = normalize_windows(build_lag_frame(base, 41), 33)
    b, _ = normalize_windows(build_lag_frame(scaled, 41), 33)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_build_dataset_shapes_and_metadata(prices):
    cfg = PreprocessConfig()
    ds = build_dataset(prices, cfg)

    assert ds.n_features == 33
    assert ds.n_targets == 8
    assert ds.X_train.shape[0] == len(ds.dates_train)
    assert ds.X_test.shape[0] == len(ds.dates_test)
    assert ds.normalized is True
    assert ds.anchors_test is not None and len(ds.anchors_test) == len(ds.X_test)
    assert ds.summary()["train_rows"] == len(ds.X_train)


def test_build_dataset_raw_mode_reproduces_the_paper(prices):
    ds = build_dataset(prices, PreprocessConfig(normalize=False))
    assert ds.normalized is False
    assert ds.anchors_train is None
    # Raw mode keeps price units.
    assert ds.y_train.max() > 10


def test_latest_features_row_ends_on_the_last_bar(prices):
    cfg = PreprocessConfig()
    row, anchor = latest_features(prices, cfg)

    assert row.shape == (1, cfg.n_input_lags)
    assert anchor == pytest.approx(prices["Close"].iloc[-1])
    # Normalised, so the anchor position is exactly 1.0.
    assert row[0, anchor_column(cfg.n_input_lags)] == pytest.approx(1.0)


def test_latest_features_needs_enough_history():
    tiny = load_synthetic("TINY", days=10)
    with pytest.raises(ValueError):
        latest_features(tiny, PreprocessConfig())


def test_forecast_horizon_is_41_minus_33():
    assert forecast_horizon(PreprocessConfig()) == 8
