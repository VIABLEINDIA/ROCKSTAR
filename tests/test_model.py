"""Random Forest training, persistence, metrics and the model/strategy fusion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.config import ModelConfig, PreprocessConfig
from algobot.data.loader import load_synthetic
from algobot.data.preprocess import build_dataset
from algobot.model.evaluate import Metrics, directional_accuracy, evaluate
from algobot.model.random_forest import TrainedModel, predict_frame, train
from algobot.strategies.base import EXIT, FLAT, LONG
from algobot.strategies.ml_filter import ModelFilteredStrategy
from algobot.strategies.registry import build_strategy, wrap_with_model

FAST_MODEL = ModelConfig(n_estimators=20, random_state=0, n_jobs=1)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return load_synthetic("MODEL", days=500)


@pytest.fixture(scope="module")
def dataset(prices):
    return build_dataset(prices, PreprocessConfig())


@pytest.fixture(scope="module")
def model(dataset) -> TrainedModel:
    return train(dataset, "MODEL", FAST_MODEL, PreprocessConfig())


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------
def test_train_produces_a_fitted_multi_output_regressor(model, dataset):
    pred = model.predict(dataset.X_test)
    assert pred.shape == (len(dataset.X_test), dataset.n_targets)


def test_feature_importances_are_named_and_normalised(model):
    importances = model.feature_importances()
    assert len(importances) == 33
    assert set(importances.index) == set(model.feature_names)
    assert importances.sum() == pytest.approx(1.0)
    assert importances.is_monotonic_decreasing


def test_metrics_are_populated(model):
    m = model.metrics
    assert isinstance(m, Metrics)
    for key, value in m.to_dict().items():
        assert isinstance(value, float), key
    assert "Explained Variance Score" in m.as_table()


def test_evaluate_is_perfect_for_an_oracle(dataset):
    class Oracle:
        def predict(self, X):
            return dataset.y_test

        def score(self, X, y):
            return 1.0

    m = evaluate(Oracle(), dataset.X_test, dataset.y_test)
    assert m.r2 == pytest.approx(1.0)
    assert m.explained_variance == pytest.approx(1.0)
    assert m.mae == pytest.approx(0.0)


def test_directional_accuracy_bounds():
    up = np.array([[1.0, 2.0, 3.0]])
    assert directional_accuracy(up, up) == pytest.approx(1.0)
    assert directional_accuracy(up, np.array([[3.0, 2.0, 1.0]])) == pytest.approx(0.0)


# ----------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------
def test_save_and_load_round_trip(model, dataset, tmp_path):
    path = model.save(tmp_path / "m.joblib")
    assert path.exists()

    loaded = TrainedModel.load(path)
    assert loaded.symbol == model.symbol
    assert loaded.feature_names == model.feature_names
    assert loaded.preprocess.n_lags == model.preprocess.n_lags
    assert loaded.metrics.to_dict() == model.metrics.to_dict()
    assert np.allclose(loaded.predict(dataset.X_test), model.predict(dataset.X_test))


def test_horizon_matches_the_config(model):
    assert model.horizon == 8


# ----------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------
def test_predict_next_returns_prices_not_ratios(model, prices):
    forecast = model.predict_next(prices)
    last_close = float(prices["Close"].iloc[-1])

    assert len(forecast) == model.horizon
    # A normalised model must be de-normalised back into price units.
    assert 0.5 * last_close < forecast[0] < 1.5 * last_close


def test_expected_return_is_a_small_fraction(model, prices):
    exp_ret = model.expected_return(prices, steps=1)
    assert -0.5 < exp_ret < 0.5


def test_predict_frame_is_in_price_units(model, dataset, prices):
    frame = predict_frame(model, dataset)
    assert list(frame.columns) == ["actual", "predicted"]
    assert len(frame) == len(dataset.X_test)
    assert frame["actual"].min() > 1.0        # prices, not ratios around 1.0
    assert frame.index.equals(dataset.dates_test)


def test_raw_price_model_also_round_trips(prices, tmp_path):
    cfg = PreprocessConfig(normalize=False)
    ds = build_dataset(prices, cfg)
    m = train(ds, "RAW", FAST_MODEL, cfg)

    forecast = m.predict_next(prices)
    assert len(forecast) == m.horizon
    assert TrainedModel.load(m.save(tmp_path / "raw.joblib")).preprocess.normalize is False


# ----------------------------------------------------------------------
# Section IV.F fusion
# ----------------------------------------------------------------------
def test_wrap_with_model_builds_a_filtered_strategy(model):
    fused = wrap_with_model(build_strategy("ma"), model)
    assert isinstance(fused, ModelFilteredStrategy)
    assert fused.base.name == "ma"
    assert "ma(" in fused.describe()


def test_ml_filter_requires_a_base_strategy(model):
    with pytest.raises(ValueError):
        ModelFilteredStrategy(model=model)


def test_expected_returns_are_aligned_and_finite(model, prices):
    fused = wrap_with_model(build_strategy("ma"), model)
    exp = fused.expected_returns(prices)

    assert exp.index.equals(prices.index)
    valid = exp.dropna()
    assert len(valid) > 0
    assert valid.abs().max() < 1.0


def test_veto_only_never_creates_new_entries(model, prices):
    base = build_strategy("ma")
    base_sig = base.generate_signals(prices)
    fused_sig = wrap_with_model(base, model, veto_only=True).generate_signals(prices)

    new_longs = (fused_sig == LONG) & (base_sig != LONG)
    assert not new_longs.any()


def test_impossible_threshold_blocks_every_entry(model, prices):
    fused = wrap_with_model(build_strategy("ma"), model, min_expected_return=10.0)
    assert not (fused.generate_signals(prices) == LONG).any()


def test_signals_stay_in_range(model, prices):
    sig = wrap_with_model(build_strategy("multiple"), model, veto_only=False).generate_signals(prices)
    assert set(np.unique(sig)) <= {EXIT, FLAT, LONG}
