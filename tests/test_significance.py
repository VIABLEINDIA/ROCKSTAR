"""Permutation test: model filter vs random vetoes of the same size."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.backtest.significance import RandomVetoStrategy, format_report, permutation_test
from algobot.config import BacktestConfig, ModelConfig, PreprocessConfig
from algobot.data.loader import load_synthetic
from algobot.data.preprocess import build_dataset
from algobot.model.random_forest import train
from algobot.strategies.base import LONG
from algobot.strategies.registry import build_strategy


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return load_synthetic("SIG", days=500)


@pytest.fixture(scope="module")
def model(prices):
    cfg = PreprocessConfig()
    return train(build_dataset(prices, cfg), "SIG", ModelConfig(n_estimators=15, n_jobs=1), cfg)


# ----------------------------------------------------------------------
# RandomVetoStrategy
# ----------------------------------------------------------------------
def test_random_veto_removes_exactly_n_entries(prices):
    base = build_strategy("donchian")
    n_entries = int((base.generate_signals(prices) == LONG).sum())
    assert n_entries > 5

    vetoed = RandomVetoStrategy(build_strategy("donchian"), n_vetoes=5, seed=1)
    n_after = int((vetoed.generate_signals(prices) == LONG).sum())
    assert n_after == n_entries - 5


def test_random_veto_is_deterministic_per_seed(prices):
    a = RandomVetoStrategy(build_strategy("donchian"), 5, seed=7).generate_signals(prices)
    b = RandomVetoStrategy(build_strategy("donchian"), 5, seed=7).generate_signals(prices)
    pd.testing.assert_series_equal(a, b)


def test_different_seeds_veto_differently(prices):
    a = RandomVetoStrategy(build_strategy("donchian"), 10, seed=1).generate_signals(prices)
    b = RandomVetoStrategy(build_strategy("donchian"), 10, seed=2).generate_signals(prices)
    assert not a.equals(b)


def test_zero_vetoes_is_a_no_op(prices):
    base = build_strategy("ma")
    pd.testing.assert_series_equal(
        base.generate_signals(prices),
        RandomVetoStrategy(build_strategy("ma"), 0, seed=0).generate_signals(prices),
        check_names=False,
    )


def test_veto_never_invents_entries(prices):
    base_sig = build_strategy("multiple").generate_signals(prices)
    vetoed = RandomVetoStrategy(build_strategy("multiple"), 20, seed=3).generate_signals(prices)
    assert not ((vetoed == LONG) & (base_sig != LONG)).any()


def test_more_vetoes_than_entries_is_safe(prices):
    vetoed = RandomVetoStrategy(build_strategy("gold"), 10_000, seed=0)
    assert (vetoed.generate_signals(prices) == LONG).sum() == 0


# ----------------------------------------------------------------------
# permutation_test
# ----------------------------------------------------------------------
def test_permutation_report_shape(prices, model):
    row = permutation_test(prices, "ma", model, BacktestConfig(), n_trials=12, symbol="SIG")

    for key in ("strategy", "entries", "vetoed", "profit_base", "profit_model",
                "null_mean", "percentile", "p_value", "skilled"):
        assert key in row
    assert 0 <= row["percentile"] <= 100
    assert 0 <= row["p_value"] <= 1
    assert row["vetoed"] <= row["entries"]


def test_p_value_and_percentile_are_consistent(prices, model):
    row = permutation_test(prices, "donchian", model, BacktestConfig(), n_trials=20, symbol="SIG")
    assert row["p_value"] == pytest.approx(1 - row["percentile"] / 100, abs=1e-9)
    assert row["skilled"] == (row["p_value"] < 0.05)


def test_permutation_is_reproducible(prices, model):
    cfg = BacktestConfig()
    a = permutation_test(prices, "ma", model, cfg, n_trials=10, symbol="SIG")
    b = permutation_test(prices, "ma", model, cfg, n_trials=10, symbol="SIG")
    assert a == b


def test_report_renders(prices, model):
    rows = [permutation_test(prices, n, model, BacktestConfig(), n_trials=8, symbol="SIG")
            for n in ("ma", "donchian")]
    text = format_report(rows, "SIG")

    assert "Permutation test" in text
    assert "PCTILE" in text
    assert "of 2 strategies" in text
