"""Strategy signal correctness -- especially the look-ahead traps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.data.loader import load_synthetic
from algobot.strategies.base import EXIT, FLAT, LONG, crossover
from algobot.strategies.donchian import Donchian
from algobot.strategies.gold_cross import GoldCross
from algobot.strategies.indicators import atr, ema, macd, rsi, sma
from algobot.strategies.moving_average import MovingAverageCrossover
from algobot.strategies.multiple import MultipleStrategy
from algobot.strategies.registry import PAPER_STRATEGIES, build_strategy, pretty


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return load_synthetic("STRAT", days=900)


def make_frame(closes: list[float]) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    close = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": 1_000},
        index=idx,
    )


# ----------------------------------------------------------------------
# generic contract
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", PAPER_STRATEGIES)
def test_signals_are_aligned_and_in_range(prices, name):
    sig = build_strategy(name).generate_signals(prices)
    assert len(sig) == len(prices)
    assert sig.index.equals(prices.index)
    assert set(np.unique(sig)) <= {EXIT, FLAT, LONG}


@pytest.mark.parametrize("name", PAPER_STRATEGIES)
def test_no_signal_before_warmup(prices, name):
    strategy = build_strategy(name)
    sig = strategy.generate_signals(prices)
    if strategy.warmup:
        assert (sig.iloc[: strategy.warmup - 1] == FLAT).all()


@pytest.mark.parametrize("name", PAPER_STRATEGIES)
def test_signals_do_not_depend_on_future_bars(prices, name):
    """Truncating the future must not change any past signal."""
    strategy = build_strategy(name)
    full = strategy.generate_signals(prices)
    cut = len(prices) - 50
    partial = strategy.generate_signals(prices.iloc[:cut])
    pd.testing.assert_series_equal(full.iloc[:cut], partial, check_names=False)


# ----------------------------------------------------------------------
# specific behaviour
# ----------------------------------------------------------------------
def test_crossover_marks_only_the_crossing_bar():
    fast = pd.Series([1.0, 1.0, 3.0, 3.0, 0.5])
    slow = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
    sig = crossover(fast, slow)
    assert sig.tolist() == [FLAT, FLAT, LONG, FLAT, EXIT]


def test_moving_average_buys_the_golden_cross():
    closes = [10] * 30 + list(np.linspace(10, 30, 30))
    sig = MovingAverageCrossover(fast=5, slow=20).generate_signals(make_frame(closes))
    assert (sig == LONG).any()
    assert sig[sig == LONG].index[0] > make_frame(closes).index[25]


def test_moving_average_rejects_inverted_windows():
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast=30, slow=10)


def test_donchian_channel_excludes_the_current_bar(prices):
    """If the channel included today, today's high would always be a breakout."""
    ch = Donchian(entry_window=20, exit_window=10).channels(prices)
    row = 100
    expected_upper = prices["High"].iloc[row - 20:row].max()
    assert ch["upper"].iloc[row] == pytest.approx(expected_upper)


def test_donchian_breaks_out_on_a_new_high():
    closes = [10] * 40 + [25]
    sig = Donchian(entry_window=20, exit_window=10).generate_signals(make_frame(closes))
    assert sig.iloc[-1] == LONG


def test_gold_cross_is_silent_without_200_bars():
    short = load_synthetic("SHORT", days=150)
    sig = GoldCross().generate_signals(short)
    assert (sig == FLAT).all()          # exactly why the paper reports NA for 1 year


def test_multiple_strategy_needs_agreement(prices):
    strategy = MultipleStrategy(min_votes=3)
    votes = strategy.votes(prices)
    sig = strategy.generate_signals(prices)

    longs = sig == LONG
    if longs.any():
        assert ((votes[longs] == LONG).sum(axis=1) >= 3).all()


def test_multiple_strategy_vote_count_changes_activity(prices):
    lenient = MultipleStrategy(min_votes=1).generate_signals(prices)
    strict = MultipleStrategy(min_votes=3).generate_signals(prices)
    assert (lenient == LONG).sum() >= (strict == LONG).sum()


# ----------------------------------------------------------------------
# indicators
# ----------------------------------------------------------------------
def test_rsi_is_bounded(prices):
    values = rsi(prices["Close"]).dropna()
    assert values.between(0, 100).all()


def test_rsi_saturates_on_a_pure_uptrend():
    closes = list(np.linspace(10, 60, 60))
    assert rsi(make_frame(closes)["Close"]).iloc[-1] > 95


def test_sma_and_ema_respect_min_periods(prices):
    assert sma(prices["Close"], 20).iloc[:19].isna().all()
    assert ema(prices["Close"], 20).iloc[:19].isna().all()


def test_macd_histogram_is_line_minus_signal(prices):
    m = macd(prices["Close"]).dropna()
    assert np.allclose(m["hist"], m["macd"] - m["signal"])


def test_atr_is_positive(prices):
    assert (atr(prices).dropna() > 0).all()


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------
def test_registry_passes_through_known_params_only():
    strategy = build_strategy("ma", fast=5, slow=15, entry_window=99)
    assert strategy.fast == 5 and strategy.slow == 15
    assert not hasattr(strategy, "entry_window")


def test_registry_rejects_unknown_strategy():
    with pytest.raises(KeyError):
        build_strategy("does-not-exist")


def test_pretty_names_the_fused_variant():
    assert pretty("ma") == "Moving Average Crossover"
    assert pretty("ma+rf") == "Moving Average Crossover + RF"
