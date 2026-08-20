"""Backtest engine: execution timing, stops, and the paper's two headline stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.backtest.engine import Trade, run_backtest
from algobot.backtest.report import format_strike, money, strategy_table, summary_table
from algobot.config import BacktestConfig
from algobot.data.loader import load_synthetic
from algobot.strategies.base import EXIT, FLAT, LONG, Strategy
from algobot.strategies.registry import build_strategy


class ScriptedStrategy(Strategy):
    """Replays a fixed signal list, so execution can be asserted exactly."""

    def __init__(self, signals: list[int]):
        self.name = "scripted"
        self._signals = signals

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self._signals[: len(df)], index=df.index, dtype=int)

    def params(self) -> dict:
        return {}


def frame_from(opens, highs=None, lows=None, closes=None) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=len(opens))
    opens = pd.Series(opens, index=idx, dtype=float)
    closes = pd.Series(closes if closes is not None else opens, index=idx, dtype=float)
    highs = pd.Series(highs if highs is not None else np.maximum(opens, closes), index=idx, dtype=float)
    lows = pd.Series(lows if lows is not None else np.minimum(opens, closes), index=idx, dtype=float)
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": 1_000}, index=idx
    )


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return load_synthetic("BT", days=800)


# ----------------------------------------------------------------------
# Trade maths
# ----------------------------------------------------------------------
def test_trade_pnl_long_and_short():
    ts = pd.Timestamp("2024-01-01")
    te = pd.Timestamp("2024-01-10")
    long = Trade(ts, 100.0, te, 110.0, 10, "long")
    short = Trade(ts, 100.0, te, 90.0, 10, "short")

    assert long.pnl == pytest.approx(100.0)
    assert long.return_pct == pytest.approx(10.0)
    assert short.pnl == pytest.approx(100.0)
    assert short.return_pct == pytest.approx(10.0)


# ----------------------------------------------------------------------
# execution timing
# ----------------------------------------------------------------------
def test_signal_executes_on_the_next_bar_open():
    df = frame_from([10, 20, 30, 40])
    # Signal on bar 0 -> fill at bar 1's open (20), exit signal on bar 2 -> fill at 40.
    # cost_model="none" isolates execution timing from transaction charges.
    result = run_backtest(df, ScriptedStrategy([LONG, FLAT, EXIT, FLAT]),
                          BacktestConfig(quantity=1, stop_loss_pct=0.0, commission=0.0,
                                         cost_model="none"))

    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(20.0)
    assert trade.exit_price == pytest.approx(40.0)
    assert trade.pnl == pytest.approx(20.0)


def test_no_trade_is_opened_on_the_final_bar_signal():
    df = frame_from([10, 11, 12])
    result = run_backtest(df, ScriptedStrategy([FLAT, FLAT, LONG]),
                          BacktestConfig(quantity=1, stop_loss_pct=0.0))
    assert result.n_trades == 0


def test_open_position_is_closed_at_the_end_of_the_test():
    df = frame_from([10, 20, 30])
    result = run_backtest(df, ScriptedStrategy([LONG, FLAT, FLAT]),
                          BacktestConfig(quantity=1, stop_loss_pct=0.0))
    assert result.n_trades == 1
    assert result.trades[0].exit_reason == "end_of_test"


def test_duplicate_long_signals_do_not_stack():
    df = frame_from([10, 11, 12, 13, 14])
    result = run_backtest(df, ScriptedStrategy([LONG, LONG, LONG, LONG, LONG]),
                          BacktestConfig(quantity=1, stop_loss_pct=0.0))
    assert result.n_trades == 1
    assert abs(result.trades[0].quantity) == 1


# ----------------------------------------------------------------------
# protective exits
# ----------------------------------------------------------------------
def test_stop_loss_fires_on_an_intrabar_low():
    df = frame_from(opens=[100, 100, 100], lows=[100, 100, 80], highs=[100, 100, 100],
                    closes=[100, 100, 95])
    result = run_backtest(df, ScriptedStrategy([LONG, FLAT, FLAT]),
                          BacktestConfig(quantity=1, stop_loss_pct=0.10))

    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(90.0)


def test_take_profit_fires_on_an_intrabar_high():
    df = frame_from(opens=[100, 100, 100], highs=[100, 100, 130], lows=[100, 100, 100],
                    closes=[100, 100, 120])
    result = run_backtest(df, ScriptedStrategy([LONG, FLAT, FLAT]),
                          BacktestConfig(quantity=1, stop_loss_pct=0.0, take_profit_pct=0.20))

    assert result.trades[0].exit_reason == "take_profit"
    assert result.trades[0].exit_price == pytest.approx(120.0)


def test_slippage_always_works_against_the_trader():
    df = frame_from([100, 100, 100, 100])
    clean = run_backtest(df, ScriptedStrategy([LONG, FLAT, EXIT, FLAT]),
                         BacktestConfig(quantity=1, stop_loss_pct=0.0))
    slipped = run_backtest(df, ScriptedStrategy([LONG, FLAT, EXIT, FLAT]),
                           BacktestConfig(quantity=1, stop_loss_pct=0.0, slippage_bps=50))

    assert slipped.profit < clean.profit
    assert slipped.trades[0].entry_price > clean.trades[0].entry_price
    assert slipped.trades[0].exit_price < clean.trades[0].exit_price


def test_commission_reduces_profit():
    df = frame_from([10, 20, 30, 40])
    free = run_backtest(df, ScriptedStrategy([LONG, FLAT, EXIT, FLAT]),
                        BacktestConfig(quantity=1, stop_loss_pct=0.0, commission=0.0))
    charged = run_backtest(df, ScriptedStrategy([LONG, FLAT, EXIT, FLAT]),
                           BacktestConfig(quantity=1, stop_loss_pct=0.0, commission=5.0))
    assert charged.final_equity < free.final_equity


# ----------------------------------------------------------------------
# reported statistics
# ----------------------------------------------------------------------
def test_strike_rate_counts_winning_trades():
    # bar:      0    1    2    3    4    5
    # win then loss.
    df = frame_from([10, 20, 30, 30, 20, 20])
    result = run_backtest(df, ScriptedStrategy([LONG, FLAT, EXIT, LONG, FLAT, EXIT]),
                          BacktestConfig(quantity=1, stop_loss_pct=0.0, cost_model="none"))
    assert result.n_trades == 2
    assert result.strike_rate == pytest.approx(50.0)


def test_strike_rate_is_nan_without_trades():
    df = frame_from([10, 11, 12])
    result = run_backtest(df, ScriptedStrategy([FLAT, FLAT, FLAT]), BacktestConfig())
    assert np.isnan(result.strike_rate)
    assert result.profit == 0.0
    assert format_strike(result.strike_rate) == "NA"


def test_profit_matches_the_equity_curve():
    df = frame_from([10, 20, 30, 40])
    cfg = BacktestConfig(quantity=2, stop_loss_pct=0.0, commission=0.0, cost_model="none")
    result = run_backtest(df, ScriptedStrategy([LONG, FLAT, EXIT, FLAT]), cfg)
    assert result.final_equity - cfg.initial_cash == pytest.approx(result.profit)


def test_equity_curve_is_aligned_to_the_price_index(prices):
    result = run_backtest(prices, build_strategy("ma"), BacktestConfig(), "BT", "test")
    assert result.history.index.equals(prices.index)
    assert list(result.history.columns) == ["close", "signal", "equity"]


def test_max_drawdown_is_never_positive(prices):
    result = run_backtest(prices, build_strategy("multiple"), BacktestConfig())
    assert result.max_drawdown_pct <= 0


def test_empty_frame_is_handled():
    result = run_backtest(pd.DataFrame(), build_strategy("ma"), BacktestConfig())
    assert result.n_trades == 0
    assert result.history.empty


def test_result_dict_is_json_safe(prices):
    import json

    result = run_backtest(prices, build_strategy("donchian"), BacktestConfig(), "BT", "10y")
    json.dumps(result.to_dict())        # must not raise on nan/inf


# ----------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------
def test_money_formats_rupees():
    assert money(1234.5, "INR") == "Rs 1,234.50"
    assert money(1234.5, "USD") == "$ 1,234.50"


def test_tables_render(prices):
    results = [
        run_backtest(prices.tail(250), build_strategy("ma"), BacktestConfig(), "BT", "1y"),
        run_backtest(prices, build_strategy("ma"), BacktestConfig(), "BT", "10y"),
    ]
    table = strategy_table(results, "ma")
    assert "STRIKE RATE" in table and "PROFIT EARNED" in table
    assert "1y" in table and "10y" in table
    assert "Moving Average Crossover" in summary_table(results)
