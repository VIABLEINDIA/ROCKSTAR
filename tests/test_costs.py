"""Dhan / NSE transaction costs and their effect on backtested P&L."""

from __future__ import annotations

import pandas as pd
import pytest

from algobot.backtest.costs import (
    DELIVERY,
    INTRADAY,
    ZERO,
    ChargeBreakdown,
    CostModel,
    get_cost_model,
)
from algobot.backtest.engine import run_backtest
from algobot.config import BacktestConfig
from algobot.data.loader import load_synthetic
from algobot.strategies.base import EXIT, FLAT, LONG, Strategy
from algobot.strategies.registry import build_strategy


class ScriptedStrategy(Strategy):
    def __init__(self, signals):
        self.name = "scripted"
        self._signals = signals

    def generate_signals(self, df):
        return pd.Series(self._signals[: len(df)], index=df.index, dtype=int)

    def params(self):
        return {}


def frame_from(prices):
    idx = pd.bdate_range("2024-01-01", periods=len(prices))
    s = pd.Series(prices, index=idx, dtype=float)
    return pd.DataFrame({"Open": s, "High": s, "Low": s, "Close": s, "Volume": 1_000}, index=idx)


# ----------------------------------------------------------------------
# rate arithmetic
# ----------------------------------------------------------------------
def test_delivery_stt_is_point_one_percent_each_side():
    buy = DELIVERY.charges(1000.0, 10, "BUY")     # turnover 10,000
    sell = DELIVERY.charges(1000.0, 10, "SELL")
    assert buy.stt == pytest.approx(10.0)
    assert sell.stt == pytest.approx(10.0)


def test_intraday_stt_is_sell_side_only():
    assert INTRADAY.charges(1000.0, 10, "BUY").stt == pytest.approx(0.0)
    assert INTRADAY.charges(1000.0, 10, "SELL").stt == pytest.approx(2.5)


def test_stamp_duty_is_buy_side_only():
    assert DELIVERY.charges(1000.0, 10, "BUY").stamp_duty == pytest.approx(1.5)
    assert DELIVERY.charges(1000.0, 10, "SELL").stamp_duty == pytest.approx(0.0)


def test_dp_charge_hits_delivery_sells_only():
    assert DELIVERY.charges(1000.0, 10, "SELL").dp_charges == pytest.approx(12.5)
    assert DELIVERY.charges(1000.0, 10, "BUY").dp_charges == pytest.approx(0.0)
    assert INTRADAY.charges(1000.0, 10, "SELL").dp_charges == pytest.approx(0.0)


def test_delivery_has_no_brokerage_but_intraday_does():
    assert DELIVERY.charges(1000.0, 10, "BUY").brokerage == pytest.approx(0.0)
    assert INTRADAY.charges(1000.0, 10, "BUY").brokerage == pytest.approx(3.0)  # 0.03%


def test_brokerage_is_capped_per_order():
    # 0.03% of 10,00,000 would be Rs 300; the cap is Rs 20.
    assert INTRADAY.charges(10_000.0, 100, "BUY").brokerage == pytest.approx(20.0)


def test_gst_applies_to_services_not_to_stt_or_stamp():
    c = INTRADAY.charges(1000.0, 10, "BUY")
    taxable = c.brokerage + c.exchange_txn + c.sebi
    assert c.gst == pytest.approx(taxable * 0.18)


def test_zero_model_charges_nothing():
    assert ZERO.charges(1000.0, 10, "BUY").total == pytest.approx(0.0)
    assert ZERO.charges(1000.0, 10, "SELL").total == pytest.approx(0.0)


def test_charges_scale_with_turnover():
    small = DELIVERY.charges(100.0, 10, "BUY").total
    big = DELIVERY.charges(1000.0, 10, "BUY").total
    assert big > small * 9          # roughly linear, flat components aside


def test_zero_turnover_is_free():
    assert DELIVERY.charges(0.0, 10, "BUY").total == pytest.approx(0.0)
    assert DELIVERY.charges(1000.0, 0, "BUY").total == pytest.approx(0.0)


def test_bad_side_rejected():
    with pytest.raises(ValueError):
        DELIVERY.charges(100.0, 1, "HOLD")


# ----------------------------------------------------------------------
# breakdown container
# ----------------------------------------------------------------------
def test_total_sums_every_component():
    c = DELIVERY.charges(1500.0, 10, "SELL")
    assert c.total == pytest.approx(
        c.brokerage + c.stt + c.exchange_txn + c.sebi + c.stamp_duty + c.gst + c.dp_charges
    )


def test_breakdowns_add():
    a = DELIVERY.charges(1000.0, 10, "BUY")
    b = DELIVERY.charges(1100.0, 10, "SELL")
    assert (a + b).total == pytest.approx(a.total + b.total)
    assert (a + b).stt == pytest.approx(a.stt + b.stt)


def test_round_trip_equals_both_legs():
    rt = DELIVERY.round_trip(1000.0, 1100.0, 10)
    legs = DELIVERY.charges(1000.0, 10, "BUY") + DELIVERY.charges(1100.0, 10, "SELL")
    assert rt.total == pytest.approx(legs.total)


def test_empty_breakdown_is_free():
    assert ChargeBreakdown().total == pytest.approx(0.0)


# ----------------------------------------------------------------------
# breakeven
# ----------------------------------------------------------------------
def test_delivery_breakeven_exceeds_intraday():
    assert DELIVERY.breakeven_pct(1300.0, 10) > INTRADAY.breakeven_pct(1300.0, 10)


def test_breakeven_is_a_plausible_fraction():
    be = DELIVERY.breakeven_pct(1300.0, 10)
    assert 0.001 < be < 0.01        # a few tenths of a percent


def test_breakeven_shrinks_as_the_flat_dp_charge_is_diluted():
    assert DELIVERY.breakeven_pct(3000.0, 10) < DELIVERY.breakeven_pct(500.0, 10)


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [("delivery", "delivery"), ("cnc", "delivery"),
                                           ("intraday", "intraday"), ("mis", "intraday"),
                                           ("none", "none"), (None, "none")])
def test_get_cost_model(name, expected):
    assert get_cost_model(name).name == expected


def test_get_cost_model_passes_instances_through():
    custom = CostModel(name="custom")
    assert get_cost_model(custom) is custom


def test_unknown_model_rejected():
    with pytest.raises(KeyError):
        get_cost_model("free-lunch")


# ----------------------------------------------------------------------
# engine integration
# ----------------------------------------------------------------------
def test_costs_reduce_net_profit_but_not_gross():
    df = frame_from([100, 100, 120, 120])
    signals = [LONG, FLAT, EXIT, FLAT]
    cfg = dict(quantity=10, stop_loss_pct=0.0)

    free = run_backtest(df, ScriptedStrategy(signals), BacktestConfig(cost_model="none", **cfg))
    charged = run_backtest(df, ScriptedStrategy(signals),
                           BacktestConfig(cost_model="delivery", **cfg))

    assert charged.gross_profit == pytest.approx(free.gross_profit)
    assert charged.profit < free.profit
    assert charged.total_costs > 0


def test_net_pnl_equals_gross_minus_costs():
    df = frame_from([100, 100, 120, 120])
    r = run_backtest(df, ScriptedStrategy([LONG, FLAT, EXIT, FLAT]),
                     BacktestConfig(quantity=10, stop_loss_pct=0.0, cost_model="delivery"))
    trade = r.trades[0]
    assert trade.pnl == pytest.approx(trade.gross_pnl - trade.costs)
    assert r.profit == pytest.approx(r.gross_profit - r.total_costs)


def test_costs_flow_through_to_the_equity_curve():
    df = frame_from([100, 100, 120, 120])
    cfg = BacktestConfig(quantity=10, stop_loss_pct=0.0, cost_model="delivery")
    r = run_backtest(df, ScriptedStrategy([LONG, FLAT, EXIT, FLAT]), cfg)
    assert r.final_equity - cfg.initial_cash == pytest.approx(r.profit)


def test_a_marginal_winner_becomes_a_loser_after_charges():
    """A move too small to clear the breakeven must not count as a win."""
    df = frame_from([1000, 1000, 1001, 1001])      # +0.1%, under the ~0.34% breakeven
    signals = [LONG, FLAT, EXIT, FLAT]

    # slippage pinned to 0: this test isolates the effect of charges alone.
    free = run_backtest(df, ScriptedStrategy(signals),
                        BacktestConfig(quantity=10, stop_loss_pct=0.0, cost_model="none",
                                       slippage_bps=0))
    charged = run_backtest(df, ScriptedStrategy(signals),
                           BacktestConfig(quantity=10, stop_loss_pct=0.0, cost_model="delivery",
                                          slippage_bps=0))

    assert free.strike_rate == pytest.approx(100.0)
    assert charged.strike_rate == pytest.approx(0.0)


def test_intraday_is_cheaper_than_delivery_on_the_same_trades():
    df = load_synthetic("COST", days=600)
    strategy = build_strategy("ma")
    base = dict(quantity=10, stop_loss_pct=0.05)

    d = run_backtest(df, strategy, BacktestConfig(cost_model="delivery", **base))
    i = run_backtest(df, strategy, BacktestConfig(cost_model="intraday", **base))
    assert i.total_costs < d.total_costs


def test_trade_dict_reports_gross_costs_and_net():
    df = frame_from([100, 100, 120, 120])
    r = run_backtest(df, ScriptedStrategy([LONG, FLAT, EXIT, FLAT]),
                     BacktestConfig(quantity=10, stop_loss_pct=0.0, cost_model="delivery"))
    d = r.trades[0].to_dict()
    for key in ("gross_pnl", "costs", "pnl"):
        assert key in d
    assert d["pnl"] == pytest.approx(d["gross_pnl"] - d["costs"], abs=0.01)


def test_result_dict_exposes_cost_drag():
    import json

    df = load_synthetic("COST2", days=600)
    r = run_backtest(df, build_strategy("donchian"),
                     BacktestConfig(cost_model="delivery"), "COST2", "test")
    payload = r.to_dict()
    assert "gross_profit" in payload and "total_costs" in payload
    json.dumps(payload)


def test_default_config_charges_costs():
    """Regression: the default must not silently report gross figures."""
    assert BacktestConfig().cost_model == "delivery"


# ----------------------------------------------------------------------
# slippage and gap-through stop fills
# ----------------------------------------------------------------------
def test_default_slippage_is_nonzero():
    """Regression: frictionless fills flatter every strategy."""
    assert BacktestConfig().slippage_bps > 0
    assert BacktestConfig().gap_through_stops is True


def test_slippage_moves_both_legs_against_the_trader():
    df = frame_from([100, 100, 100, 100])
    signals = [LONG, FLAT, EXIT, FLAT]
    base = dict(quantity=10, stop_loss_pct=0.0, cost_model="none")

    clean = run_backtest(df, ScriptedStrategy(signals), BacktestConfig(slippage_bps=0, **base))
    slipped = run_backtest(df, ScriptedStrategy(signals), BacktestConfig(slippage_bps=50, **base))

    assert slipped.trades[0].entry_price > clean.trades[0].entry_price   # pay up to buy
    assert slipped.trades[0].exit_price < clean.trades[0].exit_price     # sell lower
    assert slipped.gross_profit < clean.gross_profit


def test_slippage_scales_with_the_rate():
    df = frame_from([100, 100, 100, 100])
    signals = [LONG, FLAT, EXIT, FLAT]
    base = dict(quantity=10, stop_loss_pct=0.0, cost_model="none")

    small = run_backtest(df, ScriptedStrategy(signals), BacktestConfig(slippage_bps=10, **base))
    large = run_backtest(df, ScriptedStrategy(signals), BacktestConfig(slippage_bps=100, **base))
    assert large.gross_profit < small.gross_profit


def gap_frame():
    """Bar 2 gaps far below the 10% stop: opens at 80, never trades near 90."""
    idx = pd.bdate_range("2024-01-01", periods=3)
    return pd.DataFrame(
        {"Open": [100.0, 100.0, 80.0], "High": [100.0, 100.0, 82.0],
         "Low": [100.0, 100.0, 78.0], "Close": [100.0, 100.0, 79.0], "Volume": 1_000},
        index=idx,
    )


def test_gap_through_stop_fills_at_the_open_not_the_stop():
    r = run_backtest(gap_frame(), ScriptedStrategy([LONG, FLAT, FLAT]),
                     BacktestConfig(quantity=10, stop_loss_pct=0.10, cost_model="none",
                                    slippage_bps=0, gap_through_stops=True))
    trade = r.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(80.0)      # the gap open, not 90


def test_gap_handling_can_be_disabled():
    r = run_backtest(gap_frame(), ScriptedStrategy([LONG, FLAT, FLAT]),
                     BacktestConfig(quantity=10, stop_loss_pct=0.10, cost_model="none",
                                    slippage_bps=0, gap_through_stops=False))
    assert r.trades[0].exit_price == pytest.approx(90.0)   # optimistic stop-level fill


def test_gap_aware_stops_are_never_more_favourable():
    df = load_synthetic("GAP", days=800)
    strategy = build_strategy("multiple")
    base = dict(quantity=10, stop_loss_pct=0.05, cost_model="delivery", slippage_bps=5)

    optimistic = run_backtest(df, strategy, BacktestConfig(gap_through_stops=False, **base))
    realistic = run_backtest(df, strategy, BacktestConfig(gap_through_stops=True, **base))
    assert realistic.profit <= optimistic.profit


def test_stop_inside_the_bar_still_fills_at_the_stop():
    """No gap: the bar opens above the stop and trades down through it."""
    idx = pd.bdate_range("2024-01-01", periods=3)
    df = pd.DataFrame(
        {"Open": [100.0, 100.0, 99.0], "High": [100.0, 100.0, 99.0],
         "Low": [100.0, 100.0, 85.0], "Close": [100.0, 100.0, 88.0], "Volume": 1_000},
        index=idx,
    )
    r = run_backtest(df, ScriptedStrategy([LONG, FLAT, FLAT]),
                     BacktestConfig(quantity=10, stop_loss_pct=0.10, cost_model="none",
                                    slippage_bps=0, gap_through_stops=True))
    assert r.trades[0].exit_price == pytest.approx(90.0)


def test_take_profit_is_not_credited_for_a_favourable_gap():
    """Gapping past the target must not pay better than the target."""
    idx = pd.bdate_range("2024-01-01", periods=3)
    df = pd.DataFrame(
        {"Open": [100.0, 100.0, 130.0], "High": [100.0, 100.0, 135.0],
         "Low": [100.0, 100.0, 128.0], "Close": [100.0, 100.0, 132.0], "Volume": 1_000},
        index=idx,
    )
    r = run_backtest(df, ScriptedStrategy([LONG, FLAT, FLAT]),
                     BacktestConfig(quantity=10, stop_loss_pct=0.0, take_profit_pct=0.10,
                                    cost_model="none", slippage_bps=0))
    assert r.trades[0].exit_price == pytest.approx(110.0)   # the target, not 130
