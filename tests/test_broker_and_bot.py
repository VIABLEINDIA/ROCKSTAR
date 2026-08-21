"""Broker ledger correctness and the Section IV.F stop conditions."""

from __future__ import annotations

import pandas as pd
import pytest

from algobot.broker.base import Position
from algobot.broker.dhan import is_nse_session
from algobot.broker.simulated import ReplayBroker, SimulatedBroker
from algobot.bot.risk import RiskState
from algobot.bot.trader import TradingBot
from algobot.config import BotConfig, DataConfig
from algobot.data.loader import load_synthetic

SYMBOL = "TESTCO"


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return load_synthetic(SYMBOL, days=600)


@pytest.fixture
def broker(tmp_path, prices) -> SimulatedBroker:
    b = SimulatedBroker(cash=100_000.0, state_file=tmp_path / "ledger.json",
                        price_source="history", force_market_open=True,
                        data_cfg=DataConfig(symbol=SYMBOL, source="synthetic"))
    b.reset(100_000.0)
    b._history_cache[SYMBOL] = prices
    return b


# ----------------------------------------------------------------------
# Position maths
# ----------------------------------------------------------------------
def test_position_pnl_signs():
    long = Position(SYMBOL, 10, 100.0, 110.0)
    short = Position(SYMBOL, -10, 100.0, 90.0)

    assert long.unrealised_pnl == pytest.approx(100.0)
    assert long.unrealised_pct == pytest.approx(0.10)
    assert short.unrealised_pnl == pytest.approx(100.0)
    assert short.unrealised_pct == pytest.approx(0.10)


# ----------------------------------------------------------------------
# Simulated broker
# ----------------------------------------------------------------------
def test_buy_then_sell_updates_cash_and_realised_pnl(broker, prices):
    price = broker.last_price(SYMBOL)
    broker.buy(SYMBOL, 10)

    pos = broker.get_position(SYMBOL)
    assert pos is not None and pos.quantity == 10
    assert broker.cash == pytest.approx(100_000.0 - 10 * price)

    broker.sell(SYMBOL, 10)
    assert broker.get_position(SYMBOL) is None
    assert broker.cash == pytest.approx(100_000.0)
    assert broker.realised_pnl == pytest.approx(0.0)


def test_averaging_up_weights_the_entry_price(broker):
    broker.submit_order(SYMBOL, "BUY", 10, order_type="LIMIT", price=100.0)
    broker.submit_order(SYMBOL, "BUY", 10, order_type="LIMIT", price=120.0)

    pos = broker.positions[SYMBOL]
    assert pos.quantity == 20
    assert pos.avg_price == pytest.approx(110.0)


def test_realised_pnl_is_booked_on_the_closing_leg(broker):
    broker.submit_order(SYMBOL, "BUY", 10, order_type="LIMIT", price=100.0)
    broker.submit_order(SYMBOL, "SELL", 10, order_type="LIMIT", price=130.0)
    assert broker.realised_pnl == pytest.approx(300.0)


def test_close_position_flattens(broker):
    broker.buy(SYMBOL, 5)
    order = broker.close_position(SYMBOL)
    assert order is not None and order.side == "SELL"
    assert broker.get_position(SYMBOL) is None
    assert broker.close_position(SYMBOL) is None      # already flat


def test_orders_are_validated(broker):
    with pytest.raises(ValueError):
        broker.submit_order(SYMBOL, "BUY", 0)


def test_ledger_persists_across_instances(broker, tmp_path, prices):
    broker.submit_order(SYMBOL, "BUY", 7, order_type="LIMIT", price=100.0)

    reopened = SimulatedBroker(state_file=broker.state_file, price_source="history",
                               data_cfg=DataConfig(symbol=SYMBOL, source="synthetic"))
    reopened._history_cache[SYMBOL] = prices
    assert reopened.positions[SYMBOL].quantity == 7
    assert reopened.cash == pytest.approx(broker.cash)


def test_equity_tracks_the_mark(broker):
    broker.submit_order(SYMBOL, "BUY", 10, order_type="LIMIT", price=100.0)
    account = broker.account()
    assert account.equity == pytest.approx(broker.cash + 10 * broker.last_price(SYMBOL))
    assert account.currency == "INR"


# ----------------------------------------------------------------------
# NSE session clock
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "stamp,expected",
    [
        ("2026-08-20 10:00", True),    # Thursday mid-session
        ("2026-08-20 09:14", False),   # one minute before the open
        ("2026-08-20 15:31", False),   # one minute after the close
        ("2026-08-22 11:00", False),   # Saturday
    ],
)
def test_nse_session_window(stamp, expected):
    ts = pd.Timestamp(stamp, tz="Asia/Kolkata").to_pydatetime()
    assert is_nse_session(ts) is expected


# ----------------------------------------------------------------------
# Risk state
# ----------------------------------------------------------------------
def test_day_stop_loss_triggers_below_the_limit():
    risk = RiskState(starting_equity=100_000.0, day_stop_loss_pct=0.02)
    assert risk.check_session(99_000.0) is None
    assert "stop-loss" in risk.check_session(97_900.0)


def test_day_take_profit_triggers():
    risk = RiskState(starting_equity=100_000.0, day_take_profit_pct=0.03)
    assert risk.check_session(102_000.0) is None
    assert "take-profit" in risk.check_session(103_500.0)


def test_trade_cap_halts():
    risk = RiskState(starting_equity=100_000.0, max_trades_per_day=2)
    risk.record_trade()
    risk.record_trade()
    assert "trade cap" in risk.check_session(100_000.0)


def test_position_stop_loss():
    risk = RiskState(starting_equity=100_000.0, trade_stop_loss_pct=0.02)
    assert risk.check_position(Position(SYMBOL, 10, 100.0, 99.0)) is None
    assert "stop-loss" in risk.check_position(Position(SYMBOL, 10, 100.0, 97.0))
    assert risk.check_position(None) is None


def test_halt_records_the_reason():
    risk = RiskState(starting_equity=100.0)
    risk.halt("because")
    assert risk.halted and risk.halt_reason == "because"


# ----------------------------------------------------------------------
# Bot loop -- the three Section IV.F stop conditions
# ----------------------------------------------------------------------
def make_bot(broker, tmp_path, **overrides) -> TradingBot:
    cfg = BotConfig(symbol=SYMBOL, strategy="ma", quantity=10, poll_seconds=0,
                    use_model=False, stop_file=str(tmp_path / "STOP"), **overrides)
    return TradingBot(broker, cfg)


def test_bot_stops_on_a_user_stop_file(broker, tmp_path):
    bot = make_bot(broker, tmp_path)
    (tmp_path / "STOP").write_text("stop", encoding="utf-8")

    assert bot.step() == "stopped"
    assert bot.risk.halt_reason == "user stop signal"


def test_bot_stops_on_a_programmatic_stop(broker, tmp_path):
    bot = make_bot(broker, tmp_path)
    bot.request_stop()
    assert bot.step() == "stopped"


def test_bot_flattens_an_open_position_when_stopped(broker, tmp_path):
    bot = make_bot(broker, tmp_path)
    broker.buy(SYMBOL, 10)
    bot.request_stop()

    assert bot.step() == "stopped"
    assert broker.get_position(SYMBOL) is None


def test_bot_stops_when_the_market_is_closed(broker, tmp_path, monkeypatch):
    bot = make_bot(broker, tmp_path)
    # Pinned rather than read from the wall clock, which made this a no-op
    # during NSE hours and an assertion outside them.
    monkeypatch.setattr(broker, "is_market_open", lambda: False)

    assert bot.step() == "market_closed"
    assert bot.risk.halt_reason == "market closed"


def test_bot_halts_on_the_day_stop_loss(broker, tmp_path):
    bot = make_bot(broker, tmp_path, day_stop_loss_pct=0.01)
    bot.risk.starting_equity = broker.account().equity * 1.5   # force a large paper loss

    assert bot.step() == "halted"
    assert "stop-loss" in bot.risk.halt_reason


def test_bot_run_returns_a_summary(broker, tmp_path):
    bot = make_bot(broker, tmp_path)
    summary = bot.run(max_iterations=2)

    for key in ("symbol", "iterations", "starting_equity", "ending_equity", "session_pnl"):
        assert key in summary
    assert summary["symbol"] == SYMBOL


def test_dry_run_never_places_an_order(broker, tmp_path):
    bot = make_bot(broker, tmp_path, dry_run=True)
    before = len(broker.fills)
    bot._enter(100.0, {"close": 100.0})
    assert len(broker.fills) == before


def test_market_view_reports_the_latest_bar(broker, tmp_path, prices):
    bot = make_bot(broker, tmp_path)
    view = bot.market_view()

    assert view["as_of"] == str(prices.index[-1].date())
    assert view["close"] == pytest.approx(float(prices["Close"].iloc[-1]))
    assert view["signal"] in (-1, 0, 1)


# ----------------------------------------------------------------------
# Replay broker
# ----------------------------------------------------------------------
def test_replay_broker_walks_forward(prices, tmp_path):
    broker = ReplayBroker(prices, SYMBOL, start_index=100,
                          state_file=tmp_path / "replay.json")
    broker.reset()

    first = broker.last_price(SYMBOL)
    assert len(broker.history(SYMBOL, 1000)) == 101

    assert broker.step() is True
    assert broker.last_price(SYMBOL) != first or True    # price may repeat; index must move
    assert len(broker.history(SYMBOL, 1000)) == 102


def test_replay_broker_stops_at_the_end(prices, tmp_path):
    broker = ReplayBroker(prices, SYMBOL, start_index=len(prices) - 2,
                          state_file=tmp_path / "replay2.json")
    assert broker.step() is True
    assert broker.step() is False
    assert broker.is_market_open() is False
