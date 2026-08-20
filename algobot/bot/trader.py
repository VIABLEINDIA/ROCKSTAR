"""Section IV.F -- the live trading loop.

"The strategy parameters are entered by the user, and once the Bot starts
trading it will continue to do so until either Stop Loss is reached, Market is
closed or User sends a Stop signal to Bot. The Bot constantly checks Market
conditions and current Positions in the market to decide its action. The Random
Forest model is integrated as a joblib file with the bot and the Bot is made to
take its decision on the basis of prediction from the model as well as the
financial strategy."

One iteration of `TradingBot.step()`:

    1. stop signal? (stop file, Ctrl-C, or `bot.request_stop()`)
    2. market open?
    3. refresh account + position; enforce session and per-trade stop-loss
    4. rebuild signals from fresh bars; ask the model for its forecast
    5. act -- enter, exit, or hold
"""

from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..broker.base import Broker, Position
from ..config import ARTIFACT_DIR, BotConfig
from ..model.random_forest import TrainedModel, default_model_path
from ..strategies.base import EXIT, FLAT, LONG
from ..broker.dhan import is_at_or_after, now_ist
from ..strategies.registry import Strategy, build_strategy, wrap_with_model
from .risk import RiskState

log = logging.getLogger(__name__)


@dataclass
class BotEvent:
    time: str
    kind: str                 # signal | order | halt | info | error
    message: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"time": self.time, "kind": self.kind, "message": self.message,
                **({"detail": self.detail} if self.detail else {})}


class TradingBot:
    """Strategy + model + broker, driven by a polling loop."""

    def __init__(self, broker: Broker, cfg: BotConfig | None = None,
                 strategy: Strategy | None = None, model: TrainedModel | None = None,
                 strategy_params: dict | None = None):
        self.cfg = cfg or BotConfig()
        self.broker = broker
        self.strategy_params = strategy_params or {}
        self.model = model if model is not None else self._maybe_load_model()

        base = strategy or build_strategy(self.cfg.strategy, **self.strategy_params)
        self.base_strategy = base
        self.strategy = (
            wrap_with_model(base, self.model, **self.strategy_params)
            if self.model is not None else base
        )

        self.events: list[BotEvent] = []
        self._stop_requested = False
        self._install_signal_handlers()

        account = self.broker.account()
        self.risk = RiskState(
            starting_equity=account.equity,
            day_stop_loss_pct=self.cfg.day_stop_loss_pct,
            day_take_profit_pct=self.cfg.day_take_profit_pct,
            trade_stop_loss_pct=self.cfg.trade_stop_loss_pct,
            max_trades_per_day=self.cfg.max_trades_per_day,
        )
        self._log_event("info", f"Bot ready: {self.strategy.describe()} on {self.cfg.symbol} "
                                f"via {self.broker.name}", account.__dict__)

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    def _maybe_load_model(self) -> TrainedModel | None:
        if not self.cfg.use_model:
            return None
        path = Path(self.cfg.model_path) if self.cfg.model_path else default_model_path(self.cfg.symbol)
        if not path.exists():
            log.warning("No model at %s -- running on the financial strategy alone "
                        "(train one with: algobot train --symbol %s)", path, self.cfg.symbol)
            return None
        model = TrainedModel.load(path)
        log.info("Loaded RF model %s (trained %s, horizon %d bars)",
                 path.name, model.trained_at[:19], model.horizon)
        return model

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            log.warning("Received signal %s -- stopping after this iteration", signum)
            self._stop_requested = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass    # not on the main thread (e.g. under a test runner)

    # ------------------------------------------------------------------
    # intraday session management
    # ------------------------------------------------------------------
    @property
    def is_intraday(self) -> bool:
        """True when orders go out as Dhan MIS, which the broker force-closes."""
        return str(self.cfg.product_type).upper() == "INTRADAY"

    def broker_now(self) -> datetime:
        """Current time from the broker if it exposes one, else the IST clock.

        ReplayBroker reports the timestamp of the bar it is standing on, so a
        replay squares off on simulated time rather than wall-clock time.
        """
        ts = getattr(self.broker, "current_time", None)
        return ts if isinstance(ts, datetime) else now_ist()

    def should_square_off(self, now: datetime | None = None) -> bool:
        """Has the intraday square-off cutoff passed?"""
        if not (self.is_intraday and self.cfg.auto_square_off):
            return False
        return is_at_or_after(self.cfg.square_off_time, now or self.broker_now())

    def entries_allowed(self, now: datetime | None = None) -> bool:
        """Block new intraday entries too late in the session to manage.

        Opening a position that must be closed within the hour leaves no room
        for the trade to work and guarantees an exit at the cutoff price.
        """
        if not self.is_intraday:
            return True
        return not is_at_or_after(self.cfg.no_new_entries_after, now or self.broker_now())

    def request_stop(self) -> None:
        """Programmatic equivalent of the user's stop signal."""
        self._stop_requested = True

    def stop_signal_received(self) -> bool:
        if self._stop_requested:
            return True
        stop_file = Path(self.cfg.stop_file)
        if stop_file.exists():
            self._log_event("info", f"Stop file detected at {stop_file}")
            return True
        return False

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def _log_event(self, kind: str, message: str, detail: dict | None = None) -> BotEvent:
        event = BotEvent(datetime.now().isoformat(timespec="seconds"), kind, message, detail or {})
        self.events.append(event)
        getattr(log, "error" if kind == "error" else "info")("%s: %s", kind.upper(), message)
        return event

    def journal_path(self) -> Path:
        return ARTIFACT_DIR / f"{self.cfg.symbol.upper()}_bot_journal.json"

    def save_journal(self) -> Path:
        path = self.journal_path()
        path.write_text(
            json.dumps(
                {
                    "symbol": self.cfg.symbol,
                    "strategy": self.strategy.describe(),
                    "broker": self.broker.name,
                    "product_type": self.cfg.product_type,
                    "model": (self.model.symbol + " @ " + self.model.trained_at
                              if self.model else None),
                    "starting_equity": self.risk.starting_equity,
                    "halted": self.risk.halted,
                    "halt_reason": self.risk.halt_reason,
                    "trades_today": self.risk.trades_today,
                    "events": [e.to_dict() for e in self.events],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    # ------------------------------------------------------------------
    # decision
    # ------------------------------------------------------------------
    def market_view(self) -> dict:
        """Fresh bars -> strategy signal + model forecast."""
        bars = self.broker.history(self.cfg.symbol, lookback_days=400)
        if bars.empty:
            raise RuntimeError(f"No market history available for {self.cfg.symbol}")

        signals = self.strategy.generate_signals(bars)
        signal_now = int(signals.iloc[-1]) if len(signals) else FLAT

        view = {
            "bars": len(bars),
            "as_of": str(bars.index[-1].date()),
            "close": float(bars["Close"].iloc[-1]),
            "signal": signal_now,
            "base_signal": int(self.base_strategy.generate_signals(bars).iloc[-1]),
            "expected_return": None,
            "forecast": None,
        }

        if self.model is not None:
            try:
                forecast = self.model.predict_next(bars)
                view["forecast"] = [round(float(v), 2) for v in forecast]
                view["expected_return"] = round(self.model.expected_return(bars), 5)
            except Exception as exc:      # a bad model must not kill the session
                self._log_event("error", f"Model prediction failed: {exc}")

        return view

    def _flatten(self, reason: str, pos: Position) -> None:
        if self.cfg.dry_run:
            self._log_event("order", f"[DRY RUN] would flatten {pos.quantity} {self.cfg.symbol} "
                                     f"({reason})")
            return
        order = self.broker.close_position(self.cfg.symbol)
        if order:
            self.risk.record_trade()
            self._log_event("order", f"EXIT {order.quantity} {self.cfg.symbol} @ "
                                     f"{order.price:.2f} ({reason})",
                            {"order_id": order.order_id, "pnl": round(pos.unrealised_pnl, 2)})

    def _enter(self, price: float, view: dict) -> None:
        if self.cfg.dry_run:
            self._log_event("order", f"[DRY RUN] would buy {self.cfg.quantity} "
                                     f"{self.cfg.symbol} @ ~{price:.2f}", view)
            return
        order = self.broker.buy(self.cfg.symbol, self.cfg.quantity)
        self.risk.record_trade()
        self._log_event("order", f"BUY {order.quantity} {self.cfg.symbol} @ {order.price:.2f}",
                        {"order_id": order.order_id, **view})

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def step(self) -> str:
        """One decision cycle. Returns a short status string."""
        # 1. user stop signal
        if self.stop_signal_received():
            pos = self.broker.get_position(self.cfg.symbol)
            if pos and pos.is_open:
                self._flatten("user stop signal", pos)
            self.risk.halt("user stop signal")
            return "stopped"

        # 2. market condition
        if not self.broker.is_market_open():
            pos = self.broker.get_position(self.cfg.symbol)
            if pos and pos.is_open:
                # Nothing can be traded now. Flag it loudly: an MIS position
                # left open here is closed by Dhan at its own discretion.
                self._log_event(
                    "error" if self.is_intraday else "info",
                    f"Market closed while holding {pos.quantity} {self.cfg.symbol}"
                    + (" as INTRADAY -- Dhan will auto-square-off at its own price"
                       if self.is_intraday else " (CNC, carried overnight)"),
                )
            self.risk.halt("market closed")
            return "market_closed"

        # 2b. intraday square-off, ahead of the broker's own cutoff
        if self.should_square_off():
            pos = self.broker.get_position(self.cfg.symbol)
            reason = f"intraday square-off at {self.cfg.square_off_time} IST"
            if pos and pos.is_open:
                self._flatten(reason, pos)
            self.risk.halt(reason)
            return "squared_off"

        # 3. account + position, then risk limits
        account = self.broker.account()
        pos = self.broker.get_position(self.cfg.symbol)

        position_exit = self.risk.check_position(pos)
        if position_exit and pos:
            self._flatten(position_exit, pos)
            pos = None

        session_halt = self.risk.check_session(account.equity)
        if session_halt:
            pos = self.broker.get_position(self.cfg.symbol)
            if pos and pos.is_open:
                self._flatten(session_halt, pos)
            self.risk.halt(session_halt)
            return "halted"

        # 4. market view
        try:
            view = self.market_view()
        except Exception as exc:
            self._log_event("error", f"Could not build market view: {exc}")
            return "error"

        # 5. act
        signal_now, holding = view["signal"], bool(pos and pos.is_open)

        if signal_now == LONG and not holding:
            if not self.entries_allowed():
                self._log_event("signal", f"entry suppressed: past "
                                          f"{self.cfg.no_new_entries_after} IST, too late to "
                                          f"open an intraday position")
                return "entry_blocked"
            self._enter(view["close"], view)
            return "entered"

        if signal_now == EXIT and holding:
            self._flatten("strategy exit signal", pos)
            return "exited"

        self._log_event(
            "signal",
            f"hold ({'in position' if holding else 'flat'}) "
            f"signal={signal_now} close={view['close']:.2f}"
            + (f" exp_ret={view['expected_return']:+.3%}"
               if view["expected_return"] is not None else ""),
        )
        return "hold"

    def run(self, max_iterations: int | None = None) -> dict:
        """Poll until stop-loss, market close, or a user stop signal."""
        self._log_event("info", f"Starting loop (poll {self.cfg.poll_seconds}s, "
                                f"day stop {self.cfg.day_stop_loss_pct:.2%})")
        iterations = 0
        try:
            while not self.risk.halted:
                status = self.step()
                iterations += 1
                if status in ("stopped", "market_closed", "halted", "squared_off"):
                    break
                if max_iterations and iterations >= max_iterations:
                    self._log_event("info", f"Reached max_iterations={max_iterations}")
                    break
                time.sleep(self.cfg.poll_seconds)
        except KeyboardInterrupt:
            pos = self.broker.get_position(self.cfg.symbol)
            if pos and pos.is_open:
                self._flatten("keyboard interrupt", pos)
            self.risk.halt("keyboard interrupt")

        account = self.broker.account()
        summary = {
            "symbol": self.cfg.symbol,
            "iterations": iterations,
            "trades": self.risk.trades_today,
            "starting_equity": round(self.risk.starting_equity, 2),
            "ending_equity": round(account.equity, 2),
            "session_pnl": round(self.risk.day_pnl(account.equity), 2),
            "session_pnl_pct": round(self.risk.day_pnl_pct(account.equity) * 100, 3),
            "halt_reason": self.risk.halt_reason,
            "journal": str(self.save_journal()),
        }
        self._log_event("info", f"Session finished: {summary['session_pnl']:+.2f} "
                                f"{account.currency} over {iterations} iterations")
        return summary


def replay_session(prices: pd.DataFrame, symbol: str, cfg: BotConfig | None = None,
                   model: TrainedModel | None = None, start_index: int = 250,
                   max_bars: int = 250, strategy_params: dict | None = None) -> dict:
    """Drive the real bot loop over historical bars (dry-run rehearsal)."""
    from ..broker.simulated import ReplayBroker

    cfg = cfg or BotConfig(symbol=symbol)
    cfg.symbol = symbol
    cfg.poll_seconds = 0

    broker = ReplayBroker(prices, symbol, start_index=start_index,
                          state_file=ARTIFACT_DIR / f"{symbol.upper()}_replay_account.json")
    broker.reset(cash=100_000.0)

    bot = TradingBot(broker, cfg, model=model, strategy_params=strategy_params)
    bars = 0
    while bars < max_bars and broker.step():
        status = bot.step()
        bars += 1
        if status in ("stopped", "halted"):
            break

    pos = broker.get_position(symbol)
    if pos and pos.is_open:
        broker.close_position(symbol)

    account = broker.account()
    return {
        "symbol": symbol,
        "bars_replayed": bars,
        "trades": len(broker.fills),
        "starting_equity": 100_000.0,
        "ending_equity": round(account.equity, 2),
        "realised_pnl": round(broker.realised_pnl, 2),
        "halt_reason": bot.risk.halt_reason,
        "journal": str(bot.save_journal()),
    }
