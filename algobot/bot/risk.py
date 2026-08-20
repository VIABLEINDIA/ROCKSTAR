"""Risk guards for the live loop.

Section IV.F: the bot "will continue to do so until either Stop Loss is
reached, Market is closed or User sends a Stop signal". This module owns the
stop-loss half of that rule, at two levels:

  * per-trade  -- an open position that falls `trade_stop_loss_pct` below its
    entry is flattened immediately;
  * per-session -- once the day's realised+unrealised P&L breaches
    `day_stop_loss_pct` of starting equity, the bot flattens and halts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from ..broker.base import Position

log = logging.getLogger(__name__)


@dataclass
class RiskState:
    starting_equity: float
    day_stop_loss_pct: float = 0.02
    day_take_profit_pct: float | None = None
    trade_stop_loss_pct: float = 0.02
    trade_take_profit_pct: float | None = None
    max_trades_per_day: int = 20
    trades_today: int = 0
    halted: bool = False
    halt_reason: str = ""
    started_at: datetime = field(default_factory=datetime.now)

    # ------------------------------------------------------------------
    def day_pnl(self, equity: float) -> float:
        return equity - self.starting_equity

    def day_pnl_pct(self, equity: float) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return equity / self.starting_equity - 1.0

    def check_session(self, equity: float) -> str | None:
        """Return a halt reason if a session-level limit is breached."""
        pnl_pct = self.day_pnl_pct(equity)

        if self.day_stop_loss_pct and pnl_pct <= -abs(self.day_stop_loss_pct):
            return (f"day stop-loss hit: {pnl_pct:+.2%} "
                    f"(limit {-abs(self.day_stop_loss_pct):.2%})")

        if self.day_take_profit_pct and pnl_pct >= abs(self.day_take_profit_pct):
            return (f"day take-profit hit: {pnl_pct:+.2%} "
                    f"(target {abs(self.day_take_profit_pct):.2%})")

        if self.max_trades_per_day and self.trades_today >= self.max_trades_per_day:
            return f"daily trade cap reached ({self.trades_today})"

        return None

    def check_position(self, pos: Position | None) -> str | None:
        """Return an exit reason if the open position breached its own limits."""
        if pos is None or not pos.is_open or pos.avg_price <= 0:
            return None

        change = pos.unrealised_pct
        if self.trade_stop_loss_pct and change <= -abs(self.trade_stop_loss_pct):
            return f"position stop-loss: {change:+.2%}"
        if self.trade_take_profit_pct and change >= abs(self.trade_take_profit_pct):
            return f"position take-profit: {change:+.2%}"
        return None

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason
        log.warning("Trading halted -- %s", reason)

    def record_trade(self) -> None:
        self.trades_today += 1
