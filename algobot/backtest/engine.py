"""Event-driven backtester producing the paper's Strike Rate / Profit Earned.

Execution model
---------------
* A signal computed from bar *t* is executed at the **open of bar t+1**. Acting
  on the same bar's close would let the strategy trade on information it could
  not have had, which inflates every result table.
* Protective stop-loss and take-profit are checked against the bar's low/high
  before the signal, and fill at the level (a conservative approximation -- a
  real gap-through would fill worse).
* Long-only by default, matching the paper's strategies; `allow_short` mirrors
  the same logic for shorts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..strategies.base import EXIT, LONG, Strategy
from .costs import ChargeBreakdown, get_cost_model


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    quantity: int
    side: str = "long"
    exit_reason: str = "signal"
    entry_charges: ChargeBreakdown = field(default_factory=ChargeBreakdown)
    exit_charges: ChargeBreakdown = field(default_factory=ChargeBreakdown)

    @property
    def gross_pnl(self) -> float:
        """P&L before transaction costs -- what the paper's tables report."""
        direction = 1 if self.side == "long" else -1
        return direction * (self.exit_price - self.entry_price) * self.quantity

    @property
    def costs(self) -> float:
        return self.entry_charges.total + self.exit_charges.total

    @property
    def pnl(self) -> float:
        """P&L net of every charge on both legs."""
        return self.gross_pnl - self.costs

    @property
    def return_pct(self) -> float:
        """Net return on the capital committed at entry."""
        notional = abs(self.entry_price * self.quantity)
        if notional == 0:
            return 0.0
        return self.pnl / notional * 100

    @property
    def days_held(self) -> int:
        return max(0, (self.exit_date - self.entry_date).days)

    def to_dict(self) -> dict:
        return {
            "entry_date": self.entry_date.date().isoformat(),
            "entry_price": round(self.entry_price, 4),
            "exit_date": self.exit_date.date().isoformat(),
            "exit_price": round(self.exit_price, 4),
            "quantity": self.quantity,
            "side": self.side,
            "exit_reason": self.exit_reason,
            "gross_pnl": round(self.gross_pnl, 2),
            "costs": round(self.costs, 2),
            "pnl": round(self.pnl, 2),
            "return_pct": round(self.return_pct, 3),
            "days_held": self.days_held,
        }


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    trades: list[Trade]
    history: pd.DataFrame
    config: BacktestConfig
    period_label: str = ""

    # --- the two headline numbers in Tables 3-6 ------------------------
    @property
    def profit(self) -> float:
        """Total P&L over all closed trades, in account currency."""
        return float(sum(t.pnl for t in self.trades))

    @property
    def strike_rate(self) -> float:
        """Percentage of closed trades that were profitable."""
        if not self.trades:
            return float("nan")
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return 100.0 * wins / len(self.trades)

    # --- supporting statistics -----------------------------------------
    @property
    def gross_profit(self) -> float:
        """Total P&L before transaction costs."""
        return float(sum(t.gross_pnl for t in self.trades))

    @property
    def total_costs(self) -> float:
        return float(sum(t.costs for t in self.trades))

    @property
    def cost_drag_pct(self) -> float:
        """Share of gross profit consumed by charges."""
        gross = self.gross_profit
        if gross <= 0:
            return float("nan")
        return 100.0 * self.total_costs / gross

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def final_equity(self) -> float:
        if self.history.empty:
            return self.config.initial_cash
        return float(self.history["equity"].iloc[-1])

    @property
    def total_return_pct(self) -> float:
        start = self.config.initial_cash
        return (self.final_equity / start - 1.0) * 100 if start else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        if self.history.empty:
            return 0.0
        eq = self.history["equity"]
        peak = eq.cummax()
        return float(((eq - peak) / peak).min() * 100)

    @property
    def buy_and_hold_profit(self) -> float:
        if self.history.empty:
            return 0.0
        first = self.history["close"].iloc[0]
        last = self.history["close"].iloc[-1]
        return float((last - first) * self.config.quantity)

    @property
    def sharpe(self) -> float:
        if len(self.history) < 3:
            return float("nan")
        rets = self.history["equity"].pct_change().dropna()
        if rets.empty or rets.std() == 0:
            return float("nan")
        return float(np.sqrt(252) * rets.mean() / rets.std())

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        return float(np.mean(wins)) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.trades if t.pnl <= 0]
        return float(np.mean(losses)) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = -sum(t.pnl for t in self.trades if t.pnl < 0)
        if gross_loss == 0:
            return float("inf") if gross_win else float("nan")
        return gross_win / gross_loss

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def to_dict(self) -> dict:
        pf = self.profit_factor
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "period": self.period_label,
            "trades": self.n_trades,
            "strike_rate": None if np.isnan(self.strike_rate) else round(self.strike_rate, 4),
            "profit": round(self.profit, 2),
            "gross_profit": round(self.gross_profit, 2),
            "total_costs": round(self.total_costs, 2),
            "cost_drag_pct": (None if np.isnan(self.cost_drag_pct)
                              else round(self.cost_drag_pct, 2)),
            "buy_and_hold_profit": round(self.buy_and_hold_profit, 2),
            "total_return_pct": round(self.total_return_pct, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 3),
            "sharpe": None if np.isnan(self.sharpe) else round(self.sharpe, 3),
            "profit_factor": (None if np.isnan(pf) else "inf" if np.isinf(pf) else round(pf, 3)),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
        }


def _fill_price(price: float, side: str, slippage_bps: float) -> float:
    """Apply slippage against the trader."""
    if not slippage_bps:
        return price
    adj = price * slippage_bps / 10_000
    return price + adj if side == "buy" else price - adj


def run_backtest(df: pd.DataFrame, strategy: Strategy, cfg: BacktestConfig | None = None,
                 symbol: str = "", period_label: str = "") -> BacktestResult:
    """Run `strategy` over `df` and return trades, equity curve and statistics."""
    cfg = cfg or BacktestConfig()
    costs = get_cost_model(cfg.cost_model)
    if df.empty:
        return BacktestResult(symbol, strategy.name, [], pd.DataFrame(), cfg, period_label)

    signals = strategy.generate_signals(df).reindex(df.index).fillna(0).astype(int)

    cash = cfg.initial_cash
    position = 0                     # signed share count
    entry_price = 0.0
    entry_date = None
    trades: list[Trade] = []
    equity_rows: list[float] = []
    entry_charges = ChargeBreakdown()

    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    sig = signals.to_numpy()
    dates = df.index

    def close_position(i: int, price: float, reason: str) -> None:
        nonlocal cash, position, entry_price, entry_date, entry_charges
        side = "long" if position > 0 else "short"
        exit_side = "sell" if position > 0 else "buy"
        exit_price = _fill_price(price, exit_side, cfg.slippage_bps)
        qty = abs(position)

        exit_charges = costs.charges(exit_price, qty, exit_side.upper())
        cash += position * exit_price - exit_charges.total - cfg.commission
        trades.append(Trade(entry_date, entry_price, dates[i], exit_price, qty, side, reason,
                            entry_charges, exit_charges))
        position, entry_price, entry_date = 0, 0.0, None
        entry_charges = ChargeBreakdown()

    def open_position(i: int, price: float, direction: int) -> None:
        nonlocal cash, position, entry_price, entry_date, entry_charges
        entry_side = "buy" if direction > 0 else "sell"
        fill = _fill_price(price, entry_side, cfg.slippage_bps)
        qty = cfg.quantity * direction

        entry_charges = costs.charges(fill, abs(qty), entry_side.upper())
        cash -= qty * fill + entry_charges.total + cfg.commission
        position, entry_price, entry_date = qty, fill, dates[i]

    for i in range(len(df)):
        # 1. Protective exits, checked intrabar before any new signal.
        if position != 0 and cfg.stop_loss_pct:
            if position > 0:
                stop = entry_price * (1 - cfg.stop_loss_pct)
                if lows[i] <= stop:
                    # A bar that opened below the stop gapped through it: there
                    # was never a fill available at the stop level, so the
                    # realistic exit is the open.
                    fill = min(stop, opens[i]) if cfg.gap_through_stops else stop
                    close_position(i, fill, "stop_loss")
            else:
                stop = entry_price * (1 + cfg.stop_loss_pct)
                if highs[i] >= stop:
                    fill = max(stop, opens[i]) if cfg.gap_through_stops else stop
                    close_position(i, fill, "stop_loss")

        # Take-profits deliberately fill at the target even when the bar gaps
        # past it: crediting favourable gaps while charging adverse ones would
        # bias the test in the strategy's favour.
        if position != 0 and cfg.take_profit_pct:
            if position > 0:
                target = entry_price * (1 + cfg.take_profit_pct)
                if highs[i] >= target:
                    close_position(i, target, "take_profit")
            else:
                target = entry_price * (1 - cfg.take_profit_pct)
                if lows[i] <= target:
                    close_position(i, target, "take_profit")

        # 2. Act on the previous bar's signal at this bar's open.
        if i > 0:
            s = sig[i - 1]
            price = opens[i]
            if s == LONG:
                if position < 0:
                    close_position(i, price, "reverse")
                if position == 0:
                    open_position(i, price, 1)
            elif s == EXIT:
                if position > 0:
                    close_position(i, price, "signal")
                if position == 0 and cfg.allow_short:
                    open_position(i, price, -1)

        equity_rows.append(cash + position * closes[i])

    # 3. Close anything still open on the final bar, so every trade is realised.
    if position != 0:
        close_position(len(df) - 1, closes[-1], "end_of_test")
        equity_rows[-1] = cash

    history = pd.DataFrame({"close": closes, "signal": sig, "equity": equity_rows}, index=dates)
    return BacktestResult(symbol or "", strategy.name, trades, history, cfg, period_label)
