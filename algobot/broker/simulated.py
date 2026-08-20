"""Simulated paper-trading account.

Section IV.F has the bot trading a paper account. Dhan does not expose a paper
endpoint, so this broker fills the role: identical interface, real market
prices (Dhan quotes when credentials exist, otherwise the cached/Yahoo daily
series), and a local cash ledger. Nothing it does leaves the machine.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import CACHE_DIR, DataConfig, DhanCredentials
from ..data.loader import load_prices
from .base import AccountSnapshot, Broker, Order, Position
from .dhan import is_nse_session, now_ist

log = logging.getLogger(__name__)


class SimulatedBroker(Broker):
    """Paper account with a persisted ledger."""

    name = "sim"
    currency = "INR"

    def __init__(self, cash: float = 100_000.0, state_file: str | Path | None = None,
                 price_source: str = "auto", force_market_open: bool = False,
                 commission: float = 0.0, data_cfg: DataConfig | None = None):
        self.state_file = Path(state_file or CACHE_DIR / "paper_account.json")
        self.price_source = price_source        # auto | dhan | history
        self.force_market_open = force_market_open
        self.commission = commission
        self.data_cfg = data_cfg or DataConfig()
        self._history_cache: dict[str, pd.DataFrame] = {}
        self._order_seq = 0

        self.cash = cash
        self.positions: dict[str, Position] = {}
        self.fills: list[dict] = []
        self.realised_pnl = 0.0
        self._load_state()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read paper ledger %s (%s); starting fresh",
                        self.state_file, exc)
            return
        self.cash = state.get("cash", self.cash)
        self.realised_pnl = state.get("realised_pnl", 0.0)
        self.fills = state.get("fills", [])
        self._order_seq = state.get("order_seq", 0)
        self.positions = {
            sym: Position(**p) for sym, p in (state.get("positions") or {}).items()
        }

    def save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(
                {
                    "cash": self.cash,
                    "realised_pnl": self.realised_pnl,
                    "order_seq": self._order_seq,
                    "fills": self.fills[-500:],
                    "positions": {s: vars(p) for s, p in self.positions.items() if p.is_open},
                    "updated_at": now_ist().isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def reset(self, cash: float = 100_000.0) -> None:
        self.cash, self.positions, self.fills, self.realised_pnl = cash, {}, [], 0.0
        self._order_seq = 0
        self.save_state()

    # ------------------------------------------------------------------
    # market data
    # ------------------------------------------------------------------
    def is_market_open(self) -> bool:
        return True if self.force_market_open else is_nse_session()

    def history(self, symbol: str, lookback_days: int = 400) -> pd.DataFrame:
        if symbol not in self._history_cache:
            cfg = DataConfig(**{**vars(self.data_cfg), "symbol": symbol})
            self._history_cache[symbol] = load_prices(cfg)
        df = self._history_cache[symbol]
        return df.tail(max(lookback_days, 1))

    def last_price(self, symbol: str) -> float:
        """Live LTP when Dhan credentials exist, else the latest daily close."""
        if self.price_source in ("auto", "dhan"):
            creds = DhanCredentials()
            if creds.configured:
                try:
                    from .dhan import DhanBroker

                    return DhanBroker(creds, dry_run=True).last_price(symbol)
                except Exception as exc:
                    log.debug("Dhan LTP unavailable (%s); using last close", exc)
            elif self.price_source == "dhan":
                raise RuntimeError("price_source='dhan' but Dhan credentials are not set")

        return float(self.history(symbol)["Close"].iloc[-1])

    def refresh_marks(self) -> None:
        for symbol, pos in self.positions.items():
            if pos.is_open:
                pos.last_price = self.last_price(symbol)

    # ------------------------------------------------------------------
    # account / orders
    # ------------------------------------------------------------------
    def account(self) -> AccountSnapshot:
        self.refresh_marks()
        market_value = sum(p.market_value for p in self.positions.values() if p.is_open)
        return AccountSnapshot(
            cash=self.cash,
            equity=self.cash + market_value,
            currency=self.currency,
            buying_power=self.cash,
        )

    def get_position(self, symbol: str) -> Position | None:
        pos = self.positions.get(symbol.upper())
        if pos is None or not pos.is_open:
            return None
        try:
            pos.last_price = self.last_price(symbol)
        except Exception:  # keep the last known mark if the quote fails
            pass
        return pos

    def submit_order(self, symbol: str, side: str, quantity: int,
                     order_type: str = "MARKET", price: float = 0.0) -> Order:
        symbol, side = symbol.upper(), side.upper()
        if quantity <= 0:
            raise ValueError("quantity must be positive")

        fill_price = float(price) if (order_type.upper() == "LIMIT" and price) else self.last_price(symbol)
        signed = quantity if side == "BUY" else -quantity

        pos = self.positions.get(symbol) or Position(symbol=symbol, quantity=0, avg_price=0.0)
        new_qty = pos.quantity + signed

        if pos.quantity != 0 and (pos.quantity > 0) != (signed > 0):
            # Reducing or flipping: realise P&L on the closed portion.
            closed = min(abs(signed), abs(pos.quantity))
            direction = 1 if pos.quantity > 0 else -1
            self.realised_pnl += direction * (fill_price - pos.avg_price) * closed
        if new_qty != 0 and (pos.quantity == 0 or (pos.quantity > 0) == (signed > 0)):
            # Adding to a position: weighted-average the entry price.
            pos.avg_price = (
                (pos.avg_price * abs(pos.quantity)) + (fill_price * abs(signed))
            ) / abs(new_qty)
        elif new_qty == 0:
            pos.avg_price = 0.0

        pos.quantity = new_qty
        pos.last_price = fill_price
        self.positions[symbol] = pos
        self.cash -= signed * fill_price + self.commission

        self._order_seq += 1
        order = Order(
            order_id=f"SIM-{self._order_seq:06d}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            status="FILLED",
            price=fill_price,
            order_type=order_type,
            submitted_at=now_ist(),
        )
        self.fills.append(
            {
                "order_id": order.order_id,
                "time": order.submitted_at.isoformat(),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": round(fill_price, 4),
                "cash_after": round(self.cash, 2),
                "realised_pnl": round(self.realised_pnl, 2),
            }
        )
        self.save_state()
        log.info("[SIM] %s %d %s @ %.2f (cash %.2f)", side, quantity, symbol, fill_price, self.cash)
        return order

    def fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.fills)


class ReplayBroker(SimulatedBroker):
    """Simulated broker that walks a historical series bar by bar.

    Lets the full live loop -- signals, stops, order placement -- be exercised
    end to end without waiting for a real session.
    """

    name = "replay"

    def __init__(self, prices: pd.DataFrame, symbol: str, start_index: int = 200, **kw):
        super().__init__(**kw)
        self.prices = prices
        self.symbol = symbol.upper()
        self.i = min(start_index, len(prices) - 1)
        self.force_market_open = True

    def step(self) -> bool:
        """Advance one bar. Returns False when the series is exhausted."""
        if self.i >= len(self.prices) - 1:
            return False
        self.i += 1
        return True

    @property
    def current_time(self) -> datetime:
        return self.prices.index[self.i].to_pydatetime()

    def is_market_open(self) -> bool:
        return self.i < len(self.prices) - 1

    def history(self, symbol: str, lookback_days: int = 400) -> pd.DataFrame:
        window = self.prices.iloc[: self.i + 1]
        return window.tail(max(lookback_days, 1))

    def last_price(self, symbol: str) -> float:
        return float(self.prices["Close"].iloc[self.i])
