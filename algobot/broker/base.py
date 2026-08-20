"""Broker abstraction.

The live bot talks only to this interface, so the simulated paper account and
the real DhanHQ account are interchangeable at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    symbol: str
    quantity: int                 # signed: negative is short
    avg_price: float
    last_price: float = 0.0
    security_id: str = ""
    product_type: str = ""

    @property
    def is_open(self) -> bool:
        return self.quantity != 0

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def unrealised_pnl(self) -> float:
        return (self.last_price - self.avg_price) * self.quantity

    @property
    def unrealised_pct(self) -> float:
        if self.avg_price == 0:
            return 0.0
        direction = 1 if self.quantity >= 0 else -1
        return direction * (self.last_price / self.avg_price - 1.0)


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str                     # BUY | SELL
    quantity: int
    status: str
    price: float = 0.0
    order_type: str = "MARKET"
    submitted_at: datetime | None = None
    raw: dict | None = None


@dataclass
class AccountSnapshot:
    cash: float
    equity: float
    currency: str = "INR"
    buying_power: float = 0.0


class BrokerError(RuntimeError):
    """Raised when a broker call fails or is rejected."""


class Broker(ABC):
    """Minimum surface the trading loop needs."""

    name: str = "broker"
    currency: str = "INR"

    @abstractmethod
    def is_market_open(self) -> bool: ...

    @abstractmethod
    def last_price(self, symbol: str) -> float: ...

    @abstractmethod
    def get_position(self, symbol: str) -> Position | None: ...

    @abstractmethod
    def account(self) -> AccountSnapshot: ...

    @abstractmethod
    def submit_order(self, symbol: str, side: str, quantity: int,
                     order_type: str = "MARKET", price: float = 0.0) -> Order: ...

    def buy(self, symbol: str, quantity: int, **kw) -> Order:
        return self.submit_order(symbol, "BUY", quantity, **kw)

    def sell(self, symbol: str, quantity: int, **kw) -> Order:
        return self.submit_order(symbol, "SELL", quantity, **kw)

    def close_position(self, symbol: str) -> Order | None:
        """Flatten whatever is open in `symbol`."""
        pos = self.get_position(symbol)
        if pos is None or not pos.is_open:
            return None
        side = "SELL" if pos.quantity > 0 else "BUY"
        return self.submit_order(symbol, side, abs(pos.quantity))

    def history(self, symbol: str, lookback_days: int = 200):
        """Recent daily bars, used to build model features and signals."""
        raise NotImplementedError
