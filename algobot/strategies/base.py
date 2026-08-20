"""Strategy interface.

Every strategy turns an OHLCV frame into a signal series aligned to its index:

    +1  enter / stay long
    -1  exit long (or enter short when shorting is enabled)
     0  no opinion -- hold whatever position is open

Signals are computed from data available *up to and including* each bar; the
backtest engine executes them on the next bar's open to avoid look-ahead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

import pandas as pd

LONG, FLAT, EXIT = 1, 0, -1


@dataclass
class Strategy(ABC):
    """Base class: subclasses declare their parameters as dataclass fields."""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return an int series of {-1, 0, +1} indexed like `df`."""

    @property
    def warmup(self) -> int:
        """Bars needed before signals are meaningful."""
        return 0

    def params(self) -> dict:
        d = asdict(self)
        d.pop("name", None)
        return d

    def describe(self) -> str:
        kv = ", ".join(f"{k}={v}" for k, v in self.params().items())
        return f"{self.name}({kv})" if kv else self.name


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """+1 where fast crosses above slow, -1 where it crosses below, else 0."""
    above = fast > slow
    cross_up = above & ~above.shift(1, fill_value=False)
    cross_down = ~above & above.shift(1, fill_value=False)
    sig = pd.Series(0, index=fast.index, dtype=int)
    sig[cross_up] = LONG
    sig[cross_down] = EXIT
    # Neither series is defined during warm-up, so suppress those bars.
    sig[fast.isna() | slow.isna()] = FLAT
    return sig
