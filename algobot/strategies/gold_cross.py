"""Golden Cross / Death Cross (paper Table 6 / Figure 20).

The 200-day slow average needs roughly a year of history before it produces
anything, which is why the paper reports NA for the 1-year test.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Strategy, crossover


@dataclass
class GoldCross(Strategy):
    name: str = "gold"
    fast: int = 50
    slow: int = 200

    @property
    def warmup(self) -> int:
        return self.slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        fast = close.rolling(self.fast, min_periods=self.fast).mean()
        slow = close.rolling(self.slow, min_periods=self.slow).mean()
        return crossover(fast, slow)
