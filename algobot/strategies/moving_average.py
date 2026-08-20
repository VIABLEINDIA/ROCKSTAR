"""Moving Average Crossover (paper Table 3 / Figure 17)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Strategy, crossover


@dataclass
class MovingAverageCrossover(Strategy):
    """Buy when the fast SMA crosses above the slow SMA; exit on the reverse."""

    name: str = "ma"
    fast: int = 10
    slow: int = 30
    use_ema: bool = False

    def __post_init__(self):
        if self.fast >= self.slow:
            raise ValueError(f"fast ({self.fast}) must be shorter than slow ({self.slow})")

    @property
    def warmup(self) -> int:
        return self.slow

    def _ma(self, s: pd.Series, window: int) -> pd.Series:
        if self.use_ema:
            return s.ewm(span=window, adjust=False, min_periods=window).mean()
        return s.rolling(window, min_periods=window).mean()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        return crossover(self._ma(close, self.fast), self._ma(close, self.slow))
