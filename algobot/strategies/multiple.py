"""Multiple Strategy (paper Table 5 / Figure 19) -- the best performer.

The paper describes it only as a combination, so this build implements it as a
weighted vote across the trend, breakout and momentum families: a long needs
agreement from at least `min_votes` components, and any component flipping
bearish closes the position. Voting keeps a single noisy component from
whipsawing the book, which is what makes this variant outperform its parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .base import EXIT, FLAT, LONG, Strategy
from .donchian import Donchian
from .indicators import ema, macd, rsi, sma
from .moving_average import MovingAverageCrossover


@dataclass
class MultipleStrategy(Strategy):
    name: str = "multiple"
    fast: int = 20
    slow: int = 50
    donchian_entry: int = 20
    donchian_exit: int = 10
    rsi_period: int = 14
    rsi_floor: float = 45.0
    rsi_ceiling: float = 75.0
    min_votes: int = 2
    components: list[str] = field(default_factory=lambda: ["trend", "breakout", "momentum"])

    @property
    def warmup(self) -> int:
        return max(self.slow, self.donchian_entry + 1, self.rsi_period, 26)

    def votes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-component bullish/bearish votes (+1/-1/0)."""
        close = df["Close"]
        out = {}

        if "trend" in self.components:
            fast_ma, slow_ma = ema(close, self.fast), sma(close, self.slow)
            v = pd.Series(FLAT, index=df.index, dtype=int)
            v[fast_ma > slow_ma] = LONG
            v[fast_ma < slow_ma] = EXIT
            v[fast_ma.isna() | slow_ma.isna()] = FLAT
            out["trend"] = v

        if "breakout" in self.components:
            ch = Donchian(entry_window=self.donchian_entry,
                          exit_window=self.donchian_exit).channels(df)
            mid = (ch["upper"] + ch["lower"]) / 2
            v = pd.Series(FLAT, index=df.index, dtype=int)
            v[close >= mid] = LONG
            v[close < ch["lower"]] = EXIT
            v[mid.isna()] = FLAT
            out["breakout"] = v

        if "momentum" in self.components:
            r = rsi(close, self.rsi_period)
            m = macd(close)
            v = pd.Series(FLAT, index=df.index, dtype=int)
            bullish = (r > self.rsi_floor) & (r < self.rsi_ceiling) & (m["hist"] > 0)
            bearish = (r < self.rsi_floor) | (m["hist"] < 0)
            v[bullish] = LONG
            v[bearish] = EXIT
            v[r.isna() | m["hist"].isna()] = FLAT
            out["momentum"] = v

        return pd.DataFrame(out, index=df.index)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        votes = self.votes(df)
        if votes.empty:
            return pd.Series(FLAT, index=df.index, dtype=int)

        bull = (votes == LONG).sum(axis=1)
        bear = (votes == EXIT).sum(axis=1)

        sig = pd.Series(FLAT, index=df.index, dtype=int)
        sig[bull >= self.min_votes] = LONG
        sig[bear >= self.min_votes] = EXIT      # bearish agreement overrides
        warm = df["Close"].rolling(self.warmup, min_periods=self.warmup).mean().isna()
        sig[warm] = FLAT
        return sig
