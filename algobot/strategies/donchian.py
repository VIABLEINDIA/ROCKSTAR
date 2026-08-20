"""Donchian Channel breakout (paper Table 4 / Figure 18)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import EXIT, FLAT, LONG, Strategy


@dataclass
class Donchian(Strategy):
    """Buy a breakout above the N-bar high; exit on a break of the M-bar low."""

    name: str = "donchian"
    entry_window: int = 20
    exit_window: int = 10

    @property
    def warmup(self) -> int:
        return max(self.entry_window, self.exit_window) + 1

    def channels(self, df: pd.DataFrame) -> pd.DataFrame:
        high, low = df["High"], df["Low"]
        # shift(1): the channel must exclude the current bar, otherwise today's
        # high is compared against itself and every bar looks like a breakout.
        return pd.DataFrame(
            {
                "upper": high.rolling(self.entry_window, min_periods=self.entry_window)
                .max()
                .shift(1),
                "lower": low.rolling(self.exit_window, min_periods=self.exit_window)
                .min()
                .shift(1),
            },
            index=df.index,
        )

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ch = self.channels(df)
        close = df["Close"]
        sig = pd.Series(FLAT, index=df.index, dtype=int)
        sig[close > ch["upper"]] = LONG
        sig[close < ch["lower"]] = EXIT
        sig[ch["upper"].isna() | ch["lower"].isna()] = FLAT
        return sig
