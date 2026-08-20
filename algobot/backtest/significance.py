"""Is the model's filtering skill, or just reduced exposure?

A veto filter removes entries. In a falling market, removing entries raises P&L
whether or not the vetoes were chosen intelligently -- so "strategy + RF beat
strategy" is not by itself evidence that the forecast has any skill.

This module runs the null hypothesis: veto the *same number* of entries at
random, many times, and see where the model's result falls in that
distribution. A model with genuine skill should land in the top tail. A
percentile near the middle means the gain came from trading less, not from
choosing better.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..strategies.base import FLAT, LONG, Strategy
from ..strategies.registry import build_strategy, wrap_with_model
from .engine import run_backtest

log = logging.getLogger(__name__)


class RandomVetoStrategy(Strategy):
    """Base strategy with a random subset of its long entries suppressed."""

    def __init__(self, base: Strategy, n_vetoes: int, seed: int):
        self.name = f"{base.name}+random"
        self.base = base
        self.n_vetoes = n_vetoes
        self.seed = seed

    @property
    def warmup(self) -> int:
        return self.base.warmup

    def params(self) -> dict:
        return {"base": self.base.describe(), "n_vetoes": self.n_vetoes, "seed": self.seed}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sig = self.base.generate_signals(df)
        entries = np.flatnonzero((sig == LONG).to_numpy())
        if self.n_vetoes <= 0 or len(entries) == 0:
            return sig

        rng = np.random.default_rng(self.seed)
        chosen = rng.choice(entries, size=min(self.n_vetoes, len(entries)), replace=False)
        out = sig.copy()
        out.iloc[chosen] = FLAT
        return out


def permutation_test(df: pd.DataFrame, strategy_name: str, model,
                     cfg: BacktestConfig | None = None, n_trials: int = 200,
                     strategy_params: dict | None = None, symbol: str = "") -> dict:
    """Compare the model filter against random filters of equal strength."""
    cfg = cfg or BacktestConfig()
    strategy_params = strategy_params or {}

    base = build_strategy(strategy_name, **strategy_params)
    fused = wrap_with_model(build_strategy(strategy_name, **strategy_params), model,
                            **strategy_params)

    base_sig = base.generate_signals(df)
    fused_sig = fused.generate_signals(df)
    n_entries = int((base_sig == LONG).sum())
    n_vetoes = int(((base_sig == LONG) & (fused_sig != LONG)).sum())

    plain = run_backtest(df, base, cfg, symbol, "test")
    model_run = run_backtest(df, fused, cfg, symbol, "test")

    null_profits = []
    for seed in range(n_trials):
        random_strategy = RandomVetoStrategy(
            build_strategy(strategy_name, **strategy_params), n_vetoes, seed
        )
        null_profits.append(run_backtest(df, random_strategy, cfg, symbol, "test").profit)

    null = np.asarray(null_profits, dtype=float)
    # Share of random filters the model beat -- its percentile in the null.
    percentile = float((null < model_run.profit).mean() * 100) if len(null) else float("nan")
    # One-sided p-value: how often chance alone matches the model.
    p_value = float((null >= model_run.profit).mean()) if len(null) else float("nan")

    return {
        "strategy": strategy_name,
        "entries": n_entries,
        "vetoed": n_vetoes,
        "veto_pct": round(100 * n_vetoes / n_entries, 1) if n_entries else 0.0,
        "profit_base": round(plain.profit, 2),
        "profit_model": round(model_run.profit, 2),
        "null_mean": round(float(null.mean()), 2) if len(null) else None,
        "null_std": round(float(null.std()), 2) if len(null) else None,
        "percentile": round(percentile, 1),
        "p_value": round(p_value, 4),
        "skilled": bool(p_value < 0.05),
    }


def format_report(rows: list[dict], symbol: str, currency: str = "INR") -> str:
    from .report import money

    header = (f"{'STRATEGY':<12}{'VETOED':>10}{'BASE P&L':>14}{'MODEL P&L':>14}"
              f"{'RANDOM MEAN':>14}{'PCTILE':>9}{'p':>8}")
    lines = [
        f"Permutation test -- {symbol}: does the RF filter beat random vetoes of equal size?",
        header,
        "-" * len(header),
    ]
    for r in rows:
        lines.append(
            f"{r['strategy']:<12}"
            f"{str(r['vetoed']) + '/' + str(r['entries']):>10}"
            f"{money(r['profit_base'], currency):>14}"
            f"{money(r['profit_model'], currency):>14}"
            f"{money(r['null_mean'], currency) if r['null_mean'] is not None else 'NA':>14}"
            f"{r['percentile']:>8.1f}%{r['p_value']:>8.3f}"
        )
    skilled = sum(1 for r in rows if r["skilled"])
    lines.append("")
    lines.append(f"Beat random vetoing at p < 0.05 in {skilled} of {len(rows)} strategies.")
    return "\n".join(lines)
