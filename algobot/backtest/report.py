"""Section VI -- rendering the paper's result tables.

Reproduces Tables 3-6 (DURATION / STRIKE RATE / PROFIT EARNED) for each
strategy, and writes machine-readable JSON + per-trade CSV alongside them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..config import ARTIFACT_DIR, BacktestConfig, DataConfig
from ..data.loader import load_prices, slice_period
from ..strategies.registry import build_strategy, pretty, wrap_with_model
from .engine import BacktestResult, run_backtest

CURRENCY_SYMBOLS = {"INR": "Rs ", "USD": "$ ", "EUR": "EUR "}

log = logging.getLogger(__name__)


def enforce_out_of_sample(window: pd.DataFrame, model, allow_in_sample: bool = False):
    """Trim a backtest window to bars the model never trained on.

    A model-filtered backtest that overlaps the model's own training split is
    measuring memorisation, not skill -- the forest has already seen those
    closes and their outcomes. The model bundle records its test_start, so the
    window is cut there unless the caller explicitly opts out.
    """
    if model is None or window.empty:
        return window, None

    test_start = (getattr(model, "data_summary", {}) or {}).get("test_start")
    if not test_start:
        return window, None

    cutoff = pd.Timestamp(test_start)
    if window.index[0] >= cutoff:
        return window, None

    if allow_in_sample:
        log.warning(
            "Backtest window starts %s but the model trained through %s -- "
            "results before that date are IN-SAMPLE and not evidence of skill.",
            window.index[0].date(), cutoff.date(),
        )
        return window, None

    trimmed = window.loc[window.index >= cutoff]
    log.info("Trimmed backtest to the model's out-of-sample period (from %s): "
             "%d of %d bars", cutoff.date(), len(trimmed), len(window))
    return trimmed, cutoff


def money(value: float, currency: str = "INR") -> str:
    return f"{CURRENCY_SYMBOLS.get(currency, currency + ' ')}{value:,.2f}"


def format_strike(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.4f}%"


def strategy_table(results: list[BacktestResult], strategy: str, currency: str = "INR") -> str:
    """One strategy across durations -- the layout of Tables 3-6, plus costs.

    PROFIT EARNED is net of charges; GROSS is the figure the paper reports.
    """
    header = (f"{'DURATION':<10}{'STRIKE RATE':>14}{'PROFIT EARNED':>18}"
              f"{'GROSS':>16}{'CHARGES':>14}{'TRADES':>8}")
    lines = [f"{pretty(strategy)} Evaluation", "=" * len(header), header, "-" * len(header)]
    for r in results:
        if r.n_trades == 0:
            lines.append(f"{r.period_label:<10}{'NA':>14}{'NA':>18}{'NA':>16}{'NA':>14}{0:>8}")
            continue
        lines.append(
            f"{r.period_label:<10}{format_strike(r.strike_rate):>14}"
            f"{money(r.profit, currency):>18}"
            f"{money(r.gross_profit, currency):>16}"
            f"{money(-r.total_costs, currency):>14}{r.n_trades:>8}"
        )
    return "\n".join(lines)


def summary_table(results: list[BacktestResult], currency: str = "INR") -> str:
    """All strategies x durations in one comparison grid."""
    header = (f"{'STRATEGY':<30}{'DURATION':<10}{'STRIKE RATE':>14}{'PROFIT':>16}"
              f"{'B&H PROFIT':>16}{'MAX DD':>10}{'TRADES':>8}")
    lines = [header, "-" * len(header)]
    for r in results:
        profit = "NA" if r.n_trades == 0 else money(r.profit, currency)
        lines.append(
            f"{pretty(r.strategy)[:29]:<30}{r.period_label:<10}"
            f"{format_strike(r.strike_rate):>14}"
            f"{profit:>16}"
            f"{money(r.buy_and_hold_profit, currency):>16}"
            f"{r.max_drawdown_pct:>9.2f}%{r.n_trades:>8}"
        )
    return "\n".join(lines)


def cost_summary(results: list[BacktestResult], currency: str = "INR") -> str:
    """What the charges did to the headline numbers."""
    traded = [r for r in results if r.n_trades > 0]
    if not traded:
        return "No trades: nothing to charge."

    gross = sum(r.gross_profit for r in traded)
    charges = sum(r.total_costs for r in traded)
    net = sum(r.profit for r in traded)
    trades = sum(r.n_trades for r in traded)

    lines = [
        f"Transaction costs across {len(traded)} runs / {trades} trades",
        f"  gross P&L      {money(gross, currency):>16}",
        f"  charges        {money(-charges, currency):>16}",
        f"  net P&L        {money(net, currency):>16}",
        f"  cost per trade {money(charges / trades, currency):>16}",
    ]
    if gross > 0:
        lines.append(f"  charges ate    {100 * charges / gross:>15.1f}% of gross profit")
    profitable_gross = sum(1 for r in traded if r.gross_profit > 0)
    profitable_net = sum(1 for r in traded if r.profit > 0)
    lines.append(f"  profitable runs {profitable_gross} gross -> {profitable_net} net")
    return "\n".join(lines)


def run_suite(symbol: str, strategies: list[str], periods: list[str],
              data_cfg: DataConfig | None = None, bt_cfg: BacktestConfig | None = None,
              model=None, strategy_params: dict | None = None,
              save_plots: bool = True, out_dir: Path | None = None,
              allow_in_sample: bool = False) -> list[BacktestResult]:
    """Backtest every (strategy, period) pair and write the artifacts."""
    data_cfg = data_cfg or DataConfig(symbol=symbol)
    data_cfg.symbol = symbol
    bt_cfg = bt_cfg or BacktestConfig()
    strategy_params = strategy_params or {}
    out_dir = Path(out_dir or ARTIFACT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = load_prices(data_cfg)
    results: list[BacktestResult] = []

    for name in strategies:
        for period in periods:
            window = slice_period(prices, period)
            window, _ = enforce_out_of_sample(window, model, allow_in_sample)
            if window.empty:
                log.warning("No out-of-sample bars left for %s/%s -- skipping", name, period)
                continue
            strat = build_strategy(name, **strategy_params)
            label = name
            if model is not None:
                strat = wrap_with_model(strat, model, **strategy_params)
                label = f"{name}+rf"

            result = run_backtest(window, strat, bt_cfg, symbol, period)
            result.strategy = label
            results.append(result)

            if save_plots and not result.history.empty:
                from ..model.plots import plot_backtest

                plot_backtest(result, symbol, f"{label}_{period}",
                              out_dir / f"{symbol.upper()}_{label}_{period}.png",
                              bt_cfg.currency)
            if result.trades:
                result.trades_frame().to_csv(
                    out_dir / f"{symbol.upper()}_{label}_{period}_trades.csv", index=False
                )

    payload = {
        "symbol": symbol,
        "data": asdict(data_cfg),
        "backtest": asdict(bt_cfg),
        "results": [r.to_dict() for r in results],
    }
    (out_dir / f"{symbol.upper()}_backtest_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return results


def compare_with_model(symbol: str, strategies: list[str], model,
                       data_cfg: DataConfig | None = None,
                       bt_cfg: BacktestConfig | None = None,
                       strategy_params: dict | None = None,
                       period: str = "max") -> pd.DataFrame:
    """Section IV.F, tested honestly.

    Runs each strategy with and without the Random Forest filter over the
    *same* out-of-sample window, so the difference is attributable to the model
    rather than to a different test period.
    """
    data_cfg = data_cfg or DataConfig(symbol=symbol)
    data_cfg.symbol = symbol
    bt_cfg = bt_cfg or BacktestConfig()
    strategy_params = strategy_params or {}

    prices = load_prices(data_cfg)
    window, cutoff = enforce_out_of_sample(slice_period(prices, period), model)
    if window.empty:
        raise ValueError("No out-of-sample bars available for this model")

    rows = []
    for name in strategies:
        plain = run_backtest(window, build_strategy(name, **strategy_params),
                             bt_cfg, symbol, period)
        fused = run_backtest(
            window,
            wrap_with_model(build_strategy(name, **strategy_params), model, **strategy_params),
            bt_cfg, symbol, period,
        )
        rows.append(
            {
                "strategy": pretty(name),
                "profit": round(plain.profit, 2),
                "profit_rf": round(fused.profit, 2),
                "delta": round(fused.profit - plain.profit, 2),
                "strike": None if pd.isna(plain.strike_rate) else round(plain.strike_rate, 2),
                "strike_rf": None if pd.isna(fused.strike_rate) else round(fused.strike_rate, 2),
                "trades": plain.n_trades,
                "trades_rf": fused.n_trades,
            }
        )

    frame = pd.DataFrame(rows)
    frame.attrs["window"] = (str(window.index[0].date()), str(window.index[-1].date()))
    frame.attrs["cutoff"] = str(cutoff.date()) if cutoff is not None else None
    frame.attrs["buy_and_hold"] = round(
        float((window["Close"].iloc[-1] - window["Close"].iloc[0]) * bt_cfg.quantity), 2
    )
    return frame


def results_frame(results: list[BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results])
