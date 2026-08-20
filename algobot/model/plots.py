"""Figures: Section IV.E's actual-vs-predicted chart, feature importances,
and the equity/signal chart used for the backtest figures (17-20)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd

from ..config import ARTIFACT_DIR

# Paper's convention: red = actual movement, blue = bot prediction.
ACTUAL_COLOR = "#d62728"
PREDICTED_COLOR = "#1f77b4"


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_actual_vs_predicted(df: pd.DataFrame, symbol: str, path: str | Path | None = None,
                             currency: str = "INR") -> Path:
    """Figure 16 -- Share Price vs Date (red actual, blue predicted)."""
    path = path or ARTIFACT_DIR / f"{symbol.upper()}_actual_vs_predicted.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["actual"], color=ACTUAL_COLOR, lw=1.2, label="Actual close")
    ax.plot(df.index, df["predicted"], color=PREDICTED_COLOR, lw=1.2, alpha=0.85,
            label="Predicted close")
    ax.set_title(f"{symbol.upper()} -- Share Price vs Date (Random Forest Regressor)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Share price ({currency})")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    return _save(fig, path)


def plot_feature_importances(importances: pd.Series, symbol: str,
                             path: str | Path | None = None, top: int = 20) -> Path:
    """Section IV.C -- `regressor.feature_importances_`."""
    path = path or ARTIFACT_DIR / f"{symbol.upper()}_feature_importances.png"
    top_n = importances.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(top_n))))
    ax.barh(top_n.index, top_n.to_numpy(), color=PREDICTED_COLOR)
    ax.set_title(f"{symbol.upper()} -- Random Forest feature importances (top {len(top_n)})")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, path)


def plot_backtest(result, symbol: str, strategy: str, path: str | Path | None = None,
                  currency: str = "INR") -> Path:
    """Figures 17-20 -- price with entry/exit markers above the equity curve."""
    path = path or ARTIFACT_DIR / f"{symbol.upper()}_{strategy}_backtest.png"
    hist = result.history

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax1.plot(hist.index, hist["close"], color="#333333", lw=1.1, label="Close")

    if result.trades:
        entries = pd.Series({t.entry_date: t.entry_price for t in result.trades})
        exits = pd.Series({t.exit_date: t.exit_price for t in result.trades})
        ax1.scatter(entries.index, entries.to_numpy(), marker="^", s=48,
                    color="#2ca02c", zorder=3, label="Buy")
        ax1.scatter(exits.index, exits.to_numpy(), marker="v", s=48,
                    color=ACTUAL_COLOR, zorder=3, label="Sell")

    ax1.set_title(f"{symbol.upper()} -- {strategy} backtest "
                  f"(strike rate {result.strike_rate:.2f}%, P&L {currency} {result.profit:,.2f})")
    ax1.set_ylabel(f"Price ({currency})")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.25)

    ax2.plot(hist.index, hist["equity"], color=PREDICTED_COLOR, lw=1.2)
    ax2.axhline(hist["equity"].iloc[0], color="#888888", ls="--", lw=0.8)
    ax2.set_ylabel(f"Equity ({currency})")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.25)
    return _save(fig, path)
