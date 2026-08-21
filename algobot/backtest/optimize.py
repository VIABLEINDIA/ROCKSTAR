"""Parameter search, with the guard rails that make the result mean something.

Grid-searching parameters and reporting the best score is the single easiest
way to manufacture a backtest that looks excellent and trades badly: with
enough combinations, something always fits the noise. Every function here
therefore separates the data used to *choose* parameters from the data used to
*judge* them.

  * `grid_search`      -- rank combinations on one window (in-sample only).
  * `train_test_optimize` -- choose on the first slice, report on the untouched
    remainder, so in-sample and out-of-sample sit side by side.
  * `walk_forward`     -- re-optimise at each fold and trade the next one,
    concatenating results the optimiser never saw. This is the honest number.
  * `random_parameter_baseline` -- what a randomly chosen parameter set scores
    out-of-sample. If tuning cannot beat this, the search found noise.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..config import BacktestConfig
from ..strategies.registry import build_strategy
from .engine import BacktestResult, run_backtest

log = logging.getLogger(__name__)


# Grids are deliberately coarse. A fine grid over a short history is how
# overfitting happens; these span meaningfully different behaviours instead.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "ma": {
        "fast": [5, 10, 20, 30, 50],
        "slow": [30, 50, 100, 150, 200],
        "use_ema": [False, True],
    },
    "donchian": {
        "entry_window": [10, 20, 30, 55],
        "exit_window": [5, 10, 20, 30],
    },
    "multiple": {
        "fast": [10, 20, 50],
        "slow": [50, 100, 200],
        "min_votes": [1, 2, 3],
    },
    "gold": {
        "fast": [20, 50],
        "slow": [100, 200],
    },
}

# Intraday grids are expressed in *bars*, not days. On NSE hourly data a
# session is 7 bars, so these span roughly half a session to two weeks --
# the daily grids (up to a 200-bar slow average) would be a 29-session
# average on hourly data, which is not an intraday strategy at all.
INTRADAY_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "ma": {
        "fast": [3, 5, 7, 10, 14],
        "slow": [14, 21, 35, 70],
        "use_ema": [False, True],
    },
    "donchian": {
        "entry_window": [7, 14, 21, 35],
        "exit_window": [3, 7, 14],
    },
    "multiple": {
        "fast": [5, 7, 14],
        "slow": [21, 35, 70],
        "min_votes": [1, 2, 3],
    },
    "gold": {
        "fast": [7, 14],
        "slow": [35, 70],
    },
}

# Intraday stops are necessarily tighter: a 12% stop cannot be reached inside
# a session, so it is the same as having no stop at all.
INTRADAY_STOP_LOSS_GRID = [0.005, 0.01, 0.02, 0.03, None]
INTRADAY_TAKE_PROFIT_GRID = [0.005, 0.01, 0.02, 0.03, None]


def grids_for(intraday: bool) -> tuple[dict, list, list]:
    """Parameter grid, stop-loss grid and take-profit grid for the bar size."""
    if intraday:
        return INTRADAY_PARAM_GRIDS, INTRADAY_STOP_LOSS_GRID, INTRADAY_TAKE_PROFIT_GRID
    return PARAM_GRIDS, STOP_LOSS_GRID, TAKE_PROFIT_GRID


# Stop-loss lives on BacktestConfig rather than the strategy, but it drives
# results as hard as any strategy window, so it is searchable too.
STOP_LOSS_GRID = [0.03, 0.05, 0.08, 0.12, None]

# Take-profit is the lever that actually moves win rate: exiting into small
# gains converts many open trades into recorded wins. It does so by capping the
# winners, which is why win rate and profitability have to be read together.
TAKE_PROFIT_GRID = [0.02, 0.03, 0.05, 0.10, None]

OBJECTIVES = {
    "profit": lambda r: r.profit,
    "profit_factor": lambda r: (r.profit_factor if np.isfinite(r.profit_factor) else 0.0),
    "sharpe": lambda r: (r.sharpe if not np.isnan(r.sharpe) else -99.0),
    "return_per_trade": lambda r: (r.profit / r.n_trades if r.n_trades else -1e9),
    "win_rate": lambda r: (r.strike_rate if not np.isnan(r.strike_rate) else 0.0),
    # Win rate alone is trivially gamed by a tiny take-profit, so this variant
    # only counts win rate on parameter sets that are also profitable.
    "win_rate_profitable": lambda r: ((r.strike_rate if not np.isnan(r.strike_rate) else 0.0)
                                      if r.profit > 0 else 0.0),
}


def score(result: BacktestResult, objective: str = "profit", min_trades: int = 5) -> float:
    """Objective value, with a floor on trade count.

    A parameter set that takes two trades can post a spectacular profit factor
    on luck alone; requiring a minimum sample keeps those out of the ranking.
    """
    if result.n_trades < min_trades:
        return -np.inf
    return float(OBJECTIVES[objective](result))


def _combinations(grid: dict[str, list], search_stop_loss: bool,
                  search_take_profit: bool = False, intraday: bool = False) -> list[dict]:
    keys = list(grid)
    combos = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]

    # Strategy-specific validity: a fast window must be shorter than the slow one.
    combos = [c for c in combos
              if not ("fast" in c and "slow" in c and c["fast"] >= c["slow"])
              and not ("entry_window" in c and "exit_window" in c
                       and c["exit_window"] > c["entry_window"])]

    _, sl_grid, tp_grid = grids_for(intraday)
    if search_stop_loss:
        combos = [{**c, "stop_loss_pct": sl} for c in combos for sl in sl_grid]
    if search_take_profit:
        combos = [{**c, "take_profit_pct": tp} for c in combos for tp in tp_grid]
    return combos


def _apply(cfg: BacktestConfig, params: dict) -> tuple[BacktestConfig, dict]:
    """Split a combination into backtest config overrides and strategy params."""
    params = dict(params)
    if "stop_loss_pct" in params:
        cfg = replace(cfg, stop_loss_pct=params.pop("stop_loss_pct") or 0.0)
    if "take_profit_pct" in params:
        cfg = replace(cfg, take_profit_pct=params.pop("take_profit_pct"))
    return cfg, params


@dataclass
class SearchRow:
    params: dict
    result: BacktestResult
    objective: float

    def summary(self) -> dict:
        return {
            **self.params,
            "trades": self.result.n_trades,
            "profit": round(self.result.profit, 2),
            "strike_rate": (None if np.isnan(self.result.strike_rate)
                            else round(self.result.strike_rate, 2)),
            "objective": (None if not np.isfinite(self.objective)
                          else round(self.objective, 4)),
        }


def grid_search(df: pd.DataFrame, strategy: str, cfg: BacktestConfig | None = None,
                objective: str = "profit", grid: dict | None = None,
                search_stop_loss: bool = True, min_trades: int = 5,
                symbol: str = "", search_take_profit: bool = False,
                intraday: bool = False) -> list[SearchRow]:
    """Rank every combination on `df`. In-sample only -- never report this alone."""
    cfg = cfg or BacktestConfig()
    grid = grid or grids_for(intraday)[0][strategy]

    rows: list[SearchRow] = []
    for params in _combinations(grid, search_stop_loss, search_take_profit, intraday):
        run_cfg, strat_params = _apply(cfg, params)
        result = run_backtest(df, build_strategy(strategy, **strat_params), run_cfg,
                              symbol, "search")
        rows.append(SearchRow(params, result, score(result, objective, min_trades)))

    rows.sort(key=lambda r: r.objective, reverse=True)
    return rows


@dataclass
class OptimisationReport:
    strategy: str
    symbol: str
    objective: str
    best_params: dict
    in_sample: BacktestResult
    out_of_sample: BacktestResult
    baseline_out_of_sample: BacktestResult
    top_rows: list[SearchRow] = field(default_factory=list)
    robust_oos: float = float("nan")
    train_window: tuple[str, str] = ("", "")
    test_window: tuple[str, str] = ("", "")

    @property
    def improvement(self) -> float:
        """Out-of-sample gain over the default parameters -- the only number
        that counts."""
        return self.out_of_sample.profit - self.baseline_out_of_sample.profit

    @property
    def degradation(self) -> float:
        """How much of the in-sample edge survived. Below ~0 means it did not."""
        return self.out_of_sample.profit - self.in_sample.profit

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "objective": self.objective,
            "best_params": self.best_params,
            "train_window": list(self.train_window),
            "test_window": list(self.test_window),
            "in_sample_profit": round(self.in_sample.profit, 2),
            "out_of_sample_profit": round(self.out_of_sample.profit, 2),
            "baseline_out_of_sample_profit": round(self.baseline_out_of_sample.profit, 2),
            "improvement_vs_default": round(self.improvement, 2),
            "robust_top10_median_oos": (None if np.isnan(self.robust_oos)
                                        else round(self.robust_oos, 2)),
        }


def train_test_optimize(df: pd.DataFrame, strategy: str, cfg: BacktestConfig | None = None,
                        objective: str = "profit", train_ratio: float = 0.6,
                        search_stop_loss: bool = True, symbol: str = "",
                        top_n: int = 10, search_take_profit: bool = False) -> OptimisationReport:
    """Choose parameters on the first `train_ratio` of the data, judge on the rest."""
    cfg = cfg or BacktestConfig()
    cut = int(len(df) * train_ratio)
    train, test = df.iloc[:cut], df.iloc[cut:]
    if train.empty or test.empty:
        raise ValueError("Not enough data to split into train and test")

    rows = grid_search(train, strategy, cfg, objective, search_stop_loss=search_stop_loss,
                       symbol=symbol, search_take_profit=search_take_profit)
    if not rows or not np.isfinite(rows[0].objective):
        raise ValueError("No parameter set met the minimum trade count on the training window")

    best = rows[0]
    run_cfg, strat_params = _apply(cfg, best.params)
    oos = run_backtest(test, build_strategy(strategy, **strat_params), run_cfg, symbol, "oos")
    baseline = run_backtest(test, build_strategy(strategy), cfg, symbol, "oos-default")

    # Median out-of-sample profit of the top N in-sample sets. If the single
    # best is a fluke this exposes it -- a genuine edge shows up across
    # neighbouring parameters, not one lucky cell.
    top = [r for r in rows[:top_n] if np.isfinite(r.objective)]
    robust = []
    for row in top:
        rcfg, rparams = _apply(cfg, row.params)
        robust.append(run_backtest(test, build_strategy(strategy, **rparams), rcfg,
                                   symbol, "oos").profit)

    return OptimisationReport(
        strategy=strategy,
        symbol=symbol,
        objective=objective,
        best_params=best.params,
        in_sample=best.result,
        out_of_sample=oos,
        baseline_out_of_sample=baseline,
        top_rows=top,
        robust_oos=float(np.median(robust)) if robust else float("nan"),
        train_window=(str(train.index[0].date()), str(train.index[-1].date())),
        test_window=(str(test.index[0].date()), str(test.index[-1].date())),
    )


@dataclass
class WalkForwardReport:
    strategy: str
    symbol: str
    folds: list[dict]
    tuned_profit: float
    default_profit: float
    tuned_trades: int
    default_trades: int
    tuned_wins: int
    default_wins: int = 0

    @property
    def default_win_rate(self) -> float:
        return (100.0 * self.default_wins / self.default_trades
                if self.default_trades else float("nan"))

    @property
    def improvement(self) -> float:
        return self.tuned_profit - self.default_profit

    @property
    def win_rate(self) -> float:
        return 100.0 * self.tuned_wins / self.tuned_trades if self.tuned_trades else float("nan")

    @property
    def folds_improved(self) -> int:
        return sum(1 for f in self.folds if f["tuned_profit"] > f["default_profit"])

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "folds": len(self.folds),
            "folds_improved": self.folds_improved,
            "tuned_profit": round(self.tuned_profit, 2),
            "default_profit": round(self.default_profit, 2),
            "improvement": round(self.improvement, 2),
            "tuned_trades": self.tuned_trades,
            "win_rate": (None if np.isnan(self.win_rate) else round(self.win_rate, 2)),
            "fold_detail": self.folds,
        }


def walk_forward(df: pd.DataFrame, strategy: str, cfg: BacktestConfig | None = None,
                 objective: str = "profit", n_folds: int = 4, train_ratio: float = 0.5,
                 search_stop_loss: bool = True, symbol: str = "",
                 search_take_profit: bool = False,
                 intraday: bool = False) -> WalkForwardReport:
    """Re-optimise at each fold and trade the next one.

    Every reported trade comes from parameters fitted only on data before it,
    which is the closest a backtest gets to what live tuning would have
    produced.
    """
    cfg = cfg or BacktestConfig()
    n = len(df)
    start = int(n * train_ratio)
    if start < 100 or n - start < n_folds * 20:
        raise ValueError("Not enough bars for a walk-forward with these settings")

    edges = np.linspace(start, n, n_folds + 1).astype(int)
    folds, tuned_trades, default_trades, tuned_wins = [], 0, 0, 0
    default_wins = 0
    tuned_total = default_total = 0.0

    for i in range(n_folds):
        train = df.iloc[: edges[i]]
        test = df.iloc[edges[i]: edges[i + 1]]
        if test.empty:
            continue

        rows = grid_search(train, strategy, cfg, objective,
                           search_stop_loss=search_stop_loss, symbol=symbol,
                           search_take_profit=search_take_profit, intraday=intraday)
        rows = [r for r in rows if np.isfinite(r.objective)]
        if not rows:
            log.warning("Fold %d: no parameter set met the trade floor; using defaults", i + 1)
            params = {}
        else:
            params = rows[0].params

        run_cfg, strat_params = _apply(cfg, params)
        tuned = run_backtest(test, build_strategy(strategy, **strat_params), run_cfg,
                             symbol, f"fold{i+1}")
        default = run_backtest(test, build_strategy(strategy), cfg, symbol, f"fold{i+1}")

        tuned_total += tuned.profit
        default_total += default.profit
        tuned_trades += tuned.n_trades
        default_trades += default.n_trades
        tuned_wins += sum(1 for t in tuned.trades if t.pnl > 0)
        default_wins += sum(1 for t in default.trades if t.pnl > 0)

        folds.append({
            "fold": i + 1,
            "train_end": str(train.index[-1].date()),
            "test": [str(test.index[0].date()), str(test.index[-1].date())],
            "params": params,
            "tuned_profit": round(tuned.profit, 2),
            "default_profit": round(default.profit, 2),
            "tuned_trades": tuned.n_trades,
            "tuned_win_rate": (None if not tuned.trades else
                               round(100 * sum(1 for t in tuned.trades if t.pnl > 0)
                                     / len(tuned.trades), 2)),
        })

    return WalkForwardReport(strategy, symbol, folds, tuned_total, default_total,
                             tuned_trades, default_trades, tuned_wins, default_wins)


def random_parameter_baseline(df_train: pd.DataFrame, df_test: pd.DataFrame, strategy: str,
                              cfg: BacktestConfig | None = None, n_trials: int = 50,
                              search_stop_loss: bool = True, seed: int = 0,
                              symbol: str = "") -> dict:
    """Out-of-sample distribution of randomly chosen parameters.

    The comparison that decides whether tuning did anything: if the optimised
    set lands mid-distribution, the search bought nothing that picking blind
    would not have.
    """
    cfg = cfg or BacktestConfig()
    combos = _combinations(PARAM_GRIDS[strategy], search_stop_loss)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(combos), size=min(n_trials, len(combos)), replace=False)

    profits = []
    for idx in picks:
        run_cfg, strat_params = _apply(cfg, combos[idx])
        profits.append(run_backtest(df_test, build_strategy(strategy, **strat_params),
                                    run_cfg, symbol, "random").profit)

    arr = np.asarray(profits, dtype=float)
    return {"n": len(arr), "mean": float(arr.mean()), "median": float(np.median(arr)),
            "std": float(arr.std()), "best": float(arr.max()), "worst": float(arr.min()),
            "profits": arr}
