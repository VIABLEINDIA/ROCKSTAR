"""Parameter search, and the guard rails that stop it lying.

The point of these tests is less "does the optimiser find good parameters" than
"does it refuse to pretend an in-sample fit is evidence".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algobot.backtest.optimize import (
    OBJECTIVES,
    PARAM_GRIDS,
    STOP_LOSS_GRID,
    TAKE_PROFIT_GRID,
    _combinations,
    grid_search,
    random_parameter_baseline,
    score,
    train_test_optimize,
    walk_forward,
)
from algobot.config import BacktestConfig
from algobot.data.loader import load_synthetic


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return load_synthetic("OPT", days=1200)


# ----------------------------------------------------------------------
# grid construction
# ----------------------------------------------------------------------
def test_every_registered_strategy_has_a_grid():
    from algobot.strategies.registry import PAPER_STRATEGIES

    for name in PAPER_STRATEGIES:
        assert name in PARAM_GRIDS


def test_combinations_reject_fast_slower_than_slow():
    combos = _combinations(PARAM_GRIDS["ma"], search_stop_loss=False)
    assert combos
    assert all(c["fast"] < c["slow"] for c in combos)


def test_combinations_reject_exit_wider_than_entry():
    combos = _combinations(PARAM_GRIDS["donchian"], search_stop_loss=False)
    assert all(c["exit_window"] <= c["entry_window"] for c in combos)


def test_stop_loss_multiplies_the_grid():
    plain = _combinations(PARAM_GRIDS["gold"], search_stop_loss=False)
    with_sl = _combinations(PARAM_GRIDS["gold"], search_stop_loss=True)
    assert len(with_sl) == len(plain) * len(STOP_LOSS_GRID)


def test_take_profit_multiplies_the_grid():
    plain = _combinations(PARAM_GRIDS["gold"], search_stop_loss=False)
    with_tp = _combinations(PARAM_GRIDS["gold"], search_stop_loss=False,
                            search_take_profit=True)
    assert len(with_tp) == len(plain) * len(TAKE_PROFIT_GRID)


# ----------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------
def test_thin_samples_are_disqualified(prices):
    """A 2-trade parameter set must never win the ranking on luck."""
    from algobot.backtest.engine import run_backtest
    from algobot.strategies.registry import build_strategy

    r = run_backtest(prices.head(300), build_strategy("gold"), BacktestConfig())
    if r.n_trades < 5:
        assert score(r, "profit", min_trades=5) == -np.inf


def test_every_objective_is_callable(prices):
    from algobot.backtest.engine import run_backtest
    from algobot.strategies.registry import build_strategy

    r = run_backtest(prices, build_strategy("ma"), BacktestConfig(), "OPT", "t")
    for name in OBJECTIVES:
        value = score(r, name, min_trades=0)
        assert isinstance(value, float)


def test_win_rate_profitable_zeroes_out_losing_sets(prices):
    from algobot.backtest.engine import run_backtest
    from algobot.strategies.registry import build_strategy

    r = run_backtest(prices, build_strategy("multiple"), BacktestConfig(), "OPT", "t")
    if r.profit <= 0:
        assert score(r, "win_rate_profitable", min_trades=0) == 0.0


# ----------------------------------------------------------------------
# grid search
# ----------------------------------------------------------------------
def test_grid_search_is_ranked_best_first(prices):
    rows = grid_search(prices, "gold", BacktestConfig(), "profit", search_stop_loss=False)
    objectives = [r.objective for r in rows if np.isfinite(r.objective)]
    assert objectives == sorted(objectives, reverse=True)


def test_grid_search_covers_the_whole_grid(prices):
    rows = grid_search(prices, "gold", BacktestConfig(), "profit", search_stop_loss=False)
    assert len(rows) == len(_combinations(PARAM_GRIDS["gold"], search_stop_loss=False))


def test_search_rows_serialise(prices):
    import json

    rows = grid_search(prices, "gold", BacktestConfig(), "profit", search_stop_loss=False)
    json.dumps([r.summary() for r in rows])


# ----------------------------------------------------------------------
# train / test separation -- the important part
# ----------------------------------------------------------------------
def test_optimiser_never_sees_the_test_window(prices, monkeypatch):
    """The bars used to pick parameters must exclude the judging window."""
    seen = {}

    import algobot.backtest.optimize as opt

    original = opt.grid_search

    def spy(df, *a, **kw):
        seen["last_train_bar"] = df.index[-1]
        return original(df, *a, **kw)

    monkeypatch.setattr(opt, "grid_search", spy)
    report = opt.train_test_optimize(prices, "donchian", BacktestConfig(), "profit",
                                     train_ratio=0.6, search_stop_loss=False, symbol="OPT")

    assert seen["last_train_bar"] <= pd.Timestamp(report.test_window[0])


def test_report_exposes_in_and_out_of_sample(prices):
    report = train_test_optimize(prices, "donchian", BacktestConfig(), "profit",
                                 search_stop_loss=False, symbol="OPT")
    d = report.to_dict()

    assert "in_sample_profit" in d and "out_of_sample_profit" in d
    assert "baseline_out_of_sample_profit" in d
    assert d["improvement_vs_default"] == pytest.approx(
        d["out_of_sample_profit"] - d["baseline_out_of_sample_profit"], abs=0.01
    )


def test_train_and_test_windows_do_not_overlap(prices):
    report = train_test_optimize(prices, "donchian", BacktestConfig(), "profit",
                                 search_stop_loss=False, symbol="OPT")
    assert pd.Timestamp(report.train_window[1]) < pd.Timestamp(report.test_window[0])


def test_a_strategy_too_sparse_to_qualify_raises():
    """Gold Cross needs 200 bars to warm up; on a short history every
    combination lands under the trade floor.

    Refusing to report is the correct behaviour: choosing parameters off one or
    two trades is how a backtest gets fabricated.
    """
    short = load_synthetic("SPARSE", days=700)
    with pytest.raises(ValueError, match="minimum trade count"):
        train_test_optimize(short, "gold", BacktestConfig(), "profit",
                            train_ratio=0.6, search_stop_loss=False, symbol="SPARSE")


def test_tiny_input_is_rejected():
    with pytest.raises(ValueError):
        train_test_optimize(load_synthetic("TINY", days=5), "ma", BacktestConfig())


# ----------------------------------------------------------------------
# walk-forward
# ----------------------------------------------------------------------
def test_walk_forward_produces_the_requested_folds(prices):
    report = walk_forward(prices, "gold", BacktestConfig(), "profit", n_folds=3,
                          search_stop_loss=False, symbol="OPT")
    assert len(report.folds) == 3
    assert report.to_dict()["folds"] == 3


def test_walk_forward_test_windows_are_sequential(prices):
    report = walk_forward(prices, "gold", BacktestConfig(), "profit", n_folds=3,
                          search_stop_loss=False, symbol="OPT")
    starts = [pd.Timestamp(f["test"][0]) for f in report.folds]
    assert starts == sorted(starts)


def test_each_fold_trains_only_on_earlier_bars(prices):
    report = walk_forward(prices, "gold", BacktestConfig(), "profit", n_folds=3,
                          search_stop_loss=False, symbol="OPT")
    for fold in report.folds:
        assert pd.Timestamp(fold["train_end"]) <= pd.Timestamp(fold["test"][0])


def test_walk_forward_reports_against_the_default(prices):
    report = walk_forward(prices, "gold", BacktestConfig(), "profit", n_folds=3,
                          search_stop_loss=False, symbol="OPT")
    assert report.improvement == pytest.approx(report.tuned_profit - report.default_profit)
    assert 0 <= report.folds_improved <= len(report.folds)


def test_walk_forward_rejects_insufficient_history():
    with pytest.raises(ValueError):
        walk_forward(load_synthetic("SHORT", days=120), "ma", BacktestConfig(), n_folds=4)


# ----------------------------------------------------------------------
# random baseline
# ----------------------------------------------------------------------
def test_random_baseline_returns_a_distribution(prices):
    cut = int(len(prices) * 0.6)
    stats = random_parameter_baseline(prices.iloc[:cut], prices.iloc[cut:], "gold",
                                      BacktestConfig(), n_trials=6, search_stop_loss=False,
                                      symbol="OPT")
    assert stats["n"] > 0
    assert stats["worst"] <= stats["median"] <= stats["best"]
    assert len(stats["profits"]) == stats["n"]


def test_random_baseline_is_reproducible(prices):
    cut = int(len(prices) * 0.6)
    kw = dict(cfg=BacktestConfig(), n_trials=5, search_stop_loss=False, symbol="OPT", seed=42)
    a = random_parameter_baseline(prices.iloc[:cut], prices.iloc[cut:], "gold", **kw)
    b = random_parameter_baseline(prices.iloc[:cut], prices.iloc[cut:], "gold", **kw)
    assert np.allclose(a["profits"], b["profits"])


# ----------------------------------------------------------------------
# intraday grids
# ----------------------------------------------------------------------
def test_intraday_grids_cover_every_strategy():
    from algobot.backtest.optimize import INTRADAY_PARAM_GRIDS

    assert set(INTRADAY_PARAM_GRIDS) == set(PARAM_GRIDS)


def test_intraday_windows_are_shorter_than_daily():
    """A 200-bar average on hourly data is a 29-session average, not intraday."""
    from algobot.backtest.optimize import INTRADAY_PARAM_GRIDS

    assert max(INTRADAY_PARAM_GRIDS["ma"]["slow"]) < max(PARAM_GRIDS["ma"]["slow"])
    assert max(INTRADAY_PARAM_GRIDS["donchian"]["entry_window"]) < \
        max(PARAM_GRIDS["donchian"]["entry_window"])


def test_intraday_stops_are_tighter():
    """A 12% stop is unreachable inside a session, so it is no stop at all."""
    from algobot.backtest.optimize import INTRADAY_STOP_LOSS_GRID

    finite_intraday = [s for s in INTRADAY_STOP_LOSS_GRID if s]
    finite_daily = [s for s in STOP_LOSS_GRID if s]
    assert max(finite_intraday) < max(finite_daily)


def test_grids_for_switches_on_the_flag():
    from algobot.backtest.optimize import INTRADAY_PARAM_GRIDS, grids_for

    assert grids_for(intraday=True)[0] is INTRADAY_PARAM_GRIDS
    assert grids_for(intraday=False)[0] is PARAM_GRIDS


def test_intraday_search_uses_the_intraday_grid(prices):
    from algobot.backtest.optimize import INTRADAY_PARAM_GRIDS, _combinations

    rows = grid_search(prices, "gold", BacktestConfig(), "profit",
                       search_stop_loss=False, intraday=True)
    assert len(rows) == len(_combinations(INTRADAY_PARAM_GRIDS["gold"], False))
