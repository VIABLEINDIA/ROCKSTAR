"""Command-line interface for the Algorithmic Trading Bot.

    algobot fetch      --symbol RELIANCE --period 10y
    algobot train      --symbol RELIANCE
    algobot backtest   --symbol RELIANCE --strategy ma --period 1y
    algobot paper-run  --symbol RELIANCE --strategy ma          # reproduces Tables 3-6
    algobot replay     --symbol RELIANCE --strategy multiple
    algobot trade      --symbol RELIANCE --strategy ma --broker sim
    algobot account    --broker sim
    algobot symbols    --query RELI
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest.engine import run_backtest
from .backtest.report import (compare_with_model, cost_summary, enforce_out_of_sample, money,
                              results_frame, run_suite, strategy_table, summary_table)
from .config import ARTIFACT_DIR, BacktestConfig, BotConfig, Config, DataConfig, load_config
from .data.loader import load_prices, slice_period
from .data.preprocess import build_dataset
from .model.random_forest import TrainedModel, default_model_path, predict_frame, train
from .strategies.registry import PAPER_STRATEGIES, REGISTRY, build_strategy, pretty, wrap_with_model


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """CLI flags beat the YAML file, which beats the dataclass defaults."""
    for attr, section, field_name in [
        ("symbol", "data", "symbol"),
        ("period", "data", "period"),
        ("source", "data", "source"),
        ("interval", "data", "interval"),
        ("no_cache", "data", "use_cache"),
        ("quantity", "backtest", "quantity"),
        ("stop_loss", "backtest", "stop_loss_pct"),
        ("costs", "backtest", "cost_model"),
        ("cash", "backtest", "initial_cash"),
    ]:
        value = getattr(args, attr, None)
        if value is None:
            continue
        if attr == "no_cache":
            cfg.data.use_cache = not value
        else:
            setattr(getattr(cfg, section), field_name, value)

    if getattr(args, "symbol", None):
        cfg.bot.symbol = args.symbol
    if getattr(args, "strategy", None):
        cfg.bot.strategy = args.strategy
    return cfg


def _strategy_params(args: argparse.Namespace) -> dict:
    """Strategy parameters the user supplied (Section IV.F: 'entered by the user')."""
    keys = ("fast", "slow", "entry_window", "exit_window", "min_votes",
            "rsi_period", "steps", "min_expected_return")
    return {k: getattr(args, k) for k in keys if getattr(args, k, None) is not None}


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------
def cmd_fetch(args, cfg: Config) -> int:
    df = load_prices(cfg.data)
    print(f"\n{cfg.data.symbol.upper()}  source={cfg.data.source}  rows={len(df)}  "
          f"{df.index[0].date()} -> {df.index[-1].date()}\n")
    print(df.tail(10).round(2).to_string())
    return 0


def cmd_auth(args, cfg: Config) -> int:
    """Check Dhan credentials end to end without ever printing them."""
    from .broker.dhan import DhanBroker, token_expiry, token_is_expired

    creds = cfg.dhan
    env_file = ARTIFACT_DIR.parent / ".env"

    # Only ever report presence and length -- a credential check that prints
    # the credential defeats its own purpose.
    token_state = f"set ({len(creds.access_token)} chars)" if creds.access_token else "MISSING"

    print(f"\n.env file            {'found' if env_file.exists() else 'not found'} ({env_file})")
    print(f"DHAN_ACCESS_TOKEN    {token_state}")
    print(f"DHAN_CLIENT_ID       {'set' if creds.client_id else 'MISSING'}")
    print(f"Base URL             {creds.base_url}")

    expiry = token_expiry(creds.access_token)
    if expiry is not None:
        expired = token_is_expired(creds.access_token)
        age = (expiry - datetime.now(timezone.utc)).total_seconds() / 3600
        state = f"EXPIRED {abs(age):.1f}h ago" if expired else f"valid for {age:.1f}h"
        print(f"Token expiry         {expiry:%Y-%m-%d %H:%M UTC} -- {state}")
        if expired:
            print("\nThe token is expired. Dhan access tokens last about 24 hours.")
            print("Regenerate under Profile -> DhanHQ Trading API, then update .env "
                  "and rerun this command.")
            return 1

    if not creds.configured:
        print("\nBoth DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID are required.")
        print("Put them in .env (see .env.example) or export them, then rerun.")
        return 1

    print("\nCalling Dhan...")
    try:
        broker = DhanBroker(creds, dry_run=True)
        account = broker.account()
        print(f"  funds        OK -- cash {money(account.cash, 'INR')}")
    except Exception as exc:
        print(f"  funds        FAILED: {str(exc)[:160]}")
        if "DH-901" in str(exc):
            print("\nDH-901 means the token is invalid or expired. Dhan access tokens are "
                  "short-lived:\nregenerate under Profile -> DhanHQ Trading API and "
                  "update .env.")
        return 1

    try:
        bars = broker.history(cfg.data.symbol, lookback_days=30)
        print(f"  daily candles OK -- {len(bars)} bars for {cfg.data.symbol.upper()}, "
              f"latest {bars.index[-1].date()}")
    except Exception as exc:
        print(f"  daily candles FAILED: {str(exc)[:160]}")
        return 1

    try:
        print(f"  live LTP      OK -- {cfg.data.symbol.upper()} "
              f"{money(broker.last_price(cfg.data.symbol), 'INR')}")
    except Exception as exc:
        print(f"  live LTP      unavailable: {str(exc)[:120]}")

    print("\nCredentials work. --source dhan will now use Dhan data.")
    return 0


def cmd_symbols(args, cfg: Config) -> int:
    from .data.instruments import resolve_security_id, search_symbols

    if args.query:
        print(search_symbols(args.query, args.limit).to_string(index=False))
    else:
        sid = resolve_security_id(cfg.data.symbol, cfg.data.exchange_segment)
        print(f"{cfg.data.symbol.upper()} -> securityId {sid} ({cfg.data.exchange_segment})")
    return 0


def cmd_train(args, cfg: Config) -> int:
    df = load_prices(cfg.data)
    dataset = build_dataset(df, cfg.preprocess)
    print(f"\nDataset: {json.dumps(dataset.summary())}\n")

    model = train(dataset, cfg.data.symbol, cfg.model, cfg.preprocess)

    print("Section V -- Evaluation\n")
    print(model.metrics.as_table())
    print("\nTop feature importances:")
    print(model.feature_importances().head(10).round(5).to_string())

    path = model.save(args.out)
    print(f"\nSaved joblib bundle -> {path}")

    if not args.no_plots:
        from .model.plots import plot_actual_vs_predicted, plot_feature_importances

        frame = predict_frame(model, dataset)
        p1 = plot_actual_vs_predicted(frame, cfg.data.symbol, currency=cfg.backtest.currency)
        p2 = plot_feature_importances(model.feature_importances(), cfg.data.symbol)
        print(f"Figure 16 (actual vs predicted) -> {p1}")
        print(f"Feature importances              -> {p2}")

    metrics_path = ARTIFACT_DIR / f"{cfg.data.symbol.upper()}_metrics.json"
    metrics_path.write_text(
        json.dumps({"symbol": cfg.data.symbol.upper(), "metrics": model.metrics.to_dict(),
                    "dataset": dataset.summary()}, indent=2), encoding="utf-8")
    print(f"Metrics JSON                     -> {metrics_path}")
    return 0


def _load_model_if_requested(args, cfg: Config) -> TrainedModel | None:
    if not getattr(args, "use_model", False):
        return None
    path = Path(args.model) if getattr(args, "model", None) else default_model_path(cfg.data.symbol)
    if not path.exists():
        print(f"! No model at {path}. Run: algobot train --symbol {cfg.data.symbol}",
              file=sys.stderr)
        return None
    return TrainedModel.load(path)


def cmd_backtest(args, cfg: Config) -> int:
    model = _load_model_if_requested(args, cfg)
    df = load_prices(cfg.data)
    window = slice_period(df, args.period or cfg.data.period)
    window, _ = enforce_out_of_sample(window, model, args.allow_in_sample)
    if window.empty:
        print("! No out-of-sample bars in this window for that model.", file=sys.stderr)
        return 1

    strategy = build_strategy(args.strategy, **_strategy_params(args))
    label = args.strategy
    if model is not None:
        strategy = wrap_with_model(strategy, model, **_strategy_params(args))
        label += "+rf"

    result = run_backtest(window, strategy, cfg.backtest, cfg.data.symbol,
                          args.period or cfg.data.period)
    result.strategy = label

    print(f"\n{pretty(args.strategy)} on {cfg.data.symbol.upper()} "
          f"({window.index[0].date()} -> {window.index[-1].date()})")
    print(f"Strategy: {strategy.describe()}\n")
    print(json.dumps(result.to_dict(), indent=2))

    if result.trades:
        print(f"\nLast {min(10, len(result.trades))} trades:")
        print(result.trades_frame().tail(10).to_string(index=False))

    if not args.no_plots and not result.history.empty:
        from .model.plots import plot_backtest

        path = plot_backtest(result, cfg.data.symbol, f"{label}_{result.period_label}",
                             currency=cfg.backtest.currency)
        print(f"\nChart -> {path}")
    return 0


def cmd_paper_run(args, cfg: Config) -> int:
    """Reproduce Section VI: every strategy over 1-year and 10-year windows."""
    model = _load_model_if_requested(args, cfg)
    strategies = [args.strategy] if args.strategy else PAPER_STRATEGIES
    periods = args.periods or ["1y", "10y"]

    results = run_suite(cfg.data.symbol, strategies, periods, cfg.data, cfg.backtest,
                        model=model, strategy_params=_strategy_params(args),
                        save_plots=not args.no_plots, allow_in_sample=args.allow_in_sample)

    print(f"\nSection VI -- Backtest results for {cfg.data.symbol.upper()} "
          f"({cfg.backtest.quantity} shares/trade, "
          f"stop-loss {cfg.backtest.stop_loss_pct:.1%})\n")
    for name in strategies:
        subset = [r for r in results if r.strategy.startswith(name)]
        print(strategy_table(subset, name, cfg.backtest.currency))
        print()

    print(summary_table(results, cfg.backtest.currency))
    print()
    print(cost_summary(results, cfg.backtest.currency))
    csv_path = ARTIFACT_DIR / f"{cfg.data.symbol.upper()}_summary.csv"
    results_frame(results).to_csv(csv_path, index=False)
    print(f"\nArtifacts -> {ARTIFACT_DIR}")
    print(f"Summary CSV -> {csv_path}")
    return 0


def cmd_compare(args, cfg: Config) -> int:
    """Does the RF filter actually help? Same window, with and without."""
    args.use_model = True
    model = _load_model_if_requested(args, cfg)
    if model is None:
        return 1

    strategies = [args.strategy] if args.strategy else PAPER_STRATEGIES
    frame = compare_with_model(cfg.data.symbol, strategies, model, cfg.data, cfg.backtest,
                               _strategy_params(args), args.period or "max")

    start, end = frame.attrs["window"]
    print("\nSection IV.F -- strategy alone vs strategy + Random Forest")
    print(f"{cfg.data.symbol.upper()}  out-of-sample window {start} -> {end}  "
          f"(model trained through {frame.attrs['cutoff']})")
    print(f"Buy & hold over the same window: "
          f"{money(frame.attrs['buy_and_hold'], cfg.backtest.currency)}\n")
    print(frame.to_string(index=False))

    better = int((frame["delta"] > 0).sum())
    print(f"\nThe model improved P&L in {better} of {len(frame)} strategies.")
    return 0


def cmd_validate(args, cfg: Config) -> int:
    """Permutation test: model filter vs random vetoes of the same size."""
    from .backtest.significance import format_report, permutation_test

    args.use_model = True
    model = _load_model_if_requested(args, cfg)
    if model is None:
        return 1

    df = load_prices(cfg.data)
    window = slice_period(df, args.period or "max")
    window, cutoff = enforce_out_of_sample(window, model, args.allow_in_sample)
    if window.empty:
        print("! No out-of-sample bars for that model.", file=sys.stderr)
        return 1

    strategies = [args.strategy] if args.strategy else PAPER_STRATEGIES
    rows = [
        permutation_test(window, name, model, cfg.backtest, args.trials,
                         _strategy_params(args), cfg.data.symbol)
        for name in strategies
    ]

    print(f"\nOut-of-sample window {window.index[0].date()} -> {window.index[-1].date()}"
          f"  ({args.trials} random trials per strategy)\n")
    print(format_report(rows, cfg.data.symbol, cfg.backtest.currency))

    out = ARTIFACT_DIR / f"{cfg.data.symbol.upper()}_permutation_test.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")
    return 0


def cmd_optimize(args, cfg: Config) -> int:
    """Tune parameters, and show what survives out-of-sample."""
    from .backtest.optimize import (PARAM_GRIDS, random_parameter_baseline,
                                    train_test_optimize, walk_forward)

    df = load_prices(cfg.data)
    window = slice_period(df, args.period or cfg.data.period)
    strategies = [args.strategy] if args.strategy else list(PARAM_GRIDS)

    if args.walk_forward:
        print(f"\nWalk-forward tuning on {cfg.data.symbol.upper()} "
              f"({args.folds} folds, objective={args.objective})")
        print("Parameters are re-fitted on everything before each fold and traded on the "
              "fold itself.\n")
        header = (f"{'STRATEGY':<12}{'TUNED':>14}{'DEFAULT':>14}{'DELTA':>14}"
                  f"{'TRADES':>9}{'FOLDS WON':>11}")
        print(header)
        print("-" * len(header))
        payload = []
        for name in strategies:
            report = walk_forward(window, name, cfg.backtest, args.objective,
                                  n_folds=args.folds, symbol=cfg.data.symbol)
            payload.append(report.to_dict())
            print(f"{pretty(name)[:11]:<12}"
                  f"{money(report.tuned_profit, cfg.backtest.currency):>14}"
                  f"{money(report.default_profit, cfg.backtest.currency):>14}"
                  f"{money(report.improvement, cfg.backtest.currency):>14}"
                  f"{report.tuned_trades:>9}"
                  f"{report.folds_improved:>7}/{len(report.folds)}")
        out = ARTIFACT_DIR / f"{cfg.data.symbol.upper()}_walkforward.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved -> {out}")
        return 0

    payload = []
    for name in strategies:
        report = train_test_optimize(window, name, cfg.backtest, args.objective,
                                     train_ratio=args.train_ratio, symbol=cfg.data.symbol)
        payload.append(report.to_dict())

        print(f"\n{pretty(name)} on {cfg.data.symbol.upper()}")
        print(f"  train {report.train_window[0]} -> {report.train_window[1]}   "
              f"test {report.test_window[0]} -> {report.test_window[1]}")
        print(f"  best params        {report.best_params}")
        print(f"  in-sample          {money(report.in_sample.profit, cfg.backtest.currency)}")
        print(f"  OUT-OF-SAMPLE      {money(report.out_of_sample.profit, cfg.backtest.currency)}")
        print(f"  default OOS        "
              f"{money(report.baseline_out_of_sample.profit, cfg.backtest.currency)}")
        print(f"  improvement        {money(report.improvement, cfg.backtest.currency)}")
        print(f"  top-10 median OOS  {money(report.robust_oos, cfg.backtest.currency)}"
              "   (a lone good cell is a fluke; neighbours should agree)")

        if args.random_baseline:
            cut = int(len(window) * args.train_ratio)
            base = random_parameter_baseline(window.iloc[:cut], window.iloc[cut:], name,
                                             cfg.backtest, symbol=cfg.data.symbol)
            pct = 100.0 * float((base["profits"] < report.out_of_sample.profit).mean())
            print(f"  random params OOS  mean "
                  f"{money(base['mean'], cfg.backtest.currency)}, "
                  f"tuned beats {pct:.0f}% of {base['n']} random sets")

    out = ARTIFACT_DIR / f"{cfg.data.symbol.upper()}_optimize.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out}")
    return 0


def cmd_replay(args, cfg: Config) -> int:
    from .bot.trader import replay_session

    model = _load_model_if_requested(args, cfg)
    df = load_prices(cfg.data)
    window = slice_period(df, args.period or "2y")

    bot_cfg = cfg.bot
    bot_cfg.strategy = args.strategy
    bot_cfg.quantity = cfg.backtest.quantity
    bot_cfg.use_model = model is not None
    bot_cfg.dry_run = False        # fills land in the simulated ledger only

    start = max(int(len(window) * 0.3), 250)
    summary = replay_session(window, cfg.data.symbol, bot_cfg, model,
                             start_index=min(start, len(window) - 2),
                             max_bars=args.bars, strategy_params=_strategy_params(args))
    print("\nReplay session (live bot loop over historical bars)\n")
    print(json.dumps(summary, indent=2))
    return 0


def _build_broker(args, cfg: Config):
    if args.broker == "dhan":
        from .broker.dhan import DhanBroker

        return DhanBroker(cfg.dhan, exchange_segment=cfg.bot.exchange_segment,
                          product_type=cfg.bot.product_type, dry_run=args.dry_run,
                          confirm_live=args.i_understand_live_trading)

    from .broker.simulated import SimulatedBroker

    return SimulatedBroker(cash=cfg.backtest.initial_cash, data_cfg=cfg.data,
                           force_market_open=args.ignore_market_hours)


def cmd_trade(args, cfg: Config) -> int:
    from .bot.trader import TradingBot

    broker = _build_broker(args, cfg)

    bot_cfg = cfg.bot
    bot_cfg.symbol = cfg.data.symbol
    bot_cfg.strategy = args.strategy
    bot_cfg.quantity = cfg.backtest.quantity
    bot_cfg.dry_run = args.dry_run
    bot_cfg.broker = args.broker
    bot_cfg.use_model = args.use_model
    bot_cfg.model_path = args.model
    if args.product_type:
        bot_cfg.product_type = args.product_type.upper()
    if args.square_off_time:
        bot_cfg.square_off_time = args.square_off_time
    if args.no_square_off:
        bot_cfg.auto_square_off = False
    if args.poll is not None:
        bot_cfg.poll_seconds = args.poll
    if args.day_stop_loss is not None:
        bot_cfg.day_stop_loss_pct = args.day_stop_loss
    if args.trade_stop_loss is not None:
        bot_cfg.trade_stop_loss_pct = args.trade_stop_loss

    bot = TradingBot(broker, bot_cfg, strategy_params=_strategy_params(args))
    print(f"\nTrading {bot_cfg.symbol} with {bot.strategy.describe()} via {broker.name}")
    if bot.is_intraday:
        detail = (f"square-off {bot_cfg.square_off_time} IST, no new entries after "
                  f"{bot_cfg.no_new_entries_after} IST" if bot_cfg.auto_square_off
                  else "square-off DISABLED -- Dhan will force-close at its own price")
    else:
        detail = "positions carried overnight"
    print(f"Product: {bot_cfg.product_type} | {detail}")
    print(f"Stop the bot with Ctrl-C or: touch {bot_cfg.stop_file}\n")

    summary = bot.run(max_iterations=args.iterations)
    print("\n" + json.dumps(summary, indent=2))
    return 0


def cmd_account(args, cfg: Config) -> int:
    broker = _build_broker(args, cfg)
    account = broker.account()
    print(f"\nBroker: {broker.name}")
    print(f"Cash:   {money(account.cash, account.currency)}")
    print(f"Equity: {money(account.equity, account.currency)}")

    pos = broker.get_position(cfg.data.symbol)
    if pos and pos.is_open:
        print(f"\nPosition: {pos.quantity} {pos.symbol} @ {pos.avg_price:.2f} "
              f"(last {pos.last_price:.2f}, unrealised {pos.unrealised_pnl:+.2f}, "
              f"{pos.unrealised_pct:+.2%})")
    else:
        print(f"\nNo open position in {cfg.data.symbol.upper()}")

    if args.reset and hasattr(broker, "reset"):
        broker.reset(cfg.backtest.initial_cash)
        print(f"\nPaper ledger reset to {money(cfg.backtest.initial_cash)}")
    return 0


# ----------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="algobot",
        description="Algorithmic Trading Bot -- Random Forest + financial strategies on Dhan/NSE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", help="YAML config file")
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, with_strategy: bool = False):
        p.add_argument("--symbol", help="NSE trading symbol, e.g. RELIANCE")
        p.add_argument("--period", help="history window, e.g. 1y / 10y")
        p.add_argument("--source", choices=["dhan", "yahoo", "synthetic"])
        p.add_argument("--interval", help="bar interval (1d, 5m, ...)")
        p.add_argument("--no-cache", action="store_true")
        p.add_argument("--quantity", type=int, help="shares per trade")
        p.add_argument("--cash", type=float, help="starting cash")
        p.add_argument("--stop-loss", type=float, help="per-trade stop as a fraction, e.g. 0.05")
        p.add_argument("--costs", choices=["delivery", "intraday", "none"],
                       help="transaction cost model (default delivery; 'none' reproduces "
                            "the paper's gross tables)")
        if with_strategy:
            p.add_argument("--strategy", choices=sorted(REGISTRY), default="ma")
            p.add_argument("--fast", type=int)
            p.add_argument("--slow", type=int)
            p.add_argument("--entry-window", type=int)
            p.add_argument("--exit-window", type=int)
            p.add_argument("--min-votes", type=int)
            p.add_argument("--rsi-period", type=int)
            p.add_argument("--use-model", action="store_true",
                           help="fuse the RF model with the strategy (Section IV.F)")
            p.add_argument("--model", help="path to a joblib bundle")
            p.add_argument("--steps", type=int, help="forecast bars ahead the model is read at")
            p.add_argument("--min-expected-return", type=float,
                           help="forecast return an entry must clear")
            p.add_argument("--no-plots", action="store_true")
            p.add_argument("--allow-in-sample", action="store_true",
                           help="do not trim model backtests to out-of-sample bars "
                                "(results become optimistic)")

    p = sub.add_parser("fetch", help="download and cache OHLCV bars")
    common(p)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("auth", help="verify Dhan credentials (never prints them)")
    common(p)
    p.set_defaults(func=cmd_auth)

    p = sub.add_parser("symbols", help="resolve or search Dhan security ids")
    common(p)
    p.add_argument("--query", help="substring to search in the scrip master")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_symbols)

    p = sub.add_parser("train", help="train the Random Forest Regressor (Sections IV.C-V)")
    common(p)
    p.add_argument("--out", help="output joblib path")
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("backtest", help="backtest one strategy over one window")
    common(p, with_strategy=True)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("paper-run", help="reproduce the paper's Tables 3-6")
    common(p, with_strategy=True)
    p.add_argument("--periods", nargs="+", help="durations to test", default=None)
    p.set_defaults(func=cmd_paper_run, strategy=None)

    p = sub.add_parser("compare", help="strategy alone vs strategy + RF, same window")
    common(p, with_strategy=True)
    p.set_defaults(func=cmd_compare, strategy=None)

    p = sub.add_parser("validate", help="permutation test: is the model skilled or just quieter?")
    common(p, with_strategy=True)
    p.add_argument("--trials", type=int, default=200, help="random filters to sample")
    p.set_defaults(func=cmd_validate, strategy=None)

    p = sub.add_parser("optimize", help="tune parameters and report what survives OOS")
    common(p, with_strategy=True)
    p.add_argument("--objective", default="profit",
                   choices=["profit", "profit_factor", "sharpe", "return_per_trade"])
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--walk-forward", action="store_true",
                   help="re-optimise at each fold and trade the next (the honest test)")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--random-baseline", action="store_true",
                   help="compare against randomly chosen parameters")
    p.set_defaults(func=cmd_optimize, strategy=None)

    p = sub.add_parser("replay", help="drive the live bot loop over historical bars")
    common(p, with_strategy=True)
    p.add_argument("--bars", type=int, default=250, help="bars to replay")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("trade", help="run the live loop (Section IV.F)")
    common(p, with_strategy=True)
    p.add_argument("--broker", choices=["sim", "dhan"], default="sim")
    p.add_argument("--poll", type=int, help="seconds between iterations")
    p.add_argument("--iterations", type=int, help="stop after N iterations")
    p.add_argument("--day-stop-loss", type=float, help="session stop as a fraction")
    p.add_argument("--trade-stop-loss", type=float, help="per-position stop as a fraction")
    p.add_argument("--product-type", choices=["CNC", "INTRADAY", "MARGIN", "cnc", "intraday",
                                             "margin"],
                   help="Dhan product type (default CNC: strategies here hold ~40 days, "
                        "and INTRADAY/MIS is auto-squared-off the same day)")
    p.add_argument("--square-off-time", help="IST cutoff to flatten INTRADAY positions")
    p.add_argument("--no-square-off", action="store_true",
                   help="disable the intraday square-off (Dhan will force-close instead)")
    p.add_argument("--dry-run", action="store_true", help="log orders instead of sending them")
    p.add_argument("--ignore-market-hours", action="store_true",
                   help="sim broker only: trade outside the NSE session")
    p.add_argument("--i-understand-live-trading", action="store_true",
                   help="arm real order placement on the Dhan account")
    p.set_defaults(func=cmd_trade)

    p = sub.add_parser("account", help="show broker cash, equity and position")
    common(p)
    p.add_argument("--broker", choices=["sim", "dhan"], default="sim")
    p.add_argument("--reset", action="store_true", help="reset the paper ledger")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--ignore-market-hours", action="store_true")
    p.add_argument("--i-understand-live-trading", action="store_true")
    p.set_defaults(func=cmd_account)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)

    cfg = _apply_overrides(load_config(args.config), args)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 40)

    try:
        return args.func(args, cfg)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        logging.getLogger("algobot").error("%s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
