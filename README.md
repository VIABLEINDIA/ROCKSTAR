# Algorithmic Trading Bot

[![tests](https://github.com/VIABLEINDIA/ROCKSTAR/actions/workflows/tests.yml/badge.svg)](https://github.com/VIABLEINDIA/ROCKSTAR/actions/workflows/tests.yml)

An implementation of **"Algorithmic Trading Bot"** — Medha Mathur, Satyam Mhadalekar, Sahil Mhatre,
Vanita Mane, *ITM Web of Conferences* **40**, 03041 (2021), ICACC-2021.

The paper builds a Random Forest Regressor over lagged closing prices, backtests four financial
strategies, and fuses the two inside a bot that trades a paper account. This build follows that
design section by section, but targets **Dhan (DhanHQ API v2) and the NSE** rather than the paper's
Alpaca/US setup, so prices, orders and P&L are in ₹ on Indian equities.

> **Not investment advice.** Live order placement is disabled by default and requires an explicit
> flag. Backtested performance says nothing about future returns.

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python -m algobot.cli paper-run --symbol RELIANCE --period 10y
```

That downloads ten years of daily bars, backtests all four strategies over 1-year and 10-year
windows, and writes tables, charts and CSVs to `artifacts/`. It works with no credentials at all —
without Dhan keys it falls back to Yahoo Finance (`RELIANCE.NS`), and offline it falls back to a
deterministic synthetic series.

Full pipeline:

```bash
python -m algobot.cli train --symbol RELIANCE --period 10y
```

```bash
python -m algobot.cli compare --symbol RELIANCE
```

```bash
python -m algobot.cli validate --symbol RELIANCE
```

```bash
python -m algobot.cli trade --symbol RELIANCE --strategy ma --broker sim --use-model
```

---

## How the paper maps onto the code

| Paper section | Implementation |
|---|---|
| III — Dataset (Alpaca / Yahoo OHLCV) | [`algobot/data/loader.py`](algobot/data/loader.py) — Dhan `/charts/historical`, Yahoo, synthetic |
| III — symbol identity | [`algobot/data/instruments.py`](algobot/data/instruments.py) — Dhan scrip master → `securityId` |
| IV.A — 41 lags, `[0:33]` split | [`algobot/data/preprocess.py`](algobot/data/preprocess.py) |
| IV.B — 60:40 split | `train_test_split_ordered` (chronological, never shuffled) |
| IV.C–IV.E — RF train / predict / plot | [`algobot/model/random_forest.py`](algobot/model/random_forest.py), [`plots.py`](algobot/model/plots.py) |
| IV.F — strategy + model fusion | [`algobot/strategies/ml_filter.py`](algobot/strategies/ml_filter.py) |
| IV.F — live loop on a paper account | [`algobot/bot/trader.py`](algobot/bot/trader.py), [`risk.py`](algobot/bot/risk.py) |
| V — evaluation metrics | [`algobot/model/evaluate.py`](algobot/model/evaluate.py) |
| VI — Tables 3–6, Figures 17–20 | [`algobot/backtest/`](algobot/backtest/) |

The four strategies from Section VI live in [`algobot/strategies/`](algobot/strategies/):
Moving Average Crossover (`ma`), Donchian (`donchian`), Multiple Strategy (`multiple`),
Gold Cross (`gold`).

---

## Commands

| Command | Purpose |
|---|---|
| `fetch` | Download and cache OHLCV bars |
| `symbols` | Resolve or search Dhan `securityId`s |
| `train` | Train the Random Forest, print Section V metrics, save a joblib bundle |
| `backtest` | One strategy over one window |
| `paper-run` | Reproduce Tables 3–6 across strategies and durations |
| `compare` | Strategy alone vs strategy + RF on the *same* out-of-sample window |
| `validate` | Permutation test — is the model skilled, or just trading less? |
| `replay` | Drive the real bot loop over historical bars |
| `trade` | Run the live loop (Section IV.F) |
| `account` | Show broker cash, equity and open position |

Run `python -m algobot.cli <command> --help` for the full flag list.

---

## Findings: where this reproduces the paper, and where it doesn't

Four things in the paper's method do not survive contact with out-of-sample NSE data. Each is
implemented as described *and* addressed, with the fix documented and switchable.

### 1. Training on raw price levels cannot generalise

The paper feeds absolute closing prices to the forest. A decision tree can only ever output values it
saw in training, and a chronological 60:40 split puts the test period in a different price regime:

```
RELIANCE, 10y, paper-literal preprocessing
train target range   ₹223.5 – ₹1301.4
test  target range  ₹1015.9 – ₹1600.9
53.7% of test targets lie above anything in training
```

The forest structurally cannot reach those values. Result: **R² = −0.97, MAE ₹153**.

The fix is to normalise each 41-day window by its anchor close, so the model learns *shape* rather
than *level* (`preprocess.normalize`, on by default). Same data, same forest:

| Preprocessing | R² | MAE | Explained variance |
|---|---|---|---|
| Paper-literal (raw prices) | −0.970 | ₹153.35 | 0.138 |
| Window-normalised | −0.119 | 0.0216 (ratio) | −0.107 |

Set `preprocess.normalize: false` to reproduce the paper's original numbers.

### 2. The forecast itself has no edge

Even normalised, and at every horizon tested, the model is no better than predicting the mean
(RELIANCE; TCS and INFY land in the same place — R² −0.201 and −0.084, directional accuracy 48.9%
and 50.4%):

| Horizon | R² | Directional accuracy |
|---|---|---|
| 1 bar | −0.052 | 44.2% |
| 3 bars | −0.077 | 50.5% |
| 8 bars (paper's 41−33) | −0.120 | 49.6% |
| 20 bars | −0.158 | 49.9% |

Directional accuracy sits at a coin flip. Lagged closes alone do not predict future closes — which is
the expected result for a liquid large-cap, and worth stating plainly rather than tuning away.

Note that Figure 16-style *price* charts look convincing regardless: when a prediction is anchored to
a recent close, the level dominates and the curves overlap. That visual agreement is not evidence of
skill.

### 3. Backtesting a model over its own training data

Filtering trades with a model, then backtesting across a window that includes that model's training
split, measures memorisation. On the full 10 years that inflation is large:

| Multiple Strategy + RF | Strike rate | Profit |
|---|---|---|
| 10y window overlapping training | 63.3% | ₹20,023 |
| Out-of-sample only | 18.8% | −₹2,478 |

`compare` and `paper-run --use-model` therefore trim to the model's out-of-sample period
automatically; `--allow-in-sample` opts out and logs a warning.

### What the honest comparison shows

`compare` runs each strategy with and without the RF filter over one identical out-of-sample window
(RELIANCE, 2022-09-15 → 2026-08-20; buy & hold ₹1,305.14 over the same window):

| Strategy | Profit | Profit + RF | Δ | Trades → +RF |
|---|---|---|---|---|
| Moving Average Crossover | −₹319.38 | −₹318.36 | **+₹1.02** | 18 → 16 |
| Donchian | ₹1,489.79 | ₹1,303.81 | −₹185.99 | 11 → 11 |
| Multiple Strategy | −₹2,467.03 | −₹2,478.21 | −₹11.18 | 33 → 32 |
| Gold Cross | −₹1,317.03 | −₹1,317.03 | ₹0.00 | 2 → 2 |

**The model improved P&L in 1 of 4 strategies.** On TCS and INFY, however, it improved **3 of 4**:

| Symbol | Buy & hold (same window) | Strategies improved by RF |
|---|---|---|
| RELIANCE | +₹1,305.14 | 1 of 4 |
| TCS | −₹8,063.50 | 3 of 4 |
| INFY | −₹3,028.00 | 3 of 4 |

That pattern is itself the clue: the RF filter looks good exactly where the market fell. A veto
filter removes entries, and removing entries in a declining market raises P&L whether or not the
vetoes were chosen intelligently. See the next section.

### 4. Filtering gains are exposure reduction, not skill

`validate` runs the null hypothesis directly: veto the *same number* of entries **at random**, 200
times, and locate the model's result in that distribution. A skilled model should sit in the top
tail.

```bash
python -m algobot.cli validate --symbol TCS --trials 200
```

```
Permutation test -- TCS: does the RF filter beat random vetoes of equal size?
STRATEGY        VETOED      BASE P&L     MODEL P&L   RANDOM MEAN   PCTILE       p
---------------------------------------------------------------------------------
ma                2/14   Rs 8,050.45   Rs 8,441.45   Rs 6,815.07    66.0%   0.340
donchian         19/58  Rs -3,057.60  Rs -1,830.60  Rs -3,458.29    71.5%   0.285
multiple       102/389 Rs -11,745.40  Rs -9,287.70 Rs -10,117.60    69.5%   0.305
gold               0/1   Rs 4,224.00   Rs 4,224.00   Rs 4,224.00     0.0%   1.000
```

Across all three symbols and four strategies — **12 tests — the model beat random vetoing at
p < 0.05 exactly once** (INFY / Multiple, p = 0.025). With 12 comparisons you expect roughly one
result that good by chance alone, so it does not survive correction for multiple testing.

The forecast, in other words, adds nothing beyond trading less. This is consistent with the
directional accuracy of ~50% measured in §2 — and it is the kind of result the paper's evaluation
design (no null model, no out-of-sample separation) could not have detected.

Rerun any of this yourself — every number above comes from a command in this repo.

### Strategy results across three symbols

`paper-run` output (10 shares/trade, 5% stop, strategies only — no model), the direct analogue of
the paper's Tables 3–6:

| Strategy | RELIANCE 10y | TCS 10y | INFY 10y |
|---|---|---|---|
| Moving Average Crossover | 32.65% / ₹4,696 | 56.82% / ₹22,892 | 39.13% / ₹3,997 |
| Donchian | 43.24% / ₹5,446 | 50.00% / ₹6,145 | 36.36% / ₹3,218 |
| Multiple Strategy | 26.97% / ₹714 | 40.26% / ₹1,459 | 36.71% / ₹4,434 |
| Gold Cross | 57.14% / ₹6,496 | 33.33% / ₹543 | 28.57% / ₹4,245 |
| *Buy & hold* | *₹10,819* | *₹10,237* | *₹6,223* |

(strike rate / profit earned). Gold Cross returns **NA** on every 1-year window, exactly as the paper
reports — a 200-day average cannot form inside 250 trading days.

Two things are worth noting against the paper's Section VI, which reports profits for all four
strategies without a benchmark column. First, only one of these twelve strategy-symbol results beats
simply buying and holding the stock (TCS / MA Crossover). Second, the paper's own Donchian tables are
negative on both durations, so a mixed outcome here is consistent with it. Strike rates below 50%
are not by themselves damning — a trend-following strategy is designed to lose small and win big —
but the profit column is what the comparison turns on.

---

## Connecting a real Dhan account

```bash
cp .env.example .env      # then fill in your credentials
```

Set `DHAN_ACCESS_TOKEN` and `DHAN_CLIENT_ID` in the environment (tokens are generated in the Dhan
app under *DhanHQ Trading API* and expire, typically daily). Then:

```bash
python -m algobot.cli fetch --symbol TCS --source dhan
```

**Dhan has no paper-trading endpoint** — the API is the live account. So the paper-trading workflow
in Section IV.F is served by the built-in `sim` broker, which uses real market prices with a local
cash ledger (`cache/paper_account.json`) and never sends an order anywhere.

Live trading is guarded three ways:

1. `--broker sim` is the default; `--broker dhan` is required to reach the real API.
2. `--dry-run` logs the exact order body instead of sending it.
3. Real orders additionally require `--i-understand-live-trading`; without it the broker raises
   rather than transmitting.

```bash
python -m algobot.cli trade --symbol RELIANCE --strategy ma --broker dhan --dry-run
```

---

## How the bot decides

Each iteration of the loop:

1. **Stop signal?** — `Ctrl-C`, `bot.request_stop()`, or creating the `STOP` file.
2. **Market open?** — NSE cash session, 09:15–15:30 IST, weekdays.
3. **Risk check** — per-position stop-loss, then session stop-loss / take-profit / trade cap.
   Any breach flattens the position and halts.
4. **Market view** — rebuild signals from fresh bars; ask the model for its forecast.
5. **Act** — enter, exit, or hold.

Those are the paper's three stop conditions: *stop loss reached, market closed, or user stop signal*.
Every decision is appended to `artifacts/<SYMBOL>_bot_journal.json`.

Backtests execute a bar-*t* signal at the **open of bar t+1** — never the same bar's close, which
would leak information the strategy could not have had.

---

## Testing

```bash
python -m pytest tests/ -q
```

139 tests covering lag construction and the `[0:33]` split, normalisation invariance under a price
regime shift, look-ahead leakage in every strategy, execution timing, stop-loss and slippage
mechanics, strike-rate/profit maths, model persistence, the DhanHQ v2 order-body contract, the
live-order guard, all three bot stop conditions, and the CLI.

The suite never touches the network — synthetic data and monkeypatched brokers throughout.

CI ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)) runs the suite on Python 3.11,
3.12 and 3.13, then exercises the whole pipeline end to end — train, backtest, compare, validate,
replay, and a paper-broker session — against `--source synthetic`, so a run can never fail because
a market data provider was slow or the exchange was closed. Generated charts and tables are uploaded
as build artifacts.

---

## Project layout

```
algobot/
  config.py            dataclass config + Dhan credentials from the environment
  cli.py               all nine commands
  data/                loader (Dhan/Yahoo/synthetic), scrip master, preprocessing
  model/               Random Forest, Section V metrics, figures
  strategies/          ma, donchian, multiple, gold, indicators, RF fusion, registry
  backtest/            execution engine, Tables 3-6 reporting, permutation test
  broker/              base interface, DhanHQ v2 client, simulated + replay brokers
  bot/                 live trading loop and risk guards
tests/                 139 tests
artifacts/             generated charts, tables, journals  (git-ignored)
models/                joblib bundles                      (git-ignored)
cache/                 downloaded bars, scrip master, paper ledger (git-ignored)
```

---

## License

[MIT](LICENSE) — this implementation, © 2026.

The MIT grant covers the code in this repository only. The paper it implements is a separate work by
its authors, published under CC BY 4.0; the citation below is the appropriate way to credit it.
DhanHQ API access is governed by Dhan's own terms.

---

## Reference

Mathur, M., Mhadalekar, S., Mhatre, S., Mane, V. (2021). *Algorithmic Trading Bot.*
ITM Web of Conferences 40, 03041. https://doi.org/10.1051/itmconf/20214003041

Published by EDP Sciences under the Creative Commons Attribution License 4.0.
