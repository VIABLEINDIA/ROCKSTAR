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
| `optimize` | Grid-search parameters; `--walk-forward` for the out-of-sample answer |
| `replay` | Drive the real bot loop over historical bars |
| `trade` | Run the live loop (Section IV.F) |
| `account` | Show broker cash, equity and open position |

Run `python -m algobot.cli <command> --help` for the full flag list.

---

## Findings: where this reproduces the paper, and where it doesn't

Eight findings from the paper's method do not survive contact with out-of-sample NSE data. Each is
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

| Multiple Strategy + RF | Strike rate | Net profit | Trades |
|---|---|---|---|
| 10y window overlapping training | 58.7% | ₹15,170 | 109 |
| Out-of-sample only | 18.8% | **−₹4,304** | 32 |

`compare` and `paper-run --use-model` therefore trim to the model's out-of-sample period
automatically; `--allow-in-sample` opts out and logs a warning.

### What the honest comparison shows

`compare` runs each strategy with and without the RF filter over one identical out-of-sample window
(RELIANCE, 2022-09-15 → 2026-08-20; buy & hold ₹1,305.14 over the same window):

| Strategy | Net profit | Net + RF | Δ | Trades → +RF |
|---|---|---|---|---|
| Moving Average Crossover | −₹1,887.99 | −₹1,777.89 | **+₹110.10** | 18 → 16 |
| Donchian | ₹860.49 | ₹674.19 | −₹186.30 | 11 → 11 |
| Multiple Strategy | −₹4,859.67 | −₹4,304.40 | **+₹555.27** | 33 → 32 |
| Gold Cross | −₹1,430.15 | −₹1,430.15 | ₹0.00 | 2 → 2 |

**The model improved P&L in 2 of 4 strategies**, and only by shrinking losses — every RELIANCE
strategy bar Donchian loses money net. On TCS and INFY it improved **3 of 4**:

| Symbol | Buy & hold (same window) | Strategies improved by RF |
|---|---|---|
| RELIANCE | +₹1,305.14 | 2 of 4 |
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
ma                2/14   Rs 6,364.48   Rs 6,998.58   Rs 5,367.65    66.0%   0.340
donchian         19/58  Rs -6,273.37  Rs -3,684.60  Rs -6,082.12    81.5%   0.185
multiple       102/389 Rs -16,238.15 Rs -13,695.67 Rs -14,354.76    65.5%   0.345
gold               0/1   Rs 4,090.75   Rs 4,090.75   Rs 4,090.75     0.0%   1.000
```

Across all three symbols and four strategies — **12 tests — the model beat random vetoing at
p < 0.05 exactly once** (INFY / Multiple, p = 0.025). With 12 comparisons you expect roughly one
result that good by chance alone, so it does not survive correction for multiple testing.

The forecast, in other words, adds nothing beyond trading less. This is consistent with the
directional accuracy of ~50% measured in §2 — and it is the kind of result the paper's evaluation
design (no null model, no out-of-sample separation) could not have detected.

Rerun any of this yourself — every number above comes from a command in this repo.

### 5. Costs and slippage eat 54% of the profit

The paper's tables are gross of everything ("reduced exchange costs"). On NSE
equities the statutory charges are not a rounding error, and they are asymmetric: STT is 0.1% *per
side* on delivery, stamp duty falls on the buy, and a DP charge hits every delivery sell.

[`algobot/backtest/costs.py`](algobot/backtest/costs.py) models the full stack — brokerage, STT,
exchange transaction charges, SEBI turnover fees, stamp duty, GST, and DP charges — applied per leg.
`--costs delivery` (the default), `intraday`, or `none` to reproduce the paper's gross figures.

Three frictions are modelled, and each was defaulted **on** because leaving any of them off
flatters every strategy:

| Friction | Default | What it corrects |
|---|---|---|
| Statutory charges + brokerage | `delivery` | The paper reports gross of everything |
| Slippage | 5 bps per side | Filling at the open assumes a price you never had to compete for |
| Gap-through stops | enabled | Filling a stop *at* the stop assumes a fill exists there — untrue on exactly the gap days that trigger stops |

Across all 12 ten-year runs, frictionless → realistic:

| | Frictionless | Realistic |
|---|---|---|
| Total P&L | ₹64,285 | **₹21,988** |
| Charges | — | −₹26,264 |
| Profitable runs | 12 / 12 | **9 / 12** |

**The three frictions consume 54% of gross profit.** Charges alone turn two runs from winners into
losers — Multiple Strategy on RELIANCE goes from +₹714 to **−₹2,473**, and on TCS from +₹1,459 to
**−₹4,409**; both are the highest-turnover strategy in the set, which is the point: cost scales with
how often you trade, not how well.

Gap-aware stops cost a further ₹635–₹1,496 per run on the strategies that stop out often, and flip
INFY / Multiple Strategy from +₹219 to **−₹1,277** on their own.

The single most useful number the model produces is the **breakeven move** — how far price must
travel just to pay for the round trip:

| Price | Delivery (CNC) | Intraday (MIS) |
|---|---|---|
| ₹500 | 0.517% | 0.106% |
| ₹1,300 | 0.336% | 0.106% |
| ₹3,000 | 0.271% | 0.106% |

Any strategy whose average winning move is smaller than this cannot be profitable at any win rate.
Delivery is the *more* expensive model here despite Dhan's zero delivery brokerage, because STT
applies to both legs and the flat DP charge is punitive on small positions.

> Rate snapshot: 2026-08. Brokerage plans, exchange charges and stamp duty all change — every rate
> is a constructor argument so it can be corrected without touching the engine. Verify against
> Dhan's current pricing before trusting any figure.

### 6. Tuning: win rate and profitability pull in opposite directions

[`algobot/backtest/optimize.py`](algobot/backtest/optimize.py) grid-searches parameters, but never
reports an in-sample fit on its own. `optimize --walk-forward` re-fits at each fold and trades the
next one, so every reported trade comes from parameters chosen only on earlier data.

Optimising for **win rate** (searching take-profit as well as the strategy windows), walk-forward
across 3 symbols x 4 strategies:

| Objective | Walk-forward win rate | Walk-forward P&L | Trades |
|---|---|---|---|
| Default parameters | 33.9% | −₹43,638 | 268 |
| **Maximise win rate** | **70.91%** | **−₹37,517** | 165 |
| Maximise profit | 28.35% | −₹23,598 | 127 |

A win rate above 70% is straightforward to reach — and it is worth understanding *why* that is not
good news. A tight take-profit converts many open positions into small recorded wins while losses
run to the stop, so the payoff ratio collapses faster than the win rate improves. Individual results
show the mechanism plainly: RELIANCE / Donchian tunes to an **80.6% win rate and still loses ₹936**;
INFY / Donchian reaches **75% and loses ₹7,879**.

Optimising for profit does the opposite — 28% win rate, but ₹20,040 better than untrained defaults
over the same windows.

**Win rate is not a proxy for profitability, and targeting it directly makes the system worse.**
Neither objective produces a profitable system out-of-sample over these windows: tuning narrows the
loss, it does not create an edge that was never there.

### 7. Intraday is worse, not better

The obvious response to "costs eat the edge" is to trade a bar size where costs are lower: MIS
charges give a **0.106% breakeven** against delivery's 0.336%. So the strategies were retuned on
**hourly NSE bars** (2 years, ~490 sessions x 7 bars), with intraday-scaled parameter grids, MIS
costs, and a forced square-off at every session close — because Dhan closes MIS positions at the
bell, and a test that carries them overnight measures a strategy the broker would never allow.

Walk-forward, 3 symbols x 4 strategies:

| Objective | Win rate | Net P&L | Trades |
|---|---|---|---|
| Default parameters | 32.3% | −₹33,055 | — |
| Maximise profit | 34.11% | **−₹12,581** | 472 |
| Maximise win rate | 36.92% | −₹75,709 | 1,571 |

**1 of 24 tuned configurations was profitable** (TCS / Gold Cross, +₹470). Tuning for profit again
narrows the loss — by ₹20,474 against defaults — without crossing into profit.

Two things are worth drawing out. Cheaper per-trade costs did not help, because the moves captured
shrank faster than the charges did, and the square-off truncates exactly the multi-day trends these
strategies exist to ride. And targeting win rate failed on its own terms here: it reached only
36.92%, because with a forced session-end exit a take-profit can no longer manufacture wins — it
just tripled the trade count, and the costs with it.

Intraday grids are expressed in bars, not days ([`optimize.py`](algobot/backtest/optimize.py)): a
200-bar slow average on hourly data is a 29-session average, which is not an intraday strategy at
all.

### 8. Mid-caps, corrected for survivorship

Screening 25 NSE mid-caps showed the binding constraint is only partly volatility — it is mostly
**share price**, because a flat ₹12.5 DP charge is 0.3% of a 10-share position in a ₹400 stock and
0.008% of one in a ₹15,000 stock. YESBANK has 61.8% annualised volatility and a **6.7% breakeven**;
DIXON has 39.8% volatility and a **0.23% breakeven**. That is an artefact of fixed-share sizing, so
`position_notional` sizes each trade to a rupee amount instead — at ₹1,00,000 per trade cost drag is
~₹237 regardless of price.

Three corrections had to be made before the mid-cap result meant anything.

**a. The loader was fabricating data.** `ZOMATO` returned 3,653 bars over ten years for a company
that listed in 2021: Yahoo failed (the ticker was renamed) and the synthetic fallback silently
substituted generated bars, which were then reported as a result. `allow_synthetic_fallback=False`
now raises instead, every frame carries `attrs["source"]` provenance, and research code sets it.

**b. The universe was chosen with hindsight.** Picking mid-caps that are prominent *today* selects
for exactly the trending behaviour the strategies detect.
[`algobot/data/universe.py`](algobot/data/universe.py) rebuilds the 2016 universe as survivors
**plus casualties** — DHFL, RCOM, Jet Airways, Unitech, Gitanjali, Coffee Day and the rest.

**c. The liquidity screen re-introduced the bias.** A ₹20 median-price floor removed 13 of 15
casualties, because those companies *became* penny stocks by collapsing. Filtering on the
whole-period median uses information the trader could not have had — the same error as survivorship
bias itself. The floor is applied to the **price at the start of the window** instead.

**d. Delisted names survive on BSE.** Yahoo drops NSE history for fully delisted companies but
often keeps the BSE listing, sometimes only under the numeric scrip code (DHFL as `511072.BO`,
Cox & Kings as `533144.BO`). Adding that fallback lifted casualty coverage from 33% to **73%**.
A cache bug was hiding them further: a 1-row NSE stub for a delisted ticker was being cached and
short-circuiting the fallback, so three casualties vanished silently. Frames below 20 rows are now
treated as failed downloads and never cached.

Default parameters, no tuning, 10 years, ₹1,00,000 per trade, ₹20 entry-price floor:

| Group | Runs | Trades | Win rate | Net P&L | Buy & hold | Beat B&H |
|---|---|---|---|---|---|---|
| Large-cap | 12 | 527 | 36.81% | ₹602,858 | ₹2,676,780 | **0/12** |
| Mid-cap survivors *(biased)* | 96 | 4,014 | 26.46% | ₹13,120,522 | ₹44,595,437 | 36/96 (38%) |
| Mid-cap casualties | 44 | 1,286 | 20.22% | ₹1,257,695 | **−₹3,412,683** | **41/44** |
| **Mid-cap, honest universe** | **140** | **5,300** | **24.94%** | **₹14,378,217** | ₹41,182,755 | **77/140 (55%)** |

**Every step of correcting the bias made the strategies look better**, which is the opposite of what
survivorship correction normally does:

| Universe | Beat buy & hold |
|---|---|
| Survivors only | 36/96 — 38% |
| + casualties available on NSE | 55/116 — 47% |
| + casualties recovered from BSE | **77/140 — 55%** |

The mechanism is the one trend-following exists for: it exits a collapsing stock and stays out,
while buy & hold rides it to zero. Across the casualties it turned **−₹3.4M of buy-and-hold losses
into +₹1.26M**, beating buy & hold on 41 of 44 runs — VIDEOIND +₹530,863 against −₹92,930,
Cox & Kings +₹168,432 against −₹98,912, RCOM +₹22,089 against −₹98,386.

**The one case where it failed is the instructive one.** Gitanjali Gems lost **−₹120,472 against
buy & hold's −₹97,731** — worse than holding. Gitanjali was the Nirav Modi fraud: the collapse was
a sequence of gaps, not a trend, so every stop filled below its level and repeated re-entries were
knifed. That is precisely the failure mode a trend filter cannot cover, and it belongs in the record
alongside the successes.

**What is still wrong with it:**

1. **4 of 15 casualties have no data on either exchange** (UNITECH, PUNJLLOYD, EDUCOMP, SINTEX;
   EDUCOMP was also below the price floor). The gap is recorded in `NO_DATA_ANYWHERE` rather than
   left invisible.
2. Delisted series simply stop, and both the strategy and the benchmark are marked out at the last
   traded price. In reality a suspended holding recovers close to nothing, so **both** sides of the
   casualty comparison are optimistic — the relative ranking holds, the absolute figures do not.
3. Mid-cap absolute P&L still trails buy & hold (₹14.4M against ₹41.2M). These strategies avoid
   disasters; they do not out-earn a rising market.
4. Win rate falls to 24.94%. Still no relationship between win rate and profitability.

### Strategy results across three symbols

`paper-run` output (10 shares/trade, 5% stop, strategies only — no model), **net of delivery
costs**, the direct analogue of the paper's Tables 3–6:

| Strategy | RELIANCE 10y | TCS 10y | INFY 10y |
|---|---|---|---|
| Moving Average Crossover | 28.57% / ₹1,802 | 45.45% / **₹17,632** | 39.13% / ₹1,630 |
| Donchian | 43.24% / ₹3,874 | 50.00% / ₹815 | 36.36% / ₹754 |
| Multiple Strategy | 26.97% / −₹4,073 | 34.62% / −₹7,004 | 35.44% / −₹1,277 |
| Gold Cross | 42.86% / ₹3,969 | 33.33% / ₹27 | 28.57% / ₹3,837 |
| *Buy & hold* | *₹10,819* | *₹10,237* | *₹6,223* |

Only **1 of 12** net results beats buying and holding (TCS / MA Crossover). Gold Cross returns **NA** on every 1-year window, exactly as the paper
reports — a 200-day average cannot form inside 250 trading days.

Two things are worth noting against the paper's Section VI, which reports profits for all four
strategies without a benchmark column. First, only one of these twelve strategy-symbol results beats
simply buying and holding the stock (TCS / MA Crossover). Second, the paper's own Donchian tables are
negative on both durations, so a mixed outcome here is consistent with it. Strike rates below 50%
are not by themselves damning — a trend-following strategy is designed to lose small and win big —
but the profit column is what the comparison turns on.

### Overall performance, net of costs

Pooled across all 527 closed trades in the twelve 10-year runs:

| Metric | Value |
|---|---|
| **Win rate** | **36.43%** (192 wins / 335 losses) |
| **Net P&L** | **₹21,988** (gross ₹48,252 − ₹26,264 charges) |
| Average trade | **+₹41.72** |
| Average win | +₹1,150.08 |
| Average loss | −₹593.52 |
| Payoff ratio | 1.94 : 1 |
| Profit factor | **1.11** |
| Average cost per trade | ₹49.84 |
| Mean 10y return on a ₹100,000 account | **~1.8%** — under 0.2%/year |

A sub-50% win rate is normal for trend-following: the 1.94:1 payoff is what makes 36% profitable at
all. But the margin is now very thin. A profit factor of **1.11** means ₹1.11 earned per ₹1.00 lost,
and the average trade clears its own ₹49.84 of charges by only ₹42 — costs are larger than the edge
they are levied on.

Read alongside the fact that only **1 of 12** runs beats buy and hold, this is a working research
implementation, not a deployable edge.

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
3. **Square-off due?** — when trading INTRADAY, flatten at `square_off_time` and halt.
4. **Risk check** — per-position stop-loss, then session stop-loss / take-profit / trade cap.
   Any breach flattens the position and halts.
5. **Market view** — rebuild signals from fresh bars; ask the model for its forecast.
6. **Act** — enter, exit, or hold.

Those are the paper's three stop conditions: *stop loss reached, market closed, or user stop signal*.
Every decision is appended to `artifacts/<SYMBOL>_bot_journal.json`.

Backtests execute a bar-*t* signal at the **open of bar t+1** — never the same bar's close, which
would leak information the strategy could not have had.

### Product type: this is positional, not intraday

Every strategy here runs on **daily** bars, and across all 527 backtested trades the mean holding
period is **39.9 days** — not one closed inside a single session. So orders go out as **CNC**
(delivery) by default.

Sending them as `INTRADAY` (MIS) would put the order type in direct conflict with the strategy:
Dhan force-closes MIS positions around 15:20 IST, so a trade meant to run for weeks would be
liquidated the same afternoon, every afternoon.

If you do choose `--product-type INTRADAY` — with intraday bars and strategy windows retuned to
match — the bot manages the session itself rather than leaving it to the broker:

| Setting | Default | Behaviour |
|---|---|---|
| `no_new_entries_after` | 15:00 IST | Stops opening positions too late in the session to manage |
| `square_off_time` | 15:15 IST | Flattens and halts, ahead of Dhan's ~15:20 cutoff |
| `auto_square_off` | `true` | `--no-square-off` hands control back to Dhan's force-close |

If the market closes while a position is still open, the bot logs it as an **error** under
`INTRADAY` (Dhan will close it at a price of its choosing) and as routine information under CNC.

---

## Testing

```bash
python -m pytest tests/ -q
```

259 tests covering lag construction and the `[0:33]` split, normalisation invariance under a price
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
  backtest/            execution engine, costs, Tables 3-6 reporting, permutation test
  broker/              base interface, DhanHQ v2 client, simulated + replay brokers
  bot/                 live trading loop and risk guards
tests/                 259 tests
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
