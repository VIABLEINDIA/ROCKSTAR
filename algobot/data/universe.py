"""Point-in-time universes, for backtests that are not survivorship-biased.

Picking symbols from a list of companies that are prominent *today* selects for
exactly the behaviour a trend-following strategy is meant to detect: the names
are well known because they went up. The strategy is then tested only on the
outcomes it would have wanted, and the result is close to meaningless.

The correct construction chooses the universe by a rule applied at a past date
and keeps every constituent, including the ones that later collapsed, were
suspended, or delisted. NSE does not publish historical index constituents in
a form that can be fetched programmatically, so the lists below are assembled
manually: mid-caps that were liquid and well-known in 2016, split by what
happened next, so the failures cannot quietly drop out of the sample.

`SURVIVORS` and `CASUALTIES` together approximate the 2016 mid-cap universe.
Testing on `SURVIVORS` alone is the biased experiment; the two together are
the honest one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import DataConfig
from .loader import load_prices

log = logging.getLogger(__name__)

# Liquid NSE mid-caps circa 2016 that are still trading.
SURVIVORS_2016 = [
    "BHEL", "NBCC", "PNB", "IOB", "SUZLON", "YESBANK", "IDFCFIRSTB",
    "BANDHANBNK", "JUBLFOOD", "TATAELXSI", "PERSISTENT", "KPITTECH",
    "CDSL", "DIXON", "IRCTC", "RVNL", "MAZDOCK", "ANGELONE", "IRFC",
    "HFCL", "ZEEL", "IDEA", "RPOWER", "PCJEWELLER", "SOUTHBANK",
    "GTLINFRA", "ALOKINDS", "RUCHI", "INFIBEAM", "VAKRANGEE",
]

# Companies that were significant mid-caps in 2016 and subsequently failed:
# bankruptcy, fraud, insolvency or suspension. Yahoo may or may not still
# serve them; the ones that error out are reported rather than skipped
# silently, because a universe that quietly loses its failures is the bias
# this module exists to remove.
CASUALTIES_2016 = [
    "DHFL",          # Dewan Housing -- collapsed 2019, insolvency
    "RCOM",          # Reliance Communications -- insolvency 2019
    "RELCAPITAL",    # Reliance Capital -- insolvency 2021
    "JETAIRWAYS",    # Jet Airways -- grounded 2019
    "UNITECH",       # Unitech -- fraud, government takeover 2020
    "VIDEOIND",      # Videocon -- insolvency
    "GITANJALI",     # Gitanjali Gems -- fraud 2018, delisted
    "MANPASAND",     # Manpasand Beverages -- fraud, suspended
    "COFFEEDAY",     # Coffee Day -- collapse 2019
    "COXANDKINGS",   # Cox & Kings -- insolvency 2019
    "TALWALKARS",    # Talwalkars -- insolvency
    "PUNJLLOYD",     # Punj Lloyd -- insolvency
    "EDUCOMP",       # Educomp -- insolvency
    "BHUSANSTL",     # Bhushan Steel -- insolvency, absorbed by Tata
    "SINTEX",        # Sintex Industries -- insolvency
]

LARGE_CAPS = ["RELIANCE", "TCS", "INFY"]


@dataclass
class UniverseResult:
    """What a universe actually yielded, so gaps are visible not hidden."""

    loaded: list[str]
    missing: list[str]
    frames: dict

    @property
    def coverage(self) -> float:
        total = len(self.loaded) + len(self.missing)
        return len(self.loaded) / total if total else 0.0

    def summary(self) -> str:
        return (f"{len(self.loaded)} loaded, {len(self.missing)} unavailable "
                f"({self.coverage:.0%} coverage)"
                + (f" -- missing: {', '.join(self.missing)}" if self.missing else ""))


def midcap_universe_2016(include_casualties: bool = True) -> list[str]:
    """The 2016 mid-cap universe. Set False to reproduce the biased version."""
    return SURVIVORS_2016 + (CASUALTIES_2016 if include_casualties else [])


def load_universe(symbols: list[str], period: str = "10y", min_bars: int = 250,
                  strict: bool = True, min_price: float = 0.0) -> UniverseResult:
    """Load every symbol, recording which ones could not be sourced.

    `strict` disables the synthetic fallback, so a delisted ticker raises
    instead of being replaced by fabricated bars.

    `min_price` screens out stocks too cheap to trade at a realistic fill. On a
    Rs 2 share the tick alone is a 0.5-2.5% spread, so the few-basis-point
    slippage assumption that is fair on a Rs 500 stock understates costs by two
    orders of magnitude -- and a Rs 1,00,000 notional position would be tens of
    thousands of shares, well beyond the book. Results that depend on such
    names are artefacts of the fill model, not strategy edge.

    The filter is applied to the price at the **start** of the window, never to
    the whole-period median. Most casualties became penny stocks precisely by
    collapsing, so a median-price floor deletes the failures and smuggles
    survivorship bias back in through the liquidity screen -- the same error of
    filtering on information the trader could not have had.
    """
    loaded, missing, frames = [], [], {}

    for symbol in symbols:
        try:
            df = load_prices(DataConfig(symbol=symbol, period=period, source="yahoo",
                                        allow_synthetic_fallback=not strict))
        except Exception as exc:
            log.info("%s unavailable: %s", symbol, str(exc)[:100])
            missing.append(symbol)
            continue

        if len(df) < min_bars:
            log.info("%s has only %d bars (< %d); excluded", symbol, len(df), min_bars)
            missing.append(symbol)
            continue

        if min_price:
            # Price over the first quarter-year, i.e. what it cost when the
            # backtest would first have considered buying it.
            start_price = float(df["Close"].iloc[:60].median())
            if start_price < min_price:
                log.info("%s started at %.2f, below the Rs %.0f floor; excluded",
                         symbol, start_price, min_price)
                missing.append(symbol)
                continue

        loaded.append(symbol)
        frames[symbol] = df

    return UniverseResult(loaded, missing, frames)
