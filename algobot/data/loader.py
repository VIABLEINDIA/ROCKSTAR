"""Section III -- Dataset.

Fetches day-wise OHLCV bars (Date, Open, High, Low, Close, Volume) from the
DhanHQ v2 charts API (the paper used Alpaca; this build targets Dhan/NSE),
with Yahoo Finance as a secondary source, an on-disk CSV cache, and a
deterministic synthetic generator so the pipeline stays runnable offline
(tests, CI, demos without network or API credentials).
"""

from __future__ import annotations

import logging
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CACHE_DIR, DataConfig, DhanCredentials
from .instruments import resolve_security_id

log = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# A frame this short is a failed download, not a short listing. Treating it as
# data poisons the cache and, worse, short-circuits the fallback chain: a
# 1-row NSE stub for a delisted company would hide the BSE listing that
# actually has its history.
MIN_USABLE_ROWS = 20

_PERIOD_DAYS = {
    "1mo": 30, "3mo": 91, "6mo": 182, "1y": 365, "2y": 730,
    "5y": 1826, "10y": 3653, "max": 7305,
}


def period_to_days(period: str) -> int:
    if period in _PERIOD_DAYS:
        return _PERIOD_DAYS[period]
    if period.endswith("d") and period[:-1].isdigit():
        return int(period[:-1])
    raise ValueError(f"Unsupported period {period!r}")


# Yahoo caps how far back intraday bars go, by interval.
YAHOO_INTRADAY_MAX_DAYS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "90m": 60, "60m": 730, "1h": 730,
}


def is_intraday(interval: str) -> bool:
    return not str(interval).lower().endswith(("d", "wk", "mo"))


def clamp_period(period: str, interval: str) -> str:
    """Shrink a period Yahoo will not serve at this interval.

    Asking for 10y of 5-minute bars returns an empty frame rather than an
    error, which then looks like a dead symbol. Clamping makes the limit
    visible instead.
    """
    cap = YAHOO_INTRADAY_MAX_DAYS.get(str(interval).lower())
    if cap is None:
        return period
    try:
        wanted = period_to_days(period)
    except ValueError:
        return period
    if wanted <= cap:
        return period
    log.warning("Yahoo serves at most %d days of %s bars; clamping period %s -> %dd",
                cap, interval, period, cap)
    return f"{cap}d"


def _cache_path(symbol: str, period: str, interval: str, source: str) -> Path:
    return CACHE_DIR / f"{symbol.upper()}_{source}_{period}_{interval}.csv"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tz-naive, Date-indexed frame with exactly the OHLCV columns."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns a (field, ticker) MultiIndex for single tickers too
        df = df.droplevel(-1, axis=1)

    df = df.rename(columns={c: str(c).title() for c in df.columns})
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {missing}")

    df = df[OHLCV].copy()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["Close"])


def load_synthetic(
    symbol: str = "SYNTH", days: int = 3653, seed: int = 7, start_price: float = 100.0
) -> pd.DataFrame:
    """Deterministic geometric-random-walk bars with a mild drift and regimes.

    Used when `source="synthetic"` or when a live download fails, so every
    downstream stage (training, backtests, the bot's sim broker) stays usable
    without a network connection.
    """
    # zlib.crc32, not hash(): Python salts string hashing per process, so
    # hash() would hand a different series to every run -- silently breaking
    # the reproducibility this generator exists to provide.
    rng = np.random.default_rng((zlib.crc32(symbol.encode("utf-8")) ^ seed) % (2**32))
    idx = pd.bdate_range(end=datetime.now(timezone.utc).date(), periods=days)

    # Slowly varying drift produces trending and ranging regimes, which the
    # trend-following strategies need in order to be meaningfully exercised.
    regime = np.sin(np.linspace(0, 8 * np.pi, len(idx))) * 0.0006
    shocks = rng.normal(0.0003, 0.014, len(idx)) + regime
    close = start_price * np.exp(np.cumsum(shocks))

    spread = np.abs(rng.normal(0.006, 0.003, len(idx))) * close
    open_ = close * (1 + rng.normal(0, 0.004, len(idx)))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(1_000_000, 9_000_000, len(idx))

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )
    df.index.name = "Date"
    return df


def _load_yahoo(cfg: DataConfig) -> pd.DataFrame:
    import yfinance as yf

    ticker = cfg.symbol.upper()
    if cfg.yahoo_suffix and not ticker.endswith(cfg.yahoo_suffix) and "." not in ticker:
        ticker += cfg.yahoo_suffix          # NSE tickers are RELIANCE.NS on Yahoo

    df = yf.download(
        ticker,
        period=clamp_period(cfg.period, cfg.interval),
        interval=cfg.interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or len(df) < MIN_USABLE_ROWS:
        raise RuntimeError(
            f"Yahoo Finance returned {0 if df is None else len(df)} rows for {ticker} "
            f"(need >= {MIN_USABLE_ROWS}); treating as unavailable"
        )
    return _normalise(df)


def _load_dhan(cfg: DataConfig, creds: DhanCredentials | None = None) -> pd.DataFrame:
    """DhanHQ v2 charts API.

    Daily bars come from ``POST /charts/historical``; anything finer comes from
    ``POST /charts/intraday``. Both return column-oriented arrays
    (``open``/``high``/``low``/``close``/``volume``/``timestamp``) where
    ``timestamp`` is epoch seconds.
    """
    import requests

    creds = creds or DhanCredentials()
    if not creds.configured:
        raise RuntimeError(
            "Dhan credentials missing: set DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID"
        )

    security_id = resolve_security_id(cfg.symbol, cfg.exchange_segment)
    to_date = datetime.now(timezone.utc).date()
    from_date = to_date - timedelta(days=period_to_days(cfg.period))

    body = {
        "securityId": str(security_id),
        "exchangeSegment": cfg.exchange_segment,
        "instrument": cfg.instrument,
        "fromDate": from_date.isoformat(),
        "toDate": to_date.isoformat(),
    }

    if cfg.interval in ("1d", "1D", "day"):
        endpoint = f"{creds.base_url}/charts/historical"
        body["expiryCode"] = 0
        body["oi"] = False
    else:
        endpoint = f"{creds.base_url}/charts/intraday"
        body["interval"] = str(cfg.interval).rstrip("m") or "1"

    resp = requests.post(endpoint, headers=creds.headers(), json=body, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Dhan charts API {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()

    if not payload or not payload.get("close"):
        raise RuntimeError(f"Dhan returned no candles for {cfg.symbol} ({security_id})")

    df = pd.DataFrame(
        {
            "Open": payload["open"],
            "High": payload["high"],
            "Low": payload["low"],
            "Close": payload["close"],
            "Volume": payload.get("volume", [0] * len(payload["close"])),
        }
    )
    ts = payload.get("timestamp") or payload.get("start_Time")
    df.index = pd.to_datetime(pd.Series(ts, dtype="float64"), unit="s", utc=True)
    df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    # Only daily candles collapse to a date; intraday must keep time of day.
    if not is_intraday(cfg.interval):
        df.index = df.index.normalize()
    return _normalise(df)


def load_prices(cfg: DataConfig, creds: DhanCredentials | None = None) -> pd.DataFrame:
    """Load OHLCV bars for `cfg.symbol`, honouring the cache and falling back
    to synthetic data if the configured source is unreachable."""
    cache = _cache_path(cfg.symbol, cfg.period, cfg.interval, cfg.source)
    if cfg.use_cache and cache.exists():
        log.info("Loading cached bars from %s", cache)
        df = pd.read_csv(cache, index_col="Date", parse_dates=["Date"])
        if len(df) >= MIN_USABLE_ROWS:
            out = _normalise(df)
            out.attrs["source"] = f"cache:{cfg.source}"
            out.attrs["symbol"] = cfg.symbol
            return out

    used = cfg.source
    try:
        if cfg.source == "dhan":
            df = _load_dhan(cfg, creds)
        elif cfg.source == "yahoo":
            df = _load_yahoo(cfg)
        elif cfg.source == "synthetic":
            df = load_synthetic(cfg.symbol, days=period_to_days(cfg.period))
        else:
            raise ValueError(f"Unknown data source {cfg.source!r}")
    except Exception as exc:  # network, auth, rate limit, unknown symbol...
        log.warning("Download from %s failed (%s); trying the next source", cfg.source, exc)
        try:
            if cfg.source != "yahoo":
                df = _load_yahoo(cfg)
                used = "yahoo"
            else:
                raise
        except Exception as exc2:
            if not cfg.allow_synthetic_fallback:
                raise RuntimeError(
                    f"No real market data for {cfg.symbol!r} "
                    f"({cfg.source} failed: {exc}; yahoo failed: {exc2}). "
                    "Synthetic fallback is disabled, so this is not being substituted "
                    "with generated bars."
                ) from exc2
            log.warning("Fallback source failed too (%s); FABRICATING synthetic bars for %s "
                        "-- these are not market data", exc2, cfg.symbol)
            df = _load_synthetic_for(cfg)
            used = "synthetic"

    if len(df) < MIN_USABLE_ROWS:
        # Central guard: whatever the source, a frame this short is a failed
        # download and must not be cached or returned as data.
        raise RuntimeError(
            f"Only {len(df)} rows for {cfg.symbol!r} from {used} "
            f"(need >= {MIN_USABLE_ROWS}); treating as unavailable"
        )

    df.attrs["source"] = used          # provenance: what actually produced these bars
    df.attrs["symbol"] = cfg.symbol
    if cfg.use_cache and len(df) >= MIN_USABLE_ROWS:
        df.to_csv(cache)
    return df


def _load_synthetic_for(cfg: DataConfig) -> pd.DataFrame:
    return load_synthetic(cfg.symbol, days=period_to_days(cfg.period))


def slice_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Trailing window of `df` (e.g. the paper's 1-year and 10-year tests)."""
    if period == "max" or df.empty:
        return df
    cutoff = df.index.max() - pd.Timedelta(period_to_days(period), unit="D")
    return df.loc[df.index >= cutoff]
