"""Dhan scrip-master lookup: NSE trading symbol -> securityId.

Every DhanHQ v2 call identifies an instrument by numeric `securityId` plus an
`exchangeSegment`, so the human-friendly symbol the user types has to be
resolved against Dhan's published scrip master CSV. The file is cached on disk;
a small built-in table keeps the resolver usable offline.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from ..config import CACHE_DIR, DhanCredentials

log = logging.getLogger(__name__)

SCRIP_CACHE = CACHE_DIR / "dhan_scrip_master.csv"

# Fallback for offline runs / tests. securityId values are the NSE_EQ ids.
FALLBACK_SECURITY_IDS = {
    "RELIANCE": 2885,
    "TCS": 11536,
    "HDFCBANK": 1333,
    "INFY": 1594,
    "ICICIBANK": 4963,
    "SBIN": 3045,
    "ITC": 1660,
    "AXISBANK": 5900,
    "LT": 11483,
    "WIPRO": 3787,
}

# Column names differ between Dhan's compact and detailed scrip masters.
_SYMBOL_COLS = ("SEM_TRADING_SYMBOL", "SM_SYMBOL_NAME", "UNDERLYING_SYMBOL",
                "SEM_CUSTOM_SYMBOL", "DISPLAY_NAME")
_ID_COLS = ("SEM_SMST_SECURITY_ID", "SECURITY_ID")
_SEGMENT_COLS = ("SEM_EXM_EXCH_ID", "EXCH_ID")
_SERIES_COLS = ("SEM_SERIES", "SERIES")


def _first_present(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    return next((n for n in names if n in df.columns), None)


@lru_cache(maxsize=1)
def load_scrip_master(url: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Download (and cache) Dhan's scrip master CSV."""
    creds = DhanCredentials()
    url = url or creds.scrip_master_url

    if SCRIP_CACHE.exists() and not refresh:
        return pd.read_csv(SCRIP_CACHE, low_memory=False)

    import requests

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    SCRIP_CACHE.write_bytes(resp.content)
    return pd.read_csv(SCRIP_CACHE, low_memory=False)


def resolve_security_id(symbol: str, exchange_segment: str = "NSE_EQ") -> int:
    """Resolve an NSE trading symbol to Dhan's numeric securityId."""
    symbol = symbol.upper().strip()

    try:
        df = load_scrip_master()
        sym_col = _first_present(df, _SYMBOL_COLS)
        id_col = _first_present(df, _ID_COLS)
        if sym_col and id_col:
            hits = df[df[sym_col].astype(str).str.upper().str.strip() == symbol]

            seg_col = _first_present(df, _SEGMENT_COLS)
            if seg_col is not None and len(hits) > 1:
                exch = exchange_segment.split("_")[0]
                narrowed = hits[hits[seg_col].astype(str).str.upper() == exch]
                hits = narrowed if not narrowed.empty else hits

            ser_col = _first_present(df, _SERIES_COLS)
            if ser_col is not None and len(hits) > 1:
                narrowed = hits[hits[ser_col].astype(str).str.upper().str.strip() == "EQ"]
                hits = narrowed if not narrowed.empty else hits

            if not hits.empty:
                return int(hits.iloc[0][id_col])
    except Exception as exc:
        log.warning("Scrip master lookup failed for %s (%s); using fallback table", symbol, exc)

    if symbol in FALLBACK_SECURITY_IDS:
        return FALLBACK_SECURITY_IDS[symbol]

    raise KeyError(
        f"Could not resolve securityId for {symbol!r} on {exchange_segment}. "
        "Refresh the scrip master or pass --security-id explicitly."
    )


def search_symbols(query: str, limit: int = 20) -> pd.DataFrame:
    """Fuzzy symbol search against the scrip master (CLI helper)."""
    df = load_scrip_master()
    sym_col = _first_present(df, _SYMBOL_COLS)
    id_col = _first_present(df, _ID_COLS)
    if not sym_col or not id_col:
        raise RuntimeError("Unexpected scrip master layout")
    mask = df[sym_col].astype(str).str.upper().str.contains(query.upper(), na=False)
    cols = [c for c in (sym_col, id_col, _first_present(df, _SEGMENT_COLS),
                        _first_present(df, _SERIES_COLS)) if c]
    return df.loc[mask, cols].head(limit)
