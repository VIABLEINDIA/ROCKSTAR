"""Point-in-time universe construction and the no-fabrication guarantee."""

from __future__ import annotations

import pytest

from algobot.config import DataConfig
from algobot.data.loader import load_prices
from algobot.data.universe import (
    CASUALTIES_2016,
    SURVIVORS_2016,
    UniverseResult,
    load_universe,
    midcap_universe_2016,
)


def test_casualties_are_disjoint_from_survivors():
    assert not set(CASUALTIES_2016) & set(SURVIVORS_2016)


def test_universe_includes_failures_by_default():
    """The whole point: a universe that drops its failures is the biased one."""
    honest = midcap_universe_2016(include_casualties=True)
    biased = midcap_universe_2016(include_casualties=False)

    assert set(CASUALTIES_2016).issubset(honest)
    assert not set(CASUALTIES_2016) & set(biased)
    assert len(honest) > len(biased)


def test_casualty_list_is_not_empty():
    assert len(CASUALTIES_2016) >= 10


def test_result_reports_coverage():
    r = UniverseResult(loaded=["A", "B"], missing=["C"], frames={})
    assert r.coverage == pytest.approx(2 / 3)
    assert "unavailable" in r.summary()
    assert "C" in r.summary()


def test_empty_result_has_zero_coverage():
    assert UniverseResult([], [], {}).coverage == 0.0


def test_strict_mode_refuses_to_fabricate(monkeypatch):
    """A delisted ticker must raise, never return generated bars."""
    monkeypatch.setattr("algobot.data.loader._load_yahoo",
                        lambda cfg: (_ for _ in ()).throw(RuntimeError("delisted")))

    with pytest.raises(RuntimeError, match="Synthetic fallback is disabled"):
        load_prices(DataConfig(symbol="GONE", period="1y", source="yahoo",
                               use_cache=False, allow_synthetic_fallback=False))


def test_permissive_mode_still_falls_back(monkeypatch):
    """CI and offline demos keep working."""
    monkeypatch.setattr("algobot.data.loader._load_yahoo",
                        lambda cfg: (_ for _ in ()).throw(RuntimeError("offline")))

    df = load_prices(DataConfig(symbol="GONE2", period="1y", source="yahoo",
                                use_cache=False, allow_synthetic_fallback=True))
    assert not df.empty
    assert df.attrs["source"] == "synthetic"


def test_provenance_is_recorded():
    df = load_prices(DataConfig(symbol="PROV", period="1y", source="synthetic",
                                use_cache=False))
    assert df.attrs["source"] == "synthetic"
    assert df.attrs["symbol"] == "PROV"


def test_missing_symbols_are_listed_not_skipped(monkeypatch):
    def fake(cfg, *a, **k):
        if cfg.symbol == "BAD":
            raise RuntimeError("no data")
        from algobot.data.loader import load_synthetic
        return load_synthetic(cfg.symbol, days=400)

    monkeypatch.setattr("algobot.data.universe.load_prices", fake)
    r = load_universe(["GOOD", "BAD"], strict=True)

    assert r.loaded == ["GOOD"]
    assert r.missing == ["BAD"]
    assert r.coverage == pytest.approx(0.5)


def test_price_floor_excludes_penny_stocks(monkeypatch):
    """Rs 1L notional in a Rs 2 stock is an artefact of the fill model."""
    from algobot.data.loader import load_synthetic

    def fake(cfg, *a, **k):
        price = 2.0 if cfg.symbol == "PENNY" else 500.0
        return load_synthetic(cfg.symbol, days=400, start_price=price)

    monkeypatch.setattr("algobot.data.universe.load_prices", fake)

    assert load_universe(["PENNY", "NORMAL"], min_price=0).loaded == ["PENNY", "NORMAL"]
    assert load_universe(["PENNY", "NORMAL"], min_price=20).loaded == ["NORMAL"]


def test_short_history_is_excluded(monkeypatch):
    from algobot.data.loader import load_synthetic

    monkeypatch.setattr("algobot.data.universe.load_prices",
                        lambda cfg, *a, **k: load_synthetic(cfg.symbol, days=50))
    r = load_universe(["SHORT"], min_bars=250)
    assert r.loaded == [] and r.missing == ["SHORT"]


def test_price_floor_uses_the_start_price_not_the_median(monkeypatch):
    """A collapsed stock was tradeable when the strategy would have bought it.

    Filtering on the whole-period median deletes exactly the failures the
    universe exists to include -- survivorship bias re-entering through the
    liquidity screen.
    """
    import numpy as np
    import pandas as pd

    def fake(cfg, *a, **k):
        idx = pd.bdate_range("2016-01-01", periods=400)
        # Starts at Rs 500, collapses to Rs 2: a casualty, tradeable at entry.
        close = pd.Series(np.linspace(500, 2, len(idx)), index=idx)
        return pd.DataFrame({"Open": close, "High": close, "Low": close,
                             "Close": close, "Volume": 1000}, index=idx)

    monkeypatch.setattr("algobot.data.universe.load_prices", fake)
    result = load_universe(["COLLAPSED"], min_price=20)

    assert result.loaded == ["COLLAPSED"], "a collapsed stock must stay in the universe"
