"""Data loading, instrument resolution, Dhan request shape, and the CLI."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from algobot.config import Config, DataConfig, DhanCredentials, load_config
from algobot.data.instruments import FALLBACK_SECURITY_IDS, resolve_security_id
from algobot.data.loader import OHLCV, load_prices, load_synthetic, period_to_days, slice_period


# ----------------------------------------------------------------------
# loader
# ----------------------------------------------------------------------
def test_synthetic_bars_are_well_formed():
    df = load_synthetic("X", days=300)
    assert list(df.columns) == OHLCV
    assert len(df) == 300
    assert (df["High"] >= df["Low"]).all()
    assert (df["High"] >= df["Close"]).all()
    assert (df["Low"] <= df["Close"]).all()
    assert (df["Close"] > 0).all()
    assert df.index.is_monotonic_increasing


def test_synthetic_bars_are_deterministic():
    a = load_synthetic("SEEDED", days=100)
    b = load_synthetic("SEEDED", days=100)
    pd.testing.assert_frame_equal(a, b)


def test_different_symbols_give_different_series():
    a = load_synthetic("AAA", days=100)
    b = load_synthetic("BBB", days=100)
    assert not a["Close"].equals(b["Close"])


@pytest.mark.parametrize("period,days", [("1y", 365), ("10y", 3653), ("30d", 30)])
def test_period_to_days(period, days):
    assert period_to_days(period) == days


def test_period_to_days_rejects_nonsense():
    with pytest.raises(ValueError):
        period_to_days("banana")


def test_slice_period_takes_a_trailing_window():
    df = load_synthetic("SLICE", days=1000)
    one_year = slice_period(df, "1y")

    assert len(one_year) < len(df)
    assert one_year.index[-1] == df.index[-1]
    assert (df.index.max() - one_year.index[0]).days <= 366
    assert slice_period(df, "max").equals(df)


def test_loader_falls_back_to_synthetic_without_credentials(tmp_path, monkeypatch):
    """No Dhan keys and no network must still yield a usable frame."""
    monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("DHAN_CLIENT_ID", raising=False)
    monkeypatch.setattr("algobot.data.loader._load_yahoo",
                        lambda cfg: (_ for _ in ()).throw(RuntimeError("offline")))

    cfg = DataConfig(symbol="NOSUCH", period="1y", source="dhan", use_cache=False)
    df = load_prices(cfg)
    assert not df.empty
    assert list(df.columns) == OHLCV


def test_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("algobot.data.loader.CACHE_DIR", tmp_path)
    monkeypatch.setattr("algobot.data.loader._cache_path",
                        lambda s, p, i, src: tmp_path / f"{s}_{src}_{p}_{i}.csv")

    cfg = DataConfig(symbol="CACHED", period="1y", source="synthetic", use_cache=True)
    first = load_prices(cfg)
    assert (tmp_path / "CACHED_synthetic_1y_1d.csv").exists()

    second = load_prices(cfg)          # served from disk this time
    # A CSV round trip drops the index freq attribute; the values must match.
    pd.testing.assert_frame_equal(first, second, check_freq=False)


# ----------------------------------------------------------------------
# instruments
# ----------------------------------------------------------------------
def test_known_symbols_resolve():
    assert resolve_security_id("RELIANCE") == 2885
    assert resolve_security_id("TCS") == 11536


def test_resolution_is_case_insensitive():
    assert resolve_security_id("reliance") == resolve_security_id("RELIANCE")


def test_unknown_symbol_raises(monkeypatch):
    monkeypatch.setattr("algobot.data.instruments.load_scrip_master",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(KeyError):
        resolve_security_id("NOT-A-REAL-TICKER")


def test_fallback_table_covers_the_defaults():
    assert "RELIANCE" in FALLBACK_SECURITY_IDS


# ----------------------------------------------------------------------
# Dhan credentials / request shape
# ----------------------------------------------------------------------
def test_credentials_report_configuration(monkeypatch):
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "token")
    monkeypatch.setenv("DHAN_CLIENT_ID", "client")
    creds = DhanCredentials()

    assert creds.configured
    assert creds.headers()["access-token"] == "token"
    assert creds.headers()["client-id"] == "client"


def test_missing_credentials_are_detected(monkeypatch):
    monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("DHAN_CLIENT_ID", raising=False)
    assert not DhanCredentials().configured


def test_dhan_order_body_matches_the_v2_contract(monkeypatch):
    """Guards the field names DhanHQ v2 expects for POST /orders."""
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "token")
    monkeypatch.setenv("DHAN_CLIENT_ID", "client-42")

    from algobot.broker.dhan import DhanBroker

    broker = DhanBroker(dry_run=True)
    monkeypatch.setattr(broker, "security_id", lambda symbol: 2885)

    order = broker.submit_order("RELIANCE", "BUY", 10)
    body = order.raw

    assert order.status == "DRY_RUN"
    assert body["dhanClientId"] == "client-42"
    assert body["transactionType"] == "BUY"
    assert body["exchangeSegment"] == "NSE_EQ"
    assert body["productType"] == "INTRADAY"
    assert body["orderType"] == "MARKET"
    assert body["securityId"] == "2885"
    assert body["quantity"] == 10
    assert body["validity"] == "DAY"


def test_live_orders_are_refused_without_explicit_confirmation(monkeypatch):
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "token")
    monkeypatch.setenv("DHAN_CLIENT_ID", "client")

    from algobot.broker.base import BrokerError
    from algobot.broker.dhan import DhanBroker

    broker = DhanBroker(dry_run=False, confirm_live=False)
    monkeypatch.setattr(broker, "security_id", lambda symbol: 2885)

    with pytest.raises(BrokerError, match="confirm_live"):
        broker.submit_order("RELIANCE", "BUY", 1)


def test_broker_rejects_a_bad_side(monkeypatch):
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "token")
    monkeypatch.setenv("DHAN_CLIENT_ID", "client")

    from algobot.broker.dhan import DhanBroker

    broker = DhanBroker(dry_run=True)
    monkeypatch.setattr(broker, "security_id", lambda symbol: 1)
    with pytest.raises(ValueError):
        broker.submit_order("RELIANCE", "HOLD", 1)


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def test_defaults_follow_the_paper():
    cfg = Config()
    assert cfg.preprocess.n_lags == 41
    assert cfg.preprocess.n_input_lags == 33
    assert cfg.preprocess.train_ratio == 0.60
    assert cfg.backtest.currency == "INR"


def test_yaml_overlay(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "data:\n  symbol: INFY\n  period: 5y\npreprocess:\n  n_lags: 21\n"
        "backtest:\n  quantity: 25\n",
        encoding="utf-8",
    )
    cfg = load_config(path)

    assert cfg.data.symbol == "INFY"
    assert cfg.data.period == "5y"
    assert cfg.preprocess.n_lags == 21
    assert cfg.backtest.quantity == 25
    assert cfg.preprocess.n_input_lags == 33      # untouched keys keep defaults


def test_config_to_dict_hides_secrets(monkeypatch):
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "super-secret")
    monkeypatch.setenv("DHAN_CLIENT_ID", "client")

    payload = json.dumps(Config().to_dict())
    assert "super-secret" not in payload
    assert "configured" in payload


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def test_cli_help_exits_cleanly():
    from algobot.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_fetch_runs(capsys):
    from algobot.cli import main

    assert main(["fetch", "--symbol", "CLITEST", "--period", "1y", "--source", "synthetic"]) == 0
    assert "CLITEST" in capsys.readouterr().out


def test_cli_backtest_runs(capsys):
    from algobot.cli import main

    code = main(["backtest", "--symbol", "CLITEST", "--period", "1y",
                 "--source", "synthetic", "--strategy", "donchian", "--no-plots"])
    assert code == 0

    out = capsys.readouterr().out
    assert "strike_rate" in out and "profit" in out


def test_cli_reports_a_bad_strategy():
    from algobot.cli import main

    with pytest.raises(SystemExit):
        main(["backtest", "--strategy", "nonsense"])
