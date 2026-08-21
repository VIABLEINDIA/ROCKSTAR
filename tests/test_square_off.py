"""Product type and the intraday square-off.

Dhan force-closes INTRADAY (MIS) positions around 15:20 IST. Every strategy in
this project runs on daily bars and holds for ~40 days, so the order product
must be CNC by default; when INTRADAY is deliberately chosen, the bot has to
flatten on its own terms before the broker does it for us.
"""

from __future__ import annotations

from datetime import time

import pandas as pd
import pytest

from algobot.broker.dhan import (
    NSE_MIS_AUTO_SQUARE_OFF,
    is_at_or_after,
    parse_ist_time,
)
from algobot.broker.simulated import SimulatedBroker
from algobot.bot.trader import TradingBot
from algobot.config import BotConfig, Config, DataConfig
from algobot.data.loader import load_synthetic

SYMBOL = "SQTEST"


def ist(stamp: str):
    return pd.Timestamp(stamp, tz="Asia/Kolkata").to_pydatetime()


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return load_synthetic(SYMBOL, days=400)


@pytest.fixture
def broker(tmp_path, prices) -> SimulatedBroker:
    b = SimulatedBroker(cash=100_000.0, state_file=tmp_path / "ledger.json",
                        price_source="history", force_market_open=True,
                        data_cfg=DataConfig(symbol=SYMBOL, source="synthetic"))
    b.reset(100_000.0)
    b._history_cache[SYMBOL] = prices
    return b


def make_bot(broker, tmp_path, **overrides) -> TradingBot:
    cfg = BotConfig(symbol=SYMBOL, strategy="ma", quantity=10, poll_seconds=0,
                    use_model=False, stop_file=str(tmp_path / "STOP"), **overrides)
    return TradingBot(broker, cfg)


# ----------------------------------------------------------------------
# defaults
# ----------------------------------------------------------------------
def test_default_product_is_cnc_not_intraday():
    """Regression: MIS would be auto-squared-off under a ~40-day strategy."""
    assert Config().bot.product_type == "CNC"


def test_default_cutoffs_precede_dhans_own():
    cfg = Config().bot
    assert parse_ist_time(cfg.square_off_time) < NSE_MIS_AUTO_SQUARE_OFF
    assert parse_ist_time(cfg.no_new_entries_after) < parse_ist_time(cfg.square_off_time)


def test_dhan_broker_sends_the_configured_product(monkeypatch):
    monkeypatch.setenv("DHAN_ACCESS_TOKEN", "token")
    monkeypatch.setenv("DHAN_CLIENT_ID", "client")

    from algobot.broker.dhan import DhanBroker

    broker = DhanBroker(dry_run=True, product_type="CNC")
    monkeypatch.setattr(broker, "security_id", lambda s: 1)
    assert broker.submit_order(SYMBOL, "BUY", 1).raw["productType"] == "CNC"


# ----------------------------------------------------------------------
# time helpers
# ----------------------------------------------------------------------
def test_parse_ist_time_accepts_both_forms():
    assert parse_ist_time("15:15") == time(15, 15)
    assert parse_ist_time("15:15:30") == time(15, 15, 30)


@pytest.mark.parametrize("bad", ["15", "", "1:2:3:4", "abc"])
def test_parse_ist_time_rejects_junk(bad):
    with pytest.raises(ValueError):
        parse_ist_time(bad)


@pytest.mark.parametrize(
    "stamp,expected",
    [
        ("2026-08-20 15:14", False),   # a minute early
        ("2026-08-20 15:15", True),    # exactly on the cutoff
        ("2026-08-20 15:16", True),
        ("2026-08-22 16:00", False),   # Saturday: no session to square off
    ],
)
def test_is_at_or_after(stamp, expected):
    assert is_at_or_after("15:15", ist(stamp)) is expected


# ----------------------------------------------------------------------
# bot behaviour
# ----------------------------------------------------------------------
def test_cnc_never_squares_off(broker, tmp_path):
    bot = make_bot(broker, tmp_path, product_type="CNC")
    assert bot.is_intraday is False
    assert bot.should_square_off(ist("2026-08-20 15:29")) is False
    assert bot.entries_allowed(ist("2026-08-20 15:29")) is True


def test_intraday_squares_off_after_the_cutoff(broker, tmp_path):
    bot = make_bot(broker, tmp_path, product_type="INTRADAY", square_off_time="15:15")
    assert bot.is_intraday is True
    assert bot.should_square_off(ist("2026-08-20 15:00")) is False
    assert bot.should_square_off(ist("2026-08-20 15:16")) is True


def test_square_off_can_be_disabled(broker, tmp_path):
    bot = make_bot(broker, tmp_path, product_type="INTRADAY", auto_square_off=False)
    assert bot.should_square_off(ist("2026-08-20 15:29")) is False


def test_late_intraday_entries_are_blocked(broker, tmp_path):
    bot = make_bot(broker, tmp_path, product_type="INTRADAY", no_new_entries_after="15:00")
    assert bot.entries_allowed(ist("2026-08-20 14:59")) is True
    assert bot.entries_allowed(ist("2026-08-20 15:01")) is False


def test_step_flattens_and_halts_at_the_cutoff(broker, tmp_path, monkeypatch):
    bot = make_bot(broker, tmp_path, product_type="INTRADAY", square_off_time="15:15")
    broker.buy(SYMBOL, 10)
    assert broker.get_position(SYMBOL) is not None

    monkeypatch.setattr(bot, "broker_now", lambda: ist("2026-08-20 15:16"))

    assert bot.step() == "squared_off"
    assert broker.get_position(SYMBOL) is None          # flattened by the bot
    assert "square-off" in bot.risk.halt_reason
    assert any("square-off" in e.message for e in bot.events if e.kind == "order")


def test_step_squares_off_even_when_flat(broker, tmp_path, monkeypatch):
    bot = make_bot(broker, tmp_path, product_type="INTRADAY", square_off_time="15:15")
    monkeypatch.setattr(bot, "broker_now", lambda: ist("2026-08-20 15:16"))

    assert bot.step() == "squared_off"                  # halts without an order
    assert bot.risk.halted


def test_step_does_not_square_off_mid_session(broker, tmp_path, monkeypatch):
    bot = make_bot(broker, tmp_path, product_type="INTRADAY", square_off_time="15:15")
    monkeypatch.setattr(bot, "broker_now", lambda: ist("2026-08-20 11:00"))

    assert bot.step() != "squared_off"
    assert not bot.risk.halted


def test_late_entry_signal_is_suppressed(broker, tmp_path, monkeypatch):
    """A LONG arriving after the entry cutoff must not open a position."""
    bot = make_bot(broker, tmp_path, product_type="INTRADAY",
                   no_new_entries_after="15:00", square_off_time="15:25")
    monkeypatch.setattr(bot, "broker_now", lambda: ist("2026-08-20 15:05"))
    monkeypatch.setattr(bot, "market_view", lambda: {"signal": 1, "close": 100.0,
                                                     "expected_return": None, "bars": 400,
                                                     "as_of": "2026-08-20", "base_signal": 1,
                                                     "forecast": None})

    assert bot.step() == "entry_blocked"
    assert broker.get_position(SYMBOL) is None


def test_market_close_while_holding_is_logged(broker, tmp_path, monkeypatch):
    bot = make_bot(broker, tmp_path, product_type="INTRADAY")
    broker.buy(SYMBOL, 10)
    # Pinned rather than relying on the wall clock: this test passed at night
    # and failed during NSE hours.
    monkeypatch.setattr(broker, "is_market_open", lambda: False)

    assert bot.step() == "market_closed"
    assert any("auto-square-off" in e.message for e in bot.events if e.kind == "error")


def test_journal_records_the_product_type(broker, tmp_path):
    import json

    bot = make_bot(broker, tmp_path, product_type="INTRADAY")
    payload = json.loads(bot.save_journal().read_text(encoding="utf-8"))
    assert payload["product_type"] == "INTRADAY"


# ----------------------------------------------------------------------
# access-token expiry
# ----------------------------------------------------------------------
def make_jwt(exp_epoch: int | None) -> str:
    """A JWT-shaped token with the given exp claim. Signature is irrelevant:
    only the payload is read, and never sent anywhere."""
    import base64
    import json as _json

    def seg(obj):
        raw = _json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    claims = {"dhanClientId": "X", "iss": "dhan"}
    if exp_epoch is not None:
        claims["exp"] = exp_epoch
    return f"{seg({'alg': 'HS256'})}.{seg(claims)}.signature"


def test_expiry_is_read_from_the_token():
    from datetime import datetime, timezone

    from algobot.broker.dhan import token_expiry

    exp = int(datetime(2026, 8, 20, 16, 10, tzinfo=timezone.utc).timestamp())
    assert token_expiry(make_jwt(exp)) == datetime(2026, 8, 20, 16, 10, tzinfo=timezone.utc)


def test_expired_and_valid_tokens_are_distinguished():
    from datetime import datetime, timedelta, timezone

    from algobot.broker.dhan import token_is_expired

    now = datetime.now(timezone.utc)
    past = int((now - timedelta(hours=1)).timestamp())
    future = int((now + timedelta(hours=10)).timestamp())

    assert token_is_expired(make_jwt(past)) is True
    assert token_is_expired(make_jwt(future)) is False


def test_unreadable_tokens_return_none_rather_than_guessing():
    """An opaque token must not be reported as valid or expired."""
    from algobot.broker.dhan import token_expiry, token_is_expired

    for junk in ("", "not-a-jwt", "a.b", "a.b.c.d", "!!!.???.***"):
        assert token_expiry(junk) is None
        assert token_is_expired(junk) is None


def test_token_without_exp_claim_returns_none():
    from algobot.broker.dhan import token_expiry

    assert token_expiry(make_jwt(None)) is None


def test_expiry_check_never_returns_the_token():
    from algobot.broker.dhan import token_expiry

    secret = make_jwt(1_800_000_000)
    assert secret not in str(token_expiry(secret))
