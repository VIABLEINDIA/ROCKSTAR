"""DhanHQ v2 broker client (https://dhanhq.co/docs/v2/).

Endpoints used
--------------
    GET  /fundlimit                 account balance
    GET  /positions                 open positions
    POST /orders                    place an order
    GET  /orders                    order book
    POST /marketfeed/ltp            last traded price
    POST /charts/historical         daily candles

Dhan has no separate paper-trading host: the same API is the live account.
Guard rails here are deliberate -- `dry_run` logs orders instead of sending
them, and `confirm_live` must be set explicitly before any real order leaves
the process. Use `SimulatedBroker` for the paper-trading workflow the paper
describes.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta

import pandas as pd

from ..config import DhanCredentials
from ..data.instruments import resolve_security_id
from .base import AccountSnapshot, Broker, BrokerError, Order, Position

log = logging.getLogger(__name__)

# NSE equity cash session, IST.
NSE_OPEN = time(9, 15)
NSE_CLOSE = time(15, 30)
IST = "Asia/Kolkata"


def now_ist() -> datetime:
    return pd.Timestamp.now(tz=IST).to_pydatetime()


def is_nse_session(ts: datetime | None = None) -> bool:
    """True during the NSE cash-market session (weekday, 09:15-15:30 IST).

    Exchange holidays are not encoded; Dhan simply rejects orders on those
    days, and `SimulatedBroker` accepts an explicit holiday list.
    """
    ts = ts or now_ist()
    if ts.weekday() >= 5:
        return False
    return NSE_OPEN <= ts.time() <= NSE_CLOSE


class DhanBroker(Broker):
    name = "dhan"
    currency = "INR"

    def __init__(self, creds: DhanCredentials | None = None, exchange_segment: str = "NSE_EQ",
                 product_type: str = "INTRADAY", dry_run: bool = False,
                 confirm_live: bool = False, timeout: int = 30):
        self.creds = creds or DhanCredentials()
        if not self.creds.configured:
            raise BrokerError(
                "Dhan credentials missing: export DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID"
            )
        self.exchange_segment = exchange_segment
        self.product_type = product_type
        self.dry_run = dry_run
        self.confirm_live = confirm_live
        self.timeout = timeout
        self._security_ids: dict[str, int] = {}

        import requests

        self._session = requests.Session()
        self._session.headers.update(self.creds.headers())

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.creds.base_url}{path}"
        resp = self._session.request(method, url, timeout=self.timeout, **kwargs)
        if resp.status_code >= 400:
            raise BrokerError(f"Dhan {method} {path} -> {resp.status_code}: {resp.text[:300]}")
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError as exc:
            raise BrokerError(f"Dhan {path} returned non-JSON: {resp.text[:200]}") from exc

    def security_id(self, symbol: str) -> int:
        key = symbol.upper()
        if key not in self._security_ids:
            self._security_ids[key] = resolve_security_id(key, self.exchange_segment)
        return self._security_ids[key]

    # ------------------------------------------------------------------
    # market state
    # ------------------------------------------------------------------
    def is_market_open(self) -> bool:
        return is_nse_session()

    def last_price(self, symbol: str) -> float:
        sid = self.security_id(symbol)
        payload = self._request(
            "POST", "/marketfeed/ltp", json={self.exchange_segment: [sid]}
        )
        data = payload.get("data", payload)
        segment = data.get(self.exchange_segment, {}) if isinstance(data, dict) else {}
        quote = segment.get(str(sid)) or segment.get(sid) or {}
        price = quote.get("last_price") or quote.get("ltp")
        if price is None:
            raise BrokerError(f"No LTP in Dhan response for {symbol}: {str(payload)[:200]}")
        return float(price)

    def history(self, symbol: str, lookback_days: int = 400) -> pd.DataFrame:
        """Daily candles via /charts/historical, as an OHLCV frame."""
        sid = self.security_id(symbol)
        to_date = date.today()
        from_date = to_date - timedelta(days=lookback_days)
        payload = self._request(
            "POST",
            "/charts/historical",
            json={
                "securityId": str(sid),
                "exchangeSegment": self.exchange_segment,
                "instrument": "EQUITY",
                "expiryCode": 0,
                "oi": False,
                "fromDate": from_date.isoformat(),
                "toDate": to_date.isoformat(),
            },
        )
        if not payload.get("close"):
            raise BrokerError(f"Dhan returned no candles for {symbol}")

        df = pd.DataFrame(
            {
                "Open": payload["open"],
                "High": payload["high"],
                "Low": payload["low"],
                "Close": payload["close"],
                "Volume": payload.get("volume", [0] * len(payload["close"])),
            }
        )
        ts = pd.to_datetime(pd.Series(payload["timestamp"], dtype="float64"), unit="s", utc=True)
        df.index = ts.dt.tz_convert(IST).dt.tz_localize(None).dt.normalize()
        df.index.name = "Date"
        return df.sort_index()

    # ------------------------------------------------------------------
    # account
    # ------------------------------------------------------------------
    def account(self) -> AccountSnapshot:
        funds = self._request("GET", "/fundlimit")
        cash = float(funds.get("availabelBalance", funds.get("availableBalance", 0.0)) or 0.0)
        equity = float(funds.get("sodLimit", cash) or cash)
        return AccountSnapshot(
            cash=cash,
            equity=equity,
            currency=self.currency,
            buying_power=float(funds.get("utilizedAmount", 0.0) or 0.0) + cash,
        )

    def positions(self) -> list[Position]:
        payload = self._request("GET", "/positions") or []
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        out: list[Position] = []
        for row in rows:
            qty = int(row.get("netQty", 0) or 0)
            if qty == 0:
                continue
            out.append(
                Position(
                    symbol=str(row.get("tradingSymbol", "")).upper(),
                    quantity=qty,
                    avg_price=float(row.get("buyAvg") or row.get("costPrice") or 0.0),
                    last_price=float(row.get("lastTradedPrice") or 0.0),
                    security_id=str(row.get("securityId", "")),
                    product_type=str(row.get("productType", "")),
                )
            )
        return out

    def get_position(self, symbol: str) -> Position | None:
        symbol = symbol.upper()
        for pos in self.positions():
            if pos.symbol.upper().startswith(symbol):
                if not pos.last_price:
                    try:
                        pos.last_price = self.last_price(symbol)
                    except BrokerError:
                        pass
                return pos
        return None

    # ------------------------------------------------------------------
    # orders
    # ------------------------------------------------------------------
    def submit_order(self, symbol: str, side: str, quantity: int,
                     order_type: str = "MARKET", price: float = 0.0,
                     trigger_price: float = 0.0, validity: str = "DAY") -> Order:
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {side!r}")
        if quantity <= 0:
            raise ValueError("quantity must be positive")

        body = {
            "dhanClientId": self.creds.client_id,
            "transactionType": side,
            "exchangeSegment": self.exchange_segment,
            "productType": self.product_type,
            "orderType": order_type.upper(),
            "validity": validity,
            "securityId": str(self.security_id(symbol)),
            "quantity": int(quantity),
            "price": float(price),
            "triggerPrice": float(trigger_price),
            "disclosedQuantity": 0,
            "afterMarketOrder": False,
        }

        if self.dry_run:
            log.warning("[DRY RUN] would submit to Dhan: %s", body)
            return Order(order_id="dry-run", symbol=symbol, side=side, quantity=quantity,
                         status="DRY_RUN", price=price, order_type=order_type,
                         submitted_at=now_ist(), raw=body)

        if not self.confirm_live:
            raise BrokerError(
                "Refusing to place a real Dhan order: DhanBroker was built without "
                "confirm_live=True. Trade with --broker sim, add --dry-run, or pass "
                "--i-understand-live-trading to arm the live account."
            )

        payload = self._request("POST", "/orders", json=body)
        return Order(
            order_id=str(payload.get("orderId", "")),
            symbol=symbol,
            side=side,
            quantity=quantity,
            status=str(payload.get("orderStatus", "SUBMITTED")),
            price=price,
            order_type=order_type,
            submitted_at=now_ist(),
            raw=payload,
        )

    def orders(self) -> list[dict]:
        payload = self._request("GET", "/orders") or []
        return payload if isinstance(payload, list) else payload.get("data", [])
