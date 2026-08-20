"""Realistic Dhan / NSE equity transaction costs.

The paper assumes "reduced exchange costs" and its tables are gross of
everything. On Indian equities the statutory charges are not a rounding error:
STT alone is 0.1% per side on delivery, which on a round trip is 0.2% of
turnover before any broker fee.

Charges are applied per leg, because they are asymmetric -- STT falls on both
sides for delivery but sell-side only for intraday, stamp duty is buy-side
only, and DP charges hit the delivery sell alone.

Rate snapshot: 2026-08. **Verify against Dhan's current pricing and the
exchange circulars before trusting any number this produces** -- brokerage
plans, exchange transaction charges and stamp duty all change, and every rate
here is a constructor argument precisely so it can be corrected without
touching the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ChargeBreakdown:
    """Every component of one leg's cost, in rupees."""

    brokerage: float = 0.0
    stt: float = 0.0
    exchange_txn: float = 0.0
    sebi: float = 0.0
    stamp_duty: float = 0.0
    gst: float = 0.0
    dp_charges: float = 0.0

    @property
    def total(self) -> float:
        return (self.brokerage + self.stt + self.exchange_txn + self.sebi
                + self.stamp_duty + self.gst + self.dp_charges)

    def to_dict(self) -> dict:
        d = {k: round(v, 4) for k, v in asdict(self).items()}
        d["total"] = round(self.total, 4)
        return d

    def __add__(self, other: "ChargeBreakdown") -> "ChargeBreakdown":
        return ChargeBreakdown(
            brokerage=self.brokerage + other.brokerage,
            stt=self.stt + other.stt,
            exchange_txn=self.exchange_txn + other.exchange_txn,
            sebi=self.sebi + other.sebi,
            stamp_duty=self.stamp_duty + other.stamp_duty,
            gst=self.gst + other.gst,
            dp_charges=self.dp_charges + other.dp_charges,
        )


@dataclass(frozen=True)
class CostModel:
    """Percentage rates are fractions of turnover (0.001 == 0.1%)."""

    name: str = "delivery"

    # Broker
    brokerage_pct: float = 0.0
    brokerage_cap: float = 20.0          # rupees per order, whichever is lower
    brokerage_min: float = 0.0

    # Securities Transaction Tax
    stt_buy_pct: float = 0.001
    stt_sell_pct: float = 0.001

    # Exchange + regulator
    exchange_txn_pct: float = 0.0000297  # NSE equity cash
    sebi_pct: float = 0.000001           # Rs 10 per crore
    ipft_pct: float = 0.0                # NSE investor protection fund, cash: nil

    # State stamp duty, buy side only
    stamp_duty_buy_pct: float = 0.00015

    # GST on brokerage + exchange + SEBI + IPFT
    gst_pct: float = 0.18

    # Depository charge, levied per scrip per day on a delivery sell
    dp_charge_per_sell: float = 12.5
    dp_charge_taxable: bool = True

    def brokerage_for(self, turnover: float) -> float:
        """Percentage brokerage, capped per order (Dhan's 'whichever is lower')."""
        if self.brokerage_pct <= 0:
            return max(0.0, self.brokerage_min)
        return max(self.brokerage_min, min(turnover * self.brokerage_pct, self.brokerage_cap))

    def charges(self, price: float, quantity: int, side: str) -> ChargeBreakdown:
        """Cost of one leg. `side` is BUY or SELL."""
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {side!r}")

        turnover = abs(price) * abs(quantity)
        if turnover == 0:
            return ChargeBreakdown()

        is_buy = side == "BUY"

        brokerage = self.brokerage_for(turnover)
        stt = turnover * (self.stt_buy_pct if is_buy else self.stt_sell_pct)
        exchange_txn = turnover * self.exchange_txn_pct
        sebi = turnover * self.sebi_pct
        ipft = turnover * self.ipft_pct
        stamp_duty = turnover * self.stamp_duty_buy_pct if is_buy else 0.0
        dp = 0.0 if is_buy else self.dp_charge_per_sell

        # GST applies to the service components, never to STT or stamp duty.
        taxable = brokerage + exchange_txn + sebi + ipft
        if self.dp_charge_taxable:
            taxable += dp
        gst = taxable * self.gst_pct

        return ChargeBreakdown(
            brokerage=brokerage,
            stt=stt,
            exchange_txn=exchange_txn + ipft,
            sebi=sebi,
            stamp_duty=stamp_duty,
            gst=gst,
            dp_charges=dp,
        )

    def round_trip(self, buy_price: float, sell_price: float, quantity: int) -> ChargeBreakdown:
        return self.charges(buy_price, quantity, "BUY") + self.charges(sell_price, quantity, "SELL")

    def breakeven_pct(self, price: float, quantity: int) -> float:
        """Move required just to cover costs, as a fraction of entry price.

        The single most useful number here: any strategy whose average winning
        move is smaller than this cannot be profitable regardless of win rate.
        """
        turnover = abs(price) * abs(quantity)
        if turnover == 0:
            return 0.0
        return self.round_trip(price, price, quantity).total / turnover

    def describe(self) -> str:
        parts = [f"{self.name}"]
        if self.brokerage_pct > 0:
            parts.append(f"brokerage {self.brokerage_pct:.3%} (cap Rs {self.brokerage_cap:g})")
        else:
            parts.append("brokerage nil")
        parts.append(f"STT {self.stt_buy_pct:.3%} buy / {self.stt_sell_pct:.3%} sell")
        parts.append(f"stamp {self.stamp_duty_buy_pct:.4%} buy")
        if self.dp_charge_per_sell:
            parts.append(f"DP Rs {self.dp_charge_per_sell:g}/sell")
        return ", ".join(parts)


# Dhan equity delivery: zero brokerage, but STT on both legs and a DP charge
# on every sell make this the more expensive model for short holds.
DELIVERY = CostModel(
    name="delivery",
    brokerage_pct=0.0,
    stt_buy_pct=0.001,
    stt_sell_pct=0.001,
    stamp_duty_buy_pct=0.00015,
    dp_charge_per_sell=12.5,
)

# Dhan intraday: Rs 20 per order or 0.03%, whichever is lower. STT sell-side
# only at a quarter the delivery rate, lower stamp duty, and no DP charge.
INTRADAY = CostModel(
    name="intraday",
    brokerage_pct=0.0003,
    brokerage_cap=20.0,
    stt_buy_pct=0.0,
    stt_sell_pct=0.00025,
    stamp_duty_buy_pct=0.00003,
    dp_charge_per_sell=0.0,
)

# Gross of everything -- reproduces the paper's own tables.
ZERO = CostModel(
    name="none",
    brokerage_pct=0.0,
    stt_buy_pct=0.0,
    stt_sell_pct=0.0,
    exchange_txn_pct=0.0,
    sebi_pct=0.0,
    stamp_duty_buy_pct=0.0,
    gst_pct=0.0,
    dp_charge_per_sell=0.0,
)

PRESETS = {"delivery": DELIVERY, "intraday": INTRADAY, "none": ZERO, "cnc": DELIVERY,
           "mis": INTRADAY, "zero": ZERO}


def get_cost_model(name: str | CostModel | None) -> CostModel:
    """Resolve a preset name (or pass a CostModel straight through)."""
    if isinstance(name, CostModel):
        return name
    if name is None:
        return ZERO
    key = str(name).lower()
    if key not in PRESETS:
        raise KeyError(f"Unknown cost model {name!r}. Available: {sorted(set(PRESETS))}")
    return PRESETS[key]
