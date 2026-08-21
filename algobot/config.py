"""Central configuration for the Algorithmic Trading Bot.

Values follow the paper (Mathur, Mhadalekar, Mhatre, Mane -- ITM Web of
Conferences 40, 03041, 2021) wherever the paper is explicit, and use
documented defaults where it is not.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"

for _d in (CACHE_DIR, MODEL_DIR, ARTIFACT_DIR):
    _d.mkdir(exist_ok=True)


@dataclass
class DataConfig:
    """Section III of the paper, sourced from Dhan (NSE) instead of Alpaca."""

    symbol: str = "RELIANCE"     # NSE trading symbol
    exchange_segment: str = "NSE_EQ"
    instrument: str = "EQUITY"
    period: str = "10y"
    interval: str = "1d"
    source: str = "dhan"         # dhan | yahoo | synthetic
    yahoo_suffix: str = ".NS"    # NSE tickers on Yahoo Finance
    use_cache: bool = True


@dataclass
class PreprocessConfig:
    """Section IV.A: 41 lags, X = columns [0:33], Y = the rest."""

    n_lags: int = 41
    n_input_lags: int = 33       # the "[0:33]" split point
    train_ratio: float = 0.60    # Section IV.B: 60:40
    normalize: bool = True
    """Scale each window by its anchor close so the model learns *ratios*.

    The paper trains on raw price levels. Decision trees cannot extrapolate
    beyond the values seen in training, so on any instrument that trends out of
    its historical range the 60:40 chronological split leaves most test targets
    unreachable and R^2 goes negative. Normalising makes the target
    scale-free and restores generalisation. Set to False to reproduce the
    paper's literal behaviour.
    """


@dataclass
class ModelConfig:
    """Section IV.C: RandomForestRegressor."""

    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 1
    max_features: str | float = 1.0
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class BacktestConfig:
    quantity: int = 10           # shares per trade
    initial_cash: float = 100_000.0
    currency: str = "INR"
    cost_model: str = "delivery"
    """Transaction costs: delivery (CNC) | intraday (MIS) | none.

    The paper reports gross figures ("reduced exchange costs"); `none`
    reproduces that. On NSE equities the statutory charges are material -- STT
    alone is 0.1% per side on delivery -- so the default applies them.
    """

    commission: float = 0.0      # extra flat fee per order, on top of cost_model

    slippage_bps: float = 5.0
    """Execution slippage per side, in basis points of the fill price.

    Backtests fill at a bar's open, which assumes the trader captured a price
    that in reality is an auction print they were not part of. 5 bps per side
    (10 bps round trip) is a modest allowance for spread and impact on a liquid
    NSE large-cap at small size; illiquid names and larger orders need more.
    Set 0 to reproduce the paper's frictionless fills.
    """

    intraday_square_off: bool = False
    """Close any open position at the last bar of each session.

    Required for an honest MIS backtest: Dhan force-closes intraday positions
    at the bell, so a test that carries them overnight is measuring a strategy
    the broker would never have let run. Ignored on daily bars, where every
    bar is its own session.
    """

    gap_through_stops: bool = True
    """Fill a stop at the open when the bar gapped through it.

    Filling every stop exactly at the stop level assumes a fill is always
    available there, which is precisely untrue on the gap-down days that
    trigger stops in the first place. With this on, a bar that opens beyond
    the stop fills at the open instead -- the worse, realistic price.
    """
    stop_loss_pct: float = 0.05  # per-trade protective stop
    take_profit_pct: float | None = None
    allow_short: bool = False


@dataclass
class BotConfig:
    """Section IV.F: live loop against a paper-trading account."""

    symbol: str = "RELIANCE"
    exchange_segment: str = "NSE_EQ"
    product_type: str = "CNC"            # CNC (delivery) | INTRADAY (MIS) | MARGIN
    """Dhan product type for orders.

    CNC is the default because every strategy here runs on daily bars and holds
    for ~40 days on average. Sending those orders as INTRADAY (MIS) would have
    Dhan auto-square-off the position the same afternoon, so the order type
    would silently contradict the strategy's own exit logic.

    Set INTRADAY only alongside intraday bars (`data.interval`) and strategy
    windows tuned for them; `auto_square_off` then applies.
    """

    auto_square_off: bool = True
    """Flatten before the session ends when trading INTRADAY.

    Dhan force-closes MIS positions itself around 15:20 IST, at whatever price
    the book offers. Squaring off first keeps the exit under the bot's control
    and inside its own journal.
    """

    square_off_time: str = "15:15"        # IST, ahead of Dhan's ~15:20 MIS cutoff
    no_new_entries_after: str = "15:00"   # IST, stop opening what must close today
    strategy: str = "ma"
    quantity: int = 10
    poll_seconds: int = 60
    day_stop_loss_pct: float = 0.02      # halt trading after -2% on the day
    day_take_profit_pct: float | None = None
    trade_stop_loss_pct: float = 0.02
    use_model: bool = True
    model_path: str | None = None
    broker: str = "sim"                  # sim | dhan
    stop_file: str = str(ROOT / "STOP")
    dry_run: bool = False
    max_trades_per_day: int = 20


@dataclass
class DhanCredentials:
    """DhanHQ API v2 credentials (https://dhanhq.co/docs/v2/).

    `access_token` is the JWT generated from the Dhan web/app under
    DhanHQ Trading API; `client_id` is the Dhan client code it belongs to.
    Both are read from the environment so they never land in the repo.
    """

    access_token: str = field(default_factory=lambda: os.getenv("DHAN_ACCESS_TOKEN", ""))
    client_id: str = field(default_factory=lambda: os.getenv("DHAN_CLIENT_ID", ""))
    base_url: str = field(
        default_factory=lambda: os.getenv("DHAN_BASE_URL", "https://api.dhan.co/v2")
    )
    scrip_master_url: str = field(
        default_factory=lambda: os.getenv(
            "DHAN_SCRIP_MASTER_URL",
            "https://images.dhan.co/api-data/api-scrip-master.csv",
        )
    )

    @property
    def configured(self) -> bool:
        return bool(self.access_token and self.client_id)

    def headers(self) -> dict[str, str]:
        return {
            "access-token": self.access_token,
            "client-id": self.client_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    bot: BotConfig = field(default_factory=BotConfig)
    dhan: DhanCredentials = field(default_factory=DhanCredentials)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dhan"] = {"base_url": self.dhan.base_url, "configured": self.dhan.configured}
        return d


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load a Config, optionally overlaying a YAML file."""
    cfg = Config()
    if path is None:
        return cfg

    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    for section, values in raw.items():
        target = getattr(cfg, section, None)
        if target is None or not isinstance(values, dict):
            continue
        for k, v in values.items():
            if hasattr(target, k):
                setattr(target, k, v)
    return cfg
