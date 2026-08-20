"""Strategy registry -- name -> constructor, used by the CLI and the bot."""

from __future__ import annotations

from .base import Strategy
from .donchian import Donchian
from .gold_cross import GoldCross
from .ml_filter import ModelFilteredStrategy
from .moving_average import MovingAverageCrossover
from .multiple import MultipleStrategy

REGISTRY = {
    "ma": MovingAverageCrossover,
    "donchian": Donchian,
    "multiple": MultipleStrategy,
    "gold": GoldCross,
}

# The four the paper backtests, in the order of Tables 3-6.
PAPER_STRATEGIES = ["ma", "donchian", "multiple", "gold"]

PRETTY_NAMES = {
    "ma": "Moving Average Crossover",
    "donchian": "Donchian",
    "multiple": "Multiple Strategy",
    "gold": "Gold Cross",
    "ml": "Model-filtered",
}


def build_strategy(name: str, **params) -> Strategy:
    """Instantiate a registered strategy, passing through only known params."""
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}. Available: {', '.join(REGISTRY)}")
    cls = REGISTRY[key]
    valid = {f for f in cls.__dataclass_fields__ if f != "name"}
    return cls(**{k: v for k, v in params.items() if k in valid and v is not None})


def wrap_with_model(strategy: Strategy, model, **params) -> ModelFilteredStrategy:
    """Section IV.F fusion of a financial strategy with the RF model."""
    valid = {f for f in ModelFilteredStrategy.__dataclass_fields__
             if f not in ("name", "base", "model")}
    kwargs = {k: v for k, v in params.items() if k in valid and v is not None}
    return ModelFilteredStrategy(base=strategy, model=model, **kwargs)


def pretty(name: str) -> str:
    """Human-readable strategy label, including '<base>+rf' fusion variants."""
    key = name.lower()
    if key.endswith("+rf"):
        return f"{PRETTY_NAMES.get(key[:-3], key[:-3])} + RF"
    return PRETTY_NAMES.get(key, name)
