"""Algorithmic Trading Bot.

Implementation of "Algorithmic Trading Bot" (Mathur, Mhadalekar, Mhatre, Mane),
ITM Web of Conferences 40, 03041 (2021), targeting the Dhan (DhanHQ v2) broker
and the NSE instead of the paper's Alpaca/US setup.
"""

__version__ = "0.1.0"

from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
