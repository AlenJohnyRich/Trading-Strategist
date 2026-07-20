"""
Risk Manager - basic position sizing and stop/target utilities.
"""
from __future__ import annotations

from TradingAI.config import get_config

cfg = get_config()

class RiskManager:
    def __init__(self, account_size: float = 10000.0):
        self.account_size = account_size

    def position_size(self, entry: float, stop: float) -> float:
        risk_amt = self.account_size * cfg.trading.risk_per_trade_pct
        stop_distance = abs(entry - stop)
        if stop_distance == 0:
            return 0.0
        return risk_amt / stop_distance
