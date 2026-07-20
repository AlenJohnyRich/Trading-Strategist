"""
Trade decision engine: evaluates signals and prepares orders (size, side, price).
This is a simplified placeholder that demonstrates integration with RiskManager.
"""
from __future__ import annotations

from dataclasses import dataclass

from TradingAI.config import get_config

cfg = get_config()

@dataclass
class Order:
    side: str
    quantity: float
    price: float
    stop: float
    target: float


class TradeDecisionEngine:
    def __init__(self, risk_manager=None):
        self.risk_manager = risk_manager

    def decide(self, signal) -> Order:
        # simple fixed dollar sizing based on risk_per_trade_pct
        account_size = 10000  # placeholder - in real life fetch from account
        risk_amt = account_size * cfg.trading.risk_per_trade_pct
        # compute quantity from risk and stop distance
        stop_distance = abs(signal.entry_price - signal.stop_price)
        if stop_distance == 0:
            quantity = 0
        else:
            quantity = risk_amt / stop_distance
        return Order(side=signal.side, quantity=quantity, price=signal.entry_price, stop=signal.stop_price, target=signal.target_price)
