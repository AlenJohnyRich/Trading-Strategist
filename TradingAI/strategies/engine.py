"""
Strategy engine: glue together indicators, experts, and produce entry/exit decisions.
This module imports the previously implemented reversal strategy and wraps it into
an Engine class exposing a run() method useful for backtesting and live operation.
"""
from __future__ import annotations

from typing import List
import pandas as pd

from TradingAI.strategies.reversal import generate_signals, simple_backtest, Signal


class StrategyEngine:
    def __init__(self, config=None):
        self.config = config

    def run_on_dataframe(self, df: pd.DataFrame) -> List[Signal]:
        return generate_signals(df, cfg_override=self.config)

    def backtest(self, df: pd.DataFrame):
        signals = self.run_on_dataframe(df)
        return simple_backtest(df, signals)
