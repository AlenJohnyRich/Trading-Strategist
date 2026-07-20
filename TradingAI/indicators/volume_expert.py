"""
Volume expert utilities.
"""
from __future__ import annotations

import pandas as pd


def volume_spike(df: pd.DataFrame, lookback: int = 20, multiplier: float = 1.5):
    avg = df['volume'].rolling(lookback).mean()
    return df['volume'] > avg * multiplier
