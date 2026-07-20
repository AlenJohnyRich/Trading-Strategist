"""
EMA Expert: helper functions and small expert advisor that analyses EMA relationships.
"""
from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def ema_trend(df: pd.DataFrame, short: int = 21, mid: int = 34, long: int = 144) -> pd.Series:
    """Return series of trend labels: 1 for bullish, -1 for bearish, 0 neutral."""
    e1 = ema(df['close'], short)
    e2 = ema(df['close'], mid)
    e3 = ema(df['close'], long)
    cond_up = (e1 > e2) & (e2 > e3)
    cond_down = (e1 < e2) & (e2 < e3)
    out = pd.Series(0, index=df.index)
    out[cond_up] = 1
    out[cond_down] = -1
    return out
