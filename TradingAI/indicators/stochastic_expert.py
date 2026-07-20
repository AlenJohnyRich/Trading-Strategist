"""
Stochastic Expert: computes stochastic oscillator and provides utility signals.
"""
from __future__ import annotations

import pandas as pd


def stochastic(df: pd.DataFrame, k_period: int = 7, k_smooth: int = 3, d_smooth: int = 3):
    low_min = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    k = k.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d


def stochastic_signals(df: pd.DataFrame, k_period: int = 7):
    k, d = stochastic(df, k_period)
    sig = pd.Series(index=df.index, data=0)
    # simple cross signals
    cross_up = (k > d) & (k.shift(1) <= d.shift(1))
    cross_down = (k < d) & (k.shift(1) >= d.shift(1))
    sig[cross_up] = 1
    sig[cross_down] = -1
    return sig
