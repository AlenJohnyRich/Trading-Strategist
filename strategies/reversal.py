"""
Reversal strategy implementation using EMA(21,34,144) and Stochastic(7,3,3).
Provides signal generation functions and a simple backtester utility.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config import get_config

cfg = get_config()

@dataclass
class Signal:
    timestamp: pd.Timestamp
    side: str  # 'long' or 'short'
    entry_price: float
    stop_price: float
    target_price: float


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def stochastic_oscillator(df: pd.DataFrame, k_period: int, k_smooth: int, d_smooth: int):
    low_min = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    k = k.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d


def generate_signals(df: pd.DataFrame):
    """Generate entry signals. Expects df with columns: open, high, low, close, volume and a datetime index."""
    df = df.copy()
    # EMAs
    df['ema21'] = ema(df['close'], cfg.ema.short_window)
    df['ema34'] = ema(df['close'], cfg.ema.mid_window)
    df['ema144'] = ema(df['close'], cfg.ema.long_window)

    # Stochastic
    k, d = stochastic_oscillator(df, cfg.stochastic.k_period, cfg.stochastic.k_smooth, cfg.stochastic.d_smooth)
    df['stoch_k'] = k
    df['stoch_d'] = d

    signals = []
    traded_after_trend = False

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        # Trend confirmation
        price = row['close']
        ema21 = row['ema21']
        ema34 = row['ema34']
        ema144 = row['ema144']

        # Require non-NaN EMAs
        if np.isnan(ema21) or np.isnan(ema34) or np.isnan(ema144):
            continue

        # Detect fresh trend change: price crossing ema144
        # Use previous candle to ensure first confirmation candle
        prev_price = prev['close']
        trend_up = (prev_price <= prev['ema144']) and (price > ema144) and (ema21 > ema34 > ema144)
        trend_down = (prev_price >= prev['ema144']) and (price < ema144) and (ema21 < ema34 < ema144)

        if trend_up:
            traded_after_trend = False
            trend_dir = 'up'
        elif trend_down:
            traded_after_trend = False
            trend_dir = 'down'
        else:
            trend_dir = None

        # Only consider first pullback after trend confirmation
        # If traded_after_trend True skip further entries until trend flips
        if trend_dir and not traded_after_trend:
            # Wait for pullback and stochastic confirmation
            # For long: stochastic k < 20 then k crosses above d and closes above 20
            k_val = row['stoch_k']
            d_val = row['stoch_d']
            prev_k = prev['stoch_k']
            prev_d = prev['stoch_d']
            if trend_dir == 'up':
                if k_val is not np.nan and prev_k is not np.nan:
                    # condition: k was below 20, then crosses above d and closes above 20
                    if (prev_k < 20) and (prev_k < prev_d) and (k_val > d_val) and (k_val > 20):
                        entry = price
                        stop = min(df['low'].iloc[max(0, i-5):i+1])  # pullback low (simple heuristic)
                        risk = entry - stop
                        target = entry + cfg.trading.target_rr * risk
                        signals.append(Signal(timestamp=row.name, side='long', entry_price=entry, stop_price=stop, target_price=target))
                        traded_after_trend = True
            elif trend_dir == 'down':
                if k_val is not np.nan and prev_k is not np.nan:
                    if (prev_k > 80) and (prev_k > prev_d) and (k_val < d_val) and (k_val < 80):
                        entry = price
                        stop = max(df['high'].iloc[max(0, i-5):i+1])
                        risk = stop - entry
                        target = entry - cfg.trading.target_rr * risk
                        signals.append(Signal(timestamp=row.name, side='short', entry_price=entry, stop_price=stop, target_price=target))
                        traded_after_trend = True

    return signals


def simple_backtest(df: pd.DataFrame, signals: list[Signal]):
    """Runs a simple backtest that assumes immediate fill at entry price and exit at target or stop."""
    results = []
    for s in signals:
        # find next bars after s.timestamp to see if stop or target hit first
        window = df.loc[s.timestamp:]
        hit_price = None
        outcome = None
        for _, r in window.iterrows():
            lo = r['low']
            hi = r['high']
            if s.side == 'long':
                if lo <= s.stop_price:
                    hit_price = s.stop_price
                    outcome = - (s.entry_price - hit_price)
                    break
                if hi >= s.target_price:
                    hit_price = s.target_price
                    outcome = s.target_price - s.entry_price
                    break
            else:
                if hi >= s.stop_price:
                    hit_price = s.stop_price
                    outcome = - (hit_price - s.entry_price)
                    break
                if lo <= s.target_price:
                    hit_price = s.target_price
                    outcome = s.entry_price - s.target_price
                    break
        results.append({'signal': s, 'pnl': outcome if outcome is not None else 0})
    return results


if __name__ == '__main__':
    print('Module test: load a CSV with columns: open,high,low,close,volume and run generate_signals')
