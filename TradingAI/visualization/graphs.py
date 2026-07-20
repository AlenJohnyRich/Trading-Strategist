"""
Graphing utilities for visualizing price, EMAs, stochastic oscillator and model predictions.
Generates PNG files saved to a directory and returns the path for downstream use.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
from typing import Optional, List


def plot_price_and_indicators(df: pd.DataFrame, signals: Optional[List] = None, preds: Optional[np.ndarray] = None, out_path: str = './plots/plot.png') -> str:
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(df.index, df['close'], label='close', color='black')
    if 'ema21' in df.columns:
        ax1.plot(df.index, df['ema21'], label='ema21', color='white')
    if 'ema34' in df.columns:
        ax1.plot(df.index, df['ema34'], label='ema34', color='orange')
    if 'ema144' in df.columns:
        ax1.plot(df.index, df['ema144'], label='ema144', color='green')

    # plot signals
    if signals:
        for s in signals:
            if s.side == 'long':
                ax1.scatter(s.timestamp, s.entry_price, marker='^', color='green', s=100)
            else:
                ax1.scatter(s.timestamp, s.entry_price, marker='v', color='red', s=100)

    # predictions aligned to the rightmost windows
    if preds is not None and len(preds) > 0:
        # create x for preds near end
        x = df.index[len(df) - len(preds):]
        ax1.plot(x, preds, label='model_pred', color='purple')

    ax1.legend()
    ax1.set_title('Price and EMAs')

    # stochastic
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        ax2.plot(df.index, df['stoch_k'], label='%K')
        ax2.plot(df.index, df['stoch_d'], label='%D')
        ax2.axhline(80, color='red', linestyle='--', alpha=0.4)
        ax2.axhline(20, color='green', linestyle='--', alpha=0.4)
        ax2.legend()

    plt.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return str(p)
