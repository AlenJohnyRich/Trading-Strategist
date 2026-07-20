"""
Dataset generator - exports trading examples / features for ML training.
"""
from __future__ import annotations

import pandas as pd


def generate_dataset(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    # naive sliding-window generator - each row is features for the next bar
    rows = []
    for i in range(window, len(df)):
        win = df.iloc[i-window:i]
        target = df['close'].iloc[i]
        features = {
            'mean_close': win['close'].mean(),
            'std_close': win['close'].std(),
            'volume_sum': win['volume'].sum(),
            'target': target
        }
        rows.append(features)
    return pd.DataFrame(rows)
