"""
Performance analytics - simple P&L aggregation utilities.
"""
from __future__ import annotations

import pandas as pd
from typing import List


def aggregate_results(results: List[dict]) -> dict:
    pnl = [r['pnl'] for r in results]
    s = pd.Series(pnl)
    return {'trades': len(pnl), 'total_pnl': s.sum(), 'win_rate': (s>0).mean() if len(s)>0 else 0}
