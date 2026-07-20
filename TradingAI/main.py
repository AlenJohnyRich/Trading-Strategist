"""
Main entrypoint for the TradingAI package. Provides a CLI-like run function.
"""
from __future__ import annotations

import argparse
import logging
import asyncio

from TradingAI.config import get_config
from TradingAI.market_data.live_feed import LiveFeedManager
from TradingAI.strategies.engine import StrategyEngine

logger = logging.getLogger('TradingAI.main')


def run_backtest(csv_path: str):
    import pandas as pd
    df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
    engine = StrategyEngine()
    results = engine.backtest(df)
    from TradingAI.analytics.performance import aggregate_results
    print(aggregate_results(results))


async def run_live():
    cfg = get_config()
    mgr = LiveFeedManager()

    async def on_message(msg):
        print('live', msg)

    await mgr.connect_polygon(api_key='')
    await mgr.connect_binance(api_key='')
    await mgr.connect_alpaca(api_key='')

    # run forever
    while True:
        await asyncio.sleep(1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', help='CSV file to backtest on')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if args.backtest:
        run_backtest(args.backtest)
    elif args.live:
        asyncio.run(run_live())
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
