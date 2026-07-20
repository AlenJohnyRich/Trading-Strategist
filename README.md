# Trading-Strategist

This repository now contains a production-ready configuration module and a coded reversal strategy implementing EMA(21,34,144) + Stochastic(7,3,3).

Files added:
- config.py (application configuration)
- strategies/reversal.py (strategy logic and simple backtester)
- requirements.txt

Usage:
1. Install dependencies from requirements.txt
2. Provide OHLCV CSV files and run strategies/reversal.py as a module or import generate_signals from it.
