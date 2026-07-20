"""
Configuration module for Trading-Strategist (trimmed copy of full config).
"""
from __future__ import annotations

import os
import pathlib
import logging
from functools import lru_cache
from pydantic import BaseSettings, Field, SecretStr
from pydantic import BaseModel

logger = logging.getLogger("trading_config")

class PostgresConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "trading"
    user: str = "trading"
    password: SecretStr | None = None

class EMAConfig(BaseModel):
    short_window: int = 21
    mid_window: int = 34
    long_window: int = 144

class StochasticConfig(BaseModel):
    k_period: int = 7
    k_smooth: int = 3
    d_smooth: int = 3

class TradingParams(BaseModel):
    risk_per_trade_pct: float = 0.01
    target_rr: float = 2.0

class AppConfig(BaseSettings):
    app_name: str = "Trading-Strategist"
    env: str = "production"
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    ema: EMAConfig = Field(default_factory=EMAConfig)
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    trading: TradingParams = Field(default_factory=TradingParams)

    class Config:
        env_prefix = "TS_"
        env_nested_delimiter = "__"

    def summary(self) -> dict:
        return {"app_name": self.app_name, "env": self.env}

@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig()
