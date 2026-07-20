"""
Extend server to include selfcode router. Keep previous endpoints intact.
"""
from __future__ import annotations

import logging
import asyncio

from fastapi import FastAPI
import uvicorn

from TradingAI.api.server import app as main_app

# Import selfcode router and include
from TradingAI.api.selfcode_api import router as selfcode_router

main_app.include_router(selfcode_router)

app = main_app


def run(host='0.0.0.0', port: int = 8000):
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    run()
