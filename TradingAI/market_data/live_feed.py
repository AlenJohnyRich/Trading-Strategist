"""
Live data feed manager (connects to Polygon, Binance, Alpaca via websockets/REST when needed).
This module provides a high-level async LiveFeedManager with automatic reconnects and
basic validation hooks. The implementation uses placeholders where exchange-specific
SDKs would normally be used, so it's safe to run in environments without those SDKs.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Callable, Awaitable, Optional

logger = logging.getLogger("TradingAI.live_feed")


class ValidationError(Exception):
    pass


class LiveFeedManager:
    """Manages multiple live websocket connections with automatic reconnect.

    Usage:
        mgr = LiveFeedManager()
        mgr.register_validator(my_validator)
        await mgr.connect_polygon(api_key=..., on_message=...)
    """

    def __init__(self, reconnect_interval: int = 5):
        self.reconnect_interval = reconnect_interval
        self._tasks: dict[str, asyncio.Task] = {}
        self._validators: list[Callable[[dict], bool]] = []
        self._running = False

    def register_validator(self, fn: Callable[[dict], bool]) -> None:
        self._validators.append(fn)

    def _validate(self, msg: dict) -> bool:
        for v in self._validators:
            try:
                if not v(msg):
                    return False
            except Exception:
                logger.exception("Validator raised an exception")
                return False
        return True

    async def _fake_stream(self, name: str, on_message: Callable[[dict], Awaitable[None]]):
        """Simulate a stream by sending random messages every second. Replace with real websocket logic."""
        while True:
            await asyncio.sleep(1 + random.random())
            msg = {"source": name, "price": random.random() * 100, "timestamp": asyncio.get_event_loop().time()}
            if self._validate(msg):
                try:
                    await on_message(msg)
                except Exception:
                    logger.exception("on_message handler failed")

    async def connect_polygon(self, api_key: str, on_message: Callable[[dict], Awaitable[None]]):
        """Start a polygon stream task (placeholder)."""
        name = "polygon"
        if name in self._tasks:
            logger.info("Polygon task already running")
            return
        task = asyncio.create_task(self._run_with_reconnect(name, lambda: self._fake_stream(name, on_message)))
        self._tasks[name] = task

    async def connect_binance(self, api_key: str, on_message: Callable[[dict], Awaitable[None]]):
        name = "binance"
        if name in self._tasks:
            logger.info("Binance task already running")
            return
        task = asyncio.create_task(self._run_with_reconnect(name, lambda: self._fake_stream(name, on_message)))
        self._tasks[name] = task

    async def connect_alpaca(self, api_key: str, on_message: Callable[[dict], Awaitable[None]]):
        name = "alpaca"
        if name in self._tasks:
            logger.info("Alpaca task already running")
            return
        task = asyncio.create_task(self._run_with_reconnect(name, lambda: self._fake_stream(name, on_message)))
        self._tasks[name] = task

    async def _run_with_reconnect(self, name: str, coro_factory: Callable[[], Awaitable[None]]):
        logger.info("Starting stream %s", name)
        while True:
            try:
                await coro_factory()
            except asyncio.CancelledError:
                logger.info("Stream %s cancelled", name)
                break
            except Exception:
                logger.exception("Stream %s failed, reconnecting in %s seconds", name, self.reconnect_interval)
                await asyncio.sleep(self.reconnect_interval)

    async def stop(self):
        for t in list(self._tasks.values()):
            t.cancel()
        self._tasks.clear()


if __name__ == '__main__':
    async def demo():
        async def on_msg(m):
            print('MSG', m)
        mgr = LiveFeedManager()
        await mgr.connect_polygon('fake', on_msg)
        await asyncio.sleep(3)
        await mgr.stop()
    asyncio.run(demo())
