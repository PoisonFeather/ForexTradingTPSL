from __future__ import annotations
from datetime import datetime, timedelta, timezone
import math
import random
from ..types import Candle, PriceQuote, Timeframe
from .base import MarketDataProvider

class MockProvider(MarketDataProvider):
    def __init__(self, seed: int = 42):
        self._rnd = random.Random(seed)

    async def get_recent_candles(self, symbol: str, timeframe: Timeframe, limit: int = 500) -> list[Candle]:
        now = datetime.now(timezone.utc)
        tf_minutes = int(timeframe.replace("min", "")) if timeframe.endswith("min") else 5
        candles: list[Candle] = []
        price = 1.1000 if symbol.upper().startswith("EURUSD") else 1.2500

        for i in range(limit):
            t = now - timedelta(minutes=tf_minutes * (limit - i))
            drift = 0.00001 * (i / 50.0)
            noise = (self._rnd.random() - 0.5) * 0.0008
            price = max(0.1, price + drift + noise + 0.0005 * math.sin(i / 12))
            high = price + abs(noise) * 0.5
            low = price - abs(noise) * 0.5
            open_ = price - noise * 0.3
            close = price + noise * 0.3
            candles.append(Candle(time=t, open=open_, high=high, low=low, close=close, volume=None))
        return candles

    async def get_quote(self, symbol: str) -> PriceQuote:
        now = datetime.now(timezone.utc)
        mid = 1.1000 if symbol.upper().startswith("EURUSD") else 1.2500
        spread = 0.00008
        return PriceQuote(bid=mid - spread/2, ask=mid + spread/2, time=now)
