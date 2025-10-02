from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from typing import Literal
import httpx
import pandas as pd
from ..config import settings
from ..types import Candle, PriceQuote, Timeframe
from .base import MarketDataProvider

_TF_MAP: dict[Timeframe, str] = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "60min": "60min",
}

class AlphaVantageProvider(MarketDataProvider):
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None, timeout: float = 15.0):
        self.api_key = api_key or settings.alpha_vantage_api_key
        self._client = httpx.AsyncClient(timeout=timeout)

    async def get_recent_candles(self, symbol: str, timeframe: Timeframe, limit: int = 500) -> list[Candle]:
        if not self.api_key:
            raise RuntimeError("AlphaVantage API key absent. Set ALPHA_VANTAGE_API_KEY in .env")
        interval = _TF_MAP[timeframe]
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": symbol[:3],
            "to_symbol": symbol[3:],
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": "full" if limit > 100 else "compact",
        }
        r = await self._client.get(self.BASE_URL, params=params)
        r.raise_for_status()
        data = r.json()
        key = f"Time Series FX ({interval})"
        if key not in data:
            raise RuntimeError(f"Unexpected response: {list(data.keys())[:3]}")
        df = (
            pd.DataFrame.from_dict(data[key], orient="index")
            .rename(columns={"1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close"})
            .astype(float)
            .sort_index()
        )
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.tail(limit)
        candles: list[Candle] = [
            Candle(time=ts.to_pydatetime(), open=row.open, high=row.high, low=row.low, close=row.close, volume=None)
            for ts, row in df.iterrows()
        ]
        return candles

    async def get_quote(self, symbol: str) -> PriceQuote:
        # AlphaVantage nu oferă mereu bid/ask; aproximăm din ultima lumânare cu spread fix mic
        candles = await self.get_recent_candles(symbol, "1min", limit=1)
        last = candles[-1]
        mid = last.close
        spread = 0.0001
        return PriceQuote(bid=mid - spread/2, ask=mid + spread/2, time=datetime.now(timezone.utc))

    async def close(self) -> None:
        await self._client.aclose()
