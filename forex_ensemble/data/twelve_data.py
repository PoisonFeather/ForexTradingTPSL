from __future__ import annotations
import httpx
import pandas as pd
from datetime import datetime, timezone
from ..types import Candle, PriceQuote, Timeframe
from .base import MarketDataProvider
from ..config import settings

_TF_MAP: dict[Timeframe, str] = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "60min": "1h",
}

class TwelveDataProvider(MarketDataProvider):
    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, api_key: str | None = None, timeout: float = 15.0):
        self.api_key = api_key or getattr(settings, "twelve_data_api_key", None) or \
                       getattr(settings, "TWELVE_DATA_API_KEY", None)
        if not self.api_key:
            # încearcă și din env direct ca fallback
            import os
            self.api_key = os.getenv("TWELVE_DATA_API_KEY")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def get_recent_candles(self, symbol: str, timeframe: Timeframe, limit: int = 500) -> list[Candle]:
        if not self.api_key:
            raise RuntimeError("TwelveData API key absent. Set TWELVE_DATA_API_KEY in .env")
        interval = _TF_MAP[timeframe]
        # Twelve Data folosește format “EUR/USD” pentru forex
        fx_symbol = f"{symbol[:3]}/{symbol[3:]}"
        params = {
            "symbol": fx_symbol,
            "interval": interval,
            "outputsize": min(limit, 5000),
            "apikey": self.api_key,
            "timezone": "UTC",
            "format": "JSON",
            "dp": 6,  # decimals
        }
        r = await self._client.get(f"{self.BASE_URL}/time_series", params=params)
        r.raise_for_status()
        data = r.json()
        if "values" not in data:
            # eroare prietenoasă
            msg = data.get("message") or str(data)[:200]
            raise RuntimeError(f"TwelveData error: {msg}")
        df = pd.DataFrame(data["values"])
        # coloanele sunt stringuri
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("datetime").tail(limit)
        candles: list[Candle] = [
            Candle(time=row["datetime"].to_pydatetime(),
                   open=float(row["open"]),
                   high=float(row["high"]),
                   low=float(row["low"]),
                   close=float(row["close"]),
                   volume=float(row["volume"]) if "volume" in df.columns and pd.notna(row["volume"]) else None)
            for _, row in df.iterrows()
        ]
        return candles

    async def get_quote(self, symbol: str) -> PriceQuote:
        # Twelve Data quote
        fx_symbol = f"{symbol[:3]}/{symbol[3:]}"
        r = await self._client.get(f"{self.BASE_URL}/price", params={"symbol": fx_symbol, "apikey": self.api_key})
        r.raise_for_status()
        data = r.json()
        if "price" not in data:
            msg = data.get("message") or str(data)[:200]
            raise RuntimeError(f"TwelveData quote error: {msg}")
        mid = float(data["price"])
        spread = 0.0001
        now = datetime.now(timezone.utc)
        return PriceQuote(bid=mid - spread/2, ask=mid + spread/2, time=now)

    async def close(self) -> None:
        await self._client.aclose()
