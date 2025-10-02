from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable
from datetime import datetime
from ..types import Candle, PriceQuote, Timeframe

class MarketDataProvider(ABC):
    """Interfața comună pentru furnizori de date (live/backfill)."""

    @abstractmethod
    async def get_recent_candles(
        self, symbol: str, timeframe: Timeframe, limit: int = 500
    ) -> list[Candle]:
        """Returnează ultimele `limit` lumânări (candles)."""

    @abstractmethod
    async def get_quote(self, symbol: str) -> PriceQuote:
        """Returnează cotația curentă (bid/ask)."""

    async def close(self) -> None:
        """Curățenie; override dacă ai sesiuni HTTP etc."""
        return None
