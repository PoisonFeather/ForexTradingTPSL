from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from datetime import datetime

Timeframe = Literal["1min", "5min", "15min", "30min", "60min"]

@dataclass(slots=True)
class Candle:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

@dataclass(slots=True)
class PriceQuote:
    bid: float
    ask: float
    time: datetime

@dataclass(slots=True)
class StrategySignal:
    name: str
    direction: Literal["long", "short", "neutral"]
    entry: float | None
    stop_loss: float | None
    take_profit: float | None
    confidence: float  # 0..1
    notes: str = ""

@dataclass(slots=True)
class EnsembleDecision:
    symbol: str
    timeframe: Timeframe
    timestamp: datetime
    entry: float | None
    stop_loss: float | None
    take_profit: float | None
    direction: Literal["long", "short", "neutral"]
    confidence: float  # agregat
