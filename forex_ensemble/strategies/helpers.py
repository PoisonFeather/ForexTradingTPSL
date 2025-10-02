# forex_ensemble/strategies/helpers.py
from __future__ import annotations
from typing import Sequence, Literal
import numpy as np
from ..types import Candle

def to_arrays(candles: Sequence[Candle]):
    closes = np.array([c.close for c in candles], dtype=float)
    highs  = np.array([c.high  for c in candles], dtype=float)
    lows   = np.array([c.low   for c in candles], dtype=float)
    opens  = np.array([c.open  for c in candles], dtype=float)
    return opens, highs, lows, closes

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def price_to_tp_sl(direction: Literal["long","short"], entry: float, atr_val: float,
                   rr: float = 1.8, sl_mult: float = 1.2):
    if direction == "long":
        sl = entry - sl_mult * atr_val
        tp = entry + rr * sl_mult * atr_val
    else:
        sl = entry + sl_mult * atr_val
        tp = entry - rr * sl_mult * atr_val
    return tp, sl
