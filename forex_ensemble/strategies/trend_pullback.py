# forex_ensemble/strategies/trend_pullback.py
from __future__ import annotations
from typing import Sequence
import numpy as np
from ..types import Candle, StrategySignal
from ..utils.ta import ema, atr
from .helpers import to_arrays, clamp01, price_to_tp_sl

class TrendPullbackEMA:
    name = "Trend Pullback to EMA"
    def __init__(self, ema_p: int = 50):
        self.ema_p = ema_p

    def generate(self, candles: Sequence[Candle]) -> StrategySignal:
        o, h, l, c = to_arrays(candles)
        m = ema(c, self.ema_p)
        a = atr(h, l, c, 14)
        last = -1
        # trend up dacă close > EMA și EMA în creștere; semnal long pe pullback sub EMA anterior
        ema_slope = m[last] - m[last-3]
        dir_ = "neutral"
        if c[last] > m[last] and ema_slope > 0 and c[last-1] < m[last-1]:
            dir_ = "long"
        elif c[last] < m[last] and ema_slope < 0 and c[last-1] > m[last-1]:
            dir_ = "short"
        entry = float(c[last]) if dir_ != "neutral" else None
        tp = sl = None
        conf = 0.0
        if dir_ != "neutral":
            distance = abs(c[last]-m[last]) / max(1e-9, a[last])
            slope_strength = abs(ema_slope) / max(1e-9, a[last])
            raw = 0.4*clamp01(distance) + 0.6*clamp01(slope_strength)
            conf = clamp01(raw)
            tp, sl = price_to_tp_sl(dir_, entry, a[last], rr=1.6)
        return StrategySignal(self.name, dir_, entry, sl, tp, conf, notes=f"ema={m[last]:.5f}")
