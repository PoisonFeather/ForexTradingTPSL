# forex_ensemble/strategies/bb_mean_reversion.py
from __future__ import annotations
from typing import Sequence
import numpy as np
from ..types import Candle, StrategySignal
from ..utils.ta import bollinger, atr
from .helpers import to_arrays, clamp01, price_to_tp_sl

class BollingerMeanRev:
    name = "Bollinger Mean-Reversion"
    def __init__(self, period: int = 20, mult: float = 2.0):
        self.period, self.mult = period, mult

    def generate(self, candles: Sequence[Candle]) -> StrategySignal:
        _, h, l, c = to_arrays(candles)
        m, u, lo = bollinger(c, self.period, self.mult)
        a = atr(h, l, c, 14)
        last = -1
        dir_ = "neutral"
        if c[last] < lo[last]:  # oversold -> revert to mean
            dir_ = "long"
        elif c[last] > u[last]:
            dir_ = "short"
        entry = float(c[last]) if dir_ != "neutral" else None
        tp = sl = None
        conf = 0.0
        if dir_ != "neutral":
            band_dist = abs(c[last] - (u[last] if dir_=="short" else lo[last])) / max(1e-9, a[last])
            squeeze = (u[last]-lo[last]) / max(1e-9, m[last])
            raw = 0.6*clamp01(band_dist/1.5) + 0.4*clamp01(1.0/(1.0+squeeze))  # mai mare când benzile nu sunt enorme
            conf = clamp01(raw)
            tp, sl = price_to_tp_sl(dir_, entry, a[last], rr=1.5, sl_mult=1.0)
        return StrategySignal(self.name, dir_, entry, sl, tp, conf, notes=f"dist={band_dist if dir_!='neutral' else 0:.3f}")
