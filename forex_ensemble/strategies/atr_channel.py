# forex_ensemble/strategies/atr_channel.py
from __future__ import annotations
from typing import Sequence
from ..types import Candle, StrategySignal
from ..utils.ta import ema, atr
from .helpers import to_arrays, clamp01, price_to_tp_sl

class AtrChannelBreakout:
    name = "ATR Channel Breakout"
    def __init__(self, ma_period: int = 50, atr_mult: float = 1.5):
        self.ma_period, self.atr_mult = ma_period, atr_mult

    def generate(self, candles: Sequence[Candle]) -> StrategySignal:
        _, h, l, c = to_arrays(candles)
        m = ema(c, self.ma_period)
        a = atr(h, l, c, 14)
        upper = m + self.atr_mult * a
        lower = m - self.atr_mult * a
        last = -1
        dir_ = "neutral"
        if c[last] > upper[last]:
            dir_ = "long"
        elif c[last] < lower[last]:
            dir_ = "short"
        entry = float(c[last]) if dir_ != "neutral" else None
        tp = sl = None
        conf = 0.0
        if dir_ != "neutral":
            distance = abs(c[last] - (upper[last] if dir_=="long" else lower[last]))/max(1e-9, a[last])
            slope = (m[last]-m[last-5]) / max(1e-9, a[last])
            raw = 0.6*clamp01(abs(slope)) + 0.4*clamp01(distance)
            conf = clamp01(raw)
            tp, sl = price_to_tp_sl(dir_, entry, a[last], rr=1.8)
        return StrategySignal(self.name, dir_, entry, sl, tp, conf, notes=f"ma={m[last]:.5f}")
