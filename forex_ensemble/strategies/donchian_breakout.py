# forex_ensemble/strategies/donchian_breakout.py
from __future__ import annotations
from typing import Sequence
from ..types import Candle, StrategySignal
from ..utils.ta import donchian_high, donchian_low, atr
from .helpers import to_arrays, clamp01, price_to_tp_sl

class DonchianBreakout:
    name = "Donchian Breakout"
    def __init__(self, period: int = 20):
        self.period = period

    def generate(self, candles: Sequence[Candle]) -> StrategySignal:
        _, h, l, c = to_arrays(candles)
        dc_h = donchian_high(h, self.period)
        dc_l = donchian_low(l, self.period)
        a = atr(h, l, c, 14)
        last = -1
        dir_ = "neutral"
        if c[last] > dc_h[last-1]:  # breakout peste maximul anterior
            dir_ = "long"
        elif c[last] < dc_l[last-1]:
            dir_ = "short"
        entry = float(c[last]) if dir_ != "neutral" else None
        tp = sl = None
        conf = 0.0
        if dir_ != "neutral":
            range_w = (dc_h[last] - dc_l[last]) / max(1e-9, a[last])
            raw = 0.5 + 0.5*clamp01(1.0/range_w)  # mai mare dacă canalul este îngust relativ la ATR
            conf = clamp01(raw)
            tp, sl = price_to_tp_sl(dir_, entry, a[last], rr=2.0)
        return StrategySignal(self.name, dir_, entry, sl, tp, conf, notes=f"range_w={dc_h[last]-dc_l[last]:.5f}")
