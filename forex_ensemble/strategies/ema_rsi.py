# forex_ensemble/strategies/ema_rsi.py
from __future__ import annotations
from typing import Sequence
import numpy as np
from ..types import Candle, StrategySignal
from ..utils.ta import ema, rsi, atr
from .helpers import to_arrays, clamp01, price_to_tp_sl

class EmaRsiStrategy:
    name = "EMA Crossover + RSI filter"
    def __init__(self, fast: int = 12, slow: int = 26, rsi_p: int = 14, rsi_buy: float = 55, rsi_sell: float = 45):
        self.fast, self.slow, self.rsi_p = fast, slow, rsi_p
        self.rsi_buy, self.rsi_sell = rsi_buy, rsi_sell

    def generate(self, candles: Sequence[Candle]) -> StrategySignal:
        _, h, l, c = to_arrays(candles)
        ema_f = ema(c, self.fast)
        ema_s = ema(c, self.slow)
        r = rsi(c, self.rsi_p)
        a = atr(h, l, c, 14)
        last = -1

        dir_ = "neutral"
        if ema_f[last] > ema_s[last] and r[last] >= self.rsi_buy:
            dir_ = "long"
        elif ema_f[last] < ema_s[last] and r[last] <= self.rsi_sell:
            dir_ = "short"

        entry = float(c[last]) if dir_ != "neutral" else None
        tp = sl = None
        conf = 0.0
        if dir_ != "neutral":
            slope = (ema_f[last] - ema_f[last-3]) / 3.0
            spread = abs(ema_f[last] - ema_s[last]) / max(1e-9, a[last])
            rsi_bias = (r[last]-50)/50 if dir_=="long" else (50-r[last])/50
            raw = 0.3*min(1, abs(slope)/a[last]) + 0.5*clamp01(spread) + 0.2*clamp01(rsi_bias)
            conf = clamp01(raw)
            tp, sl = price_to_tp_sl(dir_, entry, a[last])
        return StrategySignal(self.name, dir_, entry, sl, tp, conf, notes=f"ema_f={ema_f[last]:.5f}, ema_s={ema_s[last]:.5f}, rsi={r[last]:.1f}")
