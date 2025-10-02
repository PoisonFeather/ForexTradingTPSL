# forex_ensemble/strategies/keltner_squeeze.py
from __future__ import annotations
from typing import Sequence
from ..types import Candle, StrategySignal
from ..utils.ta import ema, atr, bollinger
from .helpers import to_arrays, clamp01, price_to_tp_sl

class KeltnerSqueeze:
    name = "Keltner Squeeze"
    def __init__(self, ema_p: int=20, atr_mult: float=1.5, bb_p: int=20, bb_mult: float=2.0):
        self.ema_p, self.atr_mult = ema_p, atr_mult
        self.bb_p, self.bb_mult = bb_p, bb_mult

    def generate(self, candles: Sequence[Candle]) -> StrategySignal:
        _, h, l, c = to_arrays(candles)
        m = ema(c, self.ema_p)
        a = atr(h, l, c, 14)
        k_upper = m + self.atr_mult * a
        k_lower = m - self.atr_mult * a
        bb_m, bb_u, bb_l = bollinger(c, self.bb_p, self.bb_mult)
        last = -1
        # „squeeze” când banda BB este în interiorul canalului Keltner
        in_squeeze = bb_u[last] < k_upper[last] and bb_l[last] > k_lower[last]
        dir_ = "neutral"
        if not in_squeeze:
            if c[last] > k_upper[last]:
                dir_ = "long"
            elif c[last] < k_lower[last]:
                dir_ = "short"
        entry = float(c[last]) if dir_ != "neutral" else None
        tp = sl = None
        conf = clamp01(0.4 if in_squeeze else 0.7) if dir_ != "neutral" else 0.0
        if dir_ != "neutral":
            tp, sl = price_to_tp_sl(dir_, entry, a[last], rr=2.0)
        return StrategySignal(self.name, dir_, entry, sl, tp, conf, notes=f"squeeze={in_squeeze}")
