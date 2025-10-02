# forex_ensemble/strategies/macd_cross.py
from __future__ import annotations
from typing import Sequence
from ..types import Candle, StrategySignal
from ..utils.ta import macd, atr
from .helpers import to_arrays, clamp01, price_to_tp_sl

class MacdCross:
    name = "MACD Crossover"
    def __init__(self, fast: int = 12, slow: int = 26, signal_p: int = 9):
        self.fast, self.slow, self.signal_p = fast, slow, signal_p

    def generate(self, candles: Sequence[Candle]) -> StrategySignal:
        # Warm-up guard: avem nevoie de suficiente bare pentru MACD și ATR și pentru a accesa [-2].
        min_len = max(self.slow + self.signal_p + 2, 14 + 2)
        if len(candles) < min_len:
            return StrategySignal(
                self.name, "neutral", None, None, None, 0.0,
                notes=f"warmup<{min_len}"
            )

        _, h, l, c = to_arrays(candles)
        macd_line, signal, hist = macd(c, self.fast, self.slow, self.signal_p)
        a = atr(h, l, c, 14)

        last = -1
        prev = -2
        dir_ = "neutral"
        if macd_line[prev] < signal[prev] and macd_line[last] > signal[last]:
            dir_ = "long"
        elif macd_line[prev] > signal[prev] and macd_line[last] < signal[last]:
            dir_ = "short"

        entry = float(c[last]) if dir_ != "neutral" else None
        tp = sl = None
        conf = 0.0
        if dir_ != "neutral":
            momentum = abs(hist[last]) / max(1e-9, a[last])
            trend_conf = abs(macd_line[last]) / max(1e-9, a[last])
            raw = 0.6 * clamp01(momentum) + 0.4 * clamp01(trend_conf)
            conf = clamp01(raw)
            tp, sl = price_to_tp_sl(dir_, entry, a[last], rr=1.7)

        return StrategySignal(self.name, dir_, entry, sl, tp, conf, notes=f"hist={hist[last]:.5f}")
