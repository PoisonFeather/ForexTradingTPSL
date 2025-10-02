# forex_ensemble/strategies/opening_range.py
from __future__ import annotations
from typing import Sequence
from datetime import time
import numpy as np
from ..types import Candle, StrategySignal
from ..utils.ta import atr
from .helpers import to_arrays, clamp01, price_to_tp_sl

class OpeningRangeBreakout:
    name = "Opening Range Breakout"
    def __init__(self, minutes: int = 30):  # prima jumătate de oră a sesiunii
        self.minutes = minutes

    def generate(self, candles: Sequence[Candle]) -> StrategySignal:
        # presupunem timeframe intraday și folosim fereastra ultimei zile UTC
        _, h, l, c = to_arrays(candles)
        a = atr(h, l, c, 14)
        # găsim startul zilei UTC pentru ultimele bare
        last_day = candles[-1].time.date()
        day_idxs = [i for i, cd in enumerate(candles) if cd.time.date() == last_day]
        if not day_idxs:
            return StrategySignal(self.name, "neutral", None, None, None, 0.0, notes="no intraday bars")
        start = day_idxs[0]
        # nr. de bare în opening range = minutes / tf_minutes aproximat din diferența dintre ultimele două lumânări
        if len(day_idxs) < 3:
            return StrategySignal(self.name, "neutral", None, None, None, 0.0, notes="not enough bars today")
        tf_minutes = int((candles[start+1].time - candles[start].time).total_seconds() // 60) or 5
        bars = max(1, self.minutes // tf_minutes)
        or_high = max(cd.high for cd in candles[start:start+bars])
        or_low  = min(cd.low  for cd in candles[start:start+bars])
        price = c[-1]
        dir_ = "neutral"
        if price > or_high:
            dir_ = "long"
        elif price < or_low:
            dir_ = "short"
        entry = float(price) if dir_ != "neutral" else None
        tp = sl = None
        conf = 0.0
        if dir_ != "neutral":
            range_size = (or_high - or_low)
            vol = a[-1]
            raw = clamp01(range_size / max(1e-9, vol))  # range vs. ATR
            conf = clamp01(0.5 + 0.5 * raw)
            tp, sl = price_to_tp_sl(dir_, entry, vol, rr=2.0, sl_mult=1.0)
        return StrategySignal(self.name, dir_, entry, sl, tp, conf, notes=f"OR=({or_low:.5f},{or_high:.5f})")
