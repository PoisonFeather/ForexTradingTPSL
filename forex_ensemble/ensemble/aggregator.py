# forex_ensemble/ensemble/aggregator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal
import numpy as np
from ..types import StrategySignal, EnsembleDecision, Timeframe
from datetime import datetime, timezone

Direction = Literal["long","short","neutral"]

@dataclass(slots=True)
class AggregationConfig:
    winsor_pct: float = 0.10   # taie 10% extreme
    min_conf_to_count: float = 0.05
    min_strategies_for_signal: int = 2
    consensus_threshold: float = 0.15  # dacă diferența long vs short < 0.15 => neutral

def _dir_to_num(d: Direction) -> int:
    return 1 if d == "long" else (-1 if d == "short" else 0)

def _winsorize(x: np.ndarray, p: float) -> np.ndarray:
    if len(x) == 0 or p <= 0: return x
    lower = np.quantile(x, p)
    upper = np.quantile(x, 1 - p)
    return np.clip(x, lower, upper)

def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float | None:
    if len(values) == 0: return None
    wsum = np.sum(weights)
    if wsum == 0: return None
    return float(np.sum(values * weights) / wsum)

def aggregate_signals(
    symbol: str, timeframe: Timeframe, price_now: float,
    signals: Iterable[StrategySignal], cfg: AggregationConfig | None = None
) -> EnsembleDecision:

    cfg = cfg or AggregationConfig()
    sigs = [s for s in signals if s.confidence >= cfg.min_conf_to_count and s.direction != "neutral"]

    if len(sigs) < cfg.min_strategies_for_signal:
        return EnsembleDecision(
            symbol=symbol, timeframe=timeframe, timestamp=datetime.now(timezone.utc),
            entry=None, stop_loss=None, take_profit=None, direction="neutral", confidence=0.0
        )

    # 1) Vot pe direcție (ponderat cu confidence)
    votes = np.array([_dir_to_num(s.direction) for s in sigs], dtype=float)
    weights = np.array([s.confidence for s in sigs], dtype=float)
    dir_score = float(np.sum(votes * weights) / (np.sum(weights) or 1.0))  # -1..1
    p_long = (dir_score + 1) / 2.0  # 0..1
    p_short = 1.0 - p_long

    if abs(p_long - p_short) < cfg.consensus_threshold:
        direction: Direction = "neutral"
    else:
        direction = "long" if p_long > p_short else "short"

    if direction == "neutral":
        return EnsembleDecision(
            symbol=symbol, timeframe=timeframe, timestamp=datetime.now(timezone.utc),
            entry=None, stop_loss=None, take_profit=None, direction="neutral",
            confidence=float(abs(p_long - p_short))
        )

    # 2) Media ponderată Entry/TP/SL doar din strategiile cu aceeași direcție
    same_dir = [s for s in sigs if s.direction == direction and s.entry is not None and s.take_profit is not None and s.stop_loss is not None]
    if len(same_dir) < cfg.min_strategies_for_signal:
        # fallback: folosim toate strategille dar păstrăm direcția
        same_dir = sigs

    entries = np.array([s.entry for s in same_dir if s.entry is not None], dtype=float)
    tps     = np.array([s.take_profit for s in same_dir if s.take_profit is not None], dtype=float)
    sls     = np.array([s.stop_loss for s in same_dir if s.stop_loss is not None], dtype=float)
    wts     = np.array([s.confidence for s in same_dir], dtype=float)

    # winsorize
    entries_w = _winsorize(entries, cfg.winsor_pct) if len(entries) > 3 else entries
    tps_w     = _winsorize(tps, cfg.winsor_pct) if len(tps) > 3 else tps
    sls_w     = _winsorize(sls, cfg.winsor_pct) if len(sls) > 3 else sls

    entry_avg = _weighted_average(entries_w, wts[:len(entries_w)])
    tp_avg    = _weighted_average(tps_w,     wts[:len(tps_w)])
    sl_avg    = _weighted_average(sls_w,     wts[:len(sls_w)])

    # 3) Scor de încredere agregat
    #   - forță a consensului pe direcție
    #   - calitatea (media confidence) a strategiilor folosite
    conf_mean = float(np.mean(wts)) if len(wts) else 0.0
    consensus_strength = float(abs(p_long - p_short))  # 0..1
    final_confidence = 0.55 * consensus_strength + 0.45 * conf_mean

    return EnsembleDecision(
        symbol=symbol, timeframe=timeframe, timestamp=datetime.now(timezone.utc),
        entry=entry_avg, stop_loss=sl_avg, take_profit=tp_avg,
        direction=direction, confidence=final_confidence
    )
