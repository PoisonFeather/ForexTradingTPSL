# forex_ensemble/ensemble_v2.py
from typing import List, Dict, Optional

DIR_MAP = {"long": 1, "short": -1, "neutral": 0}

def _weighted_median(values: List[float], weights: List[float]) -> Optional[float]:
    if not values:
        return None
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return None
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= 0.5 * total_w:
            return v
    return pairs[-1][0]

def _weighted_choice(values: List[float], weights: List[float], mode: str) -> Optional[float]:
    if not values:
        return None
    if mode == "median":
        return _weighted_median(values, weights)
    wsum = sum(weights)
    if wsum <= 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / wsum

def _apply_regime_weights(base_w: Dict[str, float], regime: str) -> Dict[str, float]:
    w = dict(base_w)
    if regime == "trend":
        for k in ["EMA Crossover + RSI filter", "Donchian Breakout", "ATR Channel Breakout",
                  "Keltner Squeeze", "Trend Pullback to EMA", "MACD Crossover"]:
            if k in w: w[k] *= 1.25
        for k in ["Bollinger Mean-Reversion", "Opening Range Breakout"]:
            if k in w: w[k] *= 0.9
    elif regime == "range":
        for k in ["Bollinger Mean-Reversion", "Opening Range Breakout"]:
            if k in w: w[k] *= 1.25
        for k in ["Donchian Breakout", "ATR Channel Breakout"]:
            if k in w: w[k] *= 0.9
    return w

def infer_regime(market_features: Dict[str, float]) -> str:
    adx = market_features.get("adx", 18.0)
    atr = market_features.get("atr", 0.0)
    price = market_features.get("price", 1.0)
    vol = (atr / price) if price else 0.0
    if adx >= 22 or vol >= 0.0009:
        return "trend"
    if adx <= 16 and vol <= 0.0006:
        return "range"
    return "neutral"

def combine_signals(
    signals: List[Dict],
    *,
    last_price: float,
    base_weights: Optional[Dict[str, float]] = None,
    regime_features: Optional[Dict[str, float]] = None,
    oppose_damp: float = 0.35,
    price_mode: str = "median",
    min_conf_floor: float = 0.05,
) -> Dict:
    if not signals:
        return {"dir": "neutral", "confidence": 0.0, "entry": None, "sl": None, "tp": None}

    base_w = {s["strategy"]: 1.0 for s in signals}
    if base_weights:
        for k, v in base_weights.items():
            if k in base_w: base_w[k] = float(v)

    regime = "neutral"
    if regime_features:
        regime = infer_regime(regime_features)
        base_w = _apply_regime_weights(base_w, regime)

    vote_num = 0.0
    vote_den = 0.0
    per_strat_weight = {}
    for s in signals:
        name = s["strategy"]
        conf = float(s.get("conf", 0.0))
        if 0.0 < conf < min_conf_floor:
            conf = min_conf_floor
        w = base_w.get(name, 1.0) * max(conf, 0.0)
        per_strat_weight[name] = w
        d = DIR_MAP.get(s.get("dir", "neutral"), 0)
        vote_num += w * d
        vote_den += w

    if vote_den <= 0:
        return {"dir": "neutral", "confidence": 0.0, "entry": last_price, "sl": None, "tp": None, "regime": regime}

    dir_val = 1 if vote_num > 1e-12 else (-1 if vote_num < -1e-12 else 0)
    final_dir = "long" if dir_val > 0 else ("short" if dir_val < 0 else "neutral")
    final_conf = abs(vote_num) / vote_den

    if final_dir == "neutral":
        return {"dir": "neutral", "confidence": final_conf, "entry": last_price, "sl": None, "tp": None, "regime": regime}

    entries, e_w, stops, s_w, tps, t_w = [], [], [], [], [], []
    for s in signals:
        name = s["strategy"]
        w = per_strat_weight.get(name, 0.0)
        if w <= 0: continue
        sd = s.get("dir", "neutral")
        factor = 1.0 if sd == final_dir else (oppose_damp if sd in ("long","short") else 0.2 * oppose_damp)
        eff_w = w * factor
        if s.get("entry") is not None:
            entries.append(float(s["entry"])); e_w.append(eff_w)
        if s.get("sl") is not None:
            stops.append(float(s["sl"])); s_w.append(eff_w)
        if s.get("tp") is not None:
            tps.append(float(s["tp"])); t_w.append(eff_w)

    entry = _weighted_choice(entries, e_w, price_mode) if entries else last_price
    final = {"dir": final_dir, "confidence": round(float(final_conf), 2), "regime": regime}

    if final_dir == "long":
        sl = min(stops) if stops else None
        tp = _weighted_choice(tps, t_w, price_mode) if tps else (entry * (1 + 0.0015))
    else:
        sl = max(stops) if stops else None
        tp = _weighted_choice(tps, t_w, price_mode) if tps else (entry * (1 - 0.0015))

    atr = (regime_features or {}).get("atr")
    if sl is None and atr is not None:
        sl = entry - 1.1 * atr if final_dir == "long" else entry + 1.1 * atr

    # sanity ordering
    if final_dir == "long" and sl is not None and tp is not None:
        if sl >= entry: sl = entry - 0.25 * max(tp - entry, 1e-8)
        if tp <= entry: tp = entry + (entry - sl) * 1.5
    if final_dir == "short" and sl is not None and tp is not None:
        if sl <= entry: sl = entry + 0.25 * max(entry - tp, 1e-8)
        if tp >= entry: tp = entry - (sl - entry) * 1.5

    final.update({"entry": float(entry), "sl": float(sl) if sl is not None else None, "tp": float(tp) if tp is not None else None})
    return final
