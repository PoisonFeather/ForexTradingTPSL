# forex_ensemble/forward.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Iterable, Tuple
import math
import statistics

# Presupunem că există funcțiile tale:
# - get_candles(provider, symbol, timeframe, limit) -> List[dict(timestamp, open, high, low, close)]
# - generate_strategy_signals(candles) -> List[Dict]] per strategie la ultimul bar
#   (sau o funcție existentă pe care o chemi pentru fiecare t ca să obții semnale "la zi")
# - combine_signals(...) din ensemble_v2 (pe care l-am propus)
from ..ensemble_v2.ensemble_v2 import combine_signals

@dataclass
class Trade:
    side: str          # 'long'|'short'
    entry: float
    sl: float
    tp: float
    open_idx: int      # index bar la care intră
    close_idx: int     # index bar la care iese
    exit_price: float
    result: float      # P&L în pips sau în unități de preț
    hit: str           # 'tp'|'sl'|'time' (fallback)
    meta: dict

@dataclass
class WFResult:
    equity: List[float]
    trades: List[Trade]
    metrics: Dict[str, float]
    splits: List[Tuple[int,int,int,int]]  # (train_start, train_end, test_start, test_end)
    per_split_metrics: List[Dict[str,float]]

def _sharpe(returns: List[float], eps: float = 1e-12) -> float:
    if not returns:
        return 0.0
    mean = statistics.fmean(returns)
    std = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    return mean / (std + eps)

def _max_drawdown(equity: List[float]) -> float:
    peak = -1e18
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = (peak - v)
        mdd = max(mdd, dd)
    return mdd

def _to_pips(symbol: str, price_diff: float) -> float:
    # simplu: EURUSD -> 0.0001 pip; pentru JPY ai 0.01, etc.
    # aici facem o euristică rapidă
    if symbol.endswith("JPY"):
        scale = 0.01
    else:
        scale = 0.0001
    return price_diff / scale

def _barrier_exit(candles: List[dict], side: str, start_idx: int, entry: float, sl: float, tp: float, max_hold: Optional[int] = None) -> Tuple[int, float, str]:
    """
    Merge bar cu bar de la start_idx+1 până lovește TP sau SL.
    Dacă max_hold e setat, forțează ieșirea la close-ul ultimei bare permise.
    Returnează (close_idx, exit_price, hit_type).
    """
    n = len(candles)
    end_idx = n - 1
    limit_idx = end_idx if max_hold is None else min(end_idx, start_idx + max_hold)
    for i in range(start_idx + 1, limit_idx + 1):
        hi = candles[i]["high"]; lo = candles[i]["low"]
        if side == "long":
            if lo <= sl <= hi and sl <= entry:  # SL atins (tolerant)
                return i, sl, "sl"
            if lo <= tp <= hi and tp >= entry:  # TP atins
                return i, tp, "tp"
        else:  # short
            if lo <= tp <= hi and tp <= entry:
                return i, tp, "tp"
            if lo <= sl <= hi and sl >= entry:
                return i, sl, "sl"
    # Fallback: time exit
    return limit_idx, candles[limit_idx]["close"], "time"

def _calc_metrics_from_trades(symbol: str, trades: List[Trade]) -> Dict[str,float]:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "avg_pips": 0.0, "sharpe": 0.0, "gross_pips": 0.0, "max_dd_pips": 0.0}
    rets = [_to_pips(symbol, t.result) for t in trades]
    wins = sum(1 for t in rets if t > 0)
    gross = sum(rets)
    # equity pentru MDD
    eq = []
    s = 0.0
    for r in rets:
        s += r
        eq.append(s)
    return {
        "trades": len(trades),
        "win_rate": wins / len(trades),
        "avg_pips": statistics.fmean(rets),
        "sharpe": _sharpe(rets),
        "gross_pips": gross,
        "max_dd_pips": _max_drawdown(eq),
    }

def _strategy_performance_for_weights(symbol: str, per_bar_strat_sigs: List[List[Dict]], candles: List[dict], dir_threshold: float = 0.0) -> Dict[str, float]:
    """
    Rulăm fiecare strategie ca și cum ar fi executată singură (fără ensemble), pentru perioada curentă,
    ca să derivăm o greutate ~ performanță. Returnăm dict {strategy_name: score}.
    """
    scores: Dict[str, float] = {}
    # iterăm pe baruri; per_bar_strat_sigs[t] = lista semnalelor strategiilor la barul t
    for t in range(len(per_bar_strat_sigs) - 1):
        for s in per_bar_strat_sigs[t]:
            name = s["strategy"]
            d = s["dir"]
            conf = float(s.get("conf", 0.0))
            if d == "neutral" or conf <= dir_threshold:
                continue
            entry = s.get("entry")
            sl = s.get("sl")
            tp = s.get("tp")
            if entry is None or sl is None or tp is None:
                continue
            close_idx, exit_price, hit = _barrier_exit(candles, d, t, entry, sl, tp, max_hold=200)
            pnl = (exit_price - entry) if d == "long" else (entry - exit_price)
            scores[name] = scores.get(name, 0.0) + _to_pips(symbol, pnl)
    # normalizare simplă (relu pe pozitivi)
    if not scores:
        return {}
    min_s = min(scores.values()); max_s = max(scores.values())
    if max_s <= 0:
        # toate ≤ 0: revenim la greutăți egale
        return {k: 1.0 for k in scores.keys()}
    # clamp la [0, 1] și adăugăm un floor
    out = {}
    for k, v in scores.items():
        nv = 0.0 if v <= 0 else (v / max_s)
        out[k] = 0.25 + 0.75 * nv
    return out

class WalkForwardTester:
    def __init__(
        self,
        provider_fn,
        strategies_fn,
        ensemble_fn=combine_signals,
        symbol: str = "EURUSD",
        timeframe: str = "15min",
        fees: float = 0.0,      # comision per tranzacție (în unități de preț)
        spread: float = 0.0,    # spread aplicat pe entry/exit
        slippage: float = 0.0,  # slippage aplicat pe entry/exit
        max_hold: Optional[int] = 200,
    ):
        self.provider_fn = provider_fn
        self.strategies_fn = strategies_fn
        self.ensemble_fn = ensemble_fn
        self.symbol = symbol
        self.timeframe = timeframe
        self.fees = fees
        self.spread = spread
        self.slippage = slippage
        self.max_hold = max_hold

    def _apply_costs(self, side: str, entry: float) -> float:
        """aplică spread + slippage pe entry într-un mod simplu (buy plătește spread în sus, sell în jos)."""
        adj = self.spread + self.slippage
        if side == "long":
            return entry + adj
        else:
            return entry - adj

    def _prices_features_at(self, candles: List[dict], t: int) -> Dict[str,float]:
        # calculează câteva feature-uri rapid (ATR simplu, ADX simplificat? aici punem ATR pseudo)
        # simplu ATR(n=14) cu high-low|high-prev_close|low-prev_close
        n = min(14, t)
        if n < 2:
            return {"atr": 0.0, "adx": 18.0, "price": candles[t]["close"]}
        trs = []
        for i in range(t - n + 1, t + 1):
            h = candles[i]["high"]; l = candles[i]["low"]; c_prev = candles[i-1]["close"] if i > 0 else candles[i]["open"]
            tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
            trs.append(tr)
        atr = statistics.fmean(trs) if trs else 0.0
        # ADX proxy (foarte simplificat): raportul trend/range (doar euristică pentru weighting)
        adx_proxy = 16.0 + 20.0 * min(1.0, (atr / max(1e-6, candles[t]["close"])) / 0.001)
        return {"atr": atr, "adx": adx_proxy, "price": candles[t]["close"]}

    def _make_trade(self, candles: List[dict], t: int, sig: Dict) -> Optional[Trade]:
        side = sig["dir"]
        if side == "neutral":
            return None
        entry = sig.get("entry"); sl = sig.get("sl"); tp = sig.get("tp")
        if entry is None or sl is None or tp is None:
            return None
        entry_adj = self._apply_costs(side, float(entry))
        close_idx, exit_price, hit = _barrier_exit(candles, side, t, entry_adj, float(sl), float(tp), self.max_hold)
        pnl = (exit_price - entry_adj) if side == "long" else (entry_adj - exit_price)
        pnl -= self.fees
        return Trade(
            side=side, entry=entry_adj, sl=float(sl), tp=float(tp),
            open_idx=t, close_idx=close_idx, exit_price=exit_price,
            result=pnl, hit=hit, meta={"conf": sig.get("confidence", 0.0)}
        )

    def walkforward(
        self,
        candles: List[dict],
        train: int = 2000,
        test: int = 250,
        step: int = 125,
        mode: str = "expanding",  # 'expanding'|'rolling'
        optimize_by: str = "sharpe",  # 'sharpe'|'avg'|'gross'
        oppose_damp: float = 0.35,
    ) -> WFResult:
        """
        candles: listă completă (crescător în timp).
        """
        n = len(candles)
        equity = [0.0]
        all_trades: List[Trade] = []
        splits: List[Tuple[int,int,int,int]] = []
        per_split_metrics: List[Dict[str,float]] = []

        start_train_end = train
        while True:
            train_end = start_train_end
            test_end = min(n, train_end + test)
            if test_end - train_end < 10:
                break
            train_start = 0 if mode == "expanding" else max(0, train_end - train)
            test_start = train_end
            splits.append((train_start, train_end, test_start, test_end))

            # --- 1) Derive weights pe TRAIN ---
            # per bar generate semnale strategii
            per_bar_strat_sigs: List[List[Dict]] = []
            for t in range(train_start, train_end):
                # semnale "la zi" folosind doar istoria până la t (inclusiv)
                # strategies_fn să întoarcă: List[Dict{strategy, dir, entry, sl, tp, conf}]
                sigs_t = self.strategies_fn(candles[: t + 1])
                per_bar_strat_sigs.append(sigs_t)

            base_weights = _strategy_performance_for_weights(self.symbol, per_bar_strat_sigs, candles[train_start:train_end])
            # fallback egal dacă goale
            if not base_weights:
                # colectăm numele strategiilor din ultimul t
                last = per_bar_strat_sigs[-1] if per_bar_strat_sigs else []
                base_weights = {s["strategy"]: 1.0 for s in last}

            # --- 2) Rulează pe TEST cu ensemble v2 + weights ---
            trades_split: List[Trade] = []
            for t in range(test_start, test_end - 1):
                last_price = candles[t]["close"]
                regime_features = self._prices_features_at(candles, t)
                sigs = self.strategies_fn(candles[: t + 1])  # sig la t
                final = self.ensemble_fn(
                    sigs,
                    last_price=last_price,
                    base_weights=base_weights,
                    regime_features=regime_features,
                    oppose_damp=oppose_damp,
                    price_mode="median",
                )
                tr = self._make_trade(candles, t, final)
                if tr:
                    trades_split.append(tr)
                    equity.append(equity[-1] + _to_pips(self.symbol, tr.result))

            # metrice pe split
            m = _calc_metrics_from_trades(self.symbol, trades_split)
            per_split_metrics.append(m)
            all_trades.extend(trades_split)

            # pas înainte
            if test_end >= n:
                break
            start_train_end = test_start + step

        # agregate
        metrics = _calc_metrics_from_trades(self.symbol, all_trades)
        return WFResult(equity=equity, trades=all_trades, metrics=metrics, splits=splits, per_split_metrics=per_split_metrics)
