# forex_ensemble/forwardtest.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import statistics

from forex_ensemble.ensemble_v2 import combine_signals

# -------- util: suportă atât dict cât și obiect cu atribute --------
def _get(o, name, default=None):
    return getattr(o, name, o.get(name, default) if isinstance(o, dict) else default)

def _ohlc(c):  # returns (o,h,l,c)
    return (_get(c, "open"), _get(c, "high"), _get(c, "low"), _get(c, "close"))

def _time(c):
    return _get(c, "time")

# -------- pips helpers --------
def _to_pip_units(symbol: str) -> float:
    return 0.01 if symbol.endswith("JPY") else 0.0001

def _to_pips(symbol: str, price_diff: float) -> float:
    return price_diff / _to_pip_units(symbol)

# -------- trade structs --------
@dataclass
class Trade:
    side: str
    entry: float
    sl: float
    tp: float
    open_idx: int
    close_idx: int
    exit_price: float
    result: float
    hit: str
    meta: dict

@dataclass
class WFResult:
    equity: List[float]
    trades: List[Trade]
    metrics: Dict[str, float]
    splits: List[Tuple[int,int,int,int]]
    per_split_metrics: List[Dict[str,float]]

# -------- metrics --------
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

# -------- barrier exit --------
def _barrier_exit(candles: List, side: str, start_idx: int, entry: float, sl: float, tp: float, max_hold: Optional[int] = None):
    n = len(candles)
    end_idx = n - 1
    limit_idx = end_idx if max_hold is None else min(end_idx, start_idx + max_hold)
    for i in range(start_idx + 1, limit_idx + 1):
        _, hi, lo, _c = _ohlc(candles[i])
        if side == "long":
            if lo <= sl <= hi and sl <= entry:  # SL
                return i, sl, "sl"
            if lo <= tp <= hi and tp >= entry:  # TP
                return i, tp, "tp"
        else:
            if lo <= tp <= hi and tp <= entry:  # TP (short)
                return i, tp, "tp"
            if lo <= sl <= hi and sl >= entry:  # SL (short)
                return i, sl, "sl"
    # time exit
    _, _, _, last_close = _ohlc(candles[limit_idx])
    return limit_idx, last_close, "time"

# -------- strategy weighting on train --------
def _strategy_performance_for_weights(symbol: str, per_bar_strat_sigs: List[List[Dict]], candles_slice: List) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    pip = _to_pip_units(symbol)
    for t in range(len(per_bar_strat_sigs) - 1):
        for s in per_bar_strat_sigs[t]:
            name = s["strategy"]; d = s["dir"]; conf = float(s.get("conf", 0.0))
            if d == "neutral" or conf <= 0.0: continue
            entry, sl, tp = s.get("entry"), s.get("sl"), s.get("tp")
            if entry is None or sl is None or tp is None: continue
            close_idx, exit_price, _ = _barrier_exit(candles_slice, d, t, float(entry), float(sl), float(tp), max_hold=200)
            pnl = (exit_price - entry) if d == "long" else (entry - exit_price)
            scores[name] = scores.get(name, 0.0) + (pnl / pip)
    if not scores:
        return {}
    max_s = max(scores.values())
    if max_s <= 0:
        return {k: 1.0 for k in scores.keys()}
    out = {}
    for k, v in scores.items():
        nv = 0.0 if v <= 0 else (v / max_s)
        out[k] = 0.25 + 0.75 * nv
    return out

# -------- main tester --------
class WalkForwardTester:
    def __init__(self, provider_fn, strategies_fn, ensemble_fn=combine_signals,
                 symbol: str = "EURUSD", timeframe: str = "15min",
                 fees: float = 0.0, spread: float = 0.0, slippage: float = 0.0,
                 max_hold: Optional[int] = 200):
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
        adj = self.spread + self.slippage
        return entry + adj if side == "long" else entry - adj

    def _prices_features_at(self, candles: List, t: int) -> Dict[str, float]:
        n = min(14, t)
        _, _, _, close_t = _ohlc(candles[t])
        if n < 2:
            return {"atr": 0.0, "adx": 18.0, "price": close_t}
        trs = []
        for i in range(t - n + 1, t + 1):
            _, h, l, _c = _ohlc(candles[i])
            _, _, _, c_prev = _ohlc(candles[i-1]) if i > 0 else _ohlc(candles[i])
            tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
            trs.append(tr)
        atr = statistics.fmean(trs) if trs else 0.0
        adx_proxy = 16.0 + 20.0 * min(1.0, (atr / max(1e-6, close_t)) / 0.001)
        return {"atr": atr, "adx": adx_proxy, "price": close_t}

    def _make_trade(self, candles: List, t: int, sig: Dict) -> Optional[Trade]:
        side = sig["dir"]
        if side == "neutral": return None
        entry, sl, tp = sig.get("entry"), sig.get("sl"), sig.get("tp")
        if entry is None or sl is None or tp is None: return None
        entry_adj = self._apply_costs(side, float(entry))
        close_idx, exit_price, hit = _barrier_exit(candles, side, t, entry_adj, float(sl), float(tp), self.max_hold)
        pnl = (exit_price - entry_adj) if side == "long" else (entry_adj - exit_price)
        pnl -= self.fees
        return Trade(side=side, entry=entry_adj, sl=float(sl), tp=float(tp),
                     open_idx=t, close_idx=close_idx, exit_price=exit_price,
                     result=pnl, hit=hit, meta={"conf": sig.get("confidence", 0.0)})

    def walkforward(self, candles: List, train: int = 2000, test: int = 250, step: int = 125,
                    mode: str = "expanding", optimize_by: str = "sharpe", oppose_damp: float = 0.35) -> WFResult:
        n = len(candles)
        equity = [0.0]
        all_trades: List[Trade] = []
        splits: List[Tuple[int,int,int,int]] = []
        per_split_metrics: List[Dict[str,float]] = []

        start_train_end = train
        while True:
            train_end = start_train_end
            test_end = min(n, train_end + test)
            if test_end - train_end < 10: break
            train_start = 0 if mode == "expanding" else max(0, train_end - train)
            test_start = train_end
            splits.append((train_start, train_end, test_start, test_end))

            # --- learn weights on TRAIN
            per_bar_strat_sigs: List[List[Dict]] = []
            for t in range(train_start, train_end):
                sigs_t = self.strategies_fn(candles[: t + 1])
                per_bar_strat_sigs.append(sigs_t)
            base_weights = _strategy_performance_for_weights(self.symbol, per_bar_strat_sigs, candles[train_start:train_end])
            if not base_weights:
                last = per_bar_strat_sigs[-1] if per_bar_strat_sigs else []
                base_weights = {s["strategy"]: 1.0 for s in last}

            # --- run TEST with ensemble
            trades_split: List[Trade] = []
            for t in range(test_start, test_end - 1):
                _, _, _, last_price = _ohlc(candles[t])
                regime_features = self._prices_features_at(candles, t)
                sigs = self.strategies_fn(candles[: t + 1])
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

            m = _calc_metrics_from_trades(self.symbol, trades_split)
            per_split_metrics.append(m)
            all_trades.extend(trades_split)

            if test_end >= n: break
            start_train_end = test_start + step

        metrics = _calc_metrics_from_trades(self.symbol, all_trades)
        return WFResult(equity=equity, trades=all_trades, metrics=metrics, splits=splits, per_split_metrics=per_split_metrics)

def _calc_metrics_from_trades(symbol: str, trades: List[Trade]) -> Dict[str,float]:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "avg_pips": 0.0, "sharpe": 0.0, "gross_pips": 0.0, "max_dd_pips": 0.0}
    rets = [_to_pips(symbol, t.result) for t in trades]
    wins = sum(1 for x in rets if x > 0)
    gross = sum(rets)
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
