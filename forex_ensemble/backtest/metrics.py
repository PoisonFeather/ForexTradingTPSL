# forex_ensemble/backtest/metrics.py
from __future__ import annotations
from typing import Sequence
import math

from .engine import Trade, BTRun

def summarize(trades: Sequence[Trade], equity_curve: Sequence[tuple], risk_pct: float):
    n = len(trades)
    wins = [tr for tr in trades if (tr.result_r or 0) > 0]
    losses = [tr for tr in trades if (tr.result_r or 0) <= 0]
    win_rate = len(wins)/n if n else 0.0
    avg_r = sum((tr.result_r or 0) for tr in trades)/n if n else 0.0
    avg_win_r = (sum(tr.result_r for tr in wins)/len(wins)) if wins else 0.0
    avg_loss_r = (sum(tr.result_r for tr in losses)/len(losses)) if losses else 0.0
    profit_factor = (sum(tr.result_r for tr in wins)/abs(sum(tr.result_r for tr in losses))) if (wins and losses and sum(tr.result_r for tr in losses)!=0) else math.inf if wins and not losses else 0.0

    # max drawdown pe equity_curve
    eq = [e for _, e in equity_curve]
    peak = eq[0] if eq else 0.0
    max_dd = 0.0
    for v in eq:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    # sharpe aproximat (pe R mulțiplus, nu anualizat)
    r_series = [tr.result_r for tr in trades if tr.result_r is not None]
    mean_r = sum(r_series)/len(r_series) if r_series else 0.0
    std_r = (sum((x-mean_r)**2 for x in r_series)/len(r_series))**0.5 if r_series else 0.0
    sharpe = (mean_r/std_r) if std_r > 1e-9 else 0.0

    return {
        "trades": n,
        "win_rate": round(win_rate, 4),
        "avg_r": round(avg_r, 4),
        "avg_win_r": round(avg_win_r, 4),
        "avg_loss_r": round(avg_loss_r, 4),
        "profit_factor": round(profit_factor, 3) if profit_factor not in (math.inf, -math.inf) else profit_factor,
        "max_drawdown": round(max_dd, 4),
        "sharpe_like": round(sharpe, 3),
        "risk_pct": risk_pct,
    }
