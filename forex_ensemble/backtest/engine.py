# forex_ensemble/backtest/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Literal, Callable
from datetime import datetime, timezone

import numpy as np
from ..types import Candle, StrategySignal
# strategii individuale pentru seturi trend/range
from ..strategies.ema_rsi import EmaRsiStrategy
from ..strategies.trend_pullback import TrendPullbackEMA
from ..strategies.keltner_squeeze import KeltnerSqueeze
from ..strategies.macd_cross import MacdCross
from ..strategies.bb_mean_reversion import BollingerMeanRev
from ..strategies.opening_range import OpeningRangeBreakout

Side = Literal["long", "short"]

@dataclass(slots=True)
class BTConfig:
    entry_buffer_atr: float = 0.05
    rr_required: float = 1.2
    spread: float = 0.00008
    commission_per_trade: float = 0.0
    slippage_pips: float = 0.5
    atr_period: int = 14
    min_confidence: float = 0.05
    allow_multiple_per_bar: bool = False

    # 🔥 NOI:
    active_start_hour: int = 7     # fereastra local time (Europe/Bucharest)
    active_end_hour: int = 20
    cooldown_bars: int = 6         # așteaptă N bare după un trade închis
    max_trades_per_day: int = 10
    trend_slope_threshold: float = 0.6  # prag panta EMA50 / ATR pe 5 bare

@dataclass(slots=True)
class Trade:
    open_time: datetime
    close_time: datetime | None
    side: Side
    entry: float
    sl: float
    tp: float
    exit: float | None
    result_r: float | None
    notes: str = ""

@dataclass(slots=True)
class BTRun:
    trades: list[Trade]
    equity_curve: list[tuple[datetime, float]]
    start_equity: float
    end_equity: float

# --- helpers
def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    alpha = 2 / (period + 1.0)
    out = np.empty_like(tr)
    out[0] = tr[0]
    for i in range(1, len(tr)):
        out[i] = alpha * tr[i] + (1 - alpha) * out[i-1]
    return out

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    alpha = 2 / (period + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i-1]
    return out

def _pip_value_per_lot() -> float:
    return 10.0  # USD/pip/lot pentru major pairs (aprox.)

def backtest(
    candles: Sequence[Candle],
    strategy_builder: Callable[[], list] | None = None,  # nefolosit acum; păstrat pt compat.
    cfg: BTConfig | None = None,
    initial_equity: float = 10_000.0,
    risk_pct_per_trade: float = 1.0,
) -> BTRun:
    cfg = cfg or BTConfig()
    # arrays
    o = np.array([c.open for c in candles], float)
    h = np.array([c.high for c in candles], float)
    l = np.array([c.low for c in candles], float)
    c = np.array([c.close for c in candles], float)
    t = np.array([c.time for c in candles])

    atr = _atr(h, l, c, cfg.atr_period)
    ema50 = _ema(c, 50)
    pip = 0.0001
    slippage = cfg.slippage_pips * pip

    trades: list[Trade] = []
    equity = initial_equity
    equity_curve: list[tuple[datetime, float]] = []

    position_open: Trade | None = None
    bars_since_exit = cfg.cooldown_bars  # permite primul trade
    # contor tranzacții/zi
    from collections import defaultdict
    trades_today = defaultdict(int)
    current_day = candles[0].time.date() if candles else None

    for i in range(max(50, cfg.atr_period + 2), len(candles) - 1):
        now = t[i]
        next_open = o[i+1]

        # rollover contor la zi nouă
        if current_day != now.date():
            current_day = now.date()
            trades_today[current_day] = 0

        # equity curve
        equity_curve.append((now, equity))

        # 0) time-of-day filter (folosește ora locală a sistemului; dacă vrei tz fix, convertește)
        local_hour = now.astimezone().hour
        active_hours = (cfg.active_start_hour <= local_hour <= cfg.active_end_hour)

        # 1) dacă avem poziție deschisă, verifică SL/TP
        if position_open is not None:
            hit = None
            if position_open.side == "long":
                if l[i] <= position_open.sl:
                    hit = max(position_open.sl - slippage, position_open.sl)
                elif h[i] >= position_open.tp:
                    hit = min(position_open.tp + slippage, position_open.tp)
            else:
                if h[i] >= position_open.sl:
                    hit = min(position_open.sl + slippage, position_open.sl)
                elif l[i] <= position_open.tp:
                    hit = max(position_open.tp - slippage, position_open.tp)

            if hit is not None:
                exit_price = next_open
                risk = abs(position_open.entry - position_open.sl)
                r_mult = (exit_price - position_open.entry) / risk if position_open.side == "long" else \
                         (position_open.entry - exit_price) / risk
                position_open.close_time = t[i+1]
                position_open.exit = exit_price
                position_open.result_r = r_mult
                trades.append(position_open)

                # equity update (costs)
                risk_amount = equity * (risk_pct_per_trade / 100.0)
                pnl = r_mult * risk_amount
                pnl -= cfg.commission_per_trade
                # aproximăm cost spread: 2x spread * pip_value * lot_size
                lot_size = risk_amount / (risk * _pip_value_per_lot() * 10)  # scalare simplă
                pnl -= 2 * cfg.spread * _pip_value_per_lot() * lot_size
                equity += pnl

                position_open = None
                bars_since_exit = 0  # pornește cooldown

        # 2) condiții pentru deschiderea unei poziții
        if position_open is None and active_hours and trades_today[current_day] < cfg.max_trades_per_day:
            if bars_since_exit < cfg.cooldown_bars:
                bars_since_exit += 1
                continue

            # calc regim trend / range
            slope = (ema50[i] - ema50[i-5]) / max(1e-9, atr[i])  # normalizat
            in_trend = abs(slope) > cfg.trend_slope_threshold

            # seturi de strategii
            trend_set = [EmaRsiStrategy(), TrendPullbackEMA(), KeltnerSqueeze(), MacdCross()]
            range_set = [BollingerMeanRev(), OpeningRangeBreakout(minutes=30)]
            active_strategies = trend_set if in_trend else range_set

            # generează semnale pe istoricul până la barul i
            local_candles = candles[: i+1]
            signals: list[StrategySignal] = [s.generate(local_candles) for s in active_strategies]

            # selectează candidații buni
            def rr(sig: StrategySignal) -> float:
                risk = abs((sig.entry or 0) - (sig.stop_loss or 0)) or 1e-9
                reward = abs((sig.take_profit or 0) - (sig.entry or 0))
                return reward / risk

            candidates = [
                s for s in signals
                if s.direction in ("long","short")
                and s.entry is not None and s.stop_loss is not None and s.take_profit is not None
                and s.confidence >= cfg.min_confidence
                and rr(s) >= cfg.rr_required
            ]

            if candidates:
                best = max(candidates, key=lambda s: s.confidence * rr(s))
                buf = cfg.entry_buffer_atr * atr[i]
                if best.direction == "long":
                    entry = max(next_open + buf + cfg.spread/2 + slippage, best.entry)
                else:
                    entry = min(next_open - buf - cfg.spread/2 - slippage, best.entry)

                position_open = Trade(
                    open_time=t[i+1],
                    close_time=None,
                    side="long" if best.direction=="long" else "short",
                    entry=float(entry),
                    sl=float(best.stop_loss),
                    tp=float(best.take_profit),
                    exit=None,
                    result_r=None,
                    notes=f"{best.name} conf={best.confidence:.2f} trend={in_trend}"
                )
                trades_today[current_day] += 1

        else:
            # dacă suntem în afara intervalului orar, doar incrementăm cooldown-ul „pasiv”
            if position_open is None and bars_since_exit < cfg.cooldown_bars:
                bars_since_exit += 1

    if len(equity_curve)==0 or equity_curve[-1][0] != t[-2]:
        equity_curve.append((t[-2], equity))

    return BTRun(trades=trades, equity_curve=equity_curve,
                 start_equity=initial_equity, end_equity=equity)
