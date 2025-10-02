# forex_ensemble/utils/ta.py
from __future__ import annotations
import numpy as np

def ema(values: np.ndarray, period: int) -> np.ndarray:
    if period <= 0: raise ValueError("period must be > 0")
    alpha = 2 / (period + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i-1]
    return out

def sma(values: np.ndarray, period: int) -> np.ndarray:
    if period <= 0: raise ValueError("period must be > 0")
    out = np.full_like(values, fill_value=np.nan, dtype=float)
    cumsum = np.cumsum(values, dtype=float)
    out[period-1:] = (cumsum[period-1:] - np.concatenate(([0.0], cumsum[:-period])))/period
    # pentru primele valori, completăm cu prima medie
    first_valid = period-1
    if first_valid < len(values):
        out[:first_valid] = out[first_valid]
    return out

def rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    if period <= 1: raise ValueError("period must be > 1")
    out = np.full_like(values, np.nan, dtype=float)
    for i in range(period-1, len(values)):
        window = values[i-period+1:i+1]
        out[i] = np.std(window, ddof=0)
    # backfill
    first_valid = period-1
    if first_valid < len(values):
        out[:first_valid] = out[first_valid]
    return out

def rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    diff = np.diff(values, prepend=values[0])
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_gain = ema(gain, period)
    avg_loss = ema(loss, period)
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    return 100 - (100 / (1 + rs))

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return ema(tr, period)

def macd(values: np.ndarray, fast: int = 12, slow: int = 26, signal_p: int = 9):
    macd_line = ema(values, fast) - ema(values, slow)
    signal = ema(macd_line, signal_p)
    hist = macd_line - signal
    return macd_line, signal, hist

def bollinger(values: np.ndarray, period: int = 20, mult: float = 2.0):
    m = sma(values, period)
    s = rolling_std(values, period)
    upper = m + mult * s
    lower = m - mult * s
    return m, upper, lower

def rolling_max(values: np.ndarray, period: int) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i-period+1)
        out[i] = np.max(values[start:i+1])
    return out

def rolling_min(values: np.ndarray, period: int) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i-period+1)
        out[i] = np.min(values[start:i+1])
    return out

def donchian_high(values: np.ndarray, period: int) -> np.ndarray:
    return rolling_max(values, period)

def donchian_low(values: np.ndarray, period: int) -> np.ndarray:
    return rolling_min(values, period)
