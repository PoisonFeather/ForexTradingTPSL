# forex_ensemble/strategies/__init__.py
from .base import Strategy
from .ema_rsi import EmaRsiStrategy
from .bb_mean_reversion import BollingerMeanRev
from .macd_cross import MacdCross
from .opening_range import OpeningRangeBreakout
from .atr_channel import AtrChannelBreakout
from .keltner_squeeze import KeltnerSqueeze
from .trend_pullback import TrendPullbackEMA
from .donchian_breakout import DonchianBreakout

def default_strategies() -> list[Strategy]:
    return [
        EmaRsiStrategy(),
        BollingerMeanRev(),
        MacdCross(),
        OpeningRangeBreakout(minutes=30),
        AtrChannelBreakout(),
        KeltnerSqueeze(),
        TrendPullbackEMA(),
        DonchianBreakout(),
    ]
