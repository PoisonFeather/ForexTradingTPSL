from forex_ensemble.ensemble.aggregator import aggregate_signals, AggregationConfig
from forex_ensemble.types import StrategySignal
from datetime import datetime

def sig(name, d, e, sl, tp, c):
    return StrategySignal(name, d, e, sl, tp, c, "")

def test_consensus_long():
    signals = [
        sig("A","long",1.10,1.099,1.102,0.7),
        sig("B","long",1.101,1.100,1.103,0.6),
        sig("C","short",1.1005,1.1015,1.0995,0.2),
    ]
    dec = aggregate_signals("EURUSD","5min",1.1010,signals)
    assert dec.direction == "long"
    assert dec.entry is not None
    assert dec.take_profit is not None
    assert dec.stop_loss is not None
    assert 0.0 <= dec.confidence <= 1.0
