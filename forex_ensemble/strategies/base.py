from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence
from ..types import Candle, StrategySignal

class Strategy(ABC):
    name: str

    @abstractmethod
    def generate(self, candles: Sequence[Candle]) -> StrategySignal:
        """
        Primește o secvență de lumânări (cele mai noi la final) și întoarce
        un semnal cu entry/SL/TP + confidence 0..1.
        Implementările trebuie să fie pure (fără efecte secundare).
        """
