from __future__ import annotations
import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    alpha_vantage_api_key: str | None = os.getenv("ALPHA_VANTAGE_API_KEY")
    default_symbol: str = os.getenv("DEFAULT_SYMBOL", "EURUSD")
    default_timeframe: str = os.getenv("DEFAULT_TIMEFRAME", "5min")
    # extensibil: poți adăuga alți provideri (Oanda, TwelveData, etc.)

settings = Settings()
