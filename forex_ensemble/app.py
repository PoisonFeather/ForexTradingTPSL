# forex_ensemble/app.py
from __future__ import annotations
import asyncio
import json
from dataclasses import asdict
from rich import print
from rich.table import Table
import typer
from .config import settings
from .types import Timeframe, StrategySignal
from .data.mock_provider import MockProvider
from .strategies import default_strategies
from .ensemble.aggregator import aggregate_signals, AggregationConfig

cli = typer.Typer(help="Ensemble day-trading advisor (educational).")

@cli.command()
def advise(
    symbol: str = typer.Option(settings.default_symbol, help="Ex: EURUSD"),
    timeframe: Timeframe = typer.Option(settings.default_timeframe, help="Ex: 5min"),
    limit: int = typer.Option(500, help="Număr lumânări de analizat"),
    provider: str = typer.Option("mock", help="mock | alpha_vantage | twelve_data"),
    json_out: bool = typer.Option(False, "--json", help="Output în format JSON"),
):
    asyncio.run(_advise(symbol, timeframe, limit, provider, json_out))

async def _advise(symbol: str, timeframe: Timeframe, limit: int, provider: str, json_out: bool):
    if provider == "alpha_vantage":
        from .data.alpha_vantage import AlphaVantageProvider
        mdp = AlphaVantageProvider()
    elif provider == "twelve_data":
        from .data.twelve_data import TwelveDataProvider
        mdp = TwelveDataProvider()
    else:
        mdp = MockProvider()

    try:
        candles = await mdp.get_recent_candles(symbol, timeframe, limit=limit)
        quote = await mdp.get_quote(symbol)
        print(f"[bold cyan]Loaded {len(candles)} candles for {symbol} / {timeframe}[/]")

        # 1) rulează strategiile
        strategies = default_strategies()
        signals: list[StrategySignal] = [s.generate(candles) for s in strategies]

        # 2) afișează semnale (diagnostic)
        table = Table(title="Signals (raw)", show_lines=True)
        table.add_column("Strategy"); table.add_column("Dir")
        table.add_column("Entry"); table.add_column("SL")
        table.add_column("TP"); table.add_column("Conf"); table.add_column("Notes")
        for sg in signals:
            table.add_row(
                sg.name, sg.direction,
                f"{sg.entry:.5f}" if sg.entry else "-",
                f"{sg.stop_loss:.5f}" if sg.stop_loss else "-",
                f"{sg.take_profit:.5f}" if sg.take_profit else "-",
                f"{sg.confidence:.2f}", sg.notes[:60],
            )
        print(table)

        # 3) agregare
        decision = aggregate_signals(symbol, timeframe, price_now=quote.ask, signals=signals)

        # 4) output final
        if json_out:
            out = {
                "symbol": decision.symbol,
                "timeframe": decision.timeframe,
                "timestamp": decision.timestamp.isoformat(),
                "direction": decision.direction,
                "confidence": round(decision.confidence, 4),
                "entry": round(decision.entry, 6) if decision.entry else None,
                "stop_loss": round(decision.stop_loss, 6) if decision.stop_loss else None,
                "take_profit": round(decision.take_profit, 6) if decision.take_profit else None,
            }
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print("\n[bold magenta]Ensemble Decision[/]")
            print(f"Direction: [bold]{decision.direction.upper()}[/]  |  Confidence: [bold]{decision.confidence:.2f}[/]")
            if decision.entry and decision.stop_loss and decision.take_profit:
                rr = abs((decision.take_profit - decision.entry) / (decision.entry - decision.stop_loss))
                print(
                    f"Entry: {decision.entry:.5f} | SL: {decision.stop_loss:.5f} | "
                    f"TP: {decision.take_profit:.5f} | R:R≈{rr:.2f}"
                )
            else:
                print("[yellow]Semnal insuficient pentru TP/SL agregat (probabil consens slab sau puține strategii valabile).[/]")

    finally:
        await mdp.close()

if __name__ == "__main__":
    cli()
