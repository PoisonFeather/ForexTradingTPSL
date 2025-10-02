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
from .backtest.engine import backtest, BTConfig
from .backtest.report import save_report
from datetime import datetime, timezone
import pandas as pd


cli = typer.Typer(help="Ensemble day-trading advisor (educational).")

@cli.command(name="backtest-cmd")
def backtest_cmd(
    symbol: str = typer.Option(settings.default_symbol, help="Ex: EURUSD"),
    timeframe: Timeframe = typer.Option(settings.default_timeframe, help="Ex: 5min"),
    provider: str = typer.Option("twelve_data", help="mock | alpha_vantage | twelve_data"),
    start: str = typer.Option(..., help="Start ISO date, ex: 2025-08-01"),
    end: str = typer.Option(..., help="End ISO date, ex: 2025-10-01"),
    spread: float = typer.Option(0.00010, help="Spread aprox (majors)"),
    slippage_pips: float = typer.Option(0.8, help="Slippage în pips"),
    risk_pct: float = typer.Option(1.0, help="Risc per tranzacție (%)"),
    rr_required: float = typer.Option(1.8, help="R:R minim al semnalului"),
    entry_buffer_atr: float = typer.Option(0.10, help="Buffer ATR pentru entry"),
    active_start_hour: int = typer.Option(7, help="Oră start (local time)"),
    active_end_hour: int = typer.Option(20, help="Oră end (local time)"),
    cooldown_bars: int = typer.Option(6, help="Bare de pauză după un trade"),
    max_trades_per_day: int = typer.Option(10, help="Limită tranzacții/zi"),
    trend_slope_threshold: float = typer.Option(0.6, help="Prag panta EMA50/ATR (trend)"),
    out_dir: str = typer.Option("bt_out", help="Folder rezultate"),
):
    """
    Backtest cu filtre de regim, orar, cooldown și cap/zi. Scrie trades.csv, equity.csv, summary.txt
    """
    asyncio.run(_backtest(symbol, timeframe, provider, start, end, spread, slippage_pips,
                          risk_pct, rr_required, entry_buffer_atr,
                          active_start_hour, active_end_hour,
                          cooldown_bars, max_trades_per_day, trend_slope_threshold, out_dir))

async def _backtest(symbol, timeframe, provider, start, end, spread, slippage_pips,
                    risk_pct, rr_required, entry_buffer_atr,
                    active_start_hour, active_end_hour,
                    cooldown_bars, max_trades_per_day, trend_slope_threshold, out_dir):
    # provider
    if provider == "alpha_vantage":
        from .data.alpha_vantage import AlphaVantageProvider
        mdp = AlphaVantageProvider()
    elif provider == "twelve_data":
        from .data.twelve_data import TwelveDataProvider
        mdp = TwelveDataProvider()
    else:
        mdp = MockProvider()

    try:
        all_candles = await mdp.get_recent_candles(symbol, timeframe, limit=5000)
    except Exception as e:
        print(f"[yellow]Provider error: {e}. Comut pe mock.[/]")
        mdp = MockProvider()
        all_candles = await mdp.get_recent_candles(symbol, timeframe, limit=5000)
    finally:
        await mdp.close()

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    candles = [cd for cd in all_candles if start_dt <= cd.time <= end_dt]
    if len(candles) < 200:
        print("[red]Prea puține lumânări după filtrare; extinde intervalul sau crește limit.[/]")
        return

    print(f"[bold cyan]Backtesting {symbol} / {timeframe} între {start} și {end} cu {len(candles)} bare...[/]")

    cfg = BTConfig(
        entry_buffer_atr=entry_buffer_atr,
        rr_required=rr_required,
        spread=spread,
        slippage_pips=slippage_pips,
        min_confidence=0.05,
        active_start_hour=active_start_hour,
        active_end_hour=active_end_hour,
        cooldown_bars=cooldown_bars,
        max_trades_per_day=max_trades_per_day,
        trend_slope_threshold=trend_slope_threshold,
    )
    run = backtest(candles, cfg=cfg, initial_equity=10_000.0, risk_pct_per_trade=risk_pct)
    summary = save_report(run, out_dir, risk_pct)
    print("[bold magenta]Summary[/]")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"start_equity: {run.start_equity:.2f}  end_equity: {run.end_equity:.2f}")
    print(f"[green]Fișiere scrise în ./{out_dir}[/]")


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
