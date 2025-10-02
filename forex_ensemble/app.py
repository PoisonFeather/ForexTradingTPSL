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

try:
    from .forwardtest import WalkForwardTester
except Exception:
    WalkForwardTester = None  # will check at runtime and warn if missing

cli = typer.Typer(help="Ensemble day-trading advisor (educational).")

# ============== BACKTEST (existing) ==============

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


# ============== ADVISE (existing) ==============

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


# ============== WALK-FORWARD (NEW) ==============

def _provider_from_name(name: str):
    if name == "alpha_vantage":
        from .data.alpha_vantage import AlphaVantageProvider
        return AlphaVantageProvider()
    elif name == "twelve_data":
        from .data.twelve_data import TwelveDataProvider
        return TwelveDataProvider()
    else:
        return MockProvider()

def _signals_from_strategies(candles_slice) -> list[dict]:
    """
    Adaptează StrategySignal -> dict pentru forward tester.
    Se apelează 'la zi' pe candles[:t+1].
    """
    out: list[dict] = []
    for strat in default_strategies():
        sg: StrategySignal = strat.generate(candles_slice)
        out.append({
            "strategy": sg.name,
            "dir": sg.direction,  # 'long'|'short'|'neutral'
            "entry": sg.entry,
            "sl": sg.stop_loss,
            "tp": sg.take_profit,
            "conf": float(sg.confidence or 0.0),
        })
    return out

@cli.command(name="walkforward", help="Run walk-forward forward testing (train→test→shift).")
def walkforward_cmd(
    symbol: str = typer.Option(settings.default_symbol, help="Ex: EURUSD"),
    timeframe: Timeframe = typer.Option(settings.default_timeframe, help="Ex: 15min"),
    provider: str = typer.Option("twelve_data", help="mock | alpha_vantage | twelve_data"),
    # date range OR limit (if dates omitted)
    start: str = typer.Option("", help="Start ISO date (optional), ex: 2025-06-01"),
    end: str = typer.Option("", help="End ISO date (optional), ex: 2025-10-01"),
    limit: int = typer.Option(8000, help="Limit candles if start/end not provided"),
    # walk-forward params
    train: int = typer.Option(2500, help="Train bars"),
    test: int = typer.Option(500, help="Test bars per split"),
    step: int = typer.Option(250, help="Shift (bars)"),
    mode: str = typer.Option("expanding", help="expanding | rolling"),
    # costs & logic
    spread: float = typer.Option(0.00002, help="Spread (price units)"),
    slippage: float = typer.Option(0.00001, help="Slippage (price units)"),
    fees: float = typer.Option(0.0, help="Per-trade fee in price units"),
    max_hold: int = typer.Option(200, help="Max bars to hold a trade"),
    oppose_damp: float = typer.Option(0.35, help="Damping for opposing strategy proposals"),
    out: str = typer.Option("", help="Optional CSV path for trades"),
):
    if WalkForwardTester is None:
        print("[red]WalkForwardTester indisponibil. Adaugă mai întâi forex_ensemble/forward.py și ensemble_v2.py.[/]")
        raise typer.Exit(code=1)
    asyncio.run(_walkforward(symbol, timeframe, provider, start, end, limit,
                             train, test, step, mode, spread, slippage, fees,
                             max_hold, oppose_damp, out))

async def _walkforward(symbol: str, timeframe: Timeframe, provider: str,
                       start: str, end: str, limit: int,
                       train: int, test: int, step: int, mode: str,
                       spread: float, slippage: float, fees: float,
                       max_hold: int, oppose_damp: float, out: str):
    mdp = _provider_from_name(provider)
    try:
        if start and end:
            all_candles = await mdp.get_recent_candles(symbol, timeframe, limit=20000)
            start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
            candles = [cd for cd in all_candles if start_dt <= cd.time <= end_dt]
        else:
            candles = await mdp.get_recent_candles(symbol, timeframe, limit=limit)

        if len(candles) < train + test + 10:
            print("[red]Prea puține lumânări pentru parametrii aleși. Mărește limit sau micșorează train/test.[/]")
            return

        tester = WalkForwardTester(
            provider_fn=None,
            strategies_fn=_signals_from_strategies,
            symbol=symbol,
            timeframe=timeframe,
            fees=fees,
            spread=spread,
            slippage=slippage,
            max_hold=max_hold,
        )
        wf = tester.walkforward(
            candles=candles,
            train=train,
            test=test,
            step=step,
            mode=mode,
            optimize_by="sharpe",
            oppose_damp=oppose_damp,
        )

        print(f"\n[bold cyan]Walk-Forward Summary[/]  Symbol={symbol}  TF={timeframe}  Splits={len(wf.splits)}")
        print(f"Trades={wf.metrics['trades']}  WinRate={wf.metrics['win_rate']:.2%}  "
              f"AvgPips={wf.metrics['avg_pips']:.2f}  Sharpe={wf.metrics['sharpe']:.2f}  "
              f"GrossPips={wf.metrics['gross_pips']:.1f}  MaxDD(Pips)={wf.metrics['max_dd_pips']:.1f}")

        if out:
            import csv
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["open_idx","close_idx","side","entry","exit","pips(price_units)","hit","conf"])
                for t in wf.trades:
                    w.writerow([t.open_idx, t.close_idx, t.side,
                                f"{t.entry:.6f}", f"{t.exit_price:.6f}",
                                t.result, t.hit, t.meta.get("conf", 0.0)])
            print(f"[green]Saved trades to {out}[/]")

    finally:
        await mdp.close()


if __name__ == "__main__":
    cli()
