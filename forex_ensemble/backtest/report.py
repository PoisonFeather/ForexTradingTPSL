# forex_ensemble/backtest/report.py
from __future__ import annotations
import csv
from pathlib import Path
from .engine import BTRun, Trade
from .metrics import summarize

def save_report(run: BTRun, out_dir: str, risk_pct: float) -> dict:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    # trades.csv
    with open(p/"trades.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time","close_time","side","entry","sl","tp","exit","result_r","notes"])
        for tr in run.trades:
            w.writerow([tr.open_time.isoformat(), tr.close_time.isoformat() if tr.close_time else "",
                        tr.side, f"{tr.entry:.5f}", f"{tr.sl:.5f}", f"{tr.tp:.5f}",
                        f"{tr.exit:.5f}" if tr.exit else "", f"{tr.result_r:.3f}" if tr.result_r is not None else "", tr.notes])

    # equity.csv
    with open(p/"equity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time","equity"])
        for ts, eq in run.equity_curve:
            w.writerow([ts.isoformat(), f"{eq:.2f}"])

    summary = summarize(run.trades, run.equity_curve, risk_pct)
    with open(p/"summary.txt", "w") as f:
        f.write("Backtest summary\n")
        for k,v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write(f"start_equity: {run.start_equity:.2f}\nend_equity: {run.end_equity:.2f}\n")
    return summary
