"""Replay the live forecast routine over a past season, one weekly freeze at a time.

    python scripts/walkforward.py --season 2025-26                  # about 1-2 hours on 4 cores
    python scripts/walkforward.py --season 2025-26 --start 2026-01-05 --weeks 3   # quick check

Each Monday 12:00 UTC it runs exactly what `scripts/forecast.py freeze --days 7` runs live, using
only games played before that Monday (models refit every --refit-days), then scores every batch
with `scripts/forecast.py score`. No market lines exist for past seasons, so this compares our
model with the RAPM and team-rating baselines. The game-margin RMSE of each model is the sigma
that data/forecast_config.json uses to turn a margin into a win probability.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import forecast as FC  # noqa: E402
from nbasim import forecast as F  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--start", default=None, help="first Monday (default: third Monday of the season)")
    ap.add_argument("--weeks", type=int, default=99)
    ap.add_argument("--refit-days", type=int, default=7)
    ap.add_argument("--dir", default=str(ROOT / "data" / "backtest" / "walkforward"))
    args = ap.parse_args()

    cfg = json.loads(FC.CONFIG.read_text())
    poss = FC.load_poss()
    sched = F.schedule_from_boxscores(FC.NBA / args.season / "boxscores.jsonl.gz")
    first = pd.Timestamp(sched.date.min())
    start = pd.Timestamp(args.start) if args.start else first + pd.offsets.Week(weekday=0) + pd.Timedelta(days=14)
    last = pd.Timestamp(sched.date.max())
    out = Path(args.dir)
    cache = out / "models"
    w = start
    for _ in range(args.weeks):
        if w > last:
            break
        ns = SimpleNamespace(now=f"{w:%Y-%m-%d}T12:00:00Z", season=args.season, days=7, out="",
                             refit_days=args.refit_days, dir=str(out), schedule_from_box=True,
                             cache=str(cache), quiet=True)
        print(f"== week of {w:%Y-%m-%d}", flush=True)
        FC.cmd_freeze(ns, cfg, poss)
        w += pd.Timedelta(days=7)
    FC.cmd_score(SimpleNamespace(season=args.season, dir=str(out)), cfg)
    g = pd.read_csv(out / args.season / "scored_games.csv")
    sig = {s: round(float(((g.actual_margin - g[f"margin_{s}"]) ** 2).mean() ** 0.5), 2)
           for s in ("real", "rapm", "team") if f"margin_{s}" in g}
    print(f"\nsigma (game margin RMSE) per model: {sig}")


if __name__ == "__main__":
    main()
