"""Fit the real-data player model on every possession we have and save it.

    python scripts/fit_model.py        # -> data/models/realmodel.pkl

Regularisation comes from the latest backtest (data/backtest/cutoff_*.json) when present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nbasim.realmodel import RealModel  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "models" / "realmodel.pkl"


def main() -> None:
    poss = pd.concat([pd.read_parquet(p) for p in sorted(PROC.glob("*/possessions.parquet"))], ignore_index=True)
    c, alpha = 0.006, 9000
    runs = sorted((ROOT / "data" / "backtest").glob("cutoff_*.json"))
    if runs:
        a = json.loads(runs[-1].read_text())["alphas"]
        c, alpha = a.get("real_C", c), a.get("resid", alpha)
    print(f"fitting on {len(poss):,} possessions ({poss.date.min()} to {poss.date.max()}), C={c}, residual alpha={alpha}")
    m = RealModel.fit(poss, C={k: c for k in ("tov", "trip", "make", "oreb", "zone")})
    m.fit_residual(poss, alpha)
    m.trained_through = str(poss.date.max())
    m._tab = None
    m.save(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
