"""Backtest lineup predictions on held-out real games.

Every model sees the same training possessions (all games before --cutoff)
and predicts the expected points of each held-out possession from who was on
the floor, home court and how the possession started. Nothing from the test
period is used for fitting.

    python scripts/backtest.py                      # train before 2026-01-01, test the rest of 2025-26
    python scripts/backtest.py --cutoff 2025-10-01  # preseason: test all of 2025-26

Models:
  context   league average by how the possession started and home court
  team      ridge regression on offense team + defense team
  rapm      ridge regression on the ten players on the floor (RAPM)
  real      nbasim.realmodel: turnovers, fouls drawn, shot making by zone, offensive rebounds
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nbasim.baselines import ContextModel, RidgeModel  # noqa: E402
from nbasim.realmodel import SEASON_WEIGHT, RealModel  # noqa: E402

PROC = ROOT / "data" / "processed"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]


def load(seasons) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(PROC / s / "possessions.parquet") for s in seasons], ignore_index=True)
    df["w"] = df.season.map(SEASON_WEIGHT).fillna(1.0)
    return df


def tune_ridge(kind, train, grid):
    """Pick alpha on the last 15% of training possessions by date."""
    cut = train.date.sort_values().iloc[int(len(train) * 0.85)]
    fit_part, val = train[train.date < cut], train[train.date >= cut]
    best = None
    for a in grid:
        mse = np.mean((RidgeModel(kind, a).fit(fit_part).predict(val) - val.points) ** 2)
        if best is None or mse < best[1]:
            best = (a, mse)
    return best[0]


def tune_real(train, grid, resid_grid, rapm_alpha):
    """Pick the real model's regularisation, then the residual ridge's, on the last 15% of training."""
    cut = train.date.sort_values().iloc[int(len(train) * 0.85)]
    fit_part, val = train[train.date < cut], train[train.date >= cut]
    best = None
    for c in grid:
        m = RealModel.fit(fit_part, C={k: c for k in ("tov", "trip", "make", "oreb")}, verbose=False)
        pv = m.predict(val)
        mse = np.mean((pv - val.points) ** 2)
        print(f"  real C={c}: val mse {mse:.5f}", flush=True)
        if best is None or mse < best[1]:
            best = (c, mse, m, pv)
    c, _, m, pv = best
    fit_resid = fit_part.points - m.predict(fit_part)
    best_a = None
    for a in resid_grid:
        r = RidgeModel("rapm", a).fit(fit_part.assign(points=fit_resid))
        mse = np.mean((pv + r.predict(val) - val.points) ** 2)
        print(f"  residual alpha={a}: val mse {mse:.5f}", flush=True)
        if best_a is None or mse < best_a[1]:
            best_a = (a, mse)
    return c, best_a[0]


def paired_bootstrap(test, pred_a, pred_b, n=2000, seed=0):
    """95% interval of game-margin RMSE(a) - RMSE(b), resampling games."""
    def sq_err(pred):
        g = test.assign(pred=pred).groupby(["game_id", "home"]).agg(pts=("points", "sum"), pred=("pred", "sum"))
        g = g.unstack("home")
        return ((g[("pts", True)] - g[("pts", False)]) - (g[("pred", True)] - g[("pred", False)])).values ** 2
    ea, eb = sq_err(pred_a), sq_err(pred_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ea), size=(n, len(ea)))
    d = np.sqrt(ea[idx].mean(1)) - np.sqrt(eb[idx].mean(1))
    return float(np.sqrt(ea.mean()) - np.sqrt(eb.mean())), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def metrics(test: pd.DataFrame, pred: np.ndarray, base_mse: float) -> dict:
    t = test.assign(pred=pred)
    mse = float(np.mean((t.pred - t.points) ** 2))
    # games: predicted vs actual margin, using the possessions that were actually played
    g = t.groupby(["game_id", "home"]).agg(pts=("points", "sum"), pred=("pred", "sum"))
    g = g.unstack("home")
    margin = g[("pts", True)] - g[("pts", False)]
    pmargin = g[("pred", True)] - g[("pred", False)]
    # lineups: net rating per 100 possessions for five-man units with enough test possessions
    off = t.groupby("off_lineup").agg(n=("points", "size"), pts=("points", "sum"), pred=("pred", "sum"))
    de = t.groupby("def_lineup").agg(n=("points", "size"), pts=("points", "sum"), pred=("pred", "sum"))
    lu = off.join(de, lsuffix="_o", rsuffix="_d", how="inner")
    lu = lu[(lu.n_o + lu.n_d) >= 200]
    net = 100 * (lu.pts_o / lu.n_o - lu.pts_d / lu.n_d)
    pnet = 100 * (lu.pred_o / lu.n_o - lu.pred_d / lu.n_d)
    wts = lu.n_o + lu.n_d
    wc = np.cov(net, pnet, aweights=wts)
    return {
        "poss_mse": mse,
        "poss_skill_vs_context_%": 100 * (1 - mse / base_mse),
        "game_margin_rmse": float(np.sqrt(np.mean((margin - pmargin) ** 2))),
        "game_margin_corr": float(np.corrcoef(margin, pmargin)[0, 1]),
        "winner_accuracy": float(np.mean(np.sign(margin) == np.sign(pmargin))),
        "lineups": int(len(lu)),
        "lineup_net_corr": float(wc[0, 1] / np.sqrt(wc[0, 0] * wc[1, 1])),
        "lineup_net_rmse": float(np.sqrt(np.average((net - pnet) ** 2, weights=wts))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-01-01", help="train on games before this date")
    ap.add_argument("--test-season", default="2025-26")
    ap.add_argument("--out", default=str(ROOT / "data" / "backtest"))
    args = ap.parse_args()

    df = load(SEASONS)
    train = df[df.date < args.cutoff]
    test = df[(df.date >= args.cutoff) & (df.season == args.test_season)]
    print(f"train {len(train):,} possessions ({train.date.min()} to {train.date.max()}), "
          f"test {len(test):,} ({test.game_id.nunique()} games, {test.date.min()} to {test.date.max()})")

    preds = {}
    t0 = time.time()
    preds["context"] = ContextModel().fit(train).predict(test)
    a_team = tune_ridge("team", train, [100, 1000, 5000, 20000, 50000])
    preds["team"] = RidgeModel("team", a_team).fit(train).predict(test)
    a_rapm = tune_ridge("rapm", train, [300, 1000, 3000, 10000])
    preds["rapm"] = RidgeModel("rapm", a_rapm).fit(train).predict(test)
    print(f"baselines fit in {time.time() - t0:.0f}s (team alpha {a_team}, rapm alpha {a_rapm})")
    t0 = time.time()
    c_real, a_resid = tune_real(train, [0.002, 0.006, 0.02], [1000, 3000, 9000, 27000], a_rapm)
    model = RealModel.fit(train, C={k: c_real for k in ("tov", "trip", "make", "oreb")}, verbose=False)
    preds["real"] = model.predict(test)
    # structured model + a RAPM-style ridge on what it still misses
    train_real = model.predict(train)
    resid = RidgeModel("rapm", a_resid).fit(train.assign(points=train.points - train_real))
    preds["real+resid"] = preds["real"] + resid.predict(test)
    print(f"real model (C {c_real}, residual alpha {a_resid}) tuned, fit + predict in {time.time() - t0:.0f}s")

    base = float(np.mean((preds["context"] - test.points) ** 2))
    res = {k: metrics(test, v, base) for k, v in preds.items()}
    table = pd.DataFrame(res).T
    print(table.round(4).to_string())
    for k in ("real", "real+resid"):
        d, lo, hi = paired_bootstrap(test, preds[k], preds["rapm"])
        print(f"game margin RMSE, {k} minus rapm: {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")
    print(f"actual test PPP {test.points.mean():.4f}; mean predictions: "
          + ", ".join(f"{k} {v.mean():.4f}" for k, v in preds.items()))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"cutoff_{args.cutoff}"
    (out / f"{tag}.json").write_text(json.dumps({"cutoff": args.cutoff, "train": len(train),
                                                 "test": len(test), "alphas": {"team": a_team, "rapm": a_rapm, "real_C": c_real, "resid": a_resid},
                                                 "results": res}, indent=2))
    test.assign(**{f"pred_{k}": v for k, v in preds.items()})[
        ["poss_id", "game_id", "date", "offense", "defense", "home", "off_lineup", "def_lineup", "start_type",
         "points"] + [f"pred_{k}" for k in preds]].to_parquet(out / f"{tag}_predictions.parquet", index=False)
    model.save(out / f"realmodel_{tag}.pkl")


if __name__ == "__main__":
    main()
