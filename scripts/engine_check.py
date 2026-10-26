"""Does the spatial engine, playing real players, agree with the real-data model?

Takes each team's most-used 2025-26 five-man unit, builds engine players from
real data (nbasim.realplayers), plays pairs of them in the full engine, and
compares the engine's net rating per matchup with nbasim.realmodel's
prediction and with what those units actually did. Also prints league-level
numbers so a broken calibration shows up.

    python scripts/engine_check.py --pairs 15 --games 4
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nbasim.data.names import player_name  # noqa: E402
from nbasim.agents.team import TEND  # noqa: E402
from nbasim.game import accumulate, league_metrics, play_game, team_stats  # noqa: E402
from nbasim.realmodel import RealModel, lineup  # noqa: E402
from nbasim.realplayers import MODEL_PATH, real_lineup  # noqa: E402


def top_units(season: str) -> pd.DataFrame:
    p = pd.read_parquet(ROOT / "data" / "processed" / season / "possessions.parquet")
    off = p.groupby(["offense", "off_lineup"]).agg(n=("points", "size"), pts=("points", "sum")).reset_index()
    de = p.groupby(["defense", "def_lineup"]).agg(dn=("points", "size"), dpts=("points", "sum")).reset_index()
    u = off.merge(de, left_on=["offense", "off_lineup"], right_on=["defense", "def_lineup"])
    u = u.sort_values("n", ascending=False).groupby("offense").head(1)
    u["net"] = 100 * (u.pts / u.n - u.dpts / u.dn)
    return u.sort_values("n", ascending=False).reset_index(drop=True)


def player_check(engs, season: str) -> pd.DataFrame:
    """Per player, the engine's box score shares and percentages next to the real season's."""
    rs = json.loads(gzip.decompress((ROOT / "data" / "nba" / season / "stats" / "player_base.json.gz").read_bytes()))
    rs = rs["resultSets"][0]
    real = pd.DataFrame(rs["rowSet"], columns=rs["headers"]).set_index("PLAYER_ID")
    tot: dict[int, dict] = {}
    for e in engs:
        for i, p in enumerate(e.players):
            t = tot.setdefault(p.nba_id, {k: 0 for k in e.stats} | {"team_fga": 0, "team_tov": 0, "team_orb": 0})
            for k in e.stats:
                t[k] += int(e.stats[k][i])
            team = range(0, 5) if i < 5 else range(5, 10)
            t["team_fga"] += int(sum(e.stats["fga"][j] for j in team))
            t["team_tov"] += int(sum(e.stats["tov"][j] for j in team))
            t["team_orb"] += int(sum(e.stats["orb"][j] for j in team))
    rows = []
    for pid, t in tot.items():
        if pid not in real.index:
            continue
        r = real.loc[pid]
        rows.append({"player": r.PLAYER_NAME, "fga_share": t["fga"] / max(t["team_fga"], 1),
                     "fg3a_rate": t["fg3a"] / max(t["fga"], 1), "fg3_pct": t["fg3m"] / max(t["fg3a"], 1),
                     "fg_pct": t["fgm"] / max(t["fga"], 1), "fta_rate": t["fta"] / max(t["fga"], 1),
                     "tov_share": t["tov"] / max(t["team_tov"], 1), "orb_share": t["orb"] / max(t["team_orb"], 1),
                     "real_fg3a_rate": r.FG3A / max(r.FGA, 1), "real_fg3_pct": r.FG3_PCT, "real_fg_pct": r.FG_PCT,
                     "real_fta_rate": r.FTA / max(r.FGA, 1), "fga": t["fga"]})
    return pd.DataFrame(rows)


COMPONENTS = {  # engine rate, model rate from RealModel.breakdown
    "ppp": (lambda t: t["pts"] / t["poss"], None),
    "tov per chance": (lambda t: t["tov"] / t["chances"], lambda d: d["p_tov"]),
    "fta per chance": (lambda t: t["fta"] / t["chances"], lambda d: d["p_trip"]),
    "fg pts per fga": (lambda t: (t["pts"] - t["ftm"]) / t["fga"], lambda d: d["fg_pts_per_fga"]),
    "oreb per miss": (lambda t: t["orb"] / max(t["orb"] + t["opp_drb"], 1), lambda d: d["p_oreb_fga"]),
    "rim share": (lambda t: t["rim"] / t["fga"], lambda d: d["share_rim"]),
    "three share": (lambda t: t["fg3a"] / t["fga"], lambda d: d["share_corner3"] + d["share_above3"]),
    "rim fg": (lambda t: t["rim_made"] / max(t["rim"], 1), lambda d: d["fg_rim"]),
    "three fg": (lambda t: t["fg3m"] / max(t["fg3a"], 1),
                 lambda d: (d["share_corner3"] * d["fg_corner3"] + d["share_above3"] * d["fg_above3"])
                 / (d["share_corner3"] + d["share_above3"])),
}


def component_check(engs, pairs, games, model) -> pd.DataFrame:
    """Per offense-vs-defense matchup, engine rates against the model's; slope 1 means the engine keeps the spread."""
    sides = {}
    for k, (a, b) in enumerate(pairs):
        for g in range(games):
            e = engs[k * games + g]
            ia = 0 if g % 2 == 0 else 1
            for team, off, de in ((ia, a, b), (1 - ia, b, a)):
                s = team_stats(e, team)
                t = sides.setdefault((k, tuple(off), tuple(de)), {"pts": 0, "poss": 0, "tov": 0, "fta": 0, "ftm": 0,
                                                                   "fga": 0, "fg3a": 0, "fg3m": 0, "orb": 0,
                                                                   "opp_drb": 0, "rim": 0, "rim_made": 0})
                for key in ("pts", "poss", "tov", "fta", "ftm", "fga", "fg3a", "fg3m", "orb", "opp_drb"):
                    t[key] += s[key]
                t["rim"] += s["zone_att"]["rim"]
                t["rim_made"] += s["zone_made"]["rim"]
    rows = []
    for (k, off, de), t in sides.items():
        t["chances"] = t["poss"] + t["orb"]
        d = model.breakdown(list(off), list(de))
        r = {"pair": k, "model_ppp": model.expected_points(list(off), list(de))}
        for name, (fe, fm) in COMPONENTS.items():
            r["engine " + name] = fe(t)
            r["model " + name] = r["model_ppp"] if fm is None else fm(d)
        rows.append(r)
    return pd.DataFrame(rows)


def run(job):
    a_ids, b_ids, seed, quarters, tend = job
    TEND.update(tend)
    a, b = real_lineup(a_ids, "A"), real_lineup(b_ids, "B")
    eng = play_game(a, b, seed=seed, quarters=quarters).engine
    return eng


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--games", type=int, default=4, help="games per pair (sides alternate)")
    ap.add_argument("--quarters", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dump", help="write the per-matchup component table to this csv")
    ap.add_argument("--tend", nargs="*", default=[], help="override team AI tendencies, e.g. zone_pref_weight=0.5")
    args = ap.parse_args()
    tend = {k: float(v) for k, v in (t.split("=") for t in args.tend)}

    model = RealModel.load(MODEL_PATH)
    units = top_units(args.season).head(2 * args.pairs)
    pairs = [(lineup(units.off_lineup[2 * k]), lineup(units.off_lineup[2 * k + 1])) for k in range(args.pairs)]
    jobs = []
    for k, (a, b) in enumerate(pairs):
        for g in range(args.games):
            jobs.append((a, b, 1000 * k + g, args.quarters, tend) if g % 2 == 0 else (b, a, 1000 * k + g, args.quarters, tend))
    with ProcessPoolExecutor(args.workers) as ex:
        engs = list(ex.map(run, jobs))

    rows = []
    for k, (a, b) in enumerate(pairs):
        pts_a = pts_b = poss_a = poss_b = 0
        for g in range(args.games):
            e = engs[k * args.games + g]
            ia, ib = (0, 1) if g % 2 == 0 else (1, 0)
            pts_a += e.score[ia]
            pts_b += e.score[ib]
            poss_a += e.possessions[ia]
            poss_b += e.possessions[ib]
        eng_net = 100 * (pts_a / poss_a - pts_b / poss_b)
        model_net = 100 * (model.expected_points(a, b) - model.expected_points(b, a))
        rows.append({"A": "/".join(player_name(p).split()[-1] for p in a),
                     "B": "/".join(player_name(p).split()[-1] for p in b),
                     "engine_net": eng_net, "model_net": model_net,
                     "unit_net_A_minus_B": units.net[2 * k] - units.net[2 * k + 1],
                     "poss_per_game": (poss_a + poss_b) / 2 / args.games})
    df = pd.DataFrame(rows)
    print(df.round(1).to_string(index=False))
    print(f"\ncorr(engine, model) = {np.corrcoef(df.engine_net, df.model_net)[0, 1]:.2f}; "
          f"corr(engine, actual unit gap) = {np.corrcoef(df.engine_net, df.unit_net_A_minus_B)[0, 1]:.2f}; "
          f"corr(model, actual unit gap) = {np.corrcoef(df.model_net, df.unit_net_A_minus_B)[0, 1]:.2f}")
    pc = player_check(engs, args.season)
    print("\nper player, engine vs real (correlation over", len(pc), "players):")
    for k in ("fg3a_rate", "fg3_pct", "fg_pct", "fta_rate"):
        print(f"  {k}: corr {np.corrcoef(pc[k], pc['real_' + k])[0, 1]:.2f}, "
              f"engine mean {np.average(pc[k], weights=pc.fga):.3f} vs real {np.average(pc['real_' + k], weights=pc.fga):.3f}")
    print(pc.sort_values("fga", ascending=False).head(12).round(3).to_string(index=False))
    cc = component_check(engs, pairs, args.games, model)
    print("\nper matchup side, engine vs model (slope of engine on model; 1 = engine keeps the model's spread):")
    for name in COMPONENTS:
        x, y = cc["model " + name], cc["engine " + name]
        slope = np.polyfit(x, y, 1)[0]
        print(f"  {name:15s} slope {slope:5.2f}  corr {np.corrcoef(x, y)[0, 1]:5.2f}  "
              f"sd model {x.std():.3f} engine {y.std():.3f}  mean model {x.mean():.3f} engine {y.mean():.3f}")
    if args.dump:
        cc.to_csv(args.dump, index=False)
    lm = league_metrics(accumulate(engs))
    print("engine league numbers with real players:",
          {k: round(v, 3) for k, v in lm.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
