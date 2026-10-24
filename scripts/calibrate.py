"""Compare simulated league numbers with the 2024-25 NBA targets.

  python scripts/calibrate.py --games 16             # average vs average lineups
  python scripts/calibrate.py --games 8 --stars      # the two all-star lineups

The average lineup (data/players/calibration.yaml) is what the constants in
nbasim/resolve.py are tuned on; star lineups should come out above league
average.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nbasim.game import accumulate, league_metrics, play_game, tracking  # noqa: E402
from nbasim.players import DATA_DIR, Lineup, load_lineups, load_roster  # noqa: E402


def lineups(stars: bool):
    if stars:
        L = load_lineups()
        return L["stars_a"], L["stars_b"]
    avg = load_roster(DATA_DIR / "players" / "calibration.yaml").players
    return Lineup(name="Average A", players=avg), Lineup(name="Average B", players=avg)


def run(args):
    seed, stars, quarters = args
    a, b = lineups(stars)
    eng = play_game(a, b, seed=seed, quarters=quarters).engine
    eng.hist = []
    eng.rng = None
    return eng


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--quarters", type=int, default=4)
    ap.add_argument("--stars", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    with ProcessPoolExecutor(args.workers) as ex:
        engs = list(ex.map(run, [(s, args.stars, args.quarters) for s in range(args.games)]))
    m = league_metrics(accumulate(engs))
    target = yaml.safe_load((ROOT / "data" / "league" / "nba_2024_25.yaml").read_text())
    d = target["derived"]
    pg = target["per_game"]
    ref = {
        "points_per_possession": d["points_per_possession"], "tov_pct": d["tov_pct"], "orb_pct": d["orb_pct"],
        "ft_rate": d["ft_rate"], "ft_pct": pg["ft_pct"], "fg_pct": pg["fg_pct"],
        "fg3_rate": pg["fg3a"] / pg["fga"], "fg3_pct": pg["fg3_pct"], "stl_share_of_tov": d["stl_share_of_tov"],
        "blk_pct_of_fga": d["blk_pct_of_fga"], "ast_pct_of_fgm": d["ast_pct_of_fgm"], "pace_poss_per_48": 100.0,
    }
    print(f"{len(engs)} games, scores: {[e.score for e in engs]}")
    print(f"{'metric':24s} {'sim':>8s} {'NBA':>8s}")
    for k, v in m.items():
        if k == "shot_locations":
            continue
        print(f"{k:24s} {v:8.3f} {ref.get(k, float('nan')):8.3f}")
    for z, v in m["shot_locations"].items():
        t = target["shot_locations"][z]
        print(f"{z:10s} freq {v['freq']:.3f} ({t['freq']:.3f})   fg {v['fg_pct']:.3f} ({t['fg_pct']:.3f})")
    print("tracking (NBA ~2.8 passes/poss, ~50 drives/team-game, top option ~29% of FGA):",
          {k: round(v, 2) for k, v in tracking(engs).items()})
    tov = Counter()
    for e in engs:
        tov.update(e.tov_types)
    fouls = Counter()
    for e in engs:
        fouls.update(e.foul_types)
    per = len(engs) * 2 * args.quarters / 4
    print("turnovers per team-game by type:", {k: round(v / per, 2) for k, v in tov.most_common()})
    print("fouls per team-game by type:", {k: round(v / per, 2) for k, v in fouls.most_common()})
    viol = Counter(ev.data.get("rule") for e in engs for ev in e.events if ev.type == "violation")
    print("violations per team-game:", {k: round(v / per, 2) for k, v in viol.most_common()})
    tot = accumulate(engs)
    print("per team-game: FGA %.1f  3PA %.1f  FTA %.1f  ORB %.1f  TOV %.1f  AST %.1f  STL %.1f  BLK %.1f  PTS %.1f  poss %.1f" % tuple(
        tot[k] / per for k in ("fga", "fg3a", "fta", "orb", "tov", "ast", "stl", "blk", "pts", "poss")))


if __name__ == "__main__":
    main()
