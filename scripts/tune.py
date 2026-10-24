"""Coordinate-descent tuner for the engine and AI constants.

Plays league-average lineups against each other with fixed seeds and moves
one constant at a time while the distance to the 2024-25 NBA targets drops.
The best constants are written to data/league/tuned_params.yaml, which
nbasim loads at import.

  python scripts/tune.py --minutes 40
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nbasim import resolve  # noqa: E402
from nbasim.agents import team  # noqa: E402
from nbasim.game import accumulate, league_metrics, play_game, tracking  # noqa: E402
from nbasim.players import DATA_DIR, Lineup, load_lineups, load_roster  # noqa: E402

# (group, name, step, low, high)
SPACE = [
    ("tendencies", "cont_base", 0.03, 0.5, 1.0),
    ("tendencies", "cont_slope", 0.04, 0.1, 0.8),
    ("tendencies", "bias_short_mid", 0.03, -0.3, 0.3),
    ("tendencies", "bias_long_mid", 0.03, -0.3, 0.3),
    ("tendencies", "bias_three", 0.03, -0.3, 0.3),
    ("tendencies", "drive_bias", 0.03, -0.3, 0.3),
    ("tendencies", "pass_cost", 0.02, -0.1, 0.3),
    ("tendencies", "temp", 0.02, 0.02, 0.3),
    ("tendencies", "self_bar_exp", 0.5, 0.0, 4.0),
    ("tendencies", "setup_patience", 0.04, 0.0, 0.5),
    ("tendencies", "ball_movement", 0.03, -0.2, 0.4),
    ("tendencies", "redrive_penalty", 0.04, 0.0, 0.6),
    ("tendencies", "touch_decay", 0.02, 0.0, 0.3),
    ("tendencies", "usage_weight", 0.3, 0.0, 4.0),
    ("tendencies", "floater_min", 1.0, 4.0, 9.0),
    ("resolve", "foul_rim", 0.03, 0.05, 0.5),
    ("resolve", "foul_short_mid", 0.02, 0.02, 0.3),
    ("resolve", "reach_foul_rate", 0.01, 0.0, 0.15),
    ("resolve", "steal_base", 0.02, 0.02, 0.4),
    ("resolve", "strip_rate", 0.005, 0.005, 0.1),
    ("resolve", "defense_reb_bias", 0.3, 1.0, 6.0),
    ("resolve", "block_base", 0.02, 0.05, 0.4),
    ("resolve", "block_jumper", 0.01, 0.0, 0.15),
    ("resolve", "base_above3", 0.01, 0.3, 0.5),
    ("resolve", "base_rim", 0.01, 0.6, 0.85),
    ("resolve", "base_short_mid", 0.01, 0.35, 0.6),
    ("resolve", "base_long_mid", 0.01, 0.35, 0.6),
    ("resolve", "base_corner3", 0.01, 0.35, 0.55),
]

TARGET = yaml.safe_load((ROOT / "data" / "league" / "nba_2024_25.yaml").read_text())
T_LOC = TARGET["shot_locations"]
GOALS = [  # (metric path, target, tolerance)
    (("points_per_possession",), TARGET["derived"]["points_per_possession"], 0.03),
    (("pace_poss_per_48",), 100.0, 4.0),
    (("tov_pct",), TARGET["derived"]["tov_pct"], 0.012),
    (("orb_pct",), TARGET["derived"]["orb_pct"], 0.02),
    (("ft_rate",), TARGET["derived"]["ft_rate"], 0.025),
    (("blk_pct_of_fga",), TARGET["derived"]["blk_pct_of_fga"], 0.01),
    (("ast_pct_of_fgm",), TARGET["derived"]["ast_pct_of_fgm"], 0.05),
    (("stl_share_of_tov",), TARGET["derived"]["stl_share_of_tov"], 0.08),
    (("fg3_pct",), TARGET["per_game"]["fg3_pct"], 0.02),
] + [(("shot_locations", z, "freq"), v["freq"], 0.02) for z, v in T_LOC.items()] \
  + [(("shot_locations", z, "fg_pct"), v["fg_pct"], 0.035) for z, v in T_LOC.items()]


def _apply(params: dict) -> None:
    resolve.CAL.update(params.get("resolve", {}))
    team.TEND.update(params.get("tendencies", {}))


# NBA.com player tracking, 2024-25 league averages (approximate): ~50 drives
# and ~280 passes per team-game at ~99.6 possessions; the top option on a
# team takes roughly 28-30% of its shots.
TRACKING = {"passes_per_poss": (2.8, 0.4), "drives_per_game": (50.0, 8.0), "top_fga_share": (0.29, 0.04)}
STAR_GAMES = 6
STAR_PACE = (100.0, 5.0)  # star lineups should still play at an NBA pace


def _game(args):
    seed, params, stars = args
    _apply(params)
    if stars:
        L = load_lineups()
        a, b = L["stars_a"], L["stars_b"]
    else:
        avg = load_roster(DATA_DIR / "players" / "calibration.yaml").players
        a, b = Lineup(name="A", players=avg), Lineup(name="B", players=avg)
    eng = play_game(a, b, seed=seed, quarters=1).engine
    eng.hist = []
    eng.rng = None
    return eng


def _get(m, path):
    for k in path:
        m = m[k]
    return m


def evaluate(pool, params, games):
    jobs = [(s, params, False) for s in range(games)] + [(1000 + s, params, True) for s in range(STAR_GAMES)]
    engs = list(pool.map(_game, jobs))
    m = league_metrics(accumulate(engs[:games]))
    m["stars_pace"] = league_metrics(accumulate(engs[games:]))["pace_poss_per_48"]
    m.update(tracking(engs[:games]))
    loss = sum(((_get(m, p) - t) / tol) ** 2 for p, t, tol in GOALS)
    loss += ((m["stars_pace"] - STAR_PACE[0]) / STAR_PACE[1]) ** 2
    loss += sum(((m[k] - t) / tol) ** 2 for k, (t, tol) in TRACKING.items())
    return loss, m


def current_params() -> dict:
    out = {"resolve": {}, "tendencies": {}}
    for g, n, *_ in SPACE:
        out[g][n] = float((resolve.CAL if g == "resolve" else team.TEND)[n])
    return out


def summary(m) -> str:
    loc = m["shot_locations"]
    return (f"ppp {m['points_per_possession']:.3f} pace {m['pace_poss_per_48']:.0f} tov {m['tov_pct']:.3f} "
            f"stars {m['stars_pace']:.0f} pass {m['passes_per_poss']:.2f} drv {m['drives_per_game']:.0f} "
            f"top {m['top_fga_share']:.2f} orb {m['orb_pct']:.3f} ftr {m['ft_rate']:.3f} 3p% {m['fg3_pct']:.3f} ast {m['ast_pct_of_fgm']:.2f} | "
            + " ".join(f"{z[:5]} {v['freq']:.2f}/{v['fg_pct']:.2f}" for z, v in loc.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out_path = resolve.TUNED_PATH

    best = current_params()
    steps = {(g, n): s for g, n, s, *_ in SPACE}
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        best_loss, m = evaluate(pool, best, args.games)
        print(f"start loss {best_loss:.1f}  {summary(m)}", flush=True)
        while time.time() - t0 < args.minutes * 60:
            improved = False
            for g, n, _, lo, hi in SPACE:
                if time.time() - t0 > args.minutes * 60:
                    break
                for sign in (1, -1):
                    trial = {k: dict(v) for k, v in best.items()}
                    trial[g][n] = round(min(hi, max(lo, best[g][n] + sign * steps[(g, n)])), 4)
                    if trial[g][n] == best[g][n]:
                        continue
                    loss, m = evaluate(pool, trial, args.games)
                    if loss < best_loss:
                        best, best_loss, improved = trial, loss, True
                        print(f"{g}.{n} -> {trial[g][n]}  loss {loss:.1f}  {summary(m)}", flush=True)
                        out_path.write_text(yaml.safe_dump(
                            {"note": "written by scripts/tune.py", "loss": round(best_loss, 2), **best},
                            sort_keys=False))
                        break
            if not improved:
                for k in steps:
                    steps[k] /= 2
                print("no improvement in a sweep; halving steps", flush=True)
    print("best loss", round(best_loss, 2))


if __name__ == "__main__":
    main()
