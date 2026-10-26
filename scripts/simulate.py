"""Play a game between two lineups with the scripted team AI and save a replay.

  python scripts/simulate.py                         # one quarter, stars_a vs stars_b -> replays/sample.json
  python scripts/simulate.py --quarters 4 --seed 7   # a full game
  python scripts/simulate.py --scheme-b switch        # team B switches every screen
  python scripts/simulate.py --real --a "Gilgeous-Alexander, Jalen Williams, Dort, Holmgren, Hartenstein" \
      --b "Brunson, Bridges, Hart, Anunoby, Towns"    # real players, skills from play-by-play

Open viewer/index.html to watch the replay.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbasim.game import play_game  # noqa: E402
from nbasim.players import load_lineups  # noqa: E402
from nbasim.replay import save  # noqa: E402


def box_score(eng, team: int, name: str) -> str:
    s = eng.stats
    rows = [f"{name}  {eng.score[team]}",
            f"  {'player':22s} PTS  FG     3P    FT    REB AST STL BLK TOV PF"]
    for i in range(team * 5, team * 5 + 5):
        rows.append(
            f"  {eng.players[i].name:22s} {s['pts'][i]:3d}  {s['fgm'][i]:2d}-{s['fga'][i]:<2d}  "
            f"{s['fg3m'][i]:2d}-{s['fg3a'][i]:<2d} {s['ftm'][i]:2d}-{s['fta'][i]:<2d} "
            f"{s['orb'][i] + s['drb'][i]:3d} {s['ast'][i]:3d} {s['stl'][i]:3d} {s['blk'][i]:3d} "
            f"{s['tov'][i]:3d} {s['pf'][i]:2d}")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="stars_a")
    ap.add_argument("--b", default="stars_b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quarters", type=int, default=1)
    ap.add_argument("--scheme-a", default="drop", choices=["drop", "switch"])
    ap.add_argument("--scheme-b", default="drop", choices=["drop", "switch"])
    ap.add_argument("--real", action="store_true", help="--a/--b are comma-separated real player names")
    ap.add_argument("--out", default="replays/sample.json")
    args = ap.parse_args()

    if args.real:
        from nbasim.realplayers import real_lineup
        a, b = real_lineup(args.a), real_lineup(args.b)
    else:
        lineups = load_lineups()
        a, b = lineups[args.a], lineups[args.b]
    res = play_game(a, b, seed=args.seed, quarters=args.quarters,
                    scheme_a=args.scheme_a, scheme_b=args.scheme_b, record=True)
    eng = res.engine
    print(box_score(eng, 0, a.name))
    print(box_score(eng, 1, b.name))
    print(f"possessions {eng.possessions}, frames {len(res.replay['frames'])}")
    save(res.replay, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
