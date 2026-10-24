"""Run possessions with the scripted baseline.

  python scripts/simulate.py                 # one possession each way -> replays/sample.json
  python scripts/simulate.py --batch 1000    # batch stats, no replay
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nbasim.agents.scripted import ScriptedPolicy  # noqa: E402
from nbasim.env import HalfCourtEnv  # noqa: E402
from nbasim.players import load_lineups  # noqa: E402
from nbasim.replay import save  # noqa: E402


def run_possession(env: HalfCourtEnv, seed: int) -> dict:
    env.reset(seed=seed)
    policy = ScriptedPolicy()
    while env.agents:
        env.step(policy(env.engine))
    return {"outcome": env.engine.outcome, "points": env.engine.points,
            "events": env.engine.events, "replay": env.replay()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--a", default="mixed")
    ap.add_argument("--b", default="west")
    ap.add_argument("--out", default="replays/sample.json")
    ap.add_argument("--batch", type=int, default=0)
    args = ap.parse_args()

    lineups = load_lineups()
    a, b = lineups[args.a], lineups[args.b]

    if args.batch:
        for off, de in ((a, b), (b, a)):
            env = HalfCourtEnv(off, de)
            outcomes, zones, pts = Counter(), Counter(), 0
            for k in range(args.batch):
                r = run_possession(env, args.seed + k)
                outcomes[r["outcome"]] += 1
                pts += r["points"]
                for ev in r["events"]:
                    if ev.type == "shot":
                        zones[ev.data["zone"]] += 1
            print(f"{off.name} on offense vs {de.name}: "
                  f"{pts / args.batch:.3f} points per possession")
            print("  outcomes:", dict(outcomes.most_common()))
            print("  shot zones:", dict(zones.most_common()))
        return

    possessions = []
    for k, (off, de) in enumerate(((a, b), (b, a))):
        env = HalfCourtEnv(off, de, record=True)
        r = run_possession(env, args.seed + k)
        possessions.append(r["replay"])
        print(f"{off.name} on offense: {r['outcome']}, {r['points']} pts")
    save(possessions, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
