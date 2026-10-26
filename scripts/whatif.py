"""Lineup what-if: how two five-man units do against each other, and who to swap in.

    python scripts/whatif.py --a "Gilgeous-Alexander, Jalen Williams, Dort, Holmgren, Hartenstein" \\
                             --b "Doncic, Reaves, LeBron, Hachimura, Ayton" --home a
    python scripts/whatif.py --a ... --b ... --try "Caruso, Wallace, Joe"   # test each bench swap

Numbers are per 100 possessions from nbasim.realmodel (fit by scripts/fit_model.py), averaged over
how possessions start in the league (after a miss, a make, a turnover...).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nbasim.data.names import find_player, player_name  # noqa: E402
from nbasim.realmodel import ZONES, RealModel  # noqa: E402

ZONE_CN = {"rim": "篮下", "short_mid": "短中距", "long_mid": "长两分", "corner3": "底角三分", "above3": "弧顶三分"}
# share of possessions by how they start (2025-26)
START_MIX = {"def_rebound": 0.34, "made_fg": 0.39, "made_ft": 0.10, "turnover": 0.14, "period_start": 0.02,
             "other": 0.01}


def players(s: str) -> list[int]:
    ids = [find_player(x) for x in s.split(",") if x.strip()]
    if len(ids) != 5 or len(set(ids)) != 5:
        raise SystemExit(f"need five different players, got {[player_name(p) for p in ids]}")
    return ids


def ppp(m: RealModel, off, de, home: bool) -> float:
    return sum(w * m.expected_points(off, de, home, c) for c, w in START_MIX.items())


def side_report(m: RealModel, off, de, home: bool) -> dict:
    b = m.breakdown(off, de, home, "def_rebound")
    out = {"每回合得分": ppp(m, off, de, home), "失误/机会": b["p_tov"], "罚球回合/机会": b["p_trip"]}
    for z in ZONES:
        out[f"{ZONE_CN[z]} 出手占比"] = b[f"share_{z}"]
    for z in ZONES:
        out[f"{ZONE_CN[z]} 命中率"] = b[f"fg_{z}"]
    out["前场篮板率（投丢后）"] = b["p_oreb_fga"] / max(sum(b[f"share_{z}"] * (1 - b[f"fg_{z}"]) for z in ZONES), 1e-9)
    shares = b["shooter_share"]
    for p, s in sorted(zip(off, shares), key=lambda x: -x[1]):
        out[f"出手占比 {player_name(p)}"] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="five players, comma separated")
    ap.add_argument("--b", required=True)
    ap.add_argument("--home", choices=["a", "b", "none"], default="none")
    ap.add_argument("--try", dest="bench", default="", help="candidates to swap into lineup A, comma separated")
    ap.add_argument("--model", default=str(ROOT / "data" / "models" / "realmodel.pkl"))
    args = ap.parse_args()

    m = RealModel.load(Path(args.model))
    A, B = players(args.a), players(args.b)
    ha, hb = args.home == "a", args.home == "b"
    ra, rb = side_report(m, A, B, ha), side_report(m, B, A, hb)
    net = 100 * (ra["每回合得分"] - rb["每回合得分"])
    print(f"A: {', '.join(player_name(p) for p in A)}")
    print(f"B: {', '.join(player_name(p) for p in B)}")
    print(f"模型训练截至 {getattr(m, 'trained_through', '?')}\n")
    keys = [k for k in ra if not k.startswith("出手占比 ")]
    table = pd.DataFrame({"A 进攻": [ra[k] for k in keys], "B 进攻": [rb[k] for k in keys]}, index=keys)
    print(table.round(3).to_string())
    print(f"\nA 对 B 的净效率：{net:+.1f} 分/100 回合")
    print("\nA 的出手分配：" + "，".join(f"{k[5:]} {v:.0%}" for k, v in ra.items() if k.startswith("出手占比 ")))
    print("B 的出手分配：" + "，".join(f"{k[5:]} {v:.0%}" for k, v in rb.items() if k.startswith("出手占比 ")))

    if args.bench:
        print("\n换人推演（把 A 的某人换成候选人后的净效率变化）：")
        rows = []
        for cand in [find_player(x) for x in args.bench.split(",") if x.strip()]:
            for i, out_p in enumerate(A):
                if cand in A:
                    continue
                A2 = A[:i] + [cand] + A[i + 1:]
                n2 = 100 * (ppp(m, A2, B, ha) - ppp(m, B, A2, hb))
                rows.append((player_name(cand), player_name(out_p), n2 - net))
        df = pd.DataFrame(rows, columns=["换上", "换下", "净效率变化"]).sort_values("净效率变化", ascending=False)
        print(df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
