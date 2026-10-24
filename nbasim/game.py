"""Run whole games and summarize them like a box score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agents.team import TeamBrain
from .engine import Engine
from .players import Lineup
from .replay import Recorder


@dataclass
class GameResult:
    engine: Engine
    replay: dict | None


def play_game(a: Lineup, b: Lineup, seed: int = 0, quarters: int = 4, quarter_len: float = 720.0,
              scheme_a: str = "drop", scheme_b: str = "drop", record: bool = False,
              max_ticks: int = 200_000) -> GameResult:
    eng = Engine(a.players, b.players, seed=seed, quarters=quarters, quarter_len=quarter_len)
    brains = (TeamBrain(scheme_a, seed=seed * 2 + 1), TeamBrain(scheme_b, seed=seed * 2 + 2))
    rec = Recorder(eng, a.name, b.name, seed) if record else None
    for _ in range(max_ticks):
        if eng.done:
            break
        eng.step(brains)
        if rec:
            rec.frame()
    return GameResult(eng, rec.to_dict() if rec else None)


def team_stats(eng: Engine, team: int) -> dict:
    idx = slice(0, 5) if team == 0 else slice(5, 10)
    s = {k: int(v[idx].sum()) for k, v in eng.stats.items()}
    opp = {k: int(v[slice(5, 10) if team == 0 else slice(0, 5)].sum()) for k, v in eng.stats.items()}
    s["poss"] = eng.possessions[team]
    s["opp_drb"] = opp["drb"]
    s["zone_att"] = {z: v[team] for z, v in eng.zone_att.items()}
    s["zone_made"] = {z: v[team] for z, v in eng.zone_made.items()}
    return s


def league_metrics(totals: dict) -> dict:
    """Same derived numbers as data/league/nba_2024_25.yaml, from summed team stats."""
    t = totals
    fga = max(t["fga"], 1)
    zones = {z: {"freq": t["zone_att"][z] / fga,
                 "fg_pct": t["zone_made"][z] / max(t["zone_att"][z], 1)} for z in t["zone_att"]}
    return {
        "points_per_possession": t["pts"] / max(t["poss"], 1),
        "tov_pct": t["tov"] / max(t["poss"], 1),
        "orb_pct": t["orb"] / max(t["orb"] + t["opp_drb"], 1),
        "ft_rate": t["fta"] / fga,
        "ft_pct": t["ftm"] / max(t["fta"], 1),
        "fg_pct": t["fgm"] / fga,
        "fg3_rate": t["fg3a"] / fga,
        "fg3_pct": t["fg3m"] / max(t["fg3a"], 1),
        "stl_share_of_tov": t["stl_opp"] / max(t["tov"], 1),
        "blk_pct_of_fga": t["blk_opp"] / fga,
        "ast_pct_of_fgm": t["ast"] / max(t["fgm"], 1),
        "pace_poss_per_48": t["poss"] / max(t["games"], 1) * (48.0 / t["minutes_per_game"]),
        "shot_locations": zones,
    }


def accumulate(results: list[Engine]) -> dict:
    keys = ("pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "orb", "drb", "ast", "stl", "blk", "tov", "pf")
    tot = {k: 0 for k in keys}
    tot.update(poss=0, opp_drb=0, stl_opp=0, blk_opp=0, games=0)
    tot["zone_att"] = {z: 0 for z in ("rim", "short_mid", "long_mid", "corner3", "above3")}
    tot["zone_made"] = dict(tot["zone_att"])
    minutes = []
    for eng in results:
        minutes.append((eng.quarter_len * min(eng.quarter, eng.quarters) +
                        max(0, eng.quarter - eng.quarters) * 300.0) / 60.0)
        for team in (0, 1):
            s = team_stats(eng, team)
            o = team_stats(eng, 1 - team)
            for k in keys:
                tot[k] += s[k]
            tot["poss"] += s["poss"]
            tot["opp_drb"] += o["drb"]
            tot["stl_opp"] += o["stl"]
            tot["blk_opp"] += o["blk"]
            tot["games"] += 1
            for z in tot["zone_att"]:
                tot["zone_att"][z] += s["zone_att"][z]
                tot["zone_made"][z] += s["zone_made"][z]
    tot["minutes_per_game"] = float(np.mean(minutes)) if minutes else 48.0
    return tot


def tracking(engs) -> dict:
    """Play-type counts the box score does not have, from the event log."""
    poss = sum(sum(e.possessions) for e in engs)
    passes = sum(1 for e in engs for ev in e.events if ev.type == "pass")
    drives = sum(1 for e in engs for ev in e.events if ev.type == "drive")
    minutes = sum(e.quarter_len * min(e.quarter, e.quarters) for e in engs) / 60.0
    shares = []
    for e in engs:
        for t in (0, 1):
            f = e.stats["fga"][t * 5:t * 5 + 5]
            shares.append(f.max() / max(f.sum(), 1))
    return {"passes_per_poss": passes / max(poss, 1), "drives_per_game": drives / 2 * 48.0 / max(minutes, 1),
            "top_fga_share": float(sum(shares) / len(shares))}
