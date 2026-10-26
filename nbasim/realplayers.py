"""Engine players whose skills come from real play-by-play instead of 2K.

The spatial engine reads NBA 2K-style ratings, and `nbasim/resolve.py` turns a
rating into a log-odds shift (for example +10 three-point rating = +0.11 logit
on a three). `nbasim/realmodel.py` measures each player's log-odds shifts
directly from real games. This module inverts the engine's formulas, so a
measured shift becomes the rating that reproduces it inside the engine:

    rating = REF + 10 * measured_logit_shift / engine_logit_per_10

That covers a shooter's own make probability and rebounding. Effects that
depend on who else is on the floor (a defender lowering opponents' shooting, a
playmaker raising teammates', turnovers, fouls drawn) go to `Player.real`, and
the engine adds them up over all ten players, the way the model does.
Everything without a real-data counterpart (size, speed, post moves) comes from
the player's 2K entry when there is one, otherwise from the league-average
template for his position. The position comes from his real box score
(assists, rebounds) and shot profile.
"""
from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import resolve
from .data.names import NBA, find_player, player_name
from .players import DATA_DIR, Lineup, Player, load_roster
from .realmodel import ZONES, RealModel

MODEL_PATH = DATA_DIR / "models" / "realmodel.pkl"

# engine log-odds per +10 rating, from resolve.py, for each mapped skill
PER10 = {
    "shoot": resolve.CAL["skill_per10"],
    "rebound": resolve.CAL["reb_skill_per10"],
}
# the real model spreads a team effect over five players; one player's own share in the engine is
# larger. Tuned with scripts/engine_check.py so engine offensive rebounding keeps the model's spread.
TEAM_TO_PLAYER = {"rebound": 2.7}
# scale on the lineup terms the engine adds on top of ratings (Player.real), tuned the same way
REAL_GAIN = {"rim": 1.6, "mid": 1.0, "three": 0.7, "tov": 0.75, "trip": 0.7}


def _clip(x: float) -> int:
    return int(round(min(99, max(25, x))))


@lru_cache(maxsize=1)
def _model() -> RealModel:
    return RealModel.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _templates() -> dict[str, Player]:
    return {p.pos: p for p in load_roster(DATA_DIR / "players" / "calibration.yaml").players}


@lru_cache(maxsize=1)
def _roster_by_name() -> dict[str, Player]:
    return {p.name.lower(): p for p in load_roster().players}


@lru_cache(maxsize=1)
def _box_rates() -> dict[int, dict]:
    """Per player, per minute assists and rebounds over the two latest seasons of game logs."""
    seasons = sorted(p.name for p in NBA.iterdir() if p.is_dir())[-2:]
    tot: dict[int, dict] = {}
    for s in seasons:
        path = NBA / s / "stats" / "gamelog_player.json.gz"
        if not path.exists():
            continue
        rs = json.loads(gzip.decompress(path.read_bytes()))["resultSets"][0]
        h = rs["headers"]
        ip, im, ia, ir = (h.index(k) for k in ("PLAYER_ID", "MIN", "AST", "REB"))
        for r in rs["rowSet"]:
            t = tot.setdefault(r[ip], {"min": 0.0, "ast": 0.0, "reb": 0.0})
            t["min"] += r[im] or 0
            t["ast"] += r[ia] or 0
            t["reb"] += r[ir] or 0
    return {p: {"ast": t["ast"] / max(t["min"], 1), "reb": t["reb"] / max(t["min"], 1), "min": t["min"]}
            for p, t in tot.items()}


def position(pid: int, model: RealModel) -> str:
    box = _box_rates().get(pid, {"ast": 0.09, "reb": 0.17})
    mix = model.zone_mix.get(pid, model.league_mix)
    rim, three = mix[0], mix[3] + mix[4]
    if box["reb"] > 0.30 or (rim > 0.45 and three < 0.12):
        return "C"
    if box["ast"] > 0.17:
        return "PG"
    if box["reb"] > 0.22 or rim > 0.38:
        return "PF"
    if box["ast"] > 0.11:
        return "SG"
    return "SF"


def real_player(query: str | int, model: RealModel | None = None) -> Player:
    """An engine Player for a real NBA player, skills from real data."""
    model = model or _model()
    pid = query if isinstance(query, int) else find_player(query)
    name = player_name(pid)
    card = model.player_card(pid)
    base = _roster_by_name().get(name.lower())
    pos = base.pos if base else position(pid, model)
    if base is None:
        tmpl = _templates()[pos]
        base = tmpl.model_copy(update={"id": f"nba_{pid}", "name": name, "team": "", "url": ""})

    def shift(logit: float, per10: float) -> int:
        # the model measures shifts against the league-average player, which the engine rates REF
        return _clip(resolve.REF + 10.0 * logit / per10)

    # the shooter's own skill and his rebounding go through ratings (rating = REF + 10 * logit / per10);
    # effects on teammates and opponents, turnovers and fouls drawn go to Player.real, which the engine
    # sums over the ten players on the floor, so their ratings stay at league average
    d_three = model.make.player("shoot_three", pid)
    d_mid = model.make.player("shoot_mid", pid)
    d_rim = model.make.player("shoot_rim", pid)
    # the engine already gives taller players more rebounds; take that part out of the measured edge
    size = 0.25 * (base.height_in - 78) / 4.0
    oreb = model.oreb.player("off", pid) * TEAM_TO_PLAYER["rebound"] - size
    dreb = -model.oreb.player("def", pid) * TEAM_TO_PLAYER["rebound"] - size
    g = REAL_GAIN
    real = {f"{k}_{z}": g[z] * model.make.player(f"{k}_{z}", pid)
            for k in ("guard", "mate") for z in ("rim", "mid", "three")}
    real |= {f"tov_{s}": g["tov"] * model.tov.player(s, pid) for s in ("off", "def")}
    real |= {f"trip_{s}": g["trip"] * model.trip.player(s, pid) for s in ("off", "def")}
    avg = int(resolve.REF)

    o = base.outside_scoring.model_copy(update={
        "three_point": shift(d_three, PER10["shoot"]),
        "mid_range": shift(d_mid, PER10["shoot"]),
        "close_shot": shift(d_rim, PER10["shoot"]),
        "free_throw": _clip(100 * card["ft_pct"]),
    })
    i = base.inside_scoring.model_copy(update={
        "layup": shift(d_rim, PER10["shoot"]),
        "driving_dunk": shift(d_rim, PER10["shoot"]),
        "draw_foul": avg,
    })
    pm = base.playmaking.model_copy(update={"ball_handle": avg, "pass_accuracy": avg})
    df = base.defense.model_copy(update={"perimeter_defense": avg, "interior_defense": avg, "steal": avg})
    rb = base.rebounding.model_copy(update={
        "offensive_rebound": shift(oreb, PER10["rebound"]),
        "defensive_rebound": shift(dreb, PER10["rebound"]),
    })
    return base.model_copy(update={
        "outside_scoring": o, "inside_scoring": i, "playmaking": pm, "defense": df, "rebounding": rb,
        "nba_id": pid, "usage": float(card["fga_rate"]), "real": real,
        "zone_pref": {z: float(np.log(m / lm)) for z, m, lm in
                      zip(ZONES, model.zone_mix.get(pid, model.league_mix), model.league_mix)},
    })


def real_lineup(names: list[str] | str, label: str = "", model: RealModel | None = None) -> Lineup:
    if isinstance(names, str):
        names = [n for n in names.split(",") if n.strip()]
    players = [real_player(n.strip() if isinstance(n, str) else n, model) for n in names]
    return Lineup(name=label or " / ".join(p.name.split()[-1] for p in players), players=players)
