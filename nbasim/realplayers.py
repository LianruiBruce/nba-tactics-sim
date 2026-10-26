"""Engine players whose skills come from real play-by-play instead of 2K.

The spatial engine reads NBA 2K-style ratings, and `nbasim/resolve.py` turns a
rating into a log-odds shift (for example +10 three-point rating = +0.11 logit
on a three). `nbasim/realmodel.py` measures each player's log-odds shifts
directly from real games. This module inverts the engine's formulas, so a
measured shift becomes the rating that reproduces it inside the engine:

    rating = position template + 10 * measured_logit_shift / engine_logit_per_10

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
    "draw_foul": resolve.CAL["draw_foul_per10"],
    "ball_handle": resolve.CAL["strip_per10"],
    "steal": resolve.CAL["steal_per10"],
    "rebound": resolve.CAL["reb_skill_per10"],
    # a defender's skill scales the contest penalty: about pen * closeness * def_skill_per10
    "perimeter": resolve.CAL["contest_jumper"] * 0.5 * resolve.CAL["def_skill_per10"],
    "interior": resolve.CAL["contest_inside"] * 0.5 * resolve.CAL["def_skill_per10"],
}
# the real model spreads a team effect over five players; one player's own share in the engine is larger
TEAM_TO_PLAYER = {"rebound": 5.0, "tov": 2.5, "guard": 1.0, "trip": 1.5}


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

    def shift(group: str, attr: str, logit: float, per10: float) -> int:
        # the model measures shifts against the league-average player, which the engine rates REF
        return _clip(resolve.REF + 10.0 * logit / per10)

    # measured log-odds shifts (the model's own units)
    k = TEAM_TO_PLAYER
    d_three = model.make.player("shoot_three", pid)
    d_mid = model.make.player("shoot_mid", pid)
    d_rim = model.make.player("shoot_rim", pid)
    g_three = -model.make.player("guard_three", pid) * k["guard"]
    g_rim = -model.make.player("guard_rim", pid) * k["guard"]
    tov_off = model.tov.player("off", pid) * k["tov"]
    tov_def = model.tov.player("def", pid) * k["tov"]
    trip = model.trip.player("off", pid) * k["trip"]
    # the engine already gives taller players more rebounds; take that part out of the measured edge
    size = 0.25 * (base.height_in - 78) / 4.0
    oreb = model.oreb.player("off", pid) * k["rebound"] - size
    dreb = -model.oreb.player("def", pid) * k["rebound"] - size

    o = base.outside_scoring.model_copy(update={
        "three_point": shift("outside_scoring", "three_point", d_three, PER10["shoot"]),
        "mid_range": shift("outside_scoring", "mid_range", d_mid, PER10["shoot"]),
        "close_shot": shift("outside_scoring", "close_shot", d_rim, PER10["shoot"]),
        "free_throw": _clip(100 * card["ft_pct"]),
    })
    i = base.inside_scoring.model_copy(update={
        "layup": shift("inside_scoring", "layup", d_rim, PER10["shoot"]),
        "driving_dunk": shift("inside_scoring", "driving_dunk", d_rim, PER10["shoot"]),
        "draw_foul": shift("inside_scoring", "draw_foul", trip, PER10["draw_foul"]),
    })
    pm = base.playmaking.model_copy(update={
        "ball_handle": shift("playmaking", "ball_handle", -tov_off, PER10["ball_handle"]),
        "pass_accuracy": shift("playmaking", "pass_accuracy", -tov_off, PER10["steal"]),
    })
    df = base.defense.model_copy(update={
        "perimeter_defense": shift("defense", "perimeter_defense", g_three, PER10["perimeter"]),
        "interior_defense": shift("defense", "interior_defense", g_rim, PER10["interior"]),
        "steal": shift("defense", "steal", tov_def, PER10["steal"]),
    })
    rb = base.rebounding.model_copy(update={
        "offensive_rebound": shift("rebounding", "offensive_rebound", oreb, PER10["rebound"]),
        "defensive_rebound": shift("rebounding", "defensive_rebound", dreb, PER10["rebound"]),
    })
    return base.model_copy(update={
        "outside_scoring": o, "inside_scoring": i, "playmaking": pm, "defense": df, "rebounding": rb,
        "nba_id": pid, "usage": float(card["fga_rate"]),
        "zone_pref": {z: float(np.log(m / lm)) for z, m, lm in
                      zip(ZONES, model.zone_mix.get(pid, model.league_mix), model.league_mix)},
    })


def real_lineup(names: list[str] | str, label: str = "", model: RealModel | None = None) -> Lineup:
    if isinstance(names, str):
        names = [n for n in names.split(",") if n.strip()]
    players = [real_player(n.strip() if isinstance(n, str) else n, model) for n in names]
    return Lineup(name=label or " / ".join(p.name.split()[-1] for p in players), players=players)
