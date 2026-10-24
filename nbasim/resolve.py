"""Probability models that turn 2K ratings and on-court geometry into outcomes.

Every model is a logistic or a hazard rate with a few named constants in
`CAL`. The constants are tuned by `scripts/calibrate.py` so that two
league-average lineups reproduce the 2024-25 NBA league numbers in
`data/league/nba_2024_25.yaml`. Ratings enter relative to REF, the rating a
league-average rotation player has in that skill.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from . import court
from .players import Player

REF = 70.0  # rating treated as league average


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rel(r: float) -> float:
    """Rating relative to league average, in units of 10 rating points."""
    return (r - REF) / 10.0


CAL: dict[str, float] = {
    # base make probability for an average shooter vs an average contest
    "base_rim": 0.740,
    "base_short_mid": 0.500,
    "base_long_mid": 0.440,
    "base_corner3": 0.425,
    "base_above3": 0.390,
    "skill_per10": 0.11,          # logit change per +10 rating in the zone's skill
    "contest_jumper": 0.55,       # logit penalty for a defender in your face on a jumper
    "contest_inside": 0.85,       # same at the rim / short mid
    "contest_range": 6.0,         # ft at which contest fades to zero
    "def_skill_per10": 0.10,
    "height_per_in": 0.025,       # defender taller than shooter, per inch
    "off_dribble": 0.18,          # jumpers off the dribble vs catch-and-shoot
    "consistency_sd": 0.20,       # logit noise for a shooter with average consistency
    # blocks (per contested rim/short-mid attempt)
    "block_base": 0.160,
    "block_per10": 0.35,
    "block_jumper": 0.04,
    # shooting fouls, probability per attempt by zone
    "foul_rim": 0.300,
    "foul_short_mid": 0.110,
    "foul_long_mid": 0.030,
    "foul_three": 0.018,
    "draw_foul_per10": 0.30,      # multiplicative, per +10 draw_foul
    # passing
    "steal_base": 0.15,           # per defender close to the ball path
    "steal_per10": 0.35,
    "steal_reach": 2.5,
    "bad_pass_oob": 0.0110,       # per pass, errant out of bounds
    # ball handling hazards, per second
    "strip_rate": 0.036,          # holder pressured by a defender within 3 ft
    "strip_per10": 0.40,
    "charge_rate": 0.08,          # per second driving into a set defender
    "travel_rate": 0.0009,        # per second holding the ball
    "reach_foul_rate": 0.030,     # non-shooting defensive foul, per pressured second
    "blowby_bias": -0.4,          # logit of beating an equal defender off the dribble
    # rebounds
    "defense_reb_bias": 3.0,
    "reb_skill_per10": 0.35,
    "ft_reb_def_bias": 6.0,
}


TUNED_PATH = Path(__file__).resolve().parent.parent / "data" / "league" / "tuned_params.yaml"


def load_tuned(path: Path = TUNED_PATH) -> dict:
    """Constants found by scripts/tune.py, applied over the defaults above."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


CAL.update(load_tuned().get("resolve", {}))


def ft_prob(p: Player) -> float:
    """2K's free throw rating tracks real FT% closely."""
    return min(0.95, max(0.40, p.outside_scoring.free_throw / 100.0))


def zone_skill(p: Player, zone: str, catch_near_rim: bool = False) -> float:
    o, i = p.outside_scoring, p.inside_scoring
    if zone == "rim":
        finish = max(i.layup, i.standing_dunk if catch_near_rim else i.driving_dunk)
        return 0.5 * o.close_shot + 0.5 * finish
    if zone == "short_mid":
        return 0.6 * o.close_shot + 0.4 * max(i.post_hook, i.post_fade, o.mid_range)
    if zone == "long_mid":
        return o.mid_range
    return o.three_point


def make_prob(
    shooter: Player,
    local,
    defender: Player | None,
    def_dist: float,
    off_dribble: bool,
    catch_near_rim: bool = False,
) -> tuple[float, str]:
    """Make probability (before blocks/fouls) and zone for a shot from `local`."""
    zone = court.shot_zone(local)
    x = logit(CAL[f"base_{zone}"]) + CAL["skill_per10"] * rel(zone_skill(shooter, zone, catch_near_rim))
    jumper = zone in ("long_mid", "corner3", "above3")
    if defender is not None and def_dist < CAL["contest_range"]:
        closeness = 1.0 - def_dist / CAL["contest_range"]
        d = defender.defense
        dskill = d.perimeter_defense if jumper else 0.5 * (d.interior_defense + d.block)
        size = defender.height_in - shooter.height_in + 0.5 * (defender.wingspan_in - defender.height_in)
        pen = CAL["contest_jumper" if jumper else "contest_inside"]
        pen *= 1.0 + CAL["def_skill_per10"] * rel(dskill) + CAL["height_per_in"] * size
        x -= max(0.0, pen) * closeness
    if jumper and off_dribble:
        x -= CAL["off_dribble"]
    # distance decay past the arc (deep threes)
    if zone == "above3":
        extra = court.dist_to_basket(local) - court.THREE_ARC_RADIUS - 1.5
        if extra > 0:
            x -= 0.06 * extra
    return sigmoid(x), zone


def shot_noise_sd(p: Player) -> float:
    return CAL["consistency_sd"] * (1.0 - 0.5 * rel(p.outside_scoring.offensive_consistency) / 3.0)


def block_prob(shooter: Player, defender: Player, def_dist: float, zone: str) -> float:
    d = defender
    if zone not in ("rim", "short_mid"):
        # jumpers get blocked only by a close, long contest (about 1-2% of NBA threes)
        if def_dist > 3.0:
            return 0.0
        size = (d.height_in + 0.5 * (d.wingspan_in - d.height_in)) - shooter.height_in
        x = logit(CAL["block_jumper"]) + CAL["block_per10"] * rel(d.defense.block) + 0.08 * size
        return sigmoid(x - 0.8 * def_dist / 3.0)
    if def_dist > 4.5:
        return 0.0
    size = (d.height_in + 0.5 * (d.wingspan_in - d.height_in)) - shooter.height_in
    x = logit(CAL["block_base"]) + CAL["block_per10"] * rel(d.defense.block)
    x += 0.15 * rel(d.athleticism.vertical) + 0.06 * size
    x -= 0.6 * def_dist / 4.5
    return sigmoid(x)


def shooting_foul_prob(shooter: Player, defender: Player | None, def_dist: float, zone: str) -> float:
    if defender is None or def_dist > 6.0:
        return 0.0
    key = {"rim": "foul_rim", "short_mid": "foul_short_mid", "long_mid": "foul_long_mid"}.get(zone, "foul_three")
    p = CAL[key] * (1.0 + 0.6 * (1.0 - def_dist / 6.0))
    p *= math.exp(CAL["draw_foul_per10"] * rel(shooter.inside_scoring.draw_foul))
    p *= math.exp(-0.15 * rel(defender.defense.defensive_consistency))
    return min(0.6, p)


def steal_prob(passer: Player, defender: Player, dist_to_ball: float) -> float:
    d = defender.defense
    skill = 0.5 * (d.steal + d.pass_perception) - passer.playmaking.pass_accuracy
    x = logit(CAL["steal_base"]) + CAL["steal_per10"] * skill / 10.0
    x -= 1.5 * dist_to_ball / CAL["steal_reach"]
    return sigmoid(x)


def strip_rate(holder: Player, defender: Player) -> float:
    gap = defender.defense.steal - holder.playmaking.ball_handle
    return CAL["strip_rate"] * math.exp(CAL["strip_per10"] * gap / 10.0)


def charge_rate(holder: Player, defender: Player) -> float:
    return CAL["charge_rate"] * math.exp(0.25 * rel(defender.defense.defensive_consistency))


def reach_foul_rate(defender: Player) -> float:
    return CAL["reach_foul_rate"] * math.exp(-0.10 * rel(defender.defense.defensive_consistency))


def rebound_skill(p: Player, offensive: bool) -> float:
    r = p.rebounding.offensive_rebound if offensive else p.rebounding.defensive_rebound
    size = (p.height_in - 78) / 4.0
    return math.exp(CAL["reb_skill_per10"] * rel(r) + 0.25 * size + 0.1 * rel(p.athleticism.hustle))
