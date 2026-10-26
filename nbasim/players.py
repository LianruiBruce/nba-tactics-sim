"""Player model using NBA 2K27 attribute groups (ratings 25-99)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field

Rating = Annotated[int, Field(ge=25, le=99)]


class _Group(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutsideScoring(_Group):
    three_point: Rating
    mid_range: Rating
    close_shot: Rating
    free_throw: Rating
    offensive_consistency: Rating
    shot_iq: Rating


class Athleticism(_Group):
    speed: Rating
    strength: Rating
    agility: Rating
    vertical: Rating
    hustle: Rating
    stamina: Rating
    overall_durability: Rating


class InsideScoring(_Group):
    layup: Rating
    driving_dunk: Rating
    standing_dunk: Rating
    post_hook: Rating
    post_fade: Rating
    post_control: Rating
    draw_foul: Rating
    hands: Rating


class Playmaking(_Group):
    ball_handle: Rating
    speed_with_ball: Rating
    pass_accuracy: Rating
    pass_vision: Rating
    pass_iq: Rating


class Defense(_Group):
    block: Rating
    steal: Rating
    pass_perception: Rating
    interior_defense: Rating
    perimeter_defense: Rating
    help_defense_iq: Rating
    defensive_consistency: Rating


class Rebounding(_Group):
    defensive_rebound: Rating
    offensive_rebound: Rating


class Player(_Group):
    """NBA 2K27 attribute set as listed on 2kratings.com."""

    id: str
    name: str
    team: str
    pos: str
    overall: int
    url: str
    height_in: int
    wingspan_in: int
    weight_lb: int
    inside_scoring: InsideScoring
    outside_scoring: OutsideScoring
    playmaking: Playmaking
    defense: Defense
    rebounding: Rebounding
    athleticism: Athleticism
    # set when skills come from real play-by-play (nbasim.realplayers)
    nba_id: int | None = None
    usage: float | None = None     # share weight of his team's shots, from real data
    zone_pref: dict[str, float] | None = None  # log(his shot mix / league mix) by zone, from real data


class Roster(BaseModel):
    source: str
    players: list[Player]

    def by_id(self) -> dict[str, Player]:
        return {p.id: p for p in self.players}


class Lineup(BaseModel):
    name: str
    players: list[Player] = Field(min_length=5, max_length=5)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_roster(path: str | Path = DATA_DIR / "players" / "nba2k27.yaml") -> Roster:
    with open(path, encoding="utf-8") as f:
        return Roster.model_validate(yaml.safe_load(f))


def load_lineups(
    path: str | Path = DATA_DIR / "lineups.yaml", roster: Roster | None = None
) -> dict[str, Lineup]:
    roster = roster or load_roster()
    ids = roster.by_id()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {
        key: Lineup(name=v["name"], players=[ids[pid] for pid in v["players"]])
        for key, v in raw.items()
    }
