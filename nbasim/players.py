"""Player model using NBA 2K attribute groups (ratings 25-99)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field

Rating = Annotated[int, Field(ge=25, le=99)]


class _Group(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InsideScoring(_Group):
    close_shot: Rating
    driving_layup: Rating
    driving_dunk: Rating
    standing_dunk: Rating
    post_control: Rating


class OutsideScoring(_Group):
    mid_range: Rating
    three_point: Rating
    free_throw: Rating
    shot_iq: Rating
    offensive_consistency: Rating


class Playmaking(_Group):
    pass_accuracy: Rating
    ball_handle: Rating
    speed_with_ball: Rating
    pass_iq: Rating
    pass_vision: Rating


class Defense(_Group):
    interior_defense: Rating
    perimeter_defense: Rating
    steal: Rating
    block: Rating
    help_defense_iq: Rating
    pass_perception: Rating
    defensive_consistency: Rating


class Rebounding(_Group):
    offensive_rebound: Rating
    defensive_rebound: Rating


class Athleticism(_Group):
    speed: Rating
    acceleration: Rating
    strength: Rating
    vertical: Rating
    stamina: Rating
    hustle: Rating


class Player(_Group):
    id: str
    name: str
    pos: str
    height_in: int
    wingspan_in: int
    weight_lb: int
    inside_scoring: InsideScoring
    outside_scoring: OutsideScoring
    playmaking: Playmaking
    defense: Defense
    rebounding: Rebounding
    athleticism: Athleticism


class Roster(BaseModel):
    source: str
    players: list[Player]

    def by_id(self) -> dict[str, Player]:
        return {p.id: p for p in self.players}


class Lineup(BaseModel):
    name: str
    players: list[Player] = Field(min_length=5, max_length=5)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_roster(path: str | Path = DATA_DIR / "players" / "nba2k26.yaml") -> Roster:
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
