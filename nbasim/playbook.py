"""Offensive plays written as data (data/plays/*.yaml) and the logic that fills roles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .players import Player

PLAYS_DIR = Path(__file__).resolve().parent.parent / "data" / "plays"
SLOTS = ["PG", "SG", "SF", "PF", "C"]


@dataclass
class Phase:
    id: str
    max_time: float
    until: str | None
    roles: dict[str, dict]
    pass_to: str | None = None


@dataclass
class Play:
    name: str
    label: str
    roles: dict[str, object]
    phases: list[Phase] = field(default_factory=list)


def load_play(path: Path) -> Play:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    phases = [Phase(id=p["id"], max_time=float(p.get("max_time", 2.0)), until=p.get("until"),
                    roles=p.get("roles", {}) or {}, pass_to=p.get("pass_to")) for p in raw["phases"]]
    return Play(name=raw["name"], label=raw.get("label", raw["name"]), roles=raw["roles"], phases=phases)


def load_playbook(directory: Path = PLAYS_DIR) -> dict[str, Play]:
    return {p.name: p for p in (load_play(f) for f in sorted(directory.glob("*.yaml")))}


def _best(cands: list[int], players: list[Player], key) -> int:
    return max(cands, key=lambda k: key(players[k]))


def assign_roles(play: Play, players: list[Player], holder_slot: int | None) -> dict[str, int]:
    """Map role name -> lineup slot (0..4)."""
    free = list(range(5))
    out: dict[str, int] = {}

    def take(slot: int, role: str) -> None:
        out[role] = slot
        free.remove(slot)

    # special selectors first, then position lists in declaration order
    for role, want in play.roles.items():
        if not isinstance(want, str):
            continue
        if want == "handler":
            slot = holder_slot if holder_slot in free else _best(free, players, lambda p: p.playmaking.ball_handle)
        elif want == "shooter":
            slot = _best(free, players, lambda p: p.outside_scoring.three_point)
        elif want == "post":
            slot = _best(free, players, lambda p: p.inside_scoring.post_control + p.height_in / 2)
        elif want == "scorer":
            slot = _best(free, players, lambda p: p.overall + p.playmaking.ball_handle / 10)
        else:
            continue
        take(slot, role)
    for role, want in play.roles.items():
        if role in out:
            continue
        prefs = want if isinstance(want, list) else [want]
        slot = next((s for pos in prefs for s in free if SLOTS[s] == pos), None)
        take(slot if slot is not None else free[0], role)
    return out


def play_fit(play: Play, players: list[Player]) -> float:
    """How much a lineup suits a play (used to weight play calls)."""
    handle = max(p.playmaking.ball_handle for p in players)
    bigs = [players[3], players[4]]
    shoot = sorted((p.outside_scoring.three_point for p in players), reverse=True)
    post = max(p.inside_scoring.post_control for p in players)
    star = max(p.overall for p in players)
    roll = max(max(b.inside_scoring.standing_dunk, b.outside_scoring.three_point) for b in bigs)
    return {
        "high_pnr": 1.0 + (handle - 80) / 20 + (roll - 80) / 30,
        "horns": 0.7 + (handle - 80) / 25,
        "spain_pnr": 0.5 + (shoot[1] - 80) / 25,
        "floppy": 0.4 + (shoot[0] - 85) / 15,
        "post_up": 0.2 + max(0.0, (post - 80) / 15),
        "iso": 0.3 + max(0.0, (star - 88) / 10),
    }.get(play.name, 0.5)
