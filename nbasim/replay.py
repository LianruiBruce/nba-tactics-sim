"""Replay log: the only interface between the simulator and the viewer."""

from __future__ import annotations

import json
from pathlib import Path

from .engine import DT, N_OFF, Engine


def _r(x: float) -> float:
    return round(float(x), 2)


class Recorder:
    def __init__(self, eng: Engine, offense_name: str, defense_name: str, seed: int | None):
        self.eng = eng
        self.meta = {
            "seed": seed,
            "dt": DT,
            "offense": offense_name,
            "defense": defense_name,
            "players": [
                {"idx": i, "id": p.id, "name": p.name, "pos": p.pos,
                 "team": "offense" if i < N_OFF else "defense"}
                for i, p in enumerate(eng.players)
            ],
        }
        self.frames: list[dict] = []
        self.frame()

    def frame(self) -> None:
        e = self.eng
        self.frames.append({
            "t": _r(e.t),
            "gc": _r(e.game_clock),
            "sc": _r(e.shot_clock),
            "ball": [_r(e.ball[0]), _r(e.ball[1]), _r(e.ball_height())],
            "holder": e.holder,
            "p": [[_r(x), _r(y)] for x, y in e.pos],
        })

    def to_dict(self) -> dict:
        e = self.eng
        return {
            "meta": self.meta,
            "outcome": e.outcome,
            "points": e.points,
            "frames": self.frames,
            "events": [ev.to_dict() for ev in e.events],
        }


def save(possessions: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "possessions": possessions}, f, ensure_ascii=False,
                  separators=(",", ":"))
