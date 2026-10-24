"""Replay log: the only interface between the simulator and the viewer."""

from __future__ import annotations

import json
from pathlib import Path

from .engine import DT, Engine


def _r(x: float) -> float:
    return round(float(x), 2)


class Recorder:
    def __init__(self, eng: Engine, name_a: str, name_b: str, seed: int | None):
        self.eng = eng
        self.meta = {
            "seed": seed,
            "dt": DT,
            "court": "full",
            "teams": [name_a, name_b],
            "players": [
                {"idx": i, "id": p.id, "name": p.name, "pos": p.pos, "team": 0 if i < 5 else 1,
                 "overall": p.overall}
                for i, p in enumerate(eng.players)
            ],
        }
        self.frames: list[dict] = []
        self.frame()

    def frame(self) -> None:
        e = self.eng
        self.frames.append({
            "t": _r(e.t),
            "q": e.quarter,
            "gc": _r(e.game_clock),
            "sc": _r(e.shot_clock),
            "s": list(e.score),
            "off": e.off,
            "st": e.status,
            "ball": [_r(e.ball[0]), _r(e.ball[1]), _r(e.ball_z)],
            "h": e.holder,
            "p": [[_r(x), _r(y)] for x, y in e.pos],
            "lab": [e.labels[0], e.labels[1]],
        })

    def to_dict(self) -> dict:
        e = self.eng
        return {
            "meta": self.meta,
            "score": list(e.score),
            "frames": self.frames,
            "events": [ev.to_dict() for ev in e.events],
            "box": {k: v.tolist() for k, v in e.stats.items()},
        }


def save(replay: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 2, **replay}, f, ensure_ascii=False, separators=(",", ":"))
