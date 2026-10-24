"""Hand-written baseline policy: a simple high pick-and-roll vs man defense.

This is a placeholder until the playbook interpreter lands in phase 1. It
reads the engine state directly (privileged), which is fine for a baseline
opponent.
"""

from __future__ import annotations

import numpy as np

from .. import court
from ..engine import N_OFF, Engine
from ..env import AGENT_NAMES, START_SPOTS
from ..macro import DRIVE, GUARD, MOVE_TO, PASS, SCREEN, SHOOT, STAY, opponents, teammates

C = 4  # lineup slot that sets the screen


def _region(name: str) -> int:
    return MOVE_TO + court.REGION_NAMES.index(name)


def _nearest_def_dist(eng: Engine, i: int, at=None) -> float:
    at = eng.pos[i] if at is None else at
    return float(np.min(np.linalg.norm(eng.pos[opponents(i)] - at, axis=1)))


def _lane_open(eng: Engine, a: int, b: int, width: float = 2.0) -> bool:
    p, q = eng.pos[a], eng.pos[b]
    seg = q - p
    L2 = float(seg @ seg) or 1e-9
    for j in opponents(a):
        s = np.clip((eng.pos[j] - p) @ seg / L2, 0.0, 1.0)
        if np.linalg.norm(eng.pos[j] - (p + s * seg)) < width:
            return False
    return True


class ScriptedPolicy:
    def __init__(self):
        self.last_pass_t = -10.0
        self.last_passer = -1  # no immediate return passes

    def __call__(self, eng: Engine) -> dict[str, int]:
        acts = {}
        for i in range(N_OFF):
            acts[AGENT_NAMES[i]] = self._offense(eng, i)
        for k in range(N_OFF):
            acts[AGENT_NAMES[N_OFF + k]] = GUARD + k  # man-to-man by lineup slot
        return acts

    def _offense(self, eng: Engine, i: int) -> int:
        sc = eng.shot_clock
        if eng.holder == i:
            return self._ball(eng, i)
        # center: screen for the ball handler early, then roll to the rim
        if i == C and eng.holder is not None and eng.holder != C:
            if sc > 17.0:
                return SCREEN + teammates(i).index(eng.holder)
            return _region("rim")
        return _region(START_SPOTS[i])

    def _ball(self, eng: Engine, i: int) -> int:
        sc = eng.shot_clock
        p, _, three = eng.shot_make_prob(i)
        my_ep = p * (3 if three else 2)
        if sc < 2.5 or (sc < 16.0 and my_ep >= 1.05):
            return SHOOT

        best, best_ep = None, my_ep
        waiting_for_screen = sc > 17.0
        if not waiting_for_screen and eng.t - self.last_pass_t > 1.5:
            for k, j in enumerate(teammates(i)):
                if j == self.last_passer or not _lane_open(eng, i, j):
                    continue
                pj, _, t3 = eng.shot_make_prob(j)
                ep = pj * (3 if t3 else 2)
                if ep > best_ep + 0.05:
                    best, best_ep = k, ep
        if best is not None:
            self.last_pass_t = eng.t
            self.last_passer = i
            return PASS + best

        if court.dist_to_basket(eng.pos[i]) < 6.0:
            return SHOOT
        if sc < 17.0 and _nearest_def_dist(eng, i) > 3.5:
            return DRIVE
        return STAY if sc > 17.0 else DRIVE
