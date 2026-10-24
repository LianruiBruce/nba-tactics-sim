"""Macro actions: the discrete action space shared by scripted and RL agents.

A macro action is chosen every decision step (several engine ticks) and is
turned into engine requests by a hand-written low-level controller here.

Layout (same for every agent; illegal ones are masked):
  STAY
  MOVE_TO(region)          one per court.REGIONS entry
  PASS(teammate k)         k = 0..3, teammates in lineup order skipping self
  SHOOT
  DRIVE                    attack the rim with the ball
  SCREEN(teammate k)       set a screen on teammate k's nearest defender
  GUARD(opponent k)        k = 0..4, stay between that opponent and the basket
"""

from __future__ import annotations

import numpy as np

from . import court
from .engine import N_OFF, N_PLAYERS, Engine

N_REGIONS = len(court.REGION_NAMES)

STAY = 0
MOVE_TO = 1
PASS = MOVE_TO + N_REGIONS
SHOOT = PASS + 4
DRIVE = SHOOT + 1
SCREEN = DRIVE + 1
GUARD = SCREEN + 4
N_ACTIONS = GUARD + 5

DRIVE_SPOT = np.array([25.0, 7.0])
SCREEN_GAP = 2.6  # ft between screener and the defender being screened
GUARD_GAP_BALL = 3.0
GUARD_GAP_OFF_BALL = 5.0


def teammates(i: int) -> list[int]:
    base = 0 if i < N_OFF else N_OFF
    return [j for j in range(base, base + N_OFF) if j != i]


def opponents(i: int) -> list[int]:
    return list(range(N_OFF, N_PLAYERS)) if i < N_OFF else list(range(N_OFF))


def describe(a: int) -> str:
    if a == STAY:
        return "stay"
    if MOVE_TO <= a < PASS:
        return f"move_to:{court.REGION_NAMES[a - MOVE_TO]}"
    if PASS <= a < SHOOT:
        return f"pass:{a - PASS}"
    if a == SHOOT:
        return "shoot"
    if a == DRIVE:
        return "drive"
    if SCREEN <= a < GUARD:
        return f"screen:{a - SCREEN}"
    return f"guard:{a - GUARD}"


def action_mask(eng: Engine, i: int) -> np.ndarray:
    m = np.zeros(N_ACTIONS, dtype=np.int8)
    m[STAY] = 1
    m[MOVE_TO:PASS] = 1
    if i < N_OFF:
        if eng.holder == i:
            m[PASS:SHOOT] = 1
            m[SHOOT] = 1
            m[DRIVE] = 1
        else:
            m[SCREEN:GUARD] = 1
    else:
        m[GUARD:N_ACTIONS] = 1
    return m


class MacroController:
    """Holds each agent's current macro action and drives the engine with it."""

    def __init__(self, eng: Engine):
        self.eng = eng
        self.current = np.full(N_PLAYERS, STAY)
        self.anchor = eng.pos.copy()

    def set_actions(self, actions: dict[int, int]) -> None:
        """Apply a new decision. One-shot actions (pass, shoot) fire immediately."""
        eng = self.eng
        for i, a in actions.items():
            a = int(a)
            if not action_mask(eng, i)[a]:
                a = STAY
            if PASS <= a < SHOOT:
                eng.request_pass(i, teammates(i)[a - PASS])
                a = STAY
            elif a == SHOOT:
                eng.request_shot(i)
                a = STAY
            if a == STAY and self.current[i] != STAY:
                self.anchor[i] = eng.pos[i]
            self.current[i] = a

    def update_targets(self) -> None:
        """Called every tick: re-aim each player at its macro action's moving target."""
        eng = self.eng
        for i in range(N_PLAYERS):
            a = self.current[i]
            if a == STAY:
                tgt = self.anchor[i]
            elif MOVE_TO <= a < PASS:
                tgt = court.REGION_XY[a - MOVE_TO]
            elif a == DRIVE:
                tgt = DRIVE_SPOT if eng.holder == i else eng.pos[i]
            elif SCREEN <= a < GUARD:
                mate = teammates(i)[a - SCREEN]
                opp = opponents(i)
                d = np.linalg.norm(eng.pos[opp] - eng.pos[mate], axis=1)
                defender = opp[int(np.argmin(d))]
                away = eng.pos[i] - eng.pos[defender]
                n = np.linalg.norm(away)
                away = away / n if n > 1e-6 else np.array([0.0, 1.0])
                tgt = eng.pos[defender] + away * SCREEN_GAP
            else:
                man = opponents(i)[a - GUARD]
                gap = GUARD_GAP_BALL if eng.holder == man else GUARD_GAP_OFF_BALL
                to_rim = court.BASKET - eng.pos[man]
                n = np.linalg.norm(to_rim)
                tgt = eng.pos[man] + (to_rim / n * min(gap, n) if n > 1e-6 else 0.0)
            eng.set_target(i, tgt)
