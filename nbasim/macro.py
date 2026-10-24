"""Macro actions for RL agents on the full-court engine.

A macro action is chosen every decision step (several engine ticks) and is
turned into engine requests here. Spots are named half-court regions in the
frame of the half being played (court.REGIONS), so the same action means the
same place whichever way the teams are attacking.

Layout (same for every agent; illegal ones are masked):
  AUTO                     let the team's scripted brain (TeamBrain) play this player
  HOLD                     stay where you are
  MOVE_TO(region)          one per court.REGIONS entry
  PASS(teammate k)         k = 0..3, teammates in lineup order skipping self
  SHOOT
  DRIVE                    attack the rim off the dribble
  SCREEN(teammate k)       set a screen on teammate k's nearest defender
  GUARD(opponent k)        k = 0..4, stay between that opponent and the basket

Dead balls (inbounds, free throws, jump balls) are always played by
TeamBrain; during them only AUTO is legal.
"""

from __future__ import annotations

import numpy as np

from . import court
from .engine import HELD, LIVE, Engine, members, team_of

N_REGIONS = len(court.REGION_NAMES)

AUTO = 0
HOLD = 1
MOVE_TO = 2
PASS = MOVE_TO + N_REGIONS
SHOOT = PASS + 4
DRIVE = SHOOT + 1
SCREEN = DRIVE + 1
GUARD = SCREEN + 4
N_ACTIONS = GUARD + 5

DRIVE_LOCAL = np.array([25.0, 6.0])
SCREEN_GAP = 2.6  # ft between screener and the defender being screened
GUARD_GAP_BALL = 3.0
GUARD_GAP_OFF_BALL = 6.0


def teammates(i: int) -> list[int]:
    return [j for j in members(team_of(i)) if j != i]


def opponents(i: int) -> list[int]:
    return list(members(1 - team_of(i)))


def describe(a: int) -> str:
    if a == AUTO:
        return "auto"
    if a == HOLD:
        return "hold"
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
    m[AUTO] = 1
    if eng.status not in LIVE:
        return m
    m[HOLD] = 1
    m[MOVE_TO:PASS] = 1
    if team_of(i) == eng.off:
        if eng.holder == i and eng.status == HELD:
            m[PASS:SHOOT] = 1
            m[SHOOT] = 1
            m[DRIVE] = 1
        elif eng.holder is not None and eng.holder != i:
            m[SCREEN:GUARD] = 1
    else:
        m[GUARD:N_ACTIONS] = 1
    return m


class MacroController:
    """Holds each agent's macro action and turns it into engine targets every tick."""

    def __init__(self, eng: Engine):
        self.eng = eng
        self.current = np.full(10, AUTO)
        self.anchor = eng.pos.copy()

    def manual(self, team: int) -> set[int]:
        """Players of `team` that are not on AUTO (TeamBrain must leave their decisions alone)."""
        return {i for i in members(team) if self.current[i] != AUTO}

    def set_actions(self, actions: dict[int, int]) -> None:
        """Apply a new decision. One-shot actions (pass, shoot, drive) fire immediately."""
        eng = self.eng
        for i, a in actions.items():
            a = int(a)
            if not action_mask(eng, i)[a]:
                a = AUTO
            if PASS <= a < SHOOT:
                eng.request_pass(i, teammates(i)[a - PASS])
                a = HOLD
            elif a == SHOOT:
                eng.request_shot(i)
                a = HOLD
            elif a == DRIVE:
                eng.start_drive(i)
            if a == HOLD and self.current[i] != HOLD:
                self.anchor[i] = eng.pos[i].copy()
            self.current[i] = a

    def release_dead_ball(self) -> None:
        """Dead balls are always scripted: put everyone back on AUTO."""
        if self.eng.status not in LIVE:
            self.current[:] = AUTO

    def update_targets(self, team: int | None = None) -> None:
        """Re-aim each manual player (of `team`, default both) at its macro action's target."""
        eng = self.eng
        if eng.status not in LIVE:
            return
        G = lambda loc: eng.glob(np.asarray(loc, float))  # frame of the half being played
        for i in (range(10) if team is None else members(team)):
            a = self.current[i]
            if a == AUTO:
                continue
            if a == HOLD:
                tgt = self.anchor[i]
            elif MOVE_TO <= a < PASS:
                tgt = G(court.REGION_LOCAL[a - MOVE_TO])
            elif a == DRIVE:
                tgt = G(DRIVE_LOCAL) if eng.holder == i else eng.pos[i]
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
                to_rim = eng.basket() - eng.pos[man]
                n = np.linalg.norm(to_rim)
                tgt = eng.pos[man] + (to_rim / n * min(gap, n) if n > 1e-6 else 0.0)
            eng.set_target(i, tgt)
