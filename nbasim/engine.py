"""Deterministic half-court game engine.

Agents only *request* things (a target to move to, a pass, a shot). The
engine owns the truth: it moves bodies, runs the clocks, flies the ball and
rolls every probabilistic outcome with its own seeded RNG.

Player indices 0-4 are the offense, 5-9 the defense.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import court
from .players import Player

DT = 0.1  # seconds per tick
SHOT_CLOCK = 24.0
OREB_SHOT_CLOCK = 14.0
PASS_SPEED = 40.0  # ft/s
SHOT_FLIGHT_TIME = 1.0  # s from release to the rim
STEAL_REACH = 2.0  # ft from the ball a defender can play a pass

# League-average make rate by zone before skill and contest modifiers.
BASE_MAKE = {"rim": 0.63, "paint": 0.42, "mid": 0.41, "corner3": 0.39, "above3": 0.35}

N_OFF = 5
N_PLAYERS = 10


def _unit(r: float) -> float:
    """Map a 25-99 rating to 0..1."""
    return (r - 25.0) / 74.0


@dataclass
class Event:
    t: float
    type: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"t": round(self.t, 2), "type": self.type, **self.data}


@dataclass
class Flight:
    kind: str  # "pass" or "shot"
    passer: int
    target: int = -1  # receiver for a pass
    start: np.ndarray | None = None
    elapsed: float = 0.0
    rolled: set = field(default_factory=set)  # defenders who already tried a steal
    # shot fields
    made: bool = False
    points: int = 0


class Engine:
    def __init__(
        self,
        offense: list[Player],
        defense: list[Player],
        seed: int | None = None,
        game_clock: float = 720.0,
    ):
        assert len(offense) == N_OFF and len(defense) == N_OFF
        self.players: list[Player] = list(offense) + list(defense)
        self.rng = np.random.default_rng(seed)

        a = [p.athleticism for p in self.players]
        self.max_speed = np.array([14.0 + 8.0 * _unit(x.speed) for x in a])  # ft/s
        self.accel = np.array([10.0 + 12.0 * _unit(x.acceleration) for x in a])  # ft/s^2
        self.ball_speed_mult = np.array(
            [0.80 + 0.15 * _unit(p.playmaking.speed_with_ball) for p in self.players]
        )
        self.mass = np.array(
            [p.weight_lb * (0.5 + p.athleticism.strength / 99.0) for p in self.players]
        )

        self.pos = np.zeros((N_PLAYERS, 2))
        self.vel = np.zeros((N_PLAYERS, 2))
        self.target = np.zeros((N_PLAYERS, 2))
        self.ball = court.BASKET.copy()
        self.holder: int | None = None
        self.flight: Flight | None = None

        self.t = 0.0
        self.shot_clock = SHOT_CLOCK
        self.game_clock = game_clock
        self.done = False
        self.outcome: str | None = None
        self.points = 0
        self.events: list[Event] = []

    # ------------------------------------------------------------------ setup
    def place(self, positions: np.ndarray, holder: int = 0) -> None:
        self.pos[:] = positions
        court.clamp_to_court(self.pos)
        self.target[:] = self.pos
        self.vel[:] = 0.0
        self.holder = holder
        self.ball = self.pos[holder].copy()

    # --------------------------------------------------------------- requests
    def set_target(self, i: int, xy) -> None:
        self.target[i] = xy

    def request_pass(self, i: int, j: int) -> bool:
        if self.done or self.holder != i or i == j or (j < N_OFF) != (i < N_OFF):
            return False
        self.holder = None
        self.flight = Flight(kind="pass", passer=i, target=j, start=self.ball.copy())
        self._log("pass", by=i, to=j)
        return True

    def request_shot(self, i: int) -> bool:
        if self.done or self.holder != i:
            return False
        p, zone, three = self.shot_make_prob(i)
        made = bool(self.rng.random() < p)
        pts = 3 if three else 2
        self.holder = None
        self.flight = Flight(kind="shot", passer=i, start=self.ball.copy(), made=made, points=pts)
        self._log("shot", by=i, zone=zone, p=round(p, 3), made=made, pts=pts if made else 0,
                  x=round(float(self.pos[i, 0]), 1), y=round(float(self.pos[i, 1]), 1))
        return True

    # ------------------------------------------------------------- resolution
    def shot_make_prob(self, i: int, at=None) -> tuple[float, str, bool]:
        """Make probability for player i shooting from `at` (default: where they stand)."""
        at = self.pos[i] if at is None else np.asarray(at)
        zone = court.shot_zone(at)
        three = court.is_three(at)
        pl = self.players[i]
        skill = {
            "rim": max(pl.inside_scoring.driving_layup, pl.inside_scoring.close_shot),
            "paint": pl.inside_scoring.close_shot,
            "mid": pl.outside_scoring.mid_range,
            "corner3": pl.outside_scoring.three_point,
            "above3": pl.outside_scoring.three_point,
        }[zone]
        p = BASE_MAKE[zone] * (0.75 + 0.45 * _unit(skill))

        opp = range(N_OFF, N_PLAYERS) if i < N_OFF else range(N_OFF)
        d = np.linalg.norm(self.pos[list(opp)] - at, axis=1)
        k = int(np.argmin(d))
        nearest, dist = list(opp)[k], float(d[k])
        dp = self.players[nearest].defense
        def_skill = (
            (dp.interior_defense + dp.block) / 2 if zone in ("rim", "paint") else dp.perimeter_defense
        )
        height_edge = (self.players[nearest].height_in - pl.height_in) / 12.0
        if dist < 6.0:
            strength = 0.30 * (0.7 + 0.6 * _unit(def_skill)) * (1.0 + 0.3 * height_edge)
            p *= 1.0 - strength * (1.0 - dist / 6.0)
        return float(np.clip(p, 0.02, 0.95)), zone, three

    def _try_steals(self) -> int | None:
        f = self.flight
        passer = self.players[f.passer].playmaking.pass_accuracy
        opp = range(N_OFF, N_PLAYERS) if f.passer < N_OFF else range(N_OFF)
        for j in opp:
            if j in f.rolled:
                continue
            if np.linalg.norm(self.pos[j] - self.ball) <= STEAL_REACH:
                f.rolled.add(j)
                dp = self.players[j].defense
                skill = (dp.steal + dp.pass_perception) / 2
                p = float(np.clip(0.25 + 0.4 * (skill - passer) / 74.0, 0.05, 0.6))
                if self.rng.random() < p:
                    return j
        return None

    def _rebound(self) -> None:
        d = np.linalg.norm(self.pos - court.BASKET, axis=1)
        w = np.zeros(N_PLAYERS)
        for i, pl in enumerate(self.players):
            r = pl.rebounding.offensive_rebound if i < N_OFF else pl.rebounding.defensive_rebound
            w[i] = (r / 99.0) ** 2 * math.exp(-d[i] / 4.0)
            w[i] *= 1.0 if i < N_OFF else 3.0  # defense has inside position on a miss
        if w.sum() <= 0:
            w[:] = 1.0
        winner = int(self.rng.choice(N_PLAYERS, p=w / w.sum()))
        self._log("rebound", by=winner, offensive=winner < N_OFF)
        if winner < N_OFF:
            self.holder = winner
            self.ball = self.pos[winner].copy()
            self.shot_clock = max(self.shot_clock, OREB_SHOT_CLOCK)
        else:
            self._end("defensive_rebound")

    # ------------------------------------------------------------------- tick
    def step(self) -> None:
        if self.done:
            return
        self._move()
        self._advance_ball()
        if self.done:
            return

        self.t += DT
        self.game_clock = max(0.0, self.game_clock - DT)
        if not (self.flight and self.flight.kind == "shot"):
            self.shot_clock = max(0.0, self.shot_clock - DT)
            if self.shot_clock <= 1e-9:
                self._log("violation", rule="shot_clock")
                self._end("shot_clock_violation")
                return
        if self.game_clock <= 1e-9 and self.flight is None:
            self._end("end_of_period")

    def _move(self) -> None:
        speed = self.max_speed.copy()
        if self.holder is not None:
            speed[self.holder] *= self.ball_speed_mult[self.holder]

        to = self.target - self.pos
        dist = np.linalg.norm(to, axis=1, keepdims=True)
        direction = np.divide(to, dist, out=np.zeros_like(to), where=dist > 1e-6)
        # "arrive" steering: slow down within the last half second of travel
        want = direction * np.minimum(speed[:, None], dist / 0.5)
        dv = want - self.vel
        dv_norm = np.linalg.norm(dv, axis=1, keepdims=True)
        cap = (self.accel * DT)[:, None]
        dv = np.where(dv_norm > cap, dv / np.maximum(dv_norm, 1e-9) * cap, dv)
        self.vel += dv
        self.pos += self.vel * DT

        # body collisions: push overlapping players apart, heavier players move less
        min_d = 2 * court.PLAYER_RADIUS
        for _ in range(2):
            diff = self.pos[:, None, :] - self.pos[None, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            ii, jj = np.nonzero(d < min_d)
            for i, j in zip(ii, jj):
                if i >= j:
                    continue
                n = diff[i, j] / max(d[i, j], 1e-6)
                overlap = min_d - d[i, j]
                mi, mj = self.mass[i], self.mass[j]
                self.pos[i] += n * overlap * mj / (mi + mj)
                self.pos[j] -= n * overlap * mi / (mi + mj)
        court.clamp_to_court(self.pos)

    def _advance_ball(self) -> None:
        if self.holder is not None:
            self.ball = self.pos[self.holder].copy()
            return
        f = self.flight
        if f is None:
            return
        f.elapsed += DT
        if f.kind == "pass":
            to = self.pos[f.target] - self.ball
            d = float(np.linalg.norm(to))
            step = PASS_SPEED * DT
            if d <= step:
                self.ball = self.pos[f.target].copy()
                self.holder, self.flight = f.target, None
                self._log("catch", by=f.target)
                return
            self.ball = self.ball + to / d * step
            thief = self._try_steals()
            if thief is not None:
                self.holder, self.flight = thief, None
                self.ball = self.pos[thief].copy()
                self._log("steal", by=thief, from_=f.passer)
                self._end("turnover")
        else:
            frac = min(1.0, f.elapsed / SHOT_FLIGHT_TIME)
            self.ball = f.start + (court.BASKET - f.start) * frac
            if frac >= 1.0:
                self.flight = None
                self.ball = court.BASKET.copy()
                if f.made:
                    self.points = f.points
                    self._log("score", by=f.passer, pts=f.points)
                    self._end("made_shot")
                else:
                    self._log("miss", by=f.passer)
                    self._rebound()

    # ---------------------------------------------------------------- helpers
    def ball_height(self) -> float:
        f = self.flight
        if f is None:
            return 3.5
        if f.kind == "shot":
            s = min(1.0, f.elapsed / SHOT_FLIGHT_TIME)
            return 7.0 + (10.0 - 7.0) * s + 8.0 * 4 * s * (1 - s)
        return 5.0

    def _end(self, outcome: str) -> None:
        self.done = True
        self.outcome = outcome
        self._log("end", outcome=outcome, points=self.points)

    def _log(self, type_: str, **data) -> None:
        data = {k.rstrip("_"): v for k, v in data.items()}
        self.events.append(Event(self.t, type_, data))
