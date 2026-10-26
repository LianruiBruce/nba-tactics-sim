"""Deterministic full-court game engine.

The engine owns the truth: bodies, ball, clocks, rules and every random
outcome (rolled with its own seeded RNG). Team "brains" (scripted or RL) only
set movement targets and request passes and shots each tick.

Players 0-4 are team 0, 5-9 are team 1, each in lineup order PG, SG, SF, PF, C.
Team 0 attacks the basket at x = 88.75 in the first half.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from . import court, resolve
from .players import Player

DT = 0.1
QUARTER_LEN = 720.0
OT_LEN = 300.0
SHOT_CLOCK = 24.0
SHOT_CLOCK_RESET = 14.0
EIGHT_SECONDS = 8.0
THREE_SECONDS = 3.0
INBOUND_SECONDS = 5.0
PASS_SPEED = 38.0
N = 10

# ball status
HELD, PASS, SHOT, LOOSE, DEAD, INBOUND, FREE_THROW = (
    "held", "pass", "shot", "loose", "dead", "inbound", "free_throw")
LIVE = (HELD, PASS, SHOT, LOOSE)


class Brain(Protocol):
    def act(self, eng: "Engine", team: int) -> None: ...


def team_of(i: int) -> int:
    return 0 if i < 5 else 1


def members(team: int) -> range:
    return range(0, 5) if team == 0 else range(5, 10)


# shot zones grouped the way nbasim.realmodel measures shot making
ZGROUP = {"rim": 0, "short_mid": 1, "long_mid": 1, "corner3": 2, "above3": 2}


@dataclass
class Event:
    t: float
    q: int
    gc: float
    type: str
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"t": round(self.t, 2), "q": self.q, "gc": round(self.gc, 1), "type": self.type, **self.data}


@dataclass
class Flight:
    kind: str                     # pass | shot | loose | ft
    src: int = -1                 # passer / shooter
    dst: int = -1                 # receiver
    start: np.ndarray | None = None
    end: np.ndarray | None = None
    duration: float = 1.0
    elapsed: float = 0.0
    rolled: set = field(default_factory=set)
    made: bool = False
    points: int = 0
    zone: str = ""
    and_one: bool = False
    from_rim: bool = False        # loose ball off a missed shot (a rebound)
    ft_miss: bool = False
    inbound: bool = False


@dataclass
class Restart:
    kind: str                     # inbound | free_throws | jump
    team: int                     # team that gets the ball
    spot: np.ndarray | None = None
    inbounder: int | None = None
    setup: float = 1.5
    clock_runs: bool = False
    shot_clock: float = SHOT_CLOCK
    shooter: int = -1
    n_shots: int = 0
    attempt: int = 0
    then_inbound: bool = False    # technical FT: offense keeps the ball afterwards
    timer: float = 0.0


class Engine:
    def __init__(self, team_a: list[Player], team_b: list[Player], seed: int | None = None,
                 quarters: int = 4, quarter_len: float = QUARTER_LEN):
        assert len(team_a) == 5 and len(team_b) == 5
        self.players: list[Player] = list(team_a) + list(team_b)
        self.rng = np.random.default_rng(seed)
        self.quarters = quarters
        self.quarter_len = quarter_len

        ath = [p.athleticism for p in self.players]
        self.max_speed = np.array([14.5 + 8.0 * (a.speed - 25) / 74 for a in ath])
        self.accel = np.array([11.0 + 12.0 * (a.agility - 25) / 74 for a in ath])
        self.ball_speed = np.array([0.80 + 0.15 * (p.playmaking.speed_with_ball - 25) / 74 for p in self.players])
        self.mass = np.array([p.weight_lb * (0.5 + p.athleticism.strength / 99.0) for p in self.players])

        self.pos = np.zeros((N, 2))
        self.vel = np.zeros((N, 2))
        self.hist: list[np.ndarray] = []  # recent positions, newest last (for reaction delays)
        self.target = np.zeros((N, 2))
        self.cap = self.max_speed.copy()
        self.beaten_until = np.full(N, -1.0)  # defenders blown by on a drive move slower briefly
        self.ball = np.array([court.LENGTH / 2, court.WIDTH / 2])
        self.ball_z = 3.5
        self.status = DEAD
        self.holder: int | None = None
        self.flight: Flight | None = None
        self.restart: Restart | None = None

        self.direction = [1, -1]
        self.off = 0
        self.t = 0.0
        self.quarter = 1
        self.game_clock = quarter_len
        self.shot_clock = SHOT_CLOCK
        self.score = [0, 0]
        self.team_fouls = [0, 0]
        self.fouls_last2 = [0, 0]
        self.pf = np.zeros(N, dtype=int)
        self.done = False
        self.jump_winner = 0

        self.frontcourt = False
        self.eight_timer = 0.0
        self.paint_timer = np.zeros(N)
        self.hold_time = 0.0
        self.catch_pos = np.zeros(2)
        self.last_pass_from: int | None = None
        self.last_catch_t = -99.0
        self.inbound_timer = 0.0
        self.labels: dict[int, str] = {0: "", 1: ""}
        self.events: list[Event] = []
        self.stats = {k: np.zeros(N, dtype=int) for k in (
            "pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "orb", "drb", "ast", "stl",
            "blk", "tov", "pf")}
        self.zone_att = {z: [0, 0] for z in ("rim", "short_mid", "long_mid", "corner3", "above3")}
        self.zone_made = {z: [0, 0] for z in self.zone_att}
        self.possessions = [0, 0]
        self.tov_types: dict[str, int] = {}
        self.foul_types: dict[str, int] = {}
        self._lineup_terms()

        self._start_period(first=True)

    def _lineup_terms(self) -> None:
        """Log-odds terms from real data that depend on all ten players (zero for 2K-only players).

        nbasim.realmodel measures, per player, how his presence moves teammates' and
        opponents' shot making, turnovers and fouls drawn; ratings carry only the
        shooter's own skill, so these terms are added on top.
        """
        self.lineup_make = np.zeros((N, 3))  # added to the make logit of i's shots, by zone group
        self.lineup_tov = np.zeros(2)        # log multiplier on the team's turnover hazards
        self.lineup_trip = np.zeros(2)       # log multiplier on the team's shooting-foul chance
        if not all(p.real for p in self.players):
            return
        r = [p.real for p in self.players]
        for team in (0, 1):
            off, de = members(team), members(1 - team)
            self.lineup_tov[team] = sum(r[i]["tov_off"] for i in off) + sum(r[j]["tov_def"] for j in de)
            self.lineup_trip[team] = sum(r[i]["trip_off"] for i in off) + sum(r[j]["trip_def"] for j in de)
            for i in off:
                for g, name in enumerate(("rim", "mid", "three")):
                    self.lineup_make[i, g] = (sum(r[m][f"mate_{name}"] for m in off if m != i)
                                              + sum(r[j][f"guard_{name}"] for j in de))

    # ------------------------------------------------------------- frames
    def local(self, xy, team: int | None = None) -> np.ndarray:
        """Global -> local frame of `team`'s attack (default: the offense)."""
        return court.to_local(xy, self.direction[self.off if team is None else team])

    def glob(self, local, team: int | None = None) -> np.ndarray:
        return court.to_global(local, self.direction[self.off if team is None else team])

    def basket(self, team: int | None = None) -> np.ndarray:
        return court.basket(self.direction[self.off if team is None else team])

    @property
    def defn(self) -> int:
        return 1 - self.off

    # ----------------------------------------------------------- requests
    def set_target(self, i: int, xy, speed: float | None = None) -> None:
        """Move toward xy; `speed` caps the pace (ft/s) for this tick, e.g. a jog."""
        self.target[i] = xy
        if speed is not None:
            self.cap[i] = min(self.cap[i], speed)

    def request_pass(self, i: int, j: int) -> bool:
        if self.holder != i or i == j or team_of(i) != team_of(j) or self.status not in (HELD, INBOUND):
            return False
        inbound = self.status == INBOUND
        self.holder = None
        self.status = PASS
        self.flight = Flight(kind="pass", src=i, dst=j, start=self.ball.copy(), inbound=inbound)
        self._log("pass", by=i, to=j)
        if not inbound and self.rng.random() < resolve.CAL["bad_pass_oob"] * math.exp(self.lineup_tov[team_of(i)]) * (
                1.0 + 0.5 * (75 - self.players[i].playmaking.pass_accuracy) / 10):
            self._log("bad_pass", by=i)
            self._turnover(i, "bad_pass_oob", steal_by=None)
        return True

    def request_shot(self, i: int) -> bool:
        if self.holder != i or self.status != HELD:
            return False
        shooter = self.players[i]
        loc = self.local(self.pos[i])
        d_idx, d_dist = self.contest(i)
        defender = self.players[d_idx] if d_idx is not None else None
        near_rim_catch = self.hold_time < 1.0 and court.dist_to_basket(loc) < 6
        p, zone = resolve.make_prob(shooter, loc, defender, d_dist, self.off_dribble(i), near_rim_catch)
        p = resolve.sigmoid(resolve.logit(p) + self.lineup_make[i, ZGROUP[zone]]
                            + self.rng.normal(0.0, resolve.shot_noise_sd(shooter)))
        three = court.is_three(loc)
        pts = 3 if three else 2
        self._clear_paint_timers()
        self.holder = None

        # any defender in range can block, the best-placed one counts
        blocker, p_blk = None, 0.0
        for j in members(1 - team_of(i)):
            dj = float(np.linalg.norm(self.pos[j] - self.pos[i]))
            pj = resolve.block_prob(shooter, self.players[j], dj, zone)
            if pj > p_blk:
                blocker, p_blk = j, pj
        if blocker is not None and self.rng.random() < p_blk:
            self._count_fga(i, zone, three, made=False)
            self.stats["blk"][blocker] += 1
            self._log("shot", by=i, zone=zone, p=round(p, 3), pts=0, made=False, blocked=True)
            self._log("block", by=blocker, on=i)
            self._loose_ball(from_xy=self.pos[i], dist=(4, 12), duration=0.6, from_rim=True)
            return True

        made = bool(self.rng.random() < p)
        fouled = defender is not None and self.rng.random() < resolve.shooting_foul_prob(
            shooter, defender, d_dist, zone) * math.exp(self.lineup_trip[team_of(i)])
        if fouled and not made:
            self._log("shot", by=i, zone=zone, p=round(p, 3), pts=0, made=False, fouled=True)
            self._foul(d_idx, i, kind="shooting", shots=pts)
            return True
        self._count_fga(i, zone, three, made)
        dist = float(np.linalg.norm(self.basket() - self.pos[i]))
        self.status = SHOT
        self.flight = Flight(kind="shot", src=i, start=self.ball.copy(), end=self.basket().copy(),
                             duration=0.55 + dist / 45.0, made=made, points=pts, zone=zone,
                             and_one=made and fouled)
        self._log("shot", by=i, zone=zone, p=round(p, 3), pts=pts if made else 0, made=made,
                  x=round(float(loc[0]), 1), y=round(float(loc[1]), 1))
        if made and fouled:
            self._and_one_fouler = d_idx
        return True

    # ------------------------------------------------------------ helpers
    def contest(self, i: int) -> tuple[int | None, float]:
        """Closest defender to player i and the effective contest distance."""
        opp = list(members(1 - team_of(i)))
        d = np.linalg.norm(self.pos[opp] - self.pos[i], axis=1)
        k = int(np.argmin(d))
        j, dist = opp[k], float(d[k])
        # a defender trailing behind the shooter contests less
        to_rim = self.basket(team_of(i)) - self.pos[i]
        if float((self.pos[j] - self.pos[i]) @ to_rim) < 0:
            dist *= 1.6
        return j, dist

    def off_dribble(self, i: int) -> bool:
        caught_recently = self.t - self.last_catch_t < 1.2 and self.last_pass_from is not None
        moved = float(np.linalg.norm(self.pos[i] - self.catch_pos))
        return not (caught_recently and moved < 4.0)

    def start_drive(self, i: int) -> bool:
        """Holder attacks off the dribble: roll the first step against the nearest defender.

        Returns True if the defender is beaten (he reacts late and moves slower
        for about a second), which is how a 2K-style blow-by happens.
        """
        if self.holder != i or self.status != HELD:
            return False
        opp = list(members(1 - team_of(i)))
        d = np.linalg.norm(self.pos[opp] - self.pos[i], axis=1)
        k = int(np.argmin(d))
        j = opp[k]
        if d[k] > 6.0:
            self._log("drive", by=i, beat=None)
            return True
        p, dp = self.players[i], self.players[j]
        quick = 0.5 * (p.playmaking.ball_handle + p.playmaking.speed_with_ball) + 0.3 * p.athleticism.speed
        stop = 0.5 * (dp.defense.perimeter_defense + dp.athleticism.agility) + 0.3 * dp.athleticism.speed
        beat = bool(self.rng.random() < resolve.sigmoid((quick - stop) / 12.0 + resolve.CAL["blowby_bias"]))
        if beat:
            self.beaten_until[j] = self.t + 1.0
        self._log("drive", by=i, beat=beat, on=j)
        return beat

    def seen_pos(self, i: int, delay: float) -> np.ndarray:
        """Where player i was `delay` seconds ago (what a slow defender is reacting to)."""
        k = min(len(self.hist), int(round(delay / DT)))
        return self.hist[-k][i] if k > 0 else self.pos[i]

    def shot_value(self, i: int, at=None) -> float:
        """Expected points of a shot by i from `at` (default where he stands), no randomness."""
        at = self.pos[i] if at is None else np.asarray(at)
        loc = self.local(at, team_of(i))
        opp = list(members(1 - team_of(i)))
        d = np.linalg.norm(self.pos[opp] - at, axis=1)
        k = int(np.argmin(d))
        p, zone = resolve.make_prob(self.players[i], loc, self.players[opp[k]], float(d[k]),
                                    off_dribble=at is None and self.off_dribble(i))
        if self.lineup_make[i, ZGROUP[zone]]:
            p = resolve.sigmoid(resolve.logit(p) + self.lineup_make[i, ZGROUP[zone]])
        return p * (3 if court.is_three(loc) else 2)

    # ---------------------------------------------------------------- tick
    def step(self, brains: tuple[Brain, Brain]) -> None:
        if self.done:
            return
        self.cap[:] = self.max_speed
        if self.status == DEAD:
            self._dead_ball_setup()
        if self.status == FREE_THROW:
            self._free_throws()
        else:
            brains[0].act(self, 0)
            brains[1].act(self, 1)
            if self.restart is not None and self.restart.kind == "free_throws":
                self._line_up_free_throw(self.restart)
            if self.status == DEAD and self.restart is not None and self.restart.kind == "inbound":
                self.target[self.restart.inbounder] = self.restart.spot
            if self.status == INBOUND and self.holder is not None:
                self.target[self.holder] = self.pos[self.holder]
        self._move()
        self.hist.append(self.pos.copy())
        if len(self.hist) > 8:
            self.hist.pop(0)
        self._update_ball()
        self._rules()
        self._clocks()
        self.t += DT

    # ------------------------------------------------------------ physics
    def _move(self) -> None:
        speed = np.minimum(self.max_speed, self.cap)
        speed = np.where(self.t < self.beaten_until, speed * 0.7, speed)
        if self.holder is not None:
            speed[self.holder] *= self.ball_speed[self.holder]
        to = self.target - self.pos
        dist = np.linalg.norm(to, axis=1, keepdims=True)
        direction = np.divide(to, dist, out=np.zeros_like(to), where=dist > 1e-6)
        want = direction * np.minimum(speed[:, None], dist / 0.4)
        dv = want - self.vel
        dvn = np.linalg.norm(dv, axis=1, keepdims=True)
        cap = (self.accel * DT)[:, None]
        dv = np.where(dvn > cap, dv / np.maximum(dvn, 1e-9) * cap, dv)
        self.vel += dv
        self.pos += self.vel * DT

        keep = self.restart.inbounder if (self.restart and self.restart.kind == "inbound") else None
        for _ in range(2):  # a second pass settles bodies pushed back in from the line
            self._separate()
            saved = self.pos[keep].copy() if keep is not None else None
            court.clamp_inbounds(self.pos)
            if keep is not None:
                self.pos[keep] = np.clip(saved, -3.0, [court.LENGTH + 3.0, court.WIDTH + 3.0])

    def _separate(self) -> None:
        min_d = 2 * court.PLAYER_RADIUS
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        ii, jj = np.nonzero(np.triu(d < min_d))
        for i, j in zip(ii, jj):
            n = diff[i, j] / max(d[i, j], 1e-6)
            overlap = min_d - d[i, j]
            mi, mj = self.mass[i], self.mass[j]
            self.pos[i] += n * overlap * mj / (mi + mj)
            self.pos[j] -= n * overlap * mi / (mi + mj)

    # --------------------------------------------------------------- ball
    def _update_ball(self) -> None:
        if self.holder is not None:
            self.ball = self.pos[self.holder].copy()
            self.ball_z = 3.5
            self.hold_time += DT
            return
        f = self.flight
        if f is None:
            return
        f.elapsed += DT
        if f.kind == "pass":
            self._fly_pass(f)
        elif f.kind in ("shot", "ft"):
            s = min(1.0, f.elapsed / f.duration)
            self.ball = f.start + (f.end - f.start) * s
            self.ball_z = 7.0 + 3.0 * s + 9.0 * 4 * s * (1 - s)
            if s >= 1.0:
                self._at_rim(f)
        elif f.kind == "loose":
            s = min(1.0, f.elapsed / f.duration)
            self.ball = f.start + (f.end - f.start) * s
            self.ball_z = 3.0 + 6.0 * 4 * s * (1 - s) + (7.0 if f.from_rim else 0.0) * (1 - s)
            if s >= 1.0:
                self._resolve_loose(f)

    def _fly_pass(self, f: Flight) -> None:
        to = self.pos[f.dst] - self.ball
        d = float(np.linalg.norm(to))
        step = PASS_SPEED * DT
        self.ball_z = 5.0
        if d <= step + 1.0:
            self._catch(f.dst, f.src, f.inbound)
            return
        self.ball = self.ball + to / d * step
        for j in members(1 - team_of(f.src)):
            if j in f.rolled:
                continue
            dist = float(np.linalg.norm(self.pos[j] - self.ball))
            if dist <= resolve.CAL["steal_reach"]:
                f.rolled.add(j)
                p_stl = resolve.steal_prob(self.players[f.src], self.players[j], dist)
                if self.lineup_tov[team_of(f.src)]:
                    p_stl = resolve.sigmoid(resolve.logit(p_stl) + self.lineup_tov[team_of(f.src)])
                if self.rng.random() < p_stl:
                    self._log("steal", by=j, kind="interception")
                    self.stats["stl"][j] += 1
                    self._turnover(f.src, "bad_pass_steal", steal_by=j)
                    return

    def _catch(self, j: int, passer: int, inbound: bool) -> None:
        self.flight = None
        self.holder = j
        self.status = HELD
        self.ball = self.pos[j].copy()
        self.hold_time = 0.0
        self.catch_pos = self.pos[j].copy()
        self.last_pass_from = passer
        self.last_catch_t = self.t
        self._log("catch", by=j)
        if inbound:
            r = self.restart
            self.restart = None
            self.shot_clock = r.shot_clock
            self.frontcourt = court.in_frontcourt(self.local(self.pos[j]))
            self.eight_timer = 0.0

    def _at_rim(self, f: Flight) -> None:
        self.flight = None
        if f.kind == "ft":
            return  # handled by the free-throw sequencer
        shooter = f.src
        if f.made:
            self._score(shooter, f.points)
            ast = self.last_pass_from
            if ast is not None and team_of(ast) == team_of(shooter) and self.t - self.last_catch_t < 3.5:
                self.stats["ast"][ast] += 1
            if f.and_one:
                self._log("and_one", by=shooter)
                self._foul(self._and_one_fouler, shooter, kind="shooting", shots=1)
            else:
                self._made_basket_restart()
            return
        self._log("miss", by=shooter)
        far = {"rim": (3, 8), "short_mid": (5, 11), "long_mid": (6, 13)}.get(f.zone, (8, 16))
        self._loose_ball(from_xy=self.basket(), dist=far, duration=0.8, from_rim=True, shot_from=self.pos[shooter])

    def _loose_ball(self, from_xy, dist, duration, from_rim, shot_from=None, ft_miss=False) -> None:
        bk_local = court.BASKET_LOCAL
        if shot_from is not None:
            s_loc = self.local(shot_from)
            base = math.atan2(s_loc[1] - bk_local[1], s_loc[0] - bk_local[0])
            ang = base + math.pi + self.rng.normal(0, 0.9) if self.rng.random() < 0.45 else base + self.rng.normal(0, 0.6)
        else:
            ang = self.rng.uniform(0.15, math.pi - 0.15)
        r = self.rng.uniform(*dist)
        start_local = self.local(from_xy)
        end_local = start_local + np.array([math.cos(ang), abs(math.sin(ang))]) * r
        end_local = np.clip(end_local, [1.5, 1.5], [48.5, 60.0])
        self.holder = None
        self.status = LOOSE
        self.flight = Flight(kind="loose", start=np.asarray(from_xy, float).copy(), end=self.glob(end_local),
                             duration=duration, from_rim=from_rim, ft_miss=ft_miss)

    def _resolve_loose(self, f: Flight) -> None:
        spot = f.end
        d = np.linalg.norm(self.pos - spot, axis=1)
        w = np.zeros(N)
        for i in range(N):
            if d[i] > 9.0:
                continue
            offensive = team_of(i) == self.off
            w[i] = math.exp(-d[i] / 2.2) * resolve.rebound_skill(self.players[i], offensive)
            if not offensive and f.from_rim:
                w[i] *= resolve.CAL["ft_reb_def_bias" if f.ft_miss else "defense_reb_bias"]
        if w.sum() <= 0:
            w = np.exp(-d / 3.0)
        winner = int(self.rng.choice(N, p=w / w.sum()))
        self.flight = None
        self.holder = winner
        self.status = HELD
        self.ball = self.pos[winner].copy()
        self.hold_time = 0.0
        self.catch_pos = self.pos[winner].copy()
        self.last_pass_from = None
        offensive = team_of(winner) == self.off
        if f.from_rim:
            self.stats["orb" if offensive else "drb"][winner] += 1
            self._log("rebound", by=winner, offensive=offensive)
            if offensive:
                self.shot_clock = max(self.shot_clock, SHOT_CLOCK_RESET)
                self.frontcourt = court.in_frontcourt(self.local(self.pos[winner]))
            else:
                self._change_possession(team_of(winner))
        else:
            self._log("recover", by=winner)
            if offensive:
                # a deflection into the backcourt is not a backcourt violation
                self.frontcourt = court.in_frontcourt(self.local(self.pos[winner]))
            if not offensive:
                self.stats["stl"][winner] += 1
                self.stats["tov"][self._loose_from] += 1
                self.tov_types["lost_ball"] = self.tov_types.get("lost_ball", 0) + 1
                self._change_possession(team_of(winner))

    # -------------------------------------------------------------- rules
    def _rules(self) -> None:
        if self.status == INBOUND:
            self.inbound_timer += DT
            if self.inbound_timer > INBOUND_SECONDS:
                self._violation(self.holder, "five_seconds")
            return
        if self.status not in (HELD, PASS):
            return
        off = self.off
        h = self.holder
        if h is not None:
            loc = self.local(self.pos[h])
            infront = court.in_frontcourt(loc)
            if not self.frontcourt:
                if infront:
                    self.frontcourt = True
                else:
                    self.eight_timer += DT
                    if self.eight_timer > EIGHT_SECONDS:
                        self._violation(h, "eight_seconds")
                        return
            elif not infront:
                self._violation(h, "backcourt")
                return
            if self._hazards(h):
                return
        # three seconds
        if self.frontcourt:
            for i in range(N):
                inp = court.in_paint(self.local(self.pos[i]))
                if team_of(i) == off:
                    self.paint_timer[i] = self.paint_timer[i] + DT if inp else 0.0
                    if self.paint_timer[i] > THREE_SECONDS:
                        # a holder still moving toward the basket gets the shooting allowance
                        if self.holder == i and float(self.vel[i] @ (self.basket() - self.pos[i])) > 0.0:
                            continue
                        self._violation(i, "offensive_three_seconds")
                        return
                else:
                    near = np.min(np.linalg.norm(self.pos[list(members(off))] - self.pos[i], axis=1))
                    active = near < 4.5
                    self.paint_timer[i] = self.paint_timer[i] + DT if (inp and not active) else 0.0
                    if self.paint_timer[i] > THREE_SECONDS:
                        self._defensive_three(i)
                        return

    def _hazards(self, h: int) -> bool:
        holder = self.players[h]
        opp = list(members(1 - team_of(h)))
        d = np.linalg.norm(self.pos[opp] - self.pos[h], axis=1)
        k = int(np.argmin(d))
        j, dist = opp[k], float(d[k])
        if self.rng.random() < resolve.CAL["travel_rate"] * DT:
            self._violation(h, "traveling")
            return True
        if dist < 3.0:
            defender = self.players[j]
            if self.rng.random() < resolve.strip_rate(holder, defender) * math.exp(self.lineup_tov[team_of(h)]) * DT:
                self._log("strip", by=j, on=h)
                self._loose_from = h
                self._loose_ball(from_xy=self.pos[h], dist=(2, 7), duration=0.5, from_rim=False)
                return True
            v = self.vel[h]
            sp = float(np.linalg.norm(v))
            # hand-checks and reach-ins happen when the handler is attacking, not while he
            # stands and surveys (NBA: about 7% of drives end in a foul)
            # a defender in foul trouble stops reaching
            caution = 0.6 ** max(0, int(self.pf[j]) - 1)
            if sp > 6.0 and self.rng.random() < resolve.reach_foul_rate(defender) * caution * DT:
                self._foul(j, h, kind="personal")
                return True
            if sp > 9.0 and dist < 2.7 and float(np.linalg.norm(self.vel[j])) < 4.0:
                ahead = float((self.pos[j] - self.pos[h]) @ v) / (sp * dist)
                if ahead > 0.8 and self.rng.random() < resolve.charge_rate(holder, defender) * DT:
                    self._log("charge", by=h, on=j)
                    self.pf[h] += 1
                    self.stats["pf"][h] += 1
                    self._turnover(h, "offensive_foul", steal_by=None)
                    return True
        return False

    def _clocks(self) -> None:
        if self.done:
            return
        inbound_pass = self.flight is not None and self.flight.inbound
        running = (self.status in LIVE and not inbound_pass) or (
            self.status == DEAD and self.restart is not None and self.restart.clock_runs)
        if running:
            self.game_clock = max(0.0, self.game_clock - DT)
            if self.status in LIVE:
                self.shot_clock = max(0.0, self.shot_clock - DT)
                # the shot clock is off when less game time remains; then it never expires first
                if self.shot_clock <= 1e-9 and self.status in (HELD, PASS) and self.game_clock > 1e-9:
                    self._violation(self.holder if self.holder is not None else self.flight.src, "shot_clock")
                    return
        ft_pending = self.status == FREE_THROW or (self.restart is not None and self.restart.kind == "free_throws")
        if self.game_clock <= 1e-9 and self.status != SHOT and not ft_pending:
            self._end_period()

    # --------------------------------------------------- possession changes
    def _change_possession(self, team: int) -> None:
        self.off = team
        self.possessions[team] += 1
        self.shot_clock = SHOT_CLOCK
        self.paint_timer[:] = 0.0
        h = self.holder
        self.frontcourt = h is not None and court.in_frontcourt(self.local(self.pos[h]))
        self.eight_timer = 0.0
        self._log("possession", team=team)

    def _turnover(self, i: int, kind: str, steal_by: int | None) -> None:
        self.stats["tov"][i] += 1
        self.tov_types[kind] = self.tov_types.get(kind, 0) + 1
        self._log("turnover", by=i, kind=kind)
        if steal_by is not None:
            self.flight = None
            self.holder = steal_by
            self.status = HELD
            self.ball = self.pos[steal_by].copy()
            self.hold_time = 0.0
            self.catch_pos = self.pos[steal_by].copy()
            self.last_pass_from = None
            self._change_possession(team_of(steal_by))
        else:
            self._dead_inbound(1 - team_of(i), self._sideline_spot(self.ball), shot_clock=SHOT_CLOCK, change=True)

    def _violation(self, i: int, kind: str) -> None:
        self._log("violation", by=i, rule=kind)
        self._turnover(i, kind, steal_by=None)

    def _defensive_three(self, i: int) -> None:
        self._log("violation", by=i, rule="defensive_three_seconds")
        self.paint_timer[:] = 0.0
        shooter = max(members(self.off), key=lambda k: self.players[k].outside_scoring.free_throw)
        self.shot_clock = max(self.shot_clock, SHOT_CLOCK_RESET)
        self._free_throw_restart(shooter, 1, fouler=None, technical=True)

    def _foul(self, fouler: int, fouled: int, kind: str, shots: int = 0) -> None:
        t = team_of(fouler)
        self.pf[fouler] += 1
        self.stats["pf"][fouler] += 1
        bonus = self.team_fouls[t] >= 4 or (self.game_clock <= 120 and self.fouls_last2[t] >= 1)
        self.team_fouls[t] += 1
        if self.game_clock <= 120:
            self.fouls_last2[t] += 1
        label = kind + ("_bonus" if kind != "shooting" and bonus else "")
        self.foul_types[label] = self.foul_types.get(label, 0) + 1
        self._log("foul", by=fouler, on=fouled, kind=kind, team_fouls=self.team_fouls[t])
        self.holder = None
        self.flight = None
        if kind == "shooting":
            self._free_throw_restart(fouled, shots, fouler)
        elif bonus:
            self._free_throw_restart(fouled, 2, fouler)
        else:
            self._dead_inbound(self.off, self._sideline_spot(self.pos[fouled]),
                               shot_clock=max(self.shot_clock, SHOT_CLOCK_RESET), change=False)

    def _score(self, i: int, pts: int) -> None:
        t = team_of(i)
        self.score[t] += pts
        self.stats["pts"][i] += pts
        self._log("score", by=i, pts=pts, score=list(self.score))

    def _count_fga(self, i: int, zone: str, three: bool, made: bool) -> None:
        t = team_of(i)
        self.stats["fga"][i] += 1
        self.zone_att[zone][t] += 1
        if three:
            self.stats["fg3a"][i] += 1
        if made:
            self.stats["fgm"][i] += 1
            self.zone_made[zone][t] += 1
            if three:
                self.stats["fg3m"][i] += 1

    # ---------------------------------------------------------- restarts
    def _sideline_spot(self, xy) -> np.ndarray:
        x = float(np.clip(xy[0], 3.0, court.LENGTH - 3.0))
        y = -1.0 if xy[1] < court.WIDTH / 2 else court.WIDTH + 1.0
        return np.array([x, y])

    def _baseline_spot(self, team: int) -> np.ndarray:
        """Inbound spot behind the baseline a team defends (its backcourt)."""
        lx = 19.0 if self.rng.random() < 0.5 else 31.0
        return court.to_global([lx, court.LENGTH + 1.0], self.direction[team])

    def _made_basket_restart(self) -> None:
        new = self.defn
        late = self.game_clock <= 120 and self.quarter >= 4
        self._dead_inbound(new, self._baseline_spot(new), shot_clock=SHOT_CLOCK, change=True,
                           clock_runs=not late, setup=2.2)

    def _dead_inbound(self, team: int, spot, shot_clock: float, change: bool, clock_runs: bool = False,
                      setup: float = 2.0) -> None:
        self.holder = None
        self.flight = None
        self.status = DEAD
        self.paint_timer[:] = 0.0
        if change and team != self.off:
            self.off = team
            self.possessions[team] += 1
            self._log("possession", team=team)
        self.off = team
        mem = list(members(team))
        d = np.linalg.norm(self.pos[mem] - spot, axis=1)
        inbounder = mem[int(np.argmin(d))]
        self.restart = Restart(kind="inbound", team=team, spot=np.asarray(spot, float), inbounder=inbounder,
                               setup=setup, clock_runs=clock_runs, shot_clock=shot_clock)
        self.shot_clock = shot_clock
        self.ball = np.asarray(spot, float).copy()

    def _free_throw_restart(self, shooter: int, n: int, fouler: int | None, technical: bool = False) -> None:
        self.holder = None
        self.flight = None
        self.status = DEAD
        self.paint_timer[:] = 0.0
        self.restart = Restart(kind="free_throws", team=team_of(shooter), shooter=shooter, n_shots=n,
                               setup=1.5, then_inbound=technical)

    def _dead_ball_setup(self) -> None:
        r = self.restart
        if r is None:
            return
        r.timer += DT
        if r.kind == "inbound":
            ib = r.inbounder
            self.target[ib] = r.spot
            if r.timer >= r.setup and float(np.linalg.norm(self.pos[ib] - r.spot)) < 1.5:
                self.status = INBOUND
                self.holder = ib
                self.ball = self.pos[ib].copy()
                self.inbound_timer = 0.0
                self.frontcourt = False
        elif r.kind == "free_throws":
            self._line_up_free_throw(r)
            if r.timer >= r.setup:
                self.status = FREE_THROW
                r.timer = 0.0
        elif r.kind == "jump":
            if r.timer >= r.setup:
                self._jump_ball()

    def _ft_spots(self, r: Restart) -> dict[int, np.ndarray]:
        off = r.team
        loc = {}
        lane_d = [(16.0, 7.5), (34.0, 7.5), (16.0, 14.5)]
        lane_o = [(16.0, 11.0), (34.0, 11.0)]
        back_o = [(25.0, 40.0), (10.0, 30.0)]
        back_d = [(34.0, 14.5), (25.0, 44.0)]
        o = [k for k in members(off) if k != r.shooter]
        o.sort(key=lambda k: -self.players[k].rebounding.offensive_rebound)
        dd = sorted(members(1 - off), key=lambda k: -self.players[k].rebounding.defensive_rebound)
        loc[r.shooter] = court.FT_SPOT_LOCAL
        for k, s in zip(o, lane_o + back_o):
            loc[k] = np.array(s)
        for k, s in zip(dd, lane_d + back_d):
            loc[k] = np.array(s)
        return {k: court.to_global(v, self.direction[off]) for k, v in loc.items()}

    def _line_up_free_throw(self, r: Restart) -> None:
        for k, xy in self._ft_spots(r).items():
            self.target[k] = xy

    def _free_throws(self) -> None:
        r = self.restart
        self._line_up_free_throw(r)
        r.timer += DT
        self.ball = self.pos[r.shooter].copy()
        if r.timer < 1.4:
            return
        r.timer = 0.0
        shooter = r.shooter
        r.attempt += 1
        made = bool(self.rng.random() < resolve.ft_prob(self.players[shooter]))
        self.stats["fta"][shooter] += 1
        self._log("free_throw", by=shooter, n=r.attempt, of=r.n_shots, made=made)
        last = r.attempt >= r.n_shots
        if made:
            self.stats["ftm"][shooter] += 1
            self._score(shooter, 1)
        if not last:
            return
        self.restart = None
        if r.then_inbound:
            self._dead_inbound(r.team, self._sideline_spot(self.pos[shooter]), shot_clock=self.shot_clock,
                               change=False)
        elif made:
            self.off = r.team
            self._made_basket_restart()
        else:
            self.off = r.team
            self.shot_clock = SHOT_CLOCK_RESET
            self._loose_ball(from_xy=self.basket(), dist=(4, 9), duration=0.8, from_rim=True,
                             shot_from=self.pos[shooter], ft_miss=True)

    # ----------------------------------------------------------- periods
    def _start_period(self, first: bool = False) -> None:
        self.status = DEAD
        self.holder = None
        self.flight = None
        self.team_fouls = [0, 0]
        self.fouls_last2 = [0, 0]
        self.shot_clock = SHOT_CLOCK
        q = self.quarter
        if q == 3:
            self.direction = [-1, 1]
        elif q == 1:
            self.direction = [1, -1]
        # line everyone up around the center circle
        mid = np.array([court.LENGTH / 2, court.WIDTH / 2])
        for t in (0, 1):
            back = -self.direction[t]
            for k, i in enumerate(members(t)):
                lateral = [-14.0, 14.0, -20.0, 20.0, 0.0][k] if k < 4 else 0.0
                depth = [10.0, 8.0, 4.0, 4.0, 2.2][k]
                self.pos[i] = mid + np.array([back * depth, lateral if k < 4 else 0.0])
        self.vel[:] = 0.0
        self.target[:] = self.pos
        if q == 1 or q > self.quarters:
            self.restart = Restart(kind="jump", team=0, setup=1.5)
        else:
            loser = 1 - self.jump_winner
            team = loser if q in (2, 3) else self.jump_winner
            self.off = team
            self._dead_inbound(team, self._baseline_spot(team), shot_clock=SHOT_CLOCK, change=False, setup=2.5)
            self.possessions[team] += 1
        self._log("period_start", quarter=q)

    def _jump_ball(self) -> None:
        a, b = self.players[4], self.players[9]
        reach = lambda p: p.height_in + 0.5 * (p.wingspan_in - p.height_in) + p.athleticism.vertical / 10
        win = 0 if self.rng.random() < resolve.sigmoid(0.4 * (reach(a) - reach(b))) else 1
        if self.quarter == 1:
            self.jump_winner = win
        receiver = 0 if win == 0 else 5
        self.restart = None
        self.off = win
        self.possessions[win] += 1
        self.status = HELD
        self.holder = receiver
        self.ball = self.pos[receiver].copy()
        self.hold_time = 0.0
        self.catch_pos = self.pos[receiver].copy()
        self.last_pass_from = None
        self.shot_clock = SHOT_CLOCK
        self.frontcourt = court.in_frontcourt(self.local(self.pos[receiver]))
        self.eight_timer = 0.0
        self._log("jump_ball", won=win)
        self._log("possession", team=win)

    def _end_period(self) -> None:
        self._log("period_end", quarter=self.quarter, score=list(self.score))
        if self.quarter >= self.quarters and self.score[0] != self.score[1]:
            self.done = True
            self.status = DEAD
            self.restart = None
            self._log("game_end", score=list(self.score))
            return
        self.quarter += 1
        self.game_clock = self.quarter_len if self.quarter <= self.quarters else OT_LEN
        self._start_period()

    def _clear_paint_timers(self) -> None:
        self.paint_timer[:] = 0.0

    def _log(self, type_: str, **data) -> None:
        self.events.append(Event(self.t, self.quarter, self.game_clock, type_, data))
