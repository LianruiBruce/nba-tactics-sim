"""Scripted team AI: runs plays on offense and man-to-man principles on defense.

All tactical reasoning happens in the offense's local frame (see court.py):
lx across the floor, ly distance from the attacked baseline.
"""

from __future__ import annotations

import math

import numpy as np

from .. import court, resolve
from ..engine import (DEAD, HELD, INBOUND, LOOSE, PASS, SHOT, Engine, members, team_of)
from ..playbook import Play, assign_roles, load_playbook, play_fit

RIM_SPOT = np.array([25.0, 6.5])
SPACING_5OUT = [(25, 31), (42.5, 23), (7.5, 23), (3, 3.5), (47, 3.5)]
SPACING_4OUT = [(25, 31), (42.5, 23), (7.5, 23), (3, 3.5), (39, 6)]
TRANSITION = [(25, 33), (46, 5), (4, 5), (14, 30), (25, 9)]
SAFETY = [(25, 52), (36, 48), (14, 48), (20, 40), (30, 40)]
CRASH = [(19, 7), (31, 7), (25, 9)]

# Decision tendencies, tuned with scripts/calibrate.py so that league-average
# lineups reproduce the NBA shot mix and pace.
TEND = {
    "cont_base": 0.76,        # value of keeping the ball with 0 on the shot clock...
    "cont_slope": 0.45,       # ...plus this much (times sqrt of the share left) with a full 24
    "setup_patience": 0.22,   # extra value of holding while a play sets up
    "bias_rim": 0.00,
    "bias_short_mid": 0.00,
    "bias_long_mid": 0.04,
    "bias_three": -0.02,
    "skill_tendency": 0.04,   # players lean toward shots they are good at (per 10 rating)
    "drive_bias": 0.00,
    "pass_cost": 0.06,
    "temp": 0.16,             # per-touch judgment noise (sd), scaled down by pass IQ
    "jitter": 0.015,          # per-decision noise (sd)
    "decision_dt": 0.25,
    "self_bar_exp": 2.0,      # how much a player's own shooting raises the bar for taking a shot
    "ball_movement": 0.06,     # value of moving the ball on to a teammate who has no shot yet
    "redrive_penalty": 0.15,  # per drive already made this possession
    "touch_decay": 0.05,
    "floater_min": 5.0,       # ft from the rim; inside this a driver keeps going at help
    "star_power": 3.0,        # fair usage share grows with scoring quality to this power
    "zone_pref_weight": 0.3,  # pts of option value per unit of log(real shot mix / league mix)
    "usage_weight": 1.5,      # pts of option value per unit of usage share above fair      # per second a touch lasts beyond 2.5 s
}
TEND.update(resolve.load_tuned().get("tendencies", {}))


LEAGUE_PPP = 1.12
_REF_SPOTS = [((25.0, 7.0), 4.0), ((25.0, 15.0), 4.0), ((8.0, 20.0), 5.0), ((25.0, 30.0), 5.0)]


def matchup_quality(shooters, defenders) -> float:
    """Average expected points of a guarded shot from each shooter's best reference spot.

    Used to scale how patient an offense is: a lineup whose ordinary looks are
    worth more than league average waits longer for a good one, so star
    lineups do not end up shooting at twice the NBA pace.
    """
    vals = []
    for sh in shooters:
        best = 0.0
        for loc, dist in _REF_SPOTS:
            p = np.mean([resolve.make_prob(sh, loc, d, dist, off_dribble=False)[0] for d in defenders])
            best = max(best, p * (3 if court.is_three(loc) else 2))
        vals.append(best)
    return float(np.mean(vals))


_REF_QUALITY: list[float] = []


def reference_quality() -> float:
    if not _REF_QUALITY:
        from ..players import DATA_DIR, load_roster
        avg = load_roster(DATA_DIR / "players" / "calibration.yaml").players
        _REF_QUALITY.append(matchup_quality(avg, avg))
    return _REF_QUALITY[0]


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.zeros(2)


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-9), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


class TeamBrain:
    def __init__(self, scheme: str = "drop", seed: int | None = None, playbook: dict[str, Play] | None = None):
        self.scheme = scheme
        self.rng = np.random.default_rng(seed)
        self.playbook = playbook or load_playbook()
        self.poss_key = None
        self.team = 0
        self._reset_offense()
        self.match = list(range(5))
        self.switch_until = -1.0
        self.drop_until = -1.0
        self.helper: int | None = None
        self.manual: set[int] = set()  # players whose live-ball decisions come from outside (RL)
        self.quality: float | None = None

    # ----------------------------------------------------------------- api
    def act(self, eng: Engine, team: int) -> None:
        self.team = team
        if self.quality is None:
            mine = [eng.players[i] for i in members(team)]
            theirs = [eng.players[i] for i in members(1 - team)]
            self.quality = matchup_quality(mine, theirs) / reference_quality()
            # a scorer's ordinary look is better than an average player's, so the
            # bar for "a good shot" is higher for him too
            indiv = {i: matchup_quality([eng.players[i]], theirs) / reference_quality() for i in members(team)}
            self.self_bar = {i: q ** TEND["self_bar_exp"] for i, q in indiv.items()}
            # the share of the team's possessions each player "should" use: better
            # scorers get more, but nobody gets the ball every time
            w = {i: q ** TEND["star_power"] for i, q in indiv.items()}
            if all(eng.players[i].usage for i in members(team)):
                # measured from real games: how often each player ends a possession with a shot
                w = {i: eng.players[i].usage for i in members(team)}
            self.fair_usage = {i: v / sum(w.values()) for i, v in w.items()}
        key = (sum(eng.possessions), eng.off)
        if key != self.poss_key:
            self.poss_key = key
            self._new_possession(eng)
        if eng.status == DEAD and eng.restart is not None and eng.restart.kind == "jump":
            for i in members(team):
                eng.set_target(i, eng.pos[i])
            return
        if eng.off == team:
            self._offense(eng)
        else:
            self._defense(eng)

    def _usage_gap(self, eng: Engine, i: int) -> float:
        """How far above his fair share of the team's shots player i is so far (prior: fair share)."""
        st = eng.stats
        mine = list(members(self.team))
        use = {j: st["fga"][j] + 0.44 * st["fta"][j] + st["tov"][j] for j in mine}
        prior = 15.0
        total = sum(use.values()) + prior
        return (use[i] + prior * self.fair_usage[i]) / total - self.fair_usage[i]

    def _patience(self, eng: Engine) -> float:
        """How good this offense's ordinary looks are against this defense (1.0 = league average).

        A fixed matchup estimate: feeding back the running score made star
        lineups slower and slower as the game went on.
        """
        return self.quality

    # ------------------------------------------------------------ shared
    def _new_possession(self, eng: Engine) -> None:
        self._reset_offense()
        self.possession_t = eng.t
        if eng.status in (DEAD, INBOUND):
            self.match = list(range(5))
        else:
            # live-ball change: pick up whoever is nearest, starting with the rim protector
            mine = list(members(self.team))
            theirs = list(members(1 - self.team))
            free = list(range(5))
            match = [0] * 5
            order = sorted(range(5), key=lambda k: -mine[k])
            for k in order:
                d = [float(np.linalg.norm(eng.pos[mine[k]] - eng.pos[theirs[s]])) + 6.0 * (s != k) for s in free]
                s = free[int(np.argmin(d))]
                match[k] = s
                free.remove(s)
            self.match = match

    def _reset_offense(self) -> None:
        self.play: Play | None = None
        self.roles: dict[str, int] = {}
        self.phase_i = 0
        self.phase_t = 0.0
        self.role_spec: dict[str, dict] = {}
        self.freelance = False
        self.next_decision = 0.0
        self.driving = False
        self.drive_stall = 0.0
        self.drives = 0
        self.last_drive_t = -99.0
        self.last_passer: int | None = None
        self.last_pass_t = -99.0
        self.cut_until: dict[int, float] = {}
        self.spot_override: dict[int, np.ndarray] = {}
        self.possession_t = 0.0
        self.pass_target_role: str | None = None
        self.touch = None
        self.touch_noise: dict[str, float] = {}

    # ============================================================ offense
    def _offense(self, eng: Engine) -> None:
        team = self.team
        mem = list(members(team))
        L = lambda xy: eng.local(xy, team)
        G = lambda loc: eng.glob(np.asarray(loc, float), team)

        if eng.status in (DEAD, INBOUND):
            self._inbound(eng, mem, L, G)
            return
        if eng.status == SHOT:
            self._after_shot(eng, mem, L, G, chase=None)
            return
        if eng.status == LOOSE:
            self._after_shot(eng, mem, L, G, chase=eng.flight.end if eng.flight else None)
            return

        holder = eng.holder if (eng.holder is not None and team_of(eng.holder) == team) else None
        if eng.status == PASS and eng.flight is not None:
            recv = eng.flight.dst
            for i in mem:
                if i == recv:
                    eng.set_target(i, eng.pos[i] + _unit(eng.ball - eng.pos[i]) * 1.0)
            self._off_ball_targets(eng, mem, L, G, holder=None, skip={recv})
            self._avoid_three_seconds(eng, mem, None, L, G)
            return
        if holder is None:
            return

        hl = L(eng.pos[holder])
        if not eng.frontcourt:
            eng.labels[team] = "推进"
            self._advance(eng, mem, holder, hl, L, G)
            return

        if self.play is None and not self.freelance:
            early = eng.t - self.possession_t < 7.0 and self._advantage(eng, holder, L)
            if early or min(eng.shot_clock, eng.game_clock) < 12.0:
                self.freelance = True
            else:
                self._call_play(eng, holder)
        if self.play is not None:
            self._run_play(eng, mem, holder, L, G)
        else:
            eng.labels[team] = "快攻" if eng.t - self.possession_t < 7.0 else "自由进攻"
            self._off_ball_targets(eng, mem, L, G, holder=holder, skip={holder})
            if not self.driving:
                eng.set_target(holder, eng.pos[holder])
        self._ball_handler(eng, holder, L, G)
        self._avoid_three_seconds(eng, mem, holder, L, G)

    def _avoid_three_seconds(self, eng, mem, holder, L, G) -> None:
        for i in mem:
            if i == holder or eng.paint_timer[i] < 1.2:
                continue
            eng.set_target(i, G(self._paint_exit(L(eng.pos[i]))))

    @staticmethod
    def _paint_exit(loc) -> np.ndarray:
        """Nearest spot just outside the paint from a local position."""
        left = court.BASKET_LOCAL[0] - court.PAINT_HALF_WIDTH - 1.5
        right = court.BASKET_LOCAL[0] + court.PAINT_HALF_WIDTH + 1.5
        top = court.PAINT_LENGTH + 1.5
        cands = [np.array([left, loc[1]]), np.array([right, loc[1]]), np.array([loc[0], top])]
        return min(cands, key=lambda c: float(np.linalg.norm(c - loc)))

    # ---------------------------------------------------------- phases
    def _inbound(self, eng, mem, L, G) -> None:
        r = eng.restart
        if r is None or r.kind != "inbound":
            return
        ib = r.inbounder
        spot = L(r.spot)
        others = [i for i in mem if i != ib]
        handler = max(others, key=lambda i: eng.players[i].playmaking.ball_handle)
        eng.labels[self.team] = "界外球"
        backcourt = spot[1] > court.HALF
        if backcourt:
            recv = np.array([np.clip(spot[0], 12, 38), spot[1] - 12.0])
        else:
            recv = np.array([np.clip(spot[0], 8, 42), np.clip(spot[1], 8, 40)])
            recv = recv + _unit(np.array([25.0, 20.0]) - recv) * 9.0
        spacing = self._spacing(eng)
        for i in others:
            if i == handler:
                eng.set_target(i, G(recv))
            else:
                k = i % 5
                tgt = np.array(spacing[k]) if not backcourt else np.array(TRANSITION[k]) + [0, 14]
                eng.set_target(i, G(tgt))
        if eng.status == INBOUND and eng.holder == ib and eng.inbound_timer > 0.8:
            best, best_open = None, -1.0
            for i in others:
                lane = min(_seg_dist(eng.pos[d], eng.pos[ib], eng.pos[i]) for d in members(1 - self.team))
                near = min(float(np.linalg.norm(eng.pos[d] - eng.pos[i])) for d in members(1 - self.team))
                score = min(lane, near) + (3.0 if i == handler else 0.0)
                if score > best_open:
                    best, best_open = i, score
            if best_open > 4.0 or eng.inbound_timer > 3.2:
                eng.request_pass(ib, best)
                self.last_passer, self.last_pass_t = ib, eng.t

    def _advance(self, eng, mem, holder, hl, L, G) -> None:
        handler = max(mem, key=lambda i: eng.players[i].playmaking.ball_handle)
        # bigs who rebound look for the outlet
        if holder != handler and holder not in self.manual and eng.hold_time > 0.4 and holder % 5 >= 3:
            if self._lane_ok(eng, holder, handler) and float(np.linalg.norm(eng.pos[handler] - eng.pos[holder])) < 45:
                eng.request_pass(holder, handler)
                self.last_passer, self.last_pass_t = holder, eng.t
                return
        push = self._advantage(eng, holder, L) and eng.t - self.possession_t < 8.0
        pace = None if push else 9.5
        eng.set_target(holder, G(np.array([25 + 0.4 * (hl[0] - 25), 30.0])), speed=pace)
        for i in mem:
            if i == holder:
                continue
            eng.set_target(i, G(TRANSITION[i % 5]), speed=None if push else 13.0)
        if hl[1] < 60 and push:
            self._ball_handler(eng, holder, L, G, transition=True)

    def _advantage(self, eng, holder, L) -> bool:
        hl = L(eng.pos[holder])
        attackers = sum(1 for i in members(self.team) if L(eng.pos[i])[1] <= hl[1] + 4)
        back = sum(1 for d in members(1 - self.team) if L(eng.pos[d])[1] < hl[1])
        return attackers > back

    def _after_shot(self, eng, mem, L, G, chase) -> None:
        shooter = eng.flight.src if (eng.flight is not None and eng.flight.kind == "shot") else None
        crashers = sorted(mem, key=lambda i: -(eng.players[i].rebounding.offensive_rebound
                                              - 0.4 * court.dist_to_basket(L(eng.pos[i]))))[:2]
        for n, i in enumerate(mem):
            if chase is not None and float(np.linalg.norm(eng.pos[i] - chase)) < 12.0:
                eng.set_target(i, chase)
            elif i in crashers or i == shooter and court.dist_to_basket(L(eng.pos[i])) < 10:
                eng.set_target(i, G(CRASH[crashers.index(i)] if i in crashers else CRASH[2]))
            else:
                eng.set_target(i, G(SAFETY[i % 5]))
        eng.labels[self.team] = "冲抢/退防"

    # ------------------------------------------------------------- plays
    def _call_play(self, eng, holder) -> None:
        players = [eng.players[i] for i in members(self.team)]
        names = list(self.playbook)
        w = np.array([max(0.05, play_fit(self.playbook[n], players)) for n in names])
        self.play = self.playbook[str(self.rng.choice(names, p=w / w.sum()))]
        self.roles = assign_roles(self.play, players, holder % 5)
        self.phase_i = 0
        self.phase_t = eng.t
        self.role_spec = {}
        self.pass_target_role = None
        eng._log("play_call", team=self.team, play=self.play.name)

    def _role_player(self, role: str) -> int:
        return self.roles[role] + (0 if self.team == 0 else 5)

    def _run_play(self, eng, mem, holder, L, G) -> None:
        play = self.play
        phase = play.phases[self.phase_i]
        for role, spec in phase.roles.items():
            self.role_spec[role] = spec
        self.pass_target_role = phase.pass_to
        eng.labels[self.team] = f"{play.label}·{phase.id}"
        elapsed = eng.t - self.phase_t

        targets: dict[int, np.ndarray] = {}
        in_place = True
        screen_set = False
        for role, spec in self.role_spec.items():
            i = self._role_player(role)
            if i == holder and self.driving:
                continue
            tgt, done = self._role_target(eng, i, holder, spec, L, G)
            if tgt is None:
                continue
            targets[i] = tgt
            if "to" in spec and not done:
                in_place = False
            if "screen" in spec and done:
                screen_set = True
        for i, tgt in targets.items():
            eng.set_target(i, tgt)
        others = [i for i in mem if i not in targets and i != holder]
        if others:
            self._off_ball_targets(eng, others, L, G, holder=holder, skip={holder})

        advance = elapsed >= phase.max_time
        if phase.until == "in_place" and in_place and elapsed > 0.6:
            advance = True
        if phase.until == "screen_set" and screen_set and elapsed > 0.4:
            advance = True
        if advance:
            self.phase_i += 1
            self.phase_t = eng.t
            if self.phase_i >= len(play.phases):
                self.play = None
                self.freelance = True

    def _role_target(self, eng, i, holder, spec, L, G):
        """Target for one role in the current phase, and whether it has reached it."""
        opp = list(members(1 - self.team))
        if "to" in spec:
            tgt = G(court.REGIONS[spec["to"]])
            return tgt, float(np.linalg.norm(eng.pos[i] - tgt)) < 3.0
        if spec.get("hold"):
            return eng.pos[i], True
        if "screen" in spec:
            mate = self._role_player(spec["screen"])
            d = opp[int(np.argmin(np.linalg.norm(eng.pos[opp] - eng.pos[mate], axis=1)))]
            ml = L(eng.pos[mate])
            side = spec.get("side", "right")
            if side == "right":
                away = np.array([1.0, -0.3])
            elif side == "left":
                away = np.array([-1.0, -0.3])
            elif side == "out":
                away = _unit(np.array([ml[0] - 25.0, 0.0]) + [0, 0.5])
            else:  # "in": screen toward the basket side
                away = _unit(court.BASKET_LOCAL - ml)
            tgt_local = L(eng.pos[d]) + _unit(away) * 2.6
            tgt = G(tgt_local)
            return tgt, float(np.linalg.norm(eng.pos[i] - tgt)) < 2.0
        if "roll" in spec or "pop" in spec:
            shooter = eng.players[i].outside_scoring.three_point >= 80
            where = spec["pop"] if (shooter and "pop" in spec) else spec.get("roll", "rim")
            tgt = G(court.REGIONS[where])
            return tgt, False
        if "drive" in spec and i == holder:
            tgt = G(court.REGIONS[spec["drive"]])
            return tgt, float(np.linalg.norm(eng.pos[i] - tgt)) < 3.0
        if spec.get("post_up") and i == holder:
            hl = L(eng.pos[i])
            tgt_local = hl + _unit(np.array([25.0, 6.0]) - hl) * 2.0
            return G(tgt_local), False
        if spec.get("iso") and i == holder:
            return eng.pos[i], False
        return None, True

    # ------------------------------------------------------- off-ball
    def _spacing(self, eng) -> list:
        c = eng.players[4 if self.team == 0 else 9]
        return SPACING_5OUT if c.outside_scoring.three_point >= 75 else SPACING_4OUT

    def _off_ball_targets(self, eng, idx, L, G, holder, skip) -> None:
        spacing = self._spacing(eng)
        used = set()
        if holder is not None:
            hl = L(eng.pos[holder])
            k_h = int(np.argmin(np.linalg.norm(np.array(spacing) - hl, axis=1)))
            used.add(k_h)
        for i in idx:
            if i in skip:
                continue
            k = i % 5
            spot = np.array(spacing[k])
            if k in used:
                free = [s for s in range(5) if s not in used]
                k = min(free, key=lambda s: np.linalg.norm(np.array(spacing[s]) - L(eng.pos[i])))
                spot = np.array(spacing[k])
            used.add(k)
            tgt = G(spot)
            # backdoor cut when the defender ball-watches
            if eng.t < self.cut_until.get(i, -1):
                tgt = G(RIM_SPOT)
            elif holder is not None and self.rng.random() < 0.01:
                opp = list(members(1 - self.team))
                d = opp[int(np.argmin(np.linalg.norm(eng.pos[opp] - eng.pos[i], axis=1)))]
                gap = float(np.linalg.norm(eng.pos[d] - eng.pos[i]))
                watching = float(np.linalg.norm(eng.pos[d] - eng.ball)) < float(np.linalg.norm(eng.pos[i] - eng.ball))
                if gap > 5.0 and watching and i % 5 != 4:
                    self.cut_until[i] = eng.t + 1.6
                    eng._log("cut", by=i)
            eng.set_target(i, tgt, speed=None if eng.t < self.cut_until.get(i, -1) else 14.0)

    # ----------------------------------------------------- ball handler
    def _lane_ok(self, eng, a, b, width=2.0) -> bool:
        return all(_seg_dist(eng.pos[d], eng.pos[a], eng.pos[b]) >= width for d in members(1 - self.team))

    def _drive_target(self, eng, h, L, G) -> np.ndarray:
        """Attack the rim, angling away from a defender standing in the lane."""
        hl = L(eng.pos[h])
        to_rim = RIM_SPOT - hl
        tgt = RIM_SPOT.copy()
        for d in members(1 - self.team):
            dl = L(eng.pos[d])
            rel_v = dl - hl
            ahead = float(rel_v @ _unit(to_rim))
            if 0 < ahead < 8.0 and _seg_dist(dl, hl, RIM_SPOT) < 3.0:
                u = _unit(to_rim)
                side = np.sign(u[0] * rel_v[1] - u[1] * rel_v[0]) or 1.0
                perp = np.array([-to_rim[1], to_rim[0]])
                tgt = tgt - _unit(perp) * side * 4.0
                break
        return G(tgt)

    def _help_arrived(self, eng, h, L) -> bool:
        """A defender has stepped into the driving lane short of the rim: time for a floater, pull-up or kick."""
        hl = L(eng.pos[h])
        rim_d = court.dist_to_basket(hl)
        if not (TEND["floater_min"] < rim_d < 14.0):
            return False
        for d in members(1 - self.team):
            dl = L(eng.pos[d])
            ahead = float((dl - hl) @ _unit(RIM_SPOT - hl))
            if 0 < ahead < 5.0 and _seg_dist(dl, hl, RIM_SPOT) < 2.5:
                return True
        return False

    def _ball_handler(self, eng, h, L, G, transition: bool = False) -> None:
        """Pick shoot / pass / drive / keep running the action, by expected points.

        Each option is valued in expected points plus a tendency term (TEND);
        keeping the ball is worth what the rest of the possession is expected
        to yield, which shrinks as the shot clock runs down.
        """
        if h in self.manual:
            return
        hl = L(eng.pos[h])
        rim_d = court.dist_to_basket(hl)
        if self.driving:
            eng.set_target(h, self._drive_target(eng, h, L, G))
            spd = float(np.linalg.norm(eng.vel[h]))
            self.drive_stall = self.drive_stall + 0.1 if spd < 4.0 else 0.0
            if rim_d < 4.0 or self.drive_stall > 0.8 or self._help_arrived(eng, h, L):
                self.next_decision = 0.0
        if eng.t < self.next_decision:
            return
        self.next_decision = eng.t + TEND["decision_dt"]
        p = eng.players[h]
        sc = min(eng.shot_clock, eng.game_clock)
        phase_id = self.play.phases[self.phase_i].id if self.play is not None else None
        setting_up = phase_id in ("setup", "screen", "clear", "curl", "entry")
        drive_over = self.driving and (rim_d < 4.0 or self.drive_stall > 0.8 or self._help_arrived(eng, h, L))
        stuck_in_paint = court.in_paint(hl) and eng.paint_timer[h] > 1.4

        options: dict[str, float] = {}
        if sc > 2.5 and not drive_over and not stuck_in_paint:
            hold = (TEND["cont_base"] + TEND["cont_slope"] * math.sqrt(min(sc, 24.0) / 24.0)) * self._patience(eng) * self.self_bar[h]
            if setting_up:
                hold += TEND["setup_patience"]
            # holding the ball too long kills the possession (NBA average touch is under 3 s)
            hold -= TEND["touch_decay"] * max(0.0, eng.hold_time - 2.5)
            if transition:
                hold -= 0.08
            options["hold"] = hold

        ep_shot = eng.shot_value(h)
        zone = court.shot_zone(hl)
        bias = TEND["bias_three"] if zone in ("corner3", "above3") else TEND[f"bias_{zone}"]
        skill = resolve.rel(resolve.zone_skill(p, zone)) * TEND["skill_tendency"]
        if p.zone_pref:
            skill += TEND["zone_pref_weight"] * p.zone_pref[zone]
        hog = TEND["usage_weight"] * self._usage_gap(eng, h)
        options["shoot"] = ep_shot + bias + skill + (0.2 if sc <= 2.5 else 0.0) - hog

        best_j, best_val = None, -9.0
        for j in members(self.team):
            if j == h:
                continue
            jl = L(eng.pos[j])
            if eng.frontcourt and not court.in_frontcourt(jl):
                continue
            if j == self.last_passer and eng.t - self.last_pass_t < 1.5:
                continue
            lane = min(_seg_dist(eng.pos[d], eng.pos[h], eng.pos[j]) for d in members(1 - self.team))
            p_safe = 0.55 if lane < 1.5 else (0.85 if lane < 3.0 else 0.98)
            ep_j = eng.shot_value(j, at=eng.pos[j])
            near = min(float(np.linalg.norm(eng.pos[d] - eng.pos[j])) for d in members(1 - self.team))
            if near > 7.0:
                ep_j = max(ep_j, 1.10)
            # a pass is worth the receiver's shot, or keeping the possession alive in his hands
            keep_j = (TEND["cont_base"] + TEND["cont_slope"] * math.sqrt(max(sc - 1.0, 0.0) / 24.0)) \
                * self._patience(eng) * self.self_bar[j]
            val = max(ep_j, keep_j + TEND["ball_movement"]) * p_safe - 0.003 * float(np.linalg.norm(eng.pos[j] - eng.pos[h]))
            if self.pass_target_role is not None and j == self._role_player(self.pass_target_role) and lane >= 2.0:
                val += 0.35
            val -= TEND["usage_weight"] * self._usage_gap(eng, j)
            if val > best_val:
                best_j, best_val = j, val
        if best_j is not None and sc > 1.0:
            options["pass"] = best_val - TEND["pass_cost"]

        if rim_d > 9.0 and not self.driving and rim_d < 32 and sc > 2.0:
            lane_open = all(_seg_dist(eng.pos[d], eng.pos[h], G(RIM_SPOT)) > 3.5 or
                            court.dist_to_basket(L(eng.pos[d])) > rim_d for d in members(1 - self.team))
            ep_rim = eng.shot_value(h, at=G(RIM_SPOT))
            if lane_open:
                ep_drive = ep_rim * 0.95
            else:
                d_on = min(members(1 - self.team), key=lambda d: float(np.linalg.norm(eng.pos[d] - eng.pos[h])))
                dp = eng.players[d_on]
                quick = 0.5 * (p.playmaking.ball_handle + p.playmaking.speed_with_ball) + 0.3 * p.athleticism.speed
                stop = 0.5 * (dp.defense.perimeter_defense + dp.athleticism.agility) + 0.3 * dp.athleticism.speed
                p_beat = resolve.sigmoid((quick - stop) / 12.0 - 0.4)
                ep_drive = p_beat * max(ep_rim, 1.15) + (1 - p_beat) * 0.85
            drive_skill = resolve.rel(0.5 * (p.inside_scoring.layup + p.playmaking.ball_handle)) * TEND["skill_tendency"]
            if p.zone_pref:
                drive_skill += TEND["zone_pref_weight"] * p.zone_pref["rim"]
            # one good drive per trip is normal (NBA: ~0.5 per possession); re-attacking
            # the same set defense right after a failed drive rarely is
            redrive = TEND["redrive_penalty"] * self.drives
            if (not setting_up or ep_drive > 1.3) and eng.t - self.last_drive_t > 2.0:
                options["drive"] = ep_drive + TEND["drive_bias"] + drive_skill - redrive - hog
        elif self.driving and not drive_over:
            options["drive"] = options.get("hold", 0.9)

        # Judgment noise is drawn once per touch (a player's read of the floor),
        # plus a small per-decision jitter; redrawing it every 0.25 s would turn
        # rare misreads into near-certain ones over a long possession.
        touch = (h, eng.last_catch_t, self.poss_key)
        if touch != self.touch:
            self.touch = touch
            temp = TEND["temp"] * (1.3 - (p.playmaking.pass_iq - 25) / 74)
            self.touch_noise = {k: self.rng.normal(0, temp) for k in ("hold", "shoot", "pass", "drive")}
        noisy = {k: v + self.touch_noise[k] + self.rng.normal(0, TEND["jitter"]) for k, v in options.items()}
        choice = max(noisy, key=noisy.get)

        if choice == "shoot":
            self.driving = False
            eng.request_shot(h)
        elif choice == "pass":
            self.driving = False
            eng.request_pass(h, best_j)
            self.last_passer, self.last_pass_t = h, eng.t
        elif choice == "drive":
            if not self.driving:
                eng.start_drive(h)
                self.drives += 1
                self.last_drive_t = eng.t
            self.driving = True
            eng.set_target(h, self._drive_target(eng, h, L, G))
        else:
            self.driving = False

    # ============================================================ defense
    def _defense(self, eng: Engine) -> None:
        team = self.team
        mine = list(members(team))
        theirs = list(members(1 - team))
        off = eng.off
        L = lambda xy: eng.local(xy, off)
        G = lambda loc: eng.glob(np.asarray(loc, float), off)
        bl = court.BASKET_LOCAL
        ball_l = L(eng.ball)
        holder = eng.holder if (eng.holder is not None and team_of(eng.holder) == off) else None
        eng.labels[team] = "人盯人·" + ("换防" if self.scheme == "switch" else "Drop")

        if eng.status in (SHOT, LOOSE):
            chase = eng.flight.end if (eng.status == LOOSE and eng.flight is not None) else None
            for k, d in enumerate(mine):
                m = theirs[self.match[k]]
                if chase is not None and float(np.linalg.norm(eng.pos[d] - chase)) < 11.0:
                    eng.set_target(d, chase)
                    continue
                ml = L(eng.pos[m])
                eng.set_target(d, G(ml + _unit(bl - ml) * 2.2))
            eng.labels[team] = "卡位"
            return

        if holder is not None and eng.status == HELD and eng.frontcourt:
            self._screen_coverage(eng, mine, theirs, holder, L)
        recv = eng.flight.dst if (eng.status == PASS and eng.flight is not None and eng.flight.kind == "pass") else None

        targets = {}
        for k, d in enumerate(mine):
            m = theirs[self.match[k]]
            dp = eng.players[d]
            react = 0.45 - 0.30 * (0.5 * (dp.defense.perimeter_defense + dp.athleticism.agility) - 25) / 74
            if eng.t < eng.beaten_until[d]:
                react += 0.5
            ml = L(eng.seen_pos(m, react))
            if m == recv:
                # close out on the catch while the ball is in the air
                hiq = (dp.defense.help_defense_iq - 25) / 74
                closeout = L(eng.pos[m]) + _unit(bl - L(eng.pos[m])) * (3.5 - 1.0 * hiq)
                targets[d] = closeout
                continue
            if m == holder:
                gap = 3.5 if ml[1] < 30 else (5.0 if ml[1] < 47 else 8.0)
                if eng.players[m].outside_scoring.three_point >= 88 and court.dist_to_basket(ml) < 30:
                    gap = 3.0
                t = ml + _unit(bl - ml) * gap
                t[1] = min(t[1], 52.0)
            else:
                dbm = float(np.linalg.norm(ml - ball_l))
                hiq = (eng.players[d].defense.help_defense_iq - 25) / 74
                a = float(np.clip(0.06 + (dbm - 10) * 0.012, 0.04, 0.42)) * (0.75 + 0.5 * hiq)
                t = ml + a * (ball_l - ml) + _unit(bl - ml) * 3.0
                t[1] = min(t[1], 42.0)
            targets[d] = t

        if holder is not None and eng.t < self.drop_until:
            k_s = self._drop_defender
            hl = L(eng.pos[holder])
            if hl[1] < 32:
                targets[mine[k_s]] = bl + _unit(hl - bl) * 10.0

        self.helper = None
        if holder is not None and eng.frontcourt:
            hl = L(eng.pos[holder])
            rim_d = court.dist_to_basket(hl)
            k_on = self.match.index(holder % 5)
            d_on = mine[k_on]
            if rim_d < 15 and court.dist_to_basket(L(eng.pos[d_on])) > rim_d + 1.0:
                spot = hl + _unit(bl - hl) * 4.0
                cands = [d for d in mine if d != d_on]
                helper = min(cands, key=lambda d: float(np.linalg.norm(L(eng.pos[d]) - spot))
                             - 3.0 * (eng.players[d].defense.help_defense_iq - 70) / 10)
                targets[helper] = spot
                self.helper = helper
                eng.labels[team] += "·协防"

        for d, t in targets.items():
            if eng.paint_timer[d] > 1.5 and court.in_paint(t):
                t = self._paint_exit(L(eng.pos[d]))
            eng.set_target(d, G(t))

    def _screen_coverage(self, eng, mine, theirs, holder, L) -> None:
        k_on = self.match.index(holder % 5)
        d_on = mine[k_on]
        for s in theirs:
            if s == holder:
                continue
            if float(np.linalg.norm(eng.pos[s] - eng.pos[d_on])) > 4.0:
                continue
            if float(np.linalg.norm(eng.pos[s] - eng.pos[holder])) > 12.0:
                continue
            k_s = self.match.index(s % 5)
            if self.scheme == "switch":
                if eng.t > self.switch_until:
                    self.match[k_on], self.match[k_s] = self.match[k_s], self.match[k_on]
                    self.switch_until = eng.t + 2.5
                    eng._log("switch", team=self.team, a=mine[k_on], b=mine[k_s])
            else:
                if eng.t > self.drop_until:
                    eng._log("drop", team=self.team, by=mine[k_s])
                self.drop_until = eng.t + 2.5
                self._drop_defender = k_s
            return
