"""Turn cdn.nba.com liveData play-by-play into possessions and shots.

A possession is a maximal run of offensive actions (field goal attempts,
non-technical free throws, turnovers) by one team. A run is cut when the
period ends, when the other team acts, or after the offense scored on its
last shot or free throw. Offensive rebounds keep the run going, so
FGA - OREB + TOV + 0.44 FTA style counts come out the same way the NBA counts.

On-court lineups are tracked from substitutions. Period starters are not
in the feed: period 1 starters come from the box score, later periods are
inferred from who acts (or is subbed out) before being subbed in, the same
way pbpstats does it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from nbasim import court

CLOCK_RE = re.compile(r"PT(?:(\d+)M)?([\d.]+)S")
RETAINED_FT = ("flagrant", "clear path", "clear-path", "away-from-play", "transition take")
SKIP_ACTOR = {"timeout", "period", "game", "memo", "stoppage", "instantreplay", "ejection"}


def clock_seconds(clock: str) -> float:
    """Seconds left in the period, from 'PT11M48.00S' (liveData/v3) or '11:48' (v2)."""
    if clock and ":" in clock:
        mm, ss = clock.split(":")
        return int(mm) * 60 + float(ss)
    m = CLOCK_RE.fullmatch(clock or "")
    if not m:
        return 0.0
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def period_length(period: int) -> float:
    return 720.0 if period <= 4 else 300.0


def elapsed(period: int, remaining: float) -> float:
    before = 4 * 720.0 + (period - 5) * 300.0 if period > 4 else (period - 1) * 720.0
    return before + period_length(period) - remaining


def is_fga(a: dict) -> bool:
    return a.get("actionType") in ("2pt", "3pt")


def ft_kind(a: dict) -> tuple[bool, int, int]:
    """(technical, n, of) for a free throw subType such as '1 of 2', 'flagrant 2 of 2', 'technical'."""
    sub = (a.get("subType") or "").lower()
    tech = "technical" in sub or (a.get("descriptor") or "").lower() == "technical"
    m = re.search(r"(\d) of (\d)", sub)
    n, of = (int(m.group(1)), int(m.group(2))) if m else (1, 1)
    return tech, n, of


def made(a: dict) -> bool:
    return (a.get("shotResult") or "").lower() == "made"


def is_offensive_action(a: dict) -> bool:
    t = a.get("actionType")
    if t in ("2pt", "3pt", "turnover"):
        return bool(a.get("teamId"))
    if t == "freethrow":
        return not ft_kind(a)[0] and bool(a.get("teamId"))
    return False


def shot_zone(a: dict) -> str:
    """rim, short_mid, long_mid, corner3 or above3, matching nbasim.court.shot_zone."""
    three = a.get("actionType") == "3pt"
    xl, yl = a.get("xLegacy"), a.get("yLegacy")
    if xl is not None and yl is not None:
        local = (court.WIDTH / 2 + xl / 10.0, court.BASKET_FROM_BASELINE + yl / 10.0)
        if three:
            return "corner3" if local[1] <= court.THREE_BREAK_LY else "above3"
        d = math.hypot(xl / 10.0, yl / 10.0)
    else:
        area = (a.get("area") or "").lower()
        if three:
            return "corner3" if "corner" in area else "above3"
        d = float(a.get("shotDistance") or 0)
    if d <= court.RIM_ZONE:
        return "rim"
    return "short_mid" if d <= court.SHORT_MID_MAX else "long_mid"


@dataclass
class Possession:
    game_id: str
    period: int
    offense: int
    defense: int
    start: float  # seconds remaining in period
    end: float
    off_lineup: tuple
    def_lineup: tuple
    lineup_changed: bool = False
    points: int = 0
    fga: int = 0
    fgm: int = 0
    fg3a: int = 0
    fg3m: int = 0
    fta: int = 0
    ftm: int = 0
    tov: int = 0
    oreb: int = 0
    start_type: str = ""
    home: bool = False
    # ordered chance-level events: "T|pid" turnover, "F|pid|zone|made" field goal attempt,
    # "FT|pid|n|of|made|tag" free throw (tag a = and-one, r = ball retained), "R|pid" offensive rebound
    events: list = field(default_factory=list)
    shots: list = field(default_factory=list)


class Lineups:
    """Tracks the five players on court for each team through a game."""

    def __init__(self, box: dict):
        self.team_of: dict[int, int] = {}
        self.starters: dict[int, set] = {}
        self.home = box["homeTeam"]["teamId"]
        self.away = box["awayTeam"]["teamId"]
        for side in ("homeTeam", "awayTeam"):
            t = box[side]
            self.starters[t["teamId"]] = set()
            for p in t.get("players", []):
                self.team_of[p["personId"]] = t["teamId"]
                if str(p.get("starter")) == "1":
                    self.starters[t["teamId"]].add(p["personId"])
        self.on: dict[int, set] = {}
        self.warnings: list[str] = []

    def involved(self, a: dict) -> list[int]:
        if a.get("actionType") in SKIP_ACTOR:
            return []
        if a.get("actionType") == "foul" and "technical" in (a.get("subType") or "").lower():
            return []
        ids = [a.get(k) for k in ("personId", "assistPersonId", "stealPersonId", "blockPersonId",
                                  "foulDrawnPersonId", "jumpBallWonPersonId", "jumpBallLostPersonId")]
        return [i for i in ids if i and i in self.team_of]

    def period_starters(self, period: int, actions: list[dict], prev_end: dict[int, set]) -> dict[int, set]:
        if period == 1 and all(len(s) == 5 for s in self.starters.values()):
            return {t: set(s) for t, s in self.starters.items()}
        start = {t: [] for t in self.starters}
        subbed_in: set = set()
        touched: set = set()
        for a in actions:
            if a.get("actionType") == "substitution":
                pid = a.get("personId")
                if pid not in self.team_of:
                    continue
                touched.add(pid)
                if (a.get("subType") or "").lower() == "in":
                    subbed_in.add(pid)
                elif pid not in subbed_in and pid not in start[self.team_of[pid]]:
                    start[self.team_of[pid]].append(pid)
                continue
            for pid in self.involved(a):
                touched.add(pid)
                if pid not in subbed_in and pid not in start[self.team_of[pid]]:
                    start[self.team_of[pid]].append(pid)
        out = {}
        for t, players in start.items():
            if len(players) < 5:
                # someone played the whole period without a recorded action
                for pid in sorted(prev_end.get(t, ())):
                    if len(players) < 5 and pid not in players and pid not in touched:
                        players.append(pid)
            if len(players) != 5:
                self.warnings.append(f"period {period} team {t}: inferred {len(players)} starters")
            out[t] = set(players[:5])
        return out


def possessions_for_game(game_id: str, actions: list[dict], box: dict) -> tuple[list[Possession], dict]:
    """Returns the game's possessions and a check dict (points per team from pbp vs box score)."""
    lu = Lineups(box)
    acts = sorted(actions, key=lambda a: (a.get("period", 0), a.get("orderNumber", a.get("actionNumber", 0))))
    by_period: dict[int, list[dict]] = {}
    for a in acts:
        by_period.setdefault(a.get("period", 0), []).append(a)

    poss: list[Possession] = []
    other = {lu.home: lu.away, lu.away: lu.home}
    team_points = {lu.home: 0, lu.away: 0}
    prev_end: dict[int, set] = {}
    for period in sorted(by_period):
        pacts = by_period[period]
        on = lu.period_starters(period, pacts, prev_end)
        cur: Possession | None = None
        cut = False  # offense scored on its last chance: next action starts a new possession
        last_end = "period_start"
        pending_and1 = None
        after_made_fg = False
        ft_between = False  # inside a free throw trip (after a non-final attempt)
        for a in pacts:
            t = a.get("actionType")
            rem = clock_seconds(a.get("clock"))
            if t == "substitution":
                pid, team = a.get("personId"), lu.team_of.get(a.get("personId"))
                if team is not None:
                    if (a.get("subType") or "").lower() == "out":
                        on[team].discard(pid)
                    else:
                        on[team].add(pid)
                continue
            # points (including technical free throws, which belong to no possession)
            if t in ("2pt", "3pt", "freethrow") and made(a) and a.get("teamId") in team_points:
                team_points[a["teamId"]] += 3 if t == "3pt" else (2 if t == "2pt" else 1)

            if t == "rebound" and cur is not None:
                reb_team = a.get("teamId")
                if ft_between:
                    pass  # placeholder team rebound between free throws of one trip
                elif reb_team == cur.offense:
                    cur.oreb += 1
                    cur.events.append(f"R|{a.get('personId') or 0}")
                elif reb_team:
                    cut = True
                    last_end = "def_rebound"
                continue
            if not is_offensive_action(a):
                continue
            team = a["teamId"]
            if team not in other:
                continue
            new = cur is None or team != cur.offense or (cut and not (
                t == "freethrow" and pending_and1 == team))
            if new:
                start_type = last_end if cur is not None else "period_start"
                cur = Possession(game_id, period, team, other[team], rem, rem,
                                 tuple(sorted(on[team])), tuple(sorted(on[other[team]])),
                                 start_type=start_type, home=team == lu.home)
                poss.append(cur)
                cut, last_end = False, "other"
            elif (tuple(sorted(on[team])) != cur.off_lineup
                  or tuple(sorted(on[other[team]])) != cur.def_lineup):
                cur.lineup_changed = True
            cur.end = rem
            pending_and1 = None
            was_made_fg, after_made_fg = after_made_fg, False
            ft_between = False
            if t in ("2pt", "3pt"):
                cur.fga += 1
                cur.fg3a += t == "3pt"
                shot = {
                    "game_id": game_id, "period": period, "clock": rem,
                    "elapsed": elapsed(period, rem), "team": team, "shooter": a.get("personId"),
                    "made": made(a), "three": t == "3pt", "zone": shot_zone(a),
                    "distance": a.get("shotDistance"), "x": a.get("xLegacy"), "y": a.get("yLegacy"),
                    "area": a.get("area"), "sub_type": a.get("subType"), "descriptor": a.get("descriptor"),
                    "qualifiers": ",".join(a.get("qualifiers") or []),
                    "assist": a.get("assistPersonId"), "block": a.get("blockPersonId"),
                    "off_lineup": tuple(sorted(on[team])), "def_lineup": tuple(sorted(on[other[team]])),
                }
                cur.shots.append(shot)
                cur.events.append(f"F|{a.get('personId')}|{shot['zone']}|{int(made(a))}")
                if made(a):
                    after_made_fg = True
                    cur.fgm += 1
                    cur.fg3m += t == "3pt"
                    cur.points += 3 if t == "3pt" else 2
                    cut, last_end, pending_and1 = True, "made_fg", team
            elif t == "freethrow":
                _, n, of = ft_kind(a)
                cur.fta += 1
                if made(a):
                    cur.ftm += 1
                    cur.points += 1
                # these free throws come with the ball back, so the possession goes on
                kind = f'{a.get("subType") or ""} {a.get("descriptor") or ""}'.lower()
                flagrant = any(k in kind for k in RETAINED_FT)
                tag = "a" if (was_made_fg and n == of == 1) else ("r" if flagrant else "")
                ft_between = n < of
                cur.events.append(f"FT|{a.get('personId')}|{n}|{of}|{int(made(a))}|{tag}")
                if n == of and made(a) and not flagrant:
                    cut, last_end = True, "made_ft"
                else:
                    pending_and1 = team if n < of else None
            elif t == "turnover":
                cur.tov += 1
                cur.events.append(f"T|{a.get('personId') or 0}")
                cut, last_end = True, "turnover"
        if cur is not None and cut and pacts:
            # the other team got the ball but the period ended before it could act
            team = other[cur.offense]
            end = clock_seconds(pacts[-1].get("clock"))
            poss.append(Possession(game_id, period, team, cur.offense, cur.end, end,
                                   tuple(sorted(on[team])), tuple(sorted(on[cur.offense])),
                                   start_type=last_end, home=team == lu.home))
        prev_end = {k: set(v) for k, v in on.items()}

    def box_points(side):
        s = box[side].get("statistics") or {}
        return s.get("points", box[side].get("score"))

    check = {
        "game_id": game_id,
        "home": lu.home, "away": lu.away,
        "pbp_home_pts": team_points[lu.home], "pbp_away_pts": team_points[lu.away],
        "poss_home_pts": sum(p.points for p in poss if p.offense == lu.home),
        "poss_away_pts": sum(p.points for p in poss if p.offense == lu.away),
        "box_home_pts": box_points("homeTeam"), "box_away_pts": box_points("awayTeam"),
        "home_poss": sum(p.offense == lu.home for p in poss),
        "away_poss": sum(p.offense == lu.away for p in poss),
        "bad_lineups": sum(len(p.off_lineup) != 5 or len(p.def_lineup) != 5 for p in poss),
        "warnings": "; ".join(lu.warnings),
    }
    return poss, check
