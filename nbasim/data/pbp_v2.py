"""Convert stats.nba.com playbyplayv2 rows into the liveData action shape that
nbasim.data.pbp.possessions_for_game reads.

v2 has no shot coordinates and no starters, so coordinates come from the
league-wide shotchartdetail (joined on GAME_EVENT_ID) and every period's
starters are inferred from the play-by-play.
"""
from __future__ import annotations

# EVENTMSGTYPE
MADE, MISSED, FT, REBOUND, TURNOVER, FOUL, VIOLATION, SUB, TIMEOUT, JUMP, EJECTION, START, END = range(1, 14)

FT_SUBTYPE = {10: "1 of 1", 11: "1 of 2", 12: "2 of 2", 13: "1 of 3", 14: "2 of 3", 15: "3 of 3",
              16: "technical", 18: "flagrant 1 of 2", 19: "flagrant 2 of 2", 20: "flagrant 1 of 1",
              21: "technical 1 of 2", 22: "technical 2 of 2", 25: "clear path 1 of 2",
              26: "clear path 2 of 2", 27: "flagrant 1 of 3", 28: "flagrant 2 of 3",
              29: "flagrant 3 of 3"}


def _desc(r: dict) -> str:
    return " ".join(d for d in (r["HOMEDESCRIPTION"], r["NEUTRALDESCRIPTION"], r["VISITORDESCRIPTION"]) if d)


def _team(r: dict, n: int, team_ids: set) -> int | None:
    """Team of PLAYERn; for team events (team rebound/turnover) PLAYERn_ID is the team id itself."""
    t = r.get(f"PLAYER{n}_TEAM_ID")
    if t:
        return int(t)
    pid = r.get(f"PLAYER{n}_ID")
    return int(pid) if pid in team_ids else None


def _pid(r: dict, n: int, team_ids: set) -> int | None:
    pid = r.get(f"PLAYER{n}_ID")
    return int(pid) if pid and pid not in team_ids else None


def convert(game_id: str, columns: list[str], rows: list[list], home: int, away: int,
            home_pts: int, away_pts: int, shots: dict | None = None) -> tuple[list[dict], dict]:
    """Returns (actions, box) in the liveData shape. `shots` maps EVENTNUM -> shotchart row dict."""
    team_ids = {home, away}
    shots = shots or {}
    players: dict[int, set] = {home: set(), away: set()}
    acts: list[dict] = []
    order = 0

    def add(**kw):
        nonlocal order
        order += 1
        kw.setdefault("orderNumber", order)
        kw.setdefault("actionNumber", order)
        acts.append(kw)

    prev_ft_nonfinal_miss = False
    for raw in rows:
        r = dict(zip(columns, raw))
        et, sub = r["EVENTMSGTYPE"], r["EVENTMSGACTIONTYPE"]
        base = {"gameId": game_id, "period": r["PERIOD"], "clock": r["PCTIMESTRING"]}
        for n in (1, 2, 3):
            t, pid = _team(r, n, team_ids), _pid(r, n, team_ids)
            if pid and t in players:
                players[t].add(pid)
        desc = _desc(r)
        t1, p1 = _team(r, 1, team_ids), _pid(r, 1, team_ids)
        if et in (MADE, MISSED):
            s = shots.get(r["EVENTNUM"], {})
            three = "3PT" in desc or s.get("SHOT_TYPE", "").startswith("3")
            add(**base, actionType="3pt" if three else "2pt", teamId=t1, personId=p1,
                shotResult="Made" if et == MADE else "Missed",
                assistPersonId=_pid(r, 2, team_ids) if et == MADE else None,
                blockPersonId=_pid(r, 3, team_ids) if et == MISSED else None,
                xLegacy=s.get("LOC_X"), yLegacy=s.get("LOC_Y"), shotDistance=s.get("SHOT_DISTANCE"),
                area=s.get("SHOT_ZONE_BASIC"), subType=s.get("ACTION_TYPE"), description=desc)
            prev_ft_nonfinal_miss = False
        elif et == FT:
            sub_type = FT_SUBTYPE.get(sub, "1 of 1")
            miss = "MISS" in desc.upper()
            add(**base, actionType="freethrow", teamId=t1, personId=p1, subType=sub_type,
                shotResult="Missed" if miss else "Made", description=desc)
            n_of = sub_type.split()[-3:]
            prev_ft_nonfinal_miss = miss and len(n_of) == 3 and n_of[0] != n_of[2]
        elif et == REBOUND:
            # v2 logs a placeholder team rebound after a missed non-final free throw
            if not prev_ft_nonfinal_miss:
                add(**base, actionType="rebound", teamId=t1, personId=p1 or 0, description=desc)
            prev_ft_nonfinal_miss = False
        elif et == TURNOVER:
            if "no turnover" in desc.lower():
                continue
            add(**base, actionType="turnover", teamId=t1, personId=p1,
                stealPersonId=_pid(r, 2, team_ids), description=desc)
        elif et == FOUL:
            tech = "T.FOUL" in desc.upper() or "TECHNICAL" in desc.upper()
            add(**base, actionType="foul", teamId=t1, personId=p1,
                subType="technical" if tech else str(sub),
                foulDrawnPersonId=_pid(r, 2, team_ids), description=desc)
        elif et == SUB:
            add(**base, actionType="substitution", teamId=t1, personId=p1, subType="out")
            add(**base, actionType="substitution", teamId=t1, personId=_pid(r, 2, team_ids), subType="in")
        elif et == JUMP:
            add(**base, actionType="jumpball", teamId=t1, personId=p1,
                jumpBallLostPersonId=_pid(r, 2, team_ids), jumpBallWonPersonId=_pid(r, 3, team_ids))
        elif et == VIOLATION:
            add(**base, actionType="violation", teamId=t1, personId=p1, description=desc)

    box = {"homeTeam": {"teamId": home, "score": home_pts,
                        "players": [{"personId": p, "starter": "0"} for p in sorted(players[home])]},
           "awayTeam": {"teamId": away, "score": away_pts,
                        "players": [{"personId": p, "starter": "0"} for p in sorted(players[away])]}}
    return acts, box
