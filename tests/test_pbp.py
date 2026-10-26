from nbasim.data.pbp import clock_seconds, possessions_for_game

H, A = 1, 2
HOME = [11, 12, 13, 14, 15]
AWAY = [21, 22, 23, 24, 25]


def box():
    def team(tid, ids, score):
        players = [{"personId": p, "starter": "1"} for p in ids] + [{"personId": ids[0] + 5, "starter": "0"}]
        return {"teamId": tid, "score": score, "players": players}
    return {"homeTeam": team(H, HOME, 6), "awayTeam": team(A, AWAY, 3)}


def act(n, t, team=None, pid=None, sub=None, result=None, clock="PT11M00.00S", **kw):
    return {"actionNumber": n, "orderNumber": n, "period": 1, "clock": clock, "actionType": t,
            "teamId": team, "personId": pid, "subType": sub, "shotResult": result, **kw}


def test_clock():
    assert clock_seconds("PT11M48.00S") == 708.0
    assert clock_seconds("PT00M05.30S") == 5.3


def test_possessions_and_lineups():
    acts = [
        act(1, "period", sub="start", clock="PT12M00.00S"),
        act(2, "2pt", H, 11, result="Missed", clock="PT11M40.00S"),
        act(3, "rebound", H, 15, sub="offensive", clock="PT11M38.00S"),
        act(4, "2pt", H, 15, result="Made", clock="PT11M37.00S", xLegacy=0, yLegacy=10),
        act(5, "foul", A, 21, sub="shooting", clock="PT11M37.00S"),
        act(6, "freethrow", H, 15, sub="1 of 1", result="Made", clock="PT11M37.00S"),   # and-one
        act(7, "3pt", A, 22, result="Made", clock="PT11M20.00S", xLegacy=-220, yLegacy=20),
        act(8, "substitution", H, 12, sub="out", clock="PT11M10.00S"),
        act(9, "substitution", H, 16, sub="in", clock="PT11M10.00S"),
        act(10, "turnover", H, 16, sub="bad pass", clock="PT11M05.00S"),
        act(11, "2pt", A, 23, result="Missed", clock="PT10M50.00S"),
        act(12, "rebound", H, 13, sub="defensive", clock="PT10M48.00S"),
        act(13, "freethrow", A, 24, sub="technical", result="Missed", clock="PT10M40.00S"),
        act(14, "3pt", H, 16, result="Made", clock="PT10M30.00S"),
    ]
    poss, check = possessions_for_game("g", acts, box())
    assert [p.offense for p in poss] == [H, A, H, A, H]
    p0 = poss[0]
    assert (p0.fga, p0.oreb, p0.fta, p0.points) == (2, 1, 1, 3)
    assert p0.shots[1]["zone"] == "rim"
    assert poss[1].shots[0]["zone"] == "corner3"
    assert 16 in poss[2].off_lineup and 12 not in poss[2].off_lineup
    assert poss[2].tov == 1
    assert poss[4].start_type == "def_rebound"
    assert check["pbp_home_pts"] == check["box_home_pts"] == 6
    assert check["pbp_away_pts"] == check["box_away_pts"] == 3
    assert check["bad_lineups"] == 0


def test_later_period_starters_inferred():
    acts = [
        act(1, "2pt", H, 11, result="Made", clock="PT11M40.00S"),
        act(2, "substitution", H, 11, sub="out", clock="PT11M00.00S"),
        act(3, "substitution", H, 16, sub="in", clock="PT11M00.00S"),
        act(4, "2pt", A, 21, result="Missed", clock="PT10M40.00S"),
        act(5, "rebound", H, 16, sub="defensive", clock="PT10M38.00S"),
    ]
    q2 = [dict(a, period=2) for a in [
        act(20, "2pt", H, 12, result="Missed", clock="PT11M40.00S"),
        act(21, "rebound", A, 21, sub="defensive", clock="PT11M38.00S"),
        act(22, "turnover", A, 22, clock="PT11M30.00S"),
        act(23, "substitution", H, 16, sub="out", clock="PT11M00.00S"),
        act(24, "substitution", H, 11, sub="in", clock="PT11M00.00S"),
        act(25, "2pt", H, 11, result="Made", clock="PT10M40.00S"),
    ]]
    b = box()
    b["homeTeam"]["score"] = 4
    b["awayTeam"]["score"] = 0
    poss, check = possessions_for_game("g", acts + q2, b)
    q2_first = [p for p in poss if p.period == 2][0]
    # 16 was subbed out in Q2 before being subbed in, so he started Q2; 13-15 carried over silently
    assert set(q2_first.off_lineup) == {12, 13, 14, 15, 16}
    assert [p for p in poss if p.period == 2][-1].off_lineup == (11, 12, 13, 14, 15)
    assert check["bad_lineups"] == 0
