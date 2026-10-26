from nbasim.data.pbp import possessions_for_game
from nbasim.data.pbp_v2 import convert

COLS = ["EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD", "PCTIMESTRING",
        "HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "VISITORDESCRIPTION", "SCORE",
        "PERSON1TYPE", "PLAYER1_ID", "PLAYER1_TEAM_ID", "PERSON2TYPE", "PLAYER2_ID",
        "PLAYER2_TEAM_ID", "PERSON3TYPE", "PLAYER3_ID", "PLAYER3_TEAM_ID"]
H, A = 100, 200


def row(n, et, sub, clock, desc, p1=0, t1=None, p2=0, t2=None, p3=0, t3=None, home=True):
    return [n, et, sub, 1, clock, desc if home else None, None, None if home else desc, None,
            4, p1, t1, 4, p2, t2, 4, p3, t3]


def test_v2_game():
    rows = [
        row(1, 12, 0, "12:00", None),
        row(2, 10, 0, "12:00", "Jump Ball", 11, H, 21, A, 12, H),
        row(3, 1, 1, "11:40", "Smith 3PT Jump Shot (3 PTS) (Jones 1 AST)", 12, H, 13, H),
        row(4, 2, 1, "11:20", "MISS Brown 2' Layup", 21, A, 0, None, 14, H, home=False),
        row(5, 4, 0, "11:18", "Rebound", 15, H),
        row(6, 6, 2, "11:00", "Foul", 22, A, 11, H, home=False),
        row(7, 3, 11, "11:00", "MISS Free Throw 1 of 2", 11, H),
        row(8, 4, 0, "11:00", "Team Rebound", H, None),          # placeholder rebound
        row(9, 3, 12, "11:00", "Free Throw 2 of 2 (1 PTS)", 11, H),
        row(10, 8, 0, "10:50", "SUB: X FOR Y", 22, A, 26, A, home=False),
        row(11, 5, 1, "10:30", "Team Turnover: Shot Clock", A, None, home=False),
        row(12, 1, 1, "10:10", "Lee 2PT Dunk (2 PTS)", 13, H),
        row(13, 2, 1, "09:50", "MISS Brown 26' 3PT", 21, A, home=False),
        row(14, 4, 0, "09:48", "Brown REBOUND", 23, A, home=False),
        row(15, 2, 1, "09:45", "MISS Green 2PT", 24, A, home=False),
        row(16, 4, 0, "09:44", "Team Rebound", H, None),
        row(17, 1, 1, "09:30", "Smith 2PT (2 PTS)", 12, H),
    ]
    shots = {3: {"LOC_X": 230, "LOC_Y": 10, "SHOT_DISTANCE": 23, "SHOT_TYPE": "3PT Field Goal"}}
    # starters on court before any event: 11-15 vs 21-25 (25 never appears)
    acts, box = convert("g", COLS, rows, H, A, 8, 0, shots)
    poss, check = possessions_for_game("g", acts, box)
    assert [p.offense for p in poss] == [H, A, H, A, H, A, H, A]
    assert poss[0].points == 3 and poss[0].shots[0]["zone"] == "corner3"
    assert poss[2].fta == 2 and poss[2].oreb == 0      # placeholder rebound dropped
    assert poss[3].tov == 1 and 26 in poss[3].off_lineup
    assert poss[5].oreb == 1 and poss[5].fga == 2
    assert check["pbp_home_pts"] == 8 == check["poss_home_pts"]
    assert check["bad_lineups"] == 8  # away has only 21, 22/26, 23, 24 seen in Q1 -> 4 players
