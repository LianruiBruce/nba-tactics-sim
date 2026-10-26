import pandas as pd

from nbasim.realmodel import ZONE_PTS, chances


def poss(events, start="def_rebound"):
    return {"poss_id": "g-000", "season": "2025-26", "date": "2026-01-01", "game_id": "g", "offense": 1,
            "defense": 2, "home": True, "off_lineup": "1-2-3-4-5", "def_lineup": "6-7-8-9-10",
            "start_type": start, "events": events}


def test_chances_split_and_points():
    df = pd.DataFrame([
        poss("F|1|above3|0;R|5;F|5|rim|1;FT|5|1|1|0|a"),        # miss, offensive board, and-one (FT missed)
        poss("FT|2|1|2|0|;FT|2|2|2|1|"),                         # two-shot trip
        poss("FT|3|1|1|1|r;F|3|corner3|1", start="made_fg"),     # flagrant free throw, ball retained, then a three
        poss("T|4", start="turnover"),
    ])
    ch = chances(df)
    assert list(ch.outcome) == ["fga", "fga", "trip", "trip", "fga", "tov"]
    assert list(ch.ctx) == ["def_rebound", "oreb", "def_rebound", "made_fg", "retained", "turnover"]
    first = ch.iloc[0]
    assert first.oreb == 1 and first.miss_type == "above3"
    and1 = ch.iloc[1]
    assert and1.and1 == 1 and and1.ft_made == 0 and and1.oreb == 0   # missed and-one FT, defense rebounds
    assert ch.iloc[2].ft_n == 2 and ch.iloc[2].ft_made == 1
    fga = ch[ch.outcome == "fga"]
    pts = (fga.made * fga.zone.map(ZONE_PTS)).sum() + ch.ft_made.fillna(0).sum()
    assert pts == 2 + 1 + 1 + 3
