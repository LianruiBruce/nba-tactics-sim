import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nbasim import forecast as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_lines  # noqa: E402
import forecast as FC  # noqa: E402

A1, A2 = "1-2-3-4-5", "1-2-3-4-6"      # team 10
B1 = "11-12-13-14-15"                  # team 20


def poss_rows(game, date, off, de, off_team, def_team, n, pts=1, home=True):
    return [{"game_id": game, "date": date, "offense": off_team, "defense": def_team, "off_lineup": off,
             "def_lineup": de, "points": pts, "home": home, "start_type": "def_rebound", "season": "2025-26"}] * n


def history():
    rows = []
    # game 1: A1 plays most; game 2 (latest): A2 plays most, player 6 present
    rows += poss_rows("g1", "2026-01-01", A1, B1, 10, 20, 80) + poss_rows("g1", "2026-01-01", B1, A1, 20, 10, 80, home=False)
    rows += poss_rows("g1", "2026-01-01", A2, B1, 10, 20, 20) + poss_rows("g1", "2026-01-01", B1, A2, 20, 10, 20, home=False)
    rows += poss_rows("g2", "2026-01-03", A2, B1, 10, 20, 70) + poss_rows("g2", "2026-01-03", B1, A2, 20, 10, 70, home=False)
    rows += poss_rows("g2", "2026-01-03", A1, B1, 10, 20, 30) + poss_rows("g2", "2026-01-03", B1, A1, 20, 10, 30, home=False)
    return pd.DataFrame(rows)


class ByLineup:
    """Toy model: offense lineup containing player 1 scores 1.2, otherwise 1.0; home adds 0.02."""

    def predict(self, df):
        return np.array([1.2 if "1" in o.split("-") else 1.0 for o in df.off_lineup]) + 0.02 * df.home.values


def test_rotation_weights_recent_games_and_drops_missing_players():
    h = history()
    rot = dict(F.rotation(h, 10, games=10, half_life=1.0, cover=1.0))
    assert set(rot) == {A1, A2}
    assert rot[A2] > rot[A1]                      # A2 led the most recent game
    assert abs(sum(rot.values()) - 1) < 1e-9
    rot = dict(F.rotation(h, 10, cover=1.0, out=[6]))
    assert set(rot) == {A1}                       # player 6 ruled out


def test_predict_games_margin_and_unit_nets():
    h = history()
    games = pd.DataFrame([{"game_id": "g3", "tip_utc": "2026-01-05T00:00:00Z", "date": "2026-01-04",
                           "home": 10, "away": 20, "home_tri": "AAA", "away_tri": "BBB"}])
    g, lu = F.predict_games(h, games, {"toy": ByLineup()}, {"toy": 12.0}, rot_kw={"cover": 1.0})
    row = g.iloc[0]
    # home units all contain player 1: 1.22 vs 1.00 per possession, pace 100
    assert abs(row.margin_toy - 100 * 0.22) < 1e-6
    assert 0.5 < row.wp_toy < 1
    home_units = lu[lu.team == 10]
    assert set(home_units.lineup) == {A1, A2}
    assert np.allclose(home_units.net_toy, 22.0)
    assert np.allclose(lu[lu.team == 20].net_toy, -22.0)


def test_latest_batch_frozen_before_tip_counts(tmp_path):
    final = pd.DataFrame([{"game_id": "0022500003", "home_pts": 110, "away_pts": 100}])
    bs = []
    for name, frozen, margin in [("b1", "2026-01-03T12:00:00Z", 1.0), ("b2", "2026-01-04T12:00:00Z", 2.0),
                                 ("b3", "2026-01-05T01:00:00Z", 3.0)]:   # b3 is after tip-off: ignored
        d = tmp_path / name
        d.mkdir()
        pd.DataFrame([{"game_id": "0022500003", "tip_utc": "2026-01-05T00:00:00Z", "date": "2026-01-04",
                       "margin_real": margin}]).to_csv(d / "games.csv", index=False)
        bs.append({"dir": d, "frozen": pd.Timestamp(frozen), "committed": None})
    g = FC.assign(bs, final)
    assert len(g) == 1 and g.margin_real[0] == 2.0 and g.batch[0] == "b2" and g.actual_margin[0] == 10


def test_paired_and_metrics():
    df = pd.DataFrame({"actual_margin": [10, -5, 3, 8, -2, 7],
                       "margin_a": [9, -4, 2, 7, -1, 6], "margin_b": [0, 0, 0, 0, 0, 0],
                       "wp_a": [0.8, 0.3, 0.6, 0.7, 0.4, 0.7], "wp_b": [0.5] * 6})
    m = F.game_metrics(df, ["a", "b"])
    assert m.loc["a", "winner_acc"] == 1.0 and m.loc["a", "margin_mae"] == 1.0
    n, mean, lo, hi = F.paired(df, "a", "b", "sq")
    assert n == 6 and mean < 0 and hi < 0


def test_devig_and_espn_spread():
    assert abs(FC.devig(-200, 170) - (2 / 3) / (2 / 3 + 100 / 270)) < 1e-4
    assert fetch_lines.spread_home({"details": "BOS -6.5"}, "BOS", "NYK") == -6.5
    assert fetch_lines.spread_home({"details": "GS -3"}, "LAL", "GSW") == 3.0
    assert fetch_lines.spread_home({"details": "EVEN"}, "LAL", "GSW") == 0.0
    assert fetch_lines.moneylines({"moneyline": {"home": {"close": {"odds": "-150"}},
                                                 "away": {"close": {"odds": "+130"}}}}) == (-150.0, 130.0)


def test_schedule_from_cdn(tmp_path):
    import gzip
    import json
    js = {"leagueSchedule": {"seasonYear": "2026-27", "gameDates": [{"games": [
        {"gameId": "0022600050", "gameStatus": 1, "gameDateTimeUTC": "2026-10-29T23:30:00Z",
         "gameDateTimeEst": "2026-10-29T19:30:00Z",
         "homeTeam": {"teamId": 1610612738, "teamTricode": "BOS", "score": 0},
         "awayTeam": {"teamId": 1610612752, "teamTricode": "NYK", "score": 0}},
        {"gameId": "0012600001", "gameStatus": 3, "gameDateTimeUTC": "2026-10-05T23:30:00Z",
         "homeTeam": {"teamId": 1, "teamTricode": "X"}, "awayTeam": {"teamId": 2, "teamTricode": "Y"}},
        {"gameId": "0062600001", "gameStatus": 1, "gameDateTimeUTC": "2026-12-16T23:30:00Z",
         "homeTeam": {"teamId": 0, "teamTricode": None}, "awayTeam": {"teamId": 0, "teamTricode": None}},
    ]}]}}
    p = tmp_path / "schedule.json.gz"
    p.write_bytes(gzip.compress(json.dumps(js).encode()))
    s = F.schedule_from_cdn(p)
    assert list(s.game_id) == ["0022600050"]           # preseason and empty Cup slots dropped
    r = s.iloc[0]
    assert r.date == "2026-10-29" and r.home_tri == "BOS" and not r.final
