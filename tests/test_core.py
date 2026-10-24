import json

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test
from pydantic import ValidationError

from nbasim import court
from nbasim.agents.team import TeamBrain
from nbasim.engine import HELD, Engine, members, team_of
from nbasim.env import NBAEnv
from nbasim.game import play_game
from nbasim.players import Lineup, Player, load_lineups, load_roster


@pytest.fixture(scope="module")
def lineups():
    return load_lineups()


# ------------------------------------------------------------------- court
@pytest.mark.parametrize("p,three", [
    ((2.5, 3.0), True),     # left corner, beyond 22 ft
    ((4.0, 3.0), False),    # inside the corner line
    ((25.0, 29.5), True),   # top of the arc, 24.25 ft
    ((25.0, 28.5), False),  # 23.25 ft
    ((47.5, 10.0), True),   # right corner
    ((25.0, 6.0), False),
])
def test_three_point_line(p, three):
    assert court.is_three(p) is three


def test_shot_zones():
    assert court.shot_zone((25.0, 7.0)) == "rim"
    assert court.shot_zone((20.0, 15.0)) == "short_mid"
    assert court.shot_zone((10.0, 18.0)) == "long_mid"
    assert court.shot_zone((2.0, 3.0)) == "corner3"
    assert court.shot_zone((25.0, 32.0)) == "above3"


@pytest.mark.parametrize("direction", [1, -1])
def test_local_global_round_trip(direction):
    pts = np.array([[3.0, 4.0], [25.0, 5.25], [40.0, 60.0]])
    assert np.allclose(court.to_local(court.to_global(pts, direction), direction), pts)
    # each direction attacks the basket at its own end of the court
    assert np.allclose(court.basket(direction), court.to_global(court.BASKET_LOCAL, direction))
    assert court.basket(1)[0] > court.LENGTH / 2 > court.basket(-1)[0]


# ----------------------------------------------------------------- players
def test_roster_is_2k27_real_players():
    roster = load_roster()
    ids = {p.id for p in roster.players}
    assert {"curry", "jokic", "giannis", "luka", "gobert"} <= ids
    for p in roster.players:
        assert 60 <= p.overall <= 99


def test_rating_out_of_range_rejected():
    raw = load_roster().players[0].model_dump()
    raw["outside_scoring"]["three_point"] = 120
    with pytest.raises(ValidationError):
        Player.model_validate(raw)


# ------------------------------------------------------------------ engine
def _run(lineups, seed, seconds, manual=False):
    eng = Engine(lineups["stars_a"].players, lineups["stars_b"].players, seed=seed, quarter_len=300)
    brains = (TeamBrain(seed=seed * 2 + 1), TeamBrain(seed=seed * 2 + 2))
    if manual:  # nobody decides to shoot, pass or drive
        brains[0].manual, brains[1].manual = set(range(5)), set(range(5, 10))
    frames = []
    while not eng.done and eng.t < seconds:
        eng.step(brains)
        frames.append(eng.pos.copy())
    return eng, frames


def test_same_seed_same_game(lineups):
    a, fa = _run(lineups, 7, 60)
    b, fb = _run(lineups, 7, 60)
    assert a.score == b.score
    assert json.dumps([e.to_dict() for e in a.events]) == json.dumps([e.to_dict() for e in b.events])
    assert np.allclose(fa[-1], fb[-1])


def test_no_overlapping_bodies(lineups):
    _, frames = _run(lineups, 3, 90)
    worst = min(
        np.min(np.linalg.norm(p[:, None] - p[None], axis=2) + np.eye(10) * 99) for p in frames)
    assert worst > 2 * court.PLAYER_RADIUS - 0.8


def test_players_stay_in_bounds(lineups):
    _, frames = _run(lineups, 4, 90)
    allp = np.concatenate(frames)
    assert allp[:, 0].min() > -4 and allp[:, 0].max() < court.LENGTH + 4
    assert allp[:, 1].min() > -4 and allp[:, 1].max() < court.WIDTH + 4


def test_shot_clock_violation_when_nobody_shoots(lineups):
    eng, _ = _run(lineups, 1, 60, manual=True)
    viol = [e for e in eng.events if e.type == "violation"]
    assert viol and viol[0].data["rule"] == "shot_clock"
    # possession started at the jump ball; 24 s plus the tip
    assert 24.0 <= viol[0].t <= 27.0
    tov = [e for e in eng.events if e.type == "turnover"]
    assert tov[0].data["kind"] == "shot_clock"


class _StayBack(TeamBrain):
    """Offense that never crosses half court."""

    def act(self, eng, team):
        super().act(eng, team)
        if eng.off == team and eng.status == HELD and team_of(eng.holder) == team:
            eng.set_target(eng.holder, eng.glob(np.array([25.0, 70.0]), team))


def test_eight_second_violation(lineups):
    eng = Engine(lineups["stars_a"].players, lineups["stars_b"].players, seed=0, quarter_len=300)
    brains = (_StayBack(seed=1), _StayBack(seed=2))
    for b, t in zip(brains, (0, 1)):
        b.manual = set(members(t))
    while not eng.done and eng.t < 40:
        eng.step(brains)
    rules = [e.data["rule"] for e in eng.events if e.type == "violation"]
    assert "eight_seconds" in rules


def test_points_add_up(lineups):
    eng = play_game(lineups["stars_a"], lineups["stars_b"], seed=5, quarters=1).engine
    assert eng.score[0] == int(eng.stats["pts"][:5].sum())
    assert eng.score[1] == int(eng.stats["pts"][5:].sum())
    s = eng.stats
    assert (s["pts"] == 2 * s["fgm"] + s["fg3m"] + s["ftm"]).all()
    assert (s["fgm"] <= s["fga"]).all() and (s["fg3m"] <= s["fg3a"]).all() and (s["ftm"] <= s["fta"]).all()
    # a quarter between two teams looks like an NBA quarter
    assert 15 <= eng.score[0] <= 50 and 15 <= eng.score[1] <= 50
    assert 18 <= eng.possessions[0] <= 32


def test_stars_beat_average(lineups):
    avg = load_roster("data/players/calibration.yaml").players
    margin = 0
    for seed in range(3):
        eng = play_game(lineups["stars_a"], Lineup(name="avg", players=avg), seed=seed, quarters=1).engine
        margin += eng.score[0] - eng.score[1]
    assert margin > 0


# --------------------------------------------------------------------- env
def test_pettingzoo_parallel_api(lineups):
    env = NBAEnv(lineups["stars_a"], lineups["stars_b"], quarter_len=60)
    parallel_api_test(env, num_cycles=300)


def test_random_masked_policy_zero_sum(lineups):
    env = NBAEnv(lineups["stars_a"], lineups["stars_b"], quarter_len=60)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    total = {a: 0.0 for a in env.agents}
    while env.agents:
        acts = {a: int(rng.choice(np.flatnonzero(o["action_mask"]))) for a, o in obs.items()}
        obs, rew, *_ = env.step(acts)
        for a, r in rew.items():
            total[a] += r
    s = env.engine.score
    assert total["a_0"] == s[0] - s[1] == -total["b_3"]


def test_scripted_team_not_exposed(lineups):
    env = NBAEnv(lineups["stars_a"], lineups["stars_b"], quarter_len=60, scripted=(1,))
    obs, _ = env.reset(seed=1)
    assert set(obs) == {f"a_{k}" for k in range(5)}
