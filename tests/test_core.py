import json

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test
from pydantic import ValidationError

from nbasim import court
from nbasim.agents.scripted import ScriptedPolicy
from nbasim.engine import N_OFF, Engine
from nbasim.env import HalfCourtEnv
from nbasim.players import Player, load_lineups, load_roster

LEGAL_OUTCOMES = {"made_shot", "defensive_rebound", "turnover", "shot_clock_violation"}


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
    assert court.shot_zone((20.0, 15.0)) == "paint"
    assert court.shot_zone((10.0, 18.0)) == "mid"
    assert court.shot_zone((2.0, 3.0)) == "corner3"
    assert court.shot_zone((25.0, 32.0)) == "above3"


# ----------------------------------------------------------------- players
def test_roster_loads_ten_players():
    roster = load_roster()
    assert len(roster.players) == 10


def test_rating_out_of_range_rejected():
    raw = load_roster().players[0].model_dump()
    raw["outside_scoring"]["three_point"] = 120
    with pytest.raises(ValidationError):
        Player.model_validate(raw)


# ------------------------------------------------------------------ engine
def _scripted_replay(lineups, seed):
    env = HalfCourtEnv(lineups["mixed"], lineups["west"], record=True)
    env.reset(seed=seed)
    policy = ScriptedPolicy()
    while env.agents:
        env.step(policy(env.engine))
    return env.replay()


def test_same_seed_same_replay(lineups):
    a = json.dumps(_scripted_replay(lineups, 7))
    b = json.dumps(_scripted_replay(lineups, 7))
    assert a == b


def test_no_overlapping_bodies(lineups):
    rep = _scripted_replay(lineups, 3)
    for f in rep["frames"]:
        p = np.array(f["p"])
        d = np.linalg.norm(p[:, None] - p[None], axis=2)
        np.fill_diagonal(d, np.inf)
        assert d.min() > 2 * court.PLAYER_RADIUS - 0.6


def test_shot_clock_violation_when_nobody_shoots(lineups):
    eng = Engine(lineups["mixed"].players, lineups["west"].players, seed=0)
    eng.place(np.array([[25, 31]] * 5 + [[25, 26]] * 5, dtype=float) + np.arange(10)[:, None] * 2.5)
    while not eng.done:
        eng.step()
    assert eng.outcome == "shot_clock_violation"
    assert eng.t == pytest.approx(24.0, abs=0.11)


def test_three_counts_three(lineups):
    eng = Engine(lineups["mixed"].players, lineups["west"].players, seed=1)
    pos = np.array([[25, 32], [45, 20], [5, 20], [3, 3], [32, 7],
                    [25, 47], [45, 46], [5, 46], [10, 46], [40, 46]], dtype=float)
    for seed in range(50):
        eng = Engine(lineups["mixed"].players, lineups["west"].players, seed=seed)
        eng.place(pos, holder=0)
        eng.request_shot(0)
        while eng.flight is not None:
            eng.step()
        if eng.outcome == "made_shot":
            assert eng.points == 3
            return
    pytest.fail("Curry missed 50 open threes")


def test_contest_lowers_make_probability(lineups):
    eng = Engine(lineups["mixed"].players, lineups["west"].players, seed=0)
    far = np.array([[25, 32]] + [[45, 20], [5, 20], [3, 3], [32, 7]]
                   + [[25, 46], [45, 46], [5, 46], [10, 46], [40, 46]], dtype=float)
    eng.place(far)
    open_p, _, _ = eng.shot_make_prob(0)
    close = far.copy()
    close[N_OFF] = [25, 29.6]
    eng.place(close)
    contested_p, _, _ = eng.shot_make_prob(0)
    assert contested_p < open_p


# --------------------------------------------------------------------- env
def test_pettingzoo_parallel_api(lineups):
    parallel_api_test(HalfCourtEnv(lineups["mixed"], lineups["west"]), num_cycles=200)


def test_random_masked_policy_always_ends_legally(lineups):
    env = HalfCourtEnv(lineups["west"], lineups["mixed"])
    rng = np.random.default_rng(0)
    for seed in range(200):
        obs, _ = env.reset(seed=seed)
        while env.agents:
            acts = {a: int(rng.choice(np.flatnonzero(o["action_mask"]))) for a, o in obs.items()}
            obs, rew, *_ , infos = env.step(acts)
        assert env.engine.outcome in LEGAL_OUTCOMES
        assert sum(rew.values()) == 0  # zero-sum between the two teams
