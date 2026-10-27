"""Forward forecasts: who will play, how each matchup should go, and how to score it afterwards.

A forecast is made at a freeze time from games that finished before it, so nothing after the
freeze can leak in. For each upcoming game:

1. rotation(): each team's likely five-man units with weights, from its recent games
   (recent games weigh more; units with a player who missed the team's last games are dropped).
2. predict_games(): every model (ours and the baselines) predicts points per possession for each
   pair of units, mixed over how possessions start. Game margin = pace x weighted difference;
   win probability = Phi(margin / sigma), with sigma fixed per model in advance.
3. The same pair table gives each likely unit's expected net rating in each game (per 100 possessions,
   against the opponent's projected rotation).

Scoring (score_games / score_lineups) compares predictions with what happened.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from nbasim.realmodel import lineup

# share of possessions by how they start (2025-26)
START_MIX = {"def_rebound": 0.34, "made_fg": 0.39, "made_ft": 0.10, "turnover": 0.14, "period_start": 0.02,
             "other": 0.01}
TOP_LINEUPS = 5            # units per team and game written to lineups.csv


# ------------------------------------------------------------------ schedule
def schedule_from_boxscores(path: Path) -> pd.DataFrame:
    """Finished games from a packed boxscores.jsonl.gz (also gives the final score)."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            h, a = b["homeTeam"], b["awayTeam"]
            rows.append({"game_id": b["gameId"], "tip_utc": b.get("gameTimeUTC"),
                         "date": (b.get("gameEt") or "")[:10],
                         "home": h["teamId"], "away": a["teamId"],
                         "home_tri": h["teamTricode"], "away_tri": a["teamTricode"],
                         "home_pts": h.get("score"), "away_pts": a.get("score"),
                         "final": b.get("gameStatus") == 3})
    return pd.DataFrame(rows)


def schedule_from_cdn(path: Path, prefixes=("002",)) -> pd.DataFrame:
    """Full season schedule from cdn.nba.com scheduleLeagueV2.json (gzipped)."""
    js = json.loads(gzip.decompress(Path(path).read_bytes()))
    rows = []
    for day in js["leagueSchedule"]["gameDates"]:
        for g in day["games"]:
            if not g["gameId"].startswith(prefixes):
                continue
            h, a = g["homeTeam"], g["awayTeam"]
            if not h.get("teamId") or not a.get("teamId"):
                continue                       # NBA Cup knockout slots without teams yet
            et = g.get("gameDateTimeEst") or g.get("gameDateEst") or ""
            rows.append({"game_id": g["gameId"], "tip_utc": g.get("gameDateTimeUTC"), "date": et[:10],
                         "home": h["teamId"], "away": a["teamId"],
                         "home_tri": h.get("teamTricode"), "away_tri": a.get("teamTricode"),
                         "home_pts": h.get("score"), "away_pts": a.get("score"),
                         "final": g.get("gameStatus") == 3})
    return pd.DataFrame(rows)


def utc(s) -> pd.Timestamp:
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ------------------------------------------------------------------ rotations
def team_lineups(hist: pd.DataFrame, team: int) -> pd.DataFrame:
    """The team's possessions (offense and defense) with its own unit in `unit`."""
    t = hist[(hist.offense == team) | (hist.defense == team)]
    unit = np.where(t.offense.values == team, t.off_lineup.values, t.def_lineup.values)
    return t.assign(unit=unit)


def rotation(hist: pd.DataFrame, team: int, games: int = 10, half_life: float = 4.0, cover: float = 0.9,
             max_units: int = 30, missing_games: int = 2, out=()) -> list[tuple[str, float]]:
    """Likely five-man units and their share of the next game, from the team's last `games` games."""
    t = team_lineups(hist, team)
    if t.empty:
        return []
    order = t.groupby("game_id").date.first().sort_values(ascending=False)
    recent = list(order.index[:games])
    gw = {g: 0.5 ** (k / half_life) for k, g in enumerate(recent)}
    t = t[t.game_id.isin(gw)]
    w = t.game_id.map(gw)
    units = w.groupby(t.unit.values).sum()
    avail = {p for s in t[t.game_id.isin(recent[:missing_games])].unit.unique() for p in lineup(s)}
    avail -= set(out)
    keep = [u for u in units.index if set(lineup(u)) <= avail]
    units = units[keep].sort_values(ascending=False)
    if units.empty:
        return []
    cum = units.cumsum() / units.sum()
    n = min(max_units, int((cum < cover).sum()) + 1)
    units = units.iloc[:n]
    return list(zip(units.index, (units / units.sum()).values))


def pace(hist: pd.DataFrame, team: int, games: int = 10, default: float = 99.0) -> float:
    t = hist[hist.offense == team]
    if t.empty:
        return default
    per_game = t.groupby("game_id").agg(n=("points", "size"), d=("date", "first")).sort_values("d")
    return float(per_game.n.iloc[-games:].mean())


# ------------------------------------------------------------------ predictions
def pair_ppp(models: dict, H: list[str], A: list[str], home_id: int, away_id: int) -> dict:
    """{model: (P_home, P_away)}; P_home[i, j] = points per possession of home unit i against away unit j."""
    hh, aa = np.meshgrid(np.arange(len(H)), np.arange(len(A)), indexing="ij")
    hh, aa = hh.ravel(), aa.ravel()
    frames = []
    for side_home in (True, False):
        for ctx in START_MIX:
            off = [H[i] for i in hh] if side_home else [A[j] for j in aa]
            de = [A[j] for j in aa] if side_home else [H[i] for i in hh]
            frames.append(pd.DataFrame({"off_lineup": off, "def_lineup": de, "start_type": ctx,
                                        "home": side_home, "offense": home_id if side_home else away_id,
                                        "defense": away_id if side_home else home_id}))
    rows = pd.concat(frames, ignore_index=True)
    mix = np.array(list(START_MIX.values()))
    mix = mix / mix.sum()
    k = len(hh)
    out = {}
    for name, m in models.items():
        p = m.predict(rows).reshape(2, len(START_MIX), k)
        p = np.tensordot(mix, p, axes=([0], [1]))           # (2, k)
        out[name] = (p[0].reshape(len(H), len(A)), p[1].reshape(len(H), len(A)))
    return out


def predict_games(hist: pd.DataFrame, games: pd.DataFrame, models: dict, sigma: dict,
                  out_players=(), rot_kw: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict upcoming `games` (schedule rows) from finished history. Returns (games, lineups)."""
    rot_kw = rot_kw or {}
    rots, paces = {}, {}
    for t in set(games.home) | set(games.away):
        rots[t] = rotation(hist, t, out=out_players, **rot_kw)
        paces[t] = pace(hist, t)
    g_rows, l_rows = [], []
    for g in games.itertuples(index=False):
        rh, ra = rots[g.home], rots[g.away]
        row = {"game_id": g.game_id, "tip_utc": g.tip_utc, "date": g.date, "home": g.home, "away": g.away,
               "home_tri": g.home_tri, "away_tri": g.away_tri,
               "pace": round((paces[g.home] + paces[g.away]) / 2, 2),
               "home_units": len(rh), "away_units": len(ra)}
        if rh and ra:
            H, wh = [u for u, _ in rh], np.array([w for _, w in rh])
            A, wa = [u for u, _ in ra], np.array([w for _, w in ra])
            units = [(g.home, u, s, i, None) for i, (u, s) in enumerate(rh[:TOP_LINEUPS])]
            units += [(g.away, u, s, None, j) for j, (u, s) in enumerate(ra[:TOP_LINEUPS])]
            lr = [{"game_id": g.game_id, "team": t, "lineup": u, "proj_share": round(s, 4)} for t, u, s, _, _ in units]
            for name, (ph, pa) in pair_ppp(models, H, A, g.home, g.away).items():
                d = ph - pa                                   # home edge per possession, per unit pair
                margin = row["pace"] * float(wh @ d @ wa)
                row[f"margin_{name}"] = round(margin, 3)
                row[f"wp_{name}"] = round(norm_cdf(margin / sigma.get(name, 12.0)), 4)
                for r, (_, _, _, i, j) in zip(lr, units):
                    net = 100 * float(d[i] @ wa) if i is not None else -100 * float(wh @ d[:, j])
                    r[f"net_{name}"] = round(net, 2)
            l_rows += lr
        g_rows.append(row)
    return pd.DataFrame(g_rows), pd.DataFrame(l_rows)


def norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ------------------------------------------------------------------ scoring
def game_metrics(df: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    """Per source, on the games where that source has a prediction."""
    out = {}
    for s in sources:
        m = df.dropna(subset=[f"margin_{s}"]) if f"margin_{s}" in df else df.iloc[0:0]
        if m.empty:
            continue
        err = m.actual_margin - m[f"margin_{s}"]
        r = {"games": len(m), "margin_mae": err.abs().mean(), "margin_rmse": np.sqrt((err ** 2).mean()),
             "winner_acc": (np.sign(m[f"margin_{s}"]) == np.sign(m.actual_margin)).mean()}
        if f"wp_{s}" in m and m[f"wp_{s}"].notna().all():
            p = m[f"wp_{s}"].clip(1e-4, 1 - 1e-4)
            y = (m.actual_margin > 0).astype(float)
            r["brier"] = ((p - y) ** 2).mean()
            r["log_loss"] = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        out[s] = r
    return pd.DataFrame(out).T


def paired(df: pd.DataFrame, a: str, b: str, metric: str, n: int = 4000, seed: int = 0):
    """Mean of metric(a) - metric(b) over games both predicted, with a 95% bootstrap interval.

    metric: "sq" (squared margin error) or "brier". Negative means a is better."""
    cols = [f"margin_{a}", f"margin_{b}"] if metric == "sq" else [f"wp_{a}", f"wp_{b}"]
    if not all(c in df for c in cols):
        return None
    m = df.dropna(subset=cols)
    if len(m) < 5:
        return None
    if metric == "sq":
        d = (m.actual_margin - m[cols[0]]) ** 2 - (m.actual_margin - m[cols[1]]) ** 2
    else:
        y = (m.actual_margin > 0).astype(float)
        d = (m[cols[0]] - y) ** 2 - (m[cols[1]] - y) ** 2
    d = d.values
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return len(d), float(d.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def unit_actuals(poss: pd.DataFrame, team: int, unit: str, game_ids) -> tuple[int, int, int, int]:
    """(off possessions, off points, def possessions, def points) of a unit in some games."""
    p = poss[poss.game_id.isin(set(game_ids))]
    o = p[(p.offense == team) & (p.off_lineup == unit)]
    d = p[(p.defense == team) & (p.def_lineup == unit)]
    return len(o), int(o.points.sum()), len(d), int(d.points.sum())


def lineup_metrics(df: pd.DataFrame, sources: list[str], min_poss: int = 1) -> pd.DataFrame:
    """Possession-weighted error of predicted vs actual unit net rating."""
    df = df[(df.n_off + df.n_def) >= min_poss]
    if df.empty:
        return pd.DataFrame()
    actual = 100 * (df.pts_off / df.n_off.clip(lower=1) - df.pts_def / df.n_def.clip(lower=1))
    w = (df.n_off + df.n_def).astype(float)
    out = {}
    for s in sources:
        c = f"net_{s}"
        if c not in df or df[c].isna().all():
            continue
        ok = df[c].notna() & (df.n_off > 0) & (df.n_def > 0)
        a, p, ww = actual[ok], df[c][ok], w[ok]
        cov = np.cov(a, p, aweights=ww) if ok.sum() > 2 else np.full((2, 2), np.nan)
        out[s] = {"units": int(ok.sum()), "possessions": int(ww.sum()),
                  "net_rmse": float(np.sqrt(np.average((a - p) ** 2, weights=ww))),
                  "net_corr": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))}
    return pd.DataFrame(out).T


def paired_units(df: pd.DataFrame, a: str, b: str, min_poss: int = 1, n: int = 4000, seed: int = 0):
    """Possession-weighted mean of squared net-rating error, a minus b, resampling units."""
    df = df[((df.n_off + df.n_def) >= min_poss) & (df.n_off > 0) & (df.n_def > 0)]
    if f"net_{a}" not in df or f"net_{b}" not in df:
        return None
    df = df.dropna(subset=[f"net_{a}", f"net_{b}"])
    if len(df) < 5:
        return None
    actual = 100 * (df.pts_off / df.n_off - df.pts_def / df.n_def)
    d = ((actual - df[f"net_{a}"]) ** 2 - (actual - df[f"net_{b}"]) ** 2).values
    w = (df.n_off + df.n_def).values.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = (d[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return len(d), float((d * w).sum() / w.sum()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
