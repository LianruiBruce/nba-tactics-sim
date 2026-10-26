"""Build possession and shot tables from the packed data of scripts/fetch_nba.py.

    python scripts/build_dataset.py --seasons 2024-25 2025-26

Reads data/nba/<season>/actions*.jsonl.gz + boxscores.jsonl.gz (cdn.nba.com)
and/or pbp_stats*.jsonl.gz (stats.nba.com playbyplayv2), writes
data/processed/<season>/possessions.parquet, shots.parquet and checks.csv,
then prints the phase-1 gate: points from possessions vs the box score, and
possessions per team-game vs stats.nba.com's POSS.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nbasim.data.pbp import possessions_for_game  # noqa: E402
from nbasim.data.pbp_v2 import convert  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NBA = ROOT / "data" / "nba"
OUT = ROOT / "data" / "processed"


def read_jsonl(paths):
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def stats_table(season: str, name: str) -> pd.DataFrame | None:
    p = NBA / season / "stats" / f"{name}.json.gz"
    if not p.exists():
        return None
    js = json.loads(gzip.decompress(p.read_bytes()))
    rs = js["resultSets"][0] if "resultSets" in js else js["resultSet"]
    return pd.DataFrame(rs["rowSet"], columns=rs["headers"])


def stats_games(season: str):
    """Yield (game_id, actions, box) from stats.nba.com play-by-play."""
    src = NBA / season
    files = sorted(src.glob("pbp_stats*.jsonl.gz"))
    if not files:
        return
    columns = json.loads((src / "pbp_v2_columns.json").read_text())
    log = stats_table(season, "gamelog_team")
    teams = {}
    for r in log.itertuples():
        g = teams.setdefault(r.GAME_ID, {})
        g["home" if " vs. " in r.MATCHUP else "away"] = (int(r.TEAM_ID), int(r.PTS))
        g["date"] = r.GAME_DATE
    sc = stats_table(season, "shotchart")
    shots: dict[str, dict] = {}
    if sc is not None:
        for row in sc.to_dict("records"):
            shots.setdefault(row["GAME_ID"], {})[row["GAME_EVENT_ID"]] = row
    for rec in read_jsonl(files):
        gid = rec["gameId"]
        if "v2" not in rec or gid not in teams or len(teams[gid]) != 3:
            continue  # TODO: playbyplayv3-only games
        (home, hp), (away, ap) = teams[gid]["home"], teams[gid]["away"]
        acts, box = convert(gid, columns, rec["v2"], home, away, hp, ap, shots.get(gid), teams[gid]["date"])
        yield gid, acts, box


def cdn_games(season: str):
    src = NBA / season
    if not (src / "boxscores.jsonl.gz").exists():
        return
    boxes = {b["gameId"]: b for b in read_jsonl([src / "boxscores.jsonl.gz"])}
    actions: dict[str, list] = {}
    for a in read_jsonl(sorted(src.glob("actions*.jsonl.gz"))):
        actions.setdefault(a["gameId"], []).append(a)
    for gid in sorted(actions):
        if gid in boxes:
            yield gid, actions[gid], boxes[gid]


def build(season: str) -> pd.DataFrame:
    poss_rows, shot_rows, checks = [], [], []
    seen = set()
    for gid, acts, box in list(cdn_games(season)) + list(stats_games(season)):
        if gid in seen:
            continue
        seen.add(gid)
        poss, check = possessions_for_game(gid, acts, box)
        checks.append(check)
        for i, p in enumerate(poss):
            d = asdict(p)
            shots = d.pop("shots")
            d["poss_id"] = f"{gid}-{i:03d}"
            d["date"] = (box.get("gameEt") or box.get("date") or "")[:10]
            d["season"] = season
            d["off_lineup"] = "-".join(map(str, p.off_lineup))
            d["def_lineup"] = "-".join(map(str, p.def_lineup))
            d["events"] = ";".join(p.events)
            poss_rows.append(d)
            for s in shots:
                s["poss_id"] = d["poss_id"]
                s["off_lineup"] = "-".join(map(str, s["off_lineup"]))
                s["def_lineup"] = "-".join(map(str, s["def_lineup"]))
                shot_rows.append(s)
    out = OUT / season
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(poss_rows).to_parquet(out / "possessions.parquet", index=False)
    pd.DataFrame(shot_rows).to_parquet(out / "shots.parquet", index=False)
    chk = pd.DataFrame(checks)
    chk.to_csv(out / "checks.csv", index=False)
    return chk


def gate(season: str, chk: pd.DataFrame) -> None:
    n = len(chk)
    pts_ok = (chk.pbp_home_pts == chk.box_home_pts) & (chk.pbp_away_pts == chk.box_away_pts)
    missing = (chk.box_home_pts - chk.poss_home_pts + chk.box_away_pts - chk.poss_away_pts)
    print(f"== {season}: {n} games")
    print(f"  pbp points == box score: {pts_ok.mean():.3%} of games")
    print(f"  points outside possessions (technical FTs etc.) per game: {missing.mean():.2f}")
    print(f"  possessions with a bad lineup: {chk.bad_lineups.sum()} "
          f"({chk.bad_lineups.sum() / (chk.home_poss + chk.away_poss).sum():.3%})")
    print(f"  games with starter warnings: {(chk.warnings != '').sum()}")
    adv = stats_table(season, "team_advanced")
    if adv is not None and "POSS" in adv:
        ours = pd.concat([chk.groupby("home").home_poss.sum(), chk.groupby("away").away_poss.sum()],
                         axis=1).sum(axis=1)
        nba = adv.set_index("TEAM_ID")["POSS"]
        both = pd.concat([ours.rename("ours"), nba.rename("nba")], axis=1).dropna()
        err = (both.ours / both.nba - 1)
        print(f"  team possessions vs stats.nba.com POSS: mean error {err.mean():+.2%}, "
              f"max abs {err.abs().max():.2%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=sorted(p.name for p in NBA.iterdir() if p.is_dir()))
    args = ap.parse_args()
    for season in args.seasons:
        gate(season, build(season))


if __name__ == "__main__":
    main()
