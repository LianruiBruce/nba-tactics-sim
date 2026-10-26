"""Download public NBA data: play-by-play, box scores and season-level stats.

Sources (free, no key):
- cdn.nba.com liveData: per-game play-by-play and box score JSON.
- stats.nba.com (the API behind nba.com/stats and the nba_api package):
  game logs, player/team/lineup stats, Synergy play types, tracking, matchups.

stats.nba.com often blocks cloud IPs, so run this on a home machine:

    python scripts/fetch_nba.py --seasons 2022-23 2023-24 2024-25 2025-26 2026-27

Raw responses are cached under data/raw/ (re-runs skip what is already there).
Then everything is packed into data/nba/<season>/:
    actions.jsonl.gz    every play-by-play action, one JSON object per line, with gameId
    boxscores.jsonl.gz  one box score per game
    stats/<name>.json.gz one file per stats.nba.com call
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "nba"

CDN_PBP = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json"
CDN_BOX = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"
STATS = "https://stats.nba.com/stats/{endpoint}?{query}"

STATS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}
CDN_HEADERS = {"User-Agent": STATS_HEADERS["User-Agent"], "Referer": "https://www.nba.com/"}

PLAY_TYPES = ["Isolation", "Transition", "PRBallHandler", "PRRollman", "Postup", "Spotup",
              "Handoff", "Cut", "OffScreen", "OffRebound", "Misc"]
PT_MEASURES = ["Drives", "Passing", "Possessions", "Rebounding", "CatchShoot", "PullUpShot",
               "PaintTouch", "PostTouch", "ElbowTouch", "SpeedDistance", "Efficiency", "Defense"]
DEFEND_CATEGORIES = ["Overall", "3 Pointers", "2 Pointers", "Less Than 6Ft", "Less Than 10Ft",
                     "Greater Than 15Ft"]


def get(url: str, headers: dict, retries: int = 5, timeout: int = 30) -> bytes | None:
    """GET with retries. Returns None on 404/403 for a missing game."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return data
        except urllib.error.HTTPError as e:
            if e.code in (403, 404) and "cdn.nba.com" in url:
                return None
            err = e
        except Exception as e:  # timeouts, resets
            err = e
        time.sleep(2 ** attempt + random.random())
    print(f"  failed: {url} ({err})", file=sys.stderr)
    return None


def cached(path: Path, fetch) -> bytes | None:
    if path.exists():
        return gzip.decompress(path.read_bytes())
    data = fetch()
    if data is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(data))
    return data


def stats_call(season: str, name: str, endpoint: str, params: dict) -> None:
    path = RAW / season / "stats" / f"{name}.json.gz"
    if path.exists():
        return
    url = STATS.format(endpoint=endpoint, query=urllib.parse.urlencode(params))
    data = get(url, STATS_HEADERS, timeout=60)
    if data is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data))
    time.sleep(0.7)  # stay polite; stats.nba.com throttles bursts


def fetch_season_stats(season: str, season_type: str = "Regular Season") -> None:
    base = {"Season": season, "SeasonType": season_type, "LeagueID": "00"}
    dash = {**base, "PerMode": "Totals", "MeasureType": "Base", "PaceAdjust": "N",
            "PlusMinus": "N", "Rank": "N", "LastNGames": 0, "Month": 0, "OpponentTeamID": 0,
            "Period": 0, "DateFrom": "", "DateTo": "", "GameSegment": "", "Location": "",
            "Outcome": "", "SeasonSegment": "", "VsConference": "", "VsDivision": "",
            "PORound": 0, "TeamID": 0, "ShotClockRange": "", "Conference": "", "Division": "",
            "GameScope": "", "PlayerExperience": "", "PlayerPosition": "", "StarterBench": "",
            "DraftYear": "", "DraftPick": "", "College": "", "Country": "", "Height": "",
            "Weight": ""}
    calls = [
        ("gamelog_team", "leaguegamelog", {**base, "PlayerOrTeam": "T", "Direction": "ASC",
                                           "Sorter": "DATE", "Counter": 0, "DateFrom": "",
                                           "DateTo": ""}),
        ("gamelog_player", "leaguegamelog", {**base, "PlayerOrTeam": "P", "Direction": "ASC",
                                             "Sorter": "DATE", "Counter": 0, "DateFrom": "",
                                             "DateTo": ""}),
    ]
    for measure in ("Base", "Advanced", "Misc", "Scoring", "Usage", "Defense"):
        calls.append((f"player_{measure.lower()}", "leaguedashplayerstats",
                      {**dash, "MeasureType": measure}))
    for measure in ("Base", "Advanced", "Four Factors", "Opponent"):
        calls.append((f"team_{measure.lower().replace(' ', '_')}", "leaguedashteamstats",
                      {**dash, "MeasureType": measure}))
    for measure in ("Base", "Advanced", "Four Factors"):
        calls.append((f"lineups5_{measure.lower().replace(' ', '_')}", "leaguedashlineups",
                      {**dash, "MeasureType": measure, "GroupQuantity": 5}))
    calls.append(("player_shot_locations", "leaguedashplayershotlocations",
                  {**dash, "DistanceRange": "By Zone"}))
    for kind in ("P", "T"):
        for grouping in ("offensive", "defensive"):
            for pt in PLAY_TYPES:
                if grouping == "defensive" and pt in ("Misc", "OffRebound"):
                    continue
                calls.append((f"playtype_{kind}_{grouping}_{pt}", "synergyplaytypes",
                              {"LeagueID": "00", "PerMode": "Totals", "PlayType": pt,
                               "PlayerOrTeam": kind, "SeasonType": season_type,
                               "SeasonYear": season, "TypeGrouping": grouping}))
        for m in PT_MEASURES:
            calls.append((f"tracking_{kind}_{m}", "leaguedashptstats",
                          {**dash, "PtMeasureType": m,
                           "PlayerOrTeam": "Player" if kind == "P" else "Team"}))
    for cat in DEFEND_CATEGORIES:
        calls.append((f"defend_{cat.replace(' ', '_')}", "leaguedashptdefend",
                      {**dash, "DefenseCategory": cat}))
    calls.append(("matchups", "leagueseasonmatchups",
                  {**base, "PerMode": "Totals", "DefPlayerID": "", "DefTeamID": "",
                   "OffPlayerID": "", "OffTeamID": ""}))
    for i, (name, endpoint, params) in enumerate(calls, 1):
        print(f"  [{i}/{len(calls)}] {name}", flush=True)
        stats_call(season, name, endpoint, params)


def season_game_ids(season: str) -> list[str]:
    """Regular-season game ids from the team game log, else the standard 1..1230 range."""
    path = RAW / season / "stats" / "gamelog_team.json.gz"
    if path.exists():
        js = json.loads(gzip.decompress(path.read_bytes()))
        rs = js["resultSets"][0]
        idx = rs["headers"].index("GAME_ID")
        ids = sorted({row[idx] for row in rs["rowSet"]})
        if ids:
            return ids
    yy = season[2:4]
    return [f"002{yy}{n:05d}" for n in range(1, 1231)]


def fetch_games(season: str, workers: int) -> list[str]:
    ids = season_game_ids(season)
    got = []

    def one(gid):
        pbp = cached(RAW / season / "pbp" / f"{gid}.json.gz",
                     lambda: get(CDN_PBP.format(gid=gid), CDN_HEADERS))
        box = cached(RAW / season / "box" / f"{gid}.json.gz",
                     lambda: get(CDN_BOX.format(gid=gid), CDN_HEADERS))
        return gid, pbp is not None and box is not None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, gid) for gid in ids]
        for n, f in enumerate(as_completed(futures), 1):
            gid, ok = f.result()
            if ok:
                got.append(gid)
            if n % 100 == 0 or n == len(ids):
                print(f"  games {n}/{len(ids)}, ok {len(got)}", flush=True)
    return sorted(got)


def pack(season: str, game_ids: list[str]) -> None:
    out = OUT / season
    (out / "stats").mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "actions.jsonl.gz", "wt", encoding="utf-8") as fa, \
            gzip.open(out / "boxscores.jsonl.gz", "wt", encoding="utf-8") as fb:
        for gid in game_ids:
            pbp = json.loads(gzip.decompress((RAW / season / "pbp" / f"{gid}.json.gz").read_bytes()))
            for a in pbp["game"]["actions"]:
                a["gameId"] = gid
                fa.write(json.dumps(a, separators=(",", ":")) + "\n")
            box = json.loads(gzip.decompress((RAW / season / "box" / f"{gid}.json.gz").read_bytes()))
            fb.write(json.dumps(box["game"], separators=(",", ":")) + "\n")
    for f in sorted((RAW / season / "stats").glob("*.json.gz")):
        (out / "stats" / f.name).write_bytes(f.read_bytes())
    sizes = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
    print(f"  packed {len(game_ids)} games -> {out} {sizes}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", default=["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"])
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads from cdn.nba.com")
    ap.add_argument("--skip-stats", action="store_true", help="only play-by-play and box scores")
    args = ap.parse_args()
    for season in args.seasons:
        print(f"== {season}", flush=True)
        if not args.skip_stats:
            fetch_season_stats(season)
        ids = fetch_games(season, args.workers)
        pack(season, ids)


if __name__ == "__main__":
    main()
