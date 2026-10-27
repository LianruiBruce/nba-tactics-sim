"""Download public NBA data: play-by-play, box scores and season-level stats.

Sources (free, no key):
- cdn.nba.com liveData: per-game play-by-play and box score JSON.
- stats.nba.com (the API behind nba.com/stats and the nba_api package):
  game logs, player/team/lineup stats, Synergy play types, tracking, matchups.

stats.nba.com often blocks cloud IPs, so run this on a home machine:

    python scripts/fetch_nba.py --seasons 2022-23 2023-24 2024-25 2025-26 2026-27
    python scripts/fetch_nba.py --seasons 2026-27 --refresh     # during the season: new games + schedule

Per-game play-by-play comes from cdn.nba.com when it answers; if the CDN refuses
(403), it falls back to stats.nba.com playbyplayv2 (plus playbyplayv3 when v2 is
empty) and a league-wide shot chart for coordinates.

Raw responses are cached under data/raw/ (re-runs skip what is already there).
Then everything is packed into data/nba/<season>/:
    actions.jsonl.gz    cdn liveData actions, one JSON object per line, with gameId
    boxscores.jsonl.gz  cdn liveData box score per game
    pbp_stats.jsonl.gz  stats.nba.com play-by-play per game (when the CDN refused)
    stats/<name>.json.gz one file per stats.nba.com call
    schedule.json.gz    cdn scheduleLeagueV2 (with --refresh; the CDN only has the current season)
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
CDN_SCHEDULE = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
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
CDN_HEADERS = {k: v for k, v in STATS_HEADERS.items() if not k.startswith("x-nba")}

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
    calls.append(("shotchart", "shotchartdetail",
                  {**base, "PlayerID": 0, "TeamID": 0, "GameID": "", "ContextMeasure": "FGA",
                   "PlayerPosition": "", "RookieYear": "", "Position": "", "Outcome": "",
                   "Location": "", "Month": 0, "SeasonSegment": "", "DateFrom": "", "DateTo": "",
                   "OpponentTeamID": 0, "VsConference": "", "VsDivision": "", "GameSegment": "",
                   "Period": 0, "LastNGames": 0, "AheadBehind": "", "ClutchTime": "",
                   "PointDiff": "", "RangeType": 0, "StartPeriod": 1, "EndPeriod": 10,
                   "StartRange": 0, "EndRange": 28800, "ContextFilter": ""}))
    calls.append(("matchups", "leagueseasonmatchups",
                  {**base, "PerMode": "Totals", "DefPlayerID": "", "DefTeamID": "",
                   "OffPlayerID": "", "OffTeamID": ""}))
    for i, (name, endpoint, params) in enumerate(calls, 1):
        print(f"  [{i}/{len(calls)}] {name}", flush=True)
        stats_call(season, name, endpoint, params)


def season_game_ids(season: str) -> list[str]:
    """Regular-season game ids from the team game log, else the schedule, else the standard 1..1230 range."""
    path = RAW / season / "stats" / "gamelog_team.json.gz"
    if path.exists():
        js = json.loads(gzip.decompress(path.read_bytes()))
        rs = js["resultSets"][0]
        idx = rs["headers"].index("GAME_ID")
        ids = sorted({row[idx] for row in rs["rowSet"]})
        if ids:
            return ids
    sched = OUT / season / "schedule.json.gz"
    if sched.exists():   # finished regular-season games from the schedule
        js = json.loads(gzip.decompress(sched.read_bytes()))
        ids = sorted(g["gameId"] for d in js["leagueSchedule"]["gameDates"] for g in d["games"]
                     if g["gameId"].startswith("002") and g.get("gameStatus") == 3)
        if ids:
            return ids
    yy = season[2:4]
    return [f"002{yy}{n:05d}" for n in range(1, 1231)]


V2_KEEP = ["EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD", "PCTIMESTRING",
           "HOMEDESCRIPTION", "NEUTRALDESCRIPTION", "VISITORDESCRIPTION", "SCORE",
           "PERSON1TYPE", "PLAYER1_ID", "PLAYER1_TEAM_ID", "PERSON2TYPE", "PLAYER2_ID",
           "PLAYER2_TEAM_ID", "PERSON3TYPE", "PLAYER3_ID", "PLAYER3_TEAM_ID"]
cdn_blocked = False


def stats_game(season: str, kind: str, gid: str) -> bytes | None:
    path = RAW / season / kind / f"{gid}.json.gz"
    if path.exists():
        return gzip.decompress(path.read_bytes())
    endpoint, params = {
        "pbp_v2": ("playbyplayv2", {"GameID": gid, "StartPeriod": 0, "EndPeriod": 14}),
        "pbp_v3": ("playbyplayv3", {"GameID": gid, "StartPeriod": 0, "EndPeriod": 0}),
    }[kind]
    data = get(STATS.format(endpoint=endpoint, query=urllib.parse.urlencode(params)), STATS_HEADERS)
    time.sleep(0.6)
    if data is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(data))
    return data


def v2_rows(data: bytes) -> list:
    rs = json.loads(data)["resultSets"][0]
    return rs["rowSet"]


def fetch_games(season: str, workers: int, stats_workers: int) -> tuple[list[str], list[str]]:
    """Returns (games from the CDN, games from stats.nba.com)."""
    global cdn_blocked
    ids = season_game_ids(season)
    cdn_ok, stats_ok = [], []

    def one(gid):
        global cdn_blocked
        if not cdn_blocked or (RAW / season / "pbp" / f"{gid}.json.gz").exists():
            pbp = cached(RAW / season / "pbp" / f"{gid}.json.gz",
                         lambda: get(CDN_PBP.format(gid=gid), CDN_HEADERS))
            if pbp is not None:
                box = cached(RAW / season / "box" / f"{gid}.json.gz",
                             lambda: get(CDN_BOX.format(gid=gid), CDN_HEADERS))
                if box is not None:
                    return gid, "cdn"
            else:
                cdn_blocked = True
        v2 = stats_game(season, "pbp_v2", gid)
        if v2 is not None and v2_rows(v2):
            return gid, "stats"
        v3 = stats_game(season, "pbp_v3", gid)
        if v3 is not None and json.loads(v3).get("game", {}).get("actions"):
            return gid, "stats"
        return gid, None

    # probe the CDN once before fanning out
    first = one(ids[0])
    results = [first]
    n_workers = workers if first[1] == "cdn" else stats_workers
    if first[1] != "cdn":
        print(f"  cdn.nba.com refused; using stats.nba.com play-by-play ({n_workers} workers)", flush=True)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(one, gid) for gid in ids[1:]]
        for n, f in enumerate(as_completed(futures), 2):
            results.append(f.result())
            if n % 100 == 0 or n == len(ids):
                print(f"  games {n}/{len(ids)}", flush=True)
    for gid, src in results:
        if src == "cdn":
            cdn_ok.append(gid)
        elif src == "stats":
            stats_ok.append(gid)
    missing = len(ids) - len(cdn_ok) - len(stats_ok)
    print(f"  {len(cdn_ok)} games from cdn, {len(stats_ok)} from stats.nba.com, {missing} missing", flush=True)
    return sorted(cdn_ok), sorted(stats_ok)


def pack(season: str, cdn_ids: list[str], stats_ids: list[str]) -> None:
    out = OUT / season
    (out / "stats").mkdir(parents=True, exist_ok=True)
    if cdn_ids:
        with gzip.open(out / "actions.jsonl.gz", "wt", encoding="utf-8") as fa, \
                gzip.open(out / "boxscores.jsonl.gz", "wt", encoding="utf-8") as fb:
            for gid in cdn_ids:
                pbp = json.loads(gzip.decompress((RAW / season / "pbp" / f"{gid}.json.gz").read_bytes()))
                for a in pbp["game"]["actions"]:
                    a["gameId"] = gid
                    fa.write(json.dumps(a, separators=(",", ":")) + "\n")
                box = json.loads(gzip.decompress((RAW / season / "box" / f"{gid}.json.gz").read_bytes()))
                fb.write(json.dumps(box["game"], separators=(",", ":")) + "\n")
    if stats_ids:
        with gzip.open(out / "pbp_stats.jsonl.gz", "wt", encoding="utf-8") as fs:
            for gid in stats_ids:
                rec = {"gameId": gid}
                v2p = RAW / season / "pbp_v2" / f"{gid}.json.gz"
                rows = v2_rows(gzip.decompress(v2p.read_bytes())) if v2p.exists() else []
                if rows:
                    rs = json.loads(gzip.decompress(v2p.read_bytes()))["resultSets"][0]
                    keep = [rs["headers"].index(h) for h in V2_KEEP]
                    rec["v2"] = [[r[i] for i in keep] for r in rows]
                else:
                    v3 = json.loads(gzip.decompress((RAW / season / "pbp_v3" / f"{gid}.json.gz").read_bytes()))
                    rec["v3"] = v3["game"]["actions"]
                fs.write(json.dumps(rec, separators=(",", ":")) + "\n")
        (out / "pbp_v2_columns.json").write_text(json.dumps(V2_KEEP))
    for f in sorted((RAW / season / "stats").glob("*.json.gz")):
        (out / "stats" / f.name).write_bytes(f.read_bytes())
    sizes = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
    print(f"  packed {len(cdn_ids) + len(stats_ids)} games -> {out} {sizes}")


def refresh(season: str) -> None:
    """Drop the season-level stats cache (game log, season totals) so they are fetched again,
    and download the current schedule. Finished games stay cached."""
    stats = RAW / season / "stats"
    n = 0
    for f in stats.glob("*.json.gz") if stats.exists() else []:
        f.unlink()
        n += 1
    print(f"  refresh: dropped {n} cached season stats files", flush=True)
    data = get(CDN_SCHEDULE, CDN_HEADERS)
    if data is None:
        print("  schedule: cdn.nba.com refused", file=sys.stderr)
        return
    js = json.loads(data)
    if js.get("leagueSchedule", {}).get("seasonYear") != season:
        print(f"  schedule is for {js.get('leagueSchedule', {}).get('seasonYear')}, not {season}; skipped")
        return
    (OUT / season).mkdir(parents=True, exist_ok=True)
    (OUT / season / "schedule.json.gz").write_bytes(gzip.compress(data))
    n_games = sum(len(d["games"]) for d in js["leagueSchedule"]["gameDates"])
    print(f"  schedule: {n_games} games -> {OUT / season / 'schedule.json.gz'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", nargs="+", default=["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"])
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads from cdn.nba.com")
    ap.add_argument("--stats-workers", type=int, default=2,
                    help="parallel per-game calls to stats.nba.com when the CDN refuses")
    ap.add_argument("--skip-stats", action="store_true", help="only play-by-play and box scores")
    ap.add_argument("--refresh", action="store_true",
                    help="season in progress: re-download the game log and season stats, and the schedule")
    args = ap.parse_args()
    for season in args.seasons:
        print(f"== {season}", flush=True)
        if args.refresh:
            refresh(season)
        if not args.skip_stats or args.refresh:
            fetch_season_stats(season)
        cdn_ids, stats_ids = fetch_games(season, args.workers, args.stats_workers)
        pack(season, cdn_ids, stats_ids)


if __name__ == "__main__":
    main()
