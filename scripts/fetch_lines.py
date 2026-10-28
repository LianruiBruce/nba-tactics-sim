"""Capture public predictions for upcoming games: the betting market and ESPN BPI.

    python scripts/fetch_lines.py --days 7       # run right before scripts/forecast.py freeze

Source: ESPN's public site API (no key). For each game it keeps the first sportsbook in ESPN's
pick center (point spread and moneylines; the moneylines are de-vigged into a win probability at
freeze time) and the BPI game projection (home win probability).
It also saves ESPN's injury list (data/lines/<season>/injuries.csv); the freeze leaves players
listed "Out" out of the projected rotations.
Rows are appended to data/lines/<season>/lines.csv with the time they were fetched, and the raw
responses of the latest run go to data/lines/<season>/raw_latest.json.gz. The freeze uses the
latest capture of each game from the 36 hours before it, so baselines are frozen at the same time
as our forecast.

ESPN often blocks cloud servers; run it on the same computer as scripts/fetch_nba.py.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "lines"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={e}"
INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/124.0 Safari/537.36", "Accept": "application/json"}
ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
FIELDS = ["fetched_at", "date", "home_tri", "away_tri", "espn_event", "provider", "spread_home",
          "home_ml", "away_ml", "bpi_home_wp"]


def get_json(url: str):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001 - any failure just means no line for this game
        print(f"  failed: {url} ({e})", file=sys.stderr)
        return None


def tri(abbr: str) -> str:
    return ESPN_TO_NBA.get(abbr, abbr)


def num(x):
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return None


def spread_home(pc: dict, home: str, away: str):
    """Home point spread (negative = home favoured) from ESPN's 'details' like 'BOS -6.5'."""
    det = (pc.get("details") or "").strip()
    if det.upper() in ("EVEN", "PK", "PICK"):
        return 0.0
    m = re.match(r"([A-Z]+)\s+([-+]?\d+(?:\.\d+)?)", det)
    if m:
        fav, line = tri(m.group(1)), float(m.group(2))
        if fav == home:
            return line
        if fav == away:
            return -line
    return num(pc.get("spread"))


def moneylines(pc: dict):
    h = (pc.get("homeTeamOdds") or {}).get("moneyLine")
    a = (pc.get("awayTeamOdds") or {}).get("moneyLine")
    if h is None or a is None:   # newer layout: moneyline.home.close.odds
        ml = pc.get("moneyline") or {}
        pick = lambda side: next((num((ml.get(side) or {}).get(k, {}).get("odds"))  # noqa: E731
                                  for k in ("close", "current", "open") if (ml.get(side) or {}).get(k)), None)
        h, a = pick("home"), pick("away")
    return num(h), num(a)


def injuries(stamp: str) -> list[dict]:
    js = get_json(INJURIES) or {}
    rows = []
    for team in js.get("injuries", []):
        for i in team.get("injuries", []):
            a = i.get("athlete") or {}
            rows.append({"fetched_at": stamp, "team": tri(((a.get("team") or {}).get("abbreviation")) or ""),
                         "player": a.get("displayName"), "status": i.get("status"),
                         "detail": (i.get("shortComment") or "")[:200]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--season", default=None)
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    y = now.year if now.month >= 8 else now.year - 1
    season = args.season or f"{y}-{(y + 1) % 100:02d}"
    today = now.astimezone(ZoneInfo("America/New_York")).date()
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, raw = [], {}
    for k in range(args.days + 1):
        d = today + timedelta(days=k)
        sb = get_json(SCOREBOARD.format(d=d.strftime("%Y%m%d")))
        if not sb:
            continue
        for ev in sb.get("events", []):
            comp = ev["competitions"][0]
            if comp.get("status", {}).get("type", {}).get("state") not in (None, "pre"):
                continue          # started or finished: too late for a forecast
            teams = {c["homeAway"]: tri(c["team"]["abbreviation"]) for c in comp["competitors"]}
            summ = get_json(SUMMARY.format(e=ev["id"])) or {}
            pcs = summ.get("pickcenter") or comp.get("odds") or []
            pc = pcs[0] if pcs else {}
            hml, aml = moneylines(pc)
            bpi = num(((summ.get("predictor") or {}).get("homeTeam") or {}).get("gameProjection"))
            rows.append({"fetched_at": stamp, "date": d.isoformat(), "home_tri": teams.get("home"),
                         "away_tri": teams.get("away"), "espn_event": ev["id"],
                         "provider": (pc.get("provider") or {}).get("name"),
                         "spread_home": spread_home(pc, teams.get("home"), teams.get("away")) if pc else None,
                         "home_ml": hml, "away_ml": aml,
                         "bpi_home_wp": round(bpi / 100, 4) if bpi is not None else None})
            raw[ev["id"]] = {"event": ev, "pickcenter": summ.get("pickcenter"), "predictor": summ.get("predictor")}
    out = OUT / season
    out.mkdir(parents=True, exist_ok=True)
    inj = injuries(stamp)
    if inj:
        with open(out / "injuries.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(inj[0]))
            w.writeheader()
            w.writerows(inj)
        print(f"{len(inj)} injury entries ({sum(r['status'] == 'Out' for r in inj)} out) -> {out / 'injuries.csv'}")
    path = out / "lines.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    (out / "raw_latest.json.gz").write_bytes(gzip.compress(json.dumps(raw).encode()))
    n_line = sum(r["spread_home"] is not None for r in rows)
    n_bpi = sum(r["bpi_home_wp"] is not None for r in rows)
    print(f"{len(rows)} upcoming games: {n_line} with a market line, {n_bpi} with BPI -> {path}")


if __name__ == "__main__":
    main()
