"""Player and team names from the packed stats.nba.com game logs."""
from __future__ import annotations

import gzip
import json
import unicodedata
from functools import lru_cache
from pathlib import Path

NBA = Path(__file__).resolve().parents[2] / "data" / "nba"


def _rows(season: str, name: str):
    p = NBA / season / "stats" / f"{name}.json.gz"
    if not p.exists():
        return []
    rs = json.loads(gzip.decompress(p.read_bytes()))["resultSets"][0]
    return [dict(zip(rs["headers"], r)) for r in rs["rowSet"]]


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


@lru_cache(maxsize=1)
def directory() -> tuple[dict, dict, dict]:
    """(player id -> name, player id -> latest team abbreviation, team abbreviation -> team id)."""
    names, team, team_ids, last = {}, {}, {}, {}
    for season in sorted(p.name for p in NBA.iterdir() if p.is_dir()):
        for r in _rows(season, "gamelog_player"):
            pid = r["PLAYER_ID"]
            names[pid] = r["PLAYER_NAME"]
            if r["GAME_DATE"] >= last.get(pid, ""):
                last[pid] = r["GAME_DATE"]
                team[pid] = r["TEAM_ABBREVIATION"]
            team_ids[r["TEAM_ABBREVIATION"]] = r["TEAM_ID"]
    return names, team, team_ids


def find_player(query: str) -> int:
    """Player id by (part of) a name, accents ignored; the most recent match wins ties."""
    names, _, _ = directory()
    q = _fold(query.strip())
    exact = [p for p, n in names.items() if _fold(n) == q]
    if exact:
        return exact[-1]
    hits = [p for p, n in names.items() if q in _fold(n)]
    if not hits:
        raise KeyError(f"no player matches {query!r}")
    if len({names[p] for p in hits}) > 1:
        raise KeyError(f"{query!r} is ambiguous: " + ", ".join(sorted({names[p] for p in hits})[:8]))
    return hits[-1]


def player_name(pid: int) -> str:
    return directory()[0].get(pid, str(pid))
