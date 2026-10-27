"""One command for the live forecast routine: refresh data, score past forecasts, freeze new ones, publish.

    python scripts/weekly.py                    # next 7 days
    python scripts/weekly.py --days 2 --out "Embiid, Haliburton"
    python scripts/weekly.py --no-push          # everything except git push

Steps (each one is a script you can also run alone):
  1. fetch_nba.py --refresh      new games, season stats and the schedule (needs nba.com access)
  2. build_dataset.py            possessions for the current season
  3. fetch_lines.py              market lines and ESPN BPI for the window (optional; failures are fine)
  4. forecast.py score           scoreboard of every finished game so far
  5. forecast.py freeze          predictions for games in the window (refits models once a week)
  6. git commit + push           to the `forecasts` branch, checked out at ../nba-forecasts

The commit time on GitHub is the public proof that a forecast existed before tip-off. Run it daily
or weekly; a game counts once, for the latest forecast frozen before it started.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT.parent / "nba-forecasts"
BRANCH = "forecasts"
README = """# 前瞻预测存档 / forecast archive

nba-tactics-sim 的赛前预测，每批在开赛前提交。每个文件夹是一次冻结（UTC 时间），manifest.json 里有每个文件的
sha256、使用的代码 commit 和数据截止日期。记分牌：<赛季>/scoreboard.md。规则：main 分支 docs/forecasting.md。

Pre-game forecasts from nba-tactics-sim, committed before tip-off. Each folder is one freeze (UTC time);
manifest.json holds the sha256 of every file, the code commit and the data cut-off.
Scoreboard: <season>/scoreboard.md. Rules: docs/forecasting.md on main.
"""


def run(cmd: list[str], check: bool = True, cwd: Path = ROOT) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if check and r.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)}")
    return r.returncode


def out(cmd: list[str], cwd: Path = ROOT) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def ensure_worktree() -> None:
    if (PUB / ".git").exists():
        return
    run(["git", "fetch", "origin"], check=False)
    if out(["git", "ls-remote", "--heads", "origin", BRANCH]):
        if out(["git", "rev-parse", "--verify", BRANCH]):
            run(["git", "worktree", "add", str(PUB), BRANCH])
        else:
            run(["git", "worktree", "add", "--track", "-b", BRANCH, str(PUB), f"origin/{BRANCH}"])
        return
    # first run: a new branch with no history, holding only forecasts
    run(["git", "worktree", "add", "--detach", str(PUB)])
    run(["git", "checkout", "--orphan", BRANCH], cwd=PUB)
    run(["git", "rm", "-rf", "-q", "."], cwd=PUB)
    (PUB / "README.md").write_text(README, encoding="utf-8")
    run(["git", "add", "README.md"], cwd=PUB)
    run(["git", "commit", "-q", "-m", "Forecast archive"], cwd=PUB)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--out", default="", help="players known to be out, comma separated")
    ap.add_argument("--season", default=None)
    ap.add_argument("--skip-fetch", action="store_true", help="use the data already on disk")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    y = now.year if now.month >= 8 else now.year - 1
    season = args.season or f"{y}-{(y + 1) % 100:02d}"
    py = sys.executable

    ensure_worktree()
    if not args.skip_fetch:
        run([py, "scripts/fetch_nba.py", "--seasons", season, "--refresh"])
        run([py, "scripts/build_dataset.py", "--seasons", season])
        if run([py, "scripts/fetch_lines.py", "--days", str(int(args.days) + 1), "--season", season], check=False):
            print("market lines not captured this time; the forecast goes on without them")
    if (PUB / season).exists():
        run([py, "scripts/forecast.py", "score", "--season", season, "--dir", str(PUB)], check=False)
    freeze = [py, "scripts/forecast.py", "freeze", "--season", season, "--days", str(args.days), "--dir", str(PUB)]
    if args.out:
        freeze += ["--out", args.out]
    run(freeze)

    run(["git", "add", "-A"], cwd=PUB)
    if not out(["git", "status", "--porcelain"], cwd=PUB):
        print("nothing new to publish")
        return
    run(["git", "commit", "-q", "-m", f"Forecasts frozen {now:%Y-%m-%d %H:%M} UTC"], cwd=PUB)
    if args.no_push:
        print(f"committed in {PUB}; not pushed (--no-push)")
        return
    if out(["git", "ls-remote", "--heads", "origin", BRANCH], cwd=PUB):
        run(["git", "pull", "--rebase", "-q", "origin", BRANCH], cwd=PUB)
    run(["git", "push", "-u", "origin", BRANCH], cwd=PUB)
    print(f"\npublished: https://github.com/LianruiBruce/nba-tactics-sim/tree/{BRANCH}/{season}")


if __name__ == "__main__":
    main()
