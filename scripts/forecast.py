"""Freeze forward forecasts before games, then score them against what happened.

    python scripts/forecast.py freeze                 # games tipping off in the next 7 days
    python scripts/forecast.py freeze --days 2 --out "Haliburton, Embiid"   # also mark these players out
    python scripts/forecast.py score                  # -> <dir>/<season>/scoreboard.md

freeze writes <dir>/<season>/<UTC time>/ with
    games.csv      per game: projected pace, margin and home win probability for every model
                   (real = our player model, rapm, team, and market / bpi when scripts/fetch_lines.py ran)
    lineups.csv    per game and team: the five most likely units and their expected net rating
    models/        the exact fitted models used, so the lineup scoring can be re-run by anyone
    manifest.json  freeze time, data cut-off, code commit, settings and sha256 of every file

Players on ESPN's injury list as Out (scripts/fetch_lines.py, last 36 hours) are left out of the
projected rotations, together with any --out players.

score recomputes the whole scoreboard from every batch. A game counts once, for the latest batch
frozen before tip-off; when the batch is committed to git, the later of its own timestamp and the
commit time is used. Rules and settings: data/forecast_config.json (fixed before the season).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nbasim import forecast as F  # noqa: E402
from nbasim.baselines import RidgeModel  # noqa: E402
from nbasim.data.names import find_player  # noqa: E402
from nbasim.realmodel import RealModel  # noqa: E402

PROC = ROOT / "data" / "processed"
NBA = ROOT / "data" / "nba"
LINES = ROOT / "data" / "lines"
CACHE = ROOT / "data" / "models" / "forecast"
CONFIG = ROOT / "data" / "forecast_config.json"
OURS = "real"
SOURCE_CN = {"real": "我们的模型", "rapm": "RAPM", "team": "球队评分", "market": "博彩盘口", "bpi": "ESPN BPI"}


def default_dir() -> Path:
    side = ROOT.parent / "nba-forecasts"
    return side if side.exists() else ROOT / "forecasts"


def season_of(t: datetime) -> str:
    y = t.year if t.month >= 8 else t.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args, cwd=ROOT) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def load_poss() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted(PROC.glob("*/possessions.parquet"))], ignore_index=True)


# ------------------------------------------------------------------ models
def fit_models(poss: pd.DataFrame, cfg: dict, verbose: bool = True) -> dict:
    h = cfg["hyper"]
    real = RealModel.fit(poss, C={k: h["real_C"] for k in ("tov", "trip", "make", "oreb", "zone")}, verbose=verbose)
    real.fit_residual(poss, h["resid_alpha"])
    real.trained_through = str(poss.date.max())
    real._tab = None
    return {"real": real, "rapm": RidgeModel("rapm", h["rapm_alpha"]).fit(poss),
            "team": RidgeModel("team", h["team_alpha"]).fit(poss)}


def save_models(models: dict, d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for k, m in models.items():
        if k == "real":
            m.save(d / "real.pkl")
        else:
            pd.to_pickle(m, d / f"{k}.pkl")


def load_models(d: Path) -> dict:
    out = {}
    for p in sorted(d.glob("*.pkl")):
        out[p.stem] = RealModel.load(p) if p.stem == "real" else pd.read_pickle(p)
    return out


def cached_models(poss: pd.DataFrame, cfg: dict, refit_days: int, cache: Path = CACHE) -> tuple[dict, Path]:
    """Reuse models fit at most refit_days of data ago; otherwise refit on everything."""
    through = str(poss.date.max())
    have = sorted(p for p in cache.glob("*") if (p / "real.pkl").exists() and p.name <= through)
    if have:
        last = have[-1]
        age = (pd.Timestamp(through) - pd.Timestamp(last.name)).days
        if age < refit_days:
            print(f"models fit on data through {last.name} ({age} days old), reusing")
            return load_models(last), last
    print(f"fitting models on {len(poss):,} possessions through {through} (a few minutes)")
    models = fit_models(poss, cfg, verbose=False)
    save_models(models, cache / through)
    return models, cache / through


# ------------------------------------------------------------------ public lines
def attach_lines(games: pd.DataFrame, season: str, now: datetime, max_age_h: float) -> tuple[pd.DataFrame, str | None]:
    """Market spread / moneyline and ESPN BPI captured by scripts/fetch_lines.py, if fresh."""
    p = LINES / season / "lines.csv"
    if not p.exists():
        return games, None
    ln = pd.read_csv(p, dtype={"date": str})
    ln["fetched"] = pd.to_datetime(ln.fetched_at, utc=True)
    ln = ln[(ln.fetched <= now) & (ln.fetched >= now - timedelta(hours=max_age_h))]
    if ln.empty:
        return games, None
    ln = ln.sort_values("fetched").groupby(["date", "home_tri", "away_tri"]).last().reset_index()
    ln["margin_market"] = -ln.spread_home
    ln["wp_market"] = [devig(h, a) for h, a in zip(ln.home_ml, ln.away_ml)]
    ln["wp_bpi"] = ln.bpi_home_wp
    keep = ["date", "home_tri", "away_tri", "margin_market", "wp_market", "wp_bpi", "fetched_at", "provider"]
    out = games.merge(ln[keep], on=["date", "home_tri", "away_tri"], how="left")
    return out.rename(columns={"fetched_at": "lines_fetched_at"}), str(ln.fetched_at.max())


def devig(home_ml, away_ml) -> float:
    def implied(ml):
        ml = float(ml)
        return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)
    try:
        if pd.isna(home_ml) or pd.isna(away_ml):
            return np.nan
        h, a = implied(home_ml), implied(away_ml)
        return round(h / (h + a), 4)
    except (TypeError, ValueError):
        return np.nan


def injured_out(season: str, now: datetime, max_age_h: float) -> list[int]:
    """Players ESPN lists as Out, matched to ids by exact name (suffixes and accents ignored)."""
    p = LINES / season / "injuries.csv"
    if not p.exists():
        return []
    inj = pd.read_csv(p)
    fetched = pd.to_datetime(inj.fetched_at, utc=True)
    inj = inj[(inj.status == "Out") & (fetched <= now) & (fetched >= now - timedelta(hours=max_age_h))]
    if inj.empty:
        return []
    from nbasim.data.names import _fold, directory
    def key(n):
        return " ".join(t for t in _fold(str(n)).replace(".", "").split() if t not in ("jr", "sr", "ii", "iii", "iv"))
    by_name = {}
    for pid, n in directory()[0].items():
        by_name.setdefault(key(n), set()).add(pid)
    ids, missed = [], []
    for n in inj.player:
        hit = by_name.get(key(n), set())
        if len(hit) == 1:
            ids.append(next(iter(hit)))
        else:
            missed.append(n)
    if missed:
        print(f"injury list: no unique match for {', '.join(map(str, missed))}")
    return ids


# ------------------------------------------------------------------ freeze
def cmd_freeze(args, cfg, poss: pd.DataFrame | None = None) -> None:
    now = datetime.now(timezone.utc) if not args.now else F.utc(args.now).to_pydatetime()
    season = args.season or season_of(now)
    if getattr(args, "schedule_from_box", False):        # past seasons, for walk-forward tests
        sched = F.schedule_from_boxscores(NBA / season / "boxscores.jsonl.gz").assign(final=False)
    else:
        sched_path = NBA / season / "schedule.json.gz"
        if not sched_path.exists():
            raise SystemExit(f"{sched_path} missing: run python scripts/fetch_nba.py --seasons {season} --refresh")
        sched = F.schedule_from_cdn(sched_path)
    tip = pd.to_datetime(sched.tip_utc, utc=True)
    upcoming = sched[(tip > now) & (tip <= now + timedelta(days=args.days)) & ~sched.final]
    if upcoming.empty:
        print("no games in the window")
        return
    poss = load_poss() if poss is None else poss
    hist = poss[pd.to_datetime(poss.date) < pd.Timestamp(now.date())]
    lag = (pd.Timestamp(now.date()) - pd.Timestamp(hist.date.max())).days
    if lag > 3:
        print(f"warning: newest game in the data is {hist.date.max()} ({lag} days ago); did the fetch work?")
    models, model_dir = cached_models(hist, cfg, args.refit_days, Path(getattr(args, "cache", None) or CACHE))
    out_players = [find_player(x) for x in args.out.split(",") if x.strip()] if args.out else []
    injured = injured_out(season, now, cfg["lines_max_age_hours"])
    out_players = sorted(set(out_players) | set(injured))
    print(f"predicting {len(upcoming)} games ({upcoming.date.min()} to {upcoming.date.max()})")
    games, lineups = F.predict_games(hist, upcoming, models, cfg["sigma"], out_players, cfg["rotation"])
    games, lines_at = attach_lines(games, season, now, cfg["lines_max_age_hours"])

    stamp = now.strftime("%Y-%m-%dT%H%MZ")
    d = Path(args.dir) / season / stamp
    d.mkdir(parents=True, exist_ok=True)
    games.to_csv(d / "games.csv", index=False)
    lineups.to_csv(d / "lineups.csv", index=False)
    shutil.copytree(model_dir, d / "models", dirs_exist_ok=True)
    raw = LINES / season / "raw_latest.json.gz"
    if lines_at and raw.exists():
        shutil.copy(raw, d / "lines_raw.json.gz")
    head = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--", "nbasim", "scripts", "data/forecast_config.json"))
    manifest = {
        "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "season": season, "window_days": args.days,
        "data_through": str(hist.date.max()), "possessions": int(len(hist)),
        "models_trained_through": model_dir.name, "code_commit": head, "code_dirty": dirty,
        "config": cfg, "out_players": out_players, "out_from_injury_list": len(injured), "lines_fetched_at": lines_at,
        "files": {str(p.relative_to(d)).replace("\\", "/"): sha256(p) for p in sorted(d.rglob("*"))
                  if p.is_file() and p.name != "manifest.json"},
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    show = games[["date", "away_tri", "home_tri"] + [c for c in games if c.startswith(("margin_", "wp_"))]]
    if not getattr(args, "quiet", False):
        print(show.round(3).to_string(index=False))
    print(f"frozen -> {d}")


# ------------------------------------------------------------------ score
def batches(root: Path) -> list[dict]:
    out = []
    for m in sorted(root.glob("*/manifest.json")):
        man = json.loads(m.read_text())
        created = pd.Timestamp(man["created_at"])
        commit = git("log", "--diff-filter=A", "--format=%cI", "--", m.name, cwd=m.parent)
        committed = pd.Timestamp(commit.splitlines()[-1]).tz_convert("UTC") if commit else None
        frozen = max(created, committed) if committed is not None else created
        out.append({"dir": m.parent, "manifest": man, "created": created, "committed": committed, "frozen": frozen})
    return out


def assign(bs: list[dict], final: pd.DataFrame) -> pd.DataFrame:
    """For each finished game, the prediction from the latest batch frozen before tip-off."""
    rows = []
    for b in bs:
        g = pd.read_csv(b["dir"] / "games.csv", dtype={"game_id": str, "date": str})
        g["batch"] = b["dir"].name
        g["frozen"] = b["frozen"]
        g["published"] = b["committed"] is not None
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    g = pd.concat(rows, ignore_index=True)
    g = g[g.frozen < pd.to_datetime(g.tip_utc, utc=True)]
    g = g.sort_values("frozen").groupby("game_id").last().reset_index()
    act = final.set_index("game_id")
    g = g[g.game_id.isin(act.index)]
    g["actual_margin"] = (act.home_pts - act.away_pts).loc[g.game_id].values
    return g


def score_lineups(bs: list[dict], games: pd.DataFrame, poss: pd.DataFrame, sources: list[str]):
    """Ex-ante units (lineups.csv) and every unit's possessions re-predicted by the frozen models."""
    ex, cond = [], []
    for b in bs:
        mine = games[games.batch == b["dir"].name]
        if mine.empty:
            continue
        gids = set(mine.game_id)
        p = poss[poss.game_id.isin(gids)]
        lu = pd.read_csv(b["dir"] / "lineups.csv", dtype={"game_id": str, "lineup": str})
        lu = lu[lu.game_id.isin(gids)]
        o = p.groupby(["game_id", "offense", "off_lineup"]).points.agg(["size", "sum"])
        o.index.names = ["game_id", "team", "lineup"]
        d = p.groupby(["game_id", "defense", "def_lineup"]).points.agg(["size", "sum"])
        d.index.names = ["game_id", "team", "lineup"]
        lu = lu.join(o.rename(columns={"size": "n_off", "sum": "pts_off"}), on=["game_id", "team", "lineup"])
        lu = lu.join(d.rename(columns={"size": "n_def", "sum": "pts_def"}), on=["game_id", "team", "lineup"])
        lu[["n_off", "pts_off", "n_def", "pts_def"]] = lu[["n_off", "pts_off", "n_def", "pts_def"]].fillna(0)
        lu["batch"] = b["dir"].name
        ex.append(lu)
        models = load_models(b["dir"] / "models")
        pc = p.assign(**{f"pred_{k}": m.predict(p) for k, m in models.items()})
        cond.append(pc[["game_id", "offense", "defense", "off_lineup", "def_lineup", "points"]
                       + [f"pred_{k}" for k in models]])
    ex = pd.concat(ex, ignore_index=True) if ex else pd.DataFrame()
    if not ex.empty:
        # a unit's prediction over a batch: per-game predictions weighted by the possessions it played
        ex["n"] = ex.n_off + ex.n_def
        agg = {"n_off": "sum", "pts_off": "sum", "n_def": "sum", "pts_def": "sum", "n": "sum"}
        for s in sources:
            if f"net_{s}" in ex:
                ex[f"wnet_{s}"] = ex[f"net_{s}"] * ex.n
                agg[f"wnet_{s}"] = "sum"
        ex = ex.groupby(["batch", "team", "lineup"]).agg(agg).reset_index()
        for s in sources:
            if f"wnet_{s}" in ex:
                ex[f"net_{s}"] = ex[f"wnet_{s}"] / ex.n.clip(lower=1)
    cond = pd.concat(cond, ignore_index=True) if cond else pd.DataFrame()
    return ex, cond


def cond_units(cond: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Season-to-date unit table from possession-level predictions of the frozen models."""
    agg = {"n": ("points", "size"), "pts": ("points", "sum")} | {f"p_{k}": (f"pred_{k}", "sum") for k in models}
    o = cond.groupby(["offense", "off_lineup"]).agg(**agg)
    d = cond.groupby(["defense", "def_lineup"]).agg(**agg)
    o.index.names = d.index.names = ["team", "lineup"]
    u = o.join(d, lsuffix="_o", rsuffix="_d", how="inner")
    out = pd.DataFrame({"n_off": u.n_o, "pts_off": u.pts_o, "n_def": u.n_d, "pts_def": u.pts_d})
    for k in models:
        out[f"net_{k}"] = 100 * (u[f"p_{k}_o"] / u.n_o - u[f"p_{k}_d"] / u.n_d)
    return out.reset_index()


def md(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table without extra dependencies."""
    if index:
        df = df.reset_index().rename(columns={"index": ""})
    def cell(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.4g}" if abs(v) < 1e5 else f"{v:.0f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def fmt_paired(r, unit: str) -> str:
    if r is None:
        return "样本不足"
    n, m, lo, hi = r
    verdict = "显著更好" if hi < 0 else ("显著更差" if lo > 0 else "未分胜负")
    return f"{m:+.3f}{unit}（95% 区间 {lo:+.3f} 到 {hi:+.3f}，{n} 场）— {verdict}"


def cmd_score(args, cfg, quiet: bool = False) -> None:
    season = args.season or season_of(datetime.now(timezone.utc))
    root = Path(args.dir) / season
    bs = batches(root)
    if not bs:
        raise SystemExit(f"no forecast batches under {root}")
    final = F.schedule_from_boxscores(NBA / season / "boxscores.jsonl.gz")
    final = final[final.final]
    games = assign(bs, final)
    if games.empty:
        print("no finished games with a forecast frozen before tip-off yet")
        return
    sources = [s for s in ["real", "rapm", "team", "market", "bpi"]
               if f"margin_{s}" in games or f"wp_{s}" in games]
    tip = pd.to_datetime(games.tip_utc, utc=True)
    games["week"] = (tip.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.to_period("W-SUN")
                     .dt.start_time.dt.strftime("%Y-%m-%d"))
    games.to_csv(root / "scored_games.csv", index=False)
    poss = pd.read_parquet(PROC / season / "possessions.parquet")
    ex, cond = score_lineups(bs, games, poss, sources)

    L = [f"# {season} 前瞻预测记分牌", "",
         f"更新于 {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC。{len(bs)} 批预测，{len(games)} 场已完赛比赛"
         f"（{int(games.published.sum())} 场的预测在开赛前已提交到 git）。规则见 docs/forecasting.md。", "",
         "## 比赛：分差和胜负", ""]
    gm = F.game_metrics(games, sources)
    gm.index = [SOURCE_CN.get(s, s) for s in gm.index]
    L += [md(gm), ""]
    L += ["我们的模型减去对手（负数 = 我们更准）：", ""]
    for s in sources:
        if s == OURS:
            continue
        L.append(f"- 对 {SOURCE_CN.get(s, s)}：分差平方误差 {fmt_paired(F.paired(games, OURS, s, 'sq'), '')}；"
                 f"Brier {fmt_paired(F.paired(games, OURS, s, 'brier'), '')}")
    L += ["", "## 每周", ""]
    wk = []
    for w, g in games.groupby("week"):
        m = F.game_metrics(g, sources)
        row = {"周（周一）": w, "场数": len(g)}
        for s in sources:
            if s in m.index and "margin_rmse" in m and not pd.isna(m.loc[s, "margin_rmse"]) and f"margin_{s}" in g:
                row[f"{SOURCE_CN.get(s, s)} RMSE"] = round(m.loc[s, "margin_rmse"], 2)
            if s in m.index and "brier" in m and not pd.isna(m.loc[s].get("brier", np.nan)):
                row[f"{SOURCE_CN.get(s, s)} Brier"] = round(m.loc[s, "brier"], 4)
        wk.append(row)
    L += [md(pd.DataFrame(wk), index=False), ""]

    if not ex.empty:
        ex.to_csv(root / "scored_lineups_exante.csv", index=False)
        lm = F.lineup_metrics(ex, [s for s in sources if s in ("real", "rapm", "team")])
        lm.index = [SOURCE_CN.get(s, s) for s in lm.index]
        L += ["## 阵容（赛前预测）", "",
              "每场赛前列出每队最可能的 5 套五人组和预期净效率（每 100 回合，对手按其预计轮换）。"
              "这里把每批里同一五人组的实际回合合在一起，按回合数加权比较。单周样本很小，看累计。", "",
              md(lm), ""]
    if not cond.empty:
        mods = [c[5:] for c in cond if c.startswith("pred_")]
        cu = cond_units(cond, mods)
        cu.to_csv(root / "scored_lineups_conditional.csv", index=False)
        lm = F.lineup_metrics(cu, mods, min_poss=cfg["scoring"]["min_unit_possessions"])
        lm.index = [SOURCE_CN.get(s, s) for s in lm.index]
        L += ["## 阵容（给定实际上场人员）", "",
              f"用每批冻结的模型，按实际上场的十个人预测每个回合，赛季累计后比较五人组净效率"
              f"（至少 {cfg['scoring']['min_unit_possessions']} 回合的五人组）。这和 backtest.py 的阵容指标同一口径。", "",
              md(lm) if not lm.empty else "还没有五人组达到回合数门槛。", ""]
    L += ["## 每场明细（最近 30 场）", ""]
    cols = ["date", "away_tri", "home_tri", "actual_margin"] + [f"margin_{s}" for s in sources if f"margin_{s}" in games] \
        + [f"wp_{s}" for s in sources if f"wp_{s}" in games] + ["batch"]
    L += [md(games.sort_values("tip_utc").tail(30)[cols].round(3), index=False), ""]
    (root / "scoreboard.md").write_text("\n".join(L), encoding="utf-8")
    if not quiet:
        print("\n".join(L[:30]))
    print(f"-> {root / 'scoreboard.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fr = sub.add_parser("freeze")
    fr.add_argument("--days", type=float, default=7, help="games tipping off within this many days")
    fr.add_argument("--now", help="pretend the current UTC time is this (for testing)")
    fr.add_argument("--out", default="", help="players known to be out, comma separated")
    fr.add_argument("--refit-days", type=int, default=None, help="refit when models are older than this")
    sc = sub.add_parser("score")
    for p in (fr, sc):
        p.add_argument("--season")
        p.add_argument("--dir", default=str(default_dir()))
    args = ap.parse_args()
    cfg = json.loads(CONFIG.read_text())
    if args.cmd == "freeze":
        if args.refit_days is None:
            args.refit_days = cfg["refit_days"]
        cmd_freeze(args, cfg)
    else:
        cmd_score(args, cfg)


if __name__ == "__main__":
    main()
