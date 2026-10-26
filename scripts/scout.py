"""Opponent scouting report from real play-by-play, NBA.com play types and the player model.

    python scripts/scout.py --team OKC                      # 2025-26 season, writes data/reports/OKC_2025-26.md
    python scripts/scout.py --team BOS --season 2026-27 --since 2026-10-20

Sections: team ratings and ranks, shot profile (offense and defense) against league average,
Synergy play types, most-used lineups, key players with model ratings, and a short game plan
derived from the numbers above.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nbasim.data.names import directory, player_name  # noqa: E402
from nbasim.realmodel import ZONES, RealModel, chances, lineup  # noqa: E402

NBA = ROOT / "data" / "nba"
PROC = ROOT / "data" / "processed"
ZONE_CN = {"rim": "篮下", "short_mid": "短中距", "long_mid": "长两分", "corner3": "底角三分", "above3": "弧顶三分"}
PT_CN = {"Isolation": "单打", "Transition": "转换进攻", "PRBallHandler": "挡拆持球", "PRRollman": "挡拆顺下/外弹",
         "Postup": "低位", "Spotup": "定点投篮", "Handoff": "手递手", "Cut": "空切", "OffScreen": "绕掩护",
         "OffRebound": "二次进攻", "Misc": "其他"}


def stats(season: str, name: str) -> pd.DataFrame:
    p = NBA / season / "stats" / f"{name}.json.gz"
    if not p.exists():
        return pd.DataFrame()
    rs = json.loads(gzip.decompress(p.read_bytes()))["resultSets"][0]
    return pd.DataFrame(rs["rowSet"], columns=rs["headers"])


def short(pid: int) -> str:
    """'S. Gilgeous-Alexander': initial plus surname, so the two Williamses stay apart."""
    n = player_name(pid)
    return f"{n[0]}. {n.split(' ', 1)[1]}" if " " in n else n


def ratings(poss: pd.DataFrame) -> pd.DataFrame:
    off = poss.groupby("offense").agg(o_poss=("points", "size"), o_pts=("points", "sum"))
    de = poss.groupby("defense").agg(d_poss=("points", "size"), d_pts=("points", "sum"))
    r = off.join(de)
    games = poss.groupby("offense").game_id.nunique()
    r["ortg"] = 100 * r.o_pts / r.o_poss
    r["drtg"] = 100 * r.d_pts / r.d_poss
    r["net"] = r.ortg - r.drtg
    r["pace"] = (r.o_poss + r.d_poss) / 2 / games
    return r


def profile(ch: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per team: turnover / trip rates per chance, shot mix, FG% by zone, offensive rebound rate."""
    g = ch.groupby(key)
    out = pd.DataFrame({"tov": g.apply(lambda d: (d.outcome == "tov").mean()),
                        "trip": g.apply(lambda d: (d.outcome == "trip").mean()),
                        "oreb": g.apply(lambda d: d.oreb.mean())})
    fga = ch[ch.outcome == "fga"]
    mix = fga.groupby([key, "zone"]).size().unstack(fill_value=0)
    mix = mix.div(mix.sum(1), axis=0)
    fg = fga.groupby([key, "zone"]).made.mean().unstack()
    for z in ZONES:
        out[f"mix_{z}"] = mix.get(z, 0)
        out[f"fg_{z}"] = fg.get(z, np.nan)
    return out


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def signed(x: float, unit: str = "") -> str:
    return f"{x:+.1f}{unit}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="team abbreviation, e.g. OKC")
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--since", default="", help="only games on or after this date")
    ap.add_argument("--model", default=str(ROOT / "data" / "models" / "realmodel.pkl"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    _, _, team_ids = directory()
    tid = team_ids[args.team.upper()]
    tname = stats(args.season, "team_base").set_index("TEAM_ID").TEAM_NAME.get(tid, args.team)
    poss = pd.read_parquet(PROC / args.season / "possessions.parquet")
    if args.since:
        poss = poss[poss.date >= args.since]
    model = RealModel.load(Path(args.model))
    ch = chances(poss)

    r = ratings(poss)
    rank = lambda col, asc=False: int(r[col].rank(ascending=asc, method="min")[tid])  # noqa: E731
    me = r.loc[tid]
    off_p, def_p = profile(ch, "offense"), profile(ch, "defense")
    lg_off = profile(ch.assign(all=0), "all").iloc[0]

    L = []
    L.append(f"# {tname} 球探报告（{args.season}{'，自 ' + args.since if args.since else ''}）\n")
    L.append(f"数据：{poss[(poss.offense == tid) | (poss.defense == tid)].game_id.nunique()} 场比赛的逐回合记录、"
             f"NBA.com 战术类型统计，球员评分来自 nbasim 真实数据模型（训练截至 {getattr(model, 'trained_through', '?')}）。\n")
    L.append("## 球队概况\n")
    L.append("| 指标 | 数值 | 联盟排名 |\n| --- | --- | --- |")
    L.append(f"| 进攻效率（每 100 回合） | {me.ortg:.1f} | {rank('ortg')} |")
    L.append(f"| 防守效率 | {me.drtg:.1f} | {rank('drtg', True)} |")
    L.append(f"| 净效率 | {me.net:+.1f} | {rank('net')} |")
    L.append(f"| 节奏（每场回合） | {me.pace:.1f} | {rank('pace')} |\n")

    L.append("## 投篮结构（与联盟平均对比）\n")
    L.append("| 区域 | 进攻出手占比 | 进攻命中率 | 防守允许出手占比 | 防守允许命中率 |\n| --- | --- | --- | --- | --- |")
    o, d = off_p.loc[tid], def_p.loc[tid]
    for z in ZONES:
        L.append(f"| {ZONE_CN[z]} | {pct(o[f'mix_{z}'])}（{signed(100 * (o[f'mix_{z}'] - lg_off[f'mix_{z}']))}） "
                 f"| {pct(o[f'fg_{z}'])}（{signed(100 * (o[f'fg_{z}'] - lg_off[f'fg_{z}']))}） "
                 f"| {pct(d[f'mix_{z}'])}（{signed(100 * (d[f'mix_{z}'] - lg_off[f'mix_{z}']))}） "
                 f"| {pct(d[f'fg_{z}'])}（{signed(100 * (d[f'fg_{z}'] - lg_off[f'fg_{z}']))}） |")
    L.append(f"\n每次机会的失误率 {pct(o.tov)}（联盟 {pct(lg_off.tov)}），制造对手失误 {pct(d.tov)}；"
             f"造罚球回合 {pct(o.trip)}（联盟 {pct(lg_off.trip)}）；前场篮板率 {pct(o.oreb)}，"
             f"允许对手前场篮板 {pct(d.oreb)}（联盟 {pct(lg_off.oreb)}）。括号里是和联盟平均的差（百分点）。\n")

    pt_rows = {}
    for grouping in ("offensive", "defensive"):
        rows = []
        for pt in PT_CN:
            df = stats(args.season, f"playtype_T_{grouping}_{pt}")
            if df.empty or tid not in set(df.TEAM_ID):
                continue
            x = df[df.TEAM_ID == tid].iloc[0]
            rows.append((pt, x.POSS_PCT, x.PPP, x.PERCENTILE, df.PPP.mean()))
        pt_rows[grouping] = sorted(rows, key=lambda t: -t[1])
    if pt_rows["offensive"]:
        L.append("## 战术类型（NBA.com / Synergy）\n")
        L.append("| 进攻战术 | 使用占比 | 每回合得分 | 联盟平均 | 百分位 |\n| --- | --- | --- | --- | --- |")
        for pt, freq, ppp, perc, lg in pt_rows["offensive"][:8]:
            L.append(f"| {PT_CN[pt]} | {pct(freq)} | {ppp:.3f} | {lg:.3f} | {100 * perc:.0f} |")
        L.append("\n| 防守战术 | 对手使用占比 | 允许每回合得分 | 联盟平均 | 百分位（越高越好） |\n| --- | --- | --- | --- | --- |")
        for pt, freq, ppp, perc, lg in pt_rows["defensive"][:8]:
            L.append(f"| {PT_CN[pt]} | {pct(freq)} | {ppp:.3f} | {lg:.3f} | {100 * perc:.0f} |")
        L.append("")

    L.append("## 常用五人组\n")
    tp = poss[poss.offense == tid].groupby("off_lineup").agg(o=("points", "size"), op=("points", "sum"))
    td = poss[poss.defense == tid].groupby("def_lineup").agg(d=("points", "size"), dp=("points", "sum"))
    lu = tp.join(td, how="inner")
    lu["n"] = lu.o + lu.d
    lu = lu.sort_values("n", ascending=False).head(8)
    L.append("| 五人组 | 回合 | 进攻效率 | 防守效率 | 净效率 | 模型预期净效率 |\n| --- | --- | --- | --- | --- | --- |")
    # model expectation against a league-average opponent: the five most-used players of the opponents faced
    opp = poss[poss.offense != tid].off_lineup.value_counts().index[:200]
    opp5 = [lineup(s) for s in opp[:40]]
    for key, x in lu.iterrows():
        five = lineup(key)
        exp = np.mean([model.expected_points(five, o5, False) - model.expected_points(o5, five, False) for o5 in opp5])
        L.append(f"| {', '.join(short(p) for p in five)} | {int(x.n)} | {100 * x.op / x.o:.1f} | "
                 f"{100 * x.dp / x.d:.1f} | {100 * (x.op / x.o - x.dp / x.d):+.1f} | {100 * exp:+.1f} |")
    L.append("\n“模型预期净效率”是用球员模型算这套阵容对联盟常见五人组的平均结果，样本小的阵容看它比看实际净效率更可靠。\n")

    L.append("## 关键球员（模型评分）\n")
    on = {}
    for s in poss[poss.offense == tid].off_lineup:
        for p in lineup(s):
            on[p] = on.get(p, 0) + 1
    fga = ch[(ch.outcome == "fga") & (ch.offense == tid)]
    took = fga.groupby("shooter").size()
    top = sorted(on, key=lambda p: -on[p])[:9]
    L.append("| 球员 | 在场回合 | 出手占比 | 篮下 | 中距离 | 三分 | 失误 | 造罚球 | 篮下防守 | 三分防守 |\n"
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    fga_on = {p: sum(p in lineup(s) for s in fga.off_lineup) for p in top}
    for p in top:
        c = model.player_card(p)
        L.append(f"| {player_name(p)} | {on[p]} | {pct(took.get(p, 0) / max(fga_on[p], 1))} "
                 f"| {signed(c['shoot_rim'])} | {signed(c['shoot_mid'])} | {signed(c['shoot_three'])} "
                 f"| {signed(c['tov_off'])} | {signed(c['trip_off'])} | {signed(c['guard_rim'])} | {signed(c['guard_three'])} |")
    L.append("\n投篮列是他出手时命中率相对联盟平均的变化（百分点，已考虑区域）；失误、造罚球是他在场时球队每次机会的变化；"
             "防守列是对手在他在场时该区域命中率的变化，负数代表防得好。\n")

    L.append("## 比赛策略（由以上数据自动生成）\n")
    tips = []
    weak = sorted(ZONES, key=lambda z: -(d[f"fg_{z}"] - lg_off[f"fg_{z}"]))[0]
    tips.append(f"他们在{ZONE_CN[weak]}的防守最弱：允许命中率 {pct(d[f'fg_{weak}'])}，"
                f"比联盟高 {signed(100 * (d[f'fg_{weak}'] - lg_off[f'fg_{weak}']))} 个百分点，多找这个区域的出手。")
    if pt_rows["defensive"]:
        worst = min(pt_rows["defensive"][:6], key=lambda t: t[3])
        tips.append(f"防守最差的战术类型是{PT_CN[worst[0]]}：允许每回合 {worst[2]:.3f} 分（联盟百分位 {100 * worst[3]:.0f}），可以多打。")
    if pt_rows["offensive"]:
        good = [t for t in pt_rows["offensive"][:8] if t[3] >= 0.6] or pt_rows["offensive"][:3]
        best = max(good, key=lambda t: t[1])
        tips.append(f"他们最依赖{PT_CN[best[0]]}（占 {pct(best[1])}，每回合 {best[2]:.3f} 分），重点限制。")
    if o.tov > lg_off.tov:
        tips.append(f"他们失误偏多（{pct(o.tov)} 对联盟 {pct(lg_off.tov)}），施压持球人有回报。")
    if o.oreb > lg_off.oreb + 0.02:
        tips.append(f"他们前场篮板很凶（{pct(o.oreb)}），投篮后要先卡位。")
    worst_lu = lu.assign(net=100 * (lu.op / lu.o - lu.dp / lu.d)).sort_values("net").iloc[0]
    tips.append(f"他们最弱的常用阵容是 {', '.join(short(p) for p in lineup(worst_lu.name))}（净效率 {worst_lu.net:+.1f}），这套人在场时加快节奏。")
    L += [f"- {t}" for t in tips]

    out = Path(args.out or ROOT / "data" / "reports" / f"{args.team.upper()}_{args.season}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
