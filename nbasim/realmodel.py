"""Player-level possession model fit on real play-by-play.

A possession is a chain of chances. Each chance ends in a turnover, a free
throw trip, or a field goal attempt; a miss can be rebounded by the offense,
which starts another chance. Every step is a regularised logistic regression
on who is on the floor (the same idea as RAPM, split by component), so each
player gets interpretable numbers: how he changes turnover odds, how often
he draws fouls, his make rate by zone, what he does to opponents' shooting
by zone, and his effect on the offensive glass.

    model = RealModel.fit(possessions)          # data/processed/*/possessions.parquet
    model.expected_points(off5, def5, home=True, ctx="def_rebound")
"""
from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

ZONES = ["rim", "short_mid", "long_mid", "corner3", "above3"]
ZGROUP = {"rim": 0, "short_mid": 1, "long_mid": 1, "corner3": 2, "above3": 2}
ZGROUP_NAMES = ["rim", "mid", "three"]
ZONE_PTS = {"rim": 2, "short_mid": 2, "long_mid": 2, "corner3": 3, "above3": 3}
CONTEXTS = ["period_start", "def_rebound", "made_fg", "made_ft", "turnover", "other", "oreb", "retained"]
FIXED = 4.0  # scale for context columns, so the ridge penalty barely touches them

SEASON_WEIGHT = {"2022-23": 0.4, "2023-24": 0.6, "2024-25": 0.8, "2025-26": 1.0, "2026-27": 1.0}


def lineup(s: str) -> list[int]:
    return [int(x) for x in s.split("-") if x]


def chances(poss: pd.DataFrame) -> pd.DataFrame:
    """Expand possessions into one row per chance."""
    rows = []
    cols = ["poss_id", "season", "date", "game_id", "offense", "defense", "home",
            "off_lineup", "def_lineup", "start_type", "events"]
    for p in poss[cols].itertuples(index=False):
        if not p.events:
            continue
        evs = [e.split("|") for e in p.events.split(";")]
        ctx = p.start_type
        i = 0
        base = (p.poss_id, p.season, p.date, p.game_id, p.offense, p.defense, bool(p.home),
                p.off_lineup, p.def_lineup)
        while i < len(evs):
            e = evs[i]
            kind = e[0]
            row = dict(zip(("poss_id", "season", "date", "game_id", "offense", "defense", "home",
                            "off_lineup", "def_lineup"), base))
            row["ctx"] = ctx
            if kind == "R":  # stray rebound (e.g. after a retained trip): just continue
                ctx = "oreb"
                i += 1
                continue
            if kind == "T":
                row.update(outcome="tov", shooter=int(e[1]))
                rows.append(row)
                ctx = "other"
                i += 1
                continue
            missed = False
            if kind == "F":
                made = e[3] == "1"
                and1 = 0
                ft_made = 0
                j = i + 1
                if made and j < len(evs) and evs[j][0] == "FT" and evs[j][5] == "a":
                    and1, ft_made = 1, int(evs[j][4])
                    j += 1
                row.update(outcome="fga", shooter=int(e[1]), zone=e[2], made=int(made),
                           and1=and1, ft_made=ft_made)
                missed = not made or (and1 and not ft_made)
                i = j
            else:  # FT trip
                shooter, of, tag = int(e[1]), int(e[3]), e[5]
                n_made, n = 0, 0
                j = i
                while j < len(evs) and evs[j][0] == "FT" and int(evs[j][1]) == shooter and n < of:
                    n_made += int(evs[j][4])
                    n += 1
                    j += 1
                    if evs[j - 1][2] == evs[j - 1][3]:
                        break
                row.update(outcome="trip", shooter=shooter, ft_n=n, ft_made=n_made, retained=int(tag == "r"))
                last_missed = evs[j - 1][4] == "0"
                missed = last_missed and tag != "r"
                if tag == "r":
                    ctx = "retained"
                i = j
            if missed:
                row["oreb"] = int(i < len(evs))  # anything after a miss means the offense kept the ball
                row["miss_type"] = row.get("zone", "ft") if row["outcome"] == "fga" else "ft"
                if i < len(evs) and evs[i][0] == "R":
                    i += 1
                ctx = "oreb"
            elif row["outcome"] == "fga":
                ctx = "other"
            rows.append(row)
    df = pd.DataFrame(rows)
    df["w"] = df.season.map(SEASON_WEIGHT).fillna(1.0)
    return df


class Design:
    """Sparse design matrix builder: named fixed columns + per-player blocks."""

    def __init__(self, fixed: list[str], blocks: list[str]):
        self.fixed = {name: i for i, name in enumerate(fixed)}
        self.blocks = blocks
        self.players: dict[int, int] = {}

    def player_ids(self, df: pd.DataFrame, cols: list[str]) -> None:
        ids = set()
        for c in cols:
            for s in df[c].unique():
                if isinstance(s, str):
                    ids.update(lineup(s))
                else:
                    ids.add(int(s))
        self.players = {p: i for i, p in enumerate(sorted(ids))}

    @property
    def width(self) -> int:
        return len(self.fixed) + len(self.blocks) * len(self.players)

    def col(self, block: str, pid: int) -> int | None:
        i = self.players.get(pid)
        if i is None:
            return None
        return len(self.fixed) + self.blocks.index(block) * len(self.players) + i


@dataclass
class Component:
    """One fitted logistic regression and the design that produced it."""
    design: Design
    coef: np.ndarray
    intercept: float

    def fixed(self, name: str) -> float:
        i = self.design.fixed.get(name)
        return 0.0 if i is None else self.coef[i] * FIXED

    def player(self, block: str, pid: int) -> float:
        c = self.design.col(block, pid)
        return 0.0 if c is None else float(self.coef[c])


@dataclass
class ZoneComponent:
    """Multinomial logit over ZONES: where the lineup's shots come from."""
    design: Design
    coef: np.ndarray       # (len(ZONES), width)
    intercept: np.ndarray  # (len(ZONES),)

    def fixed(self, name: str) -> np.ndarray:
        i = self.design.fixed.get(name)
        return np.zeros(len(ZONES)) if i is None else self.coef[:, i] * FIXED

    def player(self, block: str, pid: int) -> np.ndarray:
        c = self.design.col(block, pid)
        return np.zeros(len(ZONES)) if c is None else self.coef[:, c]


def fit_logistic(rows: list[list[tuple[int, float]]], y: np.ndarray, w: np.ndarray, width: int,
                 C: float) -> tuple[np.ndarray, float]:
    indptr, indices, data = [0], [], []
    for r in rows:
        for c, v in r:
            indices.append(c)
            data.append(v)
        indptr.append(len(indices))
    X = sparse.csr_matrix((data, indices, indptr), shape=(len(rows), width))
    m = LogisticRegression(C=C, max_iter=2000, tol=1e-6)
    m.fit(X, y, sample_weight=w)
    if m.coef_.shape[0] > 1:  # multinomial: (classes, width)
        return m.coef_, m.intercept_
    return m.coef_[0], float(m.intercept_[0])


def _ctx_cols(d: Design, ctx: str, home: bool) -> list[tuple[int, float]]:
    out = [(d.fixed["ctx_" + ctx], FIXED)] if "ctx_" + ctx in d.fixed else []
    if home:
        out.append((d.fixed["home"], FIXED))
    return out


@dataclass
class RealModel:
    tov: Component
    trip: Component
    make: Component
    oreb: Component
    zone: ZoneComponent = None
    resid: dict = None                              # optional RAPM-style correction, see fit_residual
    fga_rate: dict = field(default_factory=dict)    # player -> share weight of FGAs
    trip_rate: dict = field(default_factory=dict)   # player -> share weight of FT trips
    zone_mix: dict = field(default_factory=dict)    # player -> np.array over ZONES
    ft_pct: dict = field(default_factory=dict)
    league_mix: np.ndarray = None
    league_ft: float = 0.78
    and1: dict = field(default_factory=dict)        # zone -> P(and-one | make)
    trip_n: dict = field(default_factory=dict)      # n -> P(n FTs | trip)
    names: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, poss: pd.DataFrame, C: dict | None = None, verbose: bool = True) -> "RealModel":
        C = {"tov": 0.01, "trip": 0.01, "make": 0.01, "oreb": 0.01, **(C or {})}
        ch = chances(poss)
        log = print if verbose else (lambda *a, **k: None)
        log(f"  {len(poss):,} possessions -> {len(ch):,} chances")

        ctx_fixed = ["ctx_" + c for c in CONTEXTS[1:]] + ["home"]
        onoff = Design(ctx_fixed, ["off", "def"])
        onoff.player_ids(ch, ["off_lineup", "def_lineup"])

        def onoff_rows(df):
            rows = []
            for r in df[["off_lineup", "def_lineup", "ctx", "home"]].itertuples(index=False):
                cols = _ctx_cols(onoff, r.ctx, r.home)
                cols += [(onoff.col("off", p), 1.0) for p in lineup(r.off_lineup)]
                cols += [(onoff.col("def", p), 1.0) for p in lineup(r.def_lineup)]
                rows.append(cols)
            return rows

        log("  fitting turnovers")
        tov = Component(onoff, *fit_logistic(onoff_rows(ch), (ch.outcome == "tov").values,
                                             ch.w.values, onoff.width, C["tov"]))
        nt = ch[ch.outcome != "tov"]
        log("  fitting free throw trips")
        trip = Component(onoff, *fit_logistic(onoff_rows(nt), (nt.outcome == "trip").values,
                                              nt.w.values, onoff.width, C["trip"]))

        fga = ch[ch.outcome == "fga"]
        log("  fitting shot selection")
        zd = Design(ctx_fixed, ["off", "def"])
        zd.player_ids(fga, ["off_lineup", "def_lineup"])
        rows = []
        for r in fga[["off_lineup", "def_lineup", "ctx", "home"]].itertuples(index=False):
            cols = _ctx_cols(zd, r.ctx, r.home)
            cols += [(zd.col("off", p), 1.0) for p in lineup(r.off_lineup)]
            cols += [(zd.col("def", p), 1.0) for p in lineup(r.def_lineup)]
            rows.append(cols)
        zone_y = fga.zone.map({z: i for i, z in enumerate(ZONES)}).values
        zone = ZoneComponent(zd, *fit_logistic(rows, zone_y, fga.w.values, zd.width, C.get("zone", C["make"])))

        log("  fitting shot making")
        mk = Design(["z_" + z for z in ZONES[1:]] + ctx_fixed,
                    [f"shoot_{g}" for g in ZGROUP_NAMES] + [f"guard_{g}" for g in ZGROUP_NAMES]
                    + [f"mate_{g}" for g in ZGROUP_NAMES])
        mk.player_ids(fga, ["off_lineup", "def_lineup"])
        rows = []
        for r in fga[["shooter", "off_lineup", "def_lineup", "zone", "ctx", "home"]].itertuples(index=False):
            g = ZGROUP_NAMES[ZGROUP[r.zone]]
            cols = _ctx_cols(mk, r.ctx, r.home)
            if r.zone != "rim":
                cols.append((mk.fixed["z_" + r.zone], FIXED))
            cols.append((mk.col("shoot_" + g, r.shooter), 1.0))
            cols += [(mk.col("guard_" + g, p), 1.0) for p in lineup(r.def_lineup)]
            # teammates: creation and spacing change how good a player's looks are
            cols += [(mk.col("mate_" + g, p), 1.0) for p in lineup(r.off_lineup) if p != r.shooter]
            rows.append(cols)
        make = Component(mk, *fit_logistic(rows, fga.made.values.astype(bool), fga.w.values,
                                           mk.width, C["make"]))

        log("  fitting offensive rebounds")
        miss = ch[ch.oreb.notna()]
        ob = Design(["m_" + z for z in ZONES[1:]] + ["m_ft"] + ctx_fixed, ["off", "def"])
        ob.player_ids(miss, ["off_lineup", "def_lineup"])
        rows = []
        for r in miss[["off_lineup", "def_lineup", "miss_type", "home"]].itertuples(index=False):
            cols = [(ob.fixed["home"], FIXED)] if r.home else []
            if r.miss_type != "rim":
                cols.append((ob.fixed["m_" + r.miss_type], FIXED))
            cols += [(ob.col("off", p), 1.0) for p in lineup(r.off_lineup)]
            cols += [(ob.col("def", p), 1.0) for p in lineup(r.def_lineup)]
            rows.append(cols)
        oreb = Component(ob, *fit_logistic(rows, miss.oreb.values.astype(bool), miss.w.values,
                                           ob.width, C["oreb"]))

        m = cls(tov, trip, make, oreb, zone)
        m._fit_shares(ch)
        return m

    def _fit_shares(self, ch: pd.DataFrame, k_share: float = 150.0, k_mix: float = 60.0,
                    k_ft: float = 40.0) -> None:
        nt = ch[ch.outcome != "tov"]
        fga = nt[nt.outcome == "fga"]
        trips = nt[nt.outcome == "trip"]
        on_fga, on_trip = Counter(), Counter()
        for s, w in zip(fga.off_lineup, fga.w):
            for p in lineup(s):
                on_fga[p] += w
        for s, w in zip(trips.off_lineup, trips.w):
            for p in lineup(s):
                on_trip[p] += w
        took = fga.groupby("shooter").w.sum()
        drew = trips.groupby("shooter").w.sum()
        self.fga_rate = {p: (took.get(p, 0.0) + 0.2 * k_share) / (n + k_share) for p, n in on_fga.items()}
        self.trip_rate = {p: (drew.get(p, 0.0) + 0.2 * k_share) / (n + k_share) for p, n in on_trip.items()}

        zc = fga.groupby(["shooter", "zone"]).w.sum().unstack(fill_value=0.0).reindex(columns=ZONES, fill_value=0.0)
        self.league_mix = zc.sum().values / zc.values.sum()
        self.zone_mix = {int(p): (row.values + k_mix * self.league_mix) / (row.values.sum() + k_mix)
                         for p, row in zc.iterrows()}

        a1 = fga[fga.and1 == 1]
        ftm = pd.concat([trips.groupby("shooter").ft_made.sum(), a1.groupby("shooter").ft_made.sum()]).groupby(level=0).sum()
        fta = pd.concat([trips.groupby("shooter").ft_n.sum(), a1.groupby("shooter").and1.sum()]).groupby(level=0).sum()
        self.league_ft = float(ftm.sum() / fta.sum())
        self.ft_pct = {int(p): (ftm.get(p, 0) + k_ft * self.league_ft) / (fta[p] + k_ft) for p in fta.index}
        made = fga[fga.made == 1]
        self.and1 = made.groupby("zone").and1.mean().to_dict()
        tn = trips[trips.retained == 0].ft_n.value_counts(normalize=True)
        self.trip_n = {int(n): float(p) for n, p in tn.items()}

    # ------------------------------------------------------------- prediction
    def expected_points(self, off5, def5, home: bool = False, ctx: str = "def_rebound") -> float:
        row = pd.DataFrame({"off_lineup": ["-".join(map(str, off5))], "def_lineup": ["-".join(map(str, def5))],
                            "home": [home], "start_type": [ctx]})
        return float(self.predict(row)[0])

    def breakdown(self, off5, def5, home: bool = False, ctx: str = "def_rebound") -> dict:
        """Per-chance rates behind expected_points, for reports."""
        index, _ = self._tables()
        O = np.array([[index.get(p, 0) for p in off5]])
        D = np.array([[index.get(p, 0) for p in def5]])
        A, B, det = self._terms_vec(O, D, np.array([home]), [ctx], detail=True)
        return {k: (v[0] if np.ndim(v) == 1 else v[0]) for k, v in det.items()}

    def _tables(self):
        """Per-player coefficient arrays for vectorised prediction (row 0 = unknown player)."""
        if getattr(self, "_tab", None) is not None:
            return self._tab
        ids = set(self.fga_rate) | set(self.tov.design.players) | set(self.make.design.players) \
            | set(self.oreb.design.players)
        index = {p: i + 1 for i, p in enumerate(sorted(ids))}
        n = len(index) + 1
        t = {k: np.zeros(n) for k in ("tov_off", "tov_def", "trip_off", "trip_def", "oreb_off", "oreb_def")}
        for g in ZGROUP_NAMES:
            for k in ("shoot", "guard", "mate"):
                t[f"{k}_{g}"] = np.zeros(n)
        t["fga_rate"] = np.full(n, 0.2)
        t["trip_rate"] = np.full(n, 0.2)
        t["ft"] = np.full(n, self.league_ft)
        t["mix"] = np.tile(self.league_mix, (n, 1))
        t["zone_off"] = np.zeros((n, len(ZONES)))
        t["zone_def"] = np.zeros((n, len(ZONES)))
        for p, i in index.items():
            t["tov_off"][i], t["tov_def"][i] = self.tov.player("off", p), self.tov.player("def", p)
            t["trip_off"][i], t["trip_def"][i] = self.trip.player("off", p), self.trip.player("def", p)
            t["oreb_off"][i], t["oreb_def"][i] = self.oreb.player("off", p), self.oreb.player("def", p)
            for g in ZGROUP_NAMES:
                for k in ("shoot", "guard", "mate"):
                    t[f"{k}_{g}"][i] = self.make.player(f"{k}_{g}", p)
            t["fga_rate"][i] = self.fga_rate.get(p, 0.2)
            t["trip_rate"][i] = self.trip_rate.get(p, 0.2)
            t["ft"][i] = self.ft_pct.get(p, self.league_ft)
            if p in self.zone_mix:
                t["mix"][i] = self.zone_mix[p]
            if self.zone is not None:
                t["zone_off"][i] = self.zone.player("off", p)
                t["zone_def"][i] = self.zone.player("def", p)
        self._tab = (index, t)
        return self._tab

    def _terms_vec(self, O, D, home, ctx, detail: bool = False):
        """Vectorised chance_terms: O, D (N,5) player rows; home (N,) bool; ctx list of names."""
        _, t = self._tables()
        sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        hm = home.astype(float)
        ctx_tov = np.array([self.tov.fixed("ctx_" + c) for c in ctx])
        ctx_trip = np.array([self.trip.fixed("ctx_" + c) for c in ctx])
        ctx_make = np.array([self.make.fixed("ctx_" + c) for c in ctx])
        p_tov = sig(self.tov.intercept + ctx_tov + hm * self.tov.fixed("home")
                    + t["tov_off"][O].sum(1) + t["tov_def"][D].sum(1))
        p_trip = (1 - p_tov) * sig(self.trip.intercept + ctx_trip + hm * self.trip.fixed("home")
                                   + t["trip_off"][O].sum(1) + t["trip_def"][D].sum(1))
        p_fga = 1 - p_tov - p_trip
        ob = self.oreb.intercept + hm * self.oreb.fixed("home") + t["oreb_off"][O].sum(1) + t["oreb_def"][D].sum(1)
        p_oreb = {z: sig(ob + self.oreb.fixed("m_" + z)) for z in ZONES + ["ft"]}
        ft = t["ft"][O]
        tr = t["trip_rate"][O]
        tr = tr / tr.sum(1, keepdims=True)
        mean_n = sum(n * q for n, q in self.trip_n.items())
        A = p_trip * (tr * ft).sum(1) * mean_n
        B = p_trip * (tr * (1 - ft)).sum(1) * p_oreb["ft"]
        fr = t["fga_rate"][O]
        mix = t["mix"][O]  # (N,5 players,5 zones)
        joint = fr[:, :, None] * mix  # player propensity to take a shot from each zone
        if self.zone is not None:
            # the lineups decide where shots come from; players split each zone by propensity
            zl = (self.zone.intercept[None, :] + np.stack([self.zone.fixed("ctx_" + c) for c in ctx])
                  + hm[:, None] * self.zone.fixed("home")[None, :]
                  + t["zone_off"][O].sum(1) + t["zone_def"][D].sum(1))
            pz = np.exp(zl - zl.max(1, keepdims=True))
            pz /= pz.sum(1, keepdims=True)
            w_all = joint / joint.sum(1, keepdims=True) * pz[:, None, :]
        else:
            w_all = joint / joint.sum((1, 2), keepdims=True)
        base = self.make.intercept + ctx_make + hm * self.make.fixed("home")
        fg_pts = np.zeros(len(O))
        det = {"p_tov": p_tov, "p_trip": p_trip, "p_fga": p_fga, "p_oreb_fga": np.zeros(len(O)),
               "p_oreb_ft": p_oreb["ft"]}
        for zi, z in enumerate(ZONES):
            g = ZGROUP_NAMES[ZGROUP[z]]
            mates = t[f"mate_{g}"][O]
            logit = (base[:, None] + self.make.fixed("z_" + z) + t[f"shoot_{g}"][O]
                     + t[f"guard_{g}"][D].sum(1)[:, None] + mates.sum(1)[:, None] - mates)
            pm = sig(logit)
            w = w_all[:, :, zi]
            fg_pts += (w * pm * (ZONE_PTS[z] + self.and1.get(z, 0.0) * ft)).sum(1)
            B += p_fga * (w * (1 - pm)).sum(1) * p_oreb[z]
            if detail:
                det[f"share_{z}"] = w.sum(1)
                det[f"fg_{z}"] = (w * pm).sum(1) / np.maximum(w.sum(1), 1e-12)
                det["p_oreb_fga"] = det["p_oreb_fga"] + (w * (1 - pm)).sum(1) * p_oreb[z]
        A += p_fga * fg_pts
        if detail:
            det["shooter_share"] = w_all.sum(2)
            det["fg_pts_per_fga"] = fg_pts
            return A, B, det
        return A, B

    def predict(self, poss: pd.DataFrame) -> np.ndarray:
        """Expected points of each possession given its lineups, home flag and how it started."""
        index, _ = self._tables()

        def rows(col):
            out = np.zeros((len(poss), 5), dtype=int)
            for k, s in enumerate(poss[col].values):
                ids = [index.get(p, 0) for p in lineup(s)][:5]
                out[k, :len(ids)] = ids
            return out

        O, D = rows("off_lineup"), rows("def_lineup")
        home = poss.home.values.astype(bool)
        A2, B2 = self._terms_vec(O, D, home, ["oreb"] * len(poss))
        A, B = self._terms_vec(O, D, home, list(poss.start_type.values))
        out = A + B * A2 / (1 - B2)
        if self.resid is not None:
            out = out + self._resid_predict(poss)
        return out

    def fit_residual(self, poss: pd.DataFrame, alpha: float) -> "RealModel":
        """Ridge regression of (actual - structural) points on the ten players, added at prediction."""
        from sklearn.linear_model import Ridge
        self.resid = None
        y = poss.points.values - self.predict(poss)
        ids = sorted({p for s in pd.concat([poss.off_lineup, poss.def_lineup]).unique() for p in lineup(s)})
        idx = {p: i for i, p in enumerate(ids)}
        X = self._resid_X(poss, idx)
        w = poss.season.map(SEASON_WEIGHT).fillna(1.0).values if "season" in poss else None
        r = Ridge(alpha=alpha).fit(X, y, sample_weight=w)
        k = len(CONTEXTS) - 1
        n = len(ids)
        self.resid = {"intercept": float(r.intercept_), "ctx": dict(zip(CONTEXTS[1:], r.coef_[:k] * FIXED)),
                      "home": float(r.coef_[k] * FIXED),
                      "off": dict(zip(ids, r.coef_[k + 1:k + 1 + n])), "def": dict(zip(ids, r.coef_[k + 1 + n:]))}
        return self

    @staticmethod
    def _resid_X(poss, idx):
        n, k = len(idx), len(CONTEXTS) - 1
        rows, cols, vals = [], [], []
        for i, (o, d, st, h) in enumerate(zip(poss.off_lineup.values, poss.def_lineup.values,
                                               poss.start_type.values, poss.home.values)):
            if st in CONTEXTS[1:]:
                rows.append(i), cols.append(CONTEXTS[1:].index(st)), vals.append(FIXED)
            if h:
                rows.append(i), cols.append(k), vals.append(FIXED)
            for p in lineup(o):
                rows.append(i), cols.append(k + 1 + idx[p]), vals.append(1.0)
            for p in lineup(d):
                rows.append(i), cols.append(k + 1 + n + idx[p]), vals.append(1.0)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(poss), k + 1 + 2 * n))

    def _resid_predict(self, poss: pd.DataFrame) -> np.ndarray:
        r = self.resid
        out = np.full(len(poss), r["intercept"])
        out += np.array([r["ctx"].get(c, 0.0) for c in poss.start_type.values])
        out += poss.home.values.astype(float) * r["home"]
        out += np.array([sum(r["off"].get(p, 0.0) for p in lineup(s)) for s in poss.off_lineup.values])
        out += np.array([sum(r["def"].get(p, 0.0) for p in lineup(s)) for s in poss.def_lineup.values])
        return out

    def player_card(self, pid: int) -> dict:
        """A player's fitted effects in readable units, at league-average context.

        shoot_* / guard_*: change in make probability (percentage points) on his shots / on shots he defends
        tov_off, trip_off, oreb_off: change in his team's per-chance turnover, foul-trip and offensive rebound odds
        """
        sig = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        out = {}
        for g, zs in (("rim", ["rim"]), ("mid", ["short_mid", "long_mid"]), ("three", ["corner3", "above3"])):
            base = self.make.intercept + np.mean([self.make.fixed("z_" + z) for z in zs])
            for k in ("shoot", "guard"):
                out[f"{k}_{g}"] = 100 * (sig(base + self.make.player(f"{k}_{g}", pid)) - sig(base))
        for name, comp in (("tov", self.tov), ("trip", self.trip), ("oreb", self.oreb)):
            for side in ("off", "def"):
                out[f"{name}_{side}"] = 100 * (sig(comp.intercept + comp.player(side, pid)) - sig(comp.intercept))
        if self.zone is not None:
            for side in ("off", "def"):
                zl = self.zone.intercept + self.zone.player(side, pid)
                p = np.exp(zl) / np.exp(zl).sum()
                p0 = np.exp(self.zone.intercept) / np.exp(self.zone.intercept).sum()
                for z, d in zip(ZONES, 100 * (p - p0)):
                    out[f"zone_{side}_{z}"] = d
        out["fga_rate"] = self.fga_rate.get(pid, 0.2)
        out["ft_pct"] = self.ft_pct.get(pid, self.league_ft)
        out["mix"] = dict(zip(ZONES, self.zone_mix.get(pid, self.league_mix)))
        if self.resid is not None:
            out["resid_off"] = 100 * self.resid["off"].get(pid, 0.0)
            out["resid_def"] = 100 * self.resid["def"].get(pid, 0.0)
        return out

    # ----------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "RealModel":
        with open(path, "rb") as f:
            return pickle.load(f)
