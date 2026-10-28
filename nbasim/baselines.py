"""Public-style baselines that predict a possession's points, shared by backtests and live forecasts.

  context   league average by how the possession started and home court
  team      ridge regression on offense team + defense team (an SRS-style team rating)
  rapm      ridge regression on the ten players on the floor (RAPM)

All of them, like nbasim.realmodel.RealModel, take a possession table with off_lineup, def_lineup,
start_type, home (and offense/defense team ids for "team") and return expected points per row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from nbasim.realmodel import CONTEXTS, SEASON_WEIGHT, lineup


def weights(df: pd.DataFrame) -> np.ndarray:
    if "w" in df:
        return df.w.values
    return df.season.map(SEASON_WEIGHT).fillna(1.0).values


def ctx_matrix(df: pd.DataFrame) -> sparse.csr_matrix:
    cols = [(df.start_type == c).values.astype(float) for c in CONTEXTS[1:]] + [df.home.values.astype(float)]
    return sparse.csr_matrix(np.column_stack(cols))


def onehot_lineups(df: pd.DataFrame, index: dict, cols: list[str]) -> sparse.csr_matrix:
    rows, cs = [], []
    width = len(index)
    for k, c in enumerate(cols):
        for i, s in enumerate(df[c].values):
            for p in lineup(s) if isinstance(s, str) else [s]:
                j = index.get(p)
                if j is not None:
                    rows.append(i)
                    cs.append(k * width + j)
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cs)), shape=(len(df), width * len(cols)))


class RidgeModel:
    """Linear ridge on context + one-hot blocks (team ids or player ids)."""

    def __init__(self, kind: str, alpha: float):
        self.kind, self.alpha = kind, alpha

    def _X(self, df):
        cols = ["offense", "defense"] if self.kind == "team" else ["off_lineup", "def_lineup"]
        return sparse.hstack([ctx_matrix(df) * 10, onehot_lineups(df, self.index, cols)]).tocsr()

    def fit(self, df):
        if self.kind == "team":
            ids = sorted(set(df.offense) | set(df.defense))
        else:
            ids = sorted({p for s in pd.concat([df.off_lineup, df.def_lineup]).unique() for p in lineup(s)})
        self.index = {p: i for i, p in enumerate(ids)}
        self.m = Ridge(alpha=self.alpha).fit(self._X(df), df.points.values, sample_weight=weights(df))
        return self

    def predict(self, df):
        return self.m.predict(self._X(df))


class ContextModel:
    def fit(self, df):
        w = weights(df)
        g = pd.DataFrame({"s": df.start_type.values, "h": df.home.values, "pw": df.points.values * w, "w": w})
        g = g.groupby(["s", "h"])[["pw", "w"]].sum()
        self.means = (g.pw / g.w).to_dict()
        self.overall = float((df.points.values * w).sum() / w.sum())
        return self

    def predict(self, df):
        return np.array([self.means.get((s, h), self.overall) for s, h in zip(df.start_type, df.home)])
