"""Horizon ablation vs the main grid (H=T=25): regret, exploit gap and uncovered fraction,
paired by seed. Assignment data, seeds present in every horizon. If abl_h{H}_oracle/ holds
true-dynamics oracles trained with the same short horizon, also separates the cost of the short
horizon itself (oracle_25 - oracle_H) from the model's part (oracle_H - policy_H).

Run:  python research/joint_ood/compare_horizon.py
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = (("25", "spread_grid"), ("10", "abl_h10"), ("5", "abl_h5"))
ARCHS = ("additive", "joint", "attention")
CONFIGS = (("none", 0.0), ("ens", 1.0), ("joint_knn", 1.0))


def load(sub):
    cells, oracle = [], {}
    for path in glob.glob(os.path.join(HERE, sub, "*.json")):
        d = json.load(open(path))
        cells += d["cells"]
        for r in d["reference"]:
            oracle[(r["n"], r["seed"])] = r["oracle"]["true_return"]
    m = defaultdict(dict)  # (metric, arch, n, pen, lam) -> {seed: value}
    for c in cells:
        if c["data"] != "assignment":
            continue
        m[("opt", c["arch"], c["n"])][c["seed"]] = c["diag"]["duplicate"]["optimism"]
        for p in c["policies"]:
            k = (c["arch"], c["n"], p["penalty"], p["lam"])
            m[("regret",) + k][c["seed"]] = oracle[(c["n"], c["seed"])] - p["true_return"]
            m[("gap",) + k][c["seed"]] = p["exploit_gap"]
            m[("unc",) + k][c["seed"]] = p.get("uncovered_frac", np.nan)
    return m


def load_oracle(sub):
    oracle = {}
    for path in glob.glob(os.path.join(HERE, sub, "*.json")):
        for r in json.load(open(path))["reference"]:
            oracle[(r["n"], r["seed"])] = r["oracle"]["true_return"]
    return oracle


def mean_ci(xs):
    xs = np.asarray(xs, float)
    h = stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else 0
    return f"{xs.mean():6.2f}±{h:4.2f}"


def paired(a, b, seeds):
    d = np.array([a[s] - b[s] for s in seeds])
    p = stats.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds]).pvalue
    return f"{d.mean():6.2f} (p={p:.3f})"


def main():
    data = {h: load(sub) for h, sub in RUNS}
    short = data["5"]
    seeds = sorted({s for k, v in short.items() for s in v})
    ns = sorted({k[2] for k in short if k[0] == "opt"})
    print(f"Seeds {seeds}, assignment data. Regret = oracle true return - policy true return.")
    print("World-model check, crowded-state optimism (same seed and data, so should match across runs):")
    for arch in ARCHS:
        print(f"  {arch:10}" + "".join(
            f"  n={n} H={h}: {np.mean([data[h][('opt', arch, n)][s] for s in seeds]):.3f}" for n in ns for h, _ in RUNS))
    for metric, title in (("regret", "Regret"), ("gap", "Exploit gap (model - true return)"),
                          ("unc", "Uncovered landmark fraction")):
        print(f"\n{title}: mean ± 95% CI by rollout horizon H, then H=25 minus H, paired by seed")
        print(f"{'arch':10} {'config':12} {'n':>2}" + "".join(f"{'H=' + h:>14}" for h, _ in RUNS)
              + f"{'25-10':>18}{'25-5':>18}")
        for arch in ARCHS:
            for pen, lam in CONFIGS:
                for n in ns:
                    k = (metric, arch, n, pen, lam)
                    row = f"{arch:10} {pen + ':' + f'{lam:g}':12} {n:2}"
                    row += "".join(f"  {mean_ci([data[h][k][s] for s in seeds])}" for h, _ in RUNS)
                    row += "".join(f"  {paired(data['25'][k], data[h][k], seeds)}" for h in ("10", "5"))
                    print(row)

    matched = {h: load_oracle(f"abl_h{h}_oracle") for h in ("10", "5")}
    if not all(matched.values()):
        return
    base = load_oracle("spread_grid")
    print("\nShort-horizon cost in the true dynamics: oracle_25 - oracle_H (regret of the H-step oracle)")
    for n in ns:
        print(f"  n={n}" + "".join(f"   H={h}: {mean_ci([base[(n, s)] - matched[h][(n, s)] for s in seeds])}"
                                   for h in ("10", "5")))
    print("\nModel part of the regret at matched horizon: oracle_H - policy_H, then H=25 minus H, paired by seed")
    print(f"{'arch':10} {'config':12} {'n':>2}" + "".join(f"{'H=' + h:>14}" for h, _ in RUNS)
          + f"{'25-10':>18}{'25-5':>18}")
    orc = {"25": base, **matched}
    for arch in ARCHS:
        for pen, lam in CONFIGS:
            for n in ns:
                k = ("regret", arch, n, pen, lam)
                # regret was computed against the H=25 oracle; shift it to the matched oracle
                m = {h: {s: data[h][k][s] - (base[(n, s)] - orc[h][(n, s)]) for s in seeds} for h, _ in RUNS}
                row = f"{arch:10} {pen + ':' + f'{lam:g}':12} {n:2}"
                row += "".join(f"  {mean_ci(list(m[h].values()))}" for h, _ in RUNS)
                row += "".join(f"  {paired(m['25'], m[h], seeds)}" for h in ("10", "5"))
                print(row)


if __name__ == "__main__":
    main()
