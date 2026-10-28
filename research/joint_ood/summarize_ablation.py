"""Ablation tables for the joint-action pilot: mean and 95% CI over seeds, and the
paired difference from the base config (same arch, n and seed).

Results without a "cfg" field (the main run) count as tag "base".

Run:  python research/joint_ood/summarize_ablation.py --ns 4,8 --max-seed 4 \
          research/joint_ood/results_4080.json research/joint_ood/ablations/*.json
"""

import argparse
import glob
import json
from collections import defaultdict

import numpy as np
from scipy import stats

ORDER = ["base", "k2", "k10", "k20", "data2k", "data100k", "hidden64", "hidden1024", "steps1k", "steps10k",
         "noboot", "noise0.05", "noise0.2", "init_logged", "init_mixed", "restarts1024", "lam_ext"]


def ci(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float(xs.mean()), float("nan")
    return float(xs.mean()), float(stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs)))


def fmt(xs, w=13):
    m, h = ci(xs)
    return f"{m:6.3f}±{h:5.3f}".rjust(w)


def plan_metric(r, setting, lam, key):
    for p in r["plans"]:
        if p["setting"] == setting and p["lam"] == lam:
            return float(p[key])
    return None


# headline metrics: (column name, function of one result)
METRICS = [
    ("mjo_auroc", lambda r: r["diag"]["auroc_std_mjo_vs_id"]),
    ("mjo_optim", lambda r: r["diag"]["marginal_in_joint_out"]["optimism"]),
    ("mjo_std/err", lambda r: r["diag"]["marginal_in_joint_out"]["std_over_err"]),
    ("ens10_regret", lambda r: plan_metric(r, "ensemble_penalty", 10.0, "regret")),
    ("ens10_dup", lambda r: plan_metric(r, "ensemble_penalty", 10.0, "duplicate")),
    ("agent0_regret", lambda r: plan_metric(r, "plus_per_agent_support", 0.0, "regret")),
    ("joint0_regret", lambda r: plan_metric(r, "plus_joint_support_oracle", 0.0, "regret")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ns", default="", help="keep only these n")
    ap.add_argument("--max-seed", type=int, default=None)
    args = ap.parse_args()
    results = []
    for pat in args.files:
        for path in glob.glob(pat) or [pat]:
            with open(path) as f:
                results.extend(json.load(f))
    ns = {int(x) for x in args.ns.split(",")} if args.ns else None
    results = [r for r in results if (ns is None or r["n"] in ns)
               and (args.max_seed is None or r["seed"] <= args.max_seed)]
    tag = lambda r: (r.get("cfg") or {}).get("tag", "base")

    by = defaultdict(dict)  # (tag, arch, n) -> seed -> result
    for r in results:
        by[(tag(r), r["arch"], r["n"])][r["seed"]] = r
    tags = sorted({k[0] for k in by}, key=lambda t: (ORDER.index(t) if t in ORDER else 99, t))
    cells = sorted({(k[1], k[2]) for k in by})

    print("Ablations: mean ± 95% CI over seeds (t interval)")
    print("mjo = marginal-in-joint-out actions; ens10 = ensemble penalty lam=10; "
          "agent0 / joint0 = per-agent / joint support penalty, lam=0")
    for arch, n in cells:
        print(f"\n== arch={arch} n={n}")
        print(f"{'tag':14} {'seeds':>5}" + "".join(f"{name:>14}" for name, _ in METRICS))
        for t in tags:
            rs = by.get((t, arch, n))
            if not rs:
                continue
            row = f"{t:14} {len(rs):5d}"
            for _, fn in METRICS:
                v = [fn(r) for r in rs.values()]
                v = [x for x in v if x is not None]
                row += fmt(v, 14) if v else " " * 14
            print(row)

    print("\nPaired difference from base (ablation minus base, same arch/n/seed), mean ± 95% CI; "
          "* = CI excludes 0")
    for arch, n in cells:
        base = by.get(("base", arch, n), {})
        print(f"\n== arch={arch} n={n}")
        print(f"{'tag':14} {'pairs':>5}" + "".join(f"{name:>15}" for name, _ in METRICS))
        for t in tags:
            if t == "base" or (t, arch, n) not in by:
                continue
            rs = by[(t, arch, n)]
            seeds = sorted(set(rs) & set(base))
            row = f"{t:14} {len(seeds):5d}"
            for _, fn in METRICS:
                d = [fn(rs[s]) - fn(base[s]) for s in seeds
                     if fn(rs[s]) is not None and fn(base[s]) is not None]
                if len(d) < 2:
                    row += " " * 15
                    continue
                m, h = ci(d)
                row += (f"{m:+6.3f}±{h:5.3f}" + ("*" if abs(m) > h else " ")).rjust(15)
            print(row)

    lam_ext = [r for r in results if tag(r) == "lam_ext"]
    if lam_ext:
        print("\nLarge ensemble penalties (lam_ext): regret / dup / exploit gap, mean ± 95% CI")
        g = defaultdict(list)
        for r in lam_ext:
            for p in r["plans"]:
                g[(r["arch"], r["n"], p["setting"], p["lam"])].append(p)
        print(f"{'arch':10} {'n':>2} {'setting':26} {'lam':>5} {'regret':>13} {'dup':>13} {'gap':>13}")
        for (arch, n, s, lam), ps in sorted(g.items()):
            print(f"{arch:10} {n:2d} {s:26} {lam:5.0f}" + fmt([p["regret"] for p in ps])
                  + fmt([p["duplicate"] for p in ps]) + fmt([p["exploit_gap"] for p in ps]))


if __name__ == "__main__":
    main()
