"""Paper-style tables for the spread main grid: scaling with n, causal control and
penalty effects, with paired tests over seeds.

Run:  python research/joint_ood/analyze_spread.py research/joint_ood/spread_grid/*.json
"""

import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ARCHS = ("additive", "joint", "attention")
PENS = (("none", 0.0), ("ens", 1.0), ("ens", 5.0), ("agent_knn", 1.0), ("joint_knn", 1.0))


def ms(xs):
    xs = np.asarray([x for x in xs if not np.isnan(x)], float)
    if len(xs) == 0:
        return "      -      "
    if len(xs) < 2:
        return f"{xs.mean():6.2f}       "
    h = stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs))
    return f"{xs.mean():6.2f}±{h:5.2f}"


def paired(a, b):
    """Mean of a-b over shared keys, and a paired t-test p value."""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    if len(d) < 2 or np.allclose(d, d[0]):
        return f"{d.mean() if len(d) else np.nan:7.2f} (n={len(d)})        "
    p = stats.ttest_rel([a[k] for k in keys], [b[k] for k in keys]).pvalue
    return f"{d.mean():7.2f} (n={len(d)}, p={p:.3f})"


def main():
    cells, refs = [], []
    for path in sys.argv[1:]:
        d = json.load(open(path))
        cells += d["cells"]
        refs += d["reference"]
    oracle = {(r["n"], r["seed"]): r["oracle"]["true_return"] for r in refs}
    ns = sorted({c["n"] for c in cells})

    # metric[(data, arch, n, pen, lam)][seed]
    pol = defaultdict(lambda: defaultdict(dict))
    for c in cells:
        for p in c["policies"]:
            key = (c["data"], c["arch"], c["n"], p["penalty"], p["lam"])
            pol["regret"][key][c["seed"]] = oracle.get((c["n"], c["seed"]), np.nan) - p["true_return"]
            pol["gap"][key][c["seed"]] = p["exploit_gap"]
            pol["unc"][key][c["seed"]] = p.get("uncovered_frac", np.nan)

    print("Oracle true return by n:", {n: round(np.mean([v for (m, _), v in oracle.items() if m == n]), 3) for n in ns})

    print("\n1. Reward-model optimism on crowded (duplicate) states and ensemble std/err, assignment data")
    print(f"{'arch':10}" + "".join(f"   n={n}: optimism  std/err" for n in ns))
    for arch in ARCHS:
        row = f"{arch:10}"
        for n in ns:
            cs = [c for c in cells if c["arch"] == arch and c["n"] == n and c["data"] == "assignment"]
            o = [c["diag"]["duplicate"]["optimism"] for c in cs]
            r = [c["diag"]["duplicate"]["std_over_err"] for c in cs]
            row += f"   {ms(o)} {np.mean(r) if r else np.nan:6.2f}"
        print(row)
    print("   same, independent data (crowding present in the data):")
    for arch in ARCHS:
        row = f"{arch:10}"
        for n in ns:
            cs = [c for c in cells if c["arch"] == arch and c["n"] == n and c["data"] == "independent"]
            o = [c["diag"]["duplicate"]["optimism"] for c in cs]
            r = [c["diag"]["duplicate"]["std_over_err"] for c in cs]
            row += f"   {ms(o)} {np.mean(r) if r else np.nan:6.2f}"
        print(row)

    for data in ("assignment", "independent"):
        print(f"\n2. Regret vs oracle, {data} data (mean ± 95% CI over seeds)")
        print(f"{'arch':10} {'penalty':12}" + "".join(f"{'n=' + str(n):>16}" for n in ns))
        for arch in ARCHS:
            for pen, lam in PENS:
                print(f"{arch:10} {pen + ':' + f'{lam:g}':12}" +
                      "".join(f"  {ms(pol['regret'][(data, arch, n, pen, lam)].values())}" for n in ns))

    print("\n3. Causal control: regret(assignment) - regret(independent), paired by seed. Positive = the")
    print("   crowding-free data makes things worse, as expected if the failure is joint-action OOD")
    for arch in ARCHS:
        for pen, lam in PENS:
            print(f"{arch:10} {pen + ':' + f'{lam:g}':12}" + "".join(
                f"  n={n}: {paired(pol['regret'][('assignment', arch, n, pen, lam)], pol['regret'][('independent', arch, n, pen, lam)])}"
                for n in ns))

    print("\n4. Penalty effect: regret(none) - regret(penalty), assignment data, paired by seed. Positive = penalty helps")
    for arch in ARCHS:
        for pen, lam in PENS[1:]:
            print(f"{arch:10} {pen + ':' + f'{lam:g}':12}" + "".join(
                f"  n={n}: {paired(pol['regret'][('assignment', arch, n, 'none', 0.0)], pol['regret'][('assignment', arch, n, pen, lam)])}"
                for n in ns))

    print("\n5. Exploit gap (model return - true return) and uncovered landmark fraction, assignment data")
    print(f"{'arch':10} {'penalty':12}" + "".join(f"{'gap n=' + str(n):>16}{'unc n=' + str(n):>15}" for n in ns))
    for arch in ARCHS:
        for pen, lam in PENS:
            k = lambda n: ("assignment", arch, n, pen, lam)
            print(f"{arch:10} {pen + ':' + f'{lam:g}':12}" + "".join(
                f"  {ms(pol['gap'][k(n)].values())}  {ms(pol['unc'][k(n)].values())}" for n in ns))


if __name__ == "__main__":
    main()
