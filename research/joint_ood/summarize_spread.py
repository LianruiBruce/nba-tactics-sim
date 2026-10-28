"""Tables for spread.py results: diagnostics, then policies, mean and 95% CI over seeds.

Run:  python research/joint_ood/summarize_spread.py spread.json [more.json ...]
"""

import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


def ci(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return f"{xs.mean():7.3f}        "
    h = stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs))
    return f"{xs.mean():7.3f}±{h:5.3f}"


def main():
    cells, refs = [], []
    for path in sys.argv[1:]:
        d = json.load(open(path))
        cells += d["cells"]
        refs += d["reference"]

    ref = defaultdict(list)
    for r in refs:
        ref[r["n"]].append(r)
    print("Reference returns (true env)")
    for n, rs in sorted(ref.items()):
        print(f"  n={n}: oracle {ci([r['oracle']['true_return'] for r in rs])}  "
              f"scripted best {ci([r['scripted_best'] for r in rs])}  "
              f"oracle dup {np.mean([r['oracle']['duplicate_rate'] for r in rs]):.2f}")

    print("\nReward-model diagnostics on three kinds of state (mean over seeds)")
    print(f"{'data':12} {'arch':10} {'n':>2} {'kind':11} {'abs_err':>8} {'optimism':>9} {'ens_std':>8} {'std/err':>8}")
    g = defaultdict(list)
    for c in cells:
        g[(c["data"], c["arch"], c["n"])].append(c)
    for (data, arch, n), cs in sorted(g.items()):
        for kind in ("assignment", "duplicate", "uniform"):
            m = lambda k: np.mean([c["diag"][kind][k] for c in cs])
            print(f"{data:12} {arch:10} {n:2d} {kind:11} {m('abs_err'):8.3f} {m('optimism'):9.3f} "
                  f"{m('ens_std'):8.3f} {m('std_over_err'):8.3f}")

    print("\nPolicies: true return, exploit gap (model minus true), duplicate rate; mean ± 95% CI over seeds")
    print(f"{'data':12} {'arch':10} {'n':>2} {'penalty':11} {'lam':>5} {'true_return':>14} {'exploit_gap':>14} "
          f"{'dup_rate':>14} {'seeds':>5}")
    pg = defaultdict(list)
    for c in cells:
        for p in c["policies"]:
            pg[(c["data"], c["arch"], c["n"], p["penalty"], p["lam"])].append(p)
    for (data, arch, n, pen, lam), ps in sorted(pg.items()):
        print(f"{data:12} {arch:10} {n:2d} {pen:11} {lam:5g} {ci([p['true_return'] for p in ps])} "
              f"{ci([p['exploit_gap'] for p in ps])} {ci([p['duplicate_rate'] for p in ps])} {len(ps):5d}")


if __name__ == "__main__":
    main()
