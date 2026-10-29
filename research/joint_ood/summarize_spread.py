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
    print("Reference returns (true env). Oracle = same policy class and optimiser trained through the true\n"
          "dynamics (the main reference). Scripted best = Hungarian assignment + tuned PD controller, a loose\n"
          "upper bound the policy class does not reach.")
    for n, rs in sorted(ref.items()):
        print(f"  n={n}: oracle {ci([r['oracle']['true_return'] for r in rs])}  "
              f"scripted best (loose UB) {ci([r['scripted_best'] for r in rs])}  "
              f"oracle dup {np.mean([r['oracle']['duplicate_rate'] for r in rs]):.2f}  "
              f"oracle cov {np.mean([r['oracle'].get('final_coverage', np.nan) for r in rs]):.3f}  "
              f"scripted cov {np.mean([r.get('scripted_coverage', np.nan) for r in rs]):.3f}")

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

    oracle = {(r["n"], r["seed"]): r["oracle"]["true_return"] for r in refs}
    print("\nPolicies: true return, regret vs oracle (oracle minus policy true return, paired by n and seed),\n"
          "exploit gap (model minus true), duplicate rate (any shared landmark; saturates at large n),\n"
          "final coverage, fraction of landmarks uncovered (no agent within 2*RAD; missing in early runs); mean ± 95% CI over seeds")
    print(f"{'data':12} {'arch':10} {'n':>2} {'penalty':11} {'lam':>5} {'true_return':>14} {'regret_oracle':>14} "
          f"{'exploit_gap':>14} {'dup_rate':>14} {'final_cov':>14} {'uncovered':>14} {'seeds':>5}")
    pg = defaultdict(list)
    for c in cells:
        for p in c["policies"]:
            pg[(c["data"], c["arch"], c["n"], p["penalty"], p["lam"])].append(
                dict(p, regret_oracle=oracle.get((c["n"], c["seed"]), np.nan) - p["true_return"]))
    for (data, arch, n, pen, lam), ps in sorted(pg.items()):
        print(f"{data:12} {arch:10} {n:2d} {pen:11} {lam:5g} {ci([p['true_return'] for p in ps])} "
              f"{ci([p['regret_oracle'] for p in ps])} {ci([p['exploit_gap'] for p in ps])} {ci([p['duplicate_rate'] for p in ps])} "
              f"{ci([p.get('final_coverage', np.nan) for p in ps])} {ci([p.get('uncovered_frac', np.nan) for p in ps])} {len(ps):5d}")


if __name__ == "__main__":
    main()
