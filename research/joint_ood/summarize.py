"""Print the pilot's tables from one or more result files.

Run:  python research/joint_ood/summarize.py results.json [more.json ...]
"""

import json
import sys
from collections import defaultdict

import numpy as np


def main():
    results = []
    for path in sys.argv[1:]:
        with open(path) as f:
            results.extend(json.load(f))

    print("Diagnostics: model error and ensemble std on three kinds of joint action")
    print(f"{'arch':10} {'n':>2} {'kind':22} {'abs_err':>8} {'optimism':>9} {'ens_std':>8} {'std/err':>8} {'AUROC':>6}")
    groups = defaultdict(list)
    for r in results:
        groups[(r["arch"], r["n"])].append(r)
    for (arch, n), rs in sorted(groups.items()):
        for kind, au in (("in_dist", None), ("marginal_in_joint_out", "auroc_std_mjo_vs_id"),
                         ("uniform_ood", "auroc_std_unif_vs_id")):
            m = lambda key: np.mean([r["diag"][kind][key] for r in rs])
            auc = f"{np.mean([r['diag'][au] for r in rs]):6.3f}" if au else "      "
            print(f"{arch:10} {n:2d} {kind:22} {m('abs_err'):8.3f} {m('optimism'):9.3f} {m('ens_std'):8.3f} "
                  f"{m('std_over_err'):8.3f} {auc}")

    print()
    print("Planning: exploit gap (model minus true reward), regret vs true optimum, duplicate rate")
    print(f"{'arch':10} {'n':>2} {'setting':26} {'lam':>4} {'gap':>7} {'regret':>7} {'dup':>5} {'seeds':>5}")
    pg = defaultdict(list)
    for r in results:
        for p in r["plans"]:
            pg[(r["arch"], r["n"], p["setting"], p["lam"])].append(p)
    for (arch, n, setting, lam), ps in sorted(pg.items()):
        print(f"{arch:10} {n:2d} {setting:26} {lam:4.0f} {np.mean([p['exploit_gap'] for p in ps]):7.3f} "
              f"{np.mean([p['regret'] for p in ps]):7.3f} {np.mean([p['duplicate'] for p in ps]):5.2f} {len(ps):5d}")


if __name__ == "__main__":
    main()
