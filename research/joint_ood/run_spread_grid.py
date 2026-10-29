"""Run the spread.py grid as one process per cell, several at once on one GPU.

spread.py is bound by Python and kernel-launch overhead (25 sequential steps per
iteration, small networks), so a single process leaves the GPU mostly idle.
Every cell seeds its own generator from (n, seed), so splitting the grid this
way gives the same results as one sequential run. Each job writes
<out>/<name>.json; jobs whose file is already complete are skipped.

Run:  python research/joint_ood/run_spread_grid.py --ns 4,8 --seeds 3 --workers 8 --out research/joint_ood/spread_grid
      python research/joint_ood/summarize_spread.py research/joint_ood/spread_grid/*.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "spread.py")


def complete(path, ref):
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return False
    return bool(d["reference"]) if ref else bool(d["cells"])


def job(spec, out, extra):
    name, args, ref = spec
    path = os.path.join(out, name + ".json")
    if complete(path, ref):
        return name, 0.0, "skip"
    t = time.time()
    log = os.path.join(out, name + ".log")
    with open(log, "w") as f:
        p = subprocess.run([sys.executable, SCRIPT, *args, *extra, "--out", path], stdout=f, stderr=subprocess.STDOUT)
    return name, time.time() - t, "ok" if p.returncode == 0 else f"FAIL (see {log})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="4,8")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--data", default="assignment,independent")
    ap.add_argument("--archs", default="joint,additive,attention")
    ap.add_argument("--configs", default="none:0,ens:1,ens:5,agent_knn:1,joint_knn:1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "spread_grid"))
    ap.add_argument("--exclude", default="", help="comma-separated job names to skip (e.g. already running)")
    ap.add_argument("extra", nargs="*", help="passed through to spread.py, after --")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    specs = []
    for n in args.ns.split(","):
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            base = ["--ns", n, "--seeds", "1", "--seed_start", str(seed)]
            specs.append((f"ref_n{n}_s{seed}", base + ["--reference_only"], True))
            for data in args.data.split(","):
                for arch in args.archs.split(","):
                    specs.append((f"cell_n{n}_s{seed}_{data}_{arch}",
                                  base + ["--no_reference", "--data", data, "--archs", arch,
                                          "--configs", args.configs], False))
    specs = [s for s in specs if s[0] not in args.exclude.split(",")]
    # longest jobs (large n) first so the tail is short
    specs.sort(key=lambda s: -int(s[0].split("_n")[1].split("_")[0]))
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for name, dt, status in ex.map(lambda s: job(s, args.out, args.extra), specs):
            print(f"{name:40} {dt:7.0f}s {status}", flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
