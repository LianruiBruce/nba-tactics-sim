"""Run the ablation grid for the joint-action pilot, several jobs at once on one GPU.

Each job is one (config, arch) pair and writes ablations/<tag>__<arch>.json.
Jobs whose output file already holds every seed are skipped, so the script can
be re-run after an interruption.

Run:  python research/joint_ood/run_ablations.py --workers 4
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "pilot_assignment_torch.py")
OUT = os.path.join(HERE, "ablations")
ARCHS = ["joint", "attention", "additive"]

# One axis changed at a time from the main run (k=5, hidden=256, steps=3000,
# data=20000, noise=0.1, bootstrap, 256 restarts, init=both, lams 0/1/3/10).
CONFIGS = {
    "k2": ["--k", "2"],
    "k10": ["--k", "10"],
    "k20": ["--k", "20"],
    "data2k": ["--data", "2000"],
    "data100k": ["--data", "100000"],
    "hidden64": ["--hidden", "64"],
    "hidden1024": ["--hidden", "1024"],
    "steps1k": ["--steps", "1000"],
    "steps10k": ["--steps", "10000"],
    "noboot": ["--no-bootstrap"],
    "noise0.05": ["--noise", "0.05"],
    "noise0.2": ["--noise", "0.2"],
    "init_logged": ["--init", "logged"],
    "init_mixed": ["--init", "mixed"],
    "restarts1024": ["--restarts", "1024"],
    "lam_ext": ["--lams", "0,10,30,100,300"],
}


def done(path, want):
    try:
        with open(path) as f:
            return len(json.load(f)) >= want
    except (OSError, ValueError):
        return False


def job(tag, extra, arch, ns, seeds):
    path = os.path.join(OUT, f"{tag}__{arch}.json")
    want = len(ns.split(",")) * seeds
    if done(path, want):
        return tag, arch, 0.0, "skip"
    t = time.time()
    cmd = [sys.executable, SCRIPT, "--ns", ns, "--seeds", str(seeds), "--archs", arch,
           "--tag", tag, "--out", path, *extra]
    p = subprocess.run(cmd, capture_output=True, text=True)
    status = "ok" if p.returncode == 0 else "FAIL\n" + p.stderr[-2000:]
    return tag, arch, time.time() - t, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ns", default="4,8")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--only", default="", help="comma-separated tags")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tags = args.only.split(",") if args.only else list(CONFIGS)
    jobs = [(t, CONFIGS[t], a) for t in tags for a in ARCHS]
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        for tag, arch, dt, status in ex.map(lambda j: job(*j, args.ns, args.seeds), jobs):
            print(f"{tag:14} {arch:10} {dt:7.0f}s {status}", flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
