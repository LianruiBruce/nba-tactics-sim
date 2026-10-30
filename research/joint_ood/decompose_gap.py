"""Where does the exploit gap come from? For every saved policy, splits the gap (model return -
true return, ensemble mean) into
  one_step:    sum over the TRUE trajectory of the model's reward error r_hat(s_t, a_t) - r(s_t+1).
               The model never feeds its own predictions back, so nothing compounds;
  compounding: gap - one_step, the part due to the model's states drifting from the true ones.
one_step is further split by whether the true next state is crowded (two agents share a nearest
landmark), the joint configurations the assignment data never shows.

A short rollout horizon only removes compounding, so this measures H3 directly, without the
myopia confound of retraining with short horizons.

The world model is retrained from the cell's seed, which reproduces it (checked against the stored
crowded-state optimism), and the policy weights come from the .pt files spread.py saved.

Run:  python research/joint_ood/decompose_gap.py --dirs spread_grid,abl_h10,abl_h5 --out decomp.json
      python research/joint_ood/decompose_gap.py --summary decomp_a.json,decomp_b.json
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spread import (DEV, Policy, Spread, WorldModel, collect, diagnostics, rollout,  # noqa: E402
                    sample_landmarks, train_world_model)

HERE = os.path.dirname(os.path.abspath(__file__))


def crowded(env, s):
    """Per-state flag: two agents share a nearest landmark. s: (..., n, 4) -> (...)"""
    shape = s.shape[:-2]
    s = s.reshape(-1, env.n, 4)
    nearest = torch.linalg.norm(s[:, :, None, :2] - env.L[None, None], dim=-1).argmin(-1)
    srt = nearest.sort(-1).values
    return (srt[:, 1:] == srt[:, :-1]).any(-1).reshape(shape)


@torch.no_grad()
def decompose(env, pol, wm, gen, episodes=2000):
    s0 = env.reset(episodes, gen)
    r_true, _, _, S, A = rollout(env, pol, s0)  # (B, T), (B, T, n, 4), (B, T, n, 2)
    b, t = r_true.shape
    _, r_hat = wm.predict(S.reshape(b * t, env.n, 4), A.reshape(b * t, env.n, 2))
    err = (r_hat.mean(0).reshape(b, t) - r_true)
    S_next = torch.cat([S[:, 1:], env.step(S[:, -1], A[:, -1])[:, None]], 1)
    crowd = crowded(env, S_next).float()
    r_mod = rollout(env, pol, s0, wm, member="mean")[0]
    gap = (r_mod.sum(1) - r_true.sum(1)).mean().item()
    one = err.sum(1).mean().item()
    return {"gap": gap, "one_step": one, "compounding": gap - one,
            "one_step_crowded": (err * crowd).sum(1).mean().item(),
            "one_step_clear": (err * (1 - crowd)).sum(1).mean().item(),
            "crowded_frac": crowd.mean().item()}


def run_cell(path):
    d = json.load(open(path))
    a, c = d["args"], d["cells"][0]
    n, seed, data_kind, arch = c["n"], c["seed"], c["data"], c["arch"]
    pts = {(p["penalty"], p["lam"]): os.path.splitext(path)[0] + f"_{data_kind}_{arch}_{p['penalty']}{p['lam']:g}.pt"
           for p in c["policies"]}
    pts = {k: v for k, v in pts.items() if os.path.exists(v)}
    if not pts:
        return None
    # same RNG sequence as spread.run_cell up to the trained world model
    gen = torch.Generator(device=DEV).manual_seed(seed * 1000 + n)
    torch.manual_seed(seed * 1000 + n)
    env = Spread(n, sample_landmarks(n, gen), agent_id=not a["no_agent_id"])
    data = collect(env, data_kind, a["episodes"], gen)
    wm = WorldModel(arch, n, k=a["ensemble"])
    train_world_model(wm, data, gen, a["wm_steps"])
    opt = diagnostics(env, wm, gen)["duplicate"]["optimism"]
    res = {"dir": os.path.basename(os.path.dirname(path)), "horizon": a.get("horizon", 25), "n": n,
           "seed": seed, "data": data_kind, "arch": arch, "optimism": opt,
           "optimism_stored": c["diag"]["duplicate"]["optimism"], "policies": []}
    obs_dim = env.obs(data["s"][:1, 0]).shape[-1]
    for (pen, lam), pt in pts.items():
        pol = Policy(obs_dim, a["pol_hidden"]).to(DEV)
        pol.load_state_dict(torch.load(pt, map_location=DEV))
        ev = decompose(env, pol, wm, torch.Generator(device=DEV).manual_seed(777))
        ev.update({"penalty": pen, "lam": lam,
                   "gap_stored": next(p["exploit_gap"] for p in c["policies"] if (p["penalty"], p["lam"]) == (pen, lam))})
        res["policies"].append(ev)
    return res


def ci(xs):
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return f"{xs.mean():7.2f}       " if len(xs) else "      -       "
    h = stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs))
    return f"{xs.mean():7.2f}±{h:5.2f}"


def summary(paths):
    rows = [r for p in paths.split(",") for r in json.load(open(p))]
    bad = [r for r in rows if abs(r["optimism"] - r["optimism_stored"]) > 1e-3]
    print(f"{len(rows)} cells; world model reproduced in {len(rows) - len(bad)}"
          + (f" (NOT in: {[(r['dir'], r['n'], r['seed'], r['data'], r['arch']) for r in bad]})" if bad else ""))
    g = defaultdict(list)
    for r in rows:
        if r in bad:
            continue
        for p in r["policies"]:
            g[(r["data"], r["horizon"], r["arch"], r["n"], p["penalty"], p["lam"])].append(p)
    print("\nExploit gap = one-step model error on the true trajectory + compounding. Mean ± 95% CI over seeds.")
    print("crowded share = part of one_step on steps whose true next state is crowded.")
    keys = ("gap", "one_step", "compounding", "one_step_crowded", "crowded_frac")
    print(f"{'data':11} {'H':>2} {'arch':10} {'n':>2} {'config':12} {'seeds':>5}" + "".join(f"{k:>17}" for k in keys))
    for k in sorted(g):
        ps = g[k]
        print(f"{k[0]:11} {k[1]:2} {k[2]:10} {k[3]:2} {k[4] + ':' + f'{k[5]:g}':12} {len(ps):5}"
              + "".join(f"  {ci([p[m] for p in ps]):>15}" for m in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="spread_grid,abl_h10,abl_h5")
    ap.add_argument("--out", default=os.path.join(HERE, "decomp.json"))
    ap.add_argument("--summary", help="only print the tables of existing output files, comma separated")
    args = ap.parse_args()
    if args.summary:
        return summary(args.summary)
    rows = json.load(open(args.out)) if os.path.exists(args.out) else []
    done = {(r["dir"], r["n"], r["seed"], r["data"], r["arch"]) for r in rows}
    for sub in args.dirs.split(","):
        for path in sorted(glob.glob(os.path.join(HERE, sub, "cell_*.json"))):
            c = json.load(open(path))["cells"][0]
            if (os.path.basename(sub), c["n"], c["seed"], c["data"], c["arch"]) in done:
                continue
            t = time.time()
            r = run_cell(path)
            if r is None:
                continue
            rows.append(r)
            with open(args.out + ".tmp", "w") as f:  # write then rename, as in spread.py
                json.dump(rows, f, indent=1)
            os.replace(args.out + ".tmp", args.out)
            print(f"{sub} n={r['n']} seed={r['seed']} {r['data']} {r['arch']} {time.time() - t:.0f}s "
                  f"optimism {r['optimism']:.4f} vs stored {r['optimism_stored']:.4f}", flush=True)
    summary(args.out)


if __name__ == "__main__":
    main()
