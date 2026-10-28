"""PyTorch port of pilot_assignment.py, for running on a local GPU.

Same experiment, same settings and output format; see pilot_assignment.py for
the description. JAX has no GPU build on native Windows, so this is the
version to run on the RTX 4080.

Run:  python research/joint_ood/pilot_assignment_torch.py --out results.json
"""

import argparse
import json
import math
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

ALPHA = 0.5
NOISE = 0.1
BOX = 1.5
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def landmarks(n):
    ang = 2 * np.pi * np.arange(n) / n
    return np.stack([np.cos(ang), np.sin(ang)], -1)


def true_reward(a, L, s):
    # a: (B, N, 2) tensor
    d = torch.linalg.norm(a[:, :, None, :] - L[None, None], dim=-1)  # B, agents, landmarks
    cover = d.min(dim=1).values.mean(-1)
    travel = torch.linalg.norm(a - s[None], dim=-1).mean(-1)
    return -cover - ALPHA * travel


def true_reward_np(a, L, s):
    with torch.no_grad():
        return true_reward(torch.as_tensor(a, dtype=torch.float32, device=DEV),
                           torch.as_tensor(L, dtype=torch.float32, device=DEV),
                           torch.as_tensor(s, dtype=torch.float32, device=DEV)).cpu().numpy()


def is_permutation(a, L):
    nearest = np.argmin(np.linalg.norm(a[:, :, None, :] - L[None, None], axis=-1), axis=-1)
    n = L.shape[0]
    return np.array([len(set(row)) == n for row in nearest])


def sample_perm(rng, n, size, L):
    perms = np.array([rng.permutation(n) for _ in range(size)])
    return L[perms] + NOISE * rng.standard_normal((size, n, 2))


def sample_marginal_joint_ood(rng, n, size, L):
    out = []
    while sum(len(x) for x in out) < size:
        idx = rng.integers(0, n, (size, n))
        a = L[idx] + NOISE * rng.standard_normal((size, n, 2))
        out.append(a[~is_permutation(a, L)])
    return np.concatenate(out)[:size]


def sample_uniform(rng, n, size):
    return rng.uniform(-BOX, BOX, (size, n, 2))


# ---------------------------------------------------------------- ensemble
# Every weight carries a leading ensemble axis K, so the K members train and
# predict in one batched call.

def ens_linear(k_models, din, dout, gain=2.0):
    w = torch.randn(k_models, din, dout, device=DEV) * math.sqrt(gain / din)
    b = torch.zeros(k_models, dout, device=DEV)
    return [w.requires_grad_(), b.requires_grad_()]


def apply_linear(x, wb):
    w, b = wb
    y = torch.einsum("k...d,kde->k...e", x, w)
    return y + b.view(b.shape[0], *([1] * (y.dim() - 2)), b.shape[1])


def ens_mlp(k_models, sizes):
    return [ens_linear(k_models, i, o) for i, o in zip(sizes[:-1], sizes[1:])]


def apply_mlp(x, layers, last_act=False):
    for i, wb in enumerate(layers):
        x = apply_linear(x, wb)
        if i < len(layers) - 1 or last_act:
            x = torch.nn.functional.silu(x)
    return x


class Ensemble:
    """Reward model over the joint action, three ways to treat the agents.

    joint      one MLP on the concatenated joint action
    attention  one token per agent (action + agent id), a self-attention layer,
               mean pooling, MLP head; the set-structured design of recent
               multi-agent world models
    additive   sum over agents of a shared per-agent MLP (action + agent id);
               fully factored, cannot represent interactions
    """

    def __init__(self, arch, n, k_models=5, hidden=256):
        self.arch, self.n, self.k = arch, n, k_models
        self.eye = torch.eye(n, device=DEV)
        if arch == "joint":
            self.p = {"mlp": ens_mlp(k_models, [2 * n, hidden, hidden, hidden, 1])}
        elif arch == "additive":
            self.p = {"mlp": ens_mlp(k_models, [2 + n, hidden, hidden, 1])}
        elif arch == "attention":
            d = 128
            self.d = d
            self.p = {"inp": ens_mlp(k_models, [2 + n, d, d]), "q": ens_linear(k_models, d, d, 1.0),
                      "k": ens_linear(k_models, d, d, 1.0), "v": ens_linear(k_models, d, d, 1.0),
                      "ff": ens_mlp(k_models, [d, d, d]), "head": ens_mlp(k_models, [d, hidden, 1])}
        else:
            raise ValueError(arch)

    def parameters(self):
        out = []
        for v in self.p.values():
            if isinstance(v[0], list):
                for wb in v:
                    out.extend(wb)
            else:
                out.extend(v)
        return out

    def tokens(self, x):
        a = x.reshape(*x.shape[:-1], self.n, 2)
        ids = self.eye.expand(*a.shape[:-1], self.n)
        return torch.cat([a, ids], -1)

    def __call__(self, x):
        # x: (K, B, 2n) -> (K, B)
        if self.arch == "joint":
            return apply_mlp(x, self.p["mlp"])[..., 0]
        if self.arch == "additive":
            return apply_mlp(self.tokens(x), self.p["mlp"])[..., 0].sum(-1)
        h = apply_mlp(self.tokens(x), self.p["inp"], last_act=True)  # K,B,n,d
        q, k, v = (apply_linear(h, self.p[name]) for name in ("q", "k", "v"))
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.d), -1)
        h = h + att @ v
        h = h + apply_mlp(h, self.p["ff"])
        return apply_mlp(h.mean(-2), self.p["head"])[..., 0]


def train_ensemble(model, X, y, rng, steps=3000, batch=512, lr=1e-3, bootstrap=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    if bootstrap:
        boot = torch.as_tensor(np.stack([rng.integers(0, n, n) for _ in range(model.k)]), device=DEV)
    else:  # members differ only by init and minibatch order
        boot = torch.arange(n, device=DEV).expand(model.k, n).contiguous()
    for _ in range(steps):
        pos = torch.randint(0, n, (model.k, batch), device=DEV)
        idx = torch.gather(boot, 1, pos)
        loss = ((model(X[idx]) - y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


# ---------------------------------------------------------------- planner

def plan(model, norm, data_flat, L, n, lam, mu_agent, mu_joint, rng, restarts=256, steps=300, lr=0.02,
         init="both"):
    mean_x, std_x, mean_y, std_y = norm

    def objective(a_flat):
        x = ((a_flat - mean_x) / std_x).expand(model.k, -1, -1)
        preds = model(x) * std_y + mean_y  # K,B
        m, s = preds.mean(0), preds.std(0, unbiased=False)
        a = a_flat.reshape(-1, n, 2)
        agent_sup = ((a[:, :, None, :] - L[None, None]) ** 2).sum(-1).min(-1).values.sum(-1)
        obj = m - lam * s - mu_agent * agent_sup
        if mu_joint > 0:
            obj = obj - mu_joint * torch.cdist(a_flat, data_flat).min(-1).values ** 2
        return obj, m, s

    # Half the restarts start at logged joint actions. The other half recombine
    # logged actions agent by agent (agent i's action from one sample, agent j's
    # from another), which is where independently trained decentralised
    # policies end up when each settles on its own preferred mode.
    data_np = data_flat.cpu().numpy()
    half = {"both": restarts // 2, "logged": restarts, "mixed": 0}[init]
    logged = data_np[rng.integers(0, data_np.shape[0], half)]
    rows = rng.integers(0, data_np.shape[0], (restarts - half, n))
    mixed = data_np.reshape(-1, n, 2)[rows, np.arange(n)[None]].reshape(restarts - half, 2 * n)
    a = torch.as_tensor(np.concatenate([logged, mixed]), dtype=torch.float32, device=DEV).requires_grad_()
    opt = torch.optim.Adam([a], lr=lr)
    for _ in range(steps):
        obj = objective(a)[0]
        opt.zero_grad()
        (-obj.sum()).backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(-BOX, BOX)
    with torch.no_grad():
        obj, m, s = objective(a)
        best = int(torch.argmax(obj))
        return a[best].cpu().numpy().reshape(n, 2), float(m[best]), float(s[best])


def optimum(L, s):
    cost = ALPHA * np.linalg.norm(L[None] - s[:, None], axis=-1) / len(L)  # agent i -> landmark j
    r, c = linear_sum_assignment(cost)
    a = np.zeros_like(s)
    a[r] = L[c]
    return a


def auroc(neg, pos):
    # probability a random pos sample has higher score than a random neg sample
    allv = np.concatenate([neg, pos])
    ranks = allv.argsort().argsort() + 1
    rp = ranks[len(neg):].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def run(n, seed, data_size, lams, settings, arch, cfg=None):
    cfg = cfg or {}
    global NOISE
    NOISE = cfg.get("noise", 0.1)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    L = landmarks(n)
    s = rng.uniform(-1, 1, (n, 2))
    Lt = torch.as_tensor(L, dtype=torch.float32, device=DEV)

    A = sample_perm(rng, n, data_size, L)
    R = true_reward_np(A, L, s) + 0.01 * rng.standard_normal(data_size)
    X = A.reshape(data_size, -1)
    mean_x, std_x = X.mean(0), X.std(0) + 1e-6
    mean_y, std_y = float(R.mean()), float(R.std() + 1e-6)
    tens = lambda v: torch.as_tensor(v, dtype=torch.float32, device=DEV)
    model = Ensemble(arch, n, k_models=cfg.get("k", 5), hidden=cfg.get("hidden", 256))
    train_ensemble(model, tens((X - mean_x) / std_x), tens((R - mean_y) / std_y), rng,
                   steps=cfg.get("steps", 3000), bootstrap=cfg.get("bootstrap", True))
    norm = (tens(mean_x), tens(std_x), mean_y, std_y)

    # diagnostics on three kinds of joint action
    diag = {}
    for name, sampler in [("in_dist", lambda: sample_perm(rng, n, 2000, L)),
                          ("marginal_in_joint_out", lambda: sample_marginal_joint_ood(rng, n, 2000, L)),
                          ("uniform_ood", lambda: sample_uniform(rng, n, 2000))]:
        a = sampler()
        with torch.no_grad():
            x = tens((a.reshape(len(a), -1) - mean_x) / std_x).expand(model.k, -1, -1)
            p = (model(x) * std_y + mean_y).cpu().numpy()
        signed = p.mean(0) - true_reward_np(a, L, s)
        err = np.abs(signed)
        diag[name] = {"abs_err": float(err.mean()), "optimism": float(signed.mean()),
                      "frac_optimistic": float((signed > 0).mean()), "ens_std": float(p.std(0).mean()),
                      "std_over_err": float(p.std(0).mean() / err.mean()),
                      "spearman_std_err": float(spearmanr(p.std(0), err).correlation),
                      "_std": p.std(0)}
    diag["auroc_std_mjo_vs_id"] = float(auroc(diag["in_dist"]["_std"], diag["marginal_in_joint_out"]["_std"]))
    diag["auroc_std_unif_vs_id"] = float(auroc(diag["in_dist"]["_std"], diag["uniform_ood"]["_std"]))
    for k in ("in_dist", "marginal_in_joint_out", "uniform_ood"):
        diag[k].pop("_std")

    r_star = float(true_reward_np(optimum(L, s)[None], L, s)[0])

    plans = []
    data_flat = tens(X)
    for setting in settings:
        for lam in lams:
            a, m, sd = plan(model, norm, data_flat, Lt, n, lam, setting["mu_agent"], setting["mu_joint"], rng,
                            restarts=cfg.get("restarts", 256), init=cfg.get("init", "both"))
            r = float(true_reward_np(a[None], L, s)[0])
            plans.append({"setting": setting["name"], "lam": lam, "model_r": m, "model_std": sd, "true_r": r,
                          "exploit_gap": m - r, "regret": r_star - r,
                          "duplicate": bool(not is_permutation(a[None], L)[0])})
    return {"arch": arch, "n": n, "seed": seed, "cfg": cfg, "r_star": r_star, "r_data_best": float(R.max()),
            "r_behavior": float(R.mean()),
            "frac_joint_space_in_support": float(math.factorial(n) / n ** n),
            "diag": diag, "plans": plans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="2,3,4,6,8")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--data", type=int, default=20000)
    ap.add_argument("--archs", default="joint,attention,additive")
    ap.add_argument("--out", default="pilot_assignment.json")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--lams", default="0,1,3,10")
    # ablation knobs; defaults reproduce the main run
    ap.add_argument("--tag", default="base")
    ap.add_argument("--k", type=int, default=5, help="ensemble size")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3000, help="training steps")
    ap.add_argument("--noise", type=float, default=0.1, help="action noise around landmarks")
    ap.add_argument("--no-bootstrap", action="store_true")
    ap.add_argument("--restarts", type=int, default=256, help="planner restarts")
    ap.add_argument("--init", default="both", choices=["both", "logged", "mixed"], help="planner restart init")
    args = ap.parse_args()
    print(f"device: {DEV}", flush=True)
    lams = [float(x) for x in args.lams.split(",")]
    cfg = {"tag": args.tag, "k": args.k, "hidden": args.hidden, "steps": args.steps, "noise": args.noise,
           "bootstrap": not args.no_bootstrap, "restarts": args.restarts, "init": args.init, "data": args.data}
    settings = [{"name": "ensemble_penalty", "mu_agent": 0.0, "mu_joint": 0.0},
                {"name": "plus_per_agent_support", "mu_agent": 10.0, "mu_joint": 0.0},
                {"name": "plus_joint_support_oracle", "mu_agent": 0.0, "mu_joint": 10.0}]
    results = []
    for arch in args.archs.split(","):
        for n in [int(x) for x in args.ns.split(",")]:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                t = time.time()
                results.append(run(n, seed, args.data, lams, settings, arch, cfg))
                print(f"arch={arch} n={n} seed={seed} {time.time() - t:.0f}s", flush=True)
                with open(args.out, "w") as f:
                    json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
