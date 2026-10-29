"""Pilot 2: joint-action model exploitation in multi-step model-based offline MARL.

Environment. A differentiable, batched cooperative-navigation task with the
physics of MPE "simple spread": N agents and N fixed landmarks; the team
reward each step is minus the mean distance from each landmark to its nearest
agent, minus a smooth collision penalty. Episodes last T steps.

Data. Each episode the behaviour policy assigns agents to landmarks and a
noisy PD controller drives each agent to its landmark.
  assignment   a random permutation: every agent visits every landmark across
               the dataset, but two agents never head to the same landmark
  independent  each agent picks a landmark independently, so shared targets do
               occur. Per-agent marginals are identical to "assignment"; only
               the joint coverage differs. This is the causal control.

Pipeline. Fit an ensemble world model (next state + reward) on the data, train
a decentralised policy (shared weights, agent id in the observation) by
back-propagating the imagined return through the model, then evaluate the
policy in the true environment. Uncertainty handling is the ablation axis:
  none         plain model return
  ens          MOPO-style penalty, lambda * norm of the ensemble std of the
               predicted (next state, reward)
  ens_reward   penalty on the ensemble std of the predicted reward only
  agent_knn    per-agent support: distance from each (own obs, own action) to
               the nearest logged pair
  joint_knn    joint support: distance from the joint (state, action) to the
               nearest logged joint pair
The oracle trains the same policy through the true, differentiable dynamics.

Run:  python research/joint_ood/spread.py --out spread.json
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 25
DAMP, ACC, DT = 0.75, 0.5, 0.1
RAD = 0.075
COL_W = 0.5


# ---------------------------------------------------------------- environment

class Spread:
    def __init__(self, n, landmarks, agent_id=True):
        self.n = n
        self.L = landmarks  # (n, 2) tensor, fixed for the dataset
        self.agent_id = agent_id  # policy observation only; world models always see the id

    def reset(self, b, gen):
        p = (torch.rand(b, self.n, 2, generator=gen, device=DEV) * 2 - 1)
        v = torch.zeros(b, self.n, 2, device=DEV)
        return torch.cat([p, v], -1)  # state (b, n, 4)

    def step(self, s, a):
        p, v = s[..., :2], s[..., 2:]
        v = DAMP * v + ACC * a
        p = p + DT * v
        return torch.cat([p, v], -1)

    def reward(self, s):
        p = s[..., :2]
        d = torch.linalg.norm(p[:, :, None] - self.L[None, None], dim=-1)  # b, agents, landmarks
        cover = d.min(1).values.mean(-1)
        # pairs only: the norm's gradient at the zero diagonal would be NaN
        iu = torch.triu_indices(self.n, self.n, 1, device=DEV)
        dd = torch.linalg.norm(p[:, iu[0]] - p[:, iu[1]], dim=-1)
        col = torch.relu(2 * RAD - dd).sum(-1) / (2 * RAD)
        return -cover - COL_W * col

    def obs(self, s):
        """Local observation of every agent: own pos and vel, relative landmark
        positions, relative positions of the other agents, agent id."""
        b, n = s.shape[:2]
        p, v = s[..., :2], s[..., 2:]
        rel_l = (self.L[None, None] - p[:, :, None]).reshape(b, n, -1)
        rel_a = p[:, None] - p[:, :, None]  # b, i, k, 2
        mask = ~torch.eye(n, dtype=torch.bool, device=DEV)
        rel_a = rel_a[:, mask].reshape(b, n, -1)
        if not self.agent_id:
            return torch.cat([p, v, rel_l, rel_a], -1)
        ids = torch.eye(n, device=DEV).expand(b, n, n)
        return torch.cat([p, v, rel_l, rel_a, ids], -1)


def coverage(env, s):
    """Mean distance from each landmark to its nearest agent."""
    d = torch.linalg.norm(s[..., :2][:, :, None] - env.L[None, None], dim=-1)
    return d.min(1).values.mean().item()


def duplicate_rate(env, s):
    p = s[..., :2]
    nearest = torch.linalg.norm(p[:, :, None] - env.L[None, None], dim=-1).argmin(-1)  # b, n
    srt = nearest.sort(-1).values
    return (srt[:, 1:] == srt[:, :-1]).any(-1).float().mean().item()


# ---------------------------------------------------------------- behaviour data

def controller(s, targets, noise, gen, kp=2.0, kd=1.0):
    p, v = s[..., :2], s[..., 2:]
    a = kp * (targets - p) - kd * v
    a = a + noise[:, None, None] * torch.randn(a.shape, generator=gen, device=DEV)
    return a.clamp(-1, 1)


def assignments(kind, b, n, gen):
    if kind == "assignment":
        return torch.argsort(torch.rand(b, n, generator=gen, device=DEV), -1)
    if kind == "independent":
        return torch.randint(0, n, (b, n), generator=gen, device=DEV)
    if kind == "duplicate":  # used only for diagnostics: at least one shared target
        idx = torch.argsort(torch.rand(b, n, generator=gen, device=DEV), -1)
        idx[:, 1] = idx[:, 0]
        return idx
    raise ValueError(kind)


def collect(env, kind, episodes, gen):
    s = env.reset(episodes, gen)
    tgt = env.L[assignments(kind, episodes, env.n, gen)]
    noise = 0.1 + 0.5 * torch.rand(episodes, generator=gen, device=DEV)
    S, A, R, S2 = [], [], [], []
    for _ in range(T):
        a = controller(s, tgt, noise, gen)
        s2 = env.step(s, a)
        S.append(s), A.append(a), R.append(env.reward(s2)), S2.append(s2)
        s = s2
    st = lambda xs: torch.stack(xs, 1)  # episodes, T, ...
    return {"s": st(S), "a": st(A), "r": st(R), "s2": st(S2)}


# ---------------------------------------------------------------- ensemble world models

def ens_linear(k, din, dout, gain=2.0):
    w = torch.randn(k, din, dout, device=DEV) * math.sqrt(gain / din)
    b = torch.zeros(k, dout, device=DEV)
    return [w.requires_grad_(), b.requires_grad_()]


def lin(x, wb):
    w, b = wb
    y = torch.einsum("k...d,kde->k...e", x, w)
    return y + b.view(b.shape[0], *([1] * (y.dim() - 2)), b.shape[1])


def ens_mlp(k, sizes):
    return [ens_linear(k, i, o) for i, o in zip(sizes[:-1], sizes[1:])]


def mlp(x, layers, last_act=False):
    for i, wb in enumerate(layers):
        x = lin(x, wb)
        if i < len(layers) - 1 or last_act:
            x = torch.nn.functional.silu(x)
    return x


class WorldModel:
    """Ensemble of K models predicting (next state delta, reward) from (state, action).

    joint      one MLP on the concatenated joint state and action
    additive   per-agent model on the agent's own state and action (plus the
               landmark layout, which is fixed); reward is the sum of per-agent
               heads. The fully factored extreme.
    attention  one token per agent, self-attention across agents, per-agent
               state heads and a pooled reward head; the set-structured design
               of recent multi-agent world models
    """

    def __init__(self, arch, n, k=5, hidden=256):
        self.arch, self.n, self.k = arch, n, k
        self.eye = torch.eye(n, device=DEV)
        tok = 4 + 2 + n  # own state, own action, agent id
        if arch == "joint":
            self.p = {"net": ens_mlp(k, [n * 6, hidden, hidden, hidden, n * 4 + 1])}
        elif arch == "additive":
            self.p = {"net": ens_mlp(k, [tok, hidden, hidden, 5])}
        elif arch == "attention":
            d = 128
            self.d = d
            self.p = {"inp": ens_mlp(k, [tok, d, d]), "q": ens_linear(k, d, d, 1.0), "k": ens_linear(k, d, d, 1.0),
                      "v": ens_linear(k, d, d, 1.0), "ff": ens_mlp(k, [d, d, d]),
                      "state": ens_mlp(k, [d, d, 4]), "rew": ens_mlp(k, [d, hidden, 1])}
        else:
            raise ValueError(arch)
        self.norm = None

    def parameters(self):
        out = []
        for v in self.p.values():
            for item in (v if isinstance(v[0], list) else [v]):
                out.extend(item)
        return out

    def set_norm(self, s, a, ds, r):
        f = lambda x: (x.reshape(-1, x.shape[-1]).mean(0), x.reshape(-1, x.shape[-1]).std(0) + 1e-6)
        self.norm = {"s": f(s), "a": f(a), "ds": f(ds), "r": (r.mean(), r.std() + 1e-6)}

    def raw(self, s, a):
        """s: (K, B, n, 4), a: (K, B, n, 2), normalised. Returns normalised (ds, r)."""
        K, B, n = s.shape[:3]
        if self.arch == "joint":
            out = mlp(torch.cat([s.reshape(K, B, -1), a.reshape(K, B, -1)], -1), self.p["net"])
            return out[..., :-1].reshape(K, B, n, 4), out[..., -1]
        x = torch.cat([s, a, self.eye.expand(K, B, n, n)], -1)
        if self.arch == "additive":
            out = mlp(x, self.p["net"])
            return out[..., :4], out[..., 4].sum(-1)
        h = mlp(x, self.p["inp"], last_act=True)
        q, kk, v = lin(h, self.p["q"]), lin(h, self.p["k"]), lin(h, self.p["v"])
        h = h + torch.softmax(q @ kk.transpose(-1, -2) / math.sqrt(self.d), -1) @ v
        h = h + mlp(h, self.p["ff"])
        return mlp(h, self.p["state"]), mlp(h.mean(-2), self.p["rew"])[..., 0]

    def predict(self, s, a):
        """Unnormalised per-member predictions. s: (B, n, 4) or (K, B, n, 4)."""
        if s.dim() == 3:
            s = s.expand(self.k, *s.shape)
            a = a.expand(self.k, *a.shape)
        ns, na = self.norm["s"], self.norm["a"]
        ds, r = self.raw((s - ns[0]) / ns[1], (a - na[0]) / na[1])
        nds, nr = self.norm["ds"], self.norm["r"]
        return s + ds * nds[1] + nds[0], r * nr[1] + nr[0]  # (K,B,n,4), (K,B)


def train_world_model(wm, data, gen, steps, batch=1024, lr=1e-3):
    s = data["s"].reshape(-1, wm.n, 4)
    a = data["a"].reshape(-1, wm.n, 2)
    ds = data["s2"].reshape(-1, wm.n, 4) - s
    r = data["r"].reshape(-1)
    wm.set_norm(s, a, ds, r)
    ns, na, nds, nr = wm.norm["s"], wm.norm["a"], wm.norm["ds"], wm.norm["r"]
    sn, an, dsn, rn = (s - ns[0]) / ns[1], (a - na[0]) / na[1], (ds - nds[0]) / nds[1], (r - nr[0]) / nr[1]
    m = s.shape[0]
    boot = torch.randint(0, m, (wm.k, m), generator=gen, device=DEV)
    opt = torch.optim.Adam(wm.parameters(), lr=lr)
    for _ in range(steps):
        idx = torch.gather(boot, 1, torch.randint(0, m, (wm.k, batch), generator=gen, device=DEV))
        pds, pr = wm.raw(sn[idx], an[idx])
        loss = ((pds - dsn[idx]) ** 2).mean() + ((pr - rn[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


# ---------------------------------------------------------------- policy

class Policy(torch.nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(obs_dim, hidden), torch.nn.SiLU(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
                                       torch.nn.Linear(hidden, 2))

    def forward(self, o):
        return torch.tanh(self.net(o))


class KNN:
    """Squared distance to the nearest reference point, on normalised features
    and divided by the feature count. The nearest point is found without
    gradient in chunks; the distance to it carries the gradient, which is the
    same gradient as the min over all points."""

    def __init__(self, ref, max_ref=50000, gen=None, chunk=1024):  # chunk only bounds memory (cdist is chunk x ref)
        if ref.shape[0] > max_ref:
            ref = ref[torch.randperm(ref.shape[0], generator=gen, device=DEV)[:max_ref]]
        self.mu, self.sd = ref.mean(0), ref.std(0) + 1e-6
        self.ref = (ref - self.mu) / self.sd
        self.chunk = chunk

    def __call__(self, x):
        z = (x - self.mu) / self.sd
        with torch.no_grad():
            idx = torch.cat([torch.cdist(c, self.ref).argmin(-1) for c in z.detach().split(self.chunk)])
        return ((z - self.ref[idx]) ** 2).sum(-1) / z.shape[-1]


def rollout(env, policy, s0, wm=None, member="random", gen=None):
    """Imagined (wm given) or true (wm None) episode. Returns per-step rewards,
    uncertainties and the visited (state, action) pairs."""
    s = s0
    rews, u_full, u_rew, states, actions = [], [], [], [], []
    for _ in range(T):
        a = policy(env.obs(s))
        states.append(s), actions.append(a)
        if wm is None:
            s = env.step(s, a)
            rews.append(env.reward(s))
            continue
        s2, r = wm.predict(s, a)  # K,B,...
        b = s.shape[0]
        nds, nr = wm.norm["ds"], wm.norm["r"]
        std_s = (s2 / nds[1]).std(0).reshape(b, -1)
        std_r = (r / nr[1]).std(0)
        u_full.append(torch.sqrt((std_s ** 2).sum(-1) + std_r ** 2))
        u_rew.append(std_r)
        if member == "random":
            pick = torch.randint(0, wm.k, (b,), generator=gen, device=DEV)
            s, r = s2[pick, torch.arange(b)], r[pick, torch.arange(b)]
        else:
            s, r = s2.mean(0), r.mean(0)
        rews.append(r)
    st = lambda xs: torch.stack(xs, 1) if xs else None
    return st(rews), st(u_full), st(u_rew), st(states), st(actions)


def train_policy(env, wm, penalty, lam, data, gen, iters, batch=256, lr=1e-3, member="random", hidden=128):
    obs_dim = env.obs(data["s"][:1, 0]).shape[-1]
    pol = Policy(obs_dim, hidden).to(DEV)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    s0_pool = data["s"][:, 0]
    knn = None
    if penalty == "agent_knn":
        o = env.obs(data["s"].reshape(-1, env.n, 4))
        knn = KNN(torch.cat([o, data["a"].reshape(-1, env.n, 2)], -1).reshape(-1, obs_dim + 2), gen=gen)
    elif penalty == "joint_knn":
        knn = KNN(torch.cat([data["s"].reshape(-1, env.n * 4), data["a"].reshape(-1, env.n * 2)], -1), gen=gen)
    for _ in range(iters):
        s0 = s0_pool[torch.randint(0, s0_pool.shape[0], (batch,), generator=gen, device=DEV)]
        r, uf, ur, S, A = rollout(env, pol, s0, wm, member, gen)
        obj = r.sum(1)
        if wm is not None and lam > 0:
            if penalty == "ens":
                obj = obj - lam * uf.sum(1)
            elif penalty == "ens_reward":
                obj = obj - lam * ur.sum(1)
            elif penalty == "agent_knn":
                x = torch.cat([env.obs(S.reshape(-1, env.n, 4)), A.reshape(-1, env.n, 2)], -1)
                obj = obj - lam * knn(x.reshape(-1, obs_dim + 2)).reshape(batch, T, env.n).sum((1, 2)) / env.n
            elif penalty == "joint_knn":
                x = torch.cat([S.reshape(batch * T, -1), A.reshape(batch * T, -1)], -1)
                obj = obj - lam * knn(x).reshape(batch, T).sum(1)
        loss = -obj.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 10.0)
        opt.step()
    return pol


@torch.no_grad()
def evaluate(env, pol, wm, gen, episodes=2000):
    s0 = env.reset(episodes, gen)
    r_true, _, _, S, _ = rollout(env, pol, s0)
    s_final = env.step(S[:, -1], pol(env.obs(S[:, -1])))
    out = {"true_return": r_true.sum(1).mean().item(),
           "duplicate_rate": duplicate_rate(env, s_final), "final_coverage": coverage(env, s_final)}
    if wm is not None:
        r_mod, uf, _, _, _ = rollout(env, pol, s0, wm, member="mean")
        out["model_return"] = r_mod.sum(1).mean().item()
        out["exploit_gap"] = out["model_return"] - out["true_return"]
        out["mean_uncertainty"] = uf.mean().item()
    return out


@torch.no_grad()
def diagnostics(env, wm, gen):
    """Reward-model error and ensemble spread on held-out behaviour states, on
    states where two agents converge on one landmark, and on uniform states."""
    out = {}
    for name in ("assignment", "duplicate", "uniform"):
        if name == "uniform":
            s = torch.cat([torch.rand(4000, env.n, 2, generator=gen, device=DEV) * 2 - 1,
                           torch.randn(4000, env.n, 2, generator=gen, device=DEV) * 0.5], -1)
            a = torch.rand(4000, env.n, 2, generator=gen, device=DEV) * 2 - 1
        else:
            d = collect(env, name, 400, gen)
            s, a = d["s"][:, -10:].reshape(-1, env.n, 4), d["a"][:, -10:].reshape(-1, env.n, 2)
        s2, r = wm.predict(s, a)
        err = r.mean(0) - env.reward(env.step(s, a))
        std = r.std(0)
        out[name] = {"abs_err": err.abs().mean().item(), "optimism": err.mean().item(),
                     "ens_std": std.mean().item(), "std_over_err": (std.mean() / err.abs().mean()).item()}
    return out


def scripted_best(env, gen, episodes=2000):
    """Hungarian assignment from the start positions + the noiseless controller,
    best over a grid of controller gains. Returns (return, final coverage, kp, kd)."""
    s0 = env.reset(episodes, gen)
    s = s0
    p = s[..., :2].cpu().numpy()
    L = env.L.cpu().numpy()
    tgt = np.zeros_like(p)
    for i in range(episodes):
        r, c = linear_sum_assignment(np.linalg.norm(p[i][:, None] - L[None], axis=-1))
        tgt[i, r] = L[c]
    tgt = torch.as_tensor(tgt, device=DEV)
    best = None
    for kp in (2.0, 4.0, 8.0, 16.0):
        for kd in (0.5, 1.0, 2.0):
            s, total = s0, 0
            for _ in range(T):
                s = env.step(s, controller(s, tgt, torch.zeros(episodes, device=DEV), gen, kp, kd))
                total = total + env.reward(s)
            cand = (total.mean().item(), coverage(env, s), kp, kd)
            if best is None or cand[0] > best[0]:
                best = cand
    return best


# ---------------------------------------------------------------- experiment

def sample_landmarks(n, gen, min_dist=0.3):
    # Landmarks closer than the 2*RAD collision distance cannot both be covered without a collision,
    # which makes the layout degenerate; resample the whole layout until all pairs are min_dist apart.
    while True:
        L = torch.rand(n, 2, generator=gen, device=DEV) * 1.6 - 0.8
        if n < 2 or torch.pdist(L).min().item() >= min_dist:
            return L


def run_cell(args, n, seed, data_kind, arch, configs):
    gen = torch.Generator(device=DEV).manual_seed(seed * 1000 + n)
    torch.manual_seed(seed * 1000 + n)
    landmarks = sample_landmarks(n, gen)
    env = Spread(n, landmarks, agent_id=not args.no_agent_id)
    data = collect(env, data_kind, args.episodes, gen)
    t = time.time()
    wm = WorldModel(arch, n, k=args.ensemble)
    train_world_model(wm, data, gen, args.wm_steps)
    res = {"n": n, "seed": seed, "data": data_kind, "arch": arch, "ensemble": args.ensemble,
           "agent_id": not args.no_agent_id,
           "episodes": args.episodes, "wm_time": time.time() - t,
           "behaviour_return": data["r"].sum(1).mean().item(),
           "diag": diagnostics(env, wm, gen), "policies": []}
    for penalty, lam in configs:
        t = time.time()
        pol = train_policy(env, wm, penalty, lam, data, gen, args.iters, batch=args.batch, lr=args.lr,
                           member=args.member, hidden=args.pol_hidden)
        ev = evaluate(env, pol, wm, gen)
        ev.update({"penalty": penalty, "lam": lam, "time": time.time() - t})
        res["policies"].append(ev)
        print(f"  policy {penalty}:{lam:g} {ev['time']:.0f}s", flush=True)
    return res


def run_reference(args, n, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed * 1000 + n)
    torch.manual_seed(seed * 1000 + n)
    landmarks = sample_landmarks(n, gen)
    env = Spread(n, landmarks, agent_id=not args.no_agent_id)
    data = collect(env, "assignment", min(args.episodes, 2000), gen)
    pol = train_policy(env, None, "none", 0.0, data, gen, args.iters, batch=args.batch, lr=args.lr,
                       hidden=args.pol_hidden)
    ret, cov, kp, kd = scripted_best(env, gen)
    return {"n": n, "seed": seed, "agent_id": not args.no_agent_id,
            "oracle": evaluate(env, pol, None, gen), "scripted_best": ret,
            "scripted_coverage": cov, "scripted_gains": [kp, kd]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="2,4,8")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--archs", default="joint,additive,attention")
    ap.add_argument("--data", default="assignment,independent")
    ap.add_argument("--configs", default="none:0,ens:1,ens:5,ens_reward:5,agent_knn:1,joint_knn:1")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--wm_steps", type=int, default=6000)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--member", default="random", choices=["random", "mean"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256, help="policy-training batch of start states")
    ap.add_argument("--pol_hidden", type=int, default=128)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--no_agent_id", action="store_true", help="drop the agent id from the policy observation")
    ap.add_argument("--no_reference", action="store_true")
    ap.add_argument("--reference_only", action="store_true",
                    help="only train the true-dynamics oracle, to check the policy optimiser reaches the scripted best")
    ap.add_argument("--out", default="spread.json")
    args = ap.parse_args()
    print(f"device: {DEV}", flush=True)
    configs = [(c.split(":")[0], float(c.split(":")[1])) for c in args.configs.split(",")]
    out = {"args": vars(args), "cells": [], "reference": []}
    def save():  # write then rename, so a kill mid-write never leaves a truncated file
        with open(args.out + ".tmp", "w") as f:
            json.dump(out, f, indent=1)
        os.replace(args.out + ".tmp", args.out)

    for n in [int(x) for x in args.ns.split(",")]:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            if not args.no_reference:
                t = time.time()
                out["reference"].append(run_reference(args, n, seed))
                save()
                print(f"reference n={n} seed={seed} {time.time() - t:.0f}s", flush=True)
            if args.reference_only:
                continue
            for data_kind in args.data.split(","):
                for arch in args.archs.split(","):
                    t = time.time()
                    out["cells"].append(run_cell(args, n, seed, data_kind, arch, configs))
                    save()  # before printing, so a lost stdout cannot lose the cell
                    print(f"n={n} seed={seed} data={data_kind} arch={arch} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
