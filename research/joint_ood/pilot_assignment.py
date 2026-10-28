"""Pilot: do cooperating agents jointly exploit a learned model?

One-step N-agent assignment game. N landmarks sit on the unit circle and
agent i starts at s_i. Each agent picks a target point a_i. The team reward
is coverage (every landmark should have an agent on it) minus travel cost:

    r(a) = -(1/N) sum_j min_i |a_i - L_j|  -  alpha (1/N) sum_i |a_i - s_i|

The offline data come from random assignments (a permutation of landmarks
to agents, plus noise). Every agent visits every landmark in the data, but
two agents never share a landmark. So a joint action where two agents pick
the same landmark is "marginally in-distribution, jointly out": each agent's
own action looks familiar, the combination was never seen.

We fit an ensemble reward model, then optimise the joint action against it
with the usual MOPO-style penalty (mean - lambda * std) and, optionally, a
per-agent support constraint (each agent stays near some landmark, which is
what per-agent behaviour regularisation enforces). We measure the exploit
gap (model reward minus true reward), the duplicate rate and the true regret.

Run:  python research/joint_ood/pilot_assignment.py --out results.json
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

ALPHA = 0.5
NOISE = 0.1
BOX = 1.5


def landmarks(n):
    ang = 2 * np.pi * np.arange(n) / n
    return np.stack([np.cos(ang), np.sin(ang)], -1)


def true_reward(a, L, s):
    # a: (B, N, 2)
    d = jnp.linalg.norm(a[:, :, None, :] - L[None, None], axis=-1)  # B,N(agents),N(landmarks)
    cover = d.min(axis=1).mean(-1)
    travel = jnp.linalg.norm(a - s[None], axis=-1).mean(-1)
    return -cover - ALPHA * travel


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

def init_mlp(key, sizes):
    params = []
    for din, dout in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        params.append((jax.random.normal(k, (din, dout)) * jnp.sqrt(2.0 / din), jnp.zeros(dout)))
    return params


def mlp(params, x):
    for w, b in params[:-1]:
        x = jax.nn.swish(x @ w + b)
    w, b = params[-1]
    return (x @ w + b)[..., 0]


def make_model(arch, n, hidden=256):
    """Reward model over the joint action, three ways to treat the agents.

    joint      one MLP on the concatenated joint action
    attention  one token per agent (action + agent id), a self-attention layer,
               mean pooling, MLP head; the set-structured design of recent
               multi-agent world models
    additive   sum over agents of a shared per-agent MLP (action + agent id);
               fully factored, cannot represent interactions
    """
    eye = jnp.eye(n)

    def tokens(x):
        a = x.reshape(x.shape[:-1] + (n, 2))
        ids = jnp.broadcast_to(eye, a.shape[:-1] + (n,))
        return jnp.concatenate([a, ids], -1)

    if arch == "joint":
        def init(k):
            return init_mlp(k, [2 * n, hidden, hidden, hidden, 1])

        def apply(p, x):
            return mlp(p, x)
    elif arch == "additive":
        def init(k):
            return init_mlp(k, [2 + n, hidden, hidden, 1])

        def apply(p, x):
            return mlp(p, tokens(x)).sum(-1)
    elif arch == "attention":
        d = 128

        def init(k):
            ks = jax.random.split(k, 6)
            lin = lambda kk, i, o: jax.random.normal(kk, (i, o)) * jnp.sqrt(1.0 / i)
            return {"inp": init_mlp(ks[0], [2 + n, d, d]), "q": lin(ks[1], d, d), "k": lin(ks[2], d, d),
                    "v": lin(ks[3], d, d), "ff": init_mlp(ks[4], [d, d, d]), "head": init_mlp(ks[5], [d, hidden, 1])}

        def apply(p, x):
            h = tokens(x)
            for w, b in p["inp"]:
                h = jax.nn.swish(h @ w + b)
            q, kk, v = h @ p["q"], h @ p["k"], h @ p["v"]
            att = jax.nn.softmax(q @ jnp.swapaxes(kk, -1, -2) / jnp.sqrt(d), -1)
            h = h + att @ v
            f = h
            for w, b in p["ff"][:-1]:
                f = jax.nn.swish(f @ w + b)
            w, b = p["ff"][-1]
            h = h + f @ w + b
            return mlp(p["head"], h.mean(-2))
    else:
        raise ValueError(arch)
    return init, apply


def train_ensemble(key, X, y, model, k_models=5, steps=3000, batch=512, lr=1e-3):
    init, apply = model
    keys = jax.random.split(key, k_models + 1)
    params = jax.vmap(init)(keys[:k_models])
    opt = optax.adam(lr)
    state = jax.vmap(opt.init)(params)
    n = X.shape[0]

    def loss(p, xb, yb):
        return jnp.mean((apply(p, xb) - yb) ** 2)

    @jax.jit
    def step(params, state, idx):
        # idx: (K, batch) bootstrap indices, one set per member
        def one(p, s, i):
            g = jax.grad(loss)(p, X[i], y[i])
            u, s = opt.update(g, s, p)
            return optax.apply_updates(p, u), s
        return jax.vmap(one)(params, state, idx)

    rng = np.random.default_rng(int(jax.random.randint(keys[-1], (), 0, 2**31 - 1)))
    boot = np.stack([rng.integers(0, n, n) for _ in range(k_models)])  # bootstrap resample
    for _ in range(steps):
        pos = rng.integers(0, n, (k_models, batch))
        idx = np.take_along_axis(boot, pos, axis=1)
        params, state = step(params, state, jnp.asarray(idx))
    return params


def ens_predict(params, X, apply):
    return jax.vmap(lambda p: apply(p, X))(params)  # K,B


# ---------------------------------------------------------------- planner

def plan(params, apply, norm, data_flat, L, n, lam, mu_agent, mu_joint, rng, restarts=256, steps=300, lr=0.02):
    mean_x, std_x, mean_y, std_y = norm

    def objective(a_flat):
        x = (a_flat - mean_x) / std_x
        preds = ens_predict(params, x, apply) * std_y + mean_y  # K,B
        m, s = preds.mean(0), preds.std(0)
        a = a_flat.reshape(-1, n, 2)
        agent_sup = jnp.min(jnp.sum((a[:, :, None, :] - L[None, None]) ** 2, -1), -1).sum(-1)
        obj = m - lam * s - mu_agent * agent_sup
        if mu_joint > 0:
            dj = jnp.min(jnp.sum((a_flat[:, None, :] - data_flat[None]) ** 2, -1), -1)
            obj = obj - mu_joint * dj
        return obj, (m, s)

    # Half the restarts start at logged joint actions. The other half recombine
    # logged actions agent by agent (agent i's action from one sample, agent j's
    # from another), which is where independently trained decentralised
    # policies end up when each settles on its own preferred mode.
    half = restarts // 2
    logged = np.asarray(data_flat)[rng.integers(0, data_flat.shape[0], half)]
    rows = rng.integers(0, data_flat.shape[0], (restarts - half, n))
    mixed = np.asarray(data_flat).reshape(-1, n, 2)[rows, np.arange(n)[None]].reshape(restarts - half, -1)
    a = jnp.asarray(np.concatenate([logged, mixed]))
    opt = optax.adam(lr)
    st = opt.init(a)

    @jax.jit
    def step(a, st):
        g = jax.grad(lambda z: -objective(z)[0].sum())(a)
        u, st = opt.update(g, st, a)
        return jnp.clip(optax.apply_updates(a, u), -BOX, BOX), st

    for _ in range(steps):
        a, st = step(a, st)
    obj, (m, s) = objective(a)
    best = int(jnp.argmax(obj))
    return np.asarray(a[best]).reshape(n, 2), float(m[best]), float(s[best])


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


def run(n, seed, data_size, lams, settings, arch):
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    L = landmarks(n)
    s = rng.uniform(-1, 1, (n, 2))
    Lj, sj = jnp.asarray(L), jnp.asarray(s)

    A = sample_perm(rng, n, data_size, L)
    R = np.asarray(true_reward(jnp.asarray(A), Lj, sj)) + 0.01 * rng.standard_normal(data_size)
    X = A.reshape(data_size, -1)
    mean_x, std_x = X.mean(0), X.std(0) + 1e-6
    mean_y, std_y = R.mean(), R.std() + 1e-6
    model = make_model(arch, n)
    apply = model[1]
    params = train_ensemble(key, jnp.asarray((X - mean_x) / std_x), jnp.asarray((R - mean_y) / std_y), model)
    norm = (jnp.asarray(mean_x), jnp.asarray(std_x), mean_y, std_y)

    # diagnostics on three kinds of joint action
    diag = {}
    for name, sampler in [("in_dist", lambda: sample_perm(rng, n, 2000, L)),
                          ("marginal_in_joint_out", lambda: sample_marginal_joint_ood(rng, n, 2000, L)),
                          ("uniform_ood", lambda: sample_uniform(rng, n, 2000))]:
        a = sampler()
        x = jnp.asarray((a.reshape(len(a), -1) - mean_x) / std_x)
        p = np.asarray(ens_predict(params, x, apply)) * std_y + mean_y
        signed = p.mean(0) - np.asarray(true_reward(jnp.asarray(a), Lj, sj))
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

    a_opt = optimum(L, s)
    r_star = float(true_reward(jnp.asarray(a_opt[None]), Lj, sj)[0])
    r_data_best = float(R.max())
    r_behavior = float(R.mean())

    plans = []
    data_flat = jnp.asarray(X)
    for setting in settings:
        for lam in lams:
            a, m, sd = plan(params, apply, norm, data_flat, Lj, n, lam, setting["mu_agent"], setting["mu_joint"], rng)
            r = float(true_reward(jnp.asarray(a[None]), Lj, sj)[0])
            plans.append({"setting": setting["name"], "lam": lam, "model_r": m, "model_std": sd, "true_r": r,
                          "exploit_gap": m - r, "regret": r_star - r,
                          "duplicate": bool(not is_permutation(a[None], L)[0])})
    return {"arch": arch, "n": n, "seed": seed, "r_star": r_star, "r_data_best": r_data_best, "r_behavior": r_behavior,
            "frac_joint_space_in_support": float(np.prod(np.arange(1, n + 1)) / n ** n),
            "diag": diag, "plans": plans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="2,3,4,6,8")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--data", type=int, default=20000)
    ap.add_argument("--archs", default="joint,attention,additive")
    ap.add_argument("--out", default="pilot_assignment.json")
    args = ap.parse_args()
    lams = [0.0, 1.0, 3.0, 10.0]
    settings = [{"name": "ensemble_penalty", "mu_agent": 0.0, "mu_joint": 0.0},
                {"name": "plus_per_agent_support", "mu_agent": 10.0, "mu_joint": 0.0},
                {"name": "plus_joint_support_oracle", "mu_agent": 0.0, "mu_joint": 10.0}]
    results = []
    for arch in args.archs.split(","):
        for n in [int(x) for x in args.ns.split(",")]:
            for seed in range(args.seeds):
                t = time.time()
                res = run(n, seed, args.data, lams, settings, arch)
                results.append(res)
                print(f"arch={arch} n={n} seed={seed} {time.time() - t:.0f}s", flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
