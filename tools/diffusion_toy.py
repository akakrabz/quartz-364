#!/usr/bin/env python3
"""A tiny DDPM on 2-D data, in pure numpy (no autograd).

Used by tools/figs_diff_b.py for the figures on the 8-diffusion-models
DDPM and fast-sampling pages; the page code is this code. Everything here
was run; the PyTorch version on the page mirrors it line for line.

    from diffusion_toy import train, sample_ddpm, sample_ddim, sample_heun
"""
import numpy as np

T = 100                                     # diffusion steps
SCALE = 1000 / T                            # keep total noise like T=1000
BETAS = np.linspace(SCALE * 1e-4, SCALE * 0.02, T)   # beta_1..beta_T
ALPHAS = 1.0 - BETAS
ALPHA_BAR = np.cumprod(ALPHAS)              # alpha_bar_1..alpha_bar_T
EMB = 16                                    # time-embedding size


# ---------------------------------------------------------------- data
def two_moons(n, rng, noise=0.06):
    """Two interleaved half circles, scaled to about unit variance."""
    n1 = n // 2
    a = rng.uniform(0, np.pi, n1)
    b = rng.uniform(0, np.pi, n - n1)
    x = np.concatenate([np.c_[np.cos(a), np.sin(a)],
                        np.c_[1 - np.cos(b), 0.5 - np.sin(b)]])
    x = x + noise * rng.standard_normal(x.shape)
    x = (x - np.array([0.5, 0.25])) / np.array([0.87, 0.5])  # center, scale
    return x[rng.permutation(n)]


# ---------------------------------------------------------------- model
def time_embedding(t):
    """Sinusoidal embedding of integer steps t."""
    half = EMB // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half) / half)
    ang = np.asarray(t, float)[:, None] * freqs[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


def silu(z):
    return z / (1 + np.exp(-np.clip(z, -60, 60)))


def silu_grad(z):
    s = 1 / (1 + np.exp(-np.clip(z, -60, 60)))
    return s * (1 + z * (1 - s))


def init_params(rng, hidden=128):
    sizes = [2 + EMB, hidden, hidden, hidden, 2]
    params = []
    for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        W = rng.standard_normal((m, n)) * np.sqrt(2.0 / m)
        if i == len(sizes) - 2:
            W *= 0.1                        # small last layer
        params += [W, np.zeros(n)]
    return params


def forward(params, x, t, cache=False):
    """eps_theta(x_t, t): MLP on [x_t, emb(t)]."""
    h = np.concatenate([x, time_embedding(t)], axis=1)
    acts = [h]
    pre = []
    n_layers = len(params) // 2
    for i in range(n_layers):
        W, b = params[2 * i], params[2 * i + 1]
        z = h @ W + b
        pre.append(z)
        h = silu(z) if i < n_layers - 1 else z
        acts.append(h)
    return (h, (acts, pre)) if cache else h


def loss_and_grads(params, x, t, eps):
    """L_simple on a batch, and its gradients."""
    out, (acts, pre) = forward(params, x, t, cache=True)
    diff = out - eps
    loss = np.mean(np.sum(diff ** 2, axis=1))
    g = 2 * diff / x.shape[0]               # dL/d out
    grads = [None] * len(params)
    n_layers = len(params) // 2
    for i in reversed(range(n_layers)):
        if i < n_layers - 1:
            g = g * silu_grad(pre[i])
        grads[2 * i] = acts[i].T @ g
        grads[2 * i + 1] = g.sum(axis=0)
        g = g @ params[2 * i].T
    return loss, grads


# ---------------------------------------------------------------- training
def train(steps=20000, batch=256, lr=1e-3, seed=0, n_data=8000):
    """Algorithm 1 with Adam. Returns params, losses, data."""
    rng = np.random.default_rng(seed)
    data = two_moons(n_data, rng)
    params = init_params(rng)
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    losses = []
    for k in range(1, steps + 1):
        x0 = data[rng.integers(0, n_data, batch)]           # x0 ~ q(x0)
        t = rng.integers(1, T + 1, batch)                    # t ~ U{1..T}
        eps = rng.standard_normal(x0.shape)                  # eps ~ N(0, I)
        ab = ALPHA_BAR[t - 1][:, None]
        xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps        # q(x_t | x0)
        loss, grads = loss_and_grads(params, xt, t, eps)
        lr_k = lr * min(1.0, 2 * (steps - k) / steps + 0.1)  # decay late
        for i in range(len(params)):                         # Adam step
            m[i] = b1 * m[i] + (1 - b1) * grads[i]
            v[i] = b2 * v[i] + (1 - b2) * grads[i] ** 2
            mh = m[i] / (1 - b1 ** k)
            vh = v[i] / (1 - b2 ** k)
            params[i] -= lr_k * mh / (np.sqrt(vh) + eps_adam)
        losses.append(loss)
    return params, np.array(losses), data


# ---------------------------------------------------------------- samplers
def sample_ddpm(params, n, rng, keep=()):
    """Algorithm 2 with sigma_t^2 = beta_t."""
    x = rng.standard_normal((n, 2))                          # x_T
    snaps = {T: x.copy()}
    for t in range(T, 0, -1):
        z = rng.standard_normal(x.shape) if t > 1 else 0.0
        eps = forward(params, x, np.full(n, t))
        a, ab, beta = ALPHAS[t - 1], ALPHA_BAR[t - 1], BETAS[t - 1]
        x = (x - beta / np.sqrt(1 - ab) * eps) / np.sqrt(a) + np.sqrt(beta) * z
        if t - 1 in keep:
            snaps[t - 1] = x.copy()
    return x, snaps


def _ab(t):
    """alpha_bar at step t, with alpha_bar_0 = 1."""
    return 1.0 if t == 0 else ALPHA_BAR[t - 1]


def timesteps(S, spacing="uniform"):
    """Increasing subsequence tau of 1..T, about S long.

    uniform: evenly spaced in t (DDIM paper);
    karras: evenly spaced in sigma^(1/7) (EDM), rounded to steps.
    """
    if spacing == "uniform":
        return np.unique(np.round(np.linspace(1, T, S)).astype(int))
    sig_all = np.sqrt((1 - ALPHA_BAR) / ALPHA_BAR)       # sigma at t=1..T
    rho, lo, hi = 7.0, sig_all[0], sig_all[-1]
    targets = (hi ** (1 / rho) + np.linspace(0, 1, S) * (lo ** (1 / rho) - hi ** (1 / rho))) ** rho
    idx = np.abs(np.log(sig_all)[None, :] - np.log(targets)[:, None]).argmin(1)
    return np.unique(idx + 1)


def sample_ddim(params, n, rng, S, eta=0.0, x_T=None, return_path=False, spacing="uniform"):
    """DDIM eq. 12; eta=1 is DDPM-like."""
    tau = timesteps(S, spacing)
    x = rng.standard_normal((n, 2)) if x_T is None else x_T.copy()
    path = [x.copy()]
    for i in range(len(tau) - 1, -1, -1):
        t, t_prev = tau[i], (tau[i - 1] if i > 0 else 0)
        ab, ab_prev = _ab(t), _ab(t_prev)
        eps = forward(params, x, np.full(n, t))
        x0_hat = (x - np.sqrt(1 - ab) * eps) / np.sqrt(ab)    # predicted x0
        sigma = eta * np.sqrt((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev))
        dir_xt = np.sqrt(np.maximum(1 - ab_prev - sigma ** 2, 0)) * eps
        noise = sigma * rng.standard_normal(x.shape) if t_prev > 0 else 0.0
        x = np.sqrt(ab_prev) * x0_hat + dir_xt + noise
        path.append(x.copy())
    return (x, path) if return_path else x


def sample_heun(params, n, rng, S, x_T=None, spacing="uniform"):
    """Heun (EDM-style) on the probability-flow ODE in sigma."""
    tau = timesteps(S, spacing)
    sig = lambda t: np.sqrt((1 - _ab(t)) / _ab(t))          # noise level
    x = rng.standard_normal((n, 2)) if x_T is None else x_T.copy()
    xb = x / np.sqrt(_ab(tau[-1]))                           # scaled state
    for i in range(len(tau) - 1, -1, -1):
        t, t_prev = tau[i], (tau[i - 1] if i > 0 else 0)
        s, s_prev = sig(t), sig(t_prev)
        d = forward(params, xb * np.sqrt(_ab(t)), np.full(n, t))   # dx/dsigma
        x_euler = xb + (s_prev - s) * d
        if t_prev > 0:                                       # 2nd order fix
            d2 = forward(params, x_euler * np.sqrt(_ab(t_prev)), np.full(n, t_prev))
            xb = xb + (s_prev - s) * 0.5 * (d + d2)
        else:
            xb = x_euler
    return xb                                                # sigma=0: x


def nfe_heun(S, spacing="uniform"):
    return 2 * len(timesteps(S, spacing)) - 1


# ---------------------------------------------------------------- metric
def energy_distance(a, b, rng, m=2000):
    """Unbiased 2E|X-Y| - E|X-X'| - E|Y-Y'|."""
    a = a[rng.choice(len(a), min(m, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(m, len(b)), replace=False)]
    dist = lambda p, q: np.sqrt(((p[:, None, :] - q[None, :, :]) ** 2).sum(-1))
    within = lambda p: dist(p, p).sum() / (len(p) * (len(p) - 1))  # no diagonal
    return 2 * dist(a, b).mean() - within(a) - within(b)


def grad_check(seed=1):
    """Finite-difference check of loss_and_grads."""
    rng = np.random.default_rng(seed)
    params = init_params(rng, hidden=8)
    x = rng.standard_normal((5, 2))
    t = rng.integers(1, T + 1, 5)
    eps = rng.standard_normal((5, 2))
    _, grads = loss_and_grads(params, x, t, eps)
    worst = 0.0
    for i, p in enumerate(params):
        for idx in [tuple(rng.integers(0, s) for s in p.shape) for _ in range(4)]:
            old = p[idx]
            p[idx] = old + 1e-6
            lp, _ = loss_and_grads(params, x, t, eps)
            p[idx] = old - 1e-6
            lm, _ = loss_and_grads(params, x, t, eps)
            p[idx] = old
            num = (lp - lm) / 2e-6
            worst = max(worst, abs(num - grads[i][idx]) / max(1e-8, abs(num) + abs(grads[i][idx])))
    return worst


if __name__ == "__main__":
    print("grad check rel. error:", grad_check())
    print("alpha_bar_T =", ALPHA_BAR[-1])
