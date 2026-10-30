#!/usr/bin/env python3
"""Rectified flow / flow matching on 2-D data, in pure numpy (no autograd).

Used by tools/figs_diff_c.py for the flow-matching page; the code on the page
is this code. Convention (Liu et al. 2022): X0 ~ N(0, I) noise at t = 0,
X1 ~ data at t = 1, X_t = t X1 + (1 - t) X0, regression target X1 - X0.

    from flow_toy import two_moons, train, euler, straightness, sliced_w1
"""
import numpy as np

NF = 4                                      # Fourier features of t


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


# ---------------------------------------------------------------- model: MLP v(x, t)
def features(x, t):
    """Input: position plus Fourier features of t."""
    k = 2 * np.pi * np.arange(1, NF + 1)
    tt = t[:, None] * k[None, :]
    return np.concatenate([x, t[:, None], np.sin(tt), np.cos(tt)], axis=1)


def silu(z):
    return z / (1 + np.exp(-z))


def silu_grad(z):
    s = 1 / (1 + np.exp(-z))
    return s * (1 + z * (1 - s))


def init_params(rng, hidden=128):
    sizes = [3 + 2 * NF, hidden, hidden, hidden, 2]
    params = []
    for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        W = rng.standard_normal((m, n)) * np.sqrt(2.0 / m)
        if i == len(sizes) - 2:
            W *= 0.1                        # small last layer
        params += [W, np.zeros(n)]
    return params


def forward(params, x, t):
    """Velocity v(x, t); also returns the cache for backprop."""
    h = features(x, t)
    cache = []
    n_layers = len(params) // 2
    for i in range(n_layers):
        W, b = params[2 * i], params[2 * i + 1]
        z = h @ W + b
        cache.append((h, z))
        h = silu(z) if i < n_layers - 1 else z
    return h, cache


def backward(params, cache, dout):
    """Gradients of the loss for every parameter."""
    grads = [None] * len(params)
    n_layers = len(params) // 2
    d = dout
    for i in reversed(range(n_layers)):
        h, z = cache[i]
        if i < n_layers - 1:
            d = d * silu_grad(z)
        grads[2 * i] = h.T @ d
        grads[2 * i + 1] = d.sum(0)
        d = d @ params[2 * i].T
    return grads


def adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    state["t"] += 1
    for i, g in enumerate(grads):
        state["m"][i] = b1 * state["m"][i] + (1 - b1) * g
        state["v"][i] = b2 * state["v"][i] + (1 - b2) * g * g
        mh = state["m"][i] / (1 - b1 ** state["t"])
        vh = state["v"][i] / (1 - b2 ** state["t"])
        params[i] -= lr * mh / (np.sqrt(vh) + eps)


# ---------------------------------------------------------------- training
def train(pairs, rng, steps=6000, batch=512, lr=2e-3, params=None):
    """Rectified-flow regression on a coupling (X0, X1).

    pairs: function(n, rng) -> (x0, x1), each of shape (n, 2).
    """
    if params is None:
        params = init_params(rng)
    state = {"t": 0, "m": [np.zeros_like(p) for p in params],
             "v": [np.zeros_like(p) for p in params]}
    losses = []
    for step in range(steps):
        x0, x1 = pairs(batch, rng)
        t = rng.uniform(0, 1, batch)
        xt = t[:, None] * x1 + (1 - t[:, None]) * x0   # straight path
        target = x1 - x0                               # constant velocity
        v, cache = forward(params, xt, t)
        err = v - target
        losses.append(np.mean(np.sum(err ** 2, axis=1)))
        dout = 2 * err / batch                         # d(mean ||err||^2)
        grads = backward(params, cache, dout)
        a = lr * 0.5 * (1 + np.cos(np.pi * step / steps))  # cosine decay
        adam_step(params, grads, state, a)
    return params, np.array(losses)


def independent_pairs(n_data=20000, seed=1):
    """1-rectified flow: noise and data drawn independently."""
    data = two_moons(n_data, np.random.default_rng(seed))

    def pairs(n, rng):
        x1 = data[rng.integers(0, len(data), n)]
        return rng.standard_normal((n, 2)), x1
    return pairs


def fixed_pairs(z0, z1):
    """Reflow: the coupling (Z0, Z1) produced by a previous flow."""
    def pairs(n, rng):
        i = rng.integers(0, len(z0), n)
        return z0[i], z1[i]
    return pairs


# ---------------------------------------------------------------- sampling
def euler(params, z0, n_steps, keep_path=False):
    """Integrate dZ/dt = v(Z, t) from t = 0 to 1."""
    z = z0.copy()
    path = [z.copy()]
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = np.full(len(z), k * dt)
        v, _ = forward(params, z, t)
        z = z + dt * v
        if keep_path:
            path.append(z.copy())
    return (z, np.array(path)) if keep_path else z


def straightness(params, z0, n_steps=100):
    """S = int E||(Z1 - Z0) - v(Z_t, t)||^2 dt; zero for straight paths."""
    z = z0.copy()
    vs = []
    dt = 1.0 / n_steps
    for k in range(n_steps):
        v, _ = forward(params, z, np.full(len(z), k * dt))
        vs.append(v)
        z = z + dt * v
    disp = z - z0
    return float(np.mean([np.mean(np.sum((disp - v) ** 2, 1)) for v in vs]))


def sliced_w1(a, b, rng, n_dir=128):
    """Sliced Wasserstein-1 distance between two 2-D point clouds."""
    th = rng.uniform(0, np.pi, n_dir)
    dirs = np.c_[np.cos(th), np.sin(th)]
    pa, pb = np.sort(a @ dirs.T, 0), np.sort(b @ dirs.T, 0)
    return float(np.mean(np.abs(pa - pb)))
