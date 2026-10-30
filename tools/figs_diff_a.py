#!/usr/bin/env python3
"""Figures for unit 8, pages 1-2 (Brownian motion / forward diffusion,
reverse-time SDE / score matching). Run: python3 tools/figs_diff_a.py

Writes tools/figs/diff-a-*.html; then
    python3 tools/assemble.py content/8-diffusion-models/brownian-motion-and-forward-diffusion.md
    python3 tools/assemble.py content/8-diffusion-models/reverse-time-sde-and-score-matching.md
Everything is computed here with numpy. The toy data are Gaussian mixtures,
so every score is exact: no network is trained.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mplfig import ACCENT, ACCENT2, HI, INK, MUTED, plt, save  # noqa: E402

NUMBERS = {}  # values quoted on the pages, printed at the end
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def compact(name):
    """Shrink a saved figure: round coords, group scatter styles."""
    import re
    path = os.path.join(FIGDIR, name + ".html")
    s = open(path).read()
    ids = {}
    for i, old in enumerate(dict.fromkeys(re.findall(r'id="(%s-m[0-9a-f]{6,})"' % re.escape(name), s))):
        ids[old] = f"{name}-k{i}"
    for old, new in ids.items():
        s = s.replace(old, new)
    s = s.replace("xlink:href=", "href=")
    head, sep, tail = s.partition("<figcaption>")           # never touch captions
    head = re.sub(r"<[^<>]+>", lambda t: re.sub(r"(\d+\.\d)\d+", r"\1", t.group(0)), head)
    s = head + sep + tail                                    # rounds inside tags only

    def group(m):
        uses = re.findall(r'<use href="([^"]+)" x="([^"]+)" y="([^"]+)" style="([^"]+)"/>', m.group(0))
        out, cur = [], None
        for href, x, y, st in uses:
            if (href, st) != cur:
                if cur:
                    out.append("</g>")
                out.append(f'<g style="{st}">')
                cur = (href, st)
            out.append(f'<use href="{href}" x="{x}" y="{y}"/>')
        out.append("</g>")
        return "".join(out)
    s = re.sub(r'(?:<use href="[^"]+" x="[^"]+" y="[^"]+" style="[^"]+"/>)+', group, s)
    open(path, "w").write(s)


# ======================================================================
# Shared toy model: VP SDE with linear beta(t), Gaussian-mixture data.
# (This block is shown on the reverse-SDE page; keep it in sync.)
# ======================================================================
B_MIN, B_MAX = 0.1, 20.0                     # Song et al. (2021)

def beta(t):
    return B_MIN + t * (B_MAX - B_MIN)

def vp_mean_coef(t):                         # m(t) = exp(-1/2 ∫β)
    return np.exp(-0.5 * (B_MIN * t + 0.5 * (B_MAX - B_MIN) * t**2))

def mixture_score(x, t, w, mu, s2):
    """Exact score of a Gaussian-mixture p_t."""
    m = vp_mean_coef(t)
    var = m**2 * s2 + 1 - m**2               # per-component variance
    diff = x[:, None, :] - m * mu[None]      # (n, K, d)
    logp = (np.log(w)[None] - 0.5 * (diff**2).sum(-1) / var
            - 0.5 * mu.shape[1] * np.log(var))
    r = np.exp(logp - logp.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)             # responsibilities
    return -(r[..., None] * diff).sum(1) / var

def reverse_sde(x, w, mu, s2, n_steps=1000, t_eps=1e-3, rng=None, keep=()):
    """Euler-Maruyama on the reverse-time VP SDE."""
    ts = np.linspace(1.0, t_eps, n_steps + 1)
    snaps = {}
    for i in range(n_steps):
        t, dt = ts[i], ts[i] - ts[i + 1]     # dt > 0, time runs down
        f = -0.5 * beta(t) * x
        g2 = beta(t)
        drift = f - g2 * mixture_score(x, t, w, mu, s2)
        x = x - drift * dt + np.sqrt(g2 * dt) * rng.standard_normal(x.shape)
        if i + 1 in keep:
            snaps[i + 1] = x.copy()
    return x, snaps

def prob_flow_ode(x, w, mu, s2, n_steps=1000, t_eps=1e-3, keep=()):
    """Euler on the probability-flow ODE, backwards."""
    ts = np.linspace(1.0, t_eps, n_steps + 1)
    snaps = {}
    for i in range(n_steps):
        t, dt = ts[i], ts[i] - ts[i + 1]
        drift = -0.5 * beta(t) * x - 0.5 * beta(t) * mixture_score(x, t, w, mu, s2)
        x = x - drift * dt                   # no noise term
        if i + 1 in keep:
            snaps[i + 1] = x.copy()
    return x, snaps
# ======================================================================


def ring_mixture(k=8, radius=2.0, std=0.12):
    ang = 2 * np.pi * np.arange(k) / k
    mu = radius * np.stack([np.cos(ang), np.sin(ang)], 1)
    return np.full(k, 1.0 / k), mu, std**2


def sample_mixture(n, w, mu, s2, rng):
    k = rng.choice(len(w), size=n, p=w)
    return mu[k] + np.sqrt(s2) * rng.standard_normal((n, mu.shape[1])), k


def shade(ax, dens, extent):
    """Theme-neutral density shading: grey with alpha."""
    rgba = np.zeros(dens.shape + (4,))
    rgba[..., :3] = 0.55
    rgba[..., 3] = 0.85 * dens / dens.max()
    ax.imshow(rgba, origin="lower", aspect="auto", extent=extent, interpolation="bilinear")


def square(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(MUTED)


# ---------------------------------------------------------------- page 1
def brownian_paths():
    rng = np.random.default_rng(3)
    n, T = 1000, 1.0
    dt = T / n
    t = np.linspace(0, T, n + 1)
    dW = np.sqrt(dt) * rng.standard_normal((5, n))
    W = np.concatenate([np.zeros((5, 1)), np.cumsum(dW, 1)], 1)
    # random walk of +-1 steps, rescaled: S_k / sqrt(n)
    k = 20
    steps = rng.choice([-1.0, 1.0], size=k)
    rw_t = np.arange(k + 1) / k
    rw = np.concatenate([[0], np.cumsum(steps)]) / np.sqrt(k)
    many = np.cumsum(np.sqrt(dt) * rng.standard_normal((4000, n)), 1)
    NUMBERS["bm_var_at_1"] = float(many[:, -1].var())
    NUMBERS["bm_var_at_half"] = float(many[:, n // 2 - 1].var())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
    cols = [ACCENT, HI, ACCENT2, INK, MUTED]
    for i in range(5):
        a1.plot(t, W[i], lw=1.1, color=cols[i])
    a1.fill_between(t, -2 * np.sqrt(t), 2 * np.sqrt(t), color=MUTED, alpha=0.18, lw=0)
    a1.plot(t, 2 * np.sqrt(t), color=MUTED, lw=1, ls="--")
    a1.plot(t, -2 * np.sqrt(t), color=MUTED, lw=1, ls="--")
    a1.set_xlabel("t")
    a1.set_ylabel("W(t)")
    a1.set_title("five Brownian paths, band = ±2√t", fontsize=10)
    a2.step(rw_t, rw, where="post", color=ACCENT, lw=1.5, label="±1 walk, 20 steps, ÷√20")
    a2.plot(t, W[0], color=MUTED, lw=0.8, label="Brownian path")
    a2.set_xlabel("t")
    a2.legend(fontsize=8, loc="upper left")
    a2.set_title("a scaled random walk", fontsize=10)
    save(fig, "diff-a-brownian",
         "<b>Brownian motion.</b> Left: five sample paths of a standard Wiener process on [0, 1], simulated "
         "as cumulative sums of 1,000 independent N(0, Δt) increments. The spread grows like √t (dashed: "
         "±2√t); over 4,000 paths the sample variance at t = 1 is "
         f"{NUMBERS['bm_var_at_1']:.3f} and at t = 0.5 it is {NUMBERS['bm_var_at_half']:.3f}. "
         "Right: a ±1 coin-flip walk with its steps scaled by 1/√n already looks like a Brownian path; "
         "as n grows it converges to one (Donsker's theorem).",
         alt="Five random Brownian paths with a square-root envelope, and a scaled random walk")


def linear_betas(T=1000):
    return np.linspace(1e-4, 0.02, T)


def cosine_alpha_bar(T=1000, s=0.008):
    t = np.arange(T + 1)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    betas = np.minimum(1 - ab[1:] / ab[:-1], 0.999)
    return np.cumprod(1 - betas)


def schedules():
    T = 1000
    ab_lin = np.cumprod(1 - linear_betas(T))
    ab_cos = cosine_alpha_bar(T)
    t = np.arange(1, T + 1)
    for name, ab in [("lin", ab_lin), ("cos", ab_cos)]:
        NUMBERS[f"ab_{name}_T"] = float(ab[-1])
        NUMBERS[f"ab_{name}_500"] = float(ab[499])
        NUMBERS[f"ab_{name}_250"] = float(ab[249])
        NUMBERS[f"t_half_{name}"] = int(np.argmax(ab < 0.5) + 1)
    snr = lambda ab: ab / (1 - ab)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    a1.plot(t, ab_lin, color=ACCENT, label="linear β (Ho et al.)")
    a1.plot(t, ab_cos, color=HI, ls="--", label="cosine (Nichol & Dhariwal)")
    a1.set_xlabel("step t")
    a1.set_ylabel("ᾱ_t  (signal fraction)")
    a1.legend(fontsize=8)
    a2.semilogy(t, snr(ab_lin), color=ACCENT)
    a2.semilogy(t, snr(ab_cos), color=HI, ls="--")
    a2.axhline(1, color=MUTED, lw=1)
    a2.set_xlabel("step t")
    a2.set_ylabel("SNR = ᾱ / (1 − ᾱ)")
    a2.set_ylim(1e-6, 1e5)
    save(fig, "diff-a-schedules",
         "<b>Two noise schedules over T = 1,000 steps.</b> Left: the surviving signal fraction ᾱ_t. "
         f"The linear schedule (β from 0.0001 to 0.02) has ᾱ below 0.5 by step {NUMBERS['t_half_lin']} and "
         f"ᾱ_T = {NUMBERS['ab_lin_T']:.1e}; the cosine schedule falls slowly at both ends and crosses 0.5 at "
         f"step {NUMBERS['t_half_cos']}. Right: signal-to-noise ratio on a log scale (SNR = 1 where signal "
         "and noise have equal variance). Computed from the formulas, not from a model.",
         alt="Alpha-bar and signal-to-noise ratio curves for linear and cosine schedules")


def forward_2d():
    rng = np.random.default_rng(0)
    w, mu, s2 = ring_mixture()
    x0, _ = sample_mixture(600, w, mu, s2, rng)
    ab = np.cumprod(1 - linear_betas())
    steps = [0, 50, 150, 300, 600, 1000]
    eps = rng.standard_normal(x0.shape)
    fig, axs = plt.subplots(1, len(steps), figsize=(8.4, 1.75))
    for ax, k in zip(axs, steps):
        a = 1.0 if k == 0 else ab[k - 1]
        xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * eps
        ax.scatter(xt[:, 0], xt[:, 1], s=1.6, color=ACCENT, lw=0)
        square(ax, 3.4)
        ax.set_title(f"t = {k}" + ("" if k == 0 else f"\nᾱ = {a:.3f}" if a > 1e-3 else f"\nᾱ = {a:.0e}"), fontsize=8)
    save(fig, "diff-a-forward-2d",
         "<b>The forward process on a 2-D toy dataset</b> (a ring of eight tight Gaussian blobs, 600 points), "
         "noised in one jump per panel with x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε and the linear schedule. The same ε is "
         "used in every panel, so each point drifts along a straight line from its data position toward its "
         "noise sample. By t = 1,000 nothing of the ring is left: the cloud is a standard normal.",
         alt="Six scatter plots of a ring of points dissolving into Gaussian noise", width=760)


def toy_image(n=64):
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    img = 0.15 * np.ones((n, n))
    img += 0.35 * (xx > 0.0) * yy                                   # vertical gradient
    img[((xx - 0.33) ** 2 + (yy - 0.38) ** 2) < 0.04] = 1.0           # disc
    img[(np.abs(xx - 0.72) < 0.14) & (np.abs(yy - 0.66) < 0.14)] = 0.0  # square
    img[(np.abs(xx - 0.72) < 0.06) & (np.abs(yy - 0.66) < 0.06)] = 0.85
    stripes = (np.sin(2 * np.pi * 6 * xx) > 0) & (yy > 0.86)
    img[stripes] = 0.9
    return 2 * img - 1                                               # scale to [-1, 1]


def image_strip():
    rng = np.random.default_rng(1)
    x0 = toy_image()
    eps = rng.standard_normal(x0.shape)
    steps = [0, 25, 100, 250, 500, 750, 1000]
    rows = [("linear", np.cumprod(1 - linear_betas())), ("cosine", cosine_alpha_bar())]
    fig, axs = plt.subplots(2, len(steps), figsize=(8.4, 2.75))
    for r, (name, ab) in enumerate(rows):
        for c, k in enumerate(steps):
            a = 1.0 if k == 0 else ab[k - 1]
            xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * eps
            ax = axs[r, c]
            ax.imshow(np.clip(xt, -1, 1), cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t = {k}", fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    save(fig, "diff-a-image-strip",
         "<b>Noising an image</b> (a 64×64 test image drawn in numpy, pixels scaled to [−1, 1]) with the closed "
         "form x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε, one fixed ε for every panel; values outside [−1, 1] are clipped for "
         "display. Top: linear schedule; bottom: cosine schedule. With the linear schedule the picture is "
         "essentially gone by t = 500; the cosine schedule keeps visible structure longer, which is what "
         "Nichol and Dhariwal designed it for.",
         alt="Grid of a synthetic image becoming noise at increasing steps for two schedules", width=760)


def density_1d():
    """p_t(x) of a 1-D mixture under the VP SDE, with forward paths."""
    w = np.array([0.3, 0.7])
    mu = np.array([[-2.0], [1.5]])
    s2 = 0.25 ** 2
    ts = np.linspace(0, 1, 201)
    xs = np.linspace(-3.5, 3.5, 301)
    dens = np.zeros((len(xs), len(ts)))
    for j, t in enumerate(ts):
        m = vp_mean_coef(t)
        var = m**2 * s2 + 1 - m**2
        for k in range(2):
            dens[:, j] += w[k] * np.exp(-0.5 * (xs - m * mu[k, 0]) ** 2 / var) / np.sqrt(2 * np.pi * var)
    return w, mu, s2, ts, xs, dens


def forward_1d():
    rng = np.random.default_rng(4)
    w, mu, s2, ts, xs, dens = density_1d()
    x0, _ = sample_mixture(6, w, mu, s2, rng)
    n = 1000
    tt = np.linspace(0, 1, n + 1)
    paths = np.zeros((6, n + 1))
    x = x0[:, 0].copy()
    paths[:, 0] = x
    for i in range(n):                      # Euler–Maruyama, forward in time
        t, dt = tt[i], tt[i + 1] - tt[i]
        x = x - 0.5 * beta(t) * x * dt + np.sqrt(beta(t) * dt) * rng.standard_normal(x.shape)
        paths[:, i + 1] = x
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    shade(ax, dens, [0, 1, xs[0], xs[-1]])
    for p in paths:
        ax.plot(tt, p, color=ACCENT, lw=0.9)
    ax.set_xlabel("t  (forward time)")
    ax.set_ylabel("x")
    save(fig, "diff-a-forward-1d",
         "<b>The VP SDE moving a 1-D distribution to N(0, 1).</b> Grey shading: the exact density p_t(x) of a "
         "two-component mixture (weights 0.3 and 0.7) under dx = −½β(t)x dt + √β(t) dW with β(t) = 0.1 + 19.9t. "
         "Blue: six forward paths simulated with Euler–Maruyama. Each path is a Brownian motion with a pull "
         "toward 0; together they turn two bumps into one standard normal.",
         alt="Heat map of a two-bump density merging into one Gaussian over time, with noisy paths")


# ---------------------------------------------------------------- page 2
def score_field():
    w = np.array([0.5, 0.5])
    mu = np.array([[-1.5, 0.0], [1.5, 0.6]])
    s2 = 0.35 ** 2
    g = np.linspace(-3, 3, 17)
    X, Y = np.meshgrid(g, g)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    gd = np.linspace(-3, 3, 90)
    XD, YD = np.meshgrid(gd, gd)
    pd = np.stack([XD.ravel(), YD.ravel()], 1)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5))
    for ax, t, lab in [(axs[0], 1e-3, "t ≈ 0 (data)"), (axs[1], 0.25, "t = 0.25 (noised)")]:
        m = vp_mean_coef(t)
        var = m**2 * s2 + 1 - m**2
        dens = sum(w[k] * np.exp(-0.5 * ((pd - m * mu[k]) ** 2).sum(1) / var) for k in range(2)).reshape(XD.shape)
        ax.contour(XD, YD, dens, levels=6, colors=MUTED, linewidths=0.8)
        sc = mixture_score(pts, t, w, mu, s2)
        nrm = np.linalg.norm(sc, axis=1, keepdims=True) + 1e-9
        u = sc / nrm * np.minimum(nrm, 4) / 4                      # cap length
        ax.quiver(pts[:, 0], pts[:, 1], u[:, 0], u[:, 1], color=ACCENT, scale=14, width=0.005)
        square(ax, 3)
        ax.set_title(lab, fontsize=10)
    save(fig, "diff-a-score-field",
         "<b>The score is a vector field pointing uphill in log-density.</b> Arrows: the exact score "
         "∇ log p_t(x) of a two-component Gaussian mixture (lengths capped for display), over its density "
         "(grey contours). Left, the data distribution: arrows point into the nearest blob, and between the blobs they "
         "split. Right, after VP noising to t = 0.25: the density is broader and the field is smooth "
         "everywhere — which is why scores of noised data are easier to learn.",
         alt="Two quiver plots of score vectors over a two-blob density, sharp and smoothed")


def langevin_vs_reverse():
    """Song & Ermon's example: 1/5 N((-5,-5), I) + 4/5 N((5,5), I)."""
    rng = np.random.default_rng(7)
    w = np.array([0.2, 0.8])
    mu = np.array([[-5.0, -5.0], [5.0, 5.0]])
    s2 = 1.0
    n = 2000
    # exact samples
    xe, _ = sample_mixture(n, w, mu, s2, rng)
    # plain Langevin with the exact data score (t ~ 0)
    x = rng.uniform(-8, 8, (n, 2))
    step = 0.1
    for _ in range(1000):
        x = x + 0.5 * step * mixture_score(x, 1e-9, w, mu, s2) + np.sqrt(step) * rng.standard_normal(x.shape)
    xl = x
    # annealed Langevin (Song & Ermon Alg. 1), exact noised scores
    sig = np.geomspace(10, 0.1, 10)
    x = rng.uniform(-8, 8, (n, 2))
    eps = 2e-3

    def smoothed_score(x, sigma):
        var = s2 + sigma**2
        diff = x[:, None] - mu[None]
        logp = np.log(w)[None] - 0.5 * (diff**2).sum(-1) / var
        r = np.exp(logp - logp.max(1, keepdims=True))
        r /= r.sum(1, keepdims=True)
        return -(r[..., None] * diff).sum(1) / var

    for sgm in sig:
        a = eps * sgm**2 / sig[-1] ** 2
        for _ in range(100):
            x = x + 0.5 * a * smoothed_score(x, sgm) + np.sqrt(a) * rng.standard_normal(x.shape)
    xa = x
    # reverse VP SDE from N(0, I)
    xr, _ = reverse_sde(rng.standard_normal((n, 2)), w, mu, s2, rng=rng)
    frac = lambda z: float(np.mean(z.sum(1) < 0))
    for k, z in [("exact", xe), ("langevin", xl), ("annealed", xa), ("reverse", xr)]:
        NUMBERS[f"frac_{k}"] = frac(z)
    fig, axs = plt.subplots(1, 4, figsize=(8.4, 2.35))
    for ax, z, lab in [(axs[0], xe, "exact samples"), (axs[1], xl, "Langevin, exact score"),
                       (axs[2], xa, "annealed Langevin"), (axs[3], xr, "reverse-time SDE")]:
        ax.scatter(z[:500, 0], z[:500, 1], s=1.4, color=ACCENT, lw=0)
        square(ax, 9)
        ax.set_title(f"{lab}\nlower-left share {frac(z):.2f}", fontsize=8)
    save(fig, "diff-a-langevin",
         "<b>Why noise levels matter</b> (the example of Song and Ermon, 2019: weights 0.2 and 0.8, 2,000 samples "
         "each, 500 drawn, all scores exact). Plain Langevin dynamics on the data score, started uniformly in a box, "
         "finds both modes but not their weights: the score inside each blob does not depend on the weights, "
         "and nothing moves samples across the empty gap. Annealed Langevin (10 noise levels from 10 to 0.1) "
         "and the reverse-time VP SDE both start where the noised distributions overlap and recover the 0.2 / 0.8 split.",
         alt="Four scatter plots comparing exact samples, Langevin, annealed Langevin and reverse SDE", width=760)


def reverse_1d():
    rng = np.random.default_rng(5)
    w, mu, s2, ts, xs, dens = density_1d()
    n = 1000
    start = rng.standard_normal((6, 1))
    tgrid = np.linspace(1.0, 1e-3, n + 1)
    sde = np.zeros((6, n + 1))
    ode = np.zeros((6, n + 1))
    x, y = start.copy(), start.copy()
    sde[:, 0], ode[:, 0] = x[:, 0], y[:, 0]
    for i in range(n):
        t, dt = tgrid[i], tgrid[i] - tgrid[i + 1]
        b = beta(t)
        x = x - (-0.5 * b * x - b * mixture_score(x, t, w, mu, s2)) * dt + np.sqrt(b * dt) * rng.standard_normal(x.shape)
        y = y - (-0.5 * b * y - 0.5 * b * mixture_score(y, t, w, mu, s2)) * dt
        sde[:, i + 1], ode[:, i + 1] = x[:, 0], y[:, 0]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    shade(ax, dens, [0, 1, xs[0], xs[-1]])
    for p in sde:
        ax.plot(tgrid, p, color=ACCENT, lw=0.9)
    for p in ode:
        ax.plot(tgrid, p, color=HI, lw=1.6, ls="--")
    ax.set_xlim(1, 0)                         # time runs right-to-left
    ax.set_xlabel("t  (runs backwards: 1 → 0)")
    ax.set_ylabel("x")
    save(fig, "diff-a-reverse-1d",
         "<b>Running the diffusion backwards.</b> Same density as the forward figure, with the time axis flipped "
         "so generation reads left to right. Blue: six reverse-time SDE paths (fresh noise at every step). "
         "Red dashed: six probability-flow ODE paths from the same starting points (no noise). Both end in "
         "one of the two bumps; the ODE paths are smooth and deterministic, the SDE paths jitter but follow "
         "the same density.",
         alt="Heat map with noisy reverse SDE paths and smooth ODE paths converging into two bumps")


def reverse_2d():
    rng = np.random.default_rng(2)
    w, mu, s2 = ring_mixture()
    n = 2000
    xT = rng.standard_normal((n, 2))
    keep = (400, 700, 850, 950)
    xs, snaps = reverse_sde(xT.copy(), w, mu, s2, rng=rng, keep=keep)
    xo, _ = prob_flow_ode(xT.copy(), w, mu, s2)
    ts = np.linspace(1.0, 1e-3, 1001)
    # which blob did each sample land in?
    d = ((xs[:, None] - mu[None]) ** 2).sum(-1)
    counts = np.bincount(d.argmin(1), minlength=8)
    NUMBERS["ring_counts_sde"] = counts.tolist()
    NUMBERS["ring_mean_dist_sde"] = float(np.sqrt(d.min(1)).mean())
    do = ((xo[:, None] - mu[None]) ** 2).sum(-1)
    NUMBERS["ring_counts_ode"] = np.bincount(do.argmin(1), minlength=8).tolist()
    NUMBERS["ring_mean_dist_ode"] = float(np.sqrt(do.min(1)).mean())
    panels = [("t = 1 (noise)", xT)] + [(f"t = {ts[k]:.2f}", snaps[k]) for k in keep] + \
             [("t = 0.001, SDE", xs), ("ODE, same start", xo)]
    fig, axs = plt.subplots(1, len(panels), figsize=(8.6, 1.65))
    for ax, (lab, z) in zip(axs, panels):
        ax.scatter(z[:500, 0], z[:500, 1], s=1.2, color=HI if "ODE" in lab else ACCENT, lw=0)
        square(ax, 3.4)
        ax.set_title(lab, fontsize=7.5)
    save(fig, "diff-a-reverse-2d",
         "<b>Generating the ring from pure noise</b> with the exact score: 2,000 standard-normal points (500 drawn) pushed "
         "through 1,000 Euler–Maruyama steps of the reverse-time VP SDE (blue), and the same starting points "
         "through the probability-flow ODE (red, last panel). Structure appears late — most of it after t ≈ 0.3, "
         f"when the noise is small. The SDE puts between {min(NUMBERS['ring_counts_sde'])} and "
         f"{max(NUMBERS['ring_counts_sde'])} points in each of the eight blobs (expected 250).",
         alt="Seven scatter plots of Gaussian noise turning into a ring of eight blobs", width=760)


if __name__ == "__main__":
    brownian_paths()
    schedules()
    forward_2d()
    image_strip()
    forward_1d()
    score_field()
    langevin_vs_reverse()
    reverse_1d()
    reverse_2d()
    for nm in ["diff-a-brownian", "diff-a-schedules", "diff-a-forward-2d", "diff-a-image-strip",
               "diff-a-forward-1d", "diff-a-score-field", "diff-a-langevin", "diff-a-reverse-1d",
               "diff-a-reverse-2d"]:
        compact(nm)
    for k, v in NUMBERS.items():
        print(k, v)
