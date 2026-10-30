#!/usr/bin/env python3
"""Figures for 8-diffusion-models/ddpm and fast-sampling-ddim-and-ode-solvers.

Run: python3 tools/figs_diff_b.py   (about two minutes: trains the toy model)
Writes tools/figs/diff-b-*.html and tools/figs/diff-b-results.json, then
python3 tools/assemble.py content/8-diffusion-models/ddpm.md content/8-diffusion-models/fast-sampling-ddim-and-ode-solvers.md
All numbers come from tools/diffusion_toy.py, run here.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diffusion_toy as D  # noqa: E402
from mplfig import ACCENT, ACCENT2, HI, INK, MUTED, FIGS, plt, save  # noqa: E402

LIM = (-2.6, 2.6)


PLOT_MAX = 1200  # points drawn per panel (page weight)


def scatter(ax, pts, color, s=2, alpha=0.5):
    pts = pts[:PLOT_MAX]  # iid samples: a prefix is random
    ax.scatter(pts[:, 0], pts[:, 1], s=s, color=color, alpha=alpha, linewidths=0)
    ax.set_xlim(*LIM)
    ax.set_ylim(*LIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(MUTED)


def schedules():
    t1000 = np.arange(1, 1001)
    lin1000 = np.cumprod(1 - np.linspace(1e-4, 0.02, 1000))
    s = 0.008
    f = lambda t: np.cos((t / 1000 + s) / (1 + s) * np.pi / 2) ** 2
    cos1000 = f(t1000) / f(0)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.plot(t1000 / 1000, lin1000, color=ACCENT, label="linear, T = 1000 (Ho et al.)")
    ax.plot(t1000 / 1000, cos1000, color=HI, label="cosine, T = 1000 (Nichol & Dhariwal)")
    ax.plot(np.arange(1, D.T + 1) / D.T, D.ALPHA_BAR, color=MUTED, ls="--",
            label="scaled linear, T = 100 (toy below)")
    ax.set_xlabel("t / T")
    ax.set_ylabel("alpha-bar (signal fraction)")
    ax.grid(True)
    ax.legend(fontsize=9)
    save(fig, "diff-b-schedule",
         "<b>How much of x<sub>0</sub> survives at step t.</b> ᾱ<sub>t</sub> is the product of (1 − β<sub>s</sub>) up "
         "to t: x<sub>t</sub> = √ᾱ<sub>t</sub> x<sub>0</sub> + √(1 − ᾱ<sub>t</sub>) ε. DDPM's linear schedule "
         "(β from 10<sup>−4</sup> to 0.02 over 1000 steps) leaves almost no signal for the last quarter of the "
         "chain; the cosine schedule spends the steps more evenly. The toy model uses 100 steps with the linear "
         "β range scaled by 10 so that the total noise matches (its ᾱ<sub>100</sub> ≈ 2·10<sup>−5</sup>).",
         alt="alpha-bar versus normalized step for linear, cosine and scaled-linear schedules")


def main():
    params, losses, data = D.train(steps=20000, seed=0)
    rng = np.random.default_rng(1)
    held = D.two_moons(4000, np.random.default_rng(123))
    held2 = D.two_moons(4000, np.random.default_rng(456))
    res = {}

    # loss
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    k = 200
    ma = np.convolve(losses, np.ones(k) / k, mode="valid")
    ax.plot(np.arange(len(losses)), losses, color=MUTED, lw=0.6, alpha=0.6)
    ax.plot(np.arange(k - 1, len(losses)), ma, color=ACCENT, lw=2)
    ax.set_yscale("log")
    ax.set_xticks([0, 5000, 10000, 15000, 20000], ["0", "5k", "10k", "15k", "20k"])
    ax.set_xlabel("training step (batch 256)")
    ax.set_ylabel("L_simple")
    ax.grid(True)
    res["loss_first200"] = float(losses[:200].mean())
    res["loss_last1000"] = float(losses[-1000:].mean())
    save(fig, "diff-b-loss",
         f"<b>Training the toy noise predictor</b> (Algorithm 1, Adam, 20,000 steps of 256 points; grey = per-batch "
         f"loss, blue = 200-step average). L<sub>simple</sub> falls from about {res['loss_first200']:.2f} to "
         f"{res['loss_last1000']:.2f}, not to 0: at large t the noise cannot be predicted from x<sub>t</sub>, so a "
         "perfect model still has a large loss there. The loss is a poor guide to sample quality.",
         alt="Training loss versus step on a log scale")

    # samples vs data
    x, snaps = D.sample_ddpm(params, 4000, rng, keep=(60, 30, 15, 5, 0))
    res["ed_ddpm_full"] = float(D.energy_distance(x, held, np.random.default_rng(7)))
    floors = [D.energy_distance(D.two_moons(4000, np.random.default_rng(1000 + i)), held, np.random.default_rng(7)) for i in range(5)]
    res["ed_floor_mean"], res["ed_floor_std"] = float(np.mean(floors)), float(np.std(floors))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.2))
    scatter(a1, held, MUTED)
    a1.set_title("held-out data", fontsize=11)
    scatter(a2, x, ACCENT)
    a2.set_title("DDPM samples (100 steps)", fontsize=11)
    save(fig, "diff-b-samples",
         "<b>4,000 samples from the trained toy model</b> (Algorithm 2 with σ<sub>t</sub>² = β<sub>t</sub>, all 100 "
         "steps) next to 4,000 held-out points of the two-moons data (1,200 of each drawn). Energy distance to the held-out set "
         f"(unbiased estimate, 2,000-point subsamples): {res['ed_ddpm_full']:.4f}. Fresh real samples score "
         f"about 0 ± {res['ed_floor_std']:.4f}, so the model is within the noise of the metric.",
         alt="Held-out two-moons data next to samples from the trained DDPM")

    # trajectory snapshots
    fig, axes = plt.subplots(1, 6, figsize=(9.6, 1.9))
    for ax, t in zip(axes, [100, 60, 30, 15, 5, 0]):
        scatter(ax, snaps[t], ACCENT if t == 0 else INK, s=1.5, alpha=0.4)
        ax.set_title(f"t = {t}", fontsize=10)
    save(fig, "diff-b-trajectory",
         "<b>The reverse chain, from x<sub>100</sub> ~ N(0, I) to x<sub>0</sub></b> (same run as above). Down to "
         "about t = 15 the batch still looks like a Gaussian blob, only slightly stretched; the moons form in the "
         "last 10–15 steps, as the noise level √(1 − ᾱ<sub>t</sub>) falls from 0.45 (t = 15) to 0.16 (t = 5) — "
         "below the width of the gap between the moons. "
         "Each panel is the whole batch at one step.",
         alt="Six scatter plots of the samples at t = 100, 60, 30, 15, 5 and 0")

    # epsilon / score field
    g = np.linspace(-2.4, 2.4, 17)
    gx, gy = np.meshgrid(g, g)
    pts = np.c_[gx.ravel(), gy.ravel()]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    for ax, t in zip(axes, [50, 8]):
        eps = D.forward(params, pts, np.full(len(pts), t))
        score = -eps / np.sqrt(1 - D.ALPHA_BAR[t - 1])
        nrm = np.linalg.norm(score, axis=1, keepdims=True) + 1e-9
        u = score / nrm
        scatter(ax, held[:1500], MUTED, s=1.5, alpha=0.35)
        ax.quiver(pts[:, 0], pts[:, 1], u[:, 0], u[:, 1], color=ACCENT, angles="xy", scale=22, width=0.006)
        ax.set_title(f"t = {t}", fontsize=11)
    save(fig, "diff-b-score",
         "<b>What the network learned: −ε<sub>θ</sub>(x, t)/√(1 − ᾱ<sub>t</sub>), the estimated score "
         "∇ log p<sub>t</sub>(x)</b> (arrows normalized to unit length). At t = 50 the noisy distribution is one blob "
         "and every arrow points to its center; at t = 8 the arrows point onto the two moons. Each reverse step "
         "moves a point a little along these arrows and adds fresh noise.",
         alt="Arrow fields of the learned score at t = 50 and t = 8 over the data")

    # ---------------- fast sampling
    Ss = [5, 10, 20, 50, 100]
    xT = np.random.default_rng(2024).standard_normal((4000, 2))
    rows = []
    for S in Ss:
        r = {"S": S}
        r["ddim"] = float(D.energy_distance(D.sample_ddim(params, 4000, np.random.default_rng(S), S, eta=0.0, x_T=xT), held, np.random.default_rng(7)))
        r["ddpm"] = float(D.energy_distance(D.sample_ddim(params, 4000, np.random.default_rng(S), S, eta=1.0, x_T=xT), held, np.random.default_rng(7)))
        if S >= 3:
            r["heun_nfe"] = D.nfe_heun(S)
            r["heun"] = float(D.energy_distance(D.sample_heun(params, 4000, np.random.default_rng(S), S, x_T=xT), held, np.random.default_rng(7)))
        rows.append(r)
    # Heun at matched NFE: S steps -> 2S-1 evaluations
    heun_rows = []
    for S in [3, 4, 6, 11, 26, 51]:
        heun_rows.append({"S": S, "nfe": D.nfe_heun(S),
                          "ed": float(D.energy_distance(D.sample_heun(params, 4000, np.random.default_rng(S), S, x_T=xT), held, np.random.default_rng(7)))})
    res["fast"] = rows
    res["heun"] = heun_rows

    fig, axes = plt.subplots(2, 4, figsize=(8.8, 4.6))
    for j, S in enumerate([5, 10, 20, 100]):
        for i, (eta, name) in enumerate([(1.0, "η = 1 (DDPM)"), (0.0, "η = 0 (DDIM)")]):
            xs = D.sample_ddim(params, 4000, np.random.default_rng(S), S, eta=eta, x_T=xT)
            scatter(axes[i, j], xs, ACCENT if eta == 0 else HI, s=1.5, alpha=0.4)
            if i == 0:
                axes[i, j].set_title(f"{S} steps", fontsize=11)
            if j == 0:
                axes[i, j].set_ylabel(name, fontsize=10)
    save(fig, "diff-b-steps-grid",
         "<b>Same trained model, fewer steps.</b> Top: the stochastic DDPM update (η = 1) on an evenly spaced "
         "subsequence of the 100 steps. Bottom: deterministic DDIM (η = 0) from the same starting noise. With 5–10 "
         "steps the stochastic sampler leaves visible noise around the moons; DDIM stays sharp. No retraining is "
         "involved: only the update rule and the step list change.",
         alt="Two rows of scatter plots comparing eta=1 and eta=0 sampling at 5, 10, 20 and 100 steps")

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    band = 1e-3
    def draw(xs, ys, color, marker, label):
        ys = np.asarray(ys)
        shown = np.maximum(ys, band * 0.8)
        ax.plot(xs, shown, color=color, lw=1.5, label=label)
        for xv, yv, sv in zip(xs, ys, shown):
            ax.scatter([xv], [sv], s=22, marker=marker, color=color if yv > band else "none",
                       edgecolors=color, zorder=3)
    draw([r["S"] for r in rows], [r["ddpm"] for r in rows], HI, "o", "η = 1 (DDPM update)")
    draw([r["S"] for r in rows], [r["ddim"] for r in rows], ACCENT, "o", "η = 0 (DDIM)")
    draw([h["nfe"] for h in heun_rows], [h["ed"] for h in heun_rows], ACCENT2, "s", "Heun (2nd order)")
    ax.axhspan(1e-4, band, color=MUTED, alpha=0.18, lw=0)
    ax.text(5.3, 6.2e-4, "within metric noise", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 100)
    ax.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("network evaluations (NFE)")
    ax.set_ylabel("energy distance to data")
    ax.grid(True)
    ax.legend(fontsize=9, loc="upper right")
    save(fig, "diff-b-quality",
         "<b>Sample quality against cost</b> for the toy model (energy distance to held-out data, lower is better; "
         "hollow markers sit in the grey band where the metric cannot tell samples from real data). At 10–20 "
         "evaluations DDIM is about twice as close to the data as the stochastic update. Heun's method costs two "
         "evaluations per step: with very few steps (5–11 evaluations) its large steps go badly wrong, but at "
         "about 20 evaluations it is level with or slightly ahead of DDIM.",
         alt="Energy distance versus number of network evaluations for three samplers")

    # deterministic vs stochastic paths
    rng_p = np.random.default_rng(99)
    start = rng_p.standard_normal((40, 2))
    _, path_ode = D.sample_ddim(params, 40, rng_p, 100, eta=0.0, x_T=start, return_path=True)
    _, path_sde = D.sample_ddim(params, 40, np.random.default_rng(5), 100, eta=1.0, x_T=start, return_path=True)
    end10 = D.sample_ddim(params, 40, rng_p, 10, eta=0.0, x_T=start)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.3))
    for ax, path, title, col, k in [(a1, path_sde, "η = 1: stochastic paths", HI, 8), (a2, path_ode, "η = 0: ODE paths", ACCENT, 40)]:
        P = np.stack(path)
        scatter(ax, held[:1500], MUTED, s=1.2, alpha=0.3)
        for i in range(k):
            ax.plot(P[:, i, 0], P[:, i, 1], color=col, lw=0.6, alpha=0.7)
        ax.scatter(P[-1, :k, 0], P[-1, :k, 1], s=10, color=col, zorder=3)
        ax.set_title(title, fontsize=11)
    a2.scatter(end10[:, 0], end10[:, 1], s=14, facecolors="none", edgecolors=INK, lw=0.8, zorder=4)
    res["ode_end_gap_10_vs_100"] = float(np.mean(np.linalg.norm(end10 - path_ode[-1], axis=1)))
    save(fig, "diff-b-paths",
         "<b>Starting points x<sub>T</sub> followed through 100 reverse steps.</b> Left (8 of them): the stochastic "
         "update adds fresh noise at every step, which makes the paths jagged random walks, and the same x<sub>T</sub> would land "
         "elsewhere on a rerun. Right (40): DDIM follows "
         "smooth deterministic curves (an ODE); open circles show where the same x<sub>T</sub> lands with only 10 "
         f"DDIM steps, on average {res['ode_end_gap_10_vs_100']:.2f} away — the latent fixes the sample.",
         alt="Trajectories of forty points under stochastic and deterministic sampling")

    with open(os.path.join(FIGS, "diff-b-results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    schedules()
    main()
