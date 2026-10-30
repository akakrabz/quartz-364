#!/usr/bin/env python3
"""Figures for the guidance / latent-diffusion, flow-matching and modern-generators
pages of unit 8. Run: python3 tools/figs_diff_c.py   (about two minutes)

Writes tools/figs/diff-c-*.html; then
    python3 tools/assemble.py content/8-diffusion-models/guidance-and-latent-diffusion.md \
        content/8-diffusion-models/flow-matching-and-rectified-flow.md content/8-diffusion-models/modern-image-generators.md
Everything is computed here with numpy: the guidance toy uses exact Gaussian-mixture
scores (no network); the flow toy trains the tiny MLP in tools/flow_toy.py.
Printed numbers are the ones quoted on the pages.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figlib import SVG  # noqa: E402
from mplfig import ACCENT, ACCENT2, HI, INK, MUTED, FIGS, plt, save  # noqa: E402
import flow_toy as ft  # noqa: E402

MONO = "'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace"


def write(name, html):
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, name + ".html"), "w") as f:
        f.write(html + "\n")


def arrow(s, x1, y1, x2, y2, color="currentColor", width=1.5):
    """Line with a filled head (no marker, colour always matches)."""
    s.line(x1, y1, x2, y2, color=color, width=width)
    v = np.array([x2 - x1, y2 - y1], float)
    v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]])
    tip = np.array([x2, y2])
    a, b = tip - 7 * v + 3.5 * n, tip - 7 * v - 3.5 * n
    s.add(f'<path d="M{tip[0]:.1f},{tip[1]:.1f} L{a[0]:.1f},{a[1]:.1f} L{b[0]:.1f},{b[1]:.1f} Z" fill="{color}"/>')


def box(s, x, y, w, h, lines, color="currentColor", fill="var(--paper)"):
    s.rect(x, y, w, h, rx=8, stroke=color, fill=fill, width=1.6)
    n = len(lines)
    for i, t in enumerate(lines):
        yy = y + h / 2 + (i - (n - 1) / 2) * 16
        s.text(x + w / 2, yy, t, size=12 if i else 13, color=color if i == 0 else "var(--muted)",
               weight="600" if i == 0 else None)


# ================================================================ guidance toy (exact scores)
ANG = np.arange(6) * np.pi / 3
CENTERS = 2.0 * np.c_[np.cos(ANG), np.sin(ANG)]
CLASS = np.array([0, 1, 0, 1, 0, 1])          # A at even angles, B at odd
S0 = 0.45                                       # component std
T_G = 250
BETAS_G = np.linspace(4e-4, 0.08, T_G)
ALPHA_BAR_G = np.cumprod(1 - BETAS_G)


def eps_exact(x, abar, comps):
    """Exact noise prediction -sqrt(1-abar) * score of a Gaussian mixture."""
    mu = np.sqrt(abar) * CENTERS[comps]                  # (K, 2)
    var = abar * S0 ** 2 + (1 - abar)
    d2 = ((x[:, None, :] - mu[None]) ** 2).sum(-1)       # (N, K)
    logw = -d2 / (2 * var)
    r = np.exp(logw - logw.max(1, keepdims=True))
    r /= r.sum(1, keepdims=True)                         # responsibilities
    score = -(x - r @ mu) / var
    return -np.sqrt(1 - abar) * score


def sample_cfg(w, n, rng, cls=0):
    """DDPM ancestral sampling with classifier-free guidance weight w."""
    comps_c = np.where(CLASS == cls)[0]
    comps_u = np.arange(6)
    x = rng.standard_normal((n, 2))
    for t in reversed(range(T_G)):
        ab, b = ALPHA_BAR_G[t], BETAS_G[t]
        e_c = eps_exact(x, ab, comps_c)
        e_u = eps_exact(x, ab, comps_u)
        e = (1 + w) * e_c - w * e_u                     # Ho & Salimans eq. 6
        x = (x - b / np.sqrt(1 - ab) * e) / np.sqrt(1 - b)
        if t > 0:
            x = x + np.sqrt(b) * rng.standard_normal(x.shape)
    return x


def p_class_a(x):
    """Bayes classifier p(A | x) on clean data."""
    d2 = ((x[:, None, :] - CENTERS[None]) ** 2).sum(-1)
    lik = np.exp(-d2 / (2 * S0 ** 2))
    return lik[:, CLASS == 0].sum(1) / lik.sum(1)


def guidance_toy():
    ws = [0, 1, 3, 7]
    rng = np.random.default_rng(0)
    data_rng = np.random.default_rng(1)
    comp = data_rng.integers(0, 6, 1200)
    data = CENTERS[comp] + S0 * data_rng.standard_normal((1200, 2))
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.7), sharex=True, sharey=True)
    for ax, w in zip(axes, ws):
        xs = sample_cfg(w, 1500, rng)
        pa = p_class_a(xs)
        ang = np.arctan2(xs[:, 1], xs[:, 0])
        d = np.angle(np.exp(1j * (ang[:, None] - ANG[CLASS == 0][None])))
        dang = d[np.arange(len(d)), np.argmin(np.abs(d), 1)]     # angle to nearest A blob
        r = np.linalg.norm(xs, axis=1)
        area = np.sqrt(np.linalg.det(np.cov(np.c_[r, 2.0 * dang].T)))  # blob size (polar)
        rows.append((w, np.mean(pa > 0.5), np.mean(pa), np.degrees(dang.std()), r.mean(), area))
        ax.scatter(data[CLASS[comp] == 1, 0][:1200], data[CLASS[comp] == 1, 1][:1200], s=2, color=MUTED, alpha=0.5, lw=0)
        ax.scatter(xs[:1200, 0], xs[:1200, 1], s=2, color=ACCENT, alpha=0.6, lw=0)
        ax.set_title(f"w = {w}", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(-4.2, 4.2)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    save(fig, "diff-c-cfg-toy",
         "<b>Classifier-free guidance on a 2-D toy.</b> Six Gaussian blobs on a ring; class A is the three blobs "
         "at 0°, 120°, 240°, class B (grey) the other three. Blue: 1,500 samples of class A drawn with DDPM "
         "(250 steps) using the guided noise prediction (1 + w)·ε(x, A) − w·ε(x), with exact scores of the noisy "
         "mixtures instead of a network. w = 0 is the plain conditional model. As w grows the samples move away "
         "from class B (outward, mean radius 2.07 → 2.38) and each blob shrinks (angular spread 12.8° → 8.3°): "
         "fidelity to the class goes up, diversity goes down.",
         alt="Four scatter plots of guided samples for guidance weights 0, 1, 3 and 7")
    print("guidance toy: w, frac p(A|x)>0.5, mean p(A|x), angular std (deg), mean radius, blob size")
    for r in rows:
        print("  w=%d  %.3f  %.3f  %.1f  %.2f  %.3f" % r)
    return rows


# ================================================================ latent diffusion diagram
def ldm_diagram():
    W, H = 720, 250
    s = SVG(W, H, font_size=13)
    box(s, 10, 95, 92, 56, ["image x", "512×512×3"])
    arrow(s, 104, 123, 134, 123)
    box(s, 136, 95, 74, 56, ["encoder", "ℰ"], color="var(--accent2)")
    arrow(s, 212, 123, 242, 123)
    box(s, 244, 95, 92, 56, ["latent z", "64×64×4"], color="var(--accent)")
    arrow(s, 338, 123, 368, 123)
    s.rect(370, 60, 170, 126, rx=10, stroke="var(--accent)", fill="var(--paper)", width=1.8)
    s.text(455, 80, "denoiser ε_θ", size=13, color="var(--accent)", weight="600")
    s.text(455, 99, "ε_θ(z_t, t, τ_θ(y))", size=11, color="var(--accent)", font=MONO)
    s.text(455, 118, "U-Net + cross-attention", size=12, color="var(--muted)")
    s.text(455, 138, "noise z_T → … → z_0", size=12, color="var(--muted)")
    s.text(455, 158, "T denoising steps,", size=12, color="var(--muted)")
    s.text(455, 174, "all in latent space", size=11, color="var(--muted)")
    arrow(s, 542, 123, 572, 123)
    box(s, 574, 95, 64, 56, ["decoder", "𝒟"], color="var(--accent2)")
    arrow(s, 640, 123, 662, 123)
    s.text(690, 115, "image", size=12)
    s.text(690, 131, "x̃", size=13)
    s.rect(370, 200, 170, 40, rx=8, stroke="var(--hi)", fill="var(--paper)", width=1.6)
    s.text(455, 220, "text encoder τ_θ", size=13, color="var(--hi)", weight="600")
    s.text(285, 220, "prompt y", size=12, color="var(--hi)")
    arrow(s, 318, 220, 366, 220, color="var(--hi)")
    arrow(s, 455, 198, 455, 190, color="var(--hi)")
    s.text(462, 194, "keys, values", size=10, color="var(--hi)", anchor="start")
    s.text(W / 2, 30, "training: z = ℰ(x) is noised and denoised; ℰ and 𝒟 are trained first, then frozen",
           size=11, color="var(--muted)")
    write("diff-c-ldm", s.figure(
        "<b>Latent diffusion</b> (Rombach et al., 2022), with the Stable Diffusion v1 sizes from its model card: "
        "an autoencoder with downsampling factor f = 8 maps a 512×512×3 image to a 64×64×4 latent, 48 times fewer "
        "numbers. The diffusion model (a U-Net) runs entirely on latents; the prompt enters every attention block "
        "as the keys and values of cross-attention. One decoder pass turns the final latent into pixels.",
        alt="Pipeline: image, encoder, latent, denoiser with cross-attention from a text encoder, decoder, image"))


# ================================================================ flow-matching toy
def flow_toy():
    rng = np.random.default_rng(0)
    p1, l1 = ft.train(ft.independent_pairs(), rng, steps=6000)
    test = ft.two_moons(4000, np.random.default_rng(7))
    z0 = np.random.default_rng(3).standard_normal((4000, 2))
    er = np.random.default_rng(11)
    steps = [1, 2, 5, 20, 100]
    sw1 = [ft.sliced_w1(ft.euler(p1, z0, n), test, er) for n in steps]
    s1 = ft.straightness(p1, z0)
    zz0 = np.random.default_rng(5).standard_normal((20000, 2))
    zz1 = ft.euler(p1, zz0, 100)
    p2, l2 = ft.train(ft.fixed_pairs(zz0, zz1), rng, steps=6000, params=[p.copy() for p in p1])
    sw2 = [ft.sliced_w1(ft.euler(p2, z0, n), test, er) for n in steps]
    s2 = ft.straightness(p2, z0)
    base = ft.sliced_w1(ft.two_moons(4000, np.random.default_rng(9)), test, er)
    noise = ft.sliced_w1(z0, test, er)
    print("flow toy: final loss 1-RF %.3f, 2-RF %.5f" % (l1[-200:].mean(), l2[-200:].mean()))
    print("  steps", steps)
    print("  SW1 1-RF", [round(v, 3) for v in sw1])
    print("  SW1 2-RF", [round(v, 3) for v in sw2])
    print("  straightness 1-RF %.3f  2-RF %.5f   data-vs-data %.3f   noise-vs-data %.3f" % (s1, s2, base, noise))

    # samples grid
    fig, axes = plt.subplots(2, 4, figsize=(9.4, 4.9), sharex=True, sharey=True)
    for row, (p, name) in enumerate([(p1, "1-rectified"), (p2, "2-rectified (reflow)")]):
        for col, n in enumerate([1, 2, 5, 20]):
            ax = axes[row, col]
            z = ft.euler(p, z0[:1500], n)
            ax.scatter(test[:500, 0], test[:500, 1], s=2, color=MUTED, alpha=0.35, lw=0)
            ax.scatter(z[:800, 0], z[:800, 1], s=2, color=ACCENT if row == 0 else HI, alpha=0.7, lw=0)
            ax.set_xlim(-2.6, 2.6)
            ax.set_ylim(-2.4, 2.4)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if row == 0:
                ax.set_title(f"{n} Euler step" + ("s" if n > 1 else ""), fontsize=11)
            if col == 0:
                ax.set_ylabel(name, fontsize=10)
    save(fig, "diff-c-rf-samples",
         "<b>Few-step sampling of a learned flow.</b> Top: the 1-rectified flow (trained on independent "
         "noise–data pairs) integrated with 1, 2, 5 and 20 Euler steps; grey is real data. With one step every "
         "sample lands near the data mean, because at t = 0 the learned velocity is roughly the average direction "
         "to the data. Bottom: after one reflow (retraining on the flow's own noise → sample pairs) the paths are "
         "nearly straight and a single Euler step already gives the two moons.",
         alt="Grid of samples from a 1-rectified and a 2-rectified flow with 1, 2, 5, 20 Euler steps")

    # trajectories
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.3), sharex=True, sharey=True)
    zt = np.random.default_rng(21).standard_normal((60, 2))
    paths = [ft.euler(p, zt, 100, keep_path=True)[1] for p in (p1, p2)]
    allp = np.concatenate([q.reshape(-1, 2) for q in paths] + [test[:1500]])
    lo, hi = allp.min(0) - 0.15, allp.max(0) + 0.15          # fit limits to data
    for ax, path, name, c in [(axes[0], paths[0], "1-rectified: curved paths", ACCENT),
                              (axes[1], paths[1], "after reflow: straight", HI)]:
        ax.scatter(test[:1500, 0], test[:1500, 1], s=2, color=MUTED, alpha=0.3, lw=0)
        for i in range(path.shape[1]):
            ax.plot(path[:, i, 0], path[:, i, 1], color=c, lw=0.9, alpha=0.8)
        ax.scatter(path[0, :, 0], path[0, :, 1], s=8, color=INK, lw=0, zorder=3)
        ax.scatter(path[-1, :, 0], path[-1, :, 1], s=8, color=c, lw=0, zorder=3)
        ax.set_title(name, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    save(fig, "diff-c-rf-paths",
         "<b>ODE trajectories from noise to data</b>, integrated with 100 Euler steps from 60 noise samples (small dots in the text colour). Left: the "
         "1-rectified flow; paths bend because the marginal velocity averages over all straight noise–data lines "
         "passing through a point. Right: after one reflow the coupling is (nearly) deterministic and the paths "
         "are (nearly) straight. Measured straightness ∫E‖(Z₁ − Z₀) − v‖² dt drops from %.2f to %.5f." % (s1, s2),
         alt="Trajectories of a learned flow before and after reflow")

    # error vs steps
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.loglog(steps, sw1, marker="o", ms=4, color=ACCENT, label="1-rectified flow")
    ax.loglog(steps, sw2, marker="o", ms=4, color=HI, label="2-rectified (one reflow)")
    ax.axhline(base, color=MUTED, lw=1, ls="--")
    ax.text(1.0, base * 0.72, "floor: two independent data samples", fontsize=9, color=MUTED)
    ax.set_ylim(base * 0.55, 1.2)
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0], ["0.01", "0.03", "0.1", "0.3", "1"])
    ax.set_xticks(steps, [str(s) for s in steps])
    ax.minorticks_off()
    ax.set_xlabel("Euler steps")
    ax.set_ylabel("sliced Wasserstein-1 to data")
    ax.legend(fontsize=9)
    ax.grid(True, which="major")
    save(fig, "diff-c-rf-w1",
         "<b>Sample error against the number of steps</b> (sliced Wasserstein-1 distance between 4,000 samples "
         "and 4,000 held-out data points; the dashed line is the distance between two independent data samples, "
         "i.e. the noise floor). The reflowed model reaches the floor with one or two steps; the 1-rectified flow "
         "needs about 20.",
         alt="Log-log plot of sample error versus number of Euler steps for two flows")
    return dict(sw1=sw1, sw2=sw2, s1=s1, s2=s2, base=base, noise=noise, steps=steps)


def velocity_score_check():
    """u(z,t) = (E[eps|z] - z)/(1-t) = (-t*score - z)/(1-t) for z_t=(1-t)x+t*eps (SD3 convention)."""
    s2 = 0.7 ** 2                                     # data x ~ N(0, s2)
    rng = np.random.default_rng(0)
    for t in [0.2, 0.5, 0.8]:
        x = rng.normal(0, np.sqrt(s2), 2_000_000)
        e = rng.standard_normal(x.size)
        z = (1 - t) * x + t * e
        var = (1 - t) ** 2 * s2 + t ** 2
        z0 = 0.6
        m = np.abs(z - z0) < 0.01
        mc = np.mean((e - x)[m])                       # Monte Carlo E[eps - x | z]
        formula = (-t * (-z0 / var) - z0) / (1 - t)
        assert abs(mc - formula) < 0.03, (t, mc, formula)
    print("velocity = affine function of score: checked by Monte Carlo")


def lognormal():
    t = np.linspace(1e-4, 1 - 1e-4, 800)

    def dens(m, s):
        lg = np.log(t / (1 - t))
        return np.exp(-(lg - m) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi) * t * (1 - t))
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.plot(t, np.ones_like(t), color=MUTED, lw=1.5, ls="--", label="uniform")
    ax.plot(t, dens(0.0, 1.0), color=ACCENT, label="logit-normal (m=0, s=1)")
    ax.plot(t, dens(0.5, 0.6), color=HI, label="logit-normal (m=0.5, s=0.6)")
    ax.plot(t, dens(-0.5, 1.0), color=ACCENT2, label="logit-normal (m=−0.5, s=1)")
    ax.set_xlabel("t   (SD3 convention: 0 = data, 1 = noise)")
    ax.set_ylabel("density")
    ax.set_ylim(0, 3.0)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(True)
    save(fig, "diff-c-lognorm",
         "<b>Timestep sampling densities</b> from Esser et al. (2024), eq. 19: t = sigmoid(u) with u drawn from "
         "a normal distribution N(m, s²). Compared with uniform t, the logit-normal puts more training weight on "
         "intermediate noise levels and almost none at the two ends; a positive m shifts it towards noise.",
         alt="Densities of uniform and logit-normal timestep distributions")


def shift_curve():
    tn = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for a, c in [(1.0, MUTED), (1.5, ACCENT2), (3.0, ACCENT)]:
        ax.plot(tn, a * tn / (1 + (a - 1) * tn), color=c, label=f"shift α = {a:g}")
    ax.set_xlabel("timestep t_n at base resolution")
    ax.set_ylabel("shifted timestep t_m")
    ax.legend(fontsize=9)
    ax.grid(True)
    ax.set_aspect("equal")
    save(fig, "diff-c-shift",
         "<b>Resolution-dependent timestep shift</b> (Esser et al., 2024, eq. 23) with α = √(m/n): at a higher "
         "resolution m the same timestep is mapped closer to the noise end. SD3 uses α = 3.0 at 1024×1024.",
         alt="Curves of the timestep shift function for alpha 1, 1.5 and 3")


# ================================================================ modern pipeline diagram
def pipeline_diagram():
    W, H = 740, 300
    s = SVG(W, H, font_size=13)
    box(s, 10, 30, 120, 52, ["prompt", "text"], color="var(--hi)")
    arrow(s, 132, 56, 162, 56, color="var(--hi)")
    box(s, 164, 30, 150, 52, ["text encoder(s)", "CLIP / T5 / VLM"], color="var(--hi)")
    arrow(s, 316, 56, 360, 100, color="var(--hi)")
    s.text(350, 70, "text tokens", size=11, color="var(--hi)", anchor="start")
    box(s, 10, 128, 120, 52, ["noise", "latent, t = 1"], color="currentColor")
    arrow(s, 132, 154, 162, 154)
    box(s, 164, 128, 150, 52, ["patchify", "2×2 latent patches"], color="var(--accent2)")
    arrow(s, 316, 154, 360, 154)
    s.rect(362, 92, 196, 124, rx=10, stroke="var(--accent)", fill="var(--paper)", width=1.8)
    s.text(460, 114, "diffusion transformer", size=13, color="var(--accent)", weight="600")
    s.text(460, 136, "joint attention over", size=12, color="var(--muted)")
    s.text(460, 152, "text + image tokens (MMDiT)", size=12, color="var(--muted)")
    s.text(460, 172, "AdaLN: timestep, pooled text", size=12, color="var(--muted)")
    s.text(460, 194, "predicts velocity (or ε)", size=12, color="var(--muted)")
    s.path("M 430 218 C 430 270, 540 270, 548 222",
           color="var(--accent)", width=1.4, dash="5 4")
    arrow(s, 549, 230, 551, 219, color="var(--accent)")
    s.text(590, 246, "sampler loop: N steps", size=11, color="var(--accent)", anchor="start")
    s.text(590, 262, "(Euler, DPM-Solver, …;", size=11, color="var(--accent)", anchor="start")
    s.text(590, 278, "classifier-free guidance)", size=11, color="var(--accent)", anchor="start")
    arrow(s, 560, 120, 596, 90)
    box(s, 598, 30, 130, 52, ["VAE decoder", "latent → pixels"], color="var(--accent2)")
    arrow(s, 663, 84, 663, 104)
    s.text(663, 120, "image", size=13, weight="600")
    write("diff-c-pipeline", s.figure(
        "<b>How a prompt becomes an image in a current latent flow/diffusion transformer</b> (the SD3 / FLUX "
        "family). Frozen text encoders turn the prompt into tokens; a noise latent is cut into patches; the "
        "transformer predicts a velocity (or noise) for every patch at the current timestep; a sampler repeats "
        "this for N steps, applying classifier-free guidance unless the model was guidance-distilled; the VAE "
        "decoder turns the final latent into pixels.",
        alt="Pipeline: prompt to text encoders, noise latent to patchify, diffusion transformer loop, VAE decoder"))


if __name__ == "__main__":
    guidance_toy()
    ldm_diagram()
    velocity_score_check()
    lognormal()
    shift_curve()
    pipeline_diagram()
    flow_toy()
    print("ok")
