#!/usr/bin/env python3
"""Figures for 5-deep-networks: MLPs, activations, convolution, conv layers, pooling.

usage: python3 tools/figs_deep_a.py   (then python3 tools/assemble.py <pages>)
Every number is computed here with numpy or copied from the Lecture 14/15 notebook outputs.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mplfig import plt, INK, ACCENT, ACCENT2, HI, MUTED, save  # noqa: E402
from figlib import SVG  # noqa: E402

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def write(name, html):
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, name + ".html"), "w") as f:
        f.write(html + "\n")


# --------------------------------------------------------------------- activations
def activations():
    z = np.linspace(-5, 5, 401)
    sig = 1 / (1 + np.exp(-z))
    tau = 0.1
    fns = [
        ("sigmoid", sig, sig * (1 - sig), ACCENT, "-"),
        ("tanh", np.tanh(z), 1 - np.tanh(z) ** 2, HI, "-"),
        ("ReLU", np.maximum(0, z), (z > 0).astype(float), ACCENT2, "-"),
        (r"leaky ReLU ($\tau=0.1$)", np.where(z >= 0, z, tau * z), np.where(z >= 0, 1.0, tau), MUTED, "--"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, f, df, c, ls in fns:
        axes[0].plot(z, f, color=c, ls=ls, label=name)
        axes[1].plot(z, df, color=c, ls=ls, label=name)
    axes[0].set_ylim(-1.3, 3.2)
    axes[1].set_ylim(-0.1, 1.15)
    axes[1].axhline(0.25, color=MUTED, lw=1, ls=":")
    axes[1].text(1.6, 0.31, "sigmoid max 0.25", color=MUTED, fontsize=9)
    axes[0].set_title(r"activation $\sigma(z)$", loc="left")
    axes[1].set_title("derivative σ′(z)", loc="left")
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("$z$")
    axes[0].legend(loc="upper left", fontsize=9)
    save(fig, "deepa-activations",
         "<b>Four activations and their derivatives.</b> Sigmoid's slope never exceeds 0.25 and tanh's never exceeds 1, "
         "and both flatten out for large |<i>z</i>| (saturation). ReLU's slope is exactly 1 for <i>z</i> &gt; 0 and 0 for <i>z</i> &lt; 0; "
         "leaky ReLU keeps a small slope τ on the negative side.",
         alt="Sigmoid, tanh, ReLU and leaky ReLU curves with their derivatives")


# ----------------------------------------------------------------------- two moons
def two_moons(n=100, sigma=0.1, seed=0):
    rng = np.random.default_rng(seed)
    off = -np.pi / 8
    a = rng.random(n) * (np.pi - 2 * off) + off
    b = rng.random(n) * (np.pi - 2 * off) + off
    pos = np.stack([np.cos(a), np.sin(a)], 1) + sigma * rng.standard_normal((n, 2))
    neg = np.array([1.0, 0.0]) + np.stack([np.cos(-b), np.sin(-b)], 1) + sigma * rng.standard_normal((n, 2))
    X = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    return X, y


def adam_fit(params, grad_fn, steps, lr):
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    for t in range(1, steps + 1):
        gs = grad_fn(params)
        for i, (p, g) in enumerate(zip(params, gs)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return params


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def moons_figure():
    X, y = two_moons()
    rng = np.random.default_rng(1)
    # logistic regression: p = sigmoid(w.x + b)
    w, b = np.zeros(2), np.zeros(1)

    def g_lr(ps):
        w, b = ps
        p = sigmoid(X @ w + b)
        e = (p - y) / len(y)
        return [X.T @ e, np.array([e.sum()])]

    w, b = adam_fit([w, b], g_lr, 3000, 0.05)
    # two-layer MLP: 2 -> 64 -> 1 with ReLU
    H = 64
    W1 = rng.standard_normal((2, H)) * 0.7
    b1 = np.zeros(H)
    W2 = rng.standard_normal(H) * 0.1
    b2 = np.zeros(1)

    def fwd(Xq, ps):
        W1, b1, W2, b2 = ps
        h = np.maximum(0, Xq @ W1 + b1)
        return h, sigmoid(h @ W2 + b2)

    def g_mlp(ps):
        W1, b1, W2, b2 = ps
        h, p = fwd(X, ps)
        e = (p - y) / len(y)
        gW2, gb2 = h.T @ e, np.array([e.sum()])
        dh = np.outer(e, W2) * (h > 0)
        return [X.T @ dh, dh.sum(0), gW2, gb2]

    ps = adam_fit([W1, b1, W2, b2], g_mlp, 4000, 0.01)
    gx, gy = np.meshgrid(np.linspace(-1.6, 2.6, 160), np.linspace(-1.2, 1.4, 120))
    G = np.stack([gx.ravel(), gy.ravel()], 1)
    p_lr = sigmoid(G @ w + b).reshape(gx.shape)
    p_mlp = fwd(G, ps)[1].reshape(gx.shape)
    acc_lr = ((sigmoid(X @ w + b) > 0.5) == y).mean() * 100
    acc_mlp = ((fwd(X, ps)[1] > 0.5) == y).mean() * 100
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, P, title in [(axes[0], p_lr, f"logistic regression: {acc_lr:.0f}% train acc."),
                         (axes[1], p_mlp, f"2-layer ReLU MLP (64 hidden): {acc_mlp:.0f}%")]:
        ax.contourf(gx, gy, P, levels=[0, 0.5, 1], colors=[ACCENT, HI], alpha=0.13)
        ax.contour(gx, gy, P, levels=[0.5], colors=[INK], linewidths=1.5)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=14, color=HI, marker="o", label="positive")
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=16, color=ACCENT, marker="^", label="negative")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].legend(loc="lower left", fontsize=9)
    save(fig, "deepa-two-moons",
         "<b>Two moons: linear vs non-linear boundary.</b> The dark curve is where <i>p</i> = 0.5. Logistic regression can only "
         "draw a straight line; one hidden ReLU layer bends it around the moons. (A numpy re-creation of the Lecture 14 "
         f"experiment, full-batch training; the lecture used PyTorch, 1024 hidden units and SGD with momentum.)",
         alt="Two moons dataset with a straight logistic-regression boundary and a curved MLP boundary")
    return acc_lr, acc_mlp


# ------------------------------------------------------------ 1-D conv vs correlation
def conv_vs_corr():
    x = np.array([2.0, 1.0, -3.0, 0.0, 4.0])
    h = np.array([1.0, 0.0, -1.0])
    conv = np.convolve(x, h)                     # flips h
    corr = np.correlate(np.pad(x, 2), h, "valid")  # no flip, padding 2
    assert conv.tolist() == [2, 1, -5, -1, 7, 0, -4], conv
    assert corr.tolist() == [-2, -1, 5, 1, -7, 0, 4], corr
    n = np.arange(len(conv))
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), gridspec_kw={"width_ratios": [1.1, 1.6, 1.6]})
    ax = axes[0]
    ax.stem(np.arange(5), x, linefmt=MUTED, markerfmt="o", basefmt=" ")
    ax.stem(np.arange(3) + 6.5, h, linefmt=MUTED, markerfmt="s", basefmt=" ")
    for i, v in enumerate(x):
        ax.text(i, v + (0.5 if v >= 0 else -0.9), f"{v:g}", ha="center", fontsize=9)
    for i, v in enumerate(h):
        ax.text(i + 6.5, v + (0.5 if v >= 0 else -0.9), f"{v:g}", ha="center", fontsize=9)
    ax.set_xticks([2, 7.5])
    ax.set_xticklabels(["$x$", "$h$"])
    ax.set_title("input and kernel", loc="left", fontsize=10)
    for ax, y, title, c in [(axes[1], conv, "convolution $x*h$ (kernel flipped)", ACCENT),
                            (axes[2], corr, "cross-correlation (F.conv1d)", HI)]:
        ml, sl, bl = ax.stem(n, y, basefmt=" ")
        plt.setp(ml, color=c)
        plt.setp(sl, color=c)
        for i, v in enumerate(y):
            ax.text(i, v + (0.7 if v >= 0 else -1.4), f"{v:g}", ha="center", fontsize=9)
        ax.set_xticks(n)
        ax.set_xlabel("$n$")
        ax.set_title(title, loc="left", fontsize=10)
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylim(-8.5, 8.5)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    save(fig, "deepa-conv-vs-corr",
         "<b>The Lecture 15 example.</b> <i>x</i> = [2, 1, −3, 0, 4], <i>h</i> = [1, 0, −1], full output of length 5 + 3 − 1 = 7. "
         "Convolution gives [2, 1, −5, −1, 7, 0, −4]; cross-correlation (what <code>F.conv1d</code> computes, no flip) "
         "gives the negative here, because flipping this antisymmetric <i>h</i> negates it.",
         alt="Stem plots of x, h, their convolution and cross-correlation")


# ------------------------------------------------------------------ padding and stride
def padding_stride():
    H, p, K, s = 7, 1, 3, 2
    cells = H + 2 * p
    cw, x0, y0 = 40, 30, 46
    S = SVG(cells * cw + 2 * x0 + 70, 250, font_size=13)
    S.text(x0, 16, f"H_in = {H}, padding p = {p}, kernel K = {K}, stride s = {s}", anchor="start", size=13)
    vals = ["0"] + [str(v) for v in [3, 1, 4, 1, 5, 9, 2]] + ["0"]
    for i in range(cells):
        pad = i < p or i >= p + H
        S.rect(x0 + i * cw, y0, cw, cw, stroke="var(--muted)" if pad else "currentColor",
               dash="4 3" if pad else None, fill="none")
        S.text(x0 + i * cw + cw / 2, y0 + cw / 2, vals[i], color="var(--muted)" if pad else "currentColor")
    n_out = (H + 2 * p - K) // s + 1
    colors = ["var(--accent)", "var(--hi)", "var(--accent2)", "var(--accent)"]
    for o in range(n_out):
        yy = y0 + cw + 18 + o * 26
        xa = x0 + o * s * cw
        S.rect(xa + 3, yy, K * cw - 6, 18, rx=4, stroke=colors[o % 4], fill=colors[o % 4], fill_opacity=0.15, width=1.6)
        S.text(xa + K * cw + 8, yy + 9, f"output {o + 1}", anchor="start", size=12, color=colors[o % 4])
    S.text(x0, 238, f"floor(({H} + 2·{p} − {K}) / {s}) + 1 = floor(6/2) + 1 = {n_out} outputs",
           anchor="start", size=13, weight="600")
    write("deepa-padding-stride", S.figure(
        "<b>Where the output-size formula comes from.</b> Padding adds <i>p</i> zeros on each side (dashed), so the "
        "padded row has <i>H</i> + 2<i>p</i> = 9 cells. A window of <i>K</i> = 3 cells fits at starting positions 0, 2, 4, 6 when it moves "
        "<i>s</i> = 2 cells at a time: ⌊(9 − 3)/2⌋ + 1 = 4 outputs.",
        alt="A padded row of 9 cells with four kernel windows of width 3 at stride 2"))
    return n_out


# --------------------------------------------------------------------- receptive field
def receptive_field():
    layers = [("3-wide conv", 3, 1), ("2×2 pool, stride 2", 2, 2), ("3-wide conv", 3, 1)]
    n = 13
    rows = [list(range(n))]
    for _, k, s in layers:
        prev = rows[-1]
        rows.append([sum(prev[i * s:i * s + k]) / k for i in range((len(prev) - k) // s + 1)])
    cone = [{len(rows[-1]) // 2}]
    for _, k, s in reversed(layers):
        cone.insert(0, {i * s + d for i in cone[0] for d in range(k)})
    dx, x0, dy, ytop = 34, 150, 62, 30
    S = SVG(x0 + n * dx + 20, ytop + len(rows) * dy - 10, font_size=13)
    ypos = [ytop + (len(rows) - 1 - lv) * dy for lv in range(len(rows))]
    for lv in range(1, len(rows)):
        k, s = layers[lv - 1][1:]
        for i in cone[lv]:
            for d in range(k):
                S.line(x0 + rows[lv][i] * dx, ypos[lv], x0 + rows[lv - 1][i * s + d] * dx, ypos[lv - 1],
                       color="var(--accent)", width=1.2, opacity=0.7)
    for lv, (row, used) in enumerate(zip(rows, cone)):
        for i, xp in enumerate(row):
            on = i in used
            if lv == 0:
                S.rect(x0 + xp * dx - 10, ypos[lv] - 10, 20, 20, rx=3, stroke="var(--accent)" if on else "var(--muted)",
                       fill="var(--accent)" if on else "none", fill_opacity=0.35 if on else None)
            else:
                S.circle(x0 + xp * dx, ypos[lv], 8, stroke="var(--accent)" if on else "var(--muted)",
                         fill="var(--accent)" if on else "var(--paper)", fill_opacity=0.5 if on else None)
        label = "input pixels" if lv == 0 else layers[lv - 1][0]
        S.text(x0 - 22, ypos[lv], label, anchor="end", size=12)
    rf = len(cone[0])
    write("deepa-receptive-field", S.figure(
        f"<b>Receptive field, one row of pixels.</b> Conv 3 → pool 2 (stride 2) → conv 3. The top unit depends on "
        f"the {rf} highlighted input pixels: <i>r</i> = 1 + (3−1)·1 + (2−1)·1 + (3−1)·2 = {rf}. After the "
        "pool the jump is 2, so the last 3-wide kernel adds 2·2 = 4 pixels instead of 2.",
        alt="Receptive-field cone through conv, pool, conv layers covering 8 input pixels"))
    return rf


# ---------------------------------------------------------------------------- pooling
def pooling():
    x = np.array([[1, 3, 2, 0], [4, 6, 1, 1], [0, 2, 5, 7], [1, 1, 8, 2]], dtype=float)
    mx = x.reshape(2, 2, 2, 2).max(axis=(1, 3))
    mn = x.reshape(2, 2, 2, 2).mean(axis=(1, 3))
    assert mx.tolist() == [[6, 2], [2, 8]] and mn.tolist() == [[3.5, 1], [1, 5.5]]
    c = 36
    cols = ["var(--accent)", "var(--hi)", "var(--accent2)", "var(--muted)"]
    S = SVG(560, 200, font_size=14)

    def grid(x0, y0, M, quad):
        r, k = M.shape
        for i in range(r):
            for j in range(k):
                q = cols[(i // quad) * 2 + (j // quad)]
                S.rect(x0 + j * c, y0 + i * c, c, c, stroke="currentColor", fill=q, fill_opacity=0.18, width=1)
                v = M[i, j]
                S.text(x0 + j * c + c / 2, y0 + i * c + c / 2, f"{v:g}")

    grid(20, 40, x, 2)
    S.text(20 + 2 * c, 22, "4 × 4 input", size=13)
    S.line(180, 76, 262, 60, arrow=True)
    S.line(180, 116, 262, 140, arrow=True)
    S.text(220, 50, "max", size=12)
    S.text(220, 150, "mean", size=12)
    grid(280, 18, mx, 1)
    grid(280, 110, mn, 1)
    S.text(370, 54, "2×2 max pool, stride 2", anchor="start", size=13)
    S.text(370, 146, "2×2 mean pool, stride 2", anchor="start", size=13)
    S.text(370, 74, "global max = 8", anchor="start", size=12, color="var(--muted)")
    S.text(370, 166, "global mean = 44/16 = 2.75", anchor="start", size=12, color="var(--muted)")
    assert x.sum() == 44
    write("deepa-pooling", S.figure(
        "<b>Pooling the Lecture 15 tensor.</b> Each colored 2×2 window becomes one number: its maximum "
        "(top) or its mean (bottom). The 4×4 map shrinks to 2×2; global pooling reduces the whole map "
        "to one number. No parameters are involved.",
        alt="4 by 4 grid pooled to 2 by 2 by max and mean"))


# ---------------------------------------------------------------- flatten vs GAP heads
def heads():
    S = SVG(680, 200, font_size=13)

    def stack(x, y, w, h, n, label):
        for i in range(n):
            S.rect(x + i * 6, y - i * 6, w, h, fill="var(--paper)", stroke="currentColor", width=1.2)
        S.text(x + w / 2 + n * 3, y + h + 16, label, size=12)

    stack(20, 70, 64, 64, 4, "")
    S.text(70, 158, "64 × 16 × 16", size=12)
    S.text(70, 174, "feature maps", size=12)
    # flatten path
    S.line(118, 70, 200, 40, arrow=True)
    S.text(150, 42, "flatten", size=12)
    S.rect(208, 30, 230, 18, rx=3, fill="var(--hi)", fill_opacity=0.15, stroke="var(--hi)")
    S.text(323, 39, "16,384 numbers (position kept)", size=12)
    S.line(442, 39, 500, 39, arrow=True)
    S.text(471, 26, "Linear", size=12)
    S.rect(506, 30, 44, 18, rx=3, stroke="currentColor")
    S.text(528, 39, "10", size=12)
    S.text(560, 39, "163,850 params", anchor="start", size=11, color="var(--hi)")
    # GAP path
    S.line(118, 120, 200, 140, arrow=True)
    S.text(166, 112, "global avg pool", size=12)
    S.rect(208, 131, 90, 18, rx=3, fill="var(--accent)", fill_opacity=0.15, stroke="var(--accent)")
    S.text(253, 140, "64 numbers", size=12)
    S.text(253, 164, "(where is dropped)", size=11, color="var(--muted)")
    S.line(302, 140, 360, 140, arrow=True)
    S.text(331, 127, "Linear", size=12)
    S.rect(366, 131, 44, 18, rx=3, stroke="currentColor")
    S.text(388, 140, "10", size=12)
    S.text(420, 140, "650 params", anchor="start", size=11, color="var(--accent)")
    write("deepa-heads", S.figure(
        "<b>Two heads on the same conv layers.</b> Flattening keeps every feature at its position, so the linear layer "
        "learns position-specific weights (16384·10 + 10 parameters). Global average pooling keeps only how strongly "
        "each of the 64 features fired (64·10 + 10 parameters).",
        alt="Feature maps going to a flatten head and a global-average-pool head"))


# ------------------------------------------------------------------ shift experiment
def shift_curve():
    shifts = [0, 1, 2, 3, 4, 6, 8, 12, 16]
    res = {  # Lecture 15 notebook output
        "MLP": [86.8, 70.0, 46.1, 24.3, 17.2, 15.0, 13.9, 7.2, 10.9],
        "CNN + flatten + FC": [90.2, 84.5, 58.1, 19.3, 13.2, 10.1, 8.6, 8.3, 9.7],
        "CNN + GAP + FC": [78.0, 76.6, 74.4, 76.1, 78.0, 74.6, 78.1, 78.0, 75.3],
        "CNN 5×5 + GAP + FC": [80.8, 80.5, 79.7, 79.8, 81.2, 79.5, 81.2, 81.0, 80.8],
    }
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for (name, ys), c, m in zip(res.items(), [MUTED, HI, ACCENT, ACCENT2], ["s", "D", "o", "^"]):
        ax.plot(shifts, ys, color=c, marker=m, markersize=5, label=name)
    ax.axhline(10, color=MUTED, lw=1, ls="--")
    ax.text(16, 12, "chance", color=MUTED, fontsize=9, ha="right")
    ax.set_ylim(0, 100)
    ax.set_xticks(shifts)
    ax.set_xlabel("diagonal shift from the training position (pixels)")
    ax.set_ylabel("test accuracy (%)")
    ax.grid(True, axis="y")
    ax.legend(fontsize=9, loc="center right")
    save(fig, "deepa-shift",
         "<b>Trained at the center, tested shifted</b> (Lecture 15 results, FashionMNIST on a 64×64 canvas). "
         "Only the models with a global-average-pooling head keep their accuracy when the object moves.",
         alt="Accuracy versus shift for MLP, CNN with flatten head and two CNNs with GAP heads")


if __name__ == "__main__":
    activations()
    print("two moons train acc (lr, mlp):", moons_figure())
    conv_vs_corr()
    print("padding/stride outputs:", padding_stride())
    print("receptive field:", receptive_field())
    pooling()
    heads()
    shift_curve()
    print("done")
