#!/usr/bin/env python3
"""Figures for the Midterm 2 section (content/9-midterm-2/). Run: python3 -B tools/figs_exam2.py

Writes tools/figs/exam2-*.html; then python3 tools/assemble.py content/9-midterm-2/<page>.md
The exam plots are re-drawn (the captions say so): the Spring 2025 loss curves are an
exponential fit to the printed figure; the Spring 2026 paths are a fresh GD-vs-SGD
simulation on a small least-squares problem with the same qualitative picture.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figlib import SVG  # noqa: E402
from mplfig import ACCENT, ACCENT2, HI, INK, MUTED, FIGS, plt, save  # noqa: E402

MONO = "'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace"


def write(name, html):
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, name + ".html"), "w") as f:
        f.write(html.replace("\n", " ") + "\n")


def arrow(s, x1, y1, x2, y2, color="currentColor", width=1.5):
    s.line(x1, y1, x2, y2, color=color, width=width)
    v = np.array([x2 - x1, y2 - y1], float)
    v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]])
    tip = np.array([x2, y2])
    a, b = tip - 7 * v + 3.5 * n, tip - 7 * v - 3.5 * n
    s.add(f'<path d="M{tip[0]:.1f},{tip[1]:.1f} L{a[0]:.1f},{a[1]:.1f} L{b[0]:.1f},{b[1]:.1f} Z" fill="{color}"/>')


# ---------------------------------------------------------------- Spring 2025 Q2: both losses plateau high
def sp25_loss():
    e = np.arange(0, 501)
    tr = 0.28 + 0.25 * np.exp(-e / 38)
    va = 0.287 + 0.243 * np.exp(-e / 38)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(e, tr, color=ACCENT, lw=1.8, label="training loss")
    ax.plot(e, va, color=HI, lw=1.8, ls="--", label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_ylim(0, 0.7)
    ax.set_xlim(0, 500)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True)
    save(fig, "exam2-sp25-loss",
         "Spring 2025 Problem 2, re-drawn from the exam's figure: training and validation loss fall together from about "
         "0.53 and flatten near 0.28 by epoch 200, almost on top of each other.",
         alt="Training and validation loss curves that fall together and plateau at about 0.28", width=520)


# ---------------------------------------------------------------- Spring 2026 Q2: GD vs SGD paths
def sp26_paths():
    rng = np.random.default_rng(364)
    n = 60
    x = rng.uniform(0, 2, n)
    th_true = np.array([4.4, 2.9])                      # (theta_0, theta_1)
    y = th_true[0] + th_true[1] * x + rng.normal(0, 1.2, n)
    X = np.stack([np.ones(n), x], 1)
    grad = lambda th, idx: 2 * X[idx].T @ (X[idx] @ th - y[idx]) / len(idx)
    start = np.array([-1.0, 6.0])

    def run(batch, steps, lr):
        th, path = start.copy(), [start.copy()]
        for _ in range(steps):
            idx = np.arange(n) if batch is None else rng.integers(0, n, batch)
            th = th - lr * grad(th, idx)
            path.append(th.copy())
        return np.array(path)

    gd, sgd = run(None, 400, 0.02), run(1, 400, 0.02)
    g0, g1 = np.meshgrid(np.linspace(-2, 6.5, 160), np.linspace(1.5, 6.5, 120))
    L = ((g0[..., None] + g1[..., None] * x - y) ** 2).mean(-1)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    for ax, path, title, col in [(axes[0], sgd, "Plot A", HI), (axes[1], gd, "Plot B", ACCENT)]:
        ax.contour(g0, g1, L, levels=np.geomspace(L.min() * 1.05, L.max(), 12), colors=[MUTED], linewidths=0.7)
        ax.plot(path[:, 0], path[:, 1], color=col, lw=1.3)
        ax.plot(*start, "o", color=INK, ms=4)
        ax.annotate("Start", start, (start[0] + 0.35, start[1] + 0.1), fontsize=9)
        ax.annotate("End", path[-1], (path[-1][0] + 0.25, path[-1][1] + 0.35), fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel(r"$\theta_0$")
        ax.set_xlim(-2, 6.5)
        ax.set_ylim(1.5, 6.5)
    axes[0].set_ylabel(r"$\theta_1$")
    save(fig, "exam2-sp26-paths",
         "Spring 2026 Problem 2, re-created (a fresh simulation on a small least-squares problem, not the exam's data): the "
         "parameters' path over the loss contours for two optimizers from the same start with the same step size. "
         "One path jitters at every step; the other is smooth.",
         alt="Two contour plots: a jagged path in plot A and a smooth path in plot B, both from the same start", width=760)


# ---------------------------------------------------------------- Spring 2026 Q7: encoder layer block diagram
def sp26_encoder():
    s = SVG(420, 400, font_size=13)
    cx = 230
    s.text(cx, 388, "h_l   (N × d)", size=13, font=MONO)
    blocks = [(318, "Multi-head self-attention"), (232, "Norm"), (160, "Feedforward layer"), (74, "Norm")]
    for y, lab in blocks:
        h = 34 if "Norm" not in lab else 26
        s.rect(cx - 95, y - h / 2, 190, h, rx=6, stroke="currentColor", fill="var(--paper)")
        s.text(cx, y, lab, size=13)
    # plus nodes
    for y in (272, 114):
        s.circle(cx, y, 10, stroke="var(--hi)", fill="var(--paper)")
        s.text(cx, y, "+", size=15, color="var(--hi)")
    arrow(s, cx, 376, cx, 337)
    arrow(s, cx, 301, cx, 283)
    arrow(s, cx, 262, cx, 246)
    arrow(s, cx, 219, cx, 178)
    arrow(s, cx, 143, cx, 125)
    arrow(s, cx, 104, cx, 88)
    arrow(s, cx, 61, cx, 30)
    s.text(cx, 16, "h_{l+1}   (N × d)", size=13, font=MONO)
    # skip connections
    s.path(f"M{cx},356 L{cx - 130},356 L{cx - 130},272 L{cx - 18},272", color="var(--hi)", width=1.5)
    arrow(s, cx - 20, 272, cx - 11, 272, color="var(--hi)")
    s.path(f"M{cx},200 L{cx - 130},200 L{cx - 130},114 L{cx - 18},114", color="var(--hi)", width=1.5)
    arrow(s, cx - 20, 114, cx - 11, 114, color="var(--hi)")
    s.text(cx - 138, 314, "skip", size=11.5, color="var(--hi)", anchor="end")
    s.text(cx - 138, 157, "skip", size=11.5, color="var(--hi)", anchor="end")
    write("exam2-sp26-encoder", s.figure(
        "Spring 2026 Problem 7, re-drawn: one Transformer encoder layer. The input is added back after self-attention and "
        "again after the feed-forward layer (the two skip connections), and each sum is normalized.",
        alt="Block diagram: self-attention, add, norm, feed-forward, add, norm, with two skip connections"))


# ---------------------------------------------------------------- Fall 2024 Q4(c): 2x2 windows at stride 3
def f24_pool():
    Z = [[4, 1, 8, 2, 3, 7], [0, 0, 2, 5, 6, 5], [1, 4, 9, 2, 5, 5], [2, 6, 2, 3, 3, 6], [0, 6, 6, 7, 6, 3], [6, 2, 1, 1, 9, 3]]
    s = SVG(470, 250, font_size=14)
    x0, y0, c = 30, 22, 34
    for (r, q) in [(0, 0), (0, 3), (3, 0), (3, 3)]:
        s.rect(x0 + q * c, y0 + r * c, 2 * c, 2 * c, stroke="var(--hi)", fill="var(--hi)", fill_opacity=0.14, width=2)
    for i in range(6):
        for j in range(6):
            s.rect(x0 + j * c, y0 + i * c, c, c, stroke="var(--muted)", width=0.7)
            win = (i % 3 < 2) and (j % 3 < 2)
            s.text(x0 + j * c + c / 2, y0 + i * c + c / 2 + 1, Z[i][j], size=14, font=MONO,
                   color="currentColor" if win else "var(--muted)")
    s.rect(x0, y0, 6 * c, 6 * c, stroke="currentColor", width=1.4)
    arrow(s, x0 + 6 * c + 16, y0 + 3 * c, x0 + 6 * c + 62, y0 + 3 * c)
    s.text(x0 + 6 * c + 39, y0 + 3 * c - 12, "max", size=12)
    ox, oy = x0 + 6 * c + 78, y0 + 2 * c
    for i, row in enumerate([[4, 6], [6, 7]]):
        for j, v in enumerate(row):
            s.rect(ox + j * c, oy + i * c, c, c, stroke="var(--hi)", fill="var(--hi)", fill_opacity=0.14, width=1.4)
            s.text(ox + j * c + c / 2, oy + i * c + c / 2 + 1, v, size=14, font=MONO)
    s.text(x0, y0 + 6 * c + 16, "stride 3 > kernel 2: grey entries are never read", size=12, color="var(--muted)", anchor="start")
    write("exam2-f24-pool", s.figure(
        "Fall 2024 Problem 4(c). A 2 × 2 max pool with stride 3 on a 6 × 6 map places windows at rows 0–1 and 3–4, columns "
        "0–1 and 3–4: ⌊(6 − 2)/3⌋ + 1 = 2 positions each way. Row and column 2 and 5 fall between windows.",
        alt="A 6 by 6 grid with four highlighted 2 by 2 windows and the 2 by 2 output of their maxima"))


# ---------------------------------------------------------------- Fall 2025 Q5(b): 64x64 image, 8x8 patches
def f25_patches():
    s = SVG(470, 230, font_size=13)
    x0, y0, c = 40, 26, 22
    for i in range(9):
        s.line(x0 + i * c, y0, x0 + i * c, y0 + 8 * c, width=0.8, color="var(--muted)")
        s.line(x0, y0 + i * c, x0 + 8 * c, y0 + i * c, width=0.8, color="var(--muted)")
    s.rect(x0, y0, 8 * c, 8 * c, stroke="currentColor", width=1.4)
    s.rect(x0, y0, c, c, stroke="var(--hi)", fill="var(--hi)", fill_opacity=0.25, width=1.8)
    s.rect(x0 + 7 * c, y0 + 7 * c, c, c, stroke="var(--accent)", fill="var(--accent)", fill_opacity=0.25, width=1.8)
    s.text(x0 + 4 * c, y0 - 12, "64 pixels = 8 patches of 8", size=12)
    s.text(x0 - 12, y0 + 4 * c, "64", size=12, anchor="end")
    tx = x0 + 8 * c + 26
    s.text(tx, y0 + 30, "8 × 8 = 64 tokens", size=13.5, anchor="start", weight=600)
    s.text(tx, y0 + 54, "each 3 · 8 · 8 = 192 numbers", size=13, anchor="start")
    s.text(tx, y0 + 92, "top-left and bottom-right patches", size=12, anchor="start", color="var(--muted)")
    s.text(tx, y0 + 109, "are two tokens: self-attention", size=12, anchor="start", color="var(--muted)")
    s.text(tx, y0 + 126, "relates them in the first layer", size=12, anchor="start", color="var(--muted)")
    write("exam2-f25-patches", s.figure(
        "Fall 2025 Problem 5(b), (d). A 64 × 64 image cut into 8 × 8 patches with stride 8 gives an 8 × 8 grid: 64 tokens, "
        "each a flattened 3 × 8 × 8 = 192-vector. At 128 × 128 the same model would simply see 16 × 16 = 256 tokens.",
        alt="An 8 by 8 grid of patches on a 64 by 64 image with the two opposite corner patches highlighted"))


if __name__ == "__main__":
    sp25_loss()
    sp26_paths()
    sp26_encoder()
    f24_pool()
    f25_patches()
    print("ok")
