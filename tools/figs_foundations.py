#!/usr/bin/env python3
"""Figures for unit 1-foundations (Lectures 1-2). Writes tools/figs/found-*.html.

usage: python3 tools/figs_foundations.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figlib import SVG, MONO  # noqa: E402
from mplfig import plt, INK, ACCENT, ACCENT2, HI, MUTED, save  # noqa: E402

FIGS = os.path.join(HERE, "figs")


def write(name, html):
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, name + ".html"), "w") as f:
        f.write(html + "\n")


def grid(S, x0, y0, vals, cell=30, hi=None, color=None, size=13, mono=True, ghost=False, base=False):
    """Draw a 2-D grid of numbers; hi = {(r, c): css-colour} fills."""
    hi = hi or {}
    for r, row in enumerate(vals):
        for c, v in enumerate(row):
            x, y = x0 + c * cell, y0 + r * cell
            if base:
                S.rect(x, y, cell, cell, fill="var(--paper)", stroke="none")
            fill = hi.get((r, c), "none")
            S.rect(x, y, cell, cell, fill=fill, fill_opacity=0.28 if fill != "none" else None,
                   stroke=color or "currentColor", width=1.2, dash="4 3" if ghost else None)
            S.text(x + cell / 2, y + cell / 2, str(v), size=size, font=MONO if mono else None,
                   color="var(--muted)" if ghost else "currentColor")


# ---------------------------------------------------------------- cat vs stack
def cat_stack():
    S = SVG(760, 250, font_size=13)
    A = [["a", "a", "a"], ["a", "a", "a"]]
    B = [["b", "b", "b"], ["b", "b", "b"]]
    ca, cb = "var(--accent)", "var(--hi)"
    hiA = {(r, c): ca for r in range(2) for c in range(3)}
    hiB = {(r, c): cb for r in range(2) for c in range(3)}
    cell = 24
    # inputs
    S.text(20, 22, "a, b: shape (2, 3)", anchor="start", weight=600)
    grid(S, 20, 40, A, cell, hiA)
    grid(S, 20, 100, B, cell, hiB)
    # cat dim 0
    x = 170
    S.text(x, 22, "torch.cat((a, b), dim=0)", anchor="start", font=MONO, size=12)
    grid(S, x, 40, A + B, cell, {**hiA, **{(r + 2, c): cb for r in range(2) for c in range(3)}})
    S.text(x, 150, "→ (4, 3)", anchor="start", weight=600)
    # cat dim 1
    x = 360
    S.text(x, 22, "torch.cat((a, b), dim=1)", anchor="start", font=MONO, size=12)
    AB = [A[r] + B[r] for r in range(2)]
    grid(S, x, 40, AB, cell, {**hiA, **{(r, c + 3): cb for r in range(2) for c in range(3)}})
    S.text(x, 102, "→ (2, 6)", anchor="start", weight=600)
    # stack dim 0: two sheets
    x = 550
    S.text(x, 22, "torch.stack((a, b), dim=0)", anchor="start", font=MONO, size=12)
    grid(S, x + 22, 40, B, cell, hiB)
    grid(S, x, 62, A, cell, hiA, base=True)
    S.text(x, 134, "→ (2, 2, 3)", anchor="start", weight=600)
    S.text(x, 152, "new dim 0: out[0] = a, out[1] = b", anchor="start", color="var(--muted)", size=12)
    S.text(20, 200, "cat glues along an existing dimension (that size grows); stack adds a new dimension of size = number of tensors.",
           anchor="start", size=12, color="var(--muted)")
    S.text(20, 222, "cat needs equal sizes in every other dim; stack needs identical shapes.", anchor="start", size=12,
           color="var(--muted)")
    write("found-cat-stack", S.figure(
        "<b>cat vs stack.</b> Two (2, 3) tensors: <code>cat</code> along dim 0 or 1 makes a bigger matrix; "
        "<code>stack</code> puts them on a new axis, here giving shape (2, 2, 3).",
        alt="cat along dim 0 gives 4 by 3, along dim 1 gives 2 by 6, stack gives 2 by 2 by 3"))


# ---------------------------------------------------------------- storage and views
def storage_views():
    S = SVG(720, 260, font_size=13)
    cell = 34
    col = {1, 5, 9}
    S.text(20, 20, "storage (one flat block of 12 elements)", anchor="start", weight=600)
    for i in range(12):
        x = 20 + i * cell
        S.rect(x, 34, cell, cell, fill="var(--accent)" if i in col else "none",
               fill_opacity=0.3 if i in col else None, width=1.2)
        S.text(x + cell / 2, 34 + cell / 2, str(i), font=MONO)
        S.text(x + cell / 2, 80, str(i), size=10, color="var(--muted)")
    S.text(20 + 12 * cell + 8, 51, "value", anchor="start", size=11, color="var(--muted)")
    S.text(20 + 12 * cell + 8, 80, "offset", anchor="start", size=11, color="var(--muted)")
    # view as 3x4
    vals = [[4 * r + c for c in range(4)] for r in range(3)]
    grid(S, 20, 120, vals, cell, {(r, 1): "var(--accent)" for r in range(3)}, size=13)
    S.text(20, 240, "a = torch.arange(12).view(3, 4)", anchor="start", font=MONO, size=12)
    S.text(180, 125, "shape (3, 4), stride (4, 1)", anchor="start", size=12)
    S.text(180, 145, "next row: jump 4 slots; next col: 1 slot", anchor="start", size=12, color="var(--muted)")
    S.text(180, 180, "a[:, 1] → elements 1, 5, 9", anchor="start", size=12, font=MONO)
    S.text(180, 200, "same storage: offset 1, stride (4,)", anchor="start", size=12, color="var(--muted)")
    S.text(445, 125, "a.T → shape (4, 3), stride (1, 4)", anchor="start", size=12, font=MONO)
    S.text(445, 145, "same storage, not contiguous", anchor="start", size=12, color="var(--muted)")
    S.text(445, 180, "a + 0, a[:, [1]], a.clone()", anchor="start", size=12, font=MONO)
    S.text(445, 200, "new storage (copies)", anchor="start", size=12, color="var(--hi)")
    write("found-storage-views", S.figure(
        "<b>One storage, many views.</b> A tensor is a window on a flat block of memory: an offset, a shape and a "
        "stride per dimension. Reshaping, transposing and basic slicing only change those numbers; the highlighted "
        "column <code>a[:, 1]</code> is the storage elements 1, 5, 9.",
        alt="flat storage 0 to 11 viewed as a 3 by 4 grid, column 1 highlighted"))


# ---------------------------------------------------------------- slicing grids
def slicing():
    S = SVG(700, 380, font_size=11)
    cell = 30
    vals = [[10 * r + c for c in range(10)] for r in range(10)]
    hi1 = {(r, c): "var(--accent)" for r in range(0, 10, 2) for c in range(1, 8, 3)}
    rows2 = [0, 1, 3, 4, 7]
    hi2 = {(r, c): "var(--hi)" for r in rows2 for c in range(1, 5, 3)}
    grid(S, 20, 50, vals, cell, hi1, size=11)
    S.text(20, 14, "b = a[::2, 1:8:3]", anchor="start", font=MONO, size=13, weight=600, color="var(--accent)")
    S.text(20, 34, "rows 0, 2, 4, 6, 8 · cols 1, 4, 7 → (5, 3), view", anchor="start", size=12)
    grid(S, 370, 50, vals, cell, hi2, size=11)
    S.text(370, 14, "b = a[[0, 1, 3, 4, 7], 1:5:3]", anchor="start", font=MONO, size=13, weight=600, color="var(--hi)")
    S.text(370, 34, "listed rows · cols 1, 4 → (5, 2), copy", anchor="start", size=12)
    write("found-slicing-grid", S.figure(
        "<b>Lecture 2 examples on <code>a = torch.arange(100).view(10, 10)</code>.</b> Left: a slice with steps "
        "(a view). Right: a list of rows mixed with a slice of columns (a copy). Stops are excluded: "
        "<code>1:8:3</code> is 1, 4, 7 and <code>1:5:3</code> is 1, 4.",
        alt="two 10 by 10 grids with the selected cells highlighted"))


# ---------------------------------------------------------------- broadcasting
def broadcasting():
    S = SVG(700, 230, font_size=13)
    cell = 30
    A = [[1, 2, 3], [0, 1, 0]]
    # case 1: (2,3) + (3,)
    S.text(20, 18, "(2, 3) + (3,)", anchor="start", weight=600)
    grid(S, 20, 34, A, cell)
    S.text(118, 64, "+", size=18)
    grid(S, 136, 34, [[1, 2, 3]], cell, {(0, c): "var(--accent)" for c in range(3)})
    grid(S, 136, 64, [[1, 2, 3]], cell, ghost=True)
    S.text(234, 64, "=", size=18)
    grid(S, 252, 34, [[2, 4, 6], [1, 3, 3]], cell)
    S.text(20, 120, "row copied down each row", anchor="start", size=12, color="var(--muted)")
    # case 2: (2,3) + (2,1)
    x = 370
    S.text(x, 18, "(2, 3) + (2, 1)", anchor="start", weight=600)
    grid(S, x, 34, A, cell)
    S.text(x + 98, 64, "+", size=18)
    grid(S, x + 116, 34, [[2], [4]], cell, {(0, 0): "var(--hi)", (1, 0): "var(--hi)"})
    grid(S, x + 146, 34, [[2, 2], [4, 4]], cell, ghost=True)
    S.text(x + 214, 64, "=", size=18)
    grid(S, x + 232, 34, [[3, 4, 5], [4, 5, 4]], cell)
    S.text(x, 120, "column copied across each column", anchor="start", size=12, color="var(--muted)")
    # rule
    S.text(20, 160, "Rule: line shapes up from the right; each pair of sizes must be equal, or one is 1, or one is missing.",
           anchor="start", size=12)
    S.text(20, 182, "(2, 3) + (2,) fails: trailing sizes 3 and 2 differ. Fix: torch.tensor([2, 4]).unsqueeze(1) → (2, 1).",
           anchor="start", size=12, color="var(--hi)")
    S.text(20, 204, "Dashed cells are never stored: broadcasting stretches with stride 0, no copy.", anchor="start",
           size=12, color="var(--muted)")
    write("found-broadcasting", S.figure(
        "<b>Broadcasting, Lecture 2.</b> The smaller tensor is stretched (dashed) along its size-1 or missing "
        "dimensions to match <code>A</code>, then the operation is elementwise.",
        alt="adding a row vector and a column vector to a 2 by 3 matrix"))


# ---------------------------------------------------------------- matrix product
def matmul():
    S = SVG(620, 210, font_size=13)
    cell = 34
    A = [[3, 0, 1], [0, 2, 2]]
    B = [[4, 0], [1, 3], [5, 1]]
    C = [[17, 1], [12, 8]]
    grid(S, 20, 50, A, cell, {(0, c): "var(--accent)" for c in range(3)})
    S.text(20 + 1.5 * cell, 34, "A: (2 × 3)", weight=600)
    S.text(140, 84, "@", size=18)
    grid(S, 160, 33, B, cell, {(r, 0): "var(--hi)" for r in range(3)})
    S.text(160 + cell, 20, "B: (3 × 2)", weight=600)
    S.text(250, 84, "=", size=18)
    grid(S, 270, 50, C, cell, {(0, 0): "var(--accent2)"})
    S.text(270 + cell, 34, "C: (2 × 2)", weight=600)
    S.text(360, 60, "c₁₁ = row 1 of A · column 1 of B", anchor="start", size=12)
    S.text(360, 80, "= 3·4 + 0·1 + 1·5 = 17", anchor="start", size=12, font=MONO)
    S.text(360, 115, "(m × n) @ (n × p) → (m × p)", anchor="start", size=13, weight=600)
    S.text(360, 135, "inner sizes must match", anchor="start", size=12, color="var(--muted)")
    S.text(20, 175, "B @ A is (3 × 2)(2 × 3) = (3 × 3): a different matrix. AB ≠ BA in general.", anchor="start",
           size=12, color="var(--muted)")
    write("found-matmul", S.figure(
        "<b>Matrix product, Lecture 2 example.</b> Entry <i>c<sub>ij</sub></i> is the dot product of row <i>i</i> "
        "of <i>A</i> with column <i>j</i> of <i>B</i>.",
        alt="2 by 3 times 3 by 2 equals 2 by 2, entry 17 highlighted"))


# ---------------------------------------------------------------- sigmoid variants
def sigmoid():
    x = np.linspace(-7, 7, 400)
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 2.9))
    for tau, c in [(0.25, HI), (1, ACCENT), (4, ACCENT2)]:
        axs[0].plot(x, 1 / (1 + np.exp(-x / tau)), color=c, label=rf"$\tau={tau:g}$")
    axs[0].set_title(r"$\sigma(x/\tau)$: temperature", fontsize=11)
    for b, c in [(-1, HI), (0, ACCENT), (1, ACCENT2)]:
        axs[1].plot(x, 1 / (1 + np.exp(-(x - b))), color=c, label=rf"$b={b:g}$")
    axs[1].set_title(r"$\sigma(x-b)$: shift", fontsize=11)
    for ax in axs:
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.grid(True)
        ax.set_xlabel("$x$")
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    save(fig, "found-sigmoid", "<b>Lecture 2 exercise.</b> Smaller τ sharpens the step, larger τ stretches it; "
         "<i>b</i> moves the point where σ = 0.5. τ = 1, <i>b</i> = 0 is the ordinary sigmoid.",
         alt="sigmoid curves for three temperatures and three shifts")


# ---------------------------------------------------------------- norm balls
def norm_balls():
    t = np.linspace(0, 2 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot(np.cos(t), np.sin(t), color=ACCENT, label=r"$\Vert v\Vert_2=1$")
    ax.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color=HI, label=r"$\Vert v\Vert_1=1$")
    ax.plot([1, -1, -1, 1, 1], [1, 1, -1, -1, 1], color=ACCENT2, label=r"$\Vert v\Vert_\infty=1$")
    ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.6)
    ax.set_xlabel("$v_1$")
    ax.set_ylabel("$v_2$")
    ax.legend(loc="upper center", fontsize=9, ncol=3, bbox_to_anchor=(0.5, 1.15), handlelength=1.2,
              columnspacing=0.8)
    save(fig, "found-norm-balls", "<b>Unit balls in 2-D.</b> All vectors of norm 1: a circle for "
         "<i>L</i><sub>2</sub>, a diamond for <i>L</i><sub>1</sub>, a square for <i>L</i><sub>∞</sub>.",
         alt="unit circle, diamond and square", width=360)


# ---------------------------------------------------------------- ML taxonomy
def taxonomy():
    S = SVG(700, 230, font_size=13)

    def box(x, y, w, h, title, sub=None, color="currentColor"):
        S.rect(x, y, w, h, rx=6, stroke=color, width=1.6)
        S.text(x + w / 2, y + (h / 2 if not sub else 18), title, weight=600, color=color)
        if sub:
            for i, s in enumerate(sub):
                S.text(x + w / 2, y + 38 + 17 * i, s, size=12, color="var(--muted)")

    box(270, 10, 160, 36, "Machine learning")
    box(40, 90, 270, 120, "Supervised: labels yᵢ known", ["classification (cat vs dog)", "regression (price, rating)",
                                                          "speech-to-text, recommendation"], color="var(--accent)")
    box(390, 90, 270, 120, "Unsupervised: no labels", ["clustering (how many speakers?)", "generation (new faces)",
                                                       "self-supervised proxy tasks"], color="var(--hi)")
    S.path("M350 46 L175 88", arrow=True)
    S.path("M350 46 L525 88", arrow=True)
    write("found-ml-taxonomy", S.figure(
        "<b>The two settings of Lecture 1.</b> Supervised problems learn from pairs (<i>x<sub>i</sub></i>, "
        "<i>y<sub>i</sub></i>); unsupervised ones only see <i>x<sub>i</sub></i> and rely on assumptions about its "
        "structure. This course is mostly supervised.",
        alt="machine learning splits into supervised and unsupervised, with examples"))


if __name__ == "__main__":
    cat_stack()
    storage_views()
    slicing()
    broadcasting()
    matmul()
    sigmoid()
    norm_balls()
    taxonomy()
    print("wrote", sorted(f for f in os.listdir(FIGS) if f.startswith("found-")))
