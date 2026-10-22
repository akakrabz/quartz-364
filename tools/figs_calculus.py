#!/usr/bin/env python3
"""Figures for unit 2 (calculus and optimization). Writes tools/figs/calc-*.html.

    python3 tools/figs_calculus.py && python3 tools/assemble.py content/2-calculus-and-optimization/*.md
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mplfig import ACCENT, ACCENT2, HI, INK, MUTED, plt, save  # noqa: E402
from figlib import SVG, SERIF_IT  # noqa: E402

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")


def write(name, html):
    with open(os.path.join(FIGS, name + ".html"), "w") as f:
        f.write(html + "\n")


# ------------------------------------------------------------------ plots

def gradient_field():
    f = lambda x, y: x**2 + 2 * y**2
    xs = np.linspace(-2.2, 2.2, 200)
    X, Y = np.meshgrid(xs, xs)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.contour(X, Y, f(X, Y), levels=[0.25, 1, 2, 3.5, 5.5, 8], colors=MUTED, linewidths=1)
    pts = np.array([[1.5, 0.5], [-1, 1], [0.5, -1.2], [-1.6, -0.6], [1.2, 1.3], [-0.4, 0.4]])
    g = np.stack([2 * pts[:, 0], 4 * pts[:, 1]], 1)
    ax.quiver(pts[:, 0], pts[:, 1], g[:, 0], g[:, 1], color=HI, angles="xy", scale_units="xy", scale=8, width=0.006)
    ax.quiver(pts[:, 0], pts[:, 1], -g[:, 0], -g[:, 1], color=ACCENT, angles="xy", scale_units="xy", scale=8, width=0.006)
    ax.plot(pts[:, 0], pts[:, 1], "o", color=INK, ms=3)
    ax.plot([0], [0], "*", color=ACCENT2, ms=11)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(r"$f(x,y)=x^2+2y^2$", fontsize=11)
    save(fig, "calc-gradient-field",
         "Level curves of <i>f</i>(<i>x</i>, <i>y</i>) = <i>x</i>² + 2<i>y</i>² with ∇<i>f</i> = (2<i>x</i>, 4<i>y</i>) "
         "(<b>red</b>, steepest ascent) and −∇<i>f</i> (<b>blue</b>, steepest descent) at a few points. "
         "The gradient is perpendicular to the level curve through its point; the star is the minimum, where ∇<i>f</i> = 0.",
         alt="Contour plot with gradient arrows", width=420)


def critical_points():
    x = np.linspace(-1.5, 1.5, 200)
    fig, axs = plt.subplots(1, 3, figsize=(8, 2.4), sharey=True)
    for ax, y, t, lab in zip(axs, [x**2, -x**2, x**3], ["$x^2$", "$-x^2$", "$x^3$"],
                             ["minimum, $f''=2>0$", "maximum, $f''=-2<0$", "inflection, $f''=0$"]):
        ax.plot(x, y, color=ACCENT)
        ax.plot([0], [0], "o", color=HI, ms=6)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(t + ": " + lab, fontsize=10)
        ax.set_xticks([-1, 0, 1])
        ax.set_ylim(-2.4, 2.4)
    save(fig, "calc-critical-points",
         "All three functions have <i>f</i>′(0) = 0, so <i>x</i> = 0 is a critical (stationary) point of each, "
         "but it is a minimum, a maximum and an inflection point respectively. The second-derivative test tells them apart "
         "except when <i>f</i>″ = 0.", alt="x squared, minus x squared, x cubed", width=720)


def convexity():
    x = np.linspace(-2, 2, 400)
    g = lambda x: x**2 + 2 * np.cos(np.pi * x) + x + 2
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8))
    ax = axs[0]
    ax.plot(x, x**2, color=ACCENT)
    a, b = -1.5, 1.0
    ax.plot([a, b], [a**2, b**2], "--", color=HI, lw=1.6)
    ax.plot([a, b], [a**2, b**2], "o", color=HI, ms=5)
    ax.set_title("convex: every chord lies above", fontsize=10)
    ax = axs[1]
    ax.plot(x, g(x), color=ACCENT)
    a, b = -0.954, 0.858
    ax.plot([a, b], [g(a), g(b)], "--", color=HI, lw=1.6)
    ax.plot([a, b], [g(a), g(b)], "o", color=HI, ms=5)
    ax.set_title("not convex: this chord dips below", fontsize=10)
    for ax in axs:
        ax.set_xlabel("$x$")
    save(fig, "calc-convexity",
         "Convexity as chords: <i>f</i>(λ<i>x</i>₁ + (1 − λ)<i>x</i>₂) ≤ λ<i>f</i>(<i>x</i>₁) + (1 − λ)<i>f</i>(<i>x</i>₂). "
         "Left: <i>x</i>², convex. Right: Lecture 4's <i>x</i>² + 2cos(π<i>x</i>) + <i>x</i> + 2, whose two local minima "
         "(red dots) are joined by a chord that passes <i>under</i> the bump between them, so it is not convex.",
         alt="Convex and non-convex functions with chords", width=680)


def gd_step_sizes():
    f = lambda x: 2 * x**2 - 6 * x + 5
    fp = lambda x: 4 * x - 6
    alphas = [(0.1, "α = 1/10: slow, monotone"), (1 / 3, "α = 1/3: zig-zags in"),
              (0.5, "α = 1/2: bounces 0 ↔ 3"), (0.6, "α = 3/5: diverges")]
    fig, axs = plt.subplots(1, 4, figsize=(10, 2.7), sharey=True)
    xs = np.linspace(-2.5, 5.5, 300)
    for ax, (a, t) in zip(axs, alphas):
        ax.plot(xs, f(xs), color=MUTED, lw=1.5)
        x = 0.0
        tr = [x]
        for _ in range(8 if a != 0.6 else 4):
            x = x - a * fp(x)
            tr.append(x)
        tr = np.array(tr)
        ax.plot(tr, f(tr), "-o", color=HI if a >= 0.5 else ACCENT, ms=4, lw=1.2)
        ax.plot([1.5], [f(1.5)], "*", color=ACCENT2, ms=10)
        ax.set_title(t, fontsize=10)
        ax.set_xlim(-2.5, 5.5)
        ax.set_ylim(-1, 30)
        ax.set_xlabel("$x$")
    save(fig, "calc-gd-step-sizes",
         "Gradient descent on <i>f</i>(<i>x</i>) = 2<i>x</i>² − 6<i>x</i> + 5 from <i>x</i>⁽⁰⁾ = 0 (HW 2 Q1). "
         "Here <i>f</i>″ = 4, so each step multiplies the distance to <i>x</i>* = 3/2 (star) by 1 − 4α: "
         "0.6, −1/3, −1 and −1.4 from left to right. It converges exactly when 0 &lt; α &lt; 1/2.",
         alt="Four gradient descent runs with different step sizes", width=900)


def gd_nonconvex():
    g = lambda x: x**2 + 2 * np.cos(np.pi * x) + x + 2
    gp = lambda x: 2 * x - 2 * np.pi * np.sin(np.pi * x) + 1
    xs = np.linspace(-4, 4.5, 500)
    fig, ax = plt.subplots(figsize=(6.4, 3))
    ax.plot(xs, g(xs), color=MUTED, lw=1.6)
    for x0, c in [(4.0, HI), (2.0, ACCENT2), (0.0, ACCENT), (-3.4, INK)]:
        x, tr = x0, [x0]
        for _ in range(30):
            x = x - gp(x) / 20
            tr.append(x)
        tr = np.array(tr)
        ax.plot(tr, g(tr), "-o", color=c, ms=3, lw=1, label=f"$x^{{(0)}}={x0:g}$ → {tr[-1]:.2f}")
    ax.legend(fontsize=9, loc="upper center")
    ax.set_xlabel("$x$")
    ax.set_ylim(-1, 30)
    save(fig, "calc-gd-nonconvex",
         "Lecture 4's non-convex <i>f</i>(<i>x</i>) = <i>x</i>² + 2cos(π<i>x</i>) + <i>x</i> + 2 with α = 1/20 from four starting points. "
         "Each run settles in the local minimum downhill from where it started; only the run from 0 finds the global "
         "minimum near −0.95.", alt="Gradient descent getting stuck in local minima", width=600)


def gd_contour():
    f = lambda x, y: (x + 3 * y) ** 2 + (x - 2) ** 2
    grad = lambda p: np.array([2 * (p[0] + 3 * p[1]) + 2 * (p[0] - 2), 6 * (p[0] + 3 * p[1])])
    X, Y = np.meshgrid(np.linspace(-0.6, 3.2, 300), np.linspace(-1.6, 0.6, 300))
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, a, n, c, t in [(axs[0], 1 / 16, 25, ACCENT, "α = 1/16: converges"),
                           (axs[1], 1 / 8, 7, HI, "α = 1/8: two steps fine, then diverges")]:
        ax.contour(X, Y, f(X, Y), levels=[0.1, 0.5, 1.5, 3, 5, 8, 12], colors=MUTED, linewidths=0.9)
        p = np.zeros(2)
        tr = [p.copy()]
        for _ in range(n):
            p = p - a * grad(p)
            tr.append(p.copy())
        tr = np.array(tr)
        ax.plot(tr[:, 0], tr[:, 1], "-o", color=c, ms=3, lw=1.2)
        ax.plot([2], [-2 / 3], "*", color=ACCENT2, ms=11)
        ax.set_xlim(-0.6, 3.2)
        ax.set_ylim(-1.6, 0.6)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("$x$")
    axs[0].set_ylabel("$y$")
    save(fig, "calc-gd-contour",
         "Fall 2024 Q3's <i>f</i>(<i>x</i>, <i>y</i>) = (<i>x</i> + 3<i>y</i>)² + (<i>x</i> − 2)² from (0, 0). The valley is long and "
         "narrow (Hessian eigenvalues ≈ 1.8 and 20.2), so steps zig-zag across it. GD converges only for α &lt; 2/20.2 ≈ 0.099. "
         "The exam's α = 1/8 gives the correct two steps (1/2, 0) and (3/4, −3/8), but continued it would diverge "
         "(the path leaves the plot after a few more steps).", alt="Gradient descent paths on a narrow valley", width=760)


def fit_quadratic():
    a, b, c = 3, 2, 1
    N = 20
    x = np.linspace(-2, 2, N)
    y = a * x**2 + b * x + c
    p = np.array([-1.0, 0.2, 0.08])
    alpha = 3e-2
    hist, loss = [p.copy()], []
    for _ in range(100):
        e = y - (p[0] * x**2 + p[1] * x + p[2])
        g = np.array([np.sum(-2 * x**2 * e), np.sum(-2 * x * e), np.sum(-2 * e)]) / N
        loss.append(np.mean(e**2))
        p = p - alpha * g
        hist.append(p.copy())
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.9))
    axs[0].semilogy(loss, color=ACCENT)
    axs[0].set_xlabel("iteration")
    axs[0].set_ylabel("MSE")
    axs[0].grid(True)
    axs[1].scatter(x, y, color=HI, s=14, zorder=3, label="data")
    for i, col in [(0, MUTED), (5, ACCENT2), (25, INK), (100, ACCENT)]:
        q = hist[i]
        axs[1].plot(x, q[0] * x**2 + q[1] * x + q[2], color=col, lw=1.6, label=f"iteration {i}")
    axs[1].legend(fontsize=8)
    axs[1].set_xlabel("$x$")
    save(fig, "calc-fit-quadratic",
         f"Lecture 6's first learning problem: fit <i>ax</i>² + <i>bx</i> + <i>c</i> to 20 points from 3<i>x</i>² + 2<i>x</i> + 1 by gradient "
         f"descent on the MSE (α = 0.03, start (−1, 0.2, 0.08)). After 100 steps (a, b, c) ≈ "
         f"({hist[-1][0]:.2f}, {hist[-1][1]:.2f}, {hist[-1][2]:.2f}): <i>b</i> is learned fastest, <i>a</i> and <i>c</i> trade off slowly.",
         alt="Loss curve and fitted parabolas", width=760)
    return hist[-1]


# ------------------------------------------------------------------ graphs

def sub(name, idx, bar=False):
    """Italic z with subscript, optional bar, as SVG markup."""
    b = "̄" if bar else ""
    return f'<tspan font-style="italic">{name}{b}</tspan><tspan dy="4" font-size="10">{idx}</tspan><tspan dy="-4"></tspan>'


def graph_figure(name, W, H, nodes, edges, inputs, output, caption, alt, notes=()):
    """nodes: {key: (x, y, label_markup, above_markup, below_markup)}; edges: (a, b, label, side)."""
    s = SVG(W, H, font_size=14)
    r = 21
    for a, b, lab, (dx, dy) in edges:
        x1, y1 = nodes[a][:2]
        x2, y2 = nodes[b][:2]
        d = np.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / d, (y2 - y1) / d
        s.line(x1 + ux * r, y1 + uy * r, x2 - ux * (r + 2), y2 - uy * (r + 2), arrow=True)
        if lab:
            mx, my = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
            s.text(mx, my, lab, size=12, color="var(--hi)", halo=True, raw=True)
    for key, (x, y, lab, above, below) in nodes.items():
        s.circle(x, y, r, fill="var(--paper)", stroke="currentColor", width=1.6)
        s.text(x, y, lab, size=14, raw=True)
        if above:
            s.text(x, y - r - 12, above, size=12, color="var(--accent)", raw=True, halo=True)
        if below:
            s.text(x, y + r + 13, below, size=12, raw=True, halo=True)
    for key, lab in inputs:
        x, y = nodes[key][:2]
        s.line(x - r - 46, y, x - r - 2, y, arrow=True)
        s.text(x - r - 54, y, lab, size=14, anchor="end", raw=True)
    key, lab = output
    x, y = nodes[key][:2]
    s.line(x + r, y, x + r + 40, y, arrow=True)
    s.text(x + r + 46, y, lab, size=14, anchor="start", raw=True)
    for (x, y, t, col) in notes:
        s.text(x, y, t, size=12, color=col, raw=True, anchor="start")
    html = s.figure(caption, alt=alt)
    # own marker: context-stroke is unreliable, currentColor follows the theme
    html = html.replace('id="ARR"', f'id="{name}-arr"').replace("url(#ARR)", f"url(#{name}-arr)")
    html = html.replace('fill="context-stroke"', 'fill="currentColor"')
    write(name, html)


def graph_g():
    e2 = np.exp(2)
    z = lambda i: sub("z", i)
    zb = lambda i: sub("z", i, True)
    nodes = {
        1: (110, 70, z(1), f"{zb(1)} = 6e² ≈ {6*e2:.2f}", "1"),
        2: (110, 200, z(2), f"{zb(2)} = 3e² ≈ {3*e2:.2f}", "2"),
        3: (270, 135, z(3), f"{zb(3)} = 3e² ≈ {3*e2:.2f}", "2"),
        4: (430, 70, z(4), f"{zb(4)} = {z(3)} = 2", f"e² ≈ {e2:.2f}"),
        5: (560, 170, z(5), f"{zb(5)} = 1", f"2e² ≈ {2*e2:.2f}"),
    }
    edges = [
        (1, 3, f"{z(2)}", (4, -14)),
        (2, 3, f"{z(1)}", (4, 14)),
        (3, 4, f"e<tspan dy='-5' font-size='9'>z₃</tspan><tspan dy='5'></tspan>", (-14, -10)),
        (3, 5, f"{z(4)}", (0, 14)),
        (4, 5, f"{z(3)}", (14, -8)),
    ]
    graph_figure(
        "calc-graph-g", 700, 250, nodes, edges,
        [(1, '<tspan font-style="italic">x</tspan> = 1'), (2, '<tspan font-style="italic">y</tspan> = 2')],
        (5, '<tspan font-style="italic">g</tspan>'),
        "Computational graph of <i>g</i>(<i>x</i>, <i>y</i>) = <i>xy</i>e<sup><i>xy</i></sup> (Lecture 5 exercise): "
        "<i>z</i>₃ = <i>z</i>₁<i>z</i>₂, <i>z</i>₄ = e<sup><i>z</i>₃</sup>, <i>z</i>₅ = <i>z</i>₃<i>z</i>₄. Numbers below the nodes: "
        "forward pass at (1, 2). <b style='color:var(--hi)'>Red</b>: local partial on each arrow (child w.r.t. parent). "
        "<b style='color:var(--accent)'>Blue</b>: adjoints from the backward pass. <i>z</i>₃ has two successors, so "
        "<i>z̄</i>₃ = <i>z̄</i>₄e<sup><i>z</i>₃</sup> + <i>z̄</i>₅<i>z</i>₄ = 2e² + e².",
        alt="Computational graph of x y e^(xy) with values and adjoints")


def graph_l12():
    z = lambda i: sub("z", i)
    nodes = {
        1: (90, 50, z(1), "", ""),
        2: (90, 190, z(2), "", ""),
        3: (220, 120, z(3), "", f"{z(1)}{z(2)}"),
        4: (350, 120, z(4), "", f"cos {z(3)}"),
        5: (470, 50, z(5), "", ""),
        6: (590, 50, z(6), "", ""),
        7: (530, 190, z(7), "", ""),
        8: (680, 120, z(8), "", ""),
    }
    edges = [(1, 3, "", (0, 0)), (2, 3, "", (0, 0)), (3, 4, "", (0, 0)),
             (4, 5, "", (0, 0)), (5, 6, "", (0, 0)), (6, 8, "", (0, 0)), (4, 7, "", (0, 0)), (7, 8, "", (0, 0))]
    graph_figure(
        "calc-graph-l12", 820, 240, nodes, edges,
        [(1, '<tspan font-style="italic">x</tspan>'), (2, '<tspan font-style="italic">y</tspan>')],
        (8, '<tspan font-style="italic">f</tspan>'),
        "The Lecture 12 review graph for <i>f</i>(<i>x</i>, <i>y</i>) = 2cos²(<i>xy</i>) + ln(cos(<i>xy</i>)). "
        "The original has no operation labels, so reading them off is part of the problem: <i>z</i>₅ = <i>z</i>₄², <i>z</i>₆ = 2<i>z</i>₅, "
        "<i>z</i>₇ = ln <i>z</i>₄, <i>z</i>₈ = <i>z</i>₆ + <i>z</i>₇.",
        alt="Computational graph with eight nodes",
        notes=[(450, 100, f"{z(5)} = {z(4)}²", "currentColor"), (575, 100, f"{z(6)} = 2{z(5)}", "currentColor"),
               (560, 222, f"{z(7)} = ln {z(4)}", "currentColor"), (700, 160, f"{z(6)} + {z(7)}", "currentColor")])


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    gradient_field()
    critical_points()
    convexity()
    gd_step_sizes()
    gd_nonconvex()
    gd_contour()
    print("quadratic fit final:", fit_quadratic())
    graph_g()
    graph_l12()
    print("calc figures written")
