#!/usr/bin/env python3
"""Figures for content/6-midterm-1/: past-exam computational graphs (redrawn),
SVM point sets, and two schematic plots. Run: python3 tools/figs_exam.py
Then: python3 tools/assemble.py content/6-midterm-1/*.md
"""
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

from figlib import Automaton, SERIF_IT, SANS, fmt  # noqa: E402
from mplfig import plt, INK, ACCENT, ACCENT2, HI, MUTED, save  # noqa: E402

FIGS = os.path.join(HERE, "figs")
UPRIGHT = ("sin", "cos", "ln", "exp", "log")
MATH = "KaTeX_Math,'STIX Two Math','Latin Modern Math','Cambria Math','Times New Roman',serif"


def mathlabel(s, size=14):
    """Tiny markup -> SVG tspans: x^{2}, w_{2}, e^{a_{1}}; sin/ln upright."""
    out, base = [], 0  # base: current dy offset

    def word(t, italic=True, fs=size, dy=0):
        st = f"font-size:{fs}px;font-family:{MATH if italic else SANS};" + ("font-style:italic;" if italic else "")
        d = f' dy="{dy}"' if dy else ""
        return f'<tspan style="{st}"{d}>{t}</tspan>'

    tokens = re.findall(r"\^\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|_\{[^{}]*\}|[A-Za-z]+|\d+|.", s)
    cur = 0
    for tk in tokens:
        if tk.startswith("^{"):
            inner = tk[2:-1]
            sub = re.match(r"^([a-z])_\{(\d+)\}$", inner)
            if sub:  # e^{a_1}
                out.append(word(sub.group(1), fs=size - 3, dy=-6 - cur)); cur = -6
                out.append(word(sub.group(2), italic=False, fs=size - 5, dy=3)); cur = -3
            else:
                out.append(word(inner, italic=not inner.isdigit(), fs=size - 3, dy=-6 - cur)); cur = -6
        elif tk.startswith("_{"):
            out.append(word(tk[2:-1], italic=False, fs=size - 3, dy=4 - cur)); cur = 4
        else:
            dy = -cur if cur else 0
            cur = 0
            if tk in UPRIGHT:
                out.append(word(tk, italic=False, dy=dy))
            elif tk.isalpha():
                out.append(word(tk, dy=dy))
            else:
                t = tk.replace("-", "−")
                out.append(word(t.replace("&", "&amp;").replace("<", "&lt;"), italic=False, dy=dy))
    return "".join(out)


class Graph(Automaton):
    """Computational graph: nodes, arrows, value labels on arrows."""

    def __init__(self, w, h, r=19):
        super().__init__(w, h, r=r, font_size=14)
        self.labels = []
        self.extra = []

    def node(self, name, x, y, lab):
        self.state(name, x, y, label="")
        self.labels.append((x, y + 1, lab, 14))

    def arrow(self, a, b, lab="", side=1, dist=12, pos=0.5, bend=0, dx=0, dy=0):
        self.edge(a, b, label="", bend=bend)
        if lab:
            A, B = self.states[a], self.states[b]
            x1, y1, x2, y2 = A["x"], A["y"], B["x"], B["y"]
            L = math.hypot(x2 - x1, y2 - y1)
            nx, ny = (y2 - y1) / L, -(x2 - x1) / L  # left normal (screen)
            px, py = x1 + (x2 - x1) * pos, y1 + (y2 - y1) * pos
            if bend:
                px, py = px - nx * bend, py - ny * bend
            self.labels.append((px + nx * dist * side + dx, py + ny * dist * side + dy, lab, 13))

    def inp(self, name, text, length=55, side="left"):
        st = self.states[name]
        if side == "left":
            x0 = st["x"] - st["r"] - length
            self.line(x0 + 12, st["y"], st["x"] - st["r"] - 2.5, st["y"], arrow=True)
            self.labels.append((x0, st["y"], text, 15))

    def out(self, name, text, dx=70, dy=0, lab="", labdy=-15, textdx=24):
        st = self.states[name]
        L = math.hypot(dx, dy)
        sx, sy = st["x"] + dx / L * st["r"], st["y"] + dy / L * st["r"]
        ex, ey = st["x"] + dx, st["y"] + dy
        self.line(sx, sy, ex, ey, arrow=True)
        self.labels.append((ex + textdx, ey, text, 15))
        if lab:
            self.labels.append(((sx + ex) / 2, (sy + ey) / 2 + labdy, lab, 13))

    def svg(self, alt=""):
        s = super().svg(alt)
        # halo copy first, so later glyphs' halos never cover earlier glyphs
        txt = "".join(
            f'<text x="{fmt(x)}" y="{fmt(y)}" text-anchor="middle" dominant-baseline="central" fill="var(--paper)" '
            f'stroke="var(--paper)" stroke-width="5" stroke-linejoin="round" aria-hidden="true">{mathlabel(l, fs)}</text>'
            f'<text x="{fmt(x)}" y="{fmt(y)}" text-anchor="middle" dominant-baseline="central" fill="currentColor">'
            f'{mathlabel(l, fs)}</text>'
            for x, y, l, fs in self.labels)
        return s.replace("</svg>", txt + "</svg>")


def write(name, html):
    with open(os.path.join(FIGS, name + ".html"), "w") as f:
        f.write(html.replace("\n", " ") + "\n")


# ---------------------------------------------------------------- Fall 2024 Q4
g = Graph(640, 260)
g.node("w1", 95, 130, "w_{1}")
g.node("w2", 195, 130, "w_{2}")
g.node("w3", 320, 50, "w_{3}")
g.node("w4", 320, 210, "w_{4}")
g.node("w5", 450, 50, "w_{5}")
g.node("w6", 450, 130, "w_{6}")
g.inp("w1", "x")
g.arrow("w1", "w2", "w_{1}")
g.arrow("w2", "w3", "sin(w_{1})", dist=16)
g.arrow("w2", "w6", "sin(w_{1})", pos=0.55)
g.arrow("w2", "w4", "sin(w_{1})", side=-1, dist=16)
g.arrow("w3", "w5", "w_{2}^{2}")
g.arrow("w5", "w6", "w_{3} + 364", side=-1, dist=38)
g.arrow("w4", "w6", "e^{w_{2}}", side=-1, dist=16)
g.out("w6", "f(x)", dx=125, lab="w_{2} + w_{4} + w_{5}", textdx=24)
write("exam-f24-graph", g.figure(
    "<strong>Fall 2024 Q4</strong>, redrawn. The label on an arrow is the value of the node the arrow leaves.",
    alt="Computational graph w1 to w6 for f(x) = sin x + e^(sin x) + sin^2 x + 364"))

# ---------------------------------------------------------------- Fall 2025 Q4
g = Graph(640, 330)
g.node("a1", 150, 170, "a_{1}")
g.node("a2", 150, 55, "a_{2}")
g.node("a3", 295, 170, "a_{3}")
g.node("a5", 295, 55, "a_{5}")
g.node("a4", 295, 285, "a_{4}")
g.node("a6", 435, 170, "a_{6}")
g.inp("a1", "x", length=75)
g.arrow("a1", "a2", "a_{1}", dist=14)
g.arrow("a1", "a3", "a_{1}")
g.arrow("a1", "a4", "a_{1}", side=-1, dist=14)
g.arrow("a2", "a5", "a_{1}^{2}")
g.arrow("a3", "a5", "e^{a_{1}}", side=-1, dist=18)
g.arrow("a3", "a4", "e^{a_{1}}", side=-1, dist=18)
g.arrow("a3", "a6", "e^{a_{1}}")
g.arrow("a5", "a6", "a_{2}a_{3}", dist=20)
g.arrow("a4", "a6", "a_{1}a_{3}", side=-1, dist=20)
g.out("a6", "f(x)", dx=125, lab="a_{3} + a_{4} + a_{5}", textdx=24)
write("exam-f25-graph", g.figure(
    "<strong>Fall 2025 Q4</strong>, redrawn. Each arrow carries the value of the node it leaves: "
    "a₂ → a₅ is labelled a₁², so a₂ = a₁².",
    alt="Computational graph a1 to a6 for f(x) = x^2 e^x + x e^x + e^x"))

# ---------------------------------------------------------------- Spring 2025 Q8
g = Graph(680, 260)
g.node("w1", 95, 130, "w_{1}")
g.node("w2", 195, 130, "w_{2}")
g.node("w3", 300, 45, "w_{3}")
g.node("w4", 430, 45, "w_{4}")
g.node("w5", 360, 215, "w_{5}")
g.node("w6", 480, 130, "w_{6}")
g.inp("w1", "x")
g.arrow("w1", "w2", "w_{1}")
g.arrow("w2", "w3", "sin(w_{1})", dist=16)
g.arrow("w3", "w4", "(w_{2})^{3}")
g.arrow("w4", "w6", "4w_{3}", side=-1, dist=20)
g.arrow("w2", "w6", "sin(w_{1})", side=-1, pos=0.5)
g.arrow("w2", "w5", "sin(w_{1})", side=-1, dist=18)
g.arrow("w5", "w6", "ln(w_{2})", side=-1, dist=22)
g.out("w6", "f(x)", dx=125, lab="w_{2} − w_{4} + w_{5}", textdx=24)
write("exam-sp25-graph", g.figure(
    "<strong>Spring 2025 Q8</strong>, redrawn. Note the minus sign on w₄ in the output label: ∂w₆/∂w₄ = −1.",
    alt="Computational graph w1 to w6 for f(x) = sin x - 4 sin^3 x + ln(sin x)"))

# ---------------------------------------------------------------- Spring 2026 Q9
g = Graph(560, 280)
for i, (x, y) in enumerate([(110, 60), (110, 145), (110, 230)], 1):
    g.node(f"w{i}", x, y, f"w_{{{i}}}")
for i, (x, y) in enumerate([(210, 60), (210, 145), (210, 230)], 4):
    g.node(f"w{i}", x, y, f"w_{{{i}}}")
g.node("w7", 310, 145, "w_{7}")
g.node("w8", 410, 145, "w_{8}")
g.node("w9", 410, 55, "w_{9}")
for i, v in zip((1, 2, 3), "xyz"):
    g.inp(f"w{i}", v)
g.arrow("w1", "w4")
g.arrow("w2", "w5")
g.arrow("w3", "w6")
g.arrow("w4", "w7")
g.arrow("w5", "w7")
g.arrow("w6", "w7")
g.arrow("w7", "w8")
g.arrow("w8", "w9")
g.edge("w1", "w9", bend=22)
g.out("w9", "g(x, y, z)", dx=80, dy=60, textdx=30)
write("exam-sp26-graph", g.figure(
    "<strong>Spring 2026 Q9</strong>, redrawn: log-softmax g = x − ln(eˣ + eʸ + eᶻ). "
    "The arrow w₁ → w₉ is why w̄₁ collects two terms.",
    alt="Computational graph w1 to w9 for log-softmax"))

# ---------------------------------------------------------------- SVM point sets


def svm_plot(name, X, O, lines, caption, alt, title=None, removed=()):
    fig, ax = plt.subplots(figsize=(4.4, 5.0))
    xs = np.linspace(0, 9.5, 2)
    for (b, style, lab, col) in lines:
        ax.plot(xs, xs + b, style, color=col, lw=1.6, label=lab)
    ax.scatter(*zip(*X), marker="x", s=70, color=ACCENT, linewidths=2.2, label="class ×", zorder=3)
    ax.scatter(*zip(*O), marker="o", s=60, facecolors="none", edgecolors=HI, linewidths=2, label="class ○", zorder=3)
    if removed:
        ax.scatter(*zip(*removed), marker="s", s=120, facecolors="none", edgecolors=MUTED, linewidths=1.2,
                   linestyle="--", zorder=2, label="removed")
    ax.set_xlim(0, 9.5); ax.set_ylim(0, 9.5)
    ax.set_xticks(range(0, 10)); ax.set_yticks(range(0, 10))
    ax.grid(True, lw=0.6); ax.set_aspect("equal")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=9, ncol=2, handlelength=1.6)
    if title:
        ax.set_title(title, fontsize=11)
    save(fig, name, caption, alt=alt, width=420)


X25 = [(1, 4), (2, 3), (2, 9), (3, 4), (4, 7), (6, 7)]
O25 = [(3, 2), (4, 3), (5, 3), (6, 1), (7, 4), (8, 7)]
svm_plot("exam-sp25-svm", X25, O25, [],
         "<strong>Spring 2025 Q4</strong>: the twelve points, redrawn.", alt="SVM points, Spring 2025")
svm_plot("exam-sp25-svm-sol", X25, O25,
         [(0, "-", "boundary  y = x", INK), (1, "--", "margin  y = x + 1", ACCENT), (-1, "--", "margin  y = x − 1", HI)],
         "Max-margin solution: boundary y = x, margins y = x ± 1 (width √2). Support vectors are the points on the dashed lines.",
         alt="SVM solution, Spring 2025")
X26 = [(1, 4), (2, 3), (2, 9), (3, 4), (4, 7)]
O26 = [(4, 3), (5, 3), (6, 1), (7, 4), (8, 7)]
svm_plot("exam-sp26-svm", X26, O26, [],
         "<strong>Spring 2026 Q5</strong>: the ten points, redrawn.", alt="SVM points, Spring 2026")
svm_plot("exam-sp26-svm-sol", X26, O26,
         [(0, "-", "boundary  y = x", INK), (1, "--", "margin  y = x + 1", ACCENT), (-1, "--", "margin  y = x − 1", HI)],
         "(a) and (b): boundary y = x with margins y = x ± 1. Removing (8, 7) leaves (4, 3) on the ○ margin, so nothing moves.",
         alt="SVM solution, Spring 2026")
svm_plot("exam-sp26-svm-c", [p for p in X26 if p not in [(2, 3), (3, 4)]], O26,
         [(1, "-", "boundary  y = x + 1", INK), (3, "--", "margin  y = x + 3", ACCENT), (-1, "--", "margin  y = x − 1", HI)],
         "(c): without (2, 3) and (3, 4) the × margin moves out to (1, 4) and (4, 7); the boundary shifts toward the ×'s, "
         "to y = x + 1 (width doubles to 2√2).",
         alt="SVM after removing two points, Spring 2026", removed=[(2, 3), (3, 4)])

# ---------------------------------------------------------------- Fall 2025 Q5(c): schematic overfit
rng = np.random.default_rng(2)
xd = np.sort(np.concatenate([rng.uniform(-2, -1.7, 4), rng.uniform(-1.2, 0.6, 9), rng.uniform(1.3, 2, 8)]))
yd = 5 * np.sin(2 * xd) - 3 * xd ** 2 + rng.normal(0, 1.5, xd.size)
deg = 16


def ridge(lam):
    Phi = np.vander(xd / 2, deg + 1)
    return np.linalg.solve(Phi.T @ Phi + lam * np.eye(deg + 1), Phi.T @ yd)


xs = np.linspace(-2, 2, 400)
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.scatter(xd, yd, s=18, color=ACCENT, zorder=3, label="data")
ax.plot(xs, np.vander(xs / 2, deg + 1) @ ridge(1e-10), "--", color=HI, lw=1.6, label="small λ: overfit")
ax.plot(xs, np.vander(xs / 2, deg + 1) @ ridge(3.0), color=INK, lw=1.6, label="larger λ: smooth")
ax.set_ylim(-35, 25); ax.set_xlabel("$x$"); ax.grid(True, lw=0.5)
ax.legend(fontsize=9, loc="lower center", ncol=3)
save(fig, "exam-f25-overfit", "Schematic of what Fall 2025 Q5(c) shows (not the exam's data): a degree-16 ridge fit to "
     "points with gaps. Too little regularization swings wildly in the gaps; more λ gives a smooth curve.",
     alt="Overfit polynomial versus regularized polynomial", width=560)

# ---------------------------------------------------------------- Fall 2025 Q6(b): schematic scatter
rng = np.random.default_rng(364)
fig, ax = plt.subplots(figsize=(4.2, 4.0))
for c, (mx, my), col, mk in [(1, (3.6, -0.4), ACCENT, "*"), (2, (-2.9, 3.0), HI, "s"), (3, (-2.6, -3.2), ACCENT2, "^")]:
    pts = rng.normal(0, 0.9, (18, 2)) + (mx, my)
    ax.scatter(pts[:, 0], pts[:, 1], s=10, color=col, alpha=0.7)
for (px, py), col, mk, lab in [((3, 1), ACCENT, "*", "class 1 (star)"), ((-3, 2), HI, "s", "class 2 (square)"),
                               ((-2, -3), ACCENT2, "^", "class 3 (triangle)")]:
    ax.scatter([px], [py], s=170, marker=mk, color=col, edgecolors=INK, linewidths=1, zorder=4, label=lab)
ax.set_xlim(-5.5, 5.5); ax.set_ylim(-6.5, 5.5); ax.grid(True, lw=0.5)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(fontsize=8, loc="upper right")
save(fig, "exam-f25-scatter", "Schematic of Fall 2025 Q6(b) (clusters re-sampled, not the exam's data). "
     "The three marked points sit at about (3, 1), (−3, 2) and (−2, −3).",
     alt="Three-class scatter with marked representatives", width=420)

print("ok")

# ---------------------------------------------------------------- figures inside callouts
# assemble.py only handles top-level markers; this inlines "> <!-- fig:exam-... -->" markers
# inside callouts (solution sketches that should stay folded). Idempotent.
CONTENT = os.path.join(HERE, "..", "content", "6-midterm-1")
MARK_Q = re.compile(r"^> <!--\s*fig:(exam-[a-z0-9-]+)\s*-->\s*$")
for fn in sorted(os.listdir(CONTENT)):
    if not fn.endswith(".md"):
        continue
    path = os.path.join(CONTENT, fn)
    lines = open(path, encoding="utf-8").read().split("\n")
    out, i, changed = [], 0, False
    while i < len(lines):
        out.append(lines[i])
        m = MARK_Q.match(lines[i])
        if m:
            fig = "> " + open(os.path.join(FIGS, m.group(1) + ".html"), encoding="utf-8").read().strip()
            if i + 1 < len(lines) and lines[i + 1].startswith("> <figure"):
                changed |= lines[i + 1] != fig
                i += 1
            else:
                changed = True
            out.append(fig)
        i += 1
    if changed:
        open(path, "w", encoding="utf-8").write("\n".join(out))
        print("inlined callout figures in", fn)
