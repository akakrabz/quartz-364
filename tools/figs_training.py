#!/usr/bin/env python3
"""Figures for unit 4 (PyTorch training). Run: python3 tools/figs_training.py

Writes tools/figs/train-*.html; then python3 tools/assemble.py content/4-pytorch-training/*.md
All numbers are computed here with numpy; nothing is copied from a trained network.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figlib import SVG  # noqa: E402
from mplfig import ACCENT, ACCENT2, HI, INK, MUTED, FIGS, plt, save  # noqa: E402

MONO = "'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace"


def write(name, html):
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, name + ".html"), "w") as f:
        f.write(html + "\n")


def box(s, x, y, w, h, lines, accent=False, mono_first=True):
    color = "var(--accent)" if accent else "currentColor"
    s.rect(x, y, w, h, rx=8, stroke=color, fill="var(--paper)", width=1.6)
    n = len(lines)
    for i, t in enumerate(lines):
        yy = y + h / 2 + (i - (n - 1) / 2) * 17
        if i == 0 and mono_first:
            s.text(x + w / 2, yy, t, font=MONO, size=13, color=color, weight="600")
        else:
            s.text(x + w / 2, yy, t, size=12, color="var(--muted)" if i else color)


# ---------------------------------------------------------------- 1. training loop cycle
def loop_cycle():
    W, H = 660, 400
    s = SVG(W, H, font_size=13)
    cx, cy, R = W / 2, H / 2 + 4, 148
    steps = [
        ("optimizer.zero_grad()", "clear old .grad"),
        ("pred = model(x)", "forward pass"),
        ("loss = criterion(pred, y)", "scalar loss"),
        ("loss.backward()", "fill every .grad"),
        ("optimizer.step()", "θ ← θ − α·g"),
    ]
    n = len(steps)
    pts = []
    for i in range(n):
        a = -math.pi / 2 + 2 * math.pi * i / n
        pts.append((cx + 1.45 * R * math.cos(a), cy + R * math.sin(a)))
    bw, bh = 216, 46
    # arrows along the ellipse between boxes
    for i in range(n):
        a0 = -math.pi / 2 + 2 * math.pi * i / n
        a1 = -math.pi / 2 + 2 * math.pi * (i + 1) / n
        k = 24
        seg = [a0 + (a1 - a0) * t for t in np.linspace(0.30, 0.70, k)]
        d = "M " + " L ".join(f"{cx + 1.45 * R * math.cos(t):.1f} {cy + R * math.sin(t):.1f}" for t in seg)
        s.path(d, color="var(--muted)", width=1.8, arrow=True)
    for i, ((x, y), (code, note)) in enumerate(zip(pts, steps)):
        box(s, x - bw / 2, y - bh / 2, bw, bh, [code, note], accent=(i == 3))
    s.text(cx, cy + 12, "for x, y in train_loader:", font=MONO, size=13)
    s.text(cx, cy + 34, "one iteration = one batch", size=12, color="var(--muted)")
    s.text(cx, cy + 52, "one epoch = one pass over the data", size=12, color="var(--muted)")
    write("train-loop-cycle", s.figure(
        "<b>The training loop.</b> Every batch goes once around the cycle. <code>backward()</code> only "
        "<i>computes</i> gradients (adding them to <code>.grad</code>); <code>step()</code> is what changes the "
        "parameters, and <code>zero_grad()</code> stops last batch's gradient from being added to this one.",
        alt="Training loop cycle: zero_grad, forward, loss, backward, step"))


# ---------------------------------------------------------------- 2. Dataset -> DataLoader -> batches
def dataflow():
    W, H = 640, 250
    s = SVG(W, H, font_size=13)
    # dataset column of items
    s.text(80, 18, "Dataset", font=MONO, size=14, weight="600")
    s.text(80, 36, "len(ds) = 20", size=12, color="var(--muted)")
    for i in range(6):
        y = 50 + i * 26
        s.rect(20, y, 120, 22, rx=4, fill="var(--paper)")
        lab = f"ds[{i}] → (x, y)" if i < 5 else "⋮"
        s.text(80, y + 11, lab, font=MONO, size=11)
    s.text(80, 222, "x: (2,)   y: ()", font=MONO, size=11, color="var(--muted)")
    # loader
    s.line(150, 120, 222, 120, arrow=True, width=1.8)
    s.rect(230, 70, 160, 100, rx=10, stroke="var(--accent)", fill="var(--paper)", width=1.8)
    s.text(310, 92, "DataLoader", font=MONO, size=14, weight="600", color="var(--accent)")
    s.text(310, 114, "batch_size=8", font=MONO, size=11)
    s.text(310, 132, "shuffle=True", font=MONO, size=11)
    s.text(310, 152, "stacks 8 items", size=11, color="var(--muted)")
    s.line(398, 120, 446, 120, arrow=True, width=1.8)
    # batches
    rows = [("batch 1", "x: (8, 2)", "y: (8,)"), ("batch 2", "x: (8, 2)", "y: (8,)"), ("batch 3", "x: (4, 2)", "y: (4,)")]
    for i, (b, xs, ys) in enumerate(rows):
        y = 52 + i * 50
        last = i == 2
        s.rect(455, y, 170, 40, rx=6, fill="var(--paper)", stroke="var(--hi)" if last else "currentColor")
        s.text(540, y + 13, b + ("  (last, partial)" if last else ""), size=11, color="var(--hi)" if last else "var(--muted)")
        s.text(540, y + 28, f"{xs}   {ys}", font=MONO, size=11)
    s.text(540, 222, "3 batches = len(loader)", size=12, color="var(--muted)")
    write("train-dataflow", s.figure(
        "<b>Dataset → DataLoader → batches.</b> The <code>Dataset</code> hands out one example at a time; the "
        "<code>DataLoader</code> draws indices, calls <code>ds[i]</code> for each and stacks the results along a new "
        "dimension 0. With 20 examples and <code>batch_size=8</code> there are ⌈20/8⌉ = 3 batches per epoch, the last "
        "one holding only 4 (unless <code>drop_last=True</code>).",
        alt="Dataset items flowing through a DataLoader into batches of shape (8, 2)"))


# ---------------------------------------------------------------- 3. data splits
def splits():
    W, H = 600, 140
    s = SVG(W, H, font_size=13)
    x0, w = 20, 560
    parts = [("train  60%", 0.6, ["fit the weights θ"], "var(--accent)"),
             ("val  20%", 0.2, ["choose", "hyperparameters"], "var(--accent2)"),
             ("test  20%", 0.2, ["final score,", "used once"], "var(--hi)")]
    x = x0
    for lab, frac, role, col in parts:
        ww = w * frac
        s.rect(x, 24, ww, 40, rx=4, stroke=col, fill=col, fill_opacity=0.12, width=1.8)
        s.text(x + ww / 2, 44, lab, weight="600", color=col)
        for j, r in enumerate(role):
            s.text(x + ww / 2, 80 + 15 * j, r, size=12)
        x += ww
    s.text(W / 2, 126, "shuffle the indices once, then cut; the three sets never share an example",
           size=12, color="var(--muted)")
    write("train-splits", s.figure(
        "<b>Train / validation / test.</b> The lecture's 60 : 20 : 20 split. Only the training set produces gradients; "
        "the validation set is looked at many times (to choose learning rate, epochs, weight decay, model size), so it "
        "is no longer an honest estimate. The test set is touched once.",
        alt="A bar split 60/20/20 into train, validation and test"))


# ---------------------------------------------------------------- 4. momentum vs plain GD (numpy)
def momentum():
    Hm = np.diag([1.0, 10.0])

    def run(lr, mu, steps, p0=(-9.0, 2.0)):
        p = np.array(p0)
        G = np.zeros(2)
        path = [p.copy()]
        for _ in range(steps):
            g = Hm @ p
            G = g + mu * G          # PyTorch's SGD momentum buffer
            p = p - lr * G
            path.append(p.copy())
        return np.array(path)

    def n_to(lr, mu, tol=1e-4):
        P = run(lr, mu, 2000)
        f = 0.5 * (P[:, 0] ** 2 + 10 * P[:, 1] ** 2)
        return int(np.argmax(f < tol))

    lr = 0.05
    plain, mom = run(lr, 0.0, 40), run(lr, 0.8, 40)
    k0, k1 = n_to(lr, 0.0), n_to(lr, 0.8)
    xs, ys = np.meshgrid(np.linspace(-10, 3, 300), np.linspace(-3, 3, 200))
    Z = 0.5 * (xs ** 2 + 10 * ys ** 2)
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.contour(xs, ys, Z, levels=[0.5, 2, 5, 10, 20, 35, 50], colors=MUTED, linewidths=0.8)
    ax.plot(plain[:, 0], plain[:, 1], "o-", color=HI, ms=3, lw=1.4, label=f"SGD, momentum 0  ({k0} steps to f < 1e-4)")
    ax.plot(mom[:, 0], mom[:, 1], "o-", color=ACCENT, ms=3, lw=1.4, label=f"SGD, momentum 0.8  ({k1} steps)")
    ax.plot([0], [0], "*", color=INK, ms=10)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1, fontsize=9)
    save(fig, "train-momentum",
         "<b>Momentum on a long, narrow valley.</b> Both runs minimise f(θ) = ½(θ₁² + 10θ₂²) "
         f"from (−9, 2) with α = {lr}; first 40 steps shown. Plain gradient descent crawls along the shallow "
         "θ₁ direction; with μ = 0.8 the gradients there keep pointing the same way and add up, so the "
         f"run gets there in {k1} steps instead of {k0}. (Computed with numpy, same update rule as "
         "<code>torch.optim.SGD</code>.)",
         alt="Contour plot with a plain gradient descent path and a faster momentum path")


# ---------------------------------------------------------------- 5. minibatch noise (numpy logistic regression)
def minibatch():
    rng = np.random.default_rng(3)
    N = 50
    X = np.vstack([0.75 * rng.standard_normal((N, 2)) + [-1, 1], 0.75 * rng.standard_normal((N, 2)) + [1, -1]])
    y = np.r_[np.ones(N), np.zeros(N)]

    def bce(w, b, idx):
        p = np.clip(1 / (1 + np.exp(-(X[idx] @ w + b))), 1e-12, 1 - 1e-12)
        return -np.mean(y[idx] * np.log(p) + (1 - y[idx]) * np.log(1 - p))

    def train(bs, iters=300, lr=0.1):
        w, b = np.zeros(2), 0.0
        r = np.random.default_rng(bs)
        logged, full = [], []
        for _ in range(iters):
            idx = r.choice(len(y), size=bs, replace=False)
            logged.append(bce(w, b, idx))            # what loss.item() reports
            full.append(bce(w, b, np.arange(len(y))))
            p = 1 / (1 + np.exp(-(X[idx] @ w + b)))
            w, b = w - lr * X[idx].T @ (p - y[idx]) / bs, b - lr * np.mean(p - y[idx])
        return np.array(logged), np.array(full)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for bs, col, lw in [(1, HI, 0.9), (16, ACCENT, 1.3), (100, INK, 2)]:
        lab = "batch = all 100 (plain GD)" if bs == 100 else f"batch = {bs}"
        logged, full = train(bs)
        a1.plot(logged, color=col, lw=lw, label=lab)
        a2.plot(full, color=col, lw=lw)
    a1.set_title("loss.item() of each batch", fontsize=11)
    a2.set_title("loss on all 100 points", fontsize=11)
    for a in (a1, a2):
        a.set_xlabel("iteration")
        a.grid(True)
        a.set_ylim(0, 1.2)
    a1.set_ylabel("BCE")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9)
    save(fig, "train-minibatch",
         "<b>Stochastic vs full-batch gradient descent.</b> Logistic regression on the lecture's two Gaussian "
         "clusters (50 points each), learning rate 0.1, one <code>step()</code> per iteration. Left: the per-batch "
         "loss you log inside the loop is noisy for small batches. Right: the loss on the whole dataset falls "
         "almost identically for all three, but a batch-1 step costs 1/100 of a full-batch step. "
         "(numpy re-computation of the Lecture 10 setup.)",
         alt="Per-batch and full-dataset loss curves for batch sizes 1, 16 and 100")


# ---------------------------------------------------------------- 6. LR schedules (closed forms)
def schedules():
    lr0, T = 0.1, 100
    e = np.arange(T)
    step = lr0 * 0.1 ** (e // 30)
    expo = lr0 * 0.95 ** e
    cos = 0 + (lr0 - 0) * (1 + np.cos(np.pi * e / T)) / 2
    multi = lr0 * 0.1 ** ((e >= 50).astype(int) + (e >= 80).astype(int))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.step(e, step, where="post", color=ACCENT, label="StepLR(step_size=30, gamma=0.1)")
    ax.step(e, multi, where="post", color=ACCENT2, ls="--", label="MultiStepLR(milestones=[50, 80], gamma=0.1)")
    ax.plot(e, expo, color=HI, label="ExponentialLR(gamma=0.95)")
    ax.plot(e, cos, color=INK, lw=1.5, label="CosineAnnealingLR(T_max=100)")
    ax.set_yscale("log")
    ax.set_ylim(5e-5, 0.2)
    ax.set_xlabel("epoch (one scheduler.step() per epoch)")
    ax.set_ylabel("learning rate")
    ax.grid(True, which="major")
    ax.legend(fontsize=8.5, loc="lower left")
    save(fig, "train-lr-schedules",
         "<b>Four schedulers, starting from <code>lr=0.1</code>.</b> Curves are the closed-form rules from the "
         "PyTorch docs (log scale). StepLR divides by 10 every 30 epochs; MultiStepLR at chosen epochs; "
         "ExponentialLR multiplies by 0.95 every epoch; cosine annealing glides to 0 at <code>T_max</code>.",
         alt="Learning rate versus epoch for StepLR, MultiStepLR, ExponentialLR and CosineAnnealingLR")


# ---------------------------------------------------------------- 7. overfitting (numpy)
def overfit():
    rng = np.random.default_rng(0)
    ntr = 12
    xtr = np.sort(rng.uniform(-1, 1, ntr))
    ytr = np.sin(np.pi * xtr) + 0.3 * rng.standard_normal(ntr)
    xva = rng.uniform(-1, 1, 200)
    yva = np.sin(np.pi * xva) + 0.3 * rng.standard_normal(200)
    D = 11

    def phi(x):
        return np.stack([np.polynomial.legendre.Legendre.basis(k)(x) for k in range(D + 1)], 1)

    A, B = phi(xtr), phi(xva)
    w, G = np.zeros(D + 1), np.zeros(D + 1)
    tr, va = [], []
    for _ in range(20000):
        g = 2 * A.T @ (A @ w - ytr) / ntr
        G = g + 0.9 * G
        w -= 0.05 * G
        tr.append(np.mean((A @ w - ytr) ** 2))
        va.append(np.mean((B @ w - yva) ** 2))
    tr, va = np.array(tr), np.array(va)
    best = int(va.argmin())
    it = np.arange(1, len(tr) + 1)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(it, tr, color=ACCENT, label="training MSE (12 points)")
    ax.plot(it, va, color=HI, label="validation MSE (200 held-out points)")
    ax.axvline(best + 1, color=MUTED, ls="--", lw=1.2)
    ax.text(best + 1.3, 1.6, f"best validation\n(iteration {best + 1})", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("MSE")
    ax.grid(True, which="major")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    save(fig, "train-overfit",
         "<b>Overfitting, measured.</b> A degree-11 polynomial (12 weights) fit to 12 noisy samples of "
         "sin(πx) by gradient descent with momentum. Training error keeps falling; validation error "
         "bottoms out early and then climbs as the model starts fitting the noise. Picking the iteration (or the "
         "degree, or λ) by validation error is exactly what the validation set is for. "
         "(Real numpy run, small synthetic problem, not a lecture plot.)",
         alt="Training loss falling while validation loss rises after a minimum")


if __name__ == "__main__":
    loop_cycle()
    dataflow()
    splits()
    momentum()
    minibatch()
    schedules()
    overfit()
    print("wrote", sorted(f for f in os.listdir(FIGS) if f.startswith("train-")))
