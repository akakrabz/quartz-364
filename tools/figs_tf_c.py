#!/usr/bin/env python3
"""Figures for the "Attention Is All You Need" paper page. Run: python3 tools/figs_tf_c.py

Writes tools/figs/tf-c-*.html; then python3 tools/assemble.py content/7-transformers/attention-is-all-you-need.md
Everything is computed here from the paper's formulas (eq. 3, footnote 4, Table 1) with numpy.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mplfig import ACCENT, ACCENT2, HI, INK, MUTED, plt, save  # noqa: E402


# ---------------------------------------------------------------- 1. warmup learning-rate schedule (eq. 3)
def lrate(step, d_model=512, warmup=4000):
    step = np.asarray(step, dtype=float)
    return d_model ** -0.5 * np.minimum(step ** -0.5, step * warmup ** -1.5)


def lr_schedule():
    s = np.arange(1, 100_001)
    base, big = lrate(s, 512), lrate(s, 1024)
    peak = lrate(4000, 512)
    assert abs(peak - 6.988e-4) < 1e-6
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(s, base * 1e4, color=ACCENT, label=r"base, $d_{model}=512$")
    ax.plot(s, big * 1e4, color=HI, label=r"big, $d_{model}=1024$")
    ax.axvline(4000, color=MUTED, lw=1, ls="--")
    ax.annotate(f"peak {peak * 1e4:.2f}e-4\nat step 4000", (4000, peak * 1e4), (16000, 6.2), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    ax.text(55000, 2.55, r"$\propto step^{-1/2}$", fontsize=10, color=ACCENT)
    ax.set_xlabel("training step")
    ax.set_ylabel(r"learning rate ($\times10^{-4}$)")
    ax.set_xlim(0, 100_000)
    ax.set_ylim(0, 7.6)
    ax.set_xticks([0, 4000, 25000, 50000, 75000, 100000], ["0", "4k", "25k", "50k", "75k", "100k"])
    ax.legend(loc="upper right")
    ax.grid(True)
    save(fig, "tf-c-lr-schedule",
         "<b>The paper's learning-rate schedule</b> (eq. 3, <i>warmup_steps</i> = 4000), computed here. The rate grows "
         "linearly for 4,000 steps, peaks at <i>d</i><sub>model</sub><sup>−1/2</sup> · 4000<sup>−1/2</sup> ≈ 6.99 × 10⁻⁴ for the base model, "
         "then decays like 1/√step (1.40 × 10⁻⁴ at the base model's last step, 100k). The big model's larger "
         "<i>d</i><sub>model</sub> = 1024 scales the whole curve down by 1/√2.",
         alt="Learning rate rising linearly to a peak at step 4000 and decaying as one over square root of step")


# ---------------------------------------------------------------- 2. why scale by sqrt(d_k) (footnote 4)
def scaling():
    rng = np.random.default_rng(0)
    dks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    n_keys, trials = 10, 4000
    std_dot, max_unscaled, max_scaled = [], [], []
    for dk in dks:
        q = rng.standard_normal((trials, 1, dk))
        k = rng.standard_normal((trials, n_keys, dk))
        s = (q * k).sum(-1)                      # (trials, n_keys) dot products
        std_dot.append(s.std())

        def mean_max(x):
            x = x - x.max(1, keepdims=True)
            p = np.exp(x) / np.exp(x).sum(1, keepdims=True)
            return p.max(1).mean()
        max_unscaled.append(mean_max(s))
        max_scaled.append(mean_max(s / np.sqrt(dk)))
    std_dot = np.array(std_dot)
    assert np.allclose(std_dot, np.sqrt(dks), rtol=0.05)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.0))
    a1.loglog(dks, std_dot, "o", color=ACCENT, ms=4, label=r"std of $q\cdot k$ (simulated)", base=2)
    a1.loglog(dks, np.sqrt(dks), color=MUTED, lw=1.5, label=r"$d_k^{1/2}$", base=2)
    a1.set_xlabel(r"$d_k$")
    a1.set_title(r"dot products grow like $d_k^{1/2}$", fontsize=10)
    a1.legend(fontsize=8.5, loc="upper left")
    a1.grid(True, which="major")
    a2.semilogx(dks, max_unscaled, color=HI, marker="o", ms=4, label=r"softmax$(QK^\top)$", base=2)
    a2.semilogx(dks, max_scaled, color=ACCENT, marker="o", ms=4, label=r"softmax$(QK^\top/d_k^{1/2})$", base=2)
    a2.axhline(1 / n_keys, color=MUTED, lw=1, ls="--")
    a2.text(1.1, 1 / n_keys + 0.03, "uniform (1/10)", fontsize=8.5, color=MUTED)
    a2.set_ylim(0, 1.05)
    a2.set_xlabel(r"$d_k$")
    a2.set_title("largest attention weight, 10 keys", fontsize=10)
    a2.legend(fontsize=8.5, loc="center right")
    a2.grid(True, which="major")
    save(fig, "tf-c-scaling",
         "<b>Why the paper divides by √d<sub>k</sub></b> (its footnote 4), simulated here with random queries and keys "
         "whose entries are independent with mean 0 and variance 1. Left: the dot product <i>q</i>·<i>k</i> has variance "
         "<i>d</i><sub>k</sub>, so its spread grows like √d<sub>k</sub>. Right: without scaling, the softmax over 10 keys puts almost "
         "all weight on one key once <i>d</i><sub>k</sub> is in the hundreds (a saturated softmax, with tiny gradients); after dividing "
         "by √d<sub>k</sub> the weights keep the same spread at every <i>d</i><sub>k</sub>.",
         alt="Two panels: dot-product standard deviation growing as root d_k, and maximum softmax weight saturating without scaling")


# ---------------------------------------------------------------- 3. Table 1: per-layer cost, self-attention vs recurrent
def complexity():
    d = 512
    n = np.logspace(0.5, 4, 200)
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    ax.loglog(n, n ** 2 * d, color=ACCENT, label=r"self-attention  $n^2 d$")
    ax.loglog(n, n * d ** 2, color=HI, label=r"recurrent  $n d^2$")
    ax.axvline(d, color=MUTED, lw=1, ls="--")
    ax.text(d * 1.12, 2e5, r"$n = d = 512$", fontsize=9, color=MUTED)
    ax.set_xlabel("sequence length n")
    ax.set_ylabel("operations per layer (order)")
    ax.legend(loc="upper left")
    ax.grid(True, which="major")
    save(fig, "tf-c-complexity",
         "<b>Table 1's per-layer cost</b> for <i>d</i> = 512 (constants dropped). Self-attention costs O(<i>n</i>²<i>d</i>), "
         "a recurrent layer O(<i>n</i><i>d</i>²); the curves cross at <i>n</i> = <i>d</i>. For sentences of fewer tokens than the model "
         "width — the case the paper says is typical for word-piece or byte-pair sentence representations — "
         "self-attention is the cheaper layer, and unlike the recurrent layer it needs no sequential steps.",
         alt="Log-log plot of n squared d versus n d squared crossing at n equals 512")


if __name__ == "__main__":
    lr_schedule()
    scaling()
    complexity()
    print("ok")
