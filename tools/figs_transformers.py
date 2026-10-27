#!/usr/bin/env python3
"""Figures for unit 7 (transformers). Run: python3 tools/figs_transformers.py

Writes tools/figs/tf-*.html; then python3 tools/assemble.py content/7-transformers/*.md
Accuracies are the Lecture 18 notebook's printed outputs and saved
EmbeddingTrials/*.npy results (copied below); nothing is retrained here.
"""
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


def arrow(s, x1, y1, x2, y2, color="currentColor", width=1.5):
    """Line with a filled triangular head (no marker, so colour always matches)."""
    s.line(x1, y1, x2, y2, color=color, width=width)
    v = np.array([x2 - x1, y2 - y1], float)
    v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]])
    tip = np.array([x2, y2])
    a, b = tip - 7 * v + 3.5 * n, tip - 7 * v - 3.5 * n
    s.add(f'<path d="M{tip[0]:.1f},{tip[1]:.1f} L{a[0]:.1f},{a[1]:.1f} L{b[0]:.1f},{b[1]:.1f} Z" fill="{color}"/>')


# ---------------------------------------------------------------- 1. four encodings compared
def encodings():
    names = ["one-hot\n(d = 52)", "binary\n(d = 6)", "custom 2-D\n(d = 2)", "learned\n(d = 8)"]
    train = [98.29, 97.31, 93.26, 98.07]
    val = [53.06, 49.92, 37.12, 58.90]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.bar(x - 0.2, train, 0.38, color=MUTED, label="training")
    ax.bar(x + 0.2, val, 0.38, color=ACCENT, label="validation")
    for xi, t, v in zip(x, train, val):
        ax.text(xi - 0.2, t + 1.5, f"{t:.1f}", ha="center", fontsize=9)
        ax.text(xi + 0.2, v + 1.5, f"{v:.1f}", ha="center", fontsize=9, color=ACCENT)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 108)
    ax.set_ylabel("mean category accuracy (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, "tf-encodings",
         "<b>Four ways to feed letters to the same Elman RNN</b> (Lecture 18, surname task with 8 languages, "
         "128 hidden units, 100 epochs, one run each). All four fit the training set; on held-out names the "
         "learned 8-dimensional embedding is best and the hand-designed 2-D code is worst. Mean category "
         "accuracy averages the per-language accuracies.",
         alt="Bar chart of training and validation accuracy for one-hot, binary, 2-D and learned encodings")


# ---------------------------------------------------------------- 2. embedding-dimension sweep
TRAIN = {1: [98.0795001, 97.92629924, 98.03654596], 2: [98.36752657, 98.15467611, 98.21927066],
         4: [98.45580713, 98.51359904, 98.29706593], 8: [98.16312371, 98.38479776, 98.48081236],
         16: [98.26273863, 98.46089783, 98.31384513], 32: [98.7766297, 98.47911716, 98.74284463],
         64: [98.22242924, 98.40445722, 98.22295472], 128: [98.47567142, 98.24602642, 98.73870208],
         256: [98.55831256, 98.45695361, 98.49707492], 512: [98.18058948, 97.96990062, 98.3760565],
         1024: [97.66125124, 97.24874223, 96.54197681]}
VAL = {1: [46.47091884, 43.35160656, 44.35615249], 2: [49.25045232, 48.26546977, 47.62197401],
       4: [51.67890031, 50.97264151, 53.65695839], 8: [56.09688965, 56.88421093, 54.59412119],
       16: [56.28093717, 57.89609177, 59.32065242], 32: [61.41627699, 61.34572937, 59.95267467],
       64: [63.0840106, 62.65349927, 62.29008076], 128: [63.38908073, 61.68056046, 62.65170094],
       256: [61.93588585, 63.00110647, 62.0794739], 512: [60.38217173, 61.56699206, 60.59079166],
       1024: [59.32863833, 59.52855896, 60.63784525]}


def embed_dim():
    E = np.array(sorted(VAL))
    tm = np.array([np.mean(TRAIN[e]) for e in E]); ts = np.array([np.std(TRAIN[e]) for e in E])
    vm = np.array([np.mean(VAL[e]) for e in E]); vs = np.array([np.std(VAL[e]) for e in E])
    # matches the notebook's printed means
    assert abs(vm[E.tolist().index(64)] - 62.68) < 0.01 and abs(tm[-1] - 97.15) < 0.01
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, m, sd, c, title, lim in [(a1, tm, ts, MUTED, "training", (95, 100)),
                                     (a2, vm, vs, ACCENT, "validation", (40, 66))]:
        ax.semilogx(E, m, color=c, marker="o", ms=4, base=2)
        ax.fill_between(E, m - sd, m + sd, color=c, alpha=0.25, lw=0)
        ax.set_ylim(*lim)
        ax.set_xticks(E, [str(e) if k % 2 == 0 else "" for k, e in enumerate(E)], fontsize=9)
        ax.set_xlabel("embedding dimension d")
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="major")
    a1.set_ylabel("mean category acc. (%)")
    i = int(np.argmax(vm))
    a2.annotate(f"peak {vm[i]:.1f}% at d = {E[i]}", (E[i], vm[i]), (5, 50), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    save(fig, "tf-embed-dim",
         "<b>Sweeping the embedding dimension</b> (Lecture 18, 3 runs per d; line = mean, band = ±1 std). "
         "Training accuracy is ≈98% for every d up to 512 and dips at 1024. Validation accuracy climbs from 44.7% "
         "(d = 1) to a plateau of about 62–63% for d = 64–256, then falls slightly: past a point, more dimensions add "
         "cost, not information.",
         alt="Training and validation accuracy versus embedding dimension on a log axis")


# ---------------------------------------------------------------- 3. nn.Embedding lookup
def lookup():
    W, H = 700, 300
    s = SVG(W, H, font_size=13)
    letters = ["C", "a", "m", "p", "o"]
    idx = [28, 0, 12, 15, 14]
    # input indices
    s.text(70, 22, "x  (5, 1)  int", font=MONO, size=12, color="var(--muted)")
    for i, (ch, k) in enumerate(zip(letters, idx)):
        y = 44 + i * 44
        s.text(22, y + 14, ch, font=MONO, size=15, weight="600")
        s.rect(40, y, 60, 28, rx=4, fill="var(--paper)")
        s.text(70, y + 14, str(k), font=MONO, size=13)
    # embedding table
    tx, ty, rw, rh = 250, 40, 170, 230
    s.text(tx + rw / 2, 22, "embedding.weight  (52, 8)", font=MONO, size=12, color="var(--muted)")
    s.rect(tx, ty, rw, rh, rx=4, fill="var(--paper)")
    rows = {0: ty + 8, 12: ty + 60, 14: ty + 80, 15: ty + 100, 28: ty + 170}
    for k, yy in rows.items():
        s.rect(tx + 4, yy, rw - 8, 15, rx=2, fill="var(--accent)", fill_opacity=0.18, stroke="var(--accent)", width=1)
        s.text(tx + 12, yy + 7.5, f"row {k}", font=MONO, size=10, anchor="start", color="var(--accent)")
    s.text(tx + rw / 2, ty + 140, "⋮", size=16, color="var(--muted)")
    s.text(tx + rw / 2, ty + rh - 14, "one learnable row per letter", size=11, color="var(--muted)")
    for i, k in enumerate(idx):
        arrow(s, 102, 58 + i * 44, tx + 2, rows[k] + 7.5, color="var(--muted)", width=1.2)
    # output vectors
    ox = 530
    s.text(ox + 66, 22, "z = embedding(x)  (5, 1, 8)", font=MONO, size=12, color="var(--muted)")
    for i, k in enumerate(idx):
        y = 44 + i * 44
        for j in range(8):
            s.rect(ox + j * 17, y + 4, 15, 20, rx=2, fill="var(--accent)", fill_opacity=0.18, stroke="var(--accent)", width=1)
        arrow(s, tx + rw + 4, rows[k] + 7.5, ox - 6, y + 14, color="var(--accent)", width=1.2)
    s.text(ox + 70, 278, "→ nn.RNN(input_size=8, …)", font=MONO, size=12)
    write("tf-lookup", s.figure(
        "<b><code>nn.Embedding</code> is a lookup table.</b> The surname “Campo” becomes the indices 28, 0, 12, 15, 14 "
        "(<code>string.ascii_letters</code>: a–z are 0–25, A–Z are 26–51). Each index selects one row of the "
        "52 × 8 weight matrix, so the RNN receives a (5, 1, 8) tensor. Only the selected rows get a gradient "
        "from this example.",
        alt="Five letter indices selecting rows of a 52 by 8 embedding table, giving five 8-dimensional vectors"))


# ---------------------------------------------------------------- 4. vocabulary vs min_freq
def vocab():
    mf = np.array([1, 10, 50, 100, 500, 1000, 5000])
    allw = np.array([81305, 9620, 3001, 1750, 412, 222, 49])
    ds = np.array([8310, 5041, 2617, 1673, 411, 222, 49])
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    ax.loglog(mf, allw, color=MUTED, marker="o", ms=4, label="words with count ≥ min_freq (all titles)")
    ax.loglog(mf, ds, color=ACCENT, marker="o", ms=4, label="vocabulary of the 10,000 sampled titles")
    ax.set_xticks([1, 10, 100, 1000, 5000], ["1", "10", "100", "1000", "5000"])
    ax.minorticks_off()
    ax.set_xlabel("min_freq")
    ax.set_ylabel("vocabulary size")
    ax.grid(True, which="major")
    ax.legend(fontsize=9, loc="lower left")
    save(fig, "tf-vocab",
         "<b>Vocabulary size against <code>min_freq</code></b> for the Lecture 18 <code>SentimentDataset</code> "
         "(word tokens of review titles, punctuation split off). Computed here by running the notebook's "
         "preprocessing on <code>reviews.csv</code>, with the sampling seeded (<code>np.random.seed(0)</code>), so "
         "the lower curve varies a little from run to run. The default <code>min_freq = 100</code> keeps 1,750 "
         "words and gives a vocabulary of about 1,700.",
         alt="Log-log plot of vocabulary size falling as min_freq grows")


if __name__ == "__main__":
    encodings()
    embed_dim()
    lookup()
    vocab()
    print("ok")
