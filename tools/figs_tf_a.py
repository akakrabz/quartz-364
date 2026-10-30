#!/usr/bin/env python3
"""Figures for the Lecture 19 pages (sequence-to-sequence RNNs, attention).

Run: python3 tools/figs_tf_a.py   (then tools/assemble.py on content/7-transformers/*.md)
Writes tools/figs/tf-a-*.html. BLEU curves are the Lecture 19 notebook's saved
results (*.npy); set L19_DIR if py-complete/Lecture19 is not at ../../py-complete
relative to tools/. Everything else is computed here with numpy.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figlib import SVG  # noqa: E402
from mplfig import ACCENT, ACCENT2, HI, INK, MUTED, FIGS, plt, save  # noqa: E402

MONO = "'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace"
L19 = os.environ.get("L19_DIR", os.path.join(HERE, "..", "..", "py-complete", "Lecture19"))


def write(name, html):
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, name + ".html"), "w") as f:
        f.write(html + "\n")


def arrow(s, x1, y1, x2, y2, color="currentColor", width=1.4):
    s.line(x1, y1, x2, y2, color=color, width=width)
    v = np.array([x2 - x1, y2 - y1], float)
    v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]])
    tip = np.array([x2, y2])
    a, b = tip - 7 * v + 3.5 * n, tip - 7 * v - 3.5 * n
    s.add(f'<path d="M{tip[0]:.1f},{tip[1]:.1f} L{a[0]:.1f},{a[1]:.1f} L{b[0]:.1f},{b[1]:.1f} Z" fill="{color}"/>')


# ---------------------------------------------------------------- 1. S2S RNN variants + Skinny Transformer
def s2s_bleu():
    runs = [("seq2seq_rnn", "unidirectional", MUTED, "-"),
            ("forcing_seq2seq_rnn", "unidirectional + forcing", MUTED, "--"),
            ("bidirectional_seq2seq_rnn", "bidirectional", ACCENT, "-"),
            ("bidirectional_forcing_seq2seq_rnn", "bidirectional + forcing", ACCENT, "--")]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for split, ax in zip(["train", "val"], axes):
        for d, lab, c, ls in runs:
            a = np.load(os.path.join(L19, d, f"{split}_bleu_scores.npy"))
            ax.plot(np.arange(1, 101), a, color=c, ls=ls, lw=1.6, label=lab)
        t = np.load(os.path.join(L19, "transformer", f"{split}_bleu_scores.npy"))
        ax.plot(np.arange(1, 101), t, color=HI, lw=1.6, label="Skinny Transformer")
        ax.set_title("training" if split == "train" else "validation", fontsize=11)
        ax.set_xlabel("evaluation step (of 100)")
        ax.set_ylim(0, 1.02)
        ax.grid(True)
    axes[0].set_ylabel("BLEU (0–1)")
    axes[0].legend(fontsize=8, loc="lower right")
    save(fig, "tf-a-s2s-bleu",
         "<b>English→French BLEU of the Lecture 19 models</b> (the notebook's saved runs: <code>min_freq = 200</code>, "
         "4,988 training pairs, embedding and hidden size 64, evaluated on 1,000 random sentences per step). "
         "Training BLEU separates the variants clearly — bidirectional encoders and teacher forcing learn fastest — "
         "while validation BLEU ends between about 0.27 and 0.37 for all of them. The red curve is the Skinny "
         "Transformer from the same lecture.",
         alt="Training and validation BLEU curves for four sequence-to-sequence RNNs and a Skinny Transformer")


# ---------------------------------------------------------------- 2. why scale by sqrt(d)
def scaling():
    rng = np.random.default_rng(0)
    dims = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    n_keys, trials = 10, 2000
    std, pmax_raw, pmax_scaled = [], [], []
    for d in dims:
        q = rng.standard_normal((trials, 1, d))
        k = rng.standard_normal((trials, n_keys, d))
        s = (q * k).sum(-1)                       # (trials, n_keys)
        std.append(s.std())
        for arr, scale in ((pmax_raw, 1.0), (pmax_scaled, np.sqrt(d))):
            z = s / scale
            p = np.exp(z - z.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            arr.append(p.max(1).mean())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a1.loglog(dims, std, color=ACCENT, marker="o", ms=4, label="std of qᵀk (measured)")
    a1.loglog(dims, np.sqrt(dims), color=MUTED, ls="--", label="√d")
    a1.set_xlabel("dimension d of q and k")
    a1.legend(fontsize=9)
    a1.grid(True, which="major")
    a2.semilogx(dims, pmax_raw, color=HI, marker="o", ms=4, label="softmax(qᵀk)")
    a2.semilogx(dims, pmax_scaled, color=ACCENT, marker="o", ms=4, label="softmax(qᵀk / √d)")
    a2.axhline(1 / n_keys, color=MUTED, lw=1, ls=":")
    a2.text(1.1, 1 / n_keys + 0.03, "uniform = 0.1", fontsize=8, color=MUTED)
    a2.set_ylim(0, 1.05)
    a2.set_xlabel("dimension d")
    a2.set_ylabel("mean largest weight")
    a2.legend(fontsize=9, loc="center right")
    a2.grid(True, which="major")
    for ax in (a1, a2):
        ax.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
        ax.minorticks_off()
    save(fig, "tf-a-scaling",
         "<b>Why divide by √d.</b> Left: with independent unit-variance entries, the dot product of a query and a key "
         "has variance d, so its spread grows like √d (2,000 random trials per d). Right: the largest of ten "
         "attention weights. Without scaling the softmax becomes nearly one-hot as d grows — a region where its "
         "gradients vanish; dividing by √d keeps the weights spread the same way at every d.",
         alt="Dot-product spread growing like the square root of d, and softmax saturating without scaling")


# ---------------------------------------------------------------- 3. worked example heatmaps
Z = np.array([[1., 0.], [0., 1.], [1., 1.]])
WQ = np.array([[1., 1.], [0., 1.]])
WK = np.array([[1., 0.], [1., 1.]])
WV = np.array([[1., 0.], [0., 2.]])


def softmax_rows(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def worked():
    Q, K, V = Z @ WQ, Z @ WK, Z @ WV
    S = Q @ K.T / np.sqrt(2)
    A = softmax_rows(S)
    M = np.triu(np.full((3, 3), -np.inf), 1)
    Am = softmax_rows(S + M)
    assert np.allclose(A @ V, [[0.716, 1.720], [0.599, 1.604], [0.694, 1.851]], atol=2e-3)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    toks = ["$t_1$", "$t_2$", "$t_3$"]
    from matplotlib.patches import Rectangle
    for ax, mat, title in [(axes[0], A, "self-attention $A$"), (axes[1], Am, "masked self-attention")]:
        for i in range(3):
            for j in range(3):
                masked = title.startswith("masked") and j > i
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=ACCENT, alpha=0.08 + 0.55 * mat[i, j],
                                       edgecolor=MUTED, lw=0.6))
                ax.text(j, i, "0\n(masked)" if masked else f"{mat[i, j]:.3f}", ha="center", va="center",
                        fontsize=7.5 if masked else 9.5, color=MUTED if masked else INK)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(2.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(3), toks)
        ax.set_yticks(range(3), toks)
        ax.tick_params(length=0)
        ax.set_xlabel("key $j$")
        ax.set_title(title, fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("query $i$")
    save(fig, "tf-a-worked",
         "<b>The worked example's attention weights.</b> Each row is a query and sums to 1. Left: every token "
         "attends to every token; token 3's key scores highest (or tied) against every query, so it gets the most "
         "weight in rows 1 and 3 and ties with token 2 in row 2. Right: the causal mask zeroes everything above the diagonal, so token 1 can only "
         "attend to itself and token 2 renormalizes over tokens 1–2. The last row is identical in both: the last "
         "position sees everything anyway.",
         alt="Two 3 by 3 heatmaps of attention weights, unmasked and causally masked")


# ---------------------------------------------------------------- 4. causal mask during teacher forcing
def causal():
    toks = ["<SOS>", "The", "dog", "is", "cute", "!"]
    nxt = ["The", "dog", "is", "cute", "!", "<EOS>"]
    n = len(toks)
    cell, x0, y0 = 46, 120, 70
    W, H = x0 + n * cell + 150, y0 + n * cell + 30
    s = SVG(W, H, font_size=12)
    s.text(x0 + n * cell / 2, 18, "keys: decoder inputs (target shifted right)", size=12, color="var(--muted)")
    s.text(x0 + n * cell + 75, 18, "trained to predict", size=12, color="var(--muted)")
    for j, t in enumerate(toks):
        s.text(x0 + j * cell + cell / 2, y0 - 14, t, font=MONO, size=11)
    for i, t in enumerate(toks):
        y = y0 + i * cell
        s.text(x0 - 10, y + cell / 2, t, font=MONO, size=11, anchor="end")
        for j in range(n):
            x = x0 + j * cell
            ok = j <= i
            s.rect(x, y, cell, cell, fill="var(--accent)" if ok else "none", fill_opacity=0.22 if ok else None,
                   stroke="var(--muted)", width=0.8)
            s.text(x + cell / 2, y + cell / 2, "0" if ok else "−∞", size=12,
                   color="var(--accent)" if ok else "var(--muted)")
        arrow(s, x0 + n * cell + 8, y + cell / 2, x0 + n * cell + 40, y + cell / 2, color="var(--muted)", width=1.2)
        s.text(x0 + n * cell + 46, y + cell / 2, nxt[i], font=MONO, size=11, anchor="start", color="var(--hi)")
    s.text(40, y0 + n * cell / 2, "queries", size=12, color="var(--muted)", anchor="middle")
    write("tf-a-causal", s.figure(
        "<b>The causal mask $M$ during teacher-forced training</b> for the lecture's example “The dog is cute!”. "
        "Row $i$ is the query at decoder position $i$; shaded cells (mask 0) are the keys it may attend to, "
        "−∞ cells get attention weight exactly 0 after the softmax. All six rows are computed in one pass, and row "
        "$i$ is trained to output the next word on the right — six next-token predictions in parallel, none of which "
        "can see the word it must predict.".replace("$M$", "<i>M</i>").replace("$i$", "<i>i</i>"),
        alt="Lower-triangular causal mask over six decoder tokens with the next-token target for each row"))


# ---------------------------------------------------------------- 5. bottleneck vs attention
def bottleneck():
    W, H = 720, 250
    s = SVG(W, H, font_size=12)
    enc = ["I", "am", "cold", "."]
    dec = ["J'ai", "froid", ".", "<EOS>"]
    for panel, x_off, title in [(0, 0, "seq2seq RNN: one context vector"), (1, 370, "attention: every encoder state")]:
        s.text(x_off + 175, 18, title, size=13, weight="600")
        ex = [x_off + 30 + 45 * i for i in range(4)]
        dx = [x_off + 190 + 40 * i for i in range(4)]
        for i, t in enumerate(enc):
            s.rect(ex[i] - 17, 150, 34, 26, rx=4, fill="var(--paper)")
            s.text(ex[i], 163, "h" + str(i + 1), size=11, font=MONO)
            s.text(ex[i], 196, t, font=MONO, size=11, color="var(--muted)")
            if i < 3:
                arrow(s, ex[i] + 17, 163, ex[i + 1] - 19, 163, color="var(--muted)", width=1.1)
        for i, t in enumerate(dec):
            s.rect(dx[i] - 17, 70, 34, 26, rx=4, fill="var(--paper)")
            s.text(dx[i], 83, "s" + str(i + 1), size=11, font=MONO)
            s.text(dx[i], 52, t, font=MONO, size=11, color="var(--muted)")
            if i < 3:
                arrow(s, dx[i] + 17, 83, dx[i + 1] - 19, 83, color="var(--muted)", width=1.1)
        if panel == 0:
            arrow(s, ex[3] + 17, 160, dx[0] - 19, 88, color="var(--hi)", width=2.2)
            s.text(ex[3] + 40, 118, "context", size=11, color="var(--hi)", anchor="start")
        else:
            for j in range(4):
                for i in range(4):
                    arrow(s, ex[i], 149, dx[j], 98, color="var(--accent)", width=0.8)
            s.text(x_off + 175, 232, "weights recomputed for each output word", size=11, color="var(--accent)")
    write("tf-a-bottleneck", s.figure(
        "<b>Limitation 3 in one picture.</b> Left: the encoder's last hidden state is the only thing the decoder "
        "receives about the source sentence, and the decoder keeps overwriting it as it generates. Right: with "
        "cross-attention every decoder position reads a fresh weighted combination of <i>all</i> encoder states "
        "(Bahdanau et al. 2014 added this to RNNs; the Transformer keeps only the attention).",
        alt="Encoder-decoder with a single context arrow versus all-to-all attention connections"))


if __name__ == "__main__":
    s2s_bleu()
    scaling()
    worked()
    causal()
    bottleneck()
    print("ok")
