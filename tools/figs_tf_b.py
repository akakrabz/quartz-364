#!/usr/bin/env python3
"""Figures for the Lecture 19-20 transformer pages (architecture, multi-head
attention, positional encoding, layer norm). Run: python3 tools/figs_tf_b.py [LECTURE20_DIR]

Writes tools/figs/tf-b-*.html; then python3 tools/assemble.py content/7-transformers/<page>.md
Curves are the saved Lecture 20 results (.npy) read from LECTURE20_DIR
(default: ../py-complete/Lecture20 next to the repo, as in ~/Nextcloud/Notes/ECE 364).
Nothing is retrained here; positional encodings and layer-norm numbers are computed with numpy.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figlib import SVG  # noqa: E402
from mplfig import ACCENT, ACCENT2, HI, INK, MUTED, FIGS, plt, save  # noqa: E402

MONO = "'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace"
CANDIDATES = [sys.argv[1]] if len(sys.argv) > 1 else [
    os.path.join(HERE, "..", "..", "py-complete", "Lecture20"),
    "/mnt/user-data/uploads/ECE 364/py-complete/Lecture20",
]
L20 = next((p for p in CANDIDATES if os.path.isdir(p)), None)


def write(name, html):
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, name + ".html"), "w") as f:
        f.write(html + "\n")


def load(*parts):
    return np.load(os.path.join(L20, *parts))


def arrow(s, x1, y1, x2, y2, color="currentColor", width=1.5):
    s.line(x1, y1, x2, y2, color=color, width=width)
    v = np.array([x2 - x1, y2 - y1], float)
    v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]])
    tip = np.array([x2, y2])
    a, b = tip - 7 * v + 3.5 * n, tip - 7 * v - 3.5 * n
    s.add(f'<path d="M{tip[0]:.1f},{tip[1]:.1f} L{a[0]:.1f},{a[1]:.1f} L{b[0]:.1f},{b[1]:.1f} Z" fill="{color}"/>')


# ---------------------------------------------------------------- 1. Skinny -> Medium -> Full
def progression():
    names = [("SkinnyTransformer", "Skinny", MUTED, "-"), ("MediumTransformer", "Medium", HI, "--"),
             ("FullTransformer", "Full", ACCENT, "-")]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for d, lab, c, ls in names:
        loss = load(d, "extended_losses.npy")
        tr = load(d, "extended_train_bleu.npy")
        va = load(d, "extended_val_bleu.npy")
        xs_l = (np.arange(len(loss)) + 1) * 1000 / 1e6
        xs_b = (np.arange(len(tr)) + 1) * 10000 / 1e6
        k = 20  # smooth logged loss for display
        a1.plot(xs_l[k - 1:], np.convolve(loss, np.ones(k) / k, mode="valid"), color=c, ls=ls, lw=1.6, label=lab)
        a2.plot(xs_b, tr, color=c, ls=ls, lw=1.4, label=lab)
        a3.plot(xs_b, va, color=c, ls=ls, lw=1.4, label=lab)
    a1.set_title("training loss (20-pt avg)", fontsize=10)
    a2.set_title("training BLEU", fontsize=10)
    a3.set_title("validation BLEU", fontsize=10)
    a1.set_ylim(0, 3)
    for a in (a2, a3):
        a.set_ylim(0, 1)
    for a in (a1, a2, a3):
        a.set_xlabel("training steps (millions)")
        a.grid(True)
        a.axvline(1.0, color=MUTED, lw=1, ls=":")
    a3.legend(loc="lower right", fontsize=9)
    save(fig, "tf-b-progression",
         "<b>Skinny → Medium → Full Transformer</b> on English→French (Lecture 20's saved results; the "
         "“extended” runs, 3 million single-sentence SGD steps; the dotted line marks the 1-million-step runs "
         "the lecture plots first). Adding feed-forward blocks, residuals and layer norm (Medium) is the big jump "
         "in fit; adding positional encoding and 4-head attention (Full) gives the best validation BLEU. "
         "BLEU is on torchmetrics' 0–1 scale, each point averaged over 1,000 random sentences.",
         alt="Training loss, training BLEU and validation BLEU for the Skinny, Medium and Full Transformers")


# ---------------------------------------------------------------- 2. sinusoidal heatmap
def pe_lecture(L, d):
    pos = np.arange(L)[:, None]
    i = np.arange(d)[None, :]
    ang = pos / (10000 ** (i / d))
    return np.where(i % 2 == 0, np.sin(ang), np.cos(ang))


def pe_paper(L, d):
    pos = np.arange(L)[:, None]
    i = np.arange(d)[None, :]
    ang = pos / (10000 ** ((i - i % 2) / d))
    return np.where(i % 2 == 0, np.sin(ang), np.cos(ang))


def pe_heatmap():
    p = pe_lecture(64, 128)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.imshow(p, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xlabel("embedding dimension i")
    ax.set_ylabel("token position pos")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    save(fig, "tf-b-pe-heatmap",
         "<b>Sinusoidal positional encodings</b> for 64 positions and <i>d</i> = 128, computed with the lecture's "
         "formula (the notebook's “Visualizing” cell). Each row is the vector added to one token's embedding. "
         "Left columns oscillate fast as the position grows; right columns barely change over 64 positions, "
         "so every row is a distinct pattern.",
         alt="Heatmap of sinusoidal positional encodings: fast stripes on the left, smooth on the right")


def pe_curves():
    d = 128
    pos = np.arange(0, 64)
    p = pe_lecture(64, d)
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    for i, c, ls in [(0, ACCENT, "-"), (1, ACCENT, "--"), (20, HI, "-"), (60, ACCENT2, "-"), (100, MUTED, "-")]:
        lam = 2 * np.pi * 10000 ** (i / d)
        ax.plot(pos, p[:, i], color=c, ls=ls, lw=1.6, label=f"i = {i}: wavelength {lam:,.0f}")
    ax.set_xlabel("token position pos")
    ax.set_ylabel("p[i]")
    ax.set_ylim(-1.15, 1.15)
    ax.grid(True)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    save(fig, "tf-b-pe-curves",
         "<b>Five columns of the heatmap</b> as functions of position. Dimension <i>i</i> oscillates with wavelength "
         "2π·10000<sup><i>i</i>/<i>d</i></sup> (lecture formula, <i>d</i> = 128): about 6 positions for <i>i</i> = 0 and 7 for "
         "<i>i</i> = 1 (a sine and a cosine of almost the same frequency), 26 for <i>i</i> = 20, 471 for <i>i</i> = 60. "
         "Over a 25-token sentence the high dimensions are nearly constant.",
         alt="Sine curves of different wavelengths for five positional-encoding dimensions")


# ---------------------------------------------------------------- 3. positional ablation
def pe_ablation():
    runs = [("sinusoidal", "sinusoidal", ACCENT, "-"), ("learned", "learned", HI, "--"), ("none", "none", MUTED, "-")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.9))
    xs = (np.arange(100) + 1) * 10000 / 1e6
    for key, lab, c, ls in runs:
        a1.plot(xs, load("FullTransformer", "PositionalAblation", f"{key}_train_bleu.npy"), color=c, ls=ls, lw=1.4, label=lab)
        a2.plot(xs, load("FullTransformer", "PositionalAblation", f"{key}_val_bleu.npy"), color=c, ls=ls, lw=1.4, label=lab)
    for a, t in ((a1, "training BLEU"), (a2, "validation BLEU")):
        a.set_title(t, fontsize=10)
        a.set_xlabel("training steps (millions)")
        a.set_ylim(0, 0.85)
        a.grid(True)
    a2.legend(loc="lower right", fontsize=9)
    save(fig, "tf-b-pe-ablation",
         "<b>Positional-encoding ablation</b> (Lecture 20's saved results, Full Transformer, 1 million steps). "
         "Sinusoidal and learned encodings track each other; with no positional information validation BLEU "
         "levels off about 0.05 lower.",
         alt="Training and validation BLEU for sinusoidal, learned and no positional encoding")


# ---------------------------------------------------------------- 4. teacher forcing vs inference
def shift_right():
    W, H = 720, 250
    s = SVG(W, H, font_size=13)
    tgt = ["Tu", "ne", "devrais", "pas", "aller", "à", "l'école", ".", "<EOS>"]
    inp = ["<SOS>"] + tgt[:-1]
    x0, cw = 70, 70
    s.text(10, 30, "decoder", size=12, anchor="start", color="var(--muted)")
    s.text(10, 46, "input", size=12, anchor="start", color="var(--muted)")
    s.text(10, 200, "target", size=12, anchor="start", color="var(--muted)")
    for k, (a, b) in enumerate(zip(inp, tgt)):
        x = x0 + k * cw
        s.rect(x, 24, cw - 8, 28, rx=4, fill="var(--paper)", stroke="var(--accent)" if k == 0 else "currentColor")
        s.text(x + (cw - 8) / 2, 38, a, font=MONO, size=11, color="var(--accent)" if k == 0 else "currentColor")
        s.rect(x, 186, cw - 8, 28, rx=4, fill="var(--paper)", stroke="var(--hi)" if b == "<EOS>" else "currentColor")
        s.text(x + (cw - 8) / 2, 200, b, font=MONO, size=11, color="var(--hi)" if b == "<EOS>" else "currentColor")
        arrow(s, x + (cw - 8) / 2, 56, x + (cw - 8) / 2, 180, color="var(--muted)", width=1.2)
        s.text(x + (cw - 8) / 2, 120, f"t={k}", size=10, color="var(--muted)", halo=True)
    s.text(x0 + 4.5 * cw - 4, 238, "position t sees inputs 0..t only (causal mask) — all 9 predictions in one pass",
           size=12, color="var(--muted)")
    write("tf-b-shift-right", s.figure(
        "<b>Teacher forcing in the Transformer.</b> Training feeds the target sentence shifted right by one "
        "(prepend <code>&lt;SOS&gt;</code>, drop <code>&lt;EOS&gt;</code>) and asks position <i>t</i> to predict target "
        "token <i>t</i>. The causal mask stops position <i>t</i> from looking at inputs after <i>t</i>, so one forward pass "
        "computes all nine next-token predictions at once. (Sentence from the Lecture 19 notebook.)",
        alt="Shifted-right decoder inputs above, target tokens below, one arrow per position"))


# ---------------------------------------------------------------- 5. multi-head shape trace
def mha_shapes():
    W, H = 720, 300
    s = SVG(W, H, font_size=13)

    def box(x, y, w, h, t1, t2=None, color="currentColor"):
        s.rect(x, y, w, h, rx=6, fill="var(--paper)", stroke=color, width=1.5)
        if t2 is None:
            s.text(x + w / 2, y + h / 2, t1, font=MONO, size=11, color=color)
        else:
            s.text(x + w / 2, y + h / 2 - 8, t1, font=MONO, size=11, color=color)
            s.text(x + w / 2, y + h / 2 + 9, t2, font=MONO, size=10, color="var(--muted)")

    # inputs
    box(20, 30, 110, 40, "Q", "(N, 64)")
    box(20, 130, 110, 40, "K", "(M, 64)")
    box(20, 230, 110, 40, "V", "(M, 128)")
    # heads
    for j in range(4):
        y = 22 + j * 66
        col = "var(--accent)" if j == 0 else "currentColor"
        box(200, y, 230, 52, f"head {j + 1}: Linear 64→16, 64→16,", "128→32; softmax(QKᵀ/√16)V", color=col)
        arrow(s, 132, 50, 198, y + 20, color="var(--muted)", width=1)
        arrow(s, 132, 150, 198, y + 26, color="var(--muted)", width=1)
        arrow(s, 132, 250, 198, y + 32, color="var(--muted)", width=1)
        arrow(s, 432, y + 26, 488, 150, color="var(--accent)" if j == 0 else "var(--muted)", width=1.2)
    s.text(315, 290, "each head output: (N, 32)", size=11, color="var(--muted)")
    box(492, 128, 90, 44, "concat", "(N, 128)")
    arrow(s, 584, 150, 610, 150)
    box(614, 128, 92, 44, "W_O", "128→128", color="var(--accent)")
    s.text(660, 196, "output (N, 128)", size=11, color="var(--muted)")
    write("tf-b-mha-shapes", s.figure(
        "<b>Shapes in the course's <code>MultiheadAttention</code></b> (embedding 128, hidden 64, 4 heads). "
        "Each head projects the already-projected queries and keys from 64 to 16 dimensions and the values from 128 "
        "to 32, runs scaled dot-product attention with √16, and returns (<i>N</i>, 32). Concatenating 4 heads "
        "gives back (<i>N</i>, 128), and <code>output_linear</code> (<i>W</i><sub>O</sub>) mixes them. For self-attention "
        "<i>M</i> = <i>N</i>; for cross-attention <i>M</i> is the source length.",
        alt="Q, K and V fanning out to four attention heads, concatenated and projected by W_O"))


# ---------------------------------------------------------------- 6. layer norm vs batch norm axes
def norm_axes():
    W, H = 720, 230
    s = SVG(W, H, font_size=13)
    rows, cols, c = 4, 6, 26
    for k, (title, mode, x0) in enumerate([("BatchNorm1d", "col", 150), ("LayerNorm", "row", 500)]):
        s.text(x0 + cols * c / 2, 20, title, font=MONO, size=13, weight="600")
        for r in range(rows):
            for q in range(cols):
                hit = (mode == "col" and q == 2) or (mode == "row" and r == 1)
                s.rect(x0 + q * c, 36 + r * c, c - 2, c - 2, rx=2,
                       fill="var(--accent)" if hit else "var(--paper)", fill_opacity=0.55 if hit else None,
                       stroke="var(--accent)" if hit else "currentColor", width=1)
        s.text(x0 + cols * c / 2, 36 + rows * c + 16, "features (d)  →", size=11, color="var(--muted)")
        s.text(x0 - 10, 36 + rows * c / 2, "tokens / examples ↓", size=11, color="var(--muted)", anchor="end")
        note = ("one mean, one variance per feature,\nover the batch; running stats at eval" if mode == "col"
                else "one mean, one variance per token,\nover its d features; same at train and eval")
        for j, line in enumerate(note.split("\n")):
            s.text(x0 + cols * c / 2, 180 + j * 17, line, size=12)
    write("tf-b-norm-axes", s.figure(
        "<b>Which numbers share a mean and variance.</b> For a (tokens × <i>d</i>) activation matrix, "
        "batch norm normalizes each <b>column</b> (one feature across all examples in the batch); layer norm "
        "normalizes each <b>row</b> (all <i>d</i> features of one token), so it needs no batch and behaves the same "
        "in training and evaluation.",
        alt="Two grids: a highlighted column for batch norm, a highlighted row for layer norm"))


if __name__ == "__main__":
    if L20 is None:
        sys.exit("Lecture 20 results not found; pass the Lecture20 folder as the first argument")
    progression()
    pe_heatmap()
    pe_curves()
    pe_ablation()
    shift_right()
    mha_shapes()
    norm_axes()
    # consistency checks against the numbers quoted on the pages
    assert abs(load("FullTransformer", "extended_val_bleu.npy")[-5:].mean() - 0.512) < 0.001
    assert np.allclose(load("FullTransformer", "val_bleu.npy"),
                       load("FullTransformer", "PositionalAblation", "sinusoidal_val_bleu.npy"))
    p1, p2 = pe_lecture(64, 128), pe_paper(64, 128)
    assert np.allclose(p1[:, 0::2], p2[:, 0::2]) and not np.allclose(p1[:, 1::2], p2[:, 1::2])
    print("ok", L20)
