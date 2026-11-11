#!/usr/bin/env python3
"""Figures for the Lecture 21 pages (PyTorch Transformer layers, <CLS> classification,
Vision Transformer). Run: python3 -B tools/figs_tf_d.py

Writes tools/figs/tf-d-*.html; then python3 tools/assemble.py content/7-transformers/<page>.md
Diagrams are drawn from the lecture's code; the one plot is computed from the patch arithmetic.
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


def arrow(s, x1, y1, x2, y2, color="currentColor", width=1.5, dash=None):
    s.line(x1, y1, x2, y2, color=color, width=width, dash=dash)
    v = np.array([x2 - x1, y2 - y1], float)
    v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]])
    tip = np.array([x2, y2])
    a, b = tip - 7 * v + 3.5 * n, tip - 7 * v - 3.5 * n
    s.add(f'<path d="M{tip[0]:.1f},{tip[1]:.1f} L{a[0]:.1f},{a[1]:.1f} L{b[0]:.1f},{b[1]:.1f} Z" fill="{color}"/>')


def box(s, cx, cy, w, h, label, sub=None, fill=None, stroke="currentColor", mono=False, size=13, dash=None):
    s.rect(cx - w / 2, cy - h / 2, w, h, rx=6, stroke=stroke, fill=fill or "var(--paper)",
           fill_opacity=0.14 if fill else None, dash=dash)
    if sub:
        s.text(cx, cy - 8, label, size=size, font=MONO if mono else None or "inherit")
        s.text(cx, cy + 9, sub, size=size - 2.5, color="var(--muted)")
    else:
        s.text(cx, cy, label, size=size, font=MONO if mono else "inherit")


# ---------------------------------------------------------------- 1. what the PyTorch layers cover
def layer_stack():
    s = SVG(760, 430, font_size=13)
    ex, dx = 190, 570
    # encoder column
    s.text(ex, 18, "encoder", size=14, weight=600)
    box(s, ex, 400, 200, 30, "source token ids")
    box(s, ex, 345, 220, 34, "nn.Embedding + positional", "you write this")
    s.rect(ex - 160, 150, 320, 150, rx=10, stroke="var(--accent)", fill="var(--accent)", fill_opacity=0.06, dash="5 4")
    box(s, ex, 262, 220, 34, "self-attention", "+ dropout, Add & Norm")
    box(s, ex, 206, 220, 34, "feed-forward (d_ff)", "+ dropout, Add & Norm")
    box(s, ex, 95, 200, 30, "z_enc   (N, d_emb)", mono=True, size=12)
    arrow(s, ex, 385, ex, 364)
    arrow(s, ex, 328, ex, 281)
    arrow(s, ex, 245, ex, 225)
    arrow(s, ex, 189, ex, 112)
    # decoder column
    s.text(dx, 18, "decoder", size=14, weight=600)
    box(s, dx, 400, 230, 30, "target ids, shifted right")
    box(s, dx, 345, 220, 34, "nn.Embedding + positional", "you write this")
    s.rect(dx - 160, 92, 320, 208, rx=10, stroke="var(--accent)", fill="var(--accent)", fill_opacity=0.06, dash="5 4")
    box(s, dx, 262, 220, 34, "masked self-attention", "tgt_mask = causal mask")
    box(s, dx, 206, 220, 34, "cross-attention", "memory = z_enc")
    box(s, dx, 150, 220, 34, "feed-forward (d_ff)", "each + dropout, Add & Norm")
    box(s, dx, 48, 230, 30, "nn.Linear(d_emb, vocab) → logits", size=12)
    arrow(s, dx, 385, dx, 364)
    arrow(s, dx, 328, dx, 281)
    arrow(s, dx, 245, dx, 225)
    arrow(s, dx, 189, dx, 169)
    arrow(s, dx, 133, dx, 65)
    s.text(ex, 167, "nn.TransformerEncoderLayer × n_layers", size=12, color="var(--accent)", font=MONO, halo=True)
    s.text(dx, 109, "nn.TransformerDecoderLayer × n_layers", size=12, color="var(--accent)", font=MONO, halo=True)
    # encoder output feeds every decoder layer's cross-attention
    s.path(f"M{ex + 100},95 L{ex + 185},95 L{ex + 185},206 L{dx - 118},206", color="var(--hi)", width=1.6)
    arrow(s, dx - 125, 206, dx - 111, 206, color="var(--hi)")
    s.text((ex + dx) / 2 + 12, 190, "keys, values", size=12, color="var(--hi)")
    write("tf-d-layer-stack", s.figure(
        "<b>What PyTorch's two layer classes cover.</b> Each dashed box is one object: an "
        "<code>nn.TransformerEncoderLayer</code> is self-attention plus the feed-forward network, each wrapped in dropout and "
        "Add &amp; Norm; an <code>nn.TransformerDecoderLayer</code> adds cross-attention on the encoder output (its "
        "<code>memory</code> argument). Embeddings, positional encoding, the causal mask and the final prediction layer are "
        "still yours to write, and stacking is a loop over an <code>nn.ModuleList</code>.",
        alt="Encoder and decoder columns showing which blocks are inside the PyTorch layer classes"))


# ---------------------------------------------------------------- 2. the <CLS> token
def cls_token():
    s = SVG(700, 330, font_size=13)
    toks = ["<CLS>", "love", "these", "shoes", "!"]
    xs = [110 + 120 * i for i in range(5)]
    for i, (t, x) in enumerate(zip(toks, xs)):
        hi = i == 0
        col = "var(--hi)" if hi else "currentColor"
        s.rect(x - 42, 282, 84, 30, rx=6, stroke=col, fill="var(--hi)" if hi else "var(--paper)", fill_opacity=0.14 if hi else None)
        s.text(x, 297, t, size=13, font=MONO, color=col)
        s.text(x, 322, f"position {i}", size=10.5, color="var(--muted)")
        arrow(s, x, 280, x, 243, color=col)
        arrow(s, x, 158, x, 130, color=col if hi else "var(--muted)")
        s.rect(x - 30, 100, 60, 28, rx=6, stroke=col if hi else "var(--muted)", fill="var(--hi)" if hi else "var(--paper)",
               fill_opacity=0.14 if hi else None)
        s.text(x, 114, f"z[{i}]", size=12.5, font=MONO, color=col if hi else "var(--muted)")
    s.rect(50, 160, 600, 82, rx=10, stroke="var(--accent)", fill="var(--accent)", fill_opacity=0.06)
    s.text(350, 185, "embedding + positional encoding", size=13)
    s.text(350, 207, "n_layers × nn.TransformerEncoderLayer", size=12.5, font=MONO, color="var(--accent)")
    s.text(350, 227, "self-attention: every token mixes with every other, so z[0] can collect the whole review", size=11.5,
           color="var(--muted)")
    arrow(s, xs[0], 98, xs[0], 66, color="var(--hi)")
    s.rect(xs[0] - 85, 34, 170, 30, rx=6, stroke="var(--hi)", fill="var(--paper)")
    s.text(xs[0], 49, "nn.Linear(d_emb, 2)", size=12.5, font=MONO, color="var(--hi)")
    arrow(s, xs[0] + 87, 49, xs[0] + 130, 49, color="var(--hi)")
    s.text(xs[0] + 136, 49, "logits: negative, positive", size=13, anchor="start")
    s.text(470, 82, "the other outputs are not used for the prediction", size=11.5, color="var(--muted)")
    write("tf-d-cls-token", s.figure(
        "<b>Classifying with a <code>&lt;CLS&gt;</code> token.</b> A reserved token (vocabulary index 0) is put in front of "
        "every review, so a review of <i>L</i> words enters the encoder as <i>L</i> + 1 tokens. After the encoder layers, only "
        "the output at position 0 goes to the prediction layer (<code>z_enc[:, 0]</code> in the lecture's code). Training "
        "teaches the model to gather what the label needs into that one position.",
        alt="Five tokens starting with CLS pass through an encoder; only the first output feeds a linear layer"))


# ---------------------------------------------------------------- 3. an image as tokens
def vit_patches():
    s = SVG(720, 300, font_size=13)
    gx, gy, c = 24, 40, 20
    for i in range(11):
        s.line(gx + i * c, gy, gx + i * c, gy + 10 * c, width=0.8, color="var(--muted)")
        s.line(gx, gy + i * c, gx + 10 * c, gy + i * c, width=0.8, color="var(--muted)")
    s.rect(gx, gy, 10 * c, 10 * c, stroke="currentColor", width=1.4)
    s.rect(gx + 3 * c, gy + 2 * c, c, c, stroke="var(--hi)", fill="var(--hi)", fill_opacity=0.25, width=1.8)
    s.text(gx + 5 * c, gy - 14, "160 × 160 RGB image", size=12.5)
    s.text(gx + 5 * c, gy + 10 * c + 16, "10 × 10 = 100 patches of 16 × 16", size=12)
    s.text(gx + 5 * c, gy + 10 * c + 33, "raster order: row by row", size=11, color="var(--muted)")
    arrow(s, gx + 4 * c + 2, gy + 2.5 * c, 280, gy + 2.5 * c, color="var(--hi)")
    s.rect(282, gy + 2.5 * c - 15, 150, 30, rx=6, stroke="var(--hi)", fill="var(--paper)")
    s.text(357, gy + 2.5 * c, "flatten: 3·16·16 = 768", size=12, color="var(--hi)")
    arrow(s, 357, gy + 2.5 * c + 15, 357, gy + 2.5 * c + 45)
    s.rect(282, gy + 2.5 * c + 47, 150, 30, rx=6, stroke="currentColor", fill="var(--paper)")
    s.text(357, gy + 2.5 * c + 62, "nn.Linear(768, 768)", size=12, font=MONO)
    s.text(357, gy + 2.5 * c + 92, "same weights for every patch", size=11, color="var(--muted)")
    arrow(s, 432, gy + 2.5 * c + 62, 468, gy + 2.5 * c + 62)
    # token row
    tx, ty, w = 470, gy + 2.5 * c + 47, 34
    labels = ["CLS", "1", "2", "…", "100"]
    for i, lab in enumerate(labels):
        hi = i == 0
        s.rect(tx + i * (w + 6), ty, w, 30, rx=5, stroke="var(--hi)" if hi else "currentColor",
               fill="var(--hi)" if hi else "var(--paper)", fill_opacity=0.14 if hi else None)
        s.text(tx + i * (w + 6) + w / 2, ty + 15, lab, size=11.5, color="var(--hi)" if hi else "currentColor")
    s.text(tx + 97, ty + 46, "+ learned position embedding 0…100", size=11.5)
    s.text(tx + 97, ty + 62, "101 tokens of width 768", size=11.5, color="var(--muted)")
    arrow(s, tx + 97, ty - 2, tx + 97, ty - 30)
    s.rect(tx - 4, ty - 62, 202, 30, rx=6, stroke="var(--accent)", fill="var(--accent)", fill_opacity=0.08)
    s.text(tx + 97, ty - 47, "Transformer encoder layers", size=12.5, color="var(--accent)")
    arrow(s, tx + 17, ty - 64, tx + 17, ty - 92, color="var(--hi)")
    s.text(tx + 97, ty - 104, "output at position 0 → MLP head → class scores", size=11.5, color="var(--hi)")
    write("tf-d-vit-patches", s.figure(
        "<b>An image as a sequence of tokens</b> (the lecture's Vision Transformer on a 160 × 160 crop). The image is cut "
        "into non-overlapping 16 × 16 patches; each patch is flattened to 3 · 16 · 16 = 768 numbers and passed through one "
        "shared linear layer, which plays the role <code>nn.Embedding</code> plays for words. A learned "
        "<code>&lt;CLS&gt;</code> vector is put in front, a learned position vector is added to each of the 101 tokens, and "
        "from there the model is the text classifier again.",
        alt="A 10 by 10 grid of patches, one patch flattened and projected, then a row of tokens starting with CLS"))


# ---------------------------------------------------------------- 4. patch size trade-off
def patch_cost():
    P = np.array([4, 5, 8, 10, 16, 20, 32, 40, 80])
    n = (160 // P) ** 2 + 1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a1.loglog(P, n, "o-", color=ACCENT, ms=4, label="tokens  $(160/P)^2 + 1$", base=2)
    a1.loglog(P, n.astype(float) ** 2, "s--", color=HI, ms=4, label="attention weights  $n^2$", base=2)
    a1.set_xlabel("patch side P (pixels)")
    a1.set_xticks(P[[0, 2, 4, 6, 8]], [str(p) for p in P[[0, 2, 4, 6, 8]]])
    a1.set_yscale("log", base=10)
    a1.legend(fontsize=8.5, loc="upper right")
    a1.set_title("sequence length", fontsize=10)
    a1.grid(True, which="major")
    a2.loglog(P, 3 * P ** 2, "o-", color=ACCENT2, ms=4, base=2)
    a2.set_yscale("log", base=10)
    a2.set_xticks(P[[0, 2, 4, 6, 8]], [str(p) for p in P[[0, 2, 4, 6, 8]]])
    a2.set_xlabel("patch side P (pixels)")
    a2.set_title("token length  $3P^2$", fontsize=10)
    a2.grid(True, which="major")
    assert n[4] == 101 and 3 * 16 ** 2 == 768
    save(fig, "tf-d-patch-cost",
         "<b>The patch-size trade-off</b> for a 160 × 160 RGB image, computed from the patch arithmetic. Small patches give "
         "many short tokens: <i>P</i> = 8 means 401 tokens and 160,801 attention weights per head and layer. Large patches "
         "give few long tokens: <i>P</i> = 32 means 26 tokens of 3,072 numbers each, and less spatial detail. The lecture uses "
         "<i>P</i> = 16: 101 tokens of length 768.",
         alt="Number of tokens and attention weights falling, and token length rising, as the patch size grows")


if __name__ == "__main__":
    layer_stack()
    cls_token()
    vit_patches()
    patch_cost()
    print("ok")
