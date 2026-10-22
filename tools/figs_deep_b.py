#!/usr/bin/env python3
"""Figures for Lecture 16 (branching, residuals, FCNs, segmentation, augmentation, batch norm, dropout).

Writes tools/figs/deepb-*.html; then run tools/assemble.py on the pages.
Numbers in the result charts are copied from the saved outputs of L16_Solution.ipynb.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figlib import SVG, MONO  # noqa: E402
from mplfig import plt, INK, ACCENT, HI, MUTED, save as mpl_save  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html + "\n")
    print("wrote", name)


def box(s, cx, cy, w, h, label, color="currentColor", size=12, fill_opacity=0.12, dash=None, mono=False):
    s.rect(cx - w / 2, cy - h / 2, w, h, rx=6, stroke=color, fill=color if color != "currentColor" else "none",
           fill_opacity=fill_opacity if color != "currentColor" else None, dash=dash)
    if label:
        if mono:
            s.text(cx, cy, label, size=size, font=MONO)
        else:
            s.text(cx, cy, label, size=size)


def arrow(s, x1, y1, x2, y2, color="var(--muted)", dash=None):
    s.line(x1, y1, x2, y2, color=color, width=1.4, arrow=True, dash=dash)


def seg(s, x1, y1, x2, y2, color="var(--muted)", dash=None):
    s.line(x1, y1, x2, y2, color=color, width=1.4, dash=dash)


def num(v):
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def grid(s, x, y, values, cell=24, title=None, colors=None, size=12, show=None):
    """Draw a matrix of numbers; colors[r][c] fills a cell; show(r, c) False leaves it blank."""
    values = np.asarray(values, dtype=float)
    R, C = values.shape
    if title:
        s.text(x + C * cell / 2, y - 12, title, size=12)
    for r in range(R):
        for c in range(C):
            col = colors[r][c] if colors is not None else None
            visible = show is None or show(r, c)
            s.rect(x + c * cell, y + r * cell, cell, cell, stroke="var(--muted)", width=0.8,
                   fill=col if (col and visible) else "none", fill_opacity=0.28 if (col and visible) else None)
            if visible:
                s.text(x + (c + 0.5) * cell, y + (r + 0.5) * cell, num(values[r, c]), size=size)
    return x + C * cell


ONE, KK, POOL, RELU = "var(--accent2)", "var(--accent)", "var(--hi)", "var(--muted)"

# ====================================================================== Inception block
s = SVG(760, 420, font_size=13)
xs = [105, 285, 475, 665]
box(s, 385, 28, 230, 30, "input: 64 × H × W", "var(--muted)")
seg(s, 385, 43, 385, 62)
seg(s, xs[0], 62, xs[-1], 62)
branches = [
    ("branch1", [("1×1 conv, 64 → 32", ONE)], "2,080"),
    ("branch2", [("1×1 conv, 64 → 32", ONE), ("ReLU", RELU), ("3×3 conv, pad 1, 32 → 32", KK)], "11,328"),
    ("branch3", [("1×1 conv, 64 → 32", ONE), ("ReLU", RELU), ("5×5 conv, pad 2, 32 → 32", KK)], "27,712"),
    ("branch4", [("3×3 max pool, stride 1, pad 1", POOL), ("1×1 conv, 64 → 32", ONE)], "2,080"),
]
top, step, bw, bh = 95, 46, 172, 28
for x, (name, layers, params) in zip(xs, branches):
    arrow(s, x, 62, x, top - bh / 2)
    for k, (label, color) in enumerate(layers):
        y = top + k * step
        box(s, x, y, bw if color != RELU else 70, bh, label, color, size=11.5)
        if k:
            arrow(s, x, y - step + bh / 2, x, y - bh / 2)
    last = top + (len(layers) - 1) * step
    arrow(s, x, last + bh / 2, x, 230)
    s.text(x, 243, "32 × H × W", size=12.5)
    s.text(x, 261, f"{name} · {params} params", size=11.5, color="var(--muted)")
    seg(s, x, 272, x, 292)
seg(s, xs[0], 292, xs[-1], 292)
arrow(s, 385, 292, 385, 311)
box(s, 385, 327, 330, 30, "torch.cat(branches, dim=1): 128 × H × W", "var(--muted)")
arrow(s, 385, 342, 385, 362)
box(s, 385, 377, 80, 28, "ReLU", RELU)
for k, (label, color) in enumerate([("1×1 conv (mixes channels)", ONE), ("3×3 / 5×5 conv", KK), ("max pool", POOL)]):
    s.rect(560, 352 + 20 * k, 14, 12, rx=2, stroke=color, fill=color, fill_opacity=0.25)
    s.text(582, 358 + 20 * k, label, size=11.5, anchor="start")
save("deepb-inception", s.figure(
    "<strong>The Inception block from Lecture 16</strong> (64 input channels, 32 per branch). Every branch keeps "
    "H × W, so <code>torch.cat</code> along the channels works: 4 × 32 = 128 channels. Total 43,200 parameters; one "
    "5×5 convolution from 64 to 128 channels would need 204,928.",
    alt="Inception block: four parallel branches of 1x1, 3x3, 5x5 convolutions and max pooling, concatenated"))

# ====================================================================== bottleneck arithmetic
s = SVG(660, 250, font_size=13)


def slab(x, y, ch, color, label):
    h = ch * 0.9
    s.rect(x, y - h / 2, 24, h, rx=3, stroke=color, fill=color, fill_opacity=0.22)
    s.text(x + 12, y + h / 2 + 12, label, size=11.5, color="var(--muted)")


s.text(20, 18, "Directly", size=13, anchor="start", weight=600)
slab(30, 70, 64, "var(--muted)", "64 ch")
arrow(s, 60, 70, 228, 70)
s.text(144, 58, "5×5 conv", size=12)
slab(236, 70, 32, KK, "32 ch")
s.text(290, 70, "64·32·25 + 32 = 51,232 parameters", size=12.5, anchor="start")
s.text(20, 134, "With a 1×1 convolution first", size=13, anchor="start", weight=600)
slab(30, 190, 64, "var(--muted)", "64 ch")
arrow(s, 60, 190, 118, 190)
s.text(89, 178, "1×1", size=12)
slab(126, 190, 32, ONE, "32 ch")
arrow(s, 156, 190, 228, 190)
s.text(192, 178, "5×5", size=12)
slab(236, 190, 32, KK, "32 ch")
s.text(290, 182, "(64·32 + 32) + (32·32·25 + 32)", size=12.5, anchor="start")
s.text(290, 202, "= 2,080 + 25,632 = 27,712 parameters (46% fewer)", size=12.5, anchor="start")
save("deepb-bottleneck", s.figure(
    "<strong>The GoogLeNet trick.</strong> A k×k convolution has C<sub>out</sub>·C<sub>in</sub>·k² weights, so "
    "shrinking the channels with a cheap 1×1 convolution first makes the expensive kernel cheaper. Slab "
    "height = number of channels. The 3×3 branch saves 39% the same way (18,464 → 11,328).",
    alt="Parameter count of a 5x5 branch with and without a 1x1 bottleneck"))

# ====================================================================== residual block
s = SVG(560, 430, font_size=13)
cx, sx = 200, 420
box(s, cx, 28, 150, 30, "x: C_in × H × W", "var(--muted)")
seg(s, cx, 43, cx, 66)
seg(s, cx, 66, sx, 66)
arrow(s, cx, 66, cx, 86)
body = [("3×3 conv, stride s", KK), ("BatchNorm2d", ONE), ("ReLU", RELU), ("3×3 conv", KK), ("BatchNorm2d", ONE)]
y0, st = 100, 44
for k, (label, color) in enumerate(body):
    y = y0 + k * st
    box(s, cx, y, 150 if color != RELU else 70, 28, label, color, size=12)
    if k:
        arrow(s, cx, y - st + 14, cx, y - 14)
s.text(cx - 100, y0 + 2 * st, "F(x)", size=15, italic=True, anchor="end")
s.rect(cx - 88, y0 - 22, 176, 4 * st + 44, rx=10, stroke="var(--muted)", dash="4 4", width=1)
arrow(s, sx, 66, sx, 165)
box(s, sx, 195, 180, 64, "", "var(--muted)", dash="5 4", fill_opacity=0.06)
s.text(sx, 177, "shortcut: identity, or", size=12)
s.text(sx, 195, "1×1 conv, stride s + BN", size=12)
s.text(sx, 213, "(if channels or size change)", size=11, color="var(--muted)")
plus_y = y0 + 4 * st + 50
arrow(s, cx, y0 + 4 * st + 14, cx, plus_y - 16)
seg(s, sx, 227, sx, plus_y)
arrow(s, sx, plus_y, cx + 16, plus_y)
s.circle(cx, plus_y, 15, stroke="currentColor", fill="var(--paper)")
s.text(cx, plus_y, "+", size=20)
arrow(s, cx, plus_y + 15, cx, plus_y + 36)
box(s, cx, plus_y + 50, 70, 28, "ReLU", RELU)
s.text(cx + 60, plus_y + 50, "y = ReLU(F(x) + x)", size=14, anchor="start", italic=True)
save("deepb-residual-block", s.figure(
    "<strong>ResNet's residual block</strong>, as built in Lecture 16 (<code>ResidualBlock</code>). The body learns a "
    "correction F(x); the shortcut carries x around it. Adding needs equal C, H and W, so when the body changes the "
    "channels or uses stride 2, the shortcut becomes a 1×1 convolution with the same stride (projection shortcut).",
    alt="Residual block: two 3x3 conv + batch norm layers, a shortcut, an addition, and a ReLU"))

# ====================================================================== U-Net
s = SVG(720, 400, font_size=13)
L, R = 140, 580
levels = [80, 190, 300]
box(s, L, 24, 120, 26, "input 1×64×64", "var(--muted)", size=12)
arrow(s, L, 37, L, levels[0] - 22)
enc = [("enc1: 2× (3×3 conv, ReLU)", "16 × 64 × 64"), ("enc2: pool, 2× conv", "32 × 32 × 32"),
       ("enc3: pool, 2× conv", "64 × 16 × 16")]
for k, (lab, shape) in enumerate(enc):
    x = L if k < 2 else 360
    y = levels[k]
    box(s, x, y, 190, 40, "", KK)
    s.text(x, y - 8, lab, size=12)
    s.text(x, y + 10, shape, size=12, weight=600)
arrow(s, L, levels[0] + 20, L, levels[1] - 20)
s.text(L + 10, (levels[0] + levels[1]) / 2, "MaxPool2d(2)", size=11, anchor="start", color="var(--muted)")
s.path(f"M {L} {levels[1] + 20} L {L} {levels[2]} L {360 - 95} {levels[2]}", color="var(--muted)", width=1.4, arrow=True)
s.text(L + 10, (levels[1] + levels[2]) / 2 + 10, "MaxPool2d(2)", size=11, anchor="start", color="var(--muted)")
dec = [("dec1: 3×3 conv (32 → 16)", "16 × 64 × 64"), ("dec2: 3×3 conv (64 → 32)", "32 × 32 × 32")]
for k, (lab, shape) in enumerate(dec):
    y = levels[k]
    box(s, R, y, 190, 40, "", KK)
    s.text(R, y - 8, lab, size=12)
    s.text(R, y + 10, shape, size=12, weight=600)
s.path(f"M {360 + 95} {levels[2]} L {R} {levels[2]} L {R} {levels[1] + 20}", color="var(--muted)", width=1.4, arrow=True)
s.text(R - 10, (levels[1] + levels[2]) / 2 + 10, "ConvTranspose2d(64, 32, 2, stride=2)", size=11, anchor="end", color="var(--muted)")
arrow(s, R, levels[1] - 20, R, levels[0] + 20)
s.text(R - 10, (levels[0] + levels[1]) / 2, "ConvTranspose2d(32, 16, 2, stride=2)", size=11, anchor="end", color="var(--muted)")
for k in (0, 1):
    arrow(s, L + 95, levels[k], R - 95, levels[k], color="var(--hi)", dash="6 4")
    s.text(360, levels[k] - 11, "skip (U-Net only): torch.cat(dim=1)", size=11.5, color="var(--hi)", halo=True)
    s.text(360, levels[k] + 12, ["16 + 16 = 32 channels", "32 + 32 = 64 channels"][k], size=11, color="var(--muted)", halo=True)
arrow(s, R, levels[0] - 20, R, 37)
box(s, R, 24, 200, 26, "1×1 conv → 1 × 64 × 64 logits", "var(--muted)", size=12)
s.text(360, 372, "Encoder shrinks H, W and grows channels; the decoder reverses it. "
       "Without the red skips this is a plain encoder-decoder.", size=12, color="var(--muted)")
save("deepb-unet", s.figure(
    "<strong>The encoder-decoder and U-Net from Lecture 16</strong>, with every shape for a 1×64×64 input. "
    "The skips concatenate the encoder's fine features into the decoder, so the decoder's first convolution at each "
    "level takes twice the channels. Encoder-decoder: 93,665 parameters, validation IoU 0.841; U-Net: 105,185 "
    "parameters, IoU 0.868.",
    alt="U-Net diagram with encoder, decoder, transposed convolutions and skip connections"))

# ====================================================================== IoU toy example
t = np.zeros((12, 12), int)
p = np.zeros((12, 12), int)
t[2:9, 2:9] = 1
p[4:11, 4:11] = 1
both, fa, miss = int((t & p).sum()), int(((1 - t) & p).sum()), int((t & (1 - p)).sum())
s = SVG(560, 240, font_size=13)
cell = 16
for r in range(12):
    for c in range(12):
        col = None
        if t[r, c] and p[r, c]:
            col = "var(--accent)"
        elif p[r, c]:
            col = "var(--hi)"
        elif t[r, c]:
            col = "var(--accent2)"
        s.rect(20 + c * cell, 20 + r * cell, cell, cell, stroke="var(--muted)", width=0.6,
               fill=col or "none", fill_opacity=0.55 if col else None)
x0 = 250
for k, (label, n, col) in enumerate([("in both (intersection)", both, "var(--accent)"),
                                     ("predicted only (false alarm)", fa, "var(--hi)"),
                                     ("missed (target only)", miss, "var(--accent2)")]):
    s.rect(x0, 30 + 26 * k, 16, 16, rx=2, stroke=col, fill=col, fill_opacity=0.55)
    s.text(x0 + 26, 38 + 26 * k, f"{label}: {n} pixels", size=12.5, anchor="start")
s.text(x0, 130, f"union = {both} + {fa} + {miss} = {both + fa + miss}", size=13, anchor="start")
s.text(x0, 156, f"IoU = {both} / {both + fa + miss} = {both / (both + fa + miss):.2f}", size=14, anchor="start", weight=600)
s.text(x0, 184, f"pixel accuracy = {144 - fa - miss}/144 = {(144 - fa - miss) / 144:.0%}", size=12.5, anchor="start",
       color="var(--muted)")
s.text(x0, 204, "(the background makes accuracy look good)", size=11.5, anchor="start", color="var(--muted)")
save("deepb-iou", s.figure(
    "<strong>Intersection over union, pixel by pixel</strong> (the toy example from Lecture 16): the target is a 7×7 "
    "square, the prediction the same square shifted by 2 pixels down and right. IoU ignores the correctly-empty "
    "background, so it punishes the shift far more than per-pixel accuracy does.",
    alt="IoU toy example on a 12 by 12 grid"))

# ====================================================================== transposed convolution
x = np.array([[1., 2.], [3., 4.]])
w = np.array([[1., 0.], [2., 1.]])
out = np.zeros((4, 4))
stamps = []
for i in range(2):
    for j in range(2):
        st_ = np.full((4, 4), np.nan)
        st_[2 * i:2 * i + 2, 2 * j:2 * j + 2] = x[i, j] * w
        out[2 * i:2 * i + 2, 2 * j:2 * j + 2] += x[i, j] * w
        stamps.append((f"{int(x[i, j])} × w", st_))
s = SVG(770, 160, font_size=12)
cell = 22
xx = 16
xx = grid(s, xx, 40, x, cell, "input x", [["var(--accent)"] * 2] * 2) + 30
xx = grid(s, xx, 40, w, cell, "kernel w", [["var(--accent2)"] * 2] * 2) + 26
for k, (title, st_) in enumerate(stamps):
    xx = grid(s, xx, 40, np.nan_to_num(st_), cell, title, [["var(--accent)"] * 4] * 4,
              show=lambda r, c, st_=st_: not np.isnan(st_[r, c])) + 8
    s.text(xx + 1, 40 + 2 * cell, "+" if k < 3 else "=", size=16)
    xx += 12
grid(s, xx, 40, out, cell, "output", [["var(--hi)"] * 4] * 4)
s.text(380, 140, "H_out = (H_in − 1)·s − 2p + k = (2 − 1)·2 − 0 + 2 = 4", size=12, color="var(--muted)")
save("deepb-transposed-conv", s.figure(
    "<strong>Transposed convolution, kernel 2, stride 2</strong> (Lecture 16's example). Each input pixel stamps a "
    "copy of the kernel, scaled by its value, at position (2i, 2j) of the output; the stamps add up. Here they sit "
    "side by side, so H and W double. Same result as <code>nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)</code> "
    "with that weight.",
    alt="Transposed convolution of a 2x2 input with a 2x2 kernel and stride 2 giving a 4x4 output"))
assert out.tolist() == [[1, 0, 2, 0], [2, 1, 4, 2], [3, 0, 4, 0], [6, 3, 8, 4]]

# ====================================================================== upsampling


def bilinear_up2(a):
    """PyTorch bilinear, align_corners=False, scale 2."""
    n = a.shape[0]
    src = np.clip((np.arange(2 * n) + 0.5) / 2 - 0.5, 0, n - 1)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    t_ = src - i0
    rows = a[i0] * (1 - t_)[:, None] + a[i1] * t_[:, None]
    return rows[:, i0] * (1 - t_)[None, :] + rows[:, i1] * t_[None, :]


near = np.kron(x, np.ones((2, 2)))
bil = bilinear_up2(x)
s = SVG(600, 150, font_size=12)
cell = 28
xx = grid(s, 20, 36, x, cell, "input", [["var(--accent)"] * 2] * 2) + 40
xx = grid(s, xx, 36, near, cell, "mode='nearest'", [["var(--accent)"] * 4] * 4) + 40
grid(s, xx, 36, bil, cell, "mode='bilinear'", [["var(--accent2)"] * 4] * 4, size=11)
save("deepb-upsample", s.figure(
    "<strong><code>nn.Upsample(scale_factor=2)</code> has no weights.</strong> Nearest copies every value into a 2×2 "
    "block; bilinear (default <code>align_corners=False</code>) blends neighbours. Nearest upsampling equals a "
    "transposed convolution whose 2×2 kernel is all ones.",
    alt="Nearest and bilinear upsampling of a 2x2 grid to 4x4"))
assert bil[0].tolist() == [1, 1.25, 1.75, 2]

# ====================================================================== batch norm worked example
X = np.array([  # (N=2, C=2, 2, 2)
    [[[1, 2], [3, 4]], [[10, 10], [10, 10]]],
    [[[5, 6], [7, 8]], [[20, 20], [20, 20]]],
], dtype=float)
mu = X.mean(axis=(0, 2, 3))
var = X.var(axis=(0, 2, 3))
Y = (X - mu[None, :, None, None]) / np.sqrt(var[None, :, None, None] + 1e-5)
s = SVG(760, 250, font_size=12)
cell = 32
s.text(150, 18, "input batch: N = 2 images, C = 2 channels", size=12.5, weight=600)
s.text(575, 18, "output (γ = 1, β = 0)", size=12.5, weight=600)
cols = ["var(--accent)", "var(--accent2)"]
for c in range(2):
    y = 50 + c * 100
    s.text(14, y + cell, f"ch {c}", size=12, anchor="start", color=cols[c])
    for n in range(2):
        grid(s, 60 + n * 90, y, X[n, c], cell, f"image {n}" if c == 0 else None, [[cols[c]] * 2] * 2)
        grid(s, 470 + n * 105, y, np.round(Y[n, c], 2), 42, f"image {n}" if c == 0 else None, [[cols[c]] * 2] * 2, size=11)
    s.text(330, y + 18, f"μ = {num(mu[c])}, σ² = {num(var[c])}", size=12.5, color=cols[c], weight=600)
    arrow(s, 260, y + cell + 6, 452, y + cell + 6)
    s.text(355, y + cell + 24, f"(x − {num(mu[c])}) / √{num(var[c])}", size=12, color="var(--muted)")
save("deepb-batchnorm", s.figure(
    "<strong>Batch norm statistics are per channel, over the batch and all pixels</strong> (dims 0, 2, 3). Channel 0 "
    "has 8 values 1…8 (μ = 4.5, σ² = 5.25); channel 1 is constant inside each image, so it normalizes to exactly −1 "
    "and +1. Then each channel is rescaled by its own learned γ and shifted by β.",
    alt="Batch normalization worked example with two images and two channels"))

# ====================================================================== result charts (numbers from the notebook)
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
for ax, (title, vals) in zip(axes, [("learning rate 0.02", (13.4, 65.0)), ("learning rate 0.1", (38.7, 68.3))]):
    bars = ax.bar(["no batch norm", "batch norm"], vals, color=[MUTED, ACCENT], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", color=INK)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylim(0, 100)
axes[0].set_ylabel("test accuracy after 3 epochs (%)")
fig.tight_layout()
mpl_save(fig, "deepb-bn-results",
         "<strong>Batch norm makes plain SGD fast and much less sensitive to the learning rate</strong> "
         "(small CNN on 20,000 FashionMNIST images, 3 epochs; numbers from the Lecture 16 notebook).",
         alt="Bar chart of test accuracy with and without batch norm at two learning rates", width=620)

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
for ax, (title, vals) in zip(axes, [("clean labels", (84.1, 84.1)), ("30% wrong labels", (67.4, 77.2))]):
    bars = ax.bar(["no dropout", "dropout"], vals, color=[MUTED, ACCENT], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", color=INK)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylim(50, 92)
axes[0].set_ylabel("final test accuracy (%)")
fig.tight_layout()
mpl_save(fig, "deepb-dropout-results",
         "<strong>Dropout matters when a big model can memorize bad data</strong>: a 784→512→512→10 MLP "
         "(≈670k parameters) trained 100 epochs on 2,000 images. With 30% wrong labels, the model without dropout "
         "fits 100% of the training labels and its test accuracy falls from a peak of 79.4% to 67.4%.",
         alt="Bar chart of final test accuracy with and without dropout", width=620)

names = ["none", "horizontal flip", "random crop (shift)", "rotation ±15°", "shear + scale", "blur",
         "brightness + contrast", "vertical flip (unrealistic)", "all realistic combined"]
clean = [88.2, 84.9, 79.8, 83.1, 84.1, 85.3, 86.3, 84.0, 77.4]
varied = [43.4, 56.1, 57.9, 49.5, 45.8, 42.2, 46.7, 43.6, 72.5]
fig, ax = plt.subplots(figsize=(7.2, 3.9))
yy = np.arange(len(names))[::-1]
for k in range(len(names)):
    ax.plot([varied[k], clean[k]], [yy[k], yy[k]], color=MUTED, lw=1.2)
ax.scatter(clean, yy, color=ACCENT, s=36, zorder=3, label="clean test images")
ax.scatter(varied, yy, color=HI, s=36, zorder=3, label="varied test images")
ax.set_yticks(yy)
ax.set_yticklabels(names)
ax.set_xlim(35, 95)
ax.set_xlabel("test accuracy (%), small CNN, 6,000 training images, 800 steps")
ax.grid(True, axis="x")
ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=10)
fig.tight_layout()
mpl_save(fig, "deepb-aug-grid",
         "<strong>Which augmentation helps?</strong> Each one alone, all realistic ones combined, and an unrealistic "
         "vertical flip, at the same 800 training steps. On clean test images every augmentation costs 2 to 11 points; on "
         "the <em>varied</em> test set (each image changed once by a random mix of realistic augmentations), shifts and "
         "flips help most and the combination adds about 30 points. Numbers from the Lecture 16 notebook.",
         alt="Dot plot of clean and varied test accuracy for each augmentation", width=640)
