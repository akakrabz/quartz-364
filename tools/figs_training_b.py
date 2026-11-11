#!/usr/bin/env python3
"""Figures for the Lecture 22 pages (diagnosing fit, learning-rate schedulers, TensorBoard).
Run: python3 -B tools/figs_training_b.py

Writes tools/figs/train-b-*.html; then python3 tools/assemble.py content/4-pytorch-training/<page>.md
The curves are the six training runs shipped with the lecture (py-complete/Lecture22/logs),
extracted once from their TensorBoard event files by tools/read_tfevents.py into
tools/l22_logs.json. Nothing is retrained or invented here; the one schematic figure says so.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mplfig import ACCENT, ACCENT2, HI, INK, MUTED, plt, save  # noqa: E402

LOGS = json.load(open(os.path.join(HERE, "l22_logs.json")))
RUN = "demb_{}_nlayers_1_nhead_{}_dropout_00"


def series(d_emb, heads, tag):
    v = LOGS[RUN.format(d_emb, heads)][tag]
    return np.array([s for s, _ in v]), np.array([x for _, x in v])


def marks(ax, text=False):
    for e in (70, 85):
        ax.axvline(e, color=MUTED, lw=1, ls=":")
    if text:
        ax.text(70.6, text, "lr ÷10", fontsize=8, color=MUTED, rotation=90, va="top")
        ax.text(85.6, text, "lr ÷10", fontsize=8, color=MUTED, rotation=90, va="top")


# ---------------------------------------------------------------- 1. embedding-dimension sweep
def emb_sweep():
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(9.8, 3.1))
    for d, c, ls in [(32, MUTED, "-"), (64, ACCENT2, "--"), (128, HI, "-"), (256, ACCENT, "-")]:
        e, loss = series(d, 4, "Loss")
        _, tr = series(d, 4, "BLEU Scores/Train BLEU")
        _, va = series(d, 4, "BLEU Scores/Validation BLEU")
        a1.semilogy(e, loss, color=c, ls=ls, lw=1.6, label=f"$d_{{emb}}$ = {d}")
        a2.plot(e, tr, color=c, ls=ls, lw=1.5)
        a3.plot(e, va, color=c, ls=ls, lw=1.5)
    a1.set_title("training loss (log scale)", fontsize=10)
    a2.set_title("training BLEU", fontsize=10)
    a3.set_title("validation BLEU", fontsize=10)
    a1.legend(fontsize=8.5, loc="upper right")
    for a in (a2, a3):
        a.set_ylim(0, 1)
    for a in (a1, a2, a3):
        a.set_xlabel("epoch")
        a.set_xlim(0, 100)
        a.grid(True)
        marks(a)
    save(fig, "train-b-emb-sweep",
         "<b>The lecture's embedding-dimension runs</b> (English→French Transformer, 1 layer, 4 heads, no dropout, "
         "100 epochs; read from the TensorBoard logs shipped with Lecture 22). Dotted lines: the learning rate drops "
         "tenfold after epochs 70 and 85. <i>d</i><sub>emb</sub> = 32 never fits the training set (BLEU 0.49): "
         "underfitting. 128 and 256 both reach training BLEU ≈ 0.98, and validation BLEU flattens near 0.52 and 0.57 "
         "without falling: a generalization gap, but not the falling validation curve of overfitting. "
         "Each BLEU point is a mean over 1,000 random sentences, hence the jitter.",
         alt="Training loss, training BLEU and validation BLEU against epoch for embedding dimensions 32, 64, 128 and 256")


# ---------------------------------------------------------------- 2. heads sweep
def head_sweep():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for h, c, ls in [(1, MUTED, "-"), (4, HI, "--"), (16, ACCENT, "-")]:
        e, tr = series(128, h, "BLEU Scores/Train BLEU")
        _, va = series(128, h, "BLEU Scores/Validation BLEU")
        a1.plot(e, tr, color=c, ls=ls, lw=1.5, label=f"{h} head" + ("s" if h > 1 else ""))
        a2.plot(e, va, color=c, ls=ls, lw=1.5)
    a1.set_title("training BLEU", fontsize=10)
    a2.set_title("validation BLEU", fontsize=10)
    a1.legend(fontsize=9, loc="lower right")
    for a in (a1, a2):
        a.set_ylim(0, 1)
        a.set_xlim(0, 100)
        a.set_xlabel("epoch")
        a.grid(True)
        marks(a)
    save(fig, "train-b-head-sweep",
         "<b>The lecture's attention-head runs</b> (<i>d</i><sub>emb</sub> = 128, 1 layer; same logs). More heads fit the "
         "training set faster before the first learning-rate drop (training BLEU at epoch 70: 0.76, 0.86, 0.94 for "
         "1, 4, 16 heads), but after the drops all three end within noise of each other on validation "
         "(mean of the last ten evaluations: 0.52, 0.52, 0.51). The parameter count is the same in all three.",
         alt="Training and validation BLEU for 1, 4 and 16 attention heads")


# ---------------------------------------------------------------- 3. the schedule and what it did
def lr_drop():
    e = np.arange(1, 101)
    lr = np.where(e <= 70, 1e-2, np.where(e <= 85, 1e-3, 1e-4))
    _, loss = series(128, 1, "Loss")
    _, tr = series(128, 1, "BLEU Scores/Train BLEU")
    _, va = series(128, 1, "BLEU Scores/Validation BLEU")
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(6.2, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.5, 1.7]))
    a0.semilogy(e, lr, color=INK, lw=1.8, drawstyle="steps-post")
    a0.set_ylabel("learning rate")
    a0.set_ylim(5e-5, 2e-2)
    a1.semilogy(e, loss, color=ACCENT, lw=1.6)
    a1.set_ylabel("training loss")
    a2.plot(e, tr, color=ACCENT, lw=1.6, label="training BLEU")
    a2.plot(e, va, color=HI, lw=1.6, ls="--", label="validation BLEU")
    a2.set_ylim(0, 1)
    a2.set_xlabel("epoch")
    a2.legend(fontsize=9, loc="lower right")
    for a in (a0, a1, a2):
        a.grid(True)
        marks(a)
        a.set_xlim(0, 100)
    save(fig, "train-b-lr-drop",
         "<b>What <code>MultiStepLR(milestones=[70, 85], gamma=0.1)</code> did</b> in the run trained during the "
         "lecture (<i>d</i><sub>emb</sub> = 128, 1 head; from its TensorBoard log). Top: the schedule, 10⁻² for epochs "
         "1–70, 10⁻³ for 71–85, 10⁻⁴ for 86–100. Middle: the loss had stalled near 0.28 and falls to 0.17 in the "
         "first epoch at the lower rate, then to 0.06. Bottom: training BLEU jumps from 0.75 to 0.89 in one epoch and "
         "validation BLEU from about 0.42 to about 0.52. The second drop changes little.",
         alt="Learning rate steps, training loss and BLEU scores against epoch, with jumps after epoch 70")


# ---------------------------------------------------------------- 4. schematic: the three regimes
def fit_sketch():
    e = np.linspace(0, 100, 300)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
    under = (0.34 * (1 - np.exp(-e / 14)), 0.31 * (1 - np.exp(-e / 14)))
    over_tr = 0.99 * (1 - np.exp(-e / 16))
    over_va = 0.62 * (1 - np.exp(-e / 10)) - 0.22 * np.clip((e - 35) / 65, 0, None) ** 1.3
    good = (0.95 * (1 - np.exp(-e / 16)), 0.80 * (1 - np.exp(-e / 15)))
    for ax, (tr, va), title in zip(axes, [under, (over_tr, over_va), good], ["underfitting", "overfitting", "a proper fit"]):
        ax.plot(e, tr, color=ACCENT, lw=1.8, label="training")
        ax.plot(e, va, color=HI, lw=1.8, ls="--", label="validation")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1.05)
        ax.set_xticks([])
        ax.grid(True)
    axes[0].set_ylabel("accuracy (or BLEU)")
    axes[0].legend(fontsize=9, loc="upper left")
    axes[1].annotate("stop here", (35, over_va[105]), (48, 0.28), fontsize=9, arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    save(fig, "train-b-fit-sketch",
         "<b>The three shapes to recognize</b> — a schematic drawn from the lecture's descriptions, not data. "
         "Underfitting: both curves low, close together, flat early. Overfitting: training near perfect while "
         "validation peaks and then <i>falls</i>. Proper fit: validation lags training but flattens instead of falling.",
         alt="Three sketched pairs of training and validation curves: underfitting, overfitting, proper fit")


if __name__ == "__main__":
    emb_sweep()
    head_sweep()
    lr_drop()
    fit_sketch()
    print("ok")
