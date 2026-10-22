#!/usr/bin/env python3
"""Figures for the CNN-history and RNN pages (Lecture 17). Prefix: deepc-.

    python3 tools/figs_deep_c.py && python3 tools/assemble.py content/5-deep-networks/<page>.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mplfig import plt, INK, ACCENT, ACCENT2, HI, MUTED, save  # noqa: E402
from figlib import SVG  # noqa: E402
from mplfig import FIGS  # noqa: E402


def write(name, html):
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(FIGS, name + ".html"), "w") as f:
        f.write(html + "\n")


# ------------------------------------------------------------------ ImageNet timeline
def imagenet_timeline():
    # ILSVRC classification, top-5 error (%) of each listed entry.
    # 2010/2011: Russakovsky et al. 2015 (IJCV); 2012: notebook / Krizhevsky et al.;
    # VGG 7.3 (ILSVRC-2014, 2nd); GoogLeNet 6.7 (Szegedy et al.); ResNet ensemble 3.57 (He et al.)
    names = ["2010\nNEC-UIUC\n(no CNN)", "2011\nXRCE\n(no CNN)", "2012\nAlexNet", "2014\nVGG\n(2nd place)",
             "2014\nGoogLeNet", "2015\nResNet"]
    err = [28.2, 25.8, 15.3, 7.3, 6.7, 3.57]
    cols = [MUTED, MUTED, ACCENT, ACCENT, ACCENT, HI]
    # weight layers (LeNet-5 for reference, from the lecture's figure)
    dnames = ["LeNet-5\n1998", "AlexNet\n2012", "VGG-19\n2014", "GoogLeNet\n2014", "ResNet-152\n2015"]
    depth = [5, 8, 19, 22, 152]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
    x = range(len(err))
    a.bar(x, err, color=cols, width=0.62)
    for i, e in enumerate(err):
        a.text(i, e + 0.7, f"{e:g}%", ha="center", va="bottom", fontsize=10, color=INK)
    a.axhline(5.1, color=INK, lw=1, ls="--")
    a.text(-0.45, 5.6, "human ≈ 5.1%", ha="left", va="bottom", fontsize=9, color=INK)
    a.set_xticks(list(x), names, fontsize=8.5)
    a.set_ylabel("ILSVRC top-5 error (%)")
    a.set_ylim(0, 32)
    a.set_title("Error fell once CNNs arrived", fontsize=11, loc="left")

    b.bar(range(len(depth)), depth, color=[MUTED, ACCENT, ACCENT, ACCENT, HI], width=0.62)
    for i, d in enumerate(depth):
        b.text(i, d * 1.12, str(d), ha="center", va="bottom", fontsize=10, color=INK)
    b.set_yscale("log")
    b.set_ylim(3, 400)
    b.set_yticks([5, 10, 20, 50, 100, 200], ["5", "10", "20", "50", "100", "200"])
    b.minorticks_off()
    b.set_xticks(range(len(depth)), dnames, fontsize=8.5)
    b.set_ylabel("weight layers (log scale)")
    b.set_title("…while the networks got deeper", fontsize=11, loc="left")
    fig.tight_layout()
    save(fig, "deepc-imagenet-timeline",
         "<b>ImageNet (ILSVRC) classification, 2010–2015.</b> Left: top-5 error of the winning entry in each year shown "
         "(2013 is left out; VGG placed 2nd in 2014; challenge entries were often ensembles of several networks). The 15.3% for AlexNet is from the lecture; the other values are the published "
         "challenge results (Russakovsky et al., 2015) and the original papers; the human estimate is the one trained "
         "annotator in Russakovsky et al. Right: depth counted in weight layers (conv + fully connected); "
         "ResNet-152 is 7× deeper than GoogLeNet.",
         alt="Bar charts: ImageNet top-5 error dropping from 28.2% to 3.57%, and depth growing from 5 to 152 layers")


# ------------------------------------------------------------------ nn.RNN tensor shapes
def sub(base, s, sup=None):
    t = f'{base}<tspan baseline-shift="sub" font-size="11">{s}</tspan>'
    if sup:
        t += f'<tspan baseline-shift="super" font-size="10">({sup})</tspan>'
    return t


def rnn_shapes():
    W, H = 800, 370
    g = SVG(W, H, font_size=14)
    xs = [230, 350, 470, 590]       # time steps t = 1..4
    ys = {2: 110, 1: 205}           # layer rows
    bw, bh = 82, 40
    # output row (top layer, every t)
    g.rect(xs[0] - 56, ys[2] - 34, xs[-1] - xs[0] + 112, 68, rx=10, stroke="var(--accent)", width=1.8, dash="6 4",
           fill="var(--accent)", fill_opacity=0.07)
    # h_n column (last t, every layer)
    g.rect(xs[-1] - 50, ys[2] - 42, 100, ys[1] - ys[2] + 84, rx=10, stroke="var(--hi)", width=1.8, dash="6 4",
           fill="var(--hi)", fill_opacity=0.06)
    for layer, y in ys.items():
        for t, x in enumerate(xs, 1):
            g.rect(x - bw / 2, y - bh / 2, bw, bh, rx=6, fill="var(--paper)")
            g.text(x, y, sub("h", t, layer), raw=True)
        # h0 on the left
        g.text(20, y, f"layer {layer}", anchor="start", size=12, color="var(--muted)")
        g.text(118, y, sub("h", 0, layer), raw=True, color="var(--muted)")
        g.line(138, y, xs[0] - bw / 2 - 3, y, arrow=True, color="var(--muted)")
        for x0, x1 in zip(xs, xs[1:]):
            g.line(x0 + bw / 2, y, x1 - bw / 2 - 3, y, arrow=True)
    for t, x in enumerate(xs, 1):
        g.text(x, 300, sub("x", t), raw=True)
        g.line(x, 288, x, ys[1] + bh / 2 + 3, arrow=True)
        g.line(x, ys[1] - bh / 2, x, ys[2] + bh / 2 + 3, arrow=True)
    g.text(20, 300, "input", anchor="start", color="var(--muted)", size=12)
    g.text(118, 250, "h_0: zeros", color="var(--muted)", size=12)
    g.text(118, 266, "by default", color="var(--muted)", size=12)
    # labels
    g.text(xs[0] - 50, 52, "output = top layer at every t:  (L, N, D·H_out)", anchor="start", color="var(--accent)",
           weight=600, size=13)
    g.text(652, 150, "h_n = last t,", anchor="start", color="var(--hi)", weight=600, size=13)
    g.text(652, 168, "every layer:", anchor="start", color="var(--hi)", weight=600, size=13)
    g.text(652, 188, "(D·num_layers,", anchor="start", color="var(--hi)", size=13)
    g.text(652, 206, " N, H_out)", anchor="start", color="var(--hi)", size=13)
    g.text(20, 336, "shapes (batch_first=False):  input x: (L, N, H_in)    h_0 and h_n: (D·num_layers, N, H_out)",
           anchor="start", size=12, color="var(--muted)")
    g.text(20, 354, "L = sequence length,  N = batch size,  D = 2 if bidirectional else 1", anchor="start", size=12,
           color="var(--muted)")
    write("deepc-rnn-shapes", g.figure(
        "<b>What <code>nn.RNN(H_in, H_out, num_layers=2)</code> returns</b> (unidirectional, L = 4 steps). "
        "<code>output</code> stacks the <b>top layer's</b> hidden state at every time step; <code>h_n</code> holds the "
        "<b>last time step</b> of every layer. They share one cell: <code>output[-1]</code> equals <code>h_n[-1]</code>. "
        "With <code>batch_first=True</code> only <code>input</code> and <code>output</code> become (N, L, ·); "
        "<code>h_0</code>/<code>h_n</code> keep the layer dimension first.",
        alt="Grid of hidden states over 2 layers and 4 time steps; top row marked as output, last column marked as h_n"))


# ------------------------------------------------------------------ parameter counts
def param_counts():
    # surname task: d = 52 letters, H = 64, K = 18 languages; formulas on the RNN pages
    rows = [("BasicRNN (from scratch)", 9594, MUTED),
            ("StackedRNN, 2 layers (from scratch)", 16400, MUTED),
            ("BidirectionalRNN (from scratch)", 17298, MUTED),
            ("nn.RNN, 2 layers, bidirectional", 42258, ACCENT),
            ("nn.GRU, 2 layers, bidirectional", 122130, ACCENT2),
            ("nn.LSTM, 2 layers, bidirectional", 162066, HI)]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    y = list(range(len(rows)))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    for yi, (n, v, _) in zip(y, rows):
        ax.text(v + 2500, yi, f"{v:,}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=10)
    ax.set_xlim(0, 190000)
    ax.set_xticks([0, 50000, 100000, 150000], ["0", "50k", "100k", "150k"])
    ax.set_xlabel("learnable parameters (including the final nn.Linear head)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, "deepc-param-counts",
         "<b>Parameters of the Lecture 17 surname models</b> (52-letter one-hot input, hidden size 64, 18 classes), "
         "counted by hand with the formulas on this page. At the same hidden size a GRU layer has 3× and an LSTM "
         "layer 4× the weights of a plain RNN layer, because they compute 3 and 4 gate vectors.",
         alt="Horizontal bars: parameter counts from 9,594 (BasicRNN) to 162,066 (2-layer bidirectional LSTM)")


if __name__ == "__main__":
    imagenet_timeline()
    rnn_shapes()
    param_counts()
    print("wrote deepc-imagenet-timeline, deepc-rnn-shapes, deepc-param-counts")
