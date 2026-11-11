#!/usr/bin/env python3
"""Recompute every number on the Midterm 2 pages (content/9-midterm-2/) and on the
Lecture 21-22 topic pages with numpy. Run: python3 -B tools/verify_mt2.py

PyTorch is not needed: layers are re-implemented from their definitions
(cross-correlation, max pooling, softmax attention, parameter-count formulas).
tools/verify_mt2_torch.py repeats the checks that are PyTorch behaviour with PyTorch itself.
Each block prints what it checked; any mismatch with the official keys raises.
"""
import math
from fractions import Fraction as Fr

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def out_size(n, k, s=1, p=0):
    return (n + 2 * p - k) // s + 1


def xcorr2d(x, w, stride=1, pad=0):
    x = np.pad(x, pad)
    k = w.shape[0]
    n = (x.shape[0] - k) // stride + 1
    m = (x.shape[1] - k) // stride + 1
    return np.array([[(x[i * stride:i * stride + k, j * stride:j * stride + k] * w).sum() for j in range(m)] for i in range(n)])


def pool2d(x, k, stride=None, op=np.max):
    stride = stride or k
    n = (x.shape[0] - k) // stride + 1
    m = (x.shape[1] - k) // stride + 1
    return np.array([[op(x[i * stride:i * stride + k, j * stride:j * stride + k]) for j in range(m)] for i in range(n)])


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def lin(i, o, bias=True):
    return i * o + (o if bias else 0)


def conv(ci, co, k, bias=True):
    return ci * co * k * k + (co if bias else 0)


def enc_layer(E, F):
    """nn.TransformerEncoderLayer(E, nhead, F): MHA (4E^2+4E) + two linears + two LayerNorms."""
    return 4 * E * E + 4 * E + lin(E, F) + lin(F, E) + 2 * 2 * E


def dec_layer(E, F):
    """nn.TransformerDecoderLayer: two MHAs + two linears + three LayerNorms."""
    return 2 * (4 * E * E + 4 * E) + lin(E, F) + lin(F, E) + 3 * 2 * E


print("== Fall 2024")
assert 72 // 2 == 36 and 36 // 4 == 9
N, D = 7, 3
assert lin(N, N) * 2 + lin(N, D) == 2 * N * N + 2 * N + N * D + D
assert out_size(40, 5) == 36 and conv(3, 16, 5, bias=False) == 1200
Z = np.array([[4, 1, 8, 2, 3, 7], [0, 0, 2, 5, 6, 5], [1, 4, 9, 2, 5, 5], [2, 6, 2, 3, 3, 6], [0, 6, 6, 7, 6, 3], [6, 2, 1, 1, 9, 3]])
p = pool2d(Z, 2, 3)
print("Q4c max pool 2x2 stride 3:", p.tolist(), "size", out_size(6, 2, 3))
assert p.tolist() == [[4, 6], [6, 7]]
print("   same input, 2x2 stride 2:", pool2d(Z, 2, 2).tolist(), " 3x3 stride 3:", pool2d(Z, 3, 3).tolist())
x = np.array([-2, 3, -1, 0, -3, 6.0])
g = lambda v: v.reshape(-1, 3).mean(1)
relu = lambda v: np.maximum(v, 0)
print("Q5b relu(g(x)) =", relu(g(x)), " g(relu(x)) =", g(relu(x)))
assert relu(g(x)).tolist() == [0, 1] and g(relu(x)).tolist() == [1, 2]
v1, v2, v3 = np.array([.5, 3 ** .5 / 2, 0]), np.array([0, .5, 3 ** .5 / 2]), np.array([3 ** .5 / 2, 0, .5])
print("Q1i (skipped) v1.v2 = %.4f (sqrt3/4 = %.4f), norms %.3f %.3f %.3f" % (v1 @ v2, 3 ** .5 / 4, *(np.linalg.norm(v) for v in (v1, v2, v3))))

print("== Fall 2025")
for name, o in [("i conv k3 s3 p1", out_size(30, 3, 3, 1)), ("ii conv k3 s3 p3", out_size(30, 3, 3, 3)), ("iii conv k1 s3 p0", out_size(30, 1, 3, 0)),
                ("iv pool k3 s1", out_size(30, 3, 1)), ("v pool k3 s3", out_size(30, 3, 3))]:
    print("Q2a", name, "->", o)
assert [out_size(30, 3, 3, 1), out_size(30, 3, 3, 3), out_size(30, 1, 3, 0), out_size(30, 3, 1), out_size(30, 3, 3)] == [10, 12, 10, 28, 10]
d, H = 5, 9
assert lin(d, H, False) + lin(H, H, False) == d * H + H * H
Q = np.array([[3, -1], [2, 1], [-1, 0]]); K = np.array([[1, 1], [-1, -2], [2, -1]]); V = np.array([[2, -2], [4, -1], [1, 3]])
S = Q @ K.T
print("Q4b QK^T =", S.tolist())
assert S.tolist() == [[2, -1, 7], [3, -4, 3], [-1, 1, -2]]
A = softmax(S / math.sqrt(2))
print("   attention weights (scaled by sqrt 2):\n", A, "\n   outputs A V:\n", A @ V)
print("   unscaled row 1:", softmax(S[0].astype(float)))
# Q4a: multiply by the mask instead of adding it
big = 1e9
M = np.where(np.tril(np.ones((3, 3))) > 0, 1.0, -big)
print("Q4a multiplicative mask, scores*M (1e9 stands for inf):\n", S * M)
print("   weights:\n", softmax(S * M / math.sqrt(2)))
print("   correct additive mask weights:\n", softmax((S + np.where(np.tril(np.ones((3, 3))) > 0, 0.0, -big)) / math.sqrt(2)))
assert 3 * 64 * 64 == 12288 and lin(12288, 512) == 6291968
assert (64 // 8) ** 2 == 64 and 3 * 8 * 8 == 192 and (128 // 8) ** 2 == 256
print("Q5 first MLP layer", lin(12288, 512), "params; ViT tokens", (64 // 8) ** 2, "of length", 3 * 8 * 8, "; at 128x128:", (128 // 8) ** 2, "tokens; MLP input would be", 3 * 128 * 128)
print("Q1d one-hot length 64; binary code bits", math.ceil(math.log2(64)))

print("== Spring 2025")
X = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [0, 1, 0, 1, 0]])
p = pool2d(X, 3, 2)
print("Q3 MaxPool2d(3, stride=2):", p.tolist(), "size", out_size(5, 3, 2))
assert p.tolist() == [[4, 5], [5, 5]]
assert lin(8, 16) == 144 and lin(16, 4) == 68 and lin(8, 16) + lin(16, 4) == 212
rng = np.random.default_rng(0)
for _ in range(200):
    m, b, x = rng.normal(size=3) * 3
    f = np.array([1, -1]) @ relu(np.array([m, -m]) * x + np.array([b, -b]))
    assert abs(f - (m * x + b)) < 1e-12
print("Q5 w0=[m,-m], b0=[b,-b], w1=[1,-1] reproduces mx+b on 200 random (m,b,x)")
# Q7: single-head attention with projections, shape check
B, Nn, dm = 2, 5, 8
Wq, Wk, Wv, Wo = (rng.normal(size=(dm, dm)) for _ in range(4))
bq, bk, bv, bo = (rng.normal(size=dm) for _ in range(4))
Qx, Kx, Vx = (rng.normal(size=(B, Nn, dm)) for _ in range(3))
Qw, Kw, Vw = Qx @ Wq + bq, Kx @ Wk + bk, Vx @ Wv + bv
W_ = softmax(Qw @ Kw.transpose(0, 2, 1) / math.sqrt(dm))
out = (W_ @ Vw) @ Wo + bo
assert W_.shape == (B, Nn, Nn) and out.shape == (B, Nn, dm) and np.allclose(W_.sum(-1), 1)
# scaling the queries first (the key) equals scaling the scores
assert np.allclose(((Qw * dm ** -0.5) @ Kw.transpose(0, 2, 1)), Qw @ Kw.transpose(0, 2, 1) / math.sqrt(dm))
print("Q7 shapes: weights", W_.shape, "output", out.shape, "; params 4(d^2+d) for d=8:", 4 * lin(dm, dm))

print("== Spring 2026")
X = np.array([[0, 0, 1, 0, 0], [0, 2, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
Wf = np.array([[1, 0, -1], [0, 1, 0], [-1, 0, 1]])
c = xcorr2d(X, Wf, stride=2)
print("Q3 cross-correlation stride 2:", c.tolist(), " flipped-kernel convolution:", xcorr2d(X, Wf[::-1, ::-1], stride=2).tolist())
assert c.tolist() == [[0, 2], [1, -1]] and (Wf[::-1, ::-1] == Wf).all()
print("   stride 1 output:", xcorr2d(X, Wf).tolist())
assert lin(10, 12) == 132 and lin(12, 6, False) == 72 and lin(6, 3) == 21 and 132 + 72 + 21 == 225
for x in np.linspace(-3, 3, 13):
    assert abs(np.array([1, 1]) @ relu(np.array([1, -1]) * x) - abs(x)) < 1e-12
print("Q5 w0=[1,-1], b0=0, w1=[1,1] gives |x|")
print("Q1a 'same' padding (K-1)/2: K=3 ->", out_size(32, 3, 1, 1), " K=5 ->", out_size(32, 5, 1, 2), " K=4, p=1 ->", out_size(32, 4, 1, 1), " K=4, p=2 ->", out_size(32, 4, 1, 2))
dd = 16
print("Q7a feed-forward d->d->d params (d=16):", 2 * lin(dd, dd), " encoder layer of the problem: MHA + 2 LayerNorm + FF")

print("== Review pages: the homework items kept as drills (HW5 1a-c, HW5 2a-c, HW6 2d)")
print("HW5 1a", out_size(90, 5, 3, 2), out_size(120, 5, 3, 2), " 1b", out_size(90, 7, 2), out_size(120, 7, 2))
print("HW5 1c", conv(3, 64, 7, False), conv(64, 128, 5, False), conv(128, 128, 3, False), lin(128, 5), "total", conv(3, 64, 7, False) + conv(64, 128, 5, False) + conv(128, 128, 3, False) + lin(128, 5))


def rf(layers):
    r, j = 1, 1
    for k, s in layers:
        r, j = r + (k - 1) * j, j * s
    return r, j


print("HW5 2a", rf([(7, 1), (3, 2)]), " 2b", rf([(3, 1), (3, 2)] * 3 + [(3, 1)]), " 2c blocks of k=5:", [rf([(5, 1), (5, 2)] * n)[0] for n in (1, 2, 3, 4)])
img = np.array([[7, 7, 0, 4, 3, 3, 1, 3], [8, 3, 5, 4, 6, 1, 9, 2], [5, 6, 2, 5, 1, 7, 1, 6], [8, 9, 8, 6, 6, 1, 5, 3]])
toks = [img[i:i + 2, j:j + 2].reshape(-1).tolist() for i in (0, 2) for j in (0, 2, 4, 6)]
print("HW6 2d ViT tokens (2x2 patches, row-major):", toks)
assert toks == [[7, 7, 8, 3], [0, 4, 5, 4], [3, 3, 6, 1], [1, 3, 9, 2], [5, 6, 8, 9], [2, 5, 8, 6], [1, 7, 6, 1], [1, 6, 5, 3]]
hw5c = [conv(3, 64, 7, False), conv(64, 128, 5, False), conv(128, 128, 3, False), lin(128, 5)]
assert hw5c == [9408, 204800, 147456, 645] and sum(hw5c) == 362309 and hw5c[1] > sum(hw5c) / 2
assert rf([(7, 1), (3, 2)]) == (9, 2) and rf([(3, 1), (3, 2)] * 3 + [(3, 1)]) == (45, 8)
rj, r_, j_ = [], 1, 1
for k_, st_ in [(3, 1), (3, 2)] * 3 + [(3, 1)]:
    r_, j_ = r_ + (k_ - 1) * j_, j_ * st_; rj.append((r_, j_))
assert rj == [(3, 1), (5, 2), (9, 2), (13, 4), (21, 4), (29, 8), (45, 8)]
assert [rf([(5, 1), (5, 2)] * n)[0] for n in (1, 2, 3, 4)] == [9, 25, 57, 121]
# RNN parameter formulas (nn.RNN has two bias vectors per layer and direction)
rnn = lambda d, H: H * (d + H + 2)
print("nn.RNN(10,20):", rnn(10, 20), " GRU:", 3 * rnn(10, 20), " LSTM:", 4 * rnn(10, 20), " bidirectional RNN:", 2 * rnn(10, 20), " 2-layer RNN:", rnn(10, 20) + rnn(20, 20))
# Elman forward pass used on review-recurrent-networks
Wx, Wh = np.array([[1.0, 0.0], [0.0, -1.0]]), np.array([[0.5, 0.0], [0.5, 0.5]])
h = np.zeros(2)
for t, xt in enumerate([np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])], 1):
    h = np.tanh(Wx @ xt + Wh @ h)
    print("   RNN drill h_%d =" % t, np.round(h, 4))
# Fall 2025 Q3 bug: without h = h_next every step sees h = 0
Wx_, Wh_ = rng.normal(size=(4, 3)), rng.normal(size=(4, 4))
xs = rng.normal(size=(5, 3))
hb, hg = np.zeros(4), np.zeros(4)
buggy, fixed = [], []
for xt in xs:
    z = np.tanh(Wx_ @ xt)
    buggy.append(np.tanh(Wh_ @ (z + np.zeros(4))))
    hg = np.tanh(Wh_ @ (z + hg)); fixed.append(hg)
print("   F25 Q3: buggy and fixed agree at t=1:", np.allclose(buggy[0], fixed[0]), " at t=2:", np.allclose(buggy[1], fixed[1]))

print("== Practice problems")
print("P1", out_size(64, 7, 2, 3), conv(3, 32, 7), "| pool:", out_size(32, 2, 2), "| conv2:", out_size(16, 3, 1, 1), conv(32, 64, 3, False))
Xp = np.array([[1, 3, 0, 2], [5, 2, 1, 0], [0, 1, 4, 2], [2, 0, 3, 6]])
print("P2 max 2x2 s2:", pool2d(Xp, 2).tolist(), " avg:", pool2d(Xp, 2, op=np.mean).tolist(), " max 3x3 s1:", pool2d(Xp, 3, 1).tolist())
Xc = np.array([[1, 0, 2, 1], [0, 1, 0, 3], [2, 1, 1, 0], [1, 0, 2, 2]]); Wc = np.array([[1, -1], [0, 2]])
print("P3 xcorr stride 1:", xcorr2d(Xc, Wc).tolist(), " stride 2:", xcorr2d(Xc, Wc, 2).tolist(), " padding 1 stride 2:", xcorr2d(Xc, Wc, 2, 1).tolist())
print("P4 params:", lin(20, 50), lin(50, 50, False), lin(50, 3), "total", lin(20, 50) + lin(50, 50, False) + lin(50, 3))
Q = np.array([[1, 0], [0, 2], [1, 1]]); K = np.array([[2, 1], [0, 1], [-1, 3]]); V = np.array([[1, 0], [0, 1], [2, 2]])
S = Q @ K.T
print("P5 QK^T:", S.tolist(), " row argmax:", (S.argmax(1) + 1).tolist())
Mk = np.where(np.tril(np.ones((3, 3))) > 0, 0.0, -1e9)
print("   masked weights (no scaling):\n", softmax(S + Mk), "\n   masked outputs:\n", softmax(S + Mk) @ V)
print("   unmasked weights row 2 (no scaling):", softmax(S[1].astype(float)), " with /sqrt2:", softmax(S[1] / math.sqrt(2)))
print("P6 ViT 96x96 P=16:", (96 // 16) ** 2, "patches +1 =", (96 // 16) ** 2 + 1, " token length", 3 * 16 * 16, " P=8:", (96 // 8) ** 2 + 1, 3 * 64)
print("P7 friend RNN: Linear(d,H)+Linear(H,H) with bias, d=12 H=32:", lin(12, 32) + lin(32, 32), " nn.RNN(12,32):", rnn(12, 32))
print("P8 ReLU net max(x,0)-max(x-1,0) at -1,0,.5,1,2:", [float(relu(x) - relu(x - 1)) for x in (-1, 0, .5, 1, 2)])
print("P9 DataLoader: 1000 points batch 64 ->", math.ceil(1000 / 64), "batches (last has", 1000 - 64 * 15, "), drop_last:", 1000 // 64)
print("P10 embedding 5000x64:", 5000 * 64, " one-hot Linear(5000,64) no bias:", lin(5000, 64, False))

print("== Lecture 21: parameter counts from the layer definitions")
E = 128
print("encoder layer E=128, d_ff=256:", enc_layer(E, 2 * E), "= 8E^2+11E:", 8 * E * E + 11 * E)
print("decoder layer E=128:", dec_layer(E, 2 * E), "= 12E^2+17E:", 12 * E * E + 17 * E)
Ve, Vf = 1443, 1646


def translation(E, L=1):
    enc = Ve * E + 1000 * E + L * enc_layer(E, 2 * E)
    dec = Vf * E + 1000 * E + L * dec_layer(E, 2 * E) + lin(E, Vf)
    return enc, dec, enc + dec


for E in (32, 64, 128, 256):
    print("translation Transformer d_emb=%d (1 layer):" % E, translation(E))
print("   2 layers, d_emb=128:", translation(128, 2), " 6 layers:", translation(128, 6))
Vs = 64432
cls_ = (Vs * 64, 1000 * 64, 2 * enc_layer(64, 128), lin(64, 2))
print("sentiment classifier (d_emb=64, 2 layers):", cls_, "total", sum(cls_), " embedding share %.1f%%" % (100 * cls_[0] / sum(cls_)))
print("   vocabulary per example:", round(64432 / 37647, 2), " majority-class accuracy: 60.19%")
E = 3 * 16 * 16
vit = dict(cls=E, pos=10 ** 4 * E, proj=lin(E, E), layers=2 * enc_layer(E, 2 * E), head=lin(E, 2 * E) + lin(2 * E, 4))
print("ViT (P=16, 2 layers, 4 classes):", vit, "total", sum(vit.values()), "; positional rows used:", 101, "=", 101 * E, "params; tokens per image", (160 // 16) ** 2 + 1, " per-head width", E // 8)
print("   ViT without the unused positional rows:", sum(vit.values()) - (10 ** 4 - 101) * E)
print("   P=8: d_emb", 3 * 64, "tokens", (160 // 8) ** 2 + 1, " P=32: d_emb", 3 * 32 * 32, "tokens", (160 // 32) ** 2 + 1)
print("   attention matrix entries per image: P=16:", 101 ** 2, " P=8:", 401 ** 2, " pixel tokens:", (160 * 160 + 1) ** 2)
for Wd in (160, 165, 170):
    r = Wd % 16
    print("   tokenize_image W=%d, P=16: notebook pads %d -> %d, Unfold keeps %d patches = %d columns (%s); exact pad %d"
          % (Wd, r, Wd + r, (Wd + r) // 16, 16 * ((Wd + r) // 16), "all kept" if 16 * ((Wd + r) // 16) >= Wd else "%d real columns dropped" % (Wd - 16 * ((Wd + r) // 16)), (-Wd) % 16))

print("== Lecture 22: MultiStepLR")


def multistep(lr0, milestones, gamma, n):
    return [lr0 * gamma ** sum(e >= m for m in milestones) for e in range(n)]


s = multistep(1e-2, [70, 85], 0.1, 100)
print("lr during epoch 1, 70, 71, 85, 86, 100 (1-indexed):", [s[i] for i in (0, 69, 70, 84, 85, 99)])
s2 = multistep(1e-1, [40, 80], 0.2, 100)
print("notebook example [40,80], gamma 0.2 from 0.1:", [round(s2[i], 6) for i in (0, 39, 40, 79, 80, 99)])
s3 = multistep(1e-1, [100, 150], 0.1, 200)
print("lecture example [100,150] from 0.1:", [round(s3[i], 6) for i in (0, 99, 100, 149, 150, 199)])
print("== Pages written after the exam walk-throughs (review pages, T/F bank, hub)")
# Fall 2025 Q4(b): weights and output quoted on the page
Q = np.array([[3, -1], [2, 1], [-1, 0]]); K = np.array([[1, 1], [-1, -2], [2, -1]]); V = np.array([[2, -2], [4, -1], [1, 3]])
A = softmax(Q @ K.T / math.sqrt(2))
assert np.allclose(A[0], [0.028, 0.003, 0.968], atol=6e-4) and np.allclose((A @ V)[0], [1.04, 2.85], atol=6e-3)
# Spring 2026 Q3: "top-left - top-right - bottom-left + bottom-right + centre" per window
X = np.array([[0, 0, 1, 0, 0], [0, 2, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
cc = lambda r, c: X[r, c] - X[r, c + 2] - X[r + 2, c] + X[r + 2, c + 2] + X[r + 1, c + 1]
assert [cc(0, 0), cc(0, 2), cc(2, 0), cc(2, 2)] == [0, 2, 1, -1]
assert lin(10, 12) + lin(12, 6) + lin(6, 3) == 231                      # Sp26 Q4 with every bias on
# Spring 2025 Q3 window table
X = np.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [0, 1, 0, 1, 0]])
assert [X[r:r + 3, c:c + 3].max() for r in (0, 2) for c in (0, 2)] == [4, 5, 5, 5]
assert out_size(8, 3) ** 2 == 36                                        # Sp25 1(c): 36 uses per kernel weight
# review-sizes-and-parameters
assert [out_size(90, 5, 3, 2), out_size(120, 5, 3, 2), out_size(90, 7, 2), out_size(120, 7, 2)] == [30, 40, 42, 57]
assert out_size(28, 3, 2, 1) == 14 and conv(16, 32, 3) == 4640 and out_size(7, 2, 2) == 3 and 4 * 20 * (10 + 20 + 2) == 2560
assert rf([(3, 2)] * 3)[0] == 15 and rf([(7, 1), (3, 2)])[0] == 9
seq, r, j = [], 1, 1
for k, s_ in [(3, 1), (3, 2)] * 3 + [(3, 1)]:
    r, j = r + (k - 1) * j, j * s_; seq.append(r)
assert seq == [3, 5, 9, 13, 21, 29, 45]
assert all(rf([(5, 1), (5, 2)] * n)[0] == 1 + 8 * (2 ** n - 1) for n in range(1, 6))
E, Fd = 128, 256
assert 4 * E * E + 2 * E * Fd + 9 * E + Fd == enc_layer(E, Fd) and 8 * E * E + 2 * E * Fd + 15 * E + Fd == dec_layer(E, Fd)
# review-pytorch-modules / review-recurrent-networks
n_, m_ = 9, 4
assert lin(n_, m_) + lin(m_, n_) == 2 * n_ * m_ + m_ + n_ and 3 * math.ceil(1000 / 64) == 48
assert 2 * 3 * 64 * (32 + 64 + 2) == 37632 and lin(12, 32) + lin(32, 32) == 32 * (12 + 32 + 2) == 1472
bh = rng.normal(size=4); Whh = rng.normal(size=(4, 4)); a1 = np.tanh(rng.normal(size=4))
assert np.allclose(np.tanh(Whh @ (a1 * np.zeros(4)) + bh), np.tanh(bh))  # practice P7(b)
# review-training-and-activations: the two-unit ReLU table
xs = np.linspace(-4, 4, 81)
net = lambda w0, b0, w1: np.array([np.array(w1) @ relu(np.array(w0) * x + np.array(b0)) for x in xs])
assert np.allclose(net([2, -2], [-6, 6], [1, -1]), 2 * xs - 6)
assert np.allclose(net([1, -1], [0, 0], [1, 1]), np.abs(xs))
assert np.allclose(net([1, 1], [0, -1], [1, -1]), np.clip(xs, 0, 1))
for c in (0.0, 0.5, 2.0):
    assert np.allclose(net([1, 0], [-c, c], [1, 1]), np.maximum(xs, c))
assert np.allclose(net([2, -1], [2 * 1.5, -1.5], [0.5, -1]), xs + 1.5)   # Sp25 Q5 alternative answer, m=1, b=1.5
# true/false bank extras and practice problems
assert conv(3, 16, 5) == 1216 and (64 // 16) ** 2 == 16 and 3 * 16 * 16 == 768 and 4 * 128 * 128 + 4 * 128 == 66048
assert round((145 / 37) ** 2) == 15 and 5 * math.ceil(1000 / 64) == 80
# skip-list arithmetic on the hub and the skipped-topics page
assert 3 * 3 + 5 * 2 + 6 * 2 == 31 and 100 - 10 - 15 == 75 and 100 - 12 - 15 == 73 and 30 - 9 == 21 and 30 - 10 == 20 and 30 - 12 == 18
assert 7 + 10 + 10 + 9 == 36
print("ok")
