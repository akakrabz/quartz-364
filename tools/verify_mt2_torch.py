#!/usr/bin/env python3
"""Run-it-for-real checks for the Midterm 2 pages and the Lecture 21-22 topic pages.

tools/verify_mt2.py recomputes everything with numpy; this script repeats the checks that
involve PyTorch behaviour with PyTorch itself (written and run with torch 2.15, CPU).

usage: python3 -B tools/verify_mt2_torch.py [path/to/L21_Beyond_Sequence_Transduction.ipynb]

With the notebook path, the class-definition cells of Lecture 21 (PositionalEncoding,
TransformerEncoder/Decoder, TransformerTextClassifier, VisionTransformer) are executed as written
(DEVICE = 'cpu') and their parameter counts and shapes are checked. Nothing is trained.
"""
import json
import math
import sys

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset

warnings.filterwarnings('ignore', message="Using padding='same'")
torch.manual_seed(0)
count = lambda m: sum(p.numel() for p in m.parameters())
ok = lambda name: print("ok  ", name)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc as e:
        return type(e).__name__ + ": " + str(e).split("\n")[0][:90]
    raise AssertionError("expected an error")


print("torch", torch.__version__)

# ------------------------------------------------------------------ Fall 2024
ds = TensorDataset(torch.arange(72.0).unsqueeze(1))
tr, va = list(range(0, 72, 2)), list(range(1, 72, 2))
for idx in (tr, va):
    assert len(DataLoader(ds, batch_size=9, sampler=SubsetRandomSampler(idx))) == 4
w = nn.Linear(3, 1)
opt = torch.optim.SGD(w.parameters())          # no lr given
assert opt.defaults["lr"] == 1e-3 and opt.defaults["momentum"] == 0 and opt.defaults["weight_decay"] == 0
print("     F24 Q2c: torch.optim.SGD(params) runs in this version; default lr =", opt.defaults["lr"])
print("     F24 Q2c: SGD() without params ->", raises(lambda: torch.optim.SGD()))


class CustomBlock(nn.Module):
    def __init__(self, N, D, activation_fn):
        super().__init__()
        self.W1 = nn.Linear(N, N)
        self.W2 = nn.Linear(N, N)
        self.W3 = nn.Linear(N, D)
        self.activation = activation_fn

    def forward(self, x):
        y1 = self.activation(self.W1(x)) * x
        y2 = self.activation(self.W2(y1)) * x
        return self.W3(y2)


for N, D in [(7, 3), (16, 5)]:
    b = CustomBlock(N, D, torch.tanh)
    assert count(b) == 2 * N * N + 2 * N + N * D + D and b(torch.randn(N)).shape == (D,)
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=0, stride=1, bias=False)
assert conv(torch.randn(1, 3, 40, 40)).shape == (1, 16, 36, 36) and count(conv) == 1200
Z = torch.tensor([[4, 1, 8, 2, 3, 7], [0, 0, 2, 5, 6, 5], [1, 4, 9, 2, 5, 5], [2, 6, 2, 3, 3, 6], [0, 6, 6, 7, 6, 3],
                  [6, 2, 1, 1, 9, 3.0]])
assert nn.MaxPool2d(2, stride=3)(Z[None, None])[0, 0].tolist() == [[4, 6], [6, 7]]
x = torch.tensor([-2, 3, -1, 0, -3, 6.0])
g = lambda v: F.avg_pool1d(v[None, None], 3, stride=3)[0, 0]
assert F.relu(g(x)).tolist() == [0, 1] and g(F.relu(x)).tolist() == [1, 2]
mx = lambda v: F.max_pool1d(v[None, None], 3, stride=3)[0, 0]
assert torch.equal(F.relu(mx(x)), mx(F.relu(x)))
ok("Fall 2024: DataLoader split, CustomBlock count, conv 36x36 / 1200, max pool, ReLU vs mean pool")

# ------------------------------------------------------------------ Fall 2025
X = torch.randn(64, 30, 30)                     # unbatched (C, H, W)
layers = [nn.Conv2d(64, 64, kernel_size=3, stride=3, padding=1), nn.Conv2d(64, 64, kernel_size=3, stride=3, padding=3),
          nn.Conv2d(64, 64, kernel_size=1, stride=3, padding=0), nn.MaxPool2d(kernel_size=3, stride=1),
          nn.MaxPool2d(kernel_size=3, stride=3)]
shapes = [tuple(f(X).shape) for f in layers]
assert shapes == [(64, 10, 10), (64, 12, 12), (64, 10, 10), (64, 28, 28), (64, 10, 10)], shapes
assert nn.AdaptiveAvgPool2d((1, 1))(torch.randn(7, 5, 9)).flatten().shape == (7,)       # 1(h)
assert F.one_hot(torch.tensor(3), 64).shape == (64,)                                    # 1(d)
bi = nn.RNN(10, 20, bidirectional=True)
assert count(bi) == 2 * count(nn.RNN(10, 20)) and not torch.equal(bi.weight_ih_l0, bi.weight_ih_l0_reverse)  # 1(j)


class FancyNewRNNLayer(nn.Module):
    def __init__(self, d, H, fixed=False):
        super().__init__()
        self.H, self.fixed = H, fixed
        self.tanh = nn.Tanh()
        self.W_x = nn.Linear(d, H, bias=False)
        self.W_h = nn.Linear(H, H, bias=False)

    def forward(self, x):
        T, d = x.size(0), x.size(1)
        h = torch.zeros(self.H)
        hiddens = []
        for i in range(T):
            z = self.tanh(self.W_x(x[i]))
            h_next = self.tanh(self.W_h(z + h))
            hiddens.append(h_next)
            if self.fixed:
                h = h_next
        return hiddens


d, H = 5, 9
bug, fix = FancyNewRNNLayer(d, H), FancyNewRNNLayer(d, H, fixed=True)
fix.load_state_dict(bug.state_dict())
assert count(bug) == d * H + H * H
xs = torch.randn(4, d)
hb, hf = bug(xs), fix(xs)
assert torch.allclose(hb[0], hf[0]) and not torch.allclose(hb[1], hf[1])
xs2 = xs.clone(); xs2[0] += 1.0                  # change only the first time step
assert torch.allclose(bug(xs2)[3], hb[3]) and not torch.allclose(fix(xs2)[3], hf[3])
Q = torch.tensor([[3, -1], [2, 1], [-1, 0.0]]); K = torch.tensor([[1, 1], [-1, -2], [2, -1.0]]); V = torch.tensor([[2, -2], [4, -1], [1, 3.0]])
S = Q @ K.T
assert S.tolist() == [[2, -1, 7], [3, -4, 3], [-1, 1, -2]] and S[0].argmax().item() == 2
A = torch.softmax(S / math.sqrt(2), dim=-1)
print("     F25 Q4b weights:", [[round(v, 4) for v in r] for r in A.tolist()])
Mmul = torch.where(torch.tril(torch.ones(3, 3)) > 0, 1.0, float("-inf"))
Madd = torch.where(torch.tril(torch.ones(3, 3)) > 0, 0.0, float("-inf"))
print("     F25 Q4a QK^T * M     =", (S * Mmul).tolist())
print("     F25 Q4a softmax(.)   =", torch.softmax(S * Mmul / math.sqrt(2), -1).tolist())
print("     additive mask softmax=", [[round(v, 4) for v in r] for r in torch.softmax((S + Madd) / math.sqrt(2), -1).tolist()])
assert (S * Mmul)[0].tolist() == [2.0, float("inf"), float("-inf")]
assert torch.isnan(torch.softmax(S * Mmul / math.sqrt(2), -1)[0]).all()
assert torch.isnan(torch.tensor(0.0) * float("-inf"))
fc = nn.Linear(3 * 64 * 64, 512, bias=True)
assert count(fc) == 6291968
print("     F25 Q5c MLP on 128x128 ->", raises(lambda: fc(torch.randn(1, 3, 128, 128).flatten(1))))
unf = nn.Unfold(8, stride=8)
assert unf(torch.randn(1, 3, 64, 64)).shape == (1, 192, 64) and unf(torch.randn(1, 3, 128, 128)).shape == (1, 192, 256)
cnn = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(8, 4))
assert cnn(torch.randn(2, 3, 64, 64)).shape == cnn(torch.randn(2, 3, 128, 128)).shape == (2, 4)
enc = nn.TransformerEncoderLayer(192, 4, 384, batch_first=True)
assert enc(torch.randn(2, 64, 192)).shape == (2, 64, 192) and enc(torch.randn(2, 256, 192)).shape == (2, 256, 192)
ok("Fall 2025: five layer shapes, RNN layer count and bug, QK^T, multiplicative mask -> NaN, MLP/CNN/ViT input sizes")

# ------------------------------------------------------------------ Spring 2025
X5 = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [0, 1, 0, 1, 0.0]])
assert nn.MaxPool2d(3, stride=2)(X5[None, None])[0, 0].tolist() == [[4, 5], [5, 5]]
m = nn.Sequential(nn.Linear(8, 16, bias=True), nn.ReLU(), nn.Linear(16, 4, bias=True))
h = torch.randn(1, 8); sh = []
for layer in m:
    h = layer(h); sh.append(tuple(h.shape))
assert sh == [(1, 16), (1, 16), (1, 4)] and count(m) == 212
for mm, bb in [(2.0, -3.0), (-1.5, 0.7), (0.0, 4.0)]:
    xx = torch.linspace(-5, 5, 41)
    f = (torch.tensor([1.0, -1.0]) * F.relu(torch.tensor([mm, -mm]) * xx[:, None] + torch.tensor([bb, -bb]))).sum(1)
    assert torch.allclose(f, mm * xx + bb, atol=1e-6)
c1 = nn.Conv2d(1, 4, 3)
assert count(c1) == 4 * 9 + 4 and c1(torch.randn(1, 1, 8, 8)).numel() == 4 * 36     # 1(c): 36 uses per weight
xr = torch.tensor([-2.0, 3.0], requires_grad=True); F.relu(xr).sum().backward()
assert xr.grad.tolist() == [0.0, 1.0]                                               # 1(f)
two = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 3)); one = nn.Linear(4, 3)
with torch.no_grad():
    one.weight.copy_(two[1].weight @ two[0].weight); one.bias.copy_(two[1].weight @ two[0].bias + two[1].bias)
v = torch.randn(5, 4); assert torch.allclose(two(v), one(v), atol=1e-6)             # 1(o)


class Attention(nn.Module):
    def __init__(self, d_model, call_super=True):
        if call_super:
            super().__init__()
        self.d_model = d_model
        self.wq = nn.Linear(d_model, d_model, bias=True)
        self.wk = nn.Linear(d_model, d_model, bias=True)
        self.wv = nn.Linear(d_model, d_model, bias=True)
        self.wo = nn.Linear(d_model, d_model, bias=True)
        self.scaling = d_model ** -0.5

    def forward(self, Q, K, V):
        query_states = self.wq(Q) * self.scaling
        key_states = self.wk(K)
        value_states = self.wv(V)
        attn_weights = F.softmax(query_states @ key_states.transpose(1, 2), dim=-1)
        return self.wo(attn_weights @ value_states)


att = Attention(8)
q, k, v = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)
ref = att.wo(torch.softmax(att.wq(q) @ att.wk(k).transpose(1, 2) / math.sqrt(8), -1) @ att.wv(v))
assert att(q, k, v).shape == (2, 5, 8) and torch.allclose(att(q, k, v), ref, atol=1e-6) and count(att) == 4 * (64 + 8)
print("     Sp25 Q7 without super().__init__() ->", raises(lambda: Attention(8, call_super=False)))
ok("Spring 2025: max pool, shapes and 212, ReLU line, T/F facts, attention module")

# ------------------------------------------------------------------ Spring 2026
X6 = torch.tensor([[0, 0, 1, 0, 0], [0, 2, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0.0]])
W6 = torch.tensor([[1, 0, -1], [0, 1, 0], [-1, 0, 1.0]])
assert F.conv2d(X6[None, None], W6[None, None], stride=2)[0, 0].tolist() == [[0, 2], [1, -1]]
assert F.conv2d(X6[None, None], W6.flip(0, 1)[None, None], stride=2)[0, 0].tolist() == [[0, 2], [1, -1]]
m = nn.Sequential(nn.Linear(10, 12, bias=True), nn.Tanh(), nn.Linear(12, 6, bias=False), nn.ReLU(), nn.Linear(6, 3, bias=True))
h = torch.randn(1, 10); sh = []
for layer in m:
    h = layer(h); sh.append(tuple(h.shape))
assert sh == [(1, 12), (1, 12), (1, 6), (1, 6), (1, 3)] and count(m) == 225
xx = torch.linspace(-4, 4, 33)
assert torch.allclose((torch.tensor([1.0, 1.0]) * F.relu(torch.tensor([1.0, -1.0]) * xx[:, None])).sum(1), xx.abs())
for Kk in (3, 5, 7):
    assert nn.Conv2d(1, 1, Kk, padding=(Kk - 1) // 2)(torch.randn(1, 1, 32, 32)).shape[-1] == 32
assert nn.Conv2d(1, 1, 4, padding=1)(torch.randn(1, 1, 32, 32)).shape[-1] == 31
assert nn.Conv2d(1, 1, 4, padding=2)(torch.randn(1, 1, 32, 32)).shape[-1] == 33
assert nn.Conv2d(1, 1, 4, padding="same")(torch.randn(1, 1, 32, 32)).shape[-1] == 32
print("     Sp26 1a padding=1.5 ->", raises(lambda: nn.Conv2d(1, 1, 4, padding=1.5)(torch.randn(1, 1, 32, 32))))
dr, bn = nn.Dropout(0.5).eval(), nn.BatchNorm1d(3)
bn.train(); bn(torch.randn(64, 3) * 4 + 2); bn.eval()
t = torch.randn(8, 3)
assert torch.equal(dr(t), t) and not torch.allclose(bn(t), t)                      # 1(f)
perm = torch.randperm(784); lin = nn.Linear(784, 10); lin2 = nn.Linear(784, 10)
with torch.no_grad():
    lin2.weight.copy_(lin.weight[:, perm]); lin2.bias.copy_(lin.bias)
img = torch.randn(4, 784)
assert torch.allclose(lin(img), lin2(img[:, perm]), atol=1e-5)                     # 1(b): permuted pixels, permuted columns


class FeedforwardLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.w1 = nn.Linear(d, d, bias=True)
        self.w2 = nn.Linear(d, d, bias=True)

    def forward(self, x):
        return self.w2(torch.sigmoid(self.w1(x)))


class MultiheadAttention(nn.Module):               # stand-in with the exam's signature
    def __init__(self, d, num_heads):
        super().__init__()
        self.inner = nn.MultiheadAttention(d, num_heads, batch_first=True)

    def forward(self, key, value, query):
        return self.inner(query, key, value, need_weights=False)[0]


class TransformerEncoder(nn.Module):
    def __init__(self, d, num_heads):
        super().__init__()
        self.d = d
        self.attn = MultiheadAttention(d, num_heads)
        self.norm1 = nn.LayerNorm(d)
        self.ff = FeedforwardLayer(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(key=x, value=x, query=x)
        x = self.norm1(x)
        x = x + self.ff(x)
        return self.norm2(x)


te = TransformerEncoder(16, 4)
assert te(torch.randn(3, 7, 16)).shape == (3, 7, 16) and count(FeedforwardLayer(16)) == 2 * (256 + 16)
assert count(te) == (4 * 256 + 4 * 16) + 2 * 32 + 544
print("     Sp26 Q7b key's MultiheadAttention(embed_dim=d, ...) with the given signature ->",
      raises(lambda: MultiheadAttention(embed_dim=16, num_heads=4), TypeError))
ok("Spring 2026: convolution, shapes and 225, |x| network, same-padding, eval-mode layers, encoder layer")

# ------------------------------------------------------------------ formulas used on the review pages
rnn = lambda dd, HH: HH * (dd + HH + 2)
assert count(nn.RNN(10, 20)) == rnn(10, 20) == 640 and count(nn.GRU(10, 20)) == 1920 and count(nn.LSTM(10, 20)) == 2560
assert count(nn.RNN(10, 20, bidirectional=True)) == 1280 and count(nn.RNN(10, 20, num_layers=2)) == 1480
assert count(nn.RNN(12, 32)) == 1472 and count(nn.RNN(10, 20, bias=False)) == 20 * 30
assert count(nn.GRU(32, 64, bidirectional=True)) == 37632 and count(nn.Conv2d(16, 32, 3)) == 4640 and count(nn.Conv2d(3, 16, 5)) == 1216
assert nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)(torch.randn(8, 16, 28, 28)).shape == (8, 32, 14, 14)
assert nn.MaxPool2d(2)(torch.randn(1, 1, 7, 7)).shape[-2:] == (3, 3)
assert count(nn.Embedding(5000, 64)) == 320000 and count(nn.LayerNorm(128)) == 256 and count(nn.BatchNorm2d(16)) == 32
assert count(nn.MultiheadAttention(128, 4)) == count(nn.MultiheadAttention(128, 1)) == 4 * 128 * 128 + 4 * 128
E = 128
assert count(nn.TransformerEncoderLayer(E, 4, 2 * E)) == 8 * E * E + 11 * E == 132480
assert count(nn.TransformerDecoderLayer(E, 4, 2 * E)) == 12 * E * E + 17 * E == 198784
assert count(nn.TransformerEncoderLayer(64, 4, 128)) == count(nn.TransformerEncoderLayer(64, 8, 128)) == 33472
print("     d_model=130, nhead=4 ->", raises(lambda: nn.TransformerEncoderLayer(130, 4)))
lay = nn.TransformerEncoderLayer(E, 4, 2 * E, dropout=0.0)
assert lay(torch.randn(9, E)).shape == (9, E)                                       # unbatched (tokens, d)
out = nn.Conv2d(3, 16, kernel_size=5, stride=3, padding=2)(torch.randn(1, 3, 90, 120)); assert out.shape[-2:] == (30, 40)
out = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=0)(torch.randn(1, 3, 90, 120)); assert out.shape[-2:] == (42, 57)
# practice problems
assert nn.Conv2d(3, 32, 7, stride=2, padding=3)(torch.randn(1, 3, 64, 64)).shape == (1, 32, 32, 32) and count(nn.Conv2d(3, 32, 7)) == 4736
Xp = torch.tensor([[1, 3, 0, 2], [5, 2, 1, 0], [0, 1, 4, 2], [2, 0, 3, 6.0]])[None, None]
assert nn.MaxPool2d(2)(Xp)[0, 0].tolist() == [[5, 2], [2, 6]] and nn.AvgPool2d(2)(Xp)[0, 0].tolist() == [[2.75, 0.75], [0.75, 3.75]]
assert nn.MaxPool2d(3, stride=1)(Xp)[0, 0].tolist() == [[5, 4], [5, 6]]
Xc = torch.tensor([[1, 0, 2, 1], [0, 1, 0, 3], [2, 1, 1, 0], [1, 0, 2, 2.0]])[None, None]; Wc = torch.tensor([[1, -1], [0, 2.0]])[None, None]
assert F.conv2d(Xc, Wc)[0, 0].tolist() == [[3, -2, 7], [1, 3, -3], [1, 4, 5]] and F.conv2d(Xc, Wc, stride=2)[0, 0].tolist() == [[3, 7], [1, 5]]
assert F.conv2d(Xc, Wc, stride=2, padding=1)[0, 0].tolist() == [[2, 4, 0], [4, 3, 3], [-1, -2, 2]]
assert len(DataLoader(TensorDataset(torch.zeros(1000)), batch_size=64)) == 16 and len(DataLoader(TensorDataset(torch.zeros(1000)), batch_size=64, drop_last=True)) == 15
img = torch.tensor([[7, 7, 0, 4, 3, 3, 1, 3], [8, 3, 5, 4, 6, 1, 9, 2], [5, 6, 2, 5, 1, 7, 1, 6], [8, 9, 8, 6, 6, 1, 5, 3.0]])
toks = nn.Unfold(2, stride=2)(img[None, None]).permute(0, 2, 1)[0]
assert toks.tolist() == [[7, 7, 8, 3], [0, 4, 5, 4], [3, 3, 6, 1], [1, 3, 9, 2], [5, 6, 8, 9], [2, 5, 8, 6], [1, 7, 6, 1], [1, 6, 5, 3]]
rgb = torch.arange(3 * 2 * 2.0).view(1, 3, 2, 2)
assert nn.Unfold(2, stride=2)(rgb)[0, :, 0].tolist() == list(range(12))                # channel 0 first, then 1, 2
opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1e-2, momentum=0.99)
sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 85], gamma=0.1)
lrs = []
for epoch in range(100):
    lrs.append(opt.param_groups[0]["lr"]); opt.step(); sch.step()
assert [round(lrs[i], 6) for i in (0, 69, 70, 84, 85, 99)] == [1e-2, 1e-2, 1e-3, 1e-3, 1e-4, 1e-4]
# the homework items kept as drills on the review pages
class SimpleCNN(nn.Module):                        # Homework 5, Problem 1(c), with the real name of the pooling layer
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=False)
        self.fc = nn.Linear(128, 5, bias=True)
        self.activation = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        B = x.size(0)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        return self.fc(self.gap(x).view(B, -1))


scnn = SimpleCNN()
assert [count(m) for m in (scnn.conv1, scnn.conv2, scnn.conv3, scnn.fc)] == [9408, 204800, 147456, 645] and count(scnn) == 362309
assert scnn(torch.randn(2, 3, 64, 64)).shape == scnn(torch.randn(2, 3, 90, 120)).shape == (2, 5)
# positional encoding (Homework 6, 2(c)): an encoder layer is permutation-equivariant, an RNN is order-sensitive
enc_l = nn.TransformerEncoderLayer(16, 4, 32, dropout=0.0).eval()
toks_ = torch.randn(6, 16); perm_ = torch.tensor([3, 0, 5, 1, 4, 2])
assert torch.allclose(enc_l(toks_)[perm_], enc_l(toks_[perm_]), atol=1e-5)
rnn_ = nn.RNN(16, 8)
assert not torch.allclose(rnn_(toks_)[1], rnn_(toks_[perm_])[1], atol=1e-3)


class ConvRecurrentLayer(nn.Module):               # images in a recurrence: convolutions in place of matrix products
    def __init__(self, c_in, c_hid):
        super().__init__()
        self.conv_x = nn.Conv2d(c_in, c_hid, 3, padding=1)
        self.conv_h = nn.Conv2d(c_hid, c_hid, 3, padding=1)

    def forward(self, x_t, h_prev):
        return torch.tanh(self.conv_x(x_t)) * torch.sigmoid(self.conv_h(h_prev))


crl = ConvRecurrentLayer(3, 8)
h_ = torch.zeros(1, 8, 16, 16)
for frame in torch.randn(4, 1, 3, 16, 16):
    h_ = crl(frame, h_)
assert h_.shape == (1, 8, 16, 16) and count(crl) == (3 * 8 * 9 + 8) + (8 * 8 * 9 + 8)
ok("review pages: RNN/GRU/LSTM counts, Transformer layer counts, kept homework drills, practice problems, Unfold order, MultiStepLR")

# ------------------------------------------------------------------ Lecture 21 notebook classes, as written
if len(sys.argv) > 1:
    import numpy as np
    nb = json.load(open(sys.argv[1]))
    ns = dict(np=np, torch=torch, nn=nn, F=F, DEVICE="cpu", MAX_LENGTH=25)
    for i in (6, 13, 17):
        exec("".join(nb["cells"][i]["source"]), ns)
    Ve, Vf = 1443, 1646
    totals = {}
    for dd in (32, 64, 128, 256):
        model = ns["Transformer"](ns["TransformerEncoder"](Ve, dd, 1, 4, dropout=0.0), ns["TransformerDecoder"](Vf, dd, 1, 4, dropout=0.0))
        totals[dd] = count(model)
    assert totals == {32: 238542, 64: 516398, 128: 1194990, 256: 3043694}, totals
    model = ns["Transformer"](ns["TransformerEncoder"](Ve, 128, 1, 4, dropout=0.0), ns["TransformerDecoder"](Vf, 128, 1, 4, dropout=0.0))
    assert count(model.encoder) == 445184 and count(model.decoder) == 749806
    src, tgt = torch.randint(2, Ve, (7, 1)), torch.randint(2, Vf, (5, 1))
    assert model(src, tgt).shape == (5, Vf)                                           # teacher-forced pass
    model.eval(); assert model(src).shape[1] == 1 and model(src).shape[0] <= 25       # greedy decoding
    two = ns["Transformer"](ns["TransformerEncoder"](Ve, 128, 2, 4), ns["TransformerDecoder"](Vf, 128, 2, 4)); assert count(two) == 1526254
    six = ns["Transformer"](ns["TransformerEncoder"](Ve, 128, 6, 4), ns["TransformerDecoder"](Vf, 128, 6, 4)); assert count(six) == 2851310
    clf = ns["TransformerTextClassifier"](64432, 64, 2, 4, 2, dropout=0.2)
    assert count(clf) == 4254722
    xrev = torch.randint(1, 64432, (1, 24)); xrev[0, 0] = 0
    assert clf(xrev).shape == (1, 2)
    pe = clf.positional_encoding(xrev)
    assert pe.shape == (1, 64) and torch.equal(pe[0], clf.positional_encoding.embedding.weight[0])
    print("     text classifier: positional_encoding(x) for x of shape (1, 24) has shape", tuple(pe.shape),
          "= row 0 of the table, broadcast to every token")
    clf.eval()
    perm_x = xrev.clone(); perm_x[0, 1:] = xrev[0, 1:].flip(0)                         # reverse the words, keep <CLS>
    assert torch.allclose(clf(xrev), clf(perm_x), atol=1e-5)
    print("     text classifier: reversing the word order leaves the logits unchanged (no order information)")
    pe_t = model.encoder.positional_encoding(src)
    assert pe_t.shape == (7, 128)                                                      # translation layout: one row per token
    vit = ns["VisionTransformer"](2, 8, 4, 16, dropout=0.2)
    assert count(vit) == 18912772 and count(vit.positional_encoding) == 7680000 and count(vit.patch_projector) == 590592
    assert vit(torch.randn(8, 3, 160, 160)).shape == (8, 4)
    for Wd, npatch in [(160, 10), (165, 10), (170, 11), (240, 15)]:
        assert vit.tokenize_image(torch.randn(1, 3, 160, Wd)).shape == (1, 10 * npatch, 768), Wd
    assert vit(torch.randn(1, 3, 160, 240)).shape == (1, 4)                            # 151 tokens, same weights
    ok("Lecture 21 notebook classes: parameter totals, shapes, positional-encoding behaviour, ViT tokenizer")
print("all torch checks passed")
