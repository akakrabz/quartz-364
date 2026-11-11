---
title: "Review: sizes, shapes and parameter counts"
description: "Midterm 2 problem type 1 (on every past exam, 10–25 points): the output-size formula for convolution and pooling with every past instance, how each layer changes a shape, parameter counts for Linear, Conv2d, normalization, embedding, RNN/GRU/LSTM and attention layers, doing a pooling or convolution window by hand, the receptive-field recursion from Homework 5, and Vision Transformer token counts."
tags: [exam, midterm-2, deep-nets]
aliases: ["sizes and parameter counts", "output size review", "parameter counting review"]
---

*Midterm 2 · problem type 1 of 6 · HW4 Q2, HW5 Q1–Q2, HW6 Q1–Q2 · Lectures 14–17, 20–21 · on all four past exams · topic pages: [[5-deep-networks/convolutional-layers|convolutional layers]], [[5-deep-networks/pooling|pooling]], [[5-deep-networks/multilayer-perceptrons|MLPs]] · back to [[9-midterm-2/index|Midterm 2]]*

> [!abstract] In one breath
> Three questions keep coming back: **what shape comes out** of a layer, **how many parameters** it has, and **what numbers come out** of one small pooling or convolution. All three are bookkeeping. One formula gives every spatial size, $\lfloor (n + 2p - K)/s\rfloor + 1$; one habit gives every count, "weights = outputs × (what one output looks at), plus one bias per output unless `bias=False`"; and a window table gives the numbers.

## The output-size formula

For one spatial direction of length $n$, kernel $K$, stride $s$, padding $p$:

$$
n_{out}=\left\lfloor\frac{n+2p-K}{s}\right\rfloor+1 .
$$

It holds for `nn.Conv2d` and for pooling (pooling: $p = 0$ unless stated, and `stride` defaults to the kernel size). Height and width are computed separately. The lecturer said on Nov 10 that this formula will be **given** on the formula sheet; the work is using it without slips.

| Where | Layer | Input | Computation | Output |
|---|---|---|---|---|
| [[9-midterm-2/fall-2024#problem-4--a-convolutional-layer-and-a-max-pool-15-pts\|F24 4(a)]] | conv $K{=}5$, $s{=}1$, $p{=}0$ | 40 | $\lfloor35/1\rfloor+1$ | 36 |
| [[9-midterm-2/fall-2024#problem-4--a-convolutional-layer-and-a-max-pool-15-pts\|F24 4(c)]] | max pool $K{=}2$, $s{=}3$ | 6 | $\lfloor4/3\rfloor+1$ | 2 |
| [[9-midterm-2/fall-2025#problem-2--layer-shapes-and-augmentations-17-pts\|F25 2(a)]] i | conv $K{=}3$, $s{=}3$, $p{=}1$ | 30 | $\lfloor29/3\rfloor+1$ | 10 |
| F25 2(a) ii | conv $K{=}3$, $s{=}3$, $p{=}3$ | 30 | $\lfloor33/3\rfloor+1$ | 12 |
| F25 2(a) iii | conv $K{=}1$, $s{=}3$, $p{=}0$ | 30 | $\lfloor29/3\rfloor+1$ | 10 |
| F25 2(a) iv | max pool $K{=}3$, $s{=}1$ | 30 | $\lfloor27/1\rfloor+1$ | 28 |
| F25 2(a) v | max pool $K{=}3$, $s{=}3$ | 30 | $\lfloor27/3\rfloor+1$ | 10 |
| [[9-midterm-2/spring-2025#problem-3--lifes-a-pool-dive-in-10-pts\|Sp25 3]] | max pool $K{=}3$, $s{=}2$ | 5 | $\lfloor2/2\rfloor+1$ | 2 |
| [[9-midterm-2/spring-2026#problem-3--lifes-a-pool-dive-in-10-pts\|Sp26 3]] | conv $K{=}3$, $s{=}2$, $p{=}0$ | 5 | $\lfloor2/2\rfloor+1$ | 2 |
| HW5 1(a) | conv $K{=}5$, $s{=}3$, $p{=}2$ | $90\times120$ | $\lfloor89/3\rfloor+1$, $\lfloor119/3\rfloor+1$ | $30\times40$ |
| HW5 1(b) | conv $K{=}7$, $s{=}2$, $p{=}0$ | $90\times120$ | $\lfloor83/2\rfloor+1$, $\lfloor113/2\rfloor+1$ | $42\times57$ |

The two Homework 5 rows are the only non-square inputs in the table: height and width go through the formula **separately**, and both floors round a non-integer down ($89/3 = 29.67$, $119/3 = 39.67$).

Two facts that settle true/false items:

- **"Same" size** with stride 1 needs $p = (K-1)/2$, which exists only for **odd** $K$ ([[9-midterm-2/spring-2026#problem-1--fact-or-cap-30-pts|Sp26 1(a)]]).
- The floor means inputs can be **dropped**: at $s = 3$, $K = 2$ a third of the rows and columns are never read.

## What each layer does to a shape

| Layer | Shape in → out |
|---|---|
| `nn.Linear(a, b)` | `(…, a)` → `(…, b)`: only the last dimension changes |
| activation, dropout, batch norm, layer norm | unchanged |
| `nn.Conv2d(Cin, Cout, K, s, p)` | `(B, Cin, H, W)` → `(B, Cout, H', W')` |
| `nn.MaxPool2d`, `nn.AvgPool2d` | channels unchanged; $H$, $W$ by the formula |
| global average pooling | `(B, C, H, W)` → `(B, C)` ([[9-midterm-2/fall-2025#problem-1--true-or-false-30-pts\|F25 1(h)]]) |
| flatten | `(B, C, H, W)` → `(B, C·H·W)` |
| `nn.Embedding(V, d)` | integer indices `(…)` → `(…, d)` |
| RNN layer, hidden size $H$ | `(T, d)` → `(T, H)` outputs and a final state `(H)` |
| attention / a Transformer encoder layer | `(N, d)` → `(N, d)`; the weights are $N\times N$ |

Examples: [[9-midterm-2/spring-2025#problem-4--who-wants-to-be-a-parameterionaire-10-pts|Sp25 4(a)]] `(1, 8)` → `(1, 16)` → `(1, 16)` → `(1, 4)`; [[9-midterm-2/spring-2026#problem-4--who-wants-to-be-a-parameterionaire-10-pts|Sp26 4(a)]] `(1, 10)` → `(1, 12)` → `(1, 12)` → `(1, 6)` → `(1, 6)` → `(1, 3)`.

## Counting parameters

| Layer | Parameters | Why |
|---|---|---|
| `nn.Linear(a, b)` | $ab + b$ | a $b\times a$ matrix and $b$ biases |
| `nn.Conv2d(Cin, Cout, K)` | $C_{out}(C_{in}K^2 + 1)$ | $C_{out}$ filters, each $C_{in}\times K\times K$, plus a bias each |
| pooling, activations, dropout, flatten | 0 | nothing to learn |
| `nn.BatchNorm2d(C)` / `nn.LayerNorm(d)` | $2C$ / $2d$ | a scale and a shift per channel / per feature |
| `nn.Embedding(V, d)` | $Vd$ | one row per vocabulary entry |
| `nn.RNN(d, H)` | $H(d + H + 2)$ | $W_{ih}$, $W_{hh}$ and **two** bias vectors |
| `nn.GRU(d, H)` / `nn.LSTM(d, H)` | $3\times$ / $4\times$ the RNN count | 3 / 4 gate blocks of the same size |
| bidirectional | $2\times$ | a separate set per direction |
| `nn.MultiheadAttention(E, h)` | $4E^2 + 4E$ | Q, K, V and output projections; **independent of $h$** |
| `nn.TransformerEncoderLayer(E, h, F)` | $4E^2 + 2EF + 9E + F$ | attention + two linears + two layer norms |

Subtract the bias term whenever the code says `bias=False`. The size of the **input image or sequence never enters** a conv, RNN or attention count; it does enter a `Linear` that follows a flatten.

Past instances:

| Where | Model | Count |
|---|---|---|
| [[9-midterm-2/fall-2024#problem-3--a-custom-block-20-pts\|F24 3(a)]] | `Linear(N,N)`, `Linear(N,N)`, `Linear(N,D)` | $2N^2 + 2N + ND + D$ |
| [[9-midterm-2/fall-2024#problem-4--a-convolutional-layer-and-a-max-pool-15-pts\|F24 4(b)]] | `Conv2d(3, 16, 5, bias=False)` | $16\cdot3\cdot25 = 1{,}200$ |
| [[9-midterm-2/fall-2025#problem-3--a-new-recurrent-layer-15-pts\|F25 3(a)]] | `Linear(d,H)`, `Linear(H,H)`, no biases | $dH + H^2$ |
| [[9-midterm-2/fall-2025#problem-5--mlp-cnn-or-vision-transformer-23-pts\|F25 5(a)]] | `Linear(3·64·64, 512)` | $12{,}288\cdot512 + 512 = 6{,}291{,}968$ |
| [[9-midterm-2/spring-2025#problem-4--who-wants-to-be-a-parameterionaire-10-pts\|Sp25 4(b)]] | `Linear(8,16)`, `Linear(16,4)` | $144 + 68 = 212$ |
| [[9-midterm-2/spring-2026#problem-4--who-wants-to-be-a-parameterionaire-10-pts\|Sp26 4(b)]] | `Linear(10,12)`, `Linear(12,6, bias=False)`, `Linear(6,3)` | $132 + 72 + 21 = 225$ |

**A whole CNN, layer by layer** (Homework 5, Problem 1(c)). The practice exams count one layer at a time; this is the same skill on a full model: three bias-free convolutions with ReLUs, global average pooling, and a linear layer for 5 classes.

| Layer | Count | Parameters |
|---|---|---|
| `nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False)` | $64\cdot3\cdot7^2$ | 9,408 |
| `nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1, bias=False)` | $128\cdot64\cdot5^2$ | 204,800 |
| `nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=False)` | $128\cdot128\cdot3^2$ | 147,456 |
| `nn.ReLU()`, `nn.AdaptiveAvgPool2d((1, 1))` | nothing to learn | 0 |
| `nn.Linear(128, 5, bias=True)` | $128\cdot5+5$ | 645 |
| **total** | | **362,309** |

What to take from it: stride and padding appear in no count; the middle layer holds more than half of the parameters although its kernel is smaller than the first one's, because $C_{in}\cdot C_{out}$ is what grows; and the head is `Linear(128, 5)` for any image size, because global pooling leaves one number per channel. (Built and counted with PyTorch: 362,309, and the model accepts $64\times64$ and $90\times120$ images alike.)

> [!tip] Shortcut — find the sizes from the equation first
> When a problem gives equations instead of code, the shapes are decided by the operations that need **equal sizes**: element-wise products and sums. In $y_1 = \sigma(W_1x)\odot x$ the product forces $W_1x$ to be as long as $x$, so $W_1$ is $N\times N$. In $h_t = \tanh(W_h(z + h_{t-1}))$ the sum forces $z\in\mathbb R^H$, so $W_x$ maps $d\to H$. Sizes first, then count.

## One window at a time

For pooling or convolution by hand:

1. Output size from the formula (so you know how many windows).
2. Window $(i, j)$ starts at row $i\cdot s$, column $j\cdot s$ and covers $K$ rows and $K$ columns.
3. Pooling: take the max (or mean) of the window. Convolution: multiply the window entry by entry with the kernel and add. `nn.Conv2d` does **not** flip the kernel ([[5-deep-networks/convolution#worked-example-convolution-vs-cross-correlation|cross-correlation]]).
4. Write a small table: window, entries, result. It is where the partial credit is.

| Where | Operation | Windows | Result |
|---|---|---|---|
| [[9-midterm-2/fall-2024#problem-4--a-convolutional-layer-and-a-max-pool-15-pts\|F24 4(c)]] | max, $K{=}2$, $s{=}3$ on $6\times6$ | rows 0–1 and 3–4, cols 0–1 and 3–4 (gaps) | $[[4, 6], [6, 7]]$ |
| [[9-midterm-2/spring-2025#problem-3--lifes-a-pool-dive-in-10-pts\|Sp25 3]] | max, $K{=}3$, $s{=}2$ on $5\times5$ | rows 0–2 and 2–4, cols 0–2 and 2–4 (overlap) | $[[4, 5], [5, 5]]$ |
| [[9-midterm-2/spring-2026#problem-3--lifes-a-pool-dive-in-10-pts\|Sp26 3]] | conv, $K{=}3$, $s{=}2$ on $5\times5$ | the same four windows | $[[0, 2], [1, -1]]$ |

## Receptive field

How large a patch of the input one activation depends on. No practice exam asks for a receptive field, but Homework 5 (Problem 2) does, and the lecturer listed it in class with output sizes and parameter counts, so the three homework cases are worked here. With $r_0 = j_0 = 1$, for each layer with kernel $k_l$ and stride $s_l$:

$$
r_l=r_{l-1}+(k_l-1)\,j_{l-1},\qquad j_l=j_{l-1}\,s_l .
$$

$j$ ("jump") is how many input pixels one step at this layer corresponds to. Pooling layers count with their kernel and stride; activations do nothing; padding does not matter.

**Worked: a $7\times7$ convolution with stride 1, then a $3\times3$ convolution with stride 2.** Layer 1: $r = 1 + (7-1)\cdot1 = 7$, $j = 1\cdot1 = 1$. Layer 2: $r = 7 + (3-1)\cdot1 = 9$, $j = 1\cdot2 = 2$. Receptive field $9\times9$. Note the order: a layer's own stride changes $j$ **after** its $r$ is computed, so the stride 2 here would only matter to a third layer.

| Layers $(k, s)$ in order | $(r, j)$ after each layer | Receptive field |
|---|---|---|
| $(7, 1), (3, 2)$ — HW5 2(a), worked above | $(7, 1), (9, 2)$ | $9\times9$ |
| $(3, 1), (3, 2)$ three times, then $(3, 1)$ — HW5 2(b) | $(3, 1), (5, 2), (9, 2), (13, 4), (21, 4), (29, 8), (45, 8)$ | $45\times45$ |
| how many blocks of $(5, 1), (5, 2)$ for at least $100\times100$? — HW5 2(c) | $r$ after 1, 2, 3, 4 blocks: 9, 25, 57, 121 | 4 blocks |

The pattern in the last row: each block adds $(4 + 4)\cdot j$ with $j$ doubling per block, so $r = 1 + 8(2^n - 1)$ after $n$ blocks. Early strides make the field grow fast.

## Vision Transformer tokens

For an $H\times W$ image with $C$ channels and $P\times P$ patches at stride $P$: $(H/P)(W/P)$ tokens (plus 1 with a class token), each of length $CP^2$.

- [[9-midterm-2/fall-2025#problem-5--mlp-cnn-or-vision-transformer-23-pts|F25 5(b)]]: $64/8 = 8$ per side → 64 tokens of length $3\cdot64 = 192$; at $128\times128$, 256 tokens.
- Lecture 21: $160/16 = 10$ per side → 100 patches + `<CLS>` = 101 tokens of length 768.
- HW6 2(d): a one-channel $4\times8$ image with $2\times2$ patches → 8 tokens of length 4 ([[7-transformers/vision-transformer#tokenizing-an-image-by-hand|worked]]).

> [!trap] Common mistakes
> - **Forgetting the $+1$** in the size formula, or rounding up instead of down.
> - **Using the kernel as the stride for a convolution.** Only pooling defaults to `stride = kernel_size`; `nn.Conv2d` defaults to stride 1.
> - **Counting a bias that is switched off** (`bias=False` in F24 4(b), F25 3, Sp26 4, HW5 1(c)), or forgetting it when it is on.
> - **Letting image size or sequence length into a count** for conv, RNN or attention layers.
> - **$C_{in}$ left out of a conv filter.** Each filter is $C_{in}\times K\times K$, not $K\times K$.
> - **One bias for `nn.RNN`.** PyTorch's layer has two bias vectors ($b_{ih}$, $b_{hh}$); a layer you write with two `nn.Linear` also has two, and one written from an equation has what the equation shows.

> [!question]- Check yourself
> 1. `nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)` on a `(8, 16, 28, 28)` batch: output shape and parameter count?
> 2. `nn.MaxPool2d(2)` on a $7\times7$ map: output size, and which inputs are never read?
> 3. `nn.LSTM(10, 20)`: how many parameters?
> 4. Three layers with $(k, s) = (3, 2)$ each: receptive field?
>
> > [!solution]- Answers
> > 1. $\lfloor(28 + 2 - 3)/2\rfloor + 1 = 14$, so `(8, 32, 14, 14)`; $32(16\cdot9 + 1) = 4{,}640$.
> > 2. $\lfloor(7-2)/2\rfloor + 1 = 3$, so $3\times3$; the last row and the last column.
> > 3. $4\cdot20(10 + 20 + 2) = 2{,}560$.
> > 4. $r$: 3, then $3 + 2\cdot2 = 7$, then $7 + 2\cdot4 = 15$: $15\times15$.

Related: [[9-midterm-2/review-pytorch-modules]] · [[9-midterm-2/review-recurrent-networks]] · [[9-midterm-2/formula-sheet]] · [[9-midterm-2/practice-problems]]
