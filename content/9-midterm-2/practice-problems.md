---
title: "Midterm 2 practice problems"
description: "Ten new Midterm 2 problems in the style of the past exams, with folded answers checked by computation: a CNN's shapes and parameters layer by layer, pooling three ways, a strided convolution by hand, an MLP with one bias off, attention scores with a causal mask, Vision Transformer token counts, a recurrent layer to dimension and debug, a ReLU network for a clipped ramp, DataLoader batch arithmetic, and an embedding table against a one-hot linear layer."
tags: [exam, midterm-2]
aliases: ["Midterm 2 practice", "MT2 practice problems"]
---

*Midterm 2 · new problems in the format of the [[9-midterm-2/index|past exams]] · every answer checked by `tools/verify_mt2.py` and, where it is PyTorch behaviour, by running PyTorch · the methods are on the review pages: [[9-midterm-2/review-sizes-and-parameters|sizes and parameters]], [[9-midterm-2/review-pytorch-modules|modules]], [[9-midterm-2/review-recurrent-networks|recurrent networks]], [[9-midterm-2/review-attention-and-transformers|attention]], [[9-midterm-2/review-training-and-activations|training and activations]], [[9-midterm-2/review-choosing-models|choosing a model]]*

## P1 — A CNN, layer by layer

> [!question] P1
> ```python
> net = nn.Sequential(
>     nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
>     nn.ReLU(),
>     nn.MaxPool2d(2),
>     nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
> )
> ```
> The input is a batch of shape `(8, 3, 64, 64)`. Give the shape after each of the four layers and the number of learnable parameters of each.

> [!solution]- Answer
> | layer | computation | shape | parameters |
> |---|---|---|---|
> | conv 1 | $\lfloor(64+6-7)/2\rfloor+1 = 32$ | `(8, 32, 32, 32)` | $32(3\cdot49+1) = 4{,}736$ |
> | ReLU | | `(8, 32, 32, 32)` | 0 |
> | max pool 2 (stride 2) | $\lfloor(32-2)/2\rfloor+1 = 16$ | `(8, 32, 16, 16)` | 0 |
> | conv 2 | $\lfloor(16+2-3)/1\rfloor+1 = 16$ | `(8, 64, 16, 16)` | $64\cdot32\cdot9 = 18{,}432$ (no bias) |

## P2 — Pooling three ways

> [!question] P2
> $$
> X=\begin{bmatrix}1&3&0&2\\5&2&1&0\\0&1&4&2\\2&0&3&6\end{bmatrix}
> $$
> Compute (a) `nn.MaxPool2d(2)`, (b) `nn.AvgPool2d(2)`, (c) `nn.MaxPool2d(3, stride=1)`.

> [!solution]- Answer
> (a) and (b) use four non-overlapping $2\times2$ windows; (c) uses four overlapping $3\times3$ windows ($\lfloor(4-3)/1\rfloor+1 = 2$ each way).
> $$
> \text{(a)}\ \begin{bmatrix}5&2\\2&6\end{bmatrix}\qquad\text{(b)}\ \begin{bmatrix}2.75&0.75\\0.75&3.75\end{bmatrix}\qquad\text{(c)}\ \begin{bmatrix}5&4\\5&6\end{bmatrix}
> $$

## P3 — A convolution by hand

> [!question] P3
> $$
> X=\begin{bmatrix}1&0&2&1\\0&1&0&3\\2&1&1&0\\1&0&2&2\end{bmatrix},\qquad W=\begin{bmatrix}1&-1\\0&2\end{bmatrix}
> $$
> Compute the output of the convolution (as `nn.Conv2d` computes it, no bias) with (a) stride 1, (b) stride 2, (c) stride 2 and padding 1.

> [!solution]- Answer
> Each output is $\text{top-left}-\text{top-right}+2\cdot\text{bottom-right}$ of its window.
> $$
> \text{(a)}\ \begin{bmatrix}3&-2&7\\1&3&-3\\1&4&5\end{bmatrix}\qquad\text{(b)}\ \begin{bmatrix}3&7\\1&5\end{bmatrix}\qquad\text{(c)}\ \begin{bmatrix}2&4&0\\4&3&3\\-1&-2&2\end{bmatrix}
> $$
> (b) is rows and columns 0 and 2 of (a). For (c) the padded input is $6\times6$ with a border of zeros, and the size is $\lfloor(4+2-2)/2\rfloor+1 = 3$; for example the top-left output is $0 - 0 + 2\cdot1 = 2$.

## P4 — Count the parameters

> [!question] P4
> ```python
> model = nn.Sequential(
>     nn.Linear(20, 50),
>     nn.Tanh(),
>     nn.Linear(50, 50, bias=False),
>     nn.ReLU(),
>     nn.Dropout(0.2),
>     nn.Linear(50, 3),
> )
> ```
> How many learnable parameters? What is the output shape for an input of shape `(16, 20)`?

> [!solution]- Answer
> $(20\cdot50+50)+(50\cdot50)+(50\cdot3+3) = 1{,}050+2{,}500+153 = 3{,}703$. Activations and dropout have none. Output shape `(16, 3)`.

## P5 — Attention scores and a causal mask

> [!question] P5
> A self-attention block receives, for three tokens,
> $$
> Q=\begin{bmatrix}1&0\\0&2\\1&1\end{bmatrix},\qquad K=\begin{bmatrix}2&1\\0&1\\-1&3\end{bmatrix},\qquad V=\begin{bmatrix}1&0\\0&1\\2&2\end{bmatrix}.
> $$
> (a) Which token does each token attend to most? (b) Now apply the causal mask and, for simplicity, **no** $\sqrt{d_k}$ scaling. Give the attention weights and the output for tokens 1 and 2.

> [!solution]- Answer
> **(a)**
> $$
> QK^\top=\begin{bmatrix}2&0&-1\\2&2&6\\3&1&2\end{bmatrix}
> $$
> Row maxima: token 1 → token 1, token 2 → token 3, token 3 → token 1.
>
> **(b)** The mask removes everything above the diagonal. Token 1 sees only itself: weights $(1, 0, 0)$, output $V_1 = (1, 0)$. Token 2 keeps scores $(2, 2)$: equal, so weights $(0.5, 0.5, 0)$ and output $0.5V_1 + 0.5V_2 = (0.5, 0.5)$. The mask changed token 2's answer: unmasked it attended mostly to token 3 (weight 0.96).
>
> (Token 3 is unaffected by the mask: weights $(0.665, 0.090, 0.245)$, output $(1.155, 0.580)$.)

## P6 — Vision Transformer tokens

> [!question] P6
> RGB images of size $96\times96$ go into a Vision Transformer with a class token. How many tokens enter the encoder, and how long is each flattened patch, for (a) $16\times16$ patches, (b) $8\times8$ patches? Which choice makes the attention computation more expensive, and by about how much?

> [!solution]- Answer
> (a) $(96/16)^2 = 36$ patches, 37 tokens, each patch $3\cdot16\cdot16 = 768$ numbers. (b) $(96/8)^2 = 144$ patches, 145 tokens, each $3\cdot8\cdot8 = 192$ numbers. Attention builds an $N\times N$ table, so (b) costs about $(145/37)^2\approx15$ times as much per layer.

## P7 — A recurrent layer

> [!question] P7
> $$
> a_t=\tanh(W_xx_t+b_x),\qquad h_t=\tanh(W_h(a_t\odot h_{t-1})+b_h),\qquad x_t\in\mathbb R^{12},\ h_t\in\mathbb R^{32}.
> $$
> (a) Give the `nn.Linear` layers for $W_x$ and $W_h$ and the number of parameters. (b) With $h_0 = 0$, what is $h_1$? What does that say about this design?

> [!solution]- Answer
> **(a)** $a_t\odot h_{t-1}$ forces $a_t\in\mathbb R^{32}$: `nn.Linear(12, 32)` and `nn.Linear(32, 32)`. Parameters: $(12\cdot32+32)+(32\cdot32+32) = 416+1{,}056 = 1{,}472$, the same as `nn.RNN(12, 32)`.
>
> **(b)** $a_1\odot h_0 = 0$ whatever $x_1$ is, so $h_1 = \tanh(b_h)$: the first input is ignored. Worse, if $b_h = 0$ the state stays at zero for ever. A multiplicative gate applied to a zero state passes nothing; this layer needs a non-zero initial state (or an additive path, as real RNNs have).

## P8 — A ReLU network by hand

> [!question] P8
> Find $w_0, b_0, w_1\in\mathbb R^2$ so that $f(x) = w_1^\top\mathrm{ReLU}(w_0x + b_0)$ equals
> $$
> f(x)=\begin{cases}0&x\le0\\x&0\le x\le1\\1&x\ge1\end{cases}
> $$

> [!solution]- Answer
> The function has kinks at 0 and 1, so one hidden unit switches on at each: $\mathrm{ReLU}(x)$ and $\mathrm{ReLU}(x-1)$. The slope must be 1 between the kinks and 0 after the second, so subtract:
> $$
> f(x)=\mathrm{ReLU}(x)-\mathrm{ReLU}(x-1),\qquad w_0=\begin{bmatrix}1\\1\end{bmatrix},\ b_0=\begin{bmatrix}0\\-1\end{bmatrix},\ w_1=\begin{bmatrix}1\\-1\end{bmatrix}.
> $$
> Check: $x = -1\to0$; $x = 0.5\to0.5$; $x = 1\to1$; $x = 2\to2-1 = 1$.

## P9 — Batches and steps

> [!question] P9
> A training set has 1,000 examples and the `DataLoader` uses `batch_size=64`. (a) How many batches per epoch, and how large is the last one? (b) With `drop_last=True`? (c) How many `optimizer.step()` calls in 5 epochs for (a)?

> [!solution]- Answer
> (a) $\lceil1000/64\rceil = 16$ batches: 15 of 64 and one of $1000-960 = 40$. (b) 15. (c) $5\cdot16 = 80$.

## P10 — Embedding or one-hot?

> [!question] P10
> A vocabulary has 5,000 words and the model width is 64. Compare `nn.Embedding(5000, 64)` with one-hot vectors fed to `nn.Linear(5000, 64, bias=False)`: number of parameters, and what each computes for word number 17.

> [!solution]- Answer
> Both have $5{,}000\cdot64 = 320{,}000$ parameters and compute the same thing: the linear layer multiplies its weight matrix by a one-hot vector, which picks out column 17; the embedding looks up row 17 directly. The embedding skips building a 5,000-long vector and multiplying by it. → [[7-transformers/text-encodings-and-embeddings#an-embedding-is-a-one-hot-vector-times-a-matrix|an embedding is a one-hot vector times a matrix]]

Related: [[9-midterm-2/true-false]] · [[9-midterm-2/formula-sheet]] · [[6-midterm-1/practice-problems]]
