---
title: "Review: recurrent networks"
description: "Midterm 2 problem type 3: the RNN recurrence and its shapes, a forward pass by hand with tanh, parameter counts for nn.RNN, GRU, LSTM, bidirectional and stacked layers and for layers given by equations, the missing-state-update bug, what the LSTM and GRU gates do (their equations are on the formula sheet), and the true/false facts past exams asked about recurrent models."
tags: [exam, midterm-2, deep-nets]
aliases: ["RNN review", "recurrent networks review"]
---

*Midterm 2 · problem type 3 of 6 · HW6 Q1 · Lecture 17 (and 19 for seq2seq) · 15 points on Fall 2025, true/false items on the others · topic pages: [[5-deep-networks/recurrent-neural-networks|RNNs]], [[5-deep-networks/lstm-and-gru|LSTM and GRU]], [[5-deep-networks/bidirectional-and-stacked-rnns|bidirectional and stacked RNNs]] · back to [[9-midterm-2/index|Midterm 2]]*

> [!abstract] In one breath
> A recurrent layer is **one small network applied at every time step**, with its previous output fed back in: $h_t = f(x_t, h_{t-1})$. Everything asked about it follows from that sentence. The weights are shared over time, so the parameter count does not depend on the sequence length; the hidden state must be **carried** from step to step, so code that forgets to update it has no memory; and gradients pass through the same matrix once per step, which is why plain RNNs forget and gated ones (LSTM, GRU) were invented.

## The recurrence and its shapes

$$
h_t=\tanh\big(W_{ih}x_t+b_{ih}+W_{hh}h_{t-1}+b_{hh}\big),\qquad h_0=0 .
$$

| Quantity | Shape |
|---|---|
| input sequence | $(T, d)$: $T$ steps, $d$ features each |
| $W_{ih}$, $W_{hh}$ | $H\times d$, $H\times H$ |
| hidden state $h_t$ | $H$ |
| all outputs (one per step) | $(T, H)$ |
| final state | $(H)$, the summary used for a many-to-one task |

For classification of a whole sequence, a linear layer reads the **last** hidden state; for one output per step it reads every $h_t$ ([[5-deep-networks/recurrent-neural-networks#many-to-one-and-many-to-many|many-to-one and many-to-many]]).

## A forward pass by hand

Take $H = d = 2$, no biases, $h_0 = (0, 0)$ and

$$
W_{ih}=\begin{bmatrix}1&0\\0&-1\end{bmatrix},\qquad W_{hh}=\begin{bmatrix}0.5&0\\0.5&0.5\end{bmatrix},\qquad x_1=\begin{bmatrix}1\\0\end{bmatrix},\ x_2=\begin{bmatrix}0\\1\end{bmatrix},\ x_3=\begin{bmatrix}1\\1\end{bmatrix}.
$$

| $t$ | $W_{ih}x_t$ | $W_{hh}h_{t-1}$ | sum | $h_t=\tanh(\text{sum})$ |
|---|---|---|---|---|
| 1 | $(1, 0)$ | $(0, 0)$ | $(1, 0)$ | $(0.762,\ 0)$ |
| 2 | $(0, -1)$ | $(0.381,\ 0.381)$ | $(0.381,\ -0.619)$ | $(0.363,\ -0.551)$ |
| 3 | $(1, -1)$ | $(0.182,\ -0.094)$ | $(1.182,\ -1.094)$ | $(0.828,\ -0.798)$ |

Each row uses the $h$ of the row above: that hand-off is the whole mechanism. (Useful values: $\tanh 0 = 0$, $\tanh 1\approx0.762$, $\tanh(-z) = -\tanh z$, and $\lvert\tanh\rvert < 1$.)

## Counting parameters

| Layer | Count | $d = 10$, $H = 20$ |
|---|---|---|
| `nn.RNN(d, H)` | $H(d + H + 2)$ | 640 |
| `nn.GRU(d, H)` | $3H(d + H + 2)$ | 1,920 |
| `nn.LSTM(d, H)` | $4H(d + H + 2)$ | 2,560 |
| `nn.RNN(d, H, bidirectional=True)` | $2H(d + H + 2)$ | 1,280 |
| `nn.RNN(d, H, num_layers=2)` | $H(d + H + 2) + H(H + H + 2)$ | 1,480 |

(All five were confirmed by counting the parameters of the PyTorch layers.) The second layer of a stack reads the first layer's $H$-vectors, so its input size is $H$; after a bidirectional layer it is $2H$.

**A layer given by equations** has exactly what the equations show:

- [[9-midterm-2/fall-2025#problem-3--a-new-recurrent-layer-15-pts|F25 3]]: $z = \tanh(W_xx_t)$, $h_t = \tanh(W_h(z + h_{t-1}))$, no biases → $dH + H^2$.
- With biases, add one per output unit of each linear map: check-yourself 3 below and [[9-midterm-2/practice-problems#p7--a-recurrent-layer|practice P7]] are worked cases.

$T$ never appears. "State your answer in terms of $d$, $H$ and/or $T$" is an invitation to make that mistake.

## The bug to recognize

```python
h = torch.zeros(self.H)
for i in range(T):
    z = self.tanh(self.W_x(x[i]))
    h_next = self.tanh(self.W_h(z + h))
    hiddens.append(h_next)        # h is never updated
```

`h` stays zero, so every step sees only its own input: the "RNN" is a feed-forward layer applied $T$ times. Fix: `h = h_next` in the loop ([[9-midterm-2/fall-2025#problem-3--a-new-recurrent-layer-15-pts|F25 3(b)]]). The general test for recurrent code: **which variable carries information to the next iteration, and is it reassigned?**

## LSTM and GRU

The gate equations are on the [[9-midterm-2/fall-2025#the-formula-sheet-that-came-with-the-exam|formula sheet]]. What to know without it:

- **LSTM**: two states. The **cell state** $c_t = f_t\odot c_{t-1} + i_t\odot g_t$ is the long-term memory: the forget gate $f_t$ decides what to keep, the input gate $i_t$ what to write. The hidden state $h_t = o_t\odot\tanh(c_t)$ is what the layer outputs. Gates are sigmoids (values in $[0, 1]$, soft switches); the candidate $g_t$ is a tanh.
- **GRU**: one state, two gates. $h_t = (1 - z_t)\odot n_t + z_t\odot h_{t-1}$: the update gate $z_t$ mixes old state and new candidate.
- **Why they help**: the cell state is updated by adding, with a gate near 1 passing it on almost unchanged, so information and gradients survive many steps. A plain RNN squeezes everything through $\tanh(W_{hh}\,\cdot)$ at every step.
- **Not** skip connections between layers. That is ResNet ([[9-midterm-2/fall-2024#problem-1--true-or-false-30-pts|F24 1(e)]]).
- LSTMs suffer **less** from vanishing gradients than plain RNNs ([[9-midterm-2/spring-2025#problem-1--fact-or-cap-30-pts|Sp25 1(h)]]).

## Facts the exams and the homework asked

| Statement | Answer | Where |
|---|---|---|
| RNNs are feed-forward models | **No**: the hidden state feeds back | F24 1(d) |
| Standard RNNs keep a hidden state across time steps | Yes | Sp25 1(g) |
| A bidirectional layer shares parameters between directions | **No**: separate weights, twice the count | F25 1(j) |
| Input and output lengths must match in sequence transduction | **No**: an encoder–decoder generates until `<EOS>` | F25 1(i) |
| RNNs need positional encoding | **No**: they read tokens in order; self-attention does not | Homework 6, 2(c); [[9-midterm-2/review-attention-and-transformers#positional-encoding\|worked]] |

Bidirectional layers need the **whole sequence** before they can run, so they suit classification and encoders, not step-by-step generation. For sequences of images (video), replace the matrix products $W_xx_t$ and $W_hh_{t-1}$ by convolutions, so the state is a feature map (the lecturer's own example in class on Nov 10, from a Homework 6 question).

> [!trap] Common mistakes
> - **Multiplying a count by $T$.** Weights are shared across time.
> - **One bias vector for `nn.RNN`** (it has two), or biases in a count when the code says `bias=False`.
> - **Starting step $t$ from $h_0$** instead of $h_{t-1}$ in a hand computation.
> - **Confusing $c_t$ and $h_t$** in an LSTM: the cell state is the memory, the hidden state is the output.
> - **"Bidirectional" for a task that generates** one token at a time.

> [!question]- Check yourself
> 1. `nn.GRU(32, 64, bidirectional=True)`: parameter count?
> 2. In the hand example above, what is $h_1$ if $x_1 = (0, 0)$? What does that say about an all-zero first input?
> 3. Your friend's layer is $h_t = \tanh(W_xx_t + b_x) + \tanh(W_hh_{t-1} + b_h)$ with $x_t\in\mathbb R^{12}$, $h\in\mathbb R^{32}$. Count the parameters and compare with `nn.RNN(12, 32)`.
>
> > [!solution]- Answers
> > 1. $2\cdot3\cdot64(32 + 64 + 2) = 37{,}632$.
> > 2. $h_1 = \tanh(0) = (0, 0)$: with no biases and a zero state, a zero input leaves the state at zero.
> > 3. $(12\cdot32 + 32) + (32\cdot32 + 32) = 416 + 1{,}056 = 1{,}472$, the same as `nn.RNN(12, 32)`: $32(12 + 32 + 2) = 1{,}472$. Two linear layers with biases are exactly $W_{ih}, b_{ih}, W_{hh}, b_{hh}$.

Related: [[9-midterm-2/review-sizes-and-parameters]] · [[9-midterm-2/review-pytorch-modules]] · [[9-midterm-2/review-attention-and-transformers]] · [[9-midterm-2/review-choosing-models]]
