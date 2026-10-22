---
title: "Bidirectional and stacked RNNs"
description: "Two ways to enrich a basic RNN: a bidirectional RNN reads the sequence forwards and backwards with separate weights and combines both hidden states; a stacked RNN feeds each layer's output into another recurrent layer. Equations, from-scratch code, the surname results, parameter counts, and how nn.RNN(num_layers, bidirectional=True) lays out output and h_n."
tags: [topic, deep-nets]
aliases: ["bidirectional RNN", "bi-directional RNN", "BiRNN", "stacked RNN", "multi-layer RNN", "deep RNN", "num_layers", "bidirectional=True"]
---

*Lecture 17 · [[5-deep-networks/index|Deep networks]] · builds on [[5-deep-networks/recurrent-neural-networks|recurrent neural networks]] · next: [[5-deep-networks/lstm-and-gru|LSTM and GRU]]*

> [!abstract] In one breath
> A **bidirectional** RNN runs two independent recurrences, one left-to-right ($h_t$) and one right-to-left ($h'_t$), and predicts from their concatenation $h_t\oplus h'_t$, so every step sees both past and future context. A **stacked** RNN puts recurrent layers on top of each other: layer 2 treats layer 1's per-step outputs as its input sequence, with its own hidden state. `nn.RNN(..., num_layers=L, bidirectional=True)` does both: `output` becomes `(T, N, 2H)` and `h_n` becomes `(2L, N, H)`.

## What it is

### Bidirectional: context from the future

A plain RNN is **causal**: $h_t$ depends only on $x_1,\dots,x_t$. Sometimes the disambiguating word comes later:

- *The **bark** from that short dog is loud!*
- *The **bark** from that short tree is dark!*

Reading left to right, the four words after "bark" are the same in both sentences; reading right to left, "loud" or "dark" arrives first and settles the meaning. A **bidirectional RNN** gives the model this **non-causal** view by processing the sequence both ways at once, with **separate parameters** for each direction:
$$
\begin{aligned}
v_t &= x_t\oplus h_{t-1}, & h_t &= \tanh(W_h v_t) & &\text{(forward, reads } x_1\to x_T)\\
v'_t &= x_t\oplus h'_{t+1}, & h'_t &= \tanh(W'_h v'_t) & &\text{(backward, reads } x_T\to x_1)\\
z_t &= h_t\oplus h'_t, & y_t &= W_y z_t & &\text{(output from both)}
\end{aligned}
$$

![[l17-rnn-bidirectional.png|Bidirectional RNN: forward cells A and backward cells A′ at every step]]

In the figure, $s = h$ and $A = W$: the light cells $A$ carry the forward state $s_0\to s_i$, the dark cells $A'$ carry the backward state $s'_0\to s'_i$ from the other end, and each output $y_t$ reads both.

### Stacked: depth at every time step

A **stacked** (multi-layer) RNN adds layers *at each time step*, like depth in an MLP. Layer 1 reads the input sequence; layer 2 reads layer 1's per-step outputs as *its* input sequence, with its **own** hidden state $h^{(2)}_t$ (not the same as $h^{(1)}_t$). The lecture's two-layer version:
$$
\begin{aligned}
v_t &= x_t\oplus h^{(1)}_{t-1}, & h^{(1)}_t &= \tanh\big(W^{(1)}_h v_t\big), & y^{(1)}_t &= W^{(1)}_y v_t,\\
z_t &= \tanh\big(y^{(1)}_t\big)\oplus h^{(2)}_{t-1}, & h^{(2)}_t &= \tanh\big(W^{(2)}_h z_t\big), & y^{(2)}_t &= W^{(2)}_y z_t .
\end{aligned}
$$
With $L$ layers and $T$ steps there are $T\times L$ hidden state vectors.

## What you should know

- Why a bidirectional RNN helps (future context) and when it **cannot** be used (online/causal prediction: the backward pass needs the whole sequence first).
- The forward and backward directions have **separate weights**; their states are **concatenated**, so the head sees $2H$ features.
- In a stacked RNN, each layer has its own weights and hidden state; layer $\ell\ge 2$ takes layer $\ell-1$'s outputs as input.
- `nn.RNN` shapes with $D=2$ directions and $L$ layers: `output` `(T, N, 2H)` (forward half `[..., :H]`, backward half `[..., H:]`), `h_n` `(2L, N, H)`, viewable as `(L, 2, N, H)`.
- Which positions summarize the whole sequence: forward at $t=T$, backward at $t=1$.
- Parameter counts: double per direction; layers $\ge 2$ have input size $D\cdot H$.

## Bidirectional RNN from scratch

```python
class BidirectionalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc_h = nn.Linear(input_dim + hidden_dim, hidden_dim)        # W_h
        self.fc_h_prime = nn.Linear(input_dim + hidden_dim, hidden_dim)  # W'_h
        self.fc_y = nn.Linear(2 * hidden_dim, output_dim)                # W_y on h ⊕ h'
        self.nonlinearity = nn.Tanh()

    def forward(self, x):                       # x: (T, 1, input_dim)
        H, T = [], x.size(0)
        h = torch.zeros(1, self.hidden_dim)
        h_prime = torch.zeros(1, self.hidden_dim)
        for t in range(T):
            v = torch.cat((x[t], h), dim=1)                    # forward reads x[t]
            v_prime = torch.cat((x[T - t - 1], h_prime), dim=1)  # backward reads x[T-1-t]
            h = self.nonlinearity(self.fc_h(v))
            h_prime = self.nonlinearity(self.fc_h_prime(v_prime))
            H.append([h, h_prime])
        z_0 = torch.cat((H[0][0], H[-1][1]), dim=1)   # both states at first position
        z_T = torch.cat((H[-1][0], H[0][1]), dim=1)   # both states at last position
        return [(self.fc_y(z_0) + self.fc_y(z_T)) / 2], H
```

Read the indexing carefully. At loop index `t` the forward state has read $x_1,\dots,x_{t+1}$, while the backward state has read $x_T,\dots,x_{T-t}$. So `H[i][0]` is the forward state at position $i+1$, but `H[i][1]` is the backward state at position $T-i$:

- `z_0` pairs the forward state at position 1 (`H[0][0]`, has seen only $x_1$) with the backward state at position 1 (`H[-1][1]`, has seen the **whole** sequence);
- `z_T` pairs the forward state at position $T$ (`H[-1][0]`, the whole sequence) with the backward state at position $T$ (`H[0][1]`, only $x_T$).

Each of the two positions therefore contains one full-sequence summary. The surname classifier skips the outputs in between and averages the two predictions.

**Parameters** for the surname task ($d=52$, $H=64$, $K=18$): `fc_h` and `fc_h_prime` are each `nn.Linear(116, 64)` with $116\cdot 64+64 = 7{,}488$, and `fc_y` is `nn.Linear(128, 18)` with $128\cdot 18 + 18 = 2{,}322$:
$$
7{,}488 + 7{,}488 + 2{,}322 = 17{,}298 \quad\text{(vs. } 9{,}594 \text{ for the unidirectional } \texttt{BasicRNN}).
$$

**Results** (mean per-language accuracy, 50 epochs, from the lecture's saved checkpoints):

| Model | Training | Validation |
| --- | --- | --- |
| `BasicRNN` (unidirectional) | 49.33% | 46.00% |
| `BidirectionalRNN` | 55.87% | 55.27% |

Reading a surname from both ends helps: the forward summary ends on the last letters (a suffix like *-ski* or *-ov*), the backward summary ends on the first letters, so both ends of the name reach the prediction through a short path.

## Stacked RNN from scratch

```python
class StackedRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc_h_1 = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.fc_y_1 = nn.Linear(input_dim + hidden_dim, output_dim)
        self.fc_h_2 = nn.Linear(output_dim + hidden_dim, hidden_dim)   # layer 2 reads y^(1)
        self.fc_y_2 = nn.Linear(output_dim + hidden_dim, output_dim)
        self.nonlinearity = nn.Tanh()

    def forward(self, x):                       # x: (T, 1, input_dim)
        Y, H = [], []
        h_1 = torch.zeros(1, self.hidden_dim)
        h_2 = torch.zeros(1, self.hidden_dim)   # separate state per layer
        for t in range(x.size(0)):
            v = torch.cat((x[t], h_1), dim=1)
            h_1 = self.nonlinearity(self.fc_h_1(v))
            y_1 = self.fc_y_1(v)
            z = torch.cat((self.nonlinearity(y_1), h_2), dim=1)
            h_2 = self.nonlinearity(self.fc_h_2(z))
            Y.append(self.fc_y_2(z))
            H.append(h_2)
        return torch.stack(Y, dim=0), torch.stack(H, dim=0)
```

Only the second layer's outputs and states are returned; the prediction is again `outputs[-1]`. **Parameters**: $7{,}488 + 2{,}106$ for layer 1, and `nn.Linear(18 + 64, 64)` $= 5{,}312$ plus `nn.Linear(82, 18)` $= 1{,}494$ for layer 2, **16,400** in total.

> [!note] The lecture's stacked RNN is not exactly `nn.RNN(num_layers=2)`
> The from-scratch model feeds $\tanh(y^{(1)}_t)$, a $K$-dimensional vector, into layer 2. PyTorch's `nn.RNN` feeds layer 1's **hidden state** $h^{(1)}_t$ (dimension $H$, or $2H$ if bidirectional) into layer 2, and has no per-layer output head. The idea, a recurrence on top of a recurrence, is the same.

## Both at once in PyTorch

The lecture combines the two ideas with one constructor call, `bidirectional=True` and `num_layers=2`, and adds a head on $D\cdot H$ features:

```python
class ElmanRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional):
        super().__init__()
        self.n_layers, self.hidden_dim = n_layers, hidden_dim
        self.bidirectional = bidirectional
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, nonlinearity='tanh',
                          bidirectional=bidirectional, batch_first=False)
        self.D = 2 if bidirectional else 1
        self.fc = nn.Linear(self.D * hidden_dim, output_dim)

    def forward(self, x):                       # x: (T, N, input_dim)
        h0 = torch.zeros(self.D * self.n_layers, x.size(1), self.hidden_dim)
        outputs, hiddens = self.rnn(x, h0)      # outputs: (T, N, D*hidden_dim)
        if self.bidirectional:
            y = (self.fc(outputs[-1]) + self.fc(outputs[0])) / 2
        else:
            y = self.fc(outputs[-1])
        return [y], hiddens

model = ElmanRNN(52, 64, 18, n_layers=2, bidirectional=True)
```

For a 9-letter name, `x` is `(9, 1, 52)`:

| Tensor | Shape | Layout |
| --- | --- | --- |
| `h0`, `hiddens` (`h_n`) | `(4, 1, 64)` | $D\cdot L = 2\cdot 2$ rows: layer 1 fwd, layer 1 bwd, layer 2 fwd, layer 2 bwd |
| `outputs` | `(9, 1, 128)` | top layer at every step; `[..., :64]` forward, `[..., 64:]` backward |
| `outputs[-1]`, `outputs[0]` | `(1, 128)` | input to `self.fc`, giving `(1, 18)` logits |

The rows of `h_n` are ordered layer by layer, forward (direction 0) before backward (direction 1), so `h_n.view(num_layers, num_directions, batch, hidden_size)` separates them, just as the [`nn.RNN` docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html) split `output` with `output.view(seq_len, batch, num_directions, hidden_size)`. So `h_n[-2]` is the top layer's forward state and `h_n[-1]` the top layer's backward state.

### Which output summarizes the whole sequence?

For a bidirectional RNN, `outputs[-1]` is **not** a summary of the whole sequence: its forward half has read everything, but its backward half has read only $x_T$. The full-sequence summaries are

$$
\underbrace{\texttt{outputs[-1, :, :H]}}_{\text{forward, } t = T} \qquad\text{and}\qquad \underbrace{\texttt{outputs[0, :, H:]}}_{\text{backward, } t = 1}.
$$

The lecture covers both by averaging the head's predictions at `outputs[-1]` and `outputs[0]`, which mirrors `z_T` and `z_0` in the from-scratch model. Another common choice is to concatenate the top layer's two final states, `torch.cat((h_n[-2], h_n[-1]), dim=1)`, shape `(N, 2H)`.

### Counting parameters

Each layer and direction has its own $W_{ih}$, $W_{hh}$, $b_{ih}$, $b_{hh}$. The first layer reads $d$ features; every later layer reads the concatenated $D\cdot H$ outputs of the layer below:
$$
\text{params} = D\cdot H(d + H + 2) \;+\; (L-1)\cdot D\cdot H(DH + H + 2).
$$
For `nn.RNN(52, 64, num_layers=2, bidirectional=True)`:
$$
\begin{aligned}
\text{layer 1: } & 2\cdot 64\cdot(52 + 64 + 2) = 2\cdot 7{,}552 = 15{,}104\\
\text{layer 2: } & 2\cdot 64\cdot(128 + 64 + 2) = 2\cdot 12{,}416 = 24{,}832\\
\text{total: } & 39{,}936, \quad\text{plus the head } \texttt{nn.Linear(128, 18)}: 2{,}322 \;\Rightarrow\; 42{,}258 .
\end{aligned}
$$
The LSTM and GRU versions of the same model multiply the RNN part by 4 and 3: [[5-deep-networks/lstm-and-gru#counting-parameters-of-rnn-gru-and-lstm|counting parameters of RNN, GRU and LSTM]].

`nn.RNN` also takes `dropout=p`, which applies [[5-deep-networks/dropout|dropout]] to the outputs of every layer except the last; it only does anything when `num_layers > 1`.

> [!trap] Common mistakes
> - **Bidirectional for a causal task.** Predicting the next character as someone types, or a live sensor value, cannot use the future: the backward direction would need inputs that have not arrived. Bidirectional models need the whole sequence before the first output.
> - **Head of the wrong width.** With `bidirectional=True` the output has $2H$ features; `nn.Linear(H, K)` fails with a shape error.
> - **Treating `outputs[-1]` as the full summary** in a bidirectional model (see above).
> - **Mixing up `h_n`'s rows.** They are ordered layer by layer, forward before backward: `(L, D, N, H)` after `.view`. `h_n[-1]` is the top layer's **backward** state, not the forward one.
> - **Counting layer 2 with input size $d$.** Layers above the first read $D\cdot H$ features.

## Check yourself

> [!question] 1. Shapes and parameters
> `rnn = nn.RNN(10, 20, num_layers=2, bidirectional=True)` and `x` has shape `(5, 3, 10)`. Give the shapes of `output` and `h_n`, and the number of parameters.

> [!solution]- Answer
> `output`: `(5, 3, 40)`; `h_n`: `(4, 3, 20)`. Parameters: layer 1, $2\cdot 20\cdot(10+20+2) = 1{,}280$; layer 2, $2\cdot 20\cdot(40 + 20 + 2) = 2{,}480$; total $3{,}760$.

> [!question] 2. Which state has seen what?
> In the from-scratch `BidirectionalRNN` with $T = 5$, which inputs has `H[1][1]` read?

> [!solution]- Answer
> At loop index 1 the backward state has read `x[4]` and `x[3]`, i.e. $x_5$ and $x_4$; it is the backward state at position $T - 1 = 4$.

> [!question] 3. Design
> A model must flag, after each new heartbeat in a live ECG stream, whether that beat is abnormal. Unidirectional or bidirectional? Stacked or not?

> [!solution]- Answer
> Unidirectional: each decision must be made when the beat arrives, without future beats. Stacking is allowed (it adds depth at each step, not future context) and may help if one layer is not expressive enough.

## PyTorch docs

- [`torch.nn.RNN`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html) — `num_layers`, `bidirectional`, `dropout`, and the `h_n` layout.
- [`torch.nn.LSTM`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) and [`torch.nn.GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) — the same two options for gated cells.
- [`torch.cat`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html) — concatenating forward and backward states.
- [`torch.Tensor.view`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html) — reshaping `h_n` to `(num_layers, num_directions, N, H)`.
- [`torch.nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) — the task head on $D\cdot H$ features.

Related: [[5-deep-networks/recurrent-neural-networks]] · [[5-deep-networks/lstm-and-gru]] · [[5-deep-networks/multilayer-perceptrons]] · [[5-deep-networks/dropout]]
