---
title: "LSTM and GRU"
description: "Gated recurrent cells: the LSTM's input, forget, cell and output gates and its additive cell-state update; the GRU's reset and update gates and interpolated hidden state; how gating keeps gradients alive over long sequences; nn.LSTM and nn.GRU in PyTorch; and the parameter counts of RNN, GRU and LSTM layers worked by hand."
tags: [topic, deep-nets]
aliases: ["LSTM", "long short-term memory", "GRU", "gated recurrent unit", "cell state", "forget gate", "input gate", "output gate", "update gate", "reset gate", "nn.LSTM", "nn.GRU", "gating"]
---

*Lecture 17 · [[5-deep-networks/index|Deep networks]] · builds on [[5-deep-networks/recurrent-neural-networks|recurrent neural networks]] and [[5-deep-networks/bidirectional-and-stacked-rnns|bidirectional and stacked RNNs]]*

> [!abstract] In one breath
> An LSTM or GRU is a drop-in replacement for the plain RNN cell $f(x_t, h_{t-1})$. Instead of overwriting the state with $\tanh(Wv_t)$ at every step, it computes **gates**, sigmoid vectors with entries in $[0,1]$, that decide elementwise how much of the old state to keep and how much new content to write. The **LSTM** keeps a separate **cell state** $c_t = f_t\odot c_{t-1} + i_t\odot g_t$ with 4 gate vectors; the **GRU** has no cell state and 3 gate vectors, $h_t = (1-z_t)\odot n_t + z_t\odot h_{t-1}$. The additive update lets information, and gradients, survive many time steps. In PyTorch: [`nn.LSTM`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) and [`nn.GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html), same interface as `nn.RNN`.

## What it is

A plain RNN replaces its whole state at every step, $h_t = \tanh(W_{ih}x_t + W_{hh}h_{t-1})$. Two problems follow. Information from early in the sequence is squashed again at every step, and during [[5-deep-networks/recurrent-neural-networks#backpropagation-through-time|backpropagation through time]] the gradient is multiplied by $\tanh'(\cdot)\,W_{hh}$ once per step, so it tends to vanish (or explode) over long sequences. Gated cells fix both by making "keep the old state" the easy default.

The building block is a **gate**: a vector $\sigma(W x_t + U h_{t-1}) \in [0,1]^H$ that multiplies another vector elementwise ($\odot$). An entry near 1 lets that component through; an entry near 0 blocks it. Gates are learned, so the network decides from the current input and state what to remember, forget and output.

## What you should know

- The four LSTM activations ($i_t, f_t, g_t, o_t$), which nonlinearity each uses and its range, and the two update equations for $c_t$ and $h_t$.
- What each gate *does*: forget gate keeps/erases the cell, input gate scales what is written, cell gate is the content, output gate scales what is exposed as $h_t$.
- The GRU's reset gate $r_t$, update gate $z_t$, new gate $n_t$, and $h_t = (1-z_t)\odot n_t + z_t\odot h_{t-1}$; $z_t\to 1$ keeps the old state.
- Why gating helps: the cell (or state) update is **additive**, so with $f_t\approx 1$ the gradient flows back through $c_t\to c_{t-1}$ almost unchanged, like the shortcut in a [[5-deep-networks/residual-connections|residual connection]].
- LSTM vs GRU: GRU has no cell state, 3 instead of 4 gate blocks, trains faster and can do better on small datasets; LSTM has more capacity.
- `nn.LSTM` takes and returns a **tuple** `(h, c)`: `output, (h_n, c_n) = lstm(x, (h0, c0))`; `nn.GRU` works like `nn.RNN`.
- Parameter counts: per layer and direction $G\cdot H(d + H + 2)$ with $G = 1$ (RNN), $3$ (GRU), $4$ (LSTM).

## Long short-term memory (LSTM)

At step $t$ the LSTM cell receives the input $x_t$, the previous hidden state $h_{t-1}$ and the previous **cell state** $c_{t-1}$, and computes four activations (the lecture drops the biases; PyTorch adds $b_{i\cdot}$ and $b_{h\cdot}$ to each line):
$$
\begin{aligned}
i_t &= \sigma(W_{ii}x_t + W_{hi}h_{t-1}) \in [0,1] && \text{input gate}\\
f_t &= \sigma(W_{if}x_t + W_{hf}h_{t-1}) \in [0,1] && \text{forget gate}\\
g_t &= \tanh(W_{ig}x_t + W_{hg}h_{t-1}) \in [-1,1] && \text{cell gate (candidate content)}\\
o_t &= \sigma(W_{io}x_t + W_{ho}h_{t-1}) \in [0,1] && \text{output gate}
\end{aligned}
$$
and then updates the two states:
$$
\boxed{\;c_t = f_t\odot c_{t-1} + i_t\odot g_t, \qquad h_t = o_t\odot\tanh(c_t)\;}
$$

![[l17-lstm-cell.png|An LSTM cell: three sigmoid gates and one tanh layer acting on the cell state line along the top]]

The horizontal line along the top of the figure is the cell state. It only meets an elementwise product with $f_t$ (keep or erase) and a sum with $i_t\odot g_t$ (write), so it can carry information across many steps. The orange boxes are the four learned layers, each with its own pair of weight matrices; the yellow circles are elementwise operations.

| Activation | Nonlinearity | Range | Role |
| --- | --- | --- | --- |
| $f_t$ forget | $\sigma$ | $[0,1]$ | entries near 1 **keep** $c_{t-1}$, near 0 **erase** it |
| $i_t$ input | $\sigma$ | $[0,1]$ | how much of the candidate to write, per entry |
| $g_t$ cell | $\tanh$ | $[-1,1]$ | the candidate itself; its sign lets the write **increase or decrease** the cell |
| $o_t$ output | $\sigma$ | $[0,1]$ | how much of $\tanh(c_t)$ to expose as $h_t$ |

The same eight weight matrices are used at every time step. A stacked LSTM passes each layer's hidden state up to the next layer at the same time step; an $L$-layer LSTM over $T$ steps produces $T\times L$ hidden state vectors (and as many cell states).

### Worked example: keep, then overwrite

Scalar cell, $c_{t-1} = 1.0$.

*Keep.* Suppose the gates come out as $f_t = 0.9$, $i_t = 0.2$, $g_t = 0.5$, $o_t = 0.8$:
$$
c_t = 0.9\cdot 1.0 + 0.2\cdot 0.5 = 1.0, \qquad h_t = 0.8\cdot\tanh(1.0) = 0.8\cdot 0.7616 = 0.6093 .
$$
The cell barely changes: the old memory is kept and a little is added.

*Erase and overwrite.* Now $f_t = 0$, $i_t = 1$, $g_t = -0.5$, same $o_t$:
$$
c_t = 0\cdot 1.0 + 1\cdot(-0.5) = -0.5, \qquad h_t = 0.8\cdot\tanh(-0.5) = -0.3697 .
$$
The old memory is gone and replaced by the new content.

> [!intuition] Why the cell state helps gradients
> Along the direct path $c_{t-1}\to c_t$, the local derivative is just $\partial c_t/\partial c_{t-1} = f_t$ (elementwise). When the forget gate is near 1, the gradient passes back through many steps almost unchanged, with no $\tanh'$ and no weight matrix in the way. It is the same trick as the "+ x" shortcut of a [[5-deep-networks/residual-connections|residual block]], along the time axis instead of depth. A plain RNN has no such path.

For more on the gates, the lecture recommends [Colah's "Understanding LSTM Networks"](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).

### `nn.LSTM` in PyTorch

`nn.LSTM` takes the same arguments as `nn.RNN` except `nonlinearity` (`input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional`, plus `proj_size`) but has **two** states, passed in and returned as a tuple:

```python
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional):
        super().__init__()
        self.n_layers, self.hidden_dim = n_layers, hidden_dim
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers,
                           bidirectional=bidirectional, batch_first=False)
        self.D = 2 if bidirectional else 1
        self.fc = nn.Linear(self.D * hidden_dim, output_dim)

    def forward(self, x):                         # x: (T, N, input_dim)
        shape = (self.D * self.n_layers, x.size(1), self.hidden_dim)
        h0, c0 = torch.zeros(shape), torch.zeros(shape)
        outputs, (hiddens, cell_states) = self.rnn(x, (h0, c0))   # tuple in, tuple out
        if self.bidirectional:
            y = (self.fc(outputs[-1]) + self.fc(outputs[0])) / 2
        else:
            y = self.fc(outputs[-1])
        return [y], hiddens
```

`c_n` has the same shape as `h_n`, `(D·num_layers, N, H)`, and `outputs` is the top layer's $h_t$ at every step, `(T, N, D·H)`, exactly as for [[5-deep-networks/bidirectional-and-stacked-rnns#both-at-once-in-pytorch|`nn.RNN`]]. The cell states of the intermediate steps are not returned.

## Gated recurrent unit (GRU)

The GRU keeps the LSTM's insight with fewer parts: no separate cell state and three gate vectors instead of four. In the lecture's notation ($U_r$, $U_z$ and $W_h$ act on the previous state; $W_r$, $W_z$ and $W_n$ on the input):
$$
\begin{aligned}
r_t &= \sigma(U_r h_{t-1} + W_r x_t) && \text{reset gate}\\
z_t &= \sigma(U_z h_{t-1} + W_z x_t) && \text{update gate}\\
n_t &= \tanh\big(W_n x_t + r_t\odot(W_h h_{t-1})\big) && \text{new gate (candidate state)}\\
h_t &= (1 - z_t)\odot n_t + z_t\odot h_{t-1}
\end{aligned}
$$

![[l17-gru-overview.png|A GRU cell; h′_t in the figure is the new gate n_t]]

- The **update gate** $z_t$ interpolates between the old state and the candidate: $z_t\to 1$ keeps $h_{t-1}$, $z_t\to 0$ replaces it with $n_t$. It plays the role of the LSTM's forget and input gates at once (they are tied: keep $z$, write $1-z$).
- The **reset gate** $r_t$ controls how much of $h_{t-1}$ enters the candidate; $r_t\to 0$ lets the cell compute $n_t$ as if starting fresh.

*Example.* $h_{t-1} = -0.2$, $z_t = 0.75$, $n_t = 0.4$: $h_t = 0.25\cdot 0.4 + 0.75\cdot(-0.2) = 0.10 - 0.15 = -0.05$. Three quarters of the old state survive.

> [!warning] Conventions differ
> The equation above is PyTorch's (and the lecture's): $z_t$ multiplies the **old** state. Some papers write the GRU with $z$ and $1 - z$ swapped. The mechanism is the same; read the definition before answering "what does $z \to 1$ do?".

`nn.GRU` is used exactly like `nn.RNN`, one state tensor in and out:

```python
self.rnn = nn.GRU(input_dim, hidden_dim, n_layers, bidirectional=bidirectional, batch_first=False)
...
h0 = torch.zeros(self.D * self.n_layers, x.size(1), self.hidden_dim)
outputs, hiddens = self.rnn(x, h0)      # no cell state
```

## RNN vs GRU vs LSTM

| | Plain RNN (Elman) | GRU | LSTM |
| --- | --- | --- | --- |
| States carried | $h_t$ | $h_t$ | $h_t$ and $c_t$ |
| Gate / layer blocks per cell | 1 ($\tanh$) | 3 ($r, z, n$) | 4 ($i, f, g, o$) |
| State update | overwrite: $\tanh(\cdot)$ | interpolate old and new | add to the cell, then expose |
| Long-range memory | poor (vanishing gradients) | good | good |
| PyTorch call | `out, h_n = rnn(x, h0)` | `out, h_n = gru(x, h0)` | `out, (h_n, c_n) = lstm(x, (h0, c0))` |
| When (lecture) | baseline | faster to train, can be better on smaller datasets | more representational capacity; wins with enough time, compute and data |

In the lecture, the LSTM and GRU versions of the surname classifier (2 layers, bidirectional, hidden size 64) are trained for 100 epochs, twice as long as the plain RNNs' saved 50-epoch checkpoints; the notebook's saved outputs do not include their accuracies, so run those cells to compare.

## Counting parameters of RNN, GRU and LSTM

PyTorch stores each layer's gate weights stacked in one matrix: `weight_ih_l0` has shape $(G\cdot H, d)$ and `weight_hh_l0` has shape $(G\cdot H, H)$, with $G$ blocks in a fixed order (LSTM: $W_{ii}\,\vert\,W_{if}\,\vert\,W_{ig}\,\vert\,W_{io}$; GRU: $W_{ir}\,\vert\,W_{iz}\,\vert\,W_{in}$), plus two bias vectors of length $G\cdot H$. So one layer, one direction has
$$
\boxed{\;G\cdot H\,(d + H + 2)\;}\qquad G = 1 \text{ (RNN)},\; 3 \text{ (GRU)},\; 4 \text{ (LSTM)}.
$$
With $D$ directions and $L$ layers, the first layer reads $d$ features and every later layer reads $D\cdot H$:
$$
\text{params} = D\cdot G\cdot H\,(d + H + 2) + (L - 1)\cdot D\cdot G\cdot H\,(DH + H + 2).
$$

Worked for the surname models ($d = 52$, $H = 64$, head `nn.Linear(128, 18)` $= 2{,}322$):

| Layer configuration | RNN ($G=1$) | GRU ($G=3$) | LSTM ($G=4$) |
| --- | --- | --- | --- |
| 1 layer, 1 direction: $G\cdot 64\cdot 118$ | 7,552 | 22,656 | 30,208 |
| 2 layers, bidirectional: $G\cdot(15{,}104 + 24{,}832)$ | 39,936 | 119,808 | 159,744 |
| + head (lecture models) | 42,258 | 122,130 | 162,066 |

<!-- fig:deepc-param-counts -->
<figure class="cs-fig"><svg role="img" aria-label="Horizontal bars: parameter counts from 9,594 (BasicRNN) to 162,066 (2-layer bidirectional LSTM)" style="max-width:683px;width:100%;height:auto" stroke-linejoin="round" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 512.338828 206.639063" xmlns="http://www.w3.org/2000/svg" version="1.1"><defs></defs><g id="deepc-param-counts-figure_1"><g id="deepc-param-counts-patch_1"><path d="M 0 206.639063  L 512.338828 206.639063  L 512.338828 0  L 0 0  L 0 206.639063  z " style="fill: none"/></g><g id="deepc-param-counts-axes_1"><g id="deepc-param-counts-patch_2"><path d="M 192.770312 166.437344  L 501.795312 166.437344  L 501.795312 7.2  L 192.770312 7.2  L 192.770312 166.437344  z " style="fill: none"/></g><g id="deepc-param-counts-patch_3"><path d="M 192.770312 29.948192  L 208.374449 29.948192  L 208.374449 14.438061  L 192.770312 14.438061  z " clip-path="url(#deepc-param-counts-p233b9bb4e3)" style="fill: var(--muted)"/></g><g id="deepc-param-counts-patch_4"><path d="M 192.770312 55.79841  L 219.444049 55.79841  L 219.444049 40.288279  L 192.770312 40.288279  z " clip-path="url(#deepc-param-counts-p233b9bb4e3)" style="fill: var(--muted)"/></g><g id="deepc-param-counts-patch_5"><path d="M 192.770312 81.648628  L 220.904599 81.648628  L 220.904599 66.138497  L 192.770312 66.138497  z " clip-path="url(#deepc-param-counts-p233b9bb4e3)" style="fill: var(--muted)"/></g><g id="deepc-param-counts-patch_6"><path d="M 192.770312 107.498846  L 261.500725 107.498846  L 261.500725 91.988716  L 192.770312 91.988716  z " clip-path="url(#deepc-param-counts-p233b9bb4e3)" style="fill: var(--accent)"/></g><g id="deepc-param-counts-patch_7"><path d="M 192.770312 133.349065  L 391.40833 133.349065  L 391.40833 117.838934  L 192.770312 117.838934  z " clip-path="url(#deepc-param-counts-p233b9bb4e3)" style="fill: var(--accent2)"/></g><g id="deepc-param-counts-patch_8"><path d="M 192.770312 159.199283  L 456.362132 159.199283  L 456.362132 143.689152  L 192.770312 143.689152  z " clip-path="url(#deepc-param-counts-p233b9bb4e3)" style="fill: var(--hi)"/></g><g id="deepc-param-counts-matplotlib.axis_1"><g id="deepc-param-counts-xtick_1"><g id="deepc-param-counts-line2d_1"><defs><path id="deepc-param-counts-m96f9c9ab67" d="M 0 0  L 0 3.5  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#deepc-param-counts-m96f9c9ab67" x="192.770312" y="166.437344" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepc-param-counts-text_1"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="192.770312" y="181.794766" transform="rotate(-0 192.770312 181.794766)">0</text></g></g><g id="deepc-param-counts-xtick_2"><g id="deepc-param-counts-line2d_2"><g><use xlink:href="#deepc-param-counts-m96f9c9ab67" x="274.092681" y="166.437344" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepc-param-counts-text_2"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="274.092681" y="181.795625" transform="rotate(-0 274.092681 181.795625)">50k</text></g></g><g id="deepc-param-counts-xtick_3"><g id="deepc-param-counts-line2d_3"><g><use xlink:href="#deepc-param-counts-m96f9c9ab67" x="355.415049" y="166.437344" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepc-param-counts-text_3"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="355.415049" y="181.795625" transform="rotate(-0 355.415049 181.795625)">100k</text></g></g><g id="deepc-param-counts-xtick_4"><g id="deepc-param-counts-line2d_4"><g><use xlink:href="#deepc-param-counts-m96f9c9ab67" x="436.737418" y="166.437344" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepc-param-counts-text_4"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="436.737418" y="181.795625" transform="rotate(-0 436.737418 181.795625)">150k</text></g></g><g id="deepc-param-counts-text_5"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="347.282812" y="196.796484" transform="rotate(-0 347.282812 196.796484)">learnable parameters (including the final nn.Linear head)</text></g></g><g id="deepc-param-counts-matplotlib.axis_2"><g id="deepc-param-counts-ytick_1"><g id="deepc-param-counts-line2d_5"/><g id="deepc-param-counts-text_6"><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="189.270312" y="25.992345" transform="rotate(-0 189.270312 25.992345)">BasicRNN (from scratch)</text></g></g><g id="deepc-param-counts-ytick_2"><g id="deepc-param-counts-line2d_6"/><g id="deepc-param-counts-text_7"><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="189.270312" y="51.842563" transform="rotate(-0 189.270312 51.842563)">StackedRNN, 2 layers (from scratch)</text></g></g><g id="deepc-param-counts-ytick_3"><g id="deepc-param-counts-line2d_7"/><g id="deepc-param-counts-text_8"><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="189.270312" y="77.692782" transform="rotate(-0 189.270312 77.692782)">BidirectionalRNN (from scratch)</text></g></g><g id="deepc-param-counts-ytick_4"><g id="deepc-param-counts-line2d_8"/><g id="deepc-param-counts-text_9"><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="189.270312" y="103.543" transform="rotate(-0 189.270312 103.543)">nn.RNN, 2 layers, bidirectional</text></g></g><g id="deepc-param-counts-ytick_5"><g id="deepc-param-counts-line2d_9"/><g id="deepc-param-counts-text_10"><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="189.270312" y="129.393218" transform="rotate(-0 189.270312 129.393218)">nn.GRU, 2 layers, bidirectional</text></g></g><g id="deepc-param-counts-ytick_6"><g id="deepc-param-counts-line2d_10"/><g id="deepc-param-counts-text_11"><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="189.270312" y="155.243436" transform="rotate(-0 189.270312 155.243436)">nn.LSTM, 2 layers, bidirectional</text></g></g></g><g id="deepc-param-counts-patch_9"><path d="M 192.770312 166.437344  L 501.795312 166.437344  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g id="deepc-param-counts-text_12"><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="212.440567" y="24.790783" transform="rotate(-0 212.440567 24.790783)">9,594</text></g><g id="deepc-param-counts-text_13"><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="223.510168" y="50.641001" transform="rotate(-0 223.510168 50.641001)">16,400</text></g><g id="deepc-param-counts-text_14"><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="224.970717" y="76.491219" transform="rotate(-0 224.970717 76.491219)">17,298</text></g><g id="deepc-param-counts-text_15"><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="265.566844" y="102.341437" transform="rotate(-0 265.566844 102.341437)">42,258</text></g><g id="deepc-param-counts-text_16"><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="395.474448" y="128.191655" transform="rotate(-0 395.474448 128.191655)">122,130</text></g><g id="deepc-param-counts-text_17"><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="460.42825" y="154.041873" transform="rotate(-0 460.42825 154.041873)">162,066</text></g></g></g><defs><clipPath id="deepc-param-counts-p233b9bb4e3"><rect x="192.770312" y="7.2" width="309.025" height="159.237344"/></clipPath></defs></svg> <figcaption><b>Parameters of the Lecture 17 surname models</b> (52-letter one-hot input, hidden size 64, 18 classes), counted by hand with the formulas on this page. At the same hidden size a GRU layer has 3× and an LSTM layer 4× the weights of a plain RNN layer, because they compute 3 and 4 gate vectors.</figcaption></figure>

> [!trap] Common mistakes
> - **Forgetting the tuple.** `nn.LSTM` expects `(h0, c0)` and returns `output, (h_n, c_n)`; `outputs, hiddens = lstm(x, h0)` fails, and unpacking the result as two tensors gives you a tuple where you expected `h_n`.
> - **Mixing up the gates' nonlinearities.** $i, f, o$ (LSTM) and $r, z$ (GRU) use the **sigmoid**, because they are fractions in $[0,1]$; the content $g_t$ / $n_t$ uses **tanh**, because it must be able to be negative.
> - **$h_t$ vs $c_t$.** The LSTM's output at each step is $h_t = o_t\odot\tanh(c_t)$, not $c_t$. Only $h_t$ goes to the next layer and to the head.
> - **"Forget gate = 1 means forget."** It is the opposite: $f_t = 1$ keeps the cell, $f_t = 0$ erases it.
> - **Counting one bias per gate.** PyTorch has two bias vectors ($b_{ih}$ and $b_{hh}$), hence the $+2$ in $d + H + 2$.
> - **Assuming gates remove the need to think about sequence length.** They make long-range learning possible, not free; compute still grows with $T$.

## Check yourself

> [!question] 1. One LSTM step
> $c_{t-1} = 0.4$, and the gates evaluate to $f_t = 0.5$, $i_t = 1$, $g_t = 0.6$, $o_t = 1$. Compute $c_t$ and $h_t$.

> [!solution]- Answer
> $c_t = 0.5\cdot 0.4 + 1\cdot 0.6 = 0.8$; $h_t = 1\cdot\tanh(0.8) = 0.6640$.

> [!question] 2. Parameters
> How many parameters do `nn.LSTM(10, 20)` and `nn.GRU(10, 20)` have? What are the shapes of the LSTM's `weight_ih_l0` and `weight_hh_l0`?

> [!solution]- Answer
> LSTM: $4\cdot 20\cdot(10 + 20 + 2) = 2{,}560$. GRU: $3\cdot 20\cdot 32 = 1{,}920$. `weight_ih_l0`: `(80, 10)`; `weight_hh_l0`: `(80, 20)` (four gate blocks of 20 rows stacked).

> [!question] 3. Gates
> In the GRU, what are $h_t$ when $z_t = 1$ and when $z_t = 0$? Which LSTM gate setting corresponds to "keep everything"?

> [!solution]- Answer
> $z_t = 1$: $h_t = h_{t-1}$ (state copied unchanged). $z_t = 0$: $h_t = n_t$ (state replaced by the candidate). In the LSTM, $f_t = 1$ with $i_t = 0$ gives $c_t = c_{t-1}$.

## PyTorch docs

- [`torch.nn.LSTM`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) — the gate equations with biases, the `(h_0, c_0)` tuple, shapes, `weight_ih_l[k]` layout, `proj_size`.
- [`torch.nn.GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) — the reset/update/new gate equations and weight layout.
- [`torch.nn.LSTMCell`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html) and [`torch.nn.GRUCell`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html) — one time step, for your own loop.
- [`torch.nn.RNN`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html) — the ungated baseline with the same interface.
- [`torch.sigmoid`](https://docs.pytorch.org/docs/stable/generated/torch.sigmoid.html) and [`torch.tanh`](https://docs.pytorch.org/docs/stable/generated/torch.tanh.html) — the two gate nonlinearities ([[5-deep-networks/activation-functions|activation functions]]).
- Further reading from the lecture: [Chung et al., "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling"](https://arxiv.org/abs/1412.3555) and [Colah's "Understanding LSTM Networks"](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).

Related: [[5-deep-networks/recurrent-neural-networks]] · [[5-deep-networks/bidirectional-and-stacked-rnns]] · [[5-deep-networks/residual-connections]] · [[5-deep-networks/activation-functions]]
