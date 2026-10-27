---
title: "Recurrent neural networks"
description: "Networks with memory for sequence data: the recurrence h_t = f(x_t, h_{t-1}) with the same weights at every step, a basic RNN built by hand, one-hot encoding of letters, the Lecture 17 surname classifier, backpropagation through time, and the input/output/hidden shapes of nn.RNN."
tags: [topic, deep-nets]
aliases: ["RNN", "RNNs", "recurrent neural network", "hidden state", "Elman RNN", "nn.RNN", "one-hot encoding", "backpropagation through time", "BPTT"]
---

*Lecture 17 · [[5-deep-networks/index|Deep networks]] · builds on [[5-deep-networks/multilayer-perceptrons|MLPs]], [[2-calculus-and-optimization/backpropagation|backpropagation]] and [[3-linear-models/multiclass-logistic-regression|softmax + cross-entropy]] · next: [[5-deep-networks/bidirectional-and-stacked-rnns|bidirectional and stacked RNNs]], [[5-deep-networks/lstm-and-gru|LSTM and GRU]]*

> [!abstract] In one breath
> MLPs and CNNs are **feed-forward**: each input is processed in one pass and nothing is remembered for the next one. A **recurrent** network reads a sequence $x_1,\dots,x_T$ one step at a time and carries a **hidden state** $h_t = f(x_t, h_{t-1})$ from step to step, using the **same weights at every step**. A task head (an `nn.Linear`) turns a hidden state into logits or a regression value. In PyTorch, [`nn.RNN`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html) runs the whole loop for you and returns `output` (the top layer at every step) and `h_n` (every layer at the last step).

## What it is

Sequence data has an order that matters: the letters of a name, the words of a sentence, sensor readings at a traffic intersection over time. Write one example as
$$
x = \{x_1, x_2, \ldots, x_T\} \in \mathbb{R}^{T\times d}, \qquad x_t \in \mathbb{R}^d .
$$
At each time step the model $f$ takes the current input $x_t$ and the **hidden state** of the previous step $h_{t-1}\in\mathbb{R}^H$ and produces the new hidden state:
$$
h_t = f(x_t, h_{t-1}).
$$
The computation is a **recurrence** (each output depends on all earlier ones), hence the name. The hidden state $h_t$ is a **latent representation** of everything read so far. A fully connected layer on top of $h_t$ solves the task: estimate a vehicle's speed (regression) or classify the vehicle type (classification) from the sensor readings, for example.

![[l17-rnn-unfolded.png|An RNN, folded (left) and unfolded in time (right)]]

Left: the compact picture, a hidden unit with a loop back into itself. Right: the same network **unfolded** over time. Every green box is the *same* function with the *same* weights; only its inputs change. Unfolding is how the network is actually computed and how gradients flow through it.

## What you should know

- The recurrence $h_t = f(x_t, h_{t-1})$, the meaning of the hidden state, and why a feed-forward net cannot do this (no memory between inputs).
- **Weight sharing over time**: one set of matrices, used at every step, so the parameter count does not depend on the sequence length $T$.
- The basic RNN $h_t = \tanh(W_{h,x}x_t + W_{h,h}h_{t-1})$, the shapes of its matrices, and the concatenation trick $v_t = x_t \oplus h_{t-1}$.
- How to count parameters: $H(d+H)$ weights for the hidden update (+ $H$ bias per `nn.Linear`), and PyTorch's $H\cdot d + H\cdot H + 2H$ per `nn.RNN` layer.
- **One-hot encoding** of symbols (52 letters → vectors in $\{0,1\}^{52}$), and why a name becomes a `(len(name), 1, 52)` tensor.
- **Many-to-one** use: classify the whole sequence from the **last** output; many-to-many: one prediction per step.
- Backpropagation **through time**: autograd unrolls the loop; gradients from every step add up in the shared weights; long sequences make them vanish or explode (the reason for [[5-deep-networks/lstm-and-gru|LSTM and GRU]]).
- `nn.RNN` shapes: input `(L, N, H_in)`, `h_0`/`h_n` `(D·num_layers, N, H_out)`, output `(L, N, D·H_out)`; `batch_first=True` only reorders input and output.
- Per-class ("mean category") accuracy vs overall accuracy on an **imbalanced** dataset.

## A basic RNN, built by hand

The lecture's toy RNN computes, at every step,
$$
h_t = \tanh\!\big(W_{h,x}x_t + W_{h,h}h_{t-1}\big), \qquad y_t = W_{y,x}x_t + W_{y,h}h_{t-1},
$$
with four parameter matrices (biases left out of the notation):
$$
W_{h,x}\in\mathbb{R}^{H\times d},\quad W_{h,h}\in\mathbb{R}^{H\times H},\quad W_{y,x}\in\mathbb{R}^{K\times d},\quad W_{y,h}\in\mathbb{R}^{K\times H}.
$$
Here $y_t\in\mathbb{R}^K$ are the task outputs, e.g. logits over $K$ classes. Concatenating input and previous state, $v_t = x_t\oplus h_{t-1}\in\mathbb{R}^{d+H}$, folds each pair of matrices into one:
$$
h_t = \tanh(W_h v_t), \qquad y_t = W_y v_t, \qquad W_h\in\mathbb{R}^{H\times(d+H)},\; W_y\in\mathbb{R}^{K\times(d+H)}.
$$

> [!key] The same matrices at every time step
> $W_h$ and $W_y$ do not have a time index. The network applies them $T$ times to process a length-$T$ sequence, so a 6-letter name and a 12-letter name use exactly the same parameters. This is the sequence version of a CNN's [[5-deep-networks/convolution|weight sharing over positions]].

In code, the two matrices are two `nn.Linear` layers on the concatenated vector, and the time loop is an ordinary Python `for` loop inside `forward` ([[1-foundations/tensors|`torch.cat`]] joins input and state, [`torch.stack`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html) collects the steps):

```python
class BasicRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc_h = nn.Linear(input_dim + hidden_dim, hidden_dim)  # W_h
        self.fc_y = nn.Linear(input_dim + hidden_dim, output_dim)  # W_y
        self.nonlinearity = nn.Tanh()

    def forward(self, x):                 # x: (seq_length, 1, input_dim)
        Y, H = [], []
        h = torch.zeros(1, self.hidden_dim)   # h_0 = 0
        for t in range(x.size(0)):
            v = torch.cat((x[t], h), dim=1)   # v_t = x_t ⊕ h_{t-1}
            h = self.nonlinearity(self.fc_h(v))
            y = self.fc_y(v)
            H.append(h)
            Y.append(y)
        return torch.stack(Y, dim=0), torch.stack(H, dim=0)

rnn_model = BasicRNN(input_dim=30, hidden_dim=128, output_dim=20)
outputs, hiddens = rnn_model(torch.randn(8, 1, 30))   # 8 steps
print(outputs.shape, hiddens.shape)
```

```
torch.Size([8, 1, 20]) torch.Size([8, 1, 128])
```

One output and one hidden state per time step. The hidden state starts as zeros, and both lists are stacked along a new first dimension (time), which is the layout PyTorch's own RNNs use.

> [!note] The lecture's toy output uses $h_{t-1}$, not $h_t$
> `y = self.fc_y(v)` reads the output from $v_t = x_t\oplus h_{t-1}$, the *previous* state plus the current input. PyTorch's `nn.RNN` instead returns the *new* hidden state $h_t$ as its output, and you add your own `nn.Linear` head on top. Both are valid designs; know which one a question is using.

### Worked example: memory in one number

Take $d = H = 1$, no bias, $W_{h,x} = 1$, $W_{h,h} = 0.5$, $h_0 = 0$, and the input sequence $x = (1, 0, -1)$:
$$
\begin{aligned}
h_1 &= \tanh(1\cdot 1 + 0.5\cdot 0) = \tanh(1) = 0.7616\\
h_2 &= \tanh(1\cdot 0 + 0.5\cdot 0.7616) = \tanh(0.3808) = 0.3634\\
h_3 &= \tanh(1\cdot(-1) + 0.5\cdot 0.3634) = \tanh(-0.8183) = -0.6741
\end{aligned}
$$
At $t=2$ the input is $0$, yet $h_2 = 0.3634 \ne 0$: the state *remembers* the earlier $x_1 = 1$. With $W_{h,h}=0$ the model would forget instantly ($h = (0.7616,\ 0,\ -0.7616)$, each step a function of its own input only), which is exactly an MLP applied to each step separately.

### Counting parameters

Each `nn.Linear(n_in, n_out)` has $n_{\text{out}}\,n_{\text{in}} + n_{\text{out}}$ parameters. For the surname model below ($d = 52$ letters, $H = 64$, $K = 18$ languages):
$$
\underbrace{(52+64)\cdot 64 + 64}_{\texttt{fc\_h}:\ 7{,}488} \;+\; \underbrace{(52+64)\cdot 18 + 18}_{\texttt{fc\_y}:\ 2{,}106} \;=\; 9{,}594 .
$$
The count is the same for a name of any length. Compare the RNN variants on [[5-deep-networks/lstm-and-gru#counting-parameters-of-rnn-gru-and-lstm|the LSTM/GRU page]].

## Backpropagation through time

Unfolding turns the loop into an ordinary [[2-calculus-and-optimization/computational-graphs|computational graph]]: a chain of $T$ copies of the cell, all reading the same weight tensors. Calling `loss.backward()` runs [[2-calculus-and-optimization/backpropagation|backpropagation]] backwards through that chain, which is called **backpropagation through time** (BPTT). Two consequences:

- **Gradients accumulate over time.** $W_h$ is used at every step, so its gradient is the *sum* of the contributions from all $T$ uses, the same rule as for a variable with several outgoing edges in a graph ([[2-calculus-and-optimization/backpropagation|gradient accumulation]]). Autograd does this for you.
- **Long chains shrink or blow up the gradient.** The gradient that reaches $h_1$ from a loss at step $T$ passes through $T-1$ factors of the form $\operatorname{diag}(\tanh'(\cdot))\,W_{h,h}$. Since $\lvert\tanh'\rvert \le 1$, these products tend to **vanish** (early inputs barely affect learning) or, if $W_{h,h}$ is large, **explode**. This is the same depth problem as in very deep plain CNNs ([[5-deep-networks/residual-connections|residual connections]]); for RNNs the fix is gating, in [[5-deep-networks/lstm-and-gru|LSTM and GRU]] cells.

> [!intuition] An RNN is a very deep network with tied weights
> Unrolled over $T$ steps, a one-layer RNN is a $T$-layer network in which every layer has the same weights. Everything that makes deep networks hard to train applies along the time axis.

## Encoding text: one-hot vectors

A network needs numbers, so each letter becomes a **one-hot** vector: all zeros except a single 1 at the letter's index. With `string.ascii_letters` (26 lowercase + 26 uppercase) there are 52 symbols:

| Letter | Encoding in $\{0,1\}^{52}$ |
| --- | --- |
| `a` (index 0) | $(1, 0, 0, 0, \ldots, 0)$ |
| `b` (index 1) | $(0, 1, 0, 0, \ldots, 0)$ |
| `c` (index 2) | $(0, 0, 1, 0, \ldots, 0)$ |

This is the same encoding that sits inside [[3-linear-models/multiclass-logistic-regression|cross-entropy]]: label $y = 3$ out of 5 classes is the distribution $(0,0,1,0,0)$. A name of length $n$ becomes $n$ one-hot vectors, stored as a `(n, 1, 52)` tensor: $n$ time steps, a batch of 1, 52 features, which is the layout `nn.Linear` (and `nn.RNN`) expects per step.

```python
all_letters = string.ascii_letters          # 52 symbols

def name_to_tensor(name):
    t = torch.zeros(len(name), 1, len(all_letters))
    for i, letter in enumerate(name):
        t[i][0][all_letters.find(letter)] = 1   # one 1 per step
    return t

name_to_tensor('Whitworth').shape
```

```
torch.Size([9, 1, 52])
```

[`torch.nn.functional.one_hot`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html) builds the same vectors from an index tensor. For large vocabularies (words, not letters), models usually learn a dense vector per symbol with [`nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) instead; Lecture 17 sticks with one-hot letters, and Lecture 18 makes the switch: [[7-transformers/text-encodings-and-embeddings|text encodings and learned embeddings]].

## Example: which language is a surname from?

The Lecture 17 task (adapted from the [PyTorch character-level RNN tutorial](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)): `data/names/` holds one text file per language, 18 languages in all, and the model must predict the language of a surname such as *Khoury* (Arabic), *Babineaux* (French) or *Kowalski* (Polish).

**Data.** A custom [[4-pytorch-training/datasets-and-dataloaders|`Dataset`]] (`NamesDataset`) reads every file, converts each name with `name_to_tensor`, and labels it with the language's index $0,\dots,17$ as a `torch.long` tensor. Half of the names are drawn at random for training and the rest for validation, with [`Subset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Subset). Names have different lengths, so they cannot be stacked into one batch tensor; the `DataLoader` gets a `collate_fn` that simply returns the list of examples:

```python
def collate_fn(batch_data):
    return [b for b in batch_data]   # keep variable lengths

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
```

**Training (many-to-one).** Each name is fed through the RNN one letter at a time; only the **last** output, `outputs[-1]`, is used as the prediction, because by then the hidden state has read the whole name. The loss of each name is divided by the batch size, so `backward()` on the sum gives the mini-batch **average** gradient:

```python
criterion = nn.CrossEntropyLoss()
for batch in train_loader:
    optimizer.zero_grad()
    curr_loss = 0
    for name, category, name_tensor, category_tensor in batch:
        outputs, hiddens = rnn_model(name_tensor)
        curr_loss += criterion(outputs[-1], category_tensor) / len(batch)  # last step
    curr_loss.backward()
    optimizer.step()
```

The optimizer is [`torch.optim.SGD`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) with `lr=3e-3` and `momentum=0.99`, hidden size 64. See [[4-pytorch-training/training-loop|the training loop]] and [[4-pytorch-training/loss-functions|`CrossEntropyLoss`]] for the general pattern; the only RNN-specific parts are the per-name loop and `outputs[-1]`. (Batching variable-length sequences properly means padding them, which [`pad_sequence`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html) and [`pack_padded_sequence`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html) handle; the lecture skips this.)

**Results** (mean of the 18 per-language accuracies, from the lecture's runs):

| Model | Training | Validation |
| --- | --- | --- |
| `BasicRNN`, 5 epochs | 23.08% | 22.56% |
| `BasicRNN`, 50 epochs (saved checkpoint) | 49.33% | 46.00% |
| `BidirectionalRNN`, 50 epochs | 55.87% | 55.27% |

> [!trap] Imbalanced classes: overall accuracy can hide a useless model
> After 5 epochs the basic RNN gets 7,138 of the 10,037 training names right, **71.1%** overall, yet its *mean per-language* accuracy is only **23.1%**. Russian names are 4,727 of the 10,037 (47%) and English another 1,838; the model learned to answer mostly "Russian", "English" or "Arabic" (90%, 85% and 73% correct) and got Dutch, French, Greek, Irish, Korean, Polish, Portuguese, Scottish, Spanish and Vietnamese **0%** right. Averaging the accuracy *per class* exposes this. Longer training (50 epochs) lifts the rare languages, and the [[5-deep-networks/bidirectional-and-stacked-rnns|bidirectional]] model does better still.

### Many-to-one and many-to-many

| Use | What is read out | Example |
| --- | --- | --- |
| many-to-one | the last output $y_T$ (or $h_T$) | language of a surname; sentiment of a review |
| many-to-many (aligned) | $y_t$ at every step | vehicle speed at every sensor reading; a tag per word |

The model is the same; only which outputs enter the loss changes.

## `nn.RNN` in PyTorch

The basic RNN with the output head removed is the **Elman RNN**, which [`nn.RNN`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html) implements. For each layer it computes
$$
h_t = \tanh\!\big(W_{ih}\,x_t + b_{ih} + W_{hh}\,h_{t-1} + b_{hh}\big),
$$
with two bias vectors per layer, and `nonlinearity='relu'` as the alternative to `'tanh'`. The constructor is `nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False)`.

<!-- fig:deepc-rnn-shapes -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 370" width="800" height="370" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:14px"><title>Grid of hidden states over 2 layers and 4 time steps; top row marked as output, last column marked as h_n</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><rect x="174" y="76" width="472" height="68" rx="10" fill="var(--accent)" fill-opacity="0.07" stroke="var(--accent)" stroke-width="1.8" stroke-dasharray="6 4"/><rect x="540" y="68" width="100" height="179" rx="10" fill="var(--hi)" fill-opacity="0.06" stroke="var(--hi)" stroke-width="1.8" stroke-dasharray="6 4"/><rect x="189" y="90" width="82" height="40" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="230" y="110" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">h<tspan baseline-shift="sub" font-size="11">1</tspan><tspan baseline-shift="super" font-size="10">(2)</tspan></text><rect x="309" y="90" width="82" height="40" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="350" y="110" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">h<tspan baseline-shift="sub" font-size="11">2</tspan><tspan baseline-shift="super" font-size="10">(2)</tspan></text><rect x="429" y="90" width="82" height="40" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="470" y="110" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">h<tspan baseline-shift="sub" font-size="11">3</tspan><tspan baseline-shift="super" font-size="10">(2)</tspan></text><rect x="549" y="90" width="82" height="40" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="590" y="110" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">h<tspan baseline-shift="sub" font-size="11">4</tspan><tspan baseline-shift="super" font-size="10">(2)</tspan></text><text x="20" y="110" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">layer 2</text><text x="118" y="110" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">h<tspan baseline-shift="sub" font-size="11">0</tspan><tspan baseline-shift="super" font-size="10">(2)</tspan></text><line x1="138" y1="110" x2="186" y2="110" stroke="var(--muted)" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><line x1="271" y1="110" x2="306" y2="110" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><line x1="391" y1="110" x2="426" y2="110" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><line x1="511" y1="110" x2="546" y2="110" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="189" y="185" width="82" height="40" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="230" y="205" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">h<tspan baseline-shift="sub" font-size="11">1</tspan><tspan baseline-shift="super" font-size="10">(1)</tspan></text><rect x="309" y="185" width="82" height="40" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="350" y="205" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">h<tspan baseline-shift="sub" font-size="11">2</tspan><tspan baseline-shift="super" font-size="10">(1)</tspan></text><rect x="429" y="185" width="82" height="40" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="470" y="205" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">h<tspan baseline-shift="sub" font-size="11">3</tspan><tspan baseline-shift="super" font-size="10">(1)</tspan></text><rect x="549" y="185" width="82" height="40" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="590" y="205" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">h<tspan baseline-shift="sub" font-size="11">4</tspan><tspan baseline-shift="super" font-size="10">(1)</tspan></text><text x="20" y="205" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">layer 1</text><text x="118" y="205" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">h<tspan baseline-shift="sub" font-size="11">0</tspan><tspan baseline-shift="super" font-size="10">(1)</tspan></text><line x1="138" y1="205" x2="186" y2="205" stroke="var(--muted)" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><line x1="271" y1="205" x2="306" y2="205" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><line x1="391" y1="205" x2="426" y2="205" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><line x1="511" y1="205" x2="546" y2="205" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><text x="230" y="300" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">x<tspan baseline-shift="sub" font-size="11">1</tspan></text><line x1="230" y1="288" x2="230" y2="228" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><line x1="230" y1="185" x2="230" y2="133" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><text x="350" y="300" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">x<tspan baseline-shift="sub" font-size="11">2</tspan></text><line x1="350" y1="288" x2="350" y2="228" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><line x1="350" y1="185" x2="350" y2="133" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><text x="470" y="300" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">x<tspan baseline-shift="sub" font-size="11">3</tspan></text><line x1="470" y1="288" x2="470" y2="228" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><line x1="470" y1="185" x2="470" y2="133" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><text x="590" y="300" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">x<tspan baseline-shift="sub" font-size="11">4</tspan></text><line x1="590" y1="288" x2="590" y2="228" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><line x1="590" y1="185" x2="590" y2="133" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" marker-end="url(#ARR)"/><text x="20" y="300" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">input</text><text x="118" y="250" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">h_0: zeros</text><text x="118" y="266" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">by default</text><text x="180" y="52" text-anchor="start" dominant-baseline="central" fill="var(--accent)" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">output = top layer at every t:  (L, N, D·H_out)</text><text x="652" y="150" text-anchor="start" dominant-baseline="central" fill="var(--hi)" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">h_n = last t,</text><text x="652" y="168" text-anchor="start" dominant-baseline="central" fill="var(--hi)" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">every layer:</text><text x="652" y="188" text-anchor="start" dominant-baseline="central" fill="var(--hi)" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">(D·num_layers,</text><text x="652" y="206" text-anchor="start" dominant-baseline="central" fill="var(--hi)" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;"> N, H_out)</text><text x="20" y="336" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">shapes (batch_first=False):  input x: (L, N, H_in)    h_0 and h_n: (D·num_layers, N, H_out)</text><text x="20" y="354" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">L = sequence length,  N = batch size,  D = 2 if bidirectional else 1</text></svg><figcaption><b>What <code>nn.RNN(H_in, H_out, num_layers=2)</code> returns</b> (unidirectional, L = 4 steps). <code>output</code> stacks the <b>top layer's</b> hidden state at every time step; <code>h_n</code> holds the <b>last time step</b> of every layer. They share one cell: <code>output[-1]</code> equals <code>h_n[-1]</code>. With <code>batch_first=True</code> only <code>input</code> and <code>output</code> become (N, L, ·); <code>h_0</code>/<code>h_n</code> keep the layer dimension first.</figcaption></figure>

| Tensor | Shape (`batch_first=False`) | Meaning |
| --- | --- | --- |
| input `x` | `(L, N, H_in)` | $L$ time steps, batch of $N$, $H_{\text{in}}$ features per step |
| `h_0` (optional, zeros if omitted) | `(D·num_layers, N, H_out)` | initial state of every layer (and direction) |
| `output` | `(L, N, D·H_out)` | **top** layer's $h_t$ for every $t$ |
| `h_n` | `(D·num_layers, N, H_out)` | **last** step's $h_T$ for every layer (and direction) |

$D = 2$ when `bidirectional=True` and 1 otherwise ([[5-deep-networks/bidirectional-and-stacked-rnns|bidirectional and stacked RNNs]]). An unbatched input `(L, H_in)` is also accepted, and the batch dimension then disappears from every shape.

The lecture's `ElmanRNN` wraps `nn.RNN` and adds the task head on the last output, exactly as in `BasicRNN`. Shown here without its `bidirectional` option (the full version is on [[5-deep-networks/bidirectional-and-stacked-rnns#both-at-once-in-pytorch|bidirectional and stacked RNNs]]):

```python
class ElmanRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.n_layers, self.hidden_dim = n_layers, hidden_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, output_dim)     # task head

    def forward(self, x):                               # x: (L, N, input_dim)
        h0 = torch.zeros(self.n_layers, x.size(1), self.hidden_dim)
        outputs, hiddens = self.rnn(x, h0)              # loop runs inside
        return self.fc(outputs[-1]), hiddens            # many-to-one
```

For the surname model, `nn.RNN(52, 64)` on the 9-letter input `(9, 1, 52)` gives `output` of shape `(9, 1, 64)` and `h_n` of shape `(1, 1, 64)`; `self.fc(outputs[-1])` has shape `(1, 18)`, one row of logits, which is what `nn.CrossEntropyLoss` expects with a target of shape `(1,)`.

**Parameters of one `nn.RNN` layer** (two weight matrices, two bias vectors):
$$
\underbrace{H\cdot d}_{W_{ih}} + \underbrace{H\cdot H}_{W_{hh}} + \underbrace{2H}_{b_{ih},\,b_{hh}} = H(d + H + 2).
$$
`nn.RNN(52, 64)`: $64\cdot(52 + 64 + 2) = 7{,}552$. The weights are stored as `weight_ih_l0` `(64, 52)`, `weight_hh_l0` `(64, 64)`, `bias_ih_l0` and `bias_hh_l0` `(64,)`; list them with `named_parameters()` ([[4-pytorch-training/nn-module|`nn.Module`]]). For a step-at-a-time loop of your own, [`nn.RNNCell`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNNCell.html) is one step of the same layer.

> [!trap] Common mistakes
> - **Wrong axis order.** The default is `(L, N, H_in)`, time first. A batch shaped `(N, L, H_in)` runs without error and silently treats the batch as time. Pass `batch_first=True` or permute.
> - **`batch_first` does not touch `h_0`/`h_n`.** They stay `(D·num_layers, N, H_out)` either way.
> - **`h_n` is not `output`.** `output` is every step of the top layer; `h_n` is the last step of every layer. For a 1-layer unidirectional RNN, `output[-1]` and `h_n[0]` hold the same numbers.
> - **Using every step's output in a many-to-one loss** (or only the first step's). The prediction for a whole sequence comes from the last step, where the state has read everything (for bidirectional RNNs see [[5-deep-networks/bidirectional-and-stacked-rnns#which-output-summarizes-the-whole-sequence|this subtlety]]).
> - **Forgetting `zero_grad()`.** Gradients from all time steps and all names in the batch accumulate in `.grad`; that is intended within one step, but they must be cleared before the next ([[2-calculus-and-optimization/autograd|autograd]]).
> - Thinking the parameter count grows with $T$. It does not; the compute and memory for backpropagation do.

## Check yourself

> [!question] 1. Shapes
> `rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=3)` receives `x` of shape `(7, 4, 10)`. Give the shapes of `output` and `h_n`, and say which of them contains $h^{(3)}_7$ for the second sequence in the batch.

> [!solution]- Answer
> `output`: `(7, 4, 20)`, the top (3rd) layer at all 7 steps. `h_n`: `(3, 4, 20)`, all 3 layers at the last step. $h^{(3)}_7$ of sequence 2 is in both: `output[6, 1]` and `h_n[2, 1]`.

> [!question] 2. Parameters
> How many learnable parameters does `nn.RNN(input_size=10, hidden_size=20)` have? And the from-scratch `BasicRNN(10, 20, 5)`?

> [!solution]- Answer
> `nn.RNN`: $20\cdot 10 + 20\cdot 20 + 2\cdot 20 = 200 + 400 + 40 = 640$.
> `BasicRNN`: `fc_h` is `nn.Linear(30, 20)` with $30\cdot 20 + 20 = 620$; `fc_y` is `nn.Linear(30, 5)` with $30\cdot 5 + 5 = 155$; total $775$. (`nn.RNN` has one more bias vector, $b_{hh}$, but no output head.)

> [!question] 3. Memory
> With $d=H=1$, no bias, $W_{h,x}=2$, $W_{h,h}=-1$, $h_0=0$, compute $h_1, h_2$ for $x = (0.5, 0.5)$.

> [!solution]- Answer
> $h_1 = \tanh(2\cdot 0.5 - 0) = \tanh(1) = 0.7616$; $h_2 = \tanh(1 - 0.7616) = \tanh(0.2384) = 0.2340$. Same input both times, different state: the negative recurrent weight pushes against the previous state.

## PyTorch docs

- [`torch.nn.RNN`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html) — equation, every constructor argument, input/output shapes, weight names.
- [`torch.nn.RNNCell`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNNCell.html) — a single time step, for writing the loop yourself.
- [`torch.cat`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html) and [`torch.stack`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html) — concatenating $x_t\oplus h_{t-1}$; collecting steps along a new time axis.
- [`torch.nn.functional.one_hot`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html) and [`torch.nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) — turning symbol indices into vectors.
- [`torch.nn.utils.rnn.pad_sequence`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html), [`pack_padded_sequence`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html) — batching sequences of different lengths.
- [`torch.utils.data.Subset`, `DataLoader` and `collate_fn`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) — the train/validation split and the list-returning collate function.
- [`torch.nn.CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — loss on the last step's logits.
- Tutorial: [NLP from scratch: classifying names with a character-level RNN](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) — the source of the surname task.

Related: [[5-deep-networks/bidirectional-and-stacked-rnns]] · [[5-deep-networks/lstm-and-gru]] · [[5-deep-networks/multilayer-perceptrons]] · [[2-calculus-and-optimization/backpropagation]] · [[4-pytorch-training/datasets-and-dataloaders]] · [[3-linear-models/multiclass-logistic-regression]] · [[7-transformers/text-encodings-and-embeddings]] · [[7-transformers/tokenization]]
