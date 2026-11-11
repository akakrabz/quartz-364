---
title: "Midterm 2 formula sheet"
description: "One page for Midterm 2: what the lecturer said the provided formula sheet will contain and what the Fall 2025 sheet did contain (so you need not memorize it), and next to it everything the sheet does not give and you should know cold: parameter counts, shape rules, attention roles and the mask, layer orders, train-versus-eval behaviour and the model comparison."
tags: [exam, midterm-2]
aliases: ["Midterm 2 cheat sheet", "MT2 formula sheet"]
---

*Midterm 2 · a formula sheet **will be provided** (said in class, Nov 10, 2026) · this semester's sheet has not been posted in the course folder; the Fall 2025 one is reproduced [[9-midterm-2/fall-2025#the-formula-sheet-that-came-with-the-exam|here]] · back to [[9-midterm-2/index|Midterm 2]]*

> [!exam] Said in class
> There will be a formula sheet as on Midterm 1, and a more extensive one. His example of what it will give: the output size of a convolutional layer, which he called a waste of brain space to memorize. And the hint that goes with it: what is on the sheet tells you what you need not memorize, as opposed to what you should "study and make sense of".

## Expect it on the sheet

From the Fall 2025 sheet, by the same instructor:

- Signatures: `torch.optim.SGD(params, lr=0.001, momentum=0, weight_decay=0)`, `DataLoader(dataset, batch_size, shuffle=None, sampler=None)`, `nn.Linear(in_features, out_features, bias=True)`, `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)`, `nn.MaxPool2d(kernel_size, stride=None, padding=0)`.
- Output size $\left\lfloor\dfrac{H_{in} + 2\cdot\text{padding} - \text{kernel\_size}}{\text{stride}} + 1\right\rfloor$.
- The LSTM and GRU equations.
- Self-, masked, cross- and multi-head attention, with the mask $M_{ij} = 0$ for $i\ge j$, $-\infty$ for $i<j$.
- The sinusoidal positional encoding and the Transformer diagram.

## Know without the sheet

### Sizes and counts

$$
\texttt{Linear(a, b)}:\ ab+b\qquad \texttt{Conv2d}:\ C_{out}(C_{in}K^2+1)\qquad \texttt{Embedding(V, d)}:\ Vd\qquad \texttt{RNN(d, H)}:\ H(d+H+2)
$$

GRU $3\times$, LSTM $4\times$ the RNN count; bidirectional $2\times$; `LayerNorm(d)` and `BatchNorm2d(C)`: $2d$, $2C$; pooling, activations, dropout: 0; multi-head attention $4E^2 + 4E$ for any number of heads. Drop the bias term when `bias=False`. Image size and sequence length are not in a conv, RNN or attention count.

Pooling uses the conv size formula with stride = kernel by default. Global average pooling: $(C, H, W)\to C$. Receptive field: $r_l = r_{l-1} + (k_l - 1)j_{l-1}$, $j_l = j_{l-1}s_l$, $r_0 = j_0 = 1$. Vision Transformer: $(H/P)(W/P)$ tokens (+1 class token) of length $CP^2$.

### Recurrent networks

$h_t = \tanh(W_{ih}x_t + W_{hh}h_{t-1} + b)$, the same weights at every step; the state must be handed on (`h = h_next`). LSTM: the cell state $c_t$ is the memory, gates are sigmoids. Bidirectional: two sets of weights, needs the whole sequence.

### Attention and Transformers

- $QK^\top$: row $i$ = token $i$'s scores over all tokens; softmax along rows; times $V$.
- The mask is **added** before the softmax; $-\infty$ above the diagonal; decoder self-attention only.
- Self-attention: $Q, K, V$ from the same tokens. Cross-attention: $Q$ from the decoder, $K, V$ from the encoder; weights $N_{dec}\times N_{enc}$.
- Encoder layer: attention → add → **layer** norm → feed-forward → add → layer norm. Decoder layer: masked self-attention, cross-attention, feed-forward, each with add and norm.
- Positional encoding is added to the embeddings because attention ignores order. `<CLS>`: a reserved first token whose output is used for classification. Decoder-only models have no cross-attention.

### Activations and training

- Without activations, stacked linear layers are one linear layer. $\sigma'\le\tfrac14$ (vanishing gradients); ReLU's derivative is 1 for $x>0$, 0 for $x<0$.
- $\mathrm{ReLU}(u) - \mathrm{ReLU}(-u) = u$; $\mathrm{ReLU}(u) + \mathrm{ReLU}(-u) = \lvert u\rvert$.
- Epoch = $\lceil N/B\rceil$ iterations. GD: whole dataset per step. SGD: a batch per step, noisy. Momentum: $v_t = \mu v_{t-1} + g_t$.
- Underfitting: both curves poor and together. Overfitting: validation turns worse while training improves.
- `model.eval()`: dropout off; batch norm uses running statistics. An augmentation must keep the label.
- `Dataset`: `__init__`, `__len__`, `__getitem__`. `nn.Module`: `super().__init__()`, layers in `__init__`, `forward`. `nn.CrossEntropyLoss` takes logits.

### Model comparison

| | fixed input size | first layer sees | shares weights across |
|---|---|---|---|
| MLP | yes | everything | nothing |
| CNN | no (with global pooling) | a $K\times K$ window | positions |
| RNN | no | inputs so far | time steps |
| Transformer / ViT | no | every token | tokens |

Related: [[9-midterm-2/review-sizes-and-parameters]] · [[9-midterm-2/review-attention-and-transformers]] · [[9-midterm-2/review-choosing-models]] · [[6-midterm-1/formula-sheet]]
