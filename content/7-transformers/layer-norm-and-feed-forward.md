---
title: "Layer norm, residuals and the feed-forward block"
description: "The two blocks that turn the Skinny Transformer into the Medium one: the position-wise feed-forward network (the same two-layer MLP on every token) and the Add & Norm wrapper LayerNorm(x + Sublayer(x)); layer norm's formula worked by hand and compared with batch norm, why transformers use it, parameter counts, the course code (one shared LayerNorm) versus the paper, and a correction to the lecture's description."
tags: [topic, transformers]
aliases: ["layer norm", "layer normalization", "LayerNorm", "Add & Norm", "add and norm", "position-wise feed-forward", "FFN", "feed-forward network"]
---

*Lecture 20 · [[7-transformers/index|Transformers]] · builds on [[5-deep-networks/residual-connections|residual connections]] and [[5-deep-networks/batch-normalization|batch normalization]] · part of the [[7-transformers/transformer-architecture|Medium and Full Transformers]]*

> [!abstract] In one breath
> Attention mixes information **between** tokens; the **position-wise feed-forward network** (FFN) then processes each token **on its own** with a two-layer MLP, $\mathrm{FFN}(x) = W_2\,\mathrm{ReLU}(W_1x + b_1) + b_2$, the same weights for every position. Every attention block and every FFN is wrapped in **Add & Norm**: a residual connection followed by **layer normalization**, $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$. Layer norm standardizes each token's feature vector by its own mean and variance (then rescales with learned $\gamma$, $\beta$), so unlike batch norm it needs no batch statistics and does exactly the same thing in training and evaluation. Adding these two pieces is the biggest single improvement in the course's Skinny → Medium → Full progression.

## What it is

![[l20-medium-transformer.jpg|Skinny to Medium Transformer: each attention block gets an Add & Norm, and a Feed Forward block with its own Add & Norm follows]]

The Medium Transformer adds, to every layer:

1. a **feed-forward block** after the attention block(s), and
2. an **Add & Norm** after every attention block and after the feed-forward block.

> [!definition] Definitions (FFN and Add & Norm: Vaswani et al., 2017, §3.3 and §3.1; layer norm: Ba et al., 2016)
> **Position-wise FFN.** For one token's vector $x\in\mathbb R^{d}$:
> $$
> \mathrm{FFN}(x) = \max(0,\;xW_1 + b_1)\,W_2 + b_2,\qquad W_1\in\mathbb R^{d\times d_{ff}},\; W_2\in\mathbb R^{d_{ff}\times d}.
> $$
> **Add & Norm.** For a sub-layer $\mathcal F$ (attention or FFN): $\;\mathrm{LayerNorm}\bigl(x + \mathcal F(x)\bigr)$.
>
> **Layer norm** of $x\in\mathbb R^{d}$, with learned $\gamma, \beta\in\mathbb R^{d}$:
> $$
> \mu = \frac1d\sum_{i=1}^d x_i,\qquad \sigma^2 = \frac1d\sum_{i=1}^d (x_i-\mu)^2,\qquad \mathrm{LN}(x)_i = \gamma_i\,\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta_i .
> $$

The lecture writes the FFN as $z = W_2\,\mathrm{ReLU}(W_1x)$ with column vectors; the code's `nn.Linear` layers include the biases.

## What you should know

- The FFN formula, its sizes (paper $d = 512$, $d_{ff} = 2048 = 4d$; course $d_{emb} = 128$, $d_{ff} = 256 = 2d_{emb}$) and parameter count $2dd_{ff} + d_{ff} + d$.
- "Position-wise": the same FFN is applied to every token separately; tokens do not interact inside it (equivalently, a convolution with kernel size 1 along the sequence).
- The division of labour: attention moves information between positions; the FFN transforms features within a position.
- Add & Norm = $\mathrm{LayerNorm}(x+\mathcal F(x))$: the residual lets a block refine a representation (if $\mathcal F(x)=0$ nothing changes) and gives gradients a direct path.
- Layer norm computes mean and variance **over the features of one token**; batch norm over **one feature across the batch**. Layer norm keeps no running statistics and behaves identically at train and eval time. Parameters: $2d$ ($\gamma$ and $\beta$).
- What changed from Skinny to Medium in the results: training loss 0.54 → 0.10, validation BLEU 0.34 → 0.46 after 3 million steps.

## The position-wise feed-forward network

The course's version:

```python
class PositionWiseFFN(nn.Module):
    def __init__(self, embedding_dim, ffn_dim=None):
        super(PositionWiseFFN, self).__init__()
        if ffn_dim is None:
            ffn_dim = 2*embedding_dim          # course: 2x; paper: 4x
        self.fc1 = nn.Linear(embedding_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embedding_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        z = self.activation(self.fc1(x))
        return self.fc2(z)
```

Given the $(N, 128)$ matrix of token vectors, `self.fc1` multiplies every **row** by the same weight matrix, so token $t$'s output depends only on token $t$'s input. That is what "position-wise" means, and why the paper calls it "two convolutions with kernel size 1".

Why it matters: an attention output is a softmax-weighted average of value vectors, so (for fixed weights) it is linear in the values. Attention decides which tokens to combine; the FFN, with its nonlinearity and wider hidden layer, gives the model room to recombine the features of each token afterwards. In the lecture's words, it allows "more sophisticated reasoning and blending of the features that each attention block correlate[s] between the tokens". It is also where most of a Transformer layer's parameters sit.

| | Course ($d = 128$, $d_{ff} = 256$) | Paper base ($d = 512$, $d_{ff} = 2048$) |
| --- | --- | --- |
| FFN: $2dd_{ff} + d_{ff} + d$ | 65,920 | 2,099,712 |
| one attention block (projections) | 41,344 (multi-head) | $4(d^2 + d) = 1{,}050{,}624$ |
| one LayerNorm: $2d$ | 256 | 1,024 |

## Residual connections and Add & Norm

The residual idea is the one from [[5-deep-networks/residual-connections|ResNet]]: a block computes a **change** $\mathcal F(x)$ that is added to its input. If $\mathcal F(x) = 0$ the token's vector passes through unchanged, so each attention or FFN block can be read as refining how each token is expressed. The identity path also carries gradients straight down the stack. Adding requires $\mathcal F(x)$ and $x$ to have the same shape, which is why every sub-layer in a Transformer maps $d$-dimensional vectors to $d$-dimensional vectors.

The Medium encoder in the course:

```python
def forward(self, x):
    z = self.embedding(x).squeeze(1)
    Q, K, V = self.fc_Q(z), self.fc_K(z), self.fc_V(z)
    attn_output = self.attn_head(Q, K, V)
    attn_output = self.layer_norm(attn_output + z)              # Add & Norm 1
    ffn_output = self.ffn(attn_output)
    encoder_result = self.layer_norm(ffn_output + attn_output)  # Add & Norm 2
    return self.activation(encoder_result)
```

This is the paper's **post-norm** order: add, then normalize. (Many later Transformers normalize first, $x + \mathcal F(\mathrm{LN}(x))$; that is beyond the lecture.)

Two differences from the paper:

- The course creates **one** `nn.LayerNorm` per encoder or decoder and reuses it at every Add & Norm, so all positions share one $\gamma$ and one $\beta$. The paper has a separate layer norm for each sub-layer (2 per encoder layer, 3 per decoder layer).
- The course's encoder ends with a ReLU after the last layer norm, which the paper does not have.

## Layer norm versus batch norm

<!-- fig:tf-b-norm-axes -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 230" width="720" height="230" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>Two grids: a highlighted column for batch norm, a highlighted row for layer norm</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><text x="228" y="20" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;font-weight:600;">BatchNorm1d</text><rect x="150" y="36" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="176" y="36" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="202" y="36" width="24" height="24" rx="2" fill="var(--accent)" fill-opacity="0.55" stroke="var(--accent)" stroke-width="1"/><rect x="228" y="36" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="254" y="36" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="280" y="36" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="150" y="62" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="176" y="62" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="202" y="62" width="24" height="24" rx="2" fill="var(--accent)" fill-opacity="0.55" stroke="var(--accent)" stroke-width="1"/><rect x="228" y="62" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="254" y="62" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="280" y="62" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="150" y="88" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="176" y="88" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="202" y="88" width="24" height="24" rx="2" fill="var(--accent)" fill-opacity="0.55" stroke="var(--accent)" stroke-width="1"/><rect x="228" y="88" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="254" y="88" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="280" y="88" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="150" y="114" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="176" y="114" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="202" y="114" width="24" height="24" rx="2" fill="var(--accent)" fill-opacity="0.55" stroke="var(--accent)" stroke-width="1"/><rect x="228" y="114" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="254" y="114" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="280" y="114" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><text x="228" y="156" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">features (d)  →</text><text x="140" y="88" text-anchor="end" dominant-baseline="central" fill="var(--muted)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">tokens / examples ↓</text><text x="228" y="180" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">one mean, one variance per feature,</text><text x="228" y="197" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">over the batch; running stats at eval</text><text x="578" y="20" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;font-weight:600;">LayerNorm</text><rect x="500" y="36" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="526" y="36" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="552" y="36" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="578" y="36" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="604" y="36" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="630" y="36" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="500" y="62" width="24" height="24" rx="2" fill="var(--accent)" fill-opacity="0.55" stroke="var(--accent)" stroke-width="1"/><rect x="526" y="62" width="24" height="24" rx="2" fill="var(--accent)" fill-opacity="0.55" stroke="var(--accent)" stroke-width="1"/><rect x="552" y="62" width="24" height="24" rx="2" fill="var(--accent)" fill-opacity="0.55" stroke="var(--accent)" stroke-width="1"/><rect x="578" y="62" width="24" height="24" rx="2" fill="var(--accent)" fill-opacity="0.55" stroke="var(--accent)" stroke-width="1"/><rect x="604" y="62" width="24" height="24" rx="2" fill="var(--accent)" fill-opacity="0.55" stroke="var(--accent)" stroke-width="1"/><rect x="630" y="62" width="24" height="24" rx="2" fill="var(--accent)" fill-opacity="0.55" stroke="var(--accent)" stroke-width="1"/><rect x="500" y="88" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="526" y="88" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="552" y="88" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="578" y="88" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="604" y="88" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="630" y="88" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="500" y="114" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="526" y="114" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="552" y="114" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="578" y="114" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="604" y="114" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><rect x="630" y="114" width="24" height="24" rx="2" fill="var(--paper)" stroke="currentColor" stroke-width="1"/><text x="578" y="156" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">features (d)  →</text><text x="490" y="88" text-anchor="end" dominant-baseline="central" fill="var(--muted)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">tokens / examples ↓</text><text x="578" y="180" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">one mean, one variance per token,</text><text x="578" y="197" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">over its d features; same at train and eval</text></svg><figcaption><b>Which numbers share a mean and variance.</b> For a (tokens × <i>d</i>) activation matrix, batch norm normalizes each <b>column</b> (one feature across all examples in the batch); layer norm normalizes each <b>row</b> (all <i>d</i> features of one token), so it needs no batch and behaves the same in training and evaluation.</figcaption></figure>

![[l20-batch-vs-layer-norm.png|Batch norm averages each channel over the batch (and spatial positions); layer norm averages over all features of one example]]

The lecture's figure shows image activations of shape $(N, C, H, W)$: batch norm computes statistics per channel over $N, H, W$; layer norm per example over $C, H, W$. In a Transformer the activations of one sentence are a $(\text{tokens}, d)$ matrix and `nn.LayerNorm(d)` normalizes each token's row.

**Worked example.** Three tokens, $d = 4$ features, $\gamma = 1$, $\beta = 0$, $\epsilon$ negligible:

$$
X = \begin{bmatrix} 2 & 4 & 6 & 8\\ 1 & 1 & 1 & 5\\ 0 & 3 & 0 & -3 \end{bmatrix}
$$

| | Layer norm (per row) | Batch norm (per column) |
| --- | --- | --- |
| means | 5, 2, 0 | 1, 2.667, 2.333, 3.333 |
| variances (divide by count) | 5, 3, 4.5 | 0.667, 1.556, 6.889, 21.556 |
| result row 1 | $(-1.342, -0.447, 0.447, 1.342)$ | $(1.225, 1.069, 1.397, 1.005)$ |
| result row 2 | $(-0.577, -0.577, -0.577, 1.732)$ | $(0, -1.336, -0.508, 0.359)$ |
| result row 3 | $(0, 1.414, 0, -1.414)$ | $(-1.225, 0.267, -0.889, -1.364)$ |

After layer norm every **row** has mean 0 and variance 1, whatever the other rows contain: token 1's output would be the same in a batch of one. After batch norm every **column** has mean 0 and variance 1, and token 1's output depends on tokens 2 and 3.

| | `nn.BatchNorm1d(d)` | `nn.LayerNorm(d)` |
| --- | --- | --- |
| statistics over | the batch (per feature) | the features (per token) |
| training vs eval | batch statistics vs **running** averages | identical |
| needs a batch | yes (fails on a single sample in training) | no |
| learned parameters | $\gamma, \beta\in\mathbb R^d$ | $\gamma, \beta\in\mathbb R^d$ |
| extra buffers | `running_mean`, `running_var` | none |

Why Transformers use layer norm: sentences have different lengths and the course processes one sentence at a time, so "the batch" is not a stable set to average over; layer norm only needs the token itself and behaves the same at test time.

> [!note] Correction to the lecture
> The notebook says layer norm, like batch norm, uses "a running mean and standard deviation". Batch norm keeps running averages for evaluation; **layer norm does not**. It recomputes $\mu$ and $\sigma^2$ from each token's own features every time, in training and in evaluation, and has no `running_mean` buffers.

## What the two blocks buy

From the [[7-transformers/transformer-architecture#results|saved Lecture 20 results]] (3 million training steps): Skinny → Medium lowers the training loss from 0.54 to 0.10, raises training BLEU from 0.55 to 0.94 and validation BLEU from 0.34 to 0.46. The Skinny model cannot even fit its training data; with the FFN, residuals and normalization it can (and starts to overfit).

> [!trap] Common mistakes
> - **Normalizing the wrong axis.** `nn.LayerNorm(d)` normalizes over the **last** dimension, the features. Passing the sequence length would normalize across tokens.
> - **Thinking the FFN mixes tokens.** It never does; only attention moves information between positions.
> - **Expecting running statistics.** `model.eval()` changes nothing about layer norm (it does change batch norm and dropout).
> - **Shape mismatch in the residual.** $x + \mathcal F(x)$ needs equal shapes; a sub-layer that changes the width needs a projection, as in ResNet.
> - **Post-norm vs pre-norm.** The paper (and the course) normalize **after** adding.

> [!question]- Check yourself
> 1. Layer-normalize the token $x = (1, 2, 3, 4)$ with $\gamma = 1$, $\beta = 0$.
> 2. How many parameters do the FFN and the layer norm add to one encoder layer of the paper's base model? What fraction of the layer is FFN?
> 3. A batch holds one sentence of 7 tokens, shape $(7, 128)$. How many means does `nn.LayerNorm(128)` compute? How many would `nn.BatchNorm1d(128)` compute on the same tensor?
>
> > [!solution]- Answers
> > 1. $\mu = 2.5$, $\sigma^2 = 1.25$, $\sqrt{1.25} = 1.118$: $(-1.342, -0.447, 0.447, 1.342)$ — the same as for $(2,4,6,8)$, because layer norm removes scale and shift.
> > 2. FFN 2,099,712; two layer norms 2,048; with the attention block (1,050,624) the layer has 3,152,384 parameters, two-thirds of them in the FFN.
> > 3. Layer norm: 7 means (one per token, over 128 features). Batch norm: 128 means (one per feature, over the 7 rows).

## PyTorch docs

- [`torch.nn.LayerNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) — `normalized_shape`, `eps` (default $10^{-5}$), `elementwise_affine`.
- [`torch.nn.functional.layer_norm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.layer_norm.html) — functional form.
- [`torch.nn.BatchNorm1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) — for comparison (running statistics, `momentum`).
- [`torch.nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) — acts on the last dimension, so it is automatically position-wise; [`torch.nn.Conv1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) with `kernel_size=1` is the equivalent convolution.
- [`torch.nn.ReLU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html).

Related: [[7-transformers/transformer-architecture]] · [[5-deep-networks/residual-connections]] · [[5-deep-networks/batch-normalization]] · [[5-deep-networks/multilayer-perceptrons]] · [[7-transformers/attention-is-all-you-need]]
