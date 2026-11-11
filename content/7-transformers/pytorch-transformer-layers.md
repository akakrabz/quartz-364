---
title: "PyTorch's Transformer layers"
description: "Lecture 21's rebuild of the translation Transformer with nn.TransformerEncoderLayer and nn.TransformerDecoderLayer: what one layer object contains, its constructor arguments, what you still write yourself (embeddings, positions, the causal mask, the output layer), stacking with nn.ModuleList, input shapes and batch_first, parameter counts per layer and for the whole model, and how it differs from the hand-built Lecture 20 blocks."
tags: [topic, transformers]
aliases: ["nn.TransformerEncoderLayer", "nn.TransformerDecoderLayer", "multi-layer Transformer", "stacking Transformer layers", "TransformerEncoderLayer"]
---

*Lecture 21 · [[7-transformers/index|Transformers]] · builds on [[7-transformers/transformer-architecture|the Transformer architecture]], [[7-transformers/multi-head-attention|multi-head attention]] and [[4-pytorch-training/nn-module|nn.Module]]*

> [!abstract] In one breath
> Lecture 20 wrote an encoder layer and a decoder layer by hand. PyTorch ships both as ready-made modules: [`nn.TransformerEncoderLayer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) is self-attention plus the feed-forward network, each with dropout and Add & Norm, and [`nn.TransformerDecoderLayer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html) adds cross-attention on the encoder's output. One object is **one layer** of the paper's diagram. You still write the embedding, the positional encoding, the causal mask and the final prediction layer, and you get depth by looping over an [`nn.ModuleList`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) of layers.

## What it is

The [[7-transformers/transformer-architecture|Full Transformer]] of Lecture 20 had one encoder layer and one decoder layer, each built from the course's own attention, feed-forward and layer-norm classes. The one piece left out was depth: the paper stacks $N = 6$ of each. Lecture 21 gets depth, and much shorter code, by swapping the hand-written layers for PyTorch's.

<!-- fig:tf-d-layer-stack -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 430" width="760" height="430" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>Encoder and decoder columns showing which blocks are inside the PyTorch layer classes</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><text x="190" y="18" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">encoder</text><rect x="90" y="385" width="200" height="30" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="190" y="400" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:inherit;">source token ids</text><rect x="80" y="328" width="220" height="34" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="190" y="337" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:inherit;">nn.Embedding + positional</text><text x="190" y="354" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">you write this</text><rect x="30" y="150" width="320" height="150" rx="10" fill="var(--accent)" fill-opacity="0.06" stroke="var(--accent)" stroke-width="1.4" stroke-dasharray="5 4"/><rect x="80" y="245" width="220" height="34" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="190" y="254" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:inherit;">self-attention</text><text x="190" y="271" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">+ dropout, Add &amp; Norm</text><rect x="80" y="189" width="220" height="34" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="190" y="198" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:inherit;">feed-forward (d_ff)</text><text x="190" y="215" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">+ dropout, Add &amp; Norm</text><rect x="90" y="80" width="200" height="30" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="190" y="95" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">z_enc   (N, d_emb)</text><line x1="190" y1="385" x2="190" y2="364" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M190.0,364.0 L193.5,371.0 L186.5,371.0 Z" fill="currentColor"/><line x1="190" y1="328" x2="190" y2="281" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M190.0,281.0 L193.5,288.0 L186.5,288.0 Z" fill="currentColor"/><line x1="190" y1="245" x2="190" y2="225" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M190.0,225.0 L193.5,232.0 L186.5,232.0 Z" fill="currentColor"/><line x1="190" y1="189" x2="190" y2="112" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M190.0,112.0 L193.5,119.0 L186.5,119.0 Z" fill="currentColor"/><text x="570" y="18" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">decoder</text><rect x="455" y="385" width="230" height="30" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="570" y="400" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:inherit;">target ids, shifted right</text><rect x="460" y="328" width="220" height="34" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="570" y="337" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:inherit;">nn.Embedding + positional</text><text x="570" y="354" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">you write this</text><rect x="410" y="92" width="320" height="208" rx="10" fill="var(--accent)" fill-opacity="0.06" stroke="var(--accent)" stroke-width="1.4" stroke-dasharray="5 4"/><rect x="460" y="245" width="220" height="34" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="570" y="254" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:inherit;">masked self-attention</text><text x="570" y="271" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">tgt_mask = causal mask</text><rect x="460" y="189" width="220" height="34" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="570" y="198" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:inherit;">cross-attention</text><text x="570" y="215" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">memory = z_enc</text><rect x="460" y="133" width="220" height="34" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="570" y="142" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:inherit;">feed-forward (d_ff)</text><text x="570" y="159" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">each + dropout, Add &amp; Norm</text><rect x="455" y="33" width="230" height="30" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="570" y="48" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:inherit;">nn.Linear(d_emb, vocab) → logits</text><line x1="570" y1="385" x2="570" y2="364" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M570.0,364.0 L573.5,371.0 L566.5,371.0 Z" fill="currentColor"/><line x1="570" y1="328" x2="570" y2="281" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M570.0,281.0 L573.5,288.0 L566.5,288.0 Z" fill="currentColor"/><line x1="570" y1="245" x2="570" y2="225" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M570.0,225.0 L573.5,232.0 L566.5,232.0 Z" fill="currentColor"/><line x1="570" y1="189" x2="570" y2="169" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M570.0,169.0 L573.5,176.0 L566.5,176.0 Z" fill="currentColor"/><line x1="570" y1="133" x2="570" y2="65" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M570.0,65.0 L573.5,72.0 L566.5,72.0 Z" fill="currentColor"/><text x="190" y="167" text-anchor="middle" dominant-baseline="central" fill="var(--accent)" style="font-size:12px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;paint-order:stroke;stroke:var(--paper);stroke-width:5px;stroke-linejoin:round;">nn.TransformerEncoderLayer × n_layers</text><text x="570" y="109" text-anchor="middle" dominant-baseline="central" fill="var(--accent)" style="font-size:12px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;paint-order:stroke;stroke:var(--paper);stroke-width:5px;stroke-linejoin:round;">nn.TransformerDecoderLayer × n_layers</text><path d="M290,95 L375,95 L375,206 L452,206" fill="none" stroke="var(--hi)" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><line x1="445" y1="206" x2="459" y2="206" stroke="var(--hi)" stroke-width="1.5" stroke-linecap="round"/><path d="M459.0,206.0 L452.0,209.5 L452.0,202.5 Z" fill="var(--hi)"/><text x="392" y="190" text-anchor="middle" dominant-baseline="central" fill="var(--hi)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">keys, values</text></svg><figcaption><b>What PyTorch's two layer classes cover.</b> Each dashed box is one object: an <code>nn.TransformerEncoderLayer</code> is self-attention plus the feed-forward network, each wrapped in dropout and Add &amp; Norm; an <code>nn.TransformerDecoderLayer</code> adds cross-attention on the encoder output (its <code>memory</code> argument). Embeddings, positional encoding, the causal mask and the final prediction layer are still yours to write, and stacking is a loop over an <code>nn.ModuleList</code>.</figcaption></figure>

The data flow is unchanged from [[7-transformers/transformer-architecture#training-the-whole-target-in-one-pass|Lecture 20]]. The encoder turns $N$ source tokens into $N$ vectors `z_enc`. Every decoder layer reads the (shifted) target through masked self-attention, then looks at `z_enc` through [[7-transformers/attention#cross-attention|cross-attention]], and a last linear layer turns each decoder vector into scores over the target vocabulary.

## What you should know

- One `nn.TransformerEncoderLayer` = multi-head self-attention → dropout → Add & Norm → feed-forward (`Linear`, ReLU, `Linear`) → dropout → Add & Norm. One `nn.TransformerDecoderLayer` has a third sub-block, cross-attention, between the two.
- Constructor: `nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False)`. The lecture passes `(d_emb, n_head, 2*d_emb, dropout=dropout)`.
- Calls: `layer(z)` for the encoder; `layer(z_dec, z_enc, tgt_mask=causal_mask, tgt_is_causal=True)` for the decoder. The second argument (`memory`) supplies the keys and values of cross-attention.
- **Not included:** token embedding, positional encoding, building the mask, the final `nn.Linear(d_emb, vocab)`.
- Layers must be held in an `nn.ModuleList` (or `nn.Sequential`), never a plain Python list, or their parameters are invisible to `model.parameters()`.
- Default shapes are `(tokens, batch, d_model)`; `batch_first=True` switches to `(batch, tokens, d_model)`. An unbatched `(tokens, d_model)` input also works, and is what the translation code uses.
- `d_model` must be divisible by `nhead`. The number of heads does not change the parameter count.
- Parameters with $d_{ff} = 2E$: $8E^2 + 11E$ per encoder layer, $12E^2 + 17E$ per decoder layer ($E = d_{model}$). For $E = 128$: 132,480 and 198,784.
- The layers contain dropout, so `model.train()` and `model.eval()` now matter.

## From the lecture

*From the class recording, paraphrased.*

- Think of each PyTorch layer object as **one box** of the Transformer diagram: the encoder-layer class is the left box, the decoder-layer class the right one. "N layers" means the output of one box is the input of the next.
- He did not present this as a big invention: wrapping last lecture's code in a module would have done the same. The point is speed. With these layers, building a Transformer for a new task mostly "comes down to our ability to format the data properly" (Nov 5).
- The `nn.ModuleList` is not decoration. If the layers sit in an ordinary list, PyTorch cannot find their parameters, so they would not be updated by gradient descent.
- The decoder layer asks for more than the encoder layer: the decoder features, the encoder features (for cross-attention) and the target mask. The mask is passed in because some uses of a decoder do allow looking at later positions; here the target is causal.
- The `squeeze` calls in the translation code exist only because it handles one sentence at a time. Batching text needs padding or packing of sequences, which the lecture skips; the [[7-transformers/vision-transformer|Vision Transformer]] later in the lecture is batched properly.
- The learned positional table is given 1,000 rows as a safe arbitrary cap: the model can handle inputs up to 1,000 tokens.
- His summary of the lecture's goal: once you think generally about what a token is, encoder and decoder layers let you build models for many tasks quickly ([[7-transformers/text-classification-with-transformers|text classification]] and [[7-transformers/vision-transformer|images]] follow).

> [!exam] Said in class
> Nov 10, 2026: Midterm 2 covers Homeworks 4–6, "to the end of transformers", so Lecture 21 is the last lecture in scope. See [[9-midterm-2/index|Midterm 2]].

## The layer's arguments

| Argument | Default | Meaning | Lecture 21 |
| --- | --- | --- | --- |
| `d_model` | (required) | width of every token vector, in and out | `d_emb` = 128 |
| `nhead` | (required) | heads in each attention block; each head is `d_model / nhead` wide | 4 |
| `dim_feedforward` | 2048 | hidden width of the position-wise feed-forward network | `2 * d_emb` = 256 |
| `dropout` | 0.1 | dropout probability inside the layer | 0.0 for translation, 0.2 for the classifiers |
| `activation` | ReLU | nonlinearity of the feed-forward network | default |
| `batch_first` | `False` | `True`: inputs are `(batch, tokens, d_model)` | `False` for translation, `True` for the classifiers |
| `norm_first` | `False` | `False`: Add then Norm, as in the paper and in [[7-transformers/layer-norm-and-feed-forward|Lecture 20]] | default |

The decoder layer takes the same arguments. Both use [`nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) inside, with per-head key and value width `d_model / nhead` (32 here).

## The encoder, rewritten

```python
class TransformerEncoder(nn.Module):
    def __init__(self, num_embeddings, d_emb, n_layers, n_head, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        d_ff = 2*d_emb
        self.positional_encoding = PositionalEncoding('learned', d_emb)
        self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_emb, n_head, d_ff,
                                                                        dropout=dropout,
                                                                        batch_first=False)
                                             for n in range(n_layers)])
        self.embedding = nn.Embedding(num_embeddings, d_emb)

    def forward(self, x):
        z_enc = self.embedding(x).squeeze(-2)        # (N, 1, d) -> (N, d)
        z_enc = z_enc + self.positional_encoding(x)  # add positions
        for layer in self.encoder_layers:
            z_enc = layer(z_enc)
        return z_enc
```

The whole forward pass is "embed, add positions, loop". `PositionalEncoding` is Lecture 20's class in its `'learned'` mode: an [`nn.Embedding(1000, d_emb)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) indexed by `torch.arange(len(x))` (see [[7-transformers/positional-encoding#learned-positional-embeddings|learned positional embeddings]]). A sentence is a column of token indices of shape `(N, 1)`, so the embedding returns `(N, 1, d_emb)` and `squeeze(-2)` makes it the unbatched `(N, d_emb)` input that the layers accept.

## The decoder, rewritten

```python
class TransformerDecoder(nn.Module):
    def __init__(self, num_embeddings, d_emb, n_layers, n_head, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        d_ff = 2*d_emb
        self.positional_encoding = PositionalEncoding('learned', d_emb)
        self.decoder_layers = nn.ModuleList([nn.TransformerDecoderLayer(d_emb, n_head, d_ff,
                                                                        dropout=dropout,
                                                                        batch_first=False)
                                             for n in range(n_layers)])
        self.embedding = nn.Embedding(num_embeddings, d_emb)
        self.fc_predict = nn.Linear(d_emb, num_embeddings)

    def forward(self, x, z_enc):
        z_dec = self.embedding(x).squeeze(-2)
        z_dec = z_dec + self.positional_encoding(x)
        n_tokens = z_dec.size(0)
        causal_mask = torch.zeros((n_tokens, n_tokens))
        inf_values = torch.zeros((n_tokens, n_tokens)).fill_(float('-inf'))
        causal_mask += torch.triu(inf_values, diagonal=1)   # -inf above the diagonal
        for layer in self.decoder_layers:
            if n_tokens > 1:
                z_dec = layer(z_dec, z_enc, tgt_mask=causal_mask, tgt_is_causal=True)
            else:
                z_dec = layer(z_dec, z_enc)
        word_logits = self.fc_predict(z_dec)
        return word_logits
```

- The mask is the [[7-transformers/attention#masked-self-attention|additive causal mask]]: 0 on and below the diagonal, $-\infty$ above it ([`torch.triu`](https://docs.pytorch.org/docs/stable/generated/torch.triu.html) with `diagonal=1` keeps the strictly upper triangle). For three tokens:
$$
M=\begin{bmatrix}0&-\infty&-\infty\\0&0&-\infty\\0&0&0\end{bmatrix}.
$$
- The **same** `z_enc` goes to every decoder layer: each layer does its own cross-attention on the final encoder output.
- With a single token (the first step of inference, just `<SOS>`) there is nothing to mask, so the mask is left out.
- `fc_predict` returns **logits**; the softmax lives inside [`nn.CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) during training and is replaced by `topk(1)` at inference.

The wrapping `Transformer` class is Lecture 20's, unchanged: with a target it shifts the target right and makes one parallel pass ([[7-transformers/sequence-to-sequence-rnns#teacher-forcing-and-lining-up-the-loss|teacher forcing]]); without one it generates greedily, one token at a time, until `<EOS>` or 25 tokens.

## Why an nn.ModuleList

An `nn.Module` finds its parameters by looking at the attributes assigned to it: tensors wrapped in `nn.Parameter`, and sub-modules. A Python `list` is neither, so layers stored in one are never registered. The model would still run, but `model.parameters()` would not return those layers' weights: the optimizer would never update them, and `model.to(DEVICE)` would not move them. [`nn.ModuleList`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) is a list that registers everything put in it (see [[4-pytorch-training/nn-module#stacking-layers-nnsequential-and-nnmodulelist|nn.Sequential and nn.ModuleList]]).

```python
self.layers = [nn.TransformerEncoderLayer(128, 4) for _ in range(6)]                 # wrong
self.layers = nn.ModuleList([nn.TransformerEncoderLayer(128, 4) for _ in range(6)])  # right
```

Note the list comprehension: it builds `n_layers` **separate** layers. Writing `[layer] * n_layers` would repeat one object, so all "layers" would share the same weights.

## Counting parameters

Write $E$ for `d_model` and $F$ for `dim_feedforward`. Each piece is a layer you already know how to count ([[5-deep-networks/multilayer-perceptrons#counting-parameters|linear layers]], [[7-transformers/multi-head-attention#in-pytorch|multi-head attention]], [[7-transformers/layer-norm-and-feed-forward#layer-norm-versus-batch-norm|layer norm]]):

| Piece | Parameters | $E = 128$, $F = 256$ |
| --- | --- | --- |
| multi-head attention: Q, K, V projections ($3E^2 + 3E$) and output projection ($E^2 + E$) | $4E^2 + 4E$ | 66,048 |
| feed-forward: `Linear(E, F)` and `Linear(F, E)` | $2EF + F + E$ | 65,920 |
| one layer norm (scale and shift) | $2E$ | 256 |
| **encoder layer** = attention + feed-forward + 2 norms | $4E^2 + 2EF + 9E + F$ | **132,480** |
| **decoder layer** = 2 attentions + feed-forward + 3 norms | $8E^2 + 2EF + 15E + F$ | **198,784** |

With the lecture's $F = 2E$ these are $8E^2 + 11E$ and $12E^2 + 17E$. The whole one-layer translation model of the lecture (English vocabulary 1,443, French 1,646, $E = 128$):

| Part | Parameters |
| --- | --- |
| encoder: embedding $1{,}443 \cdot 128$ + positions $1{,}000 \cdot 128$ + 1 layer | $184{,}704 + 128{,}000 + 132{,}480 = 445{,}184$ |
| decoder: embedding $1{,}646 \cdot 128$ + positions + 1 layer + `fc_predict` ($128 \cdot 1{,}646 + 1{,}646$) | $210{,}688 + 128{,}000 + 198{,}784 + 212{,}334 = 749{,}806$ |
| **total** | **1,194,990** |

Each extra layer adds 132,480 (encoder) or 198,784 (decoder); two layers each give 1,526,254. These counts are derived from the layer definitions in the PyTorch docs (recomputed in `tools/verify_mt2.py`), not printed by the notebook.

## How it differs from the Lecture 20 layers

| | Lecture 20, hand-built | Lecture 21, PyTorch layers |
| --- | --- | --- |
| query/key width | separate hidden size $H = 64$, then per-head 16 | `d_model / nhead` = 32 per head |
| extra projections | `fc_Q`, `fc_K`, `fc_V` before the heads ([[7-transformers/multi-head-attention#the-course-implementation|redundant]]) | none |
| dropout | none | inside every sub-block |
| depth | one layer each | `n_layers`, a constructor argument |
| batches | one sentence at a time | supported (`batch_first`), used by the classifiers |
| positional encoding | sinusoidal or learned (the ablation) | learned, 1,000 rows |

The mathematics is the same: [[7-transformers/attention|scaled dot-product attention]] in several heads, a [[7-transformers/layer-norm-and-feed-forward|two-layer feed-forward network]], residual connections and layer norm.

## Training it

The lecture reruns English→French translation ([[7-transformers/sequence-to-sequence-rnns#the-englishfrench-data|same data]], `min_freq = 50`):

```
18878 training examples
18878 validation examples
English vocabulary has 1443 words.
French vocabulary has 1646 words.
```

```python
d_emb, n_layers, n_head, dropout = 128, 1, 4, 0.0
encoder = TransformerEncoder(num_embeddings_enc, d_emb, n_layers, n_head, dropout=dropout)
decoder = TransformerDecoder(num_embeddings_dec, d_emb, n_layers, n_head, dropout=dropout)
transformer_model = Transformer(encoder, decoder).to(DEVICE)
optimizer = torch.optim.SGD(transformer_model.parameters(), lr=1e-2, momentum=0.99)
```

Sentences still go through the model one at a time. A "batch" of 64 is simulated by adding up 64 per-sentence losses (each divided by 64) and calling `backward()` once, which averages the gradient exactly as a real batch would ([[4-pytorch-training/training-loop#gradient-accumulation|gradient accumulation]]).

Five epochs took about 14 minutes on the notebook's GPU. Read off the notebook's plots, the loss falls from about 4.4 to 1.3, training BLEU reaches about 0.32 and validation BLEU about 0.24, all still improving. That is a check that the rebuilt model learns, not a finished run. Lecture 22 trains the same model for 100 epochs with a learning-rate schedule and reaches validation BLEU above 0.5: see [[4-pytorch-training/tensorboard-and-logging#reading-the-lectures-six-runs|the six logged runs]].

## One level higher

PyTorch also has containers that do the stacking loop for you (not used in the lecture):

```python
layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, batch_first=True)
encoder = nn.TransformerEncoder(layer, num_layers=6)   # deep-copies the layer 6 times
z = encoder(torch.randn(8, 20, 128))                   # (batch, tokens, d_model)
```

[`nn.TransformerEncoder`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html), [`nn.TransformerDecoder`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html) and the complete [`nn.Transformer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html) still leave embeddings, positions and the output layer to you.

> [!trap] Common mistakes
> - **Layers in a plain list.** No error, but those layers never train.
> - **Forgetting `batch_first`.** The default is `(tokens, batch, d_model)`. Feeding `(batch, tokens, d_model)` to a default layer silently mixes examples in a batch.
> - **`d_model` not divisible by `nhead`.** PyTorch raises an error (the hand-built Lecture 20 code silently dropped dimensions).
> - **Expecting positions to be handled.** Without your own [[7-transformers/positional-encoding|positional encoding]] the model cannot tell word order.
> - **Leaving the model in training mode for evaluation.** The layers contain dropout; call `model.eval()` first, and `model.train()` afterwards.
> - **Passing probabilities to the loss.** `fc_predict` outputs logits, which is what `nn.CrossEntropyLoss` expects.

> [!exam] On the exams
> [[9-midterm-2/spring-2026#problem-7--attention-is-all-you-need-15-pts|Spring 2026 Midterm 2, Problem 7]] asks you to write one encoder layer from a given attention module, `nn.LayerNorm` and a feed-forward layer: exactly the contents of the dashed box above. [[9-midterm-2/fall-2025#problem-1--true-or-false-30-pts|Fall 2025 Problem 1(f)]] tests that Add & Norm uses layer norm, not batch norm. Drill: [[9-midterm-2/review-attention-and-transformers|attention and Transformers review]].

> [!question]- Check yourself
> 1. How many parameters does `nn.TransformerEncoderLayer(64, 4, 128)` have? Does the answer change with 8 heads?
> 2. A 6-layer encoder and 6-layer decoder with the lecture's settings ($E = 128$, $F = 256$, vocabularies 1,443 and 1,646): how many parameters in total?
> 3. Why does the decoder pass `z_enc` to every layer instead of only the first?
>
> > [!solution]- Answers
> > 1. $8\cdot64^2 + 11\cdot64 = 32{,}768 + 704 = 33{,}472$. No: the heads split the same projection matrices.
> > 2. Encoder $184{,}704 + 128{,}000 + 6\cdot132{,}480 = 1{,}107{,}584$; decoder $210{,}688 + 128{,}000 + 6\cdot198{,}784 + 212{,}334 = 1{,}743{,}726$; total 2,851,310.
> > 3. Every decoder layer has its own cross-attention block, and each one needs the source sentence as keys and values. Only the queries (the decoder's own features) change from layer to layer.

## PyTorch docs

- [`torch.nn.TransformerEncoderLayer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) — arguments and the `forward(src, src_mask, src_key_padding_mask)` signature.
- [`torch.nn.TransformerDecoderLayer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html) — `forward(tgt, memory, tgt_mask, ..., tgt_is_causal)`.
- [`torch.nn.TransformerEncoder`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html), [`torch.nn.TransformerDecoder`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html), [`torch.nn.Transformer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html) — the stacking containers; `nn.Transformer.generate_square_subsequent_mask` builds the causal mask.
- [`torch.nn.ModuleList`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) — a list whose modules are registered.
- [`torch.nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) — the attention block inside both layers.
- [`torch.triu`](https://docs.pytorch.org/docs/stable/generated/torch.triu.html) — the upper triangle used for the mask.

Related: [[7-transformers/transformer-architecture]] · [[7-transformers/text-classification-with-transformers]] · [[7-transformers/vision-transformer]] · [[7-transformers/multi-head-attention]] · [[7-transformers/attention-is-all-you-need]]
