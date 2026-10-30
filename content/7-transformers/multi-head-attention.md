---
title: "Multi-head attention"
description: "Running h smaller attention heads in parallel instead of one big one: the definition from the paper, per-head dimensions, concatenation and the output projection W_O, why several heads help (Table 3 of the paper), why the cost equals one full-width head, a shape trace and parameter count for the course's 4-head block, the course implementation and its redundant projection, and PyTorch's nn.MultiheadAttention and scaled_dot_product_attention."
tags: [topic, transformers]
aliases: ["multi-head", "multihead attention", "attention heads", "W_O", "nn.MultiheadAttention"]
---

*Lecture 20 · [[7-transformers/index|Transformers]] · builds on [[7-transformers/attention|attention]] · part of the [[7-transformers/transformer-architecture|Full Transformer]]*

> [!abstract] In one breath
> One attention head compares tokens through one fixed pair of query/key projections, so it can express one notion of "relevant". **Multi-head attention** runs $h$ heads side by side, each with its own projections into a smaller space ($d_k/h$ for queries and keys, $d_v/h$ for values, with $d_k$, $d_v$ the full widths), concatenates their outputs back to full width and mixes them with one more linear map $W_O$. Because each head is $h$ times narrower, the parameter count and compute match a single full-width head. The paper uses 8 heads of size 64; the course's Full Transformer uses 4 heads with 16-dimensional queries and keys and 32-dimensional values.

## What it is

> [!definition] Definition — multi-head attention (Vaswani et al., 2017, §3.2.2)
> $$
> \mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^O,\qquad \mathrm{head}_j=\mathrm{Attention}\bigl(QW_j^Q,\;KW_j^K,\;VW_j^V\bigr)
> $$
> with $W_j^Q, W_j^K\in\mathbb R^{d_{model}\times d_k}$, $W_j^V\in\mathbb R^{d_{model}\times d_v}$, $W^O\in\mathbb R^{hd_v\times d_{model}}$, and $\mathrm{Attention}$ the [[7-transformers/attention|scaled dot-product attention]] $\mathrm{softmax}\bigl(QK^\top/\sqrt{d_k}\bigr)V$, computed separately in every head. The paper takes $h = 8$ and $d_k = d_v = d_{model}/h = 64$.

![[l20-multi-head-attention.png|Multi-head attention (Vaswani et al., 2017, Figure 2): h parallel sets of linear maps and scaled dot-product attention, concatenated, then one more linear layer]]

Every head sees the same tokens but compares them through its own learned projections, so one head can line up a verb with its object while another tracks the previous word. The lecture's caveat applies: heads "do not in practice provide such clean and informative representations", but having several gives the layer room to represent several relations at once. In the paper's words, multi-head attention "allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this": a single softmax-weighted average has to blend everything into one mixture per token.

## What you should know

- The formula, and that the scaling uses the **per-head** key size (16 in the course, 64 in the paper).
- Per-head sizes (lecture notation: $d_k$, $d_v$ = **total** widths; the paper's $d_k$, $d_v$ are per head): queries and keys $d_k/h$, values $d_v/h$; concatenating $h$ outputs of size $d_v/h$ restores width $d_v$; $W_O$ maps it back to $d_{model}$.
- Splitting into $h$ heads keeps the parameter count and the matrix-multiply cost the same as one head of full width; only the number of softmaxes ($h$ attention matrices of size $n\times m$) grows.
- Shapes for the course's block (embedding 128, hidden 64, 4 heads): each head $(N, 16)$ queries, $(M, 16)$ keys, $(M, 32)$ values → $(N, 32)$ output; concatenated $(N, 128)$; after $W_O$ $(N, 128)$. 41,344 parameters.
- Paper evidence (Table 3, rows A): one head is 0.9 BLEU worse than the best setting; too many heads (32 of size 16) also hurts.
- The causal mask is applied identically inside every head.
- PyTorch's [`nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) does the per-head projections with one big matrix and needs `embed_dim % num_heads == 0`.

## In the course's Full Transformer

<!-- fig:tf-b-mha-shapes -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 300" width="720" height="300" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>Q, K and V fanning out to four attention heads, concatenated and projected by W_O</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><rect x="20" y="30" width="110" height="40" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.5"/><text x="75" y="42" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">Q</text><text x="75" y="59" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">(N, 64)</text><rect x="20" y="130" width="110" height="40" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.5"/><text x="75" y="142" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">K</text><text x="75" y="159" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">(M, 64)</text><rect x="20" y="230" width="110" height="40" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.5"/><text x="75" y="242" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">V</text><text x="75" y="259" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">(M, 128)</text><rect x="200" y="22" width="230" height="52" rx="6" fill="var(--paper)" stroke="var(--accent)" stroke-width="1.5"/><text x="315" y="40" text-anchor="middle" dominant-baseline="central" fill="var(--accent)" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">head 1: Linear 64→16, 64→16,</text><text x="315" y="57" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">128→32; softmax(QKᵀ/√16)V</text><line x1="132" y1="50" x2="198" y2="42" stroke="var(--muted)" stroke-width="1" stroke-linecap="round"/><path d="M198.0,42.0 L191.5,46.3 L190.6,39.4 Z" fill="var(--muted)"/><line x1="132" y1="150" x2="198" y2="48" stroke="var(--muted)" stroke-width="1" stroke-linecap="round"/><path d="M198.0,48.0 L197.1,55.8 L191.3,52.0 Z" fill="var(--muted)"/><line x1="132" y1="250" x2="198" y2="54" stroke="var(--muted)" stroke-width="1" stroke-linecap="round"/><path d="M198.0,54.0 L199.1,61.8 L192.4,59.5 Z" fill="var(--muted)"/><line x1="432" y1="48" x2="488" y2="150" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round"/><path d="M488.0,150.0 L481.6,145.5 L487.7,142.2 Z" fill="var(--accent)"/><rect x="200" y="88" width="230" height="52" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.5"/><text x="315" y="106" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">head 2: Linear 64→16, 64→16,</text><text x="315" y="123" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">128→32; softmax(QKᵀ/√16)V</text><line x1="132" y1="50" x2="198" y2="108" stroke="var(--muted)" stroke-width="1" stroke-linecap="round"/><path d="M198.0,108.0 L190.4,106.0 L195.1,100.8 Z" fill="var(--muted)"/><line x1="132" y1="150" x2="198" y2="114" stroke="var(--muted)" stroke-width="1" stroke-linecap="round"/><path d="M198.0,114.0 L193.5,120.4 L190.2,114.3 Z" fill="var(--muted)"/><line x1="132" y1="250" x2="198" y2="120" stroke="var(--muted)" stroke-width="1" stroke-linecap="round"/><path d="M198.0,120.0 L198.0,127.8 L191.7,124.7 Z" fill="var(--muted)"/><line x1="432" y1="114" x2="488" y2="150" stroke="var(--muted)" stroke-width="1.2" stroke-linecap="round"/><path d="M488.0,150.0 L480.2,149.2 L484.0,143.3 Z" fill="var(--muted)"/><rect x="200" y="154" width="230" height="52" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.5"/><text x="315" y="172" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">head 3: Linear 64→16, 64→16,</text><text x="315" y="189" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">128→32; softmax(QKᵀ/√16)V</text><line x1="132" y1="50" x2="198" y2="174" stroke="var(--muted)" stroke-width="1" stroke-linecap="round"/><path d="M198.0,174.0 L191.6,169.5 L197.8,166.2 Z" fill="var(--muted)"/><line x1="132" y1="150" x2="198" y2="180" stroke="var(--muted)" stroke-width="1" stroke-linecap="round"/><path d="M198.0,180.0 L190.2,180.3 L193.1,173.9 Z" fill="var(--muted)"/><line x1="132" y1="250" x2="198" y2="186" stroke="var(--muted)" stroke-width="1" stroke-linecap="round"/><path d="M198.0,186.0 L195.4,193.4 L190.5,188.4 Z" fill="var(--muted)"/><line x1="432" y1="180" x2="488" y2="150" stroke="var(--muted)" stroke-width="1.2" stroke-linecap="round"/><path d="M488.0,150.0 L483.5,156.4 L480.2,150.2 Z" fill="var(--muted)"/><rect x="200" y="220" width="230" height="52" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.5"/><text x="315" y="238" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">head 4: Linear 64→16, 64→16,</text><text x="315" y="255" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">128→32; softmax(QKᵀ/√16)V</text><line x1="132" y1="50" x2="198" y2="240" stroke="var(--muted)" stroke-width="1" stroke-linecap="round"/><path d="M198.0,240.0 L192.4,234.5 L199.0,232.2 Z" fill="var(--muted)"/><line x1="132" y1="150" x2="198" y2="246" stroke="var(--muted)" stroke-width="1" stroke-linecap="round"/><path d="M198.0,246.0 L191.2,242.2 L196.9,238.2 Z" fill="var(--muted)"/><line x1="132" y1="250" x2="198" y2="252" stroke="var(--muted)" stroke-width="1" stroke-linecap="round"/><path d="M198.0,252.0 L190.9,255.3 L191.1,248.3 Z" fill="var(--muted)"/><line x1="432" y1="246" x2="488" y2="150" stroke="var(--muted)" stroke-width="1.2" stroke-linecap="round"/><path d="M488.0,150.0 L487.5,157.8 L481.4,154.3 Z" fill="var(--muted)"/><text x="315" y="290" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">each head output: (N, 32)</text><rect x="492" y="128" width="90" height="44" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.5"/><text x="537" y="142" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">concat</text><text x="537" y="159" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">(N, 128)</text><line x1="584" y1="150" x2="610" y2="150" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M610.0,150.0 L603.0,153.5 L603.0,146.5 Z" fill="currentColor"/><rect x="614" y="128" width="92" height="44" rx="6" fill="var(--paper)" stroke="var(--accent)" stroke-width="1.5"/><text x="660" y="142" text-anchor="middle" dominant-baseline="central" fill="var(--accent)" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">W_O</text><text x="660" y="159" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">128→128</text><text x="660" y="196" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">output (N, 128)</text></svg><figcaption><b>Shapes in the course's <code>MultiheadAttention</code></b> (embedding 128, hidden 64, 4 heads). Each head projects the already-projected queries and keys from 64 to 16 dimensions and the values from 128 to 32, runs scaled dot-product attention with √16, and returns (<i>N</i>, 32). Concatenating 4 heads gives back (<i>N</i>, 128), and <code>output_linear</code> (<i>W</i><sub>O</sub>) mixes them. For self-attention <i>M</i> = <i>N</i>; for cross-attention <i>M</i> is the source length.</figcaption></figure>

With $d_{emb} = 128$, $H = 64$ (the course's query/key width) and $h = 4$:

| Step | Shape | Parameters |
| --- | --- | --- |
| input queries `Q` (after `fc_Q`) | $(N, 64)$ | — |
| input keys `K`, values `V` | $(M, 64)$, $(M, 128)$ | — |
| per head: `Linear(64, 16)` on Q and on K | $(N, 16)$, $(M, 16)$ | $2\cdot(64\cdot16+16) = 2{,}080$ |
| per head: `Linear(128, 32)` on V | $(M, 32)$ | $128\cdot32+32 = 4{,}128$ |
| per head: $\mathrm{softmax}(QK^\top/\sqrt{16})V$ | scores $(N, M)$, output $(N, 32)$ | — |
| concatenate 4 heads | $(N, 128)$ | — |
| `output_linear` ($W_O$): `Linear(128, 128)` | $(N, 128)$ | 16,512 |
| **total** | | $4\cdot(2{,}080+4{,}128) + 16{,}512 = 41{,}344$ |

For the encoder's self-attention $M = N$ (the source length); for the decoder's cross-attention $N$ is the target length and $M$ the source length. The Full Transformer has three of these blocks (one in the encoder, masked and cross in the decoder): 124,032 parameters.

### Same cost as one head

Run the same counting with $h = 1$: `Linear(64, 64)` for queries and keys, `Linear(128, 128)` for values, `Linear(128, 128)` for $W_O$ — again 41,344 parameters. Splitting a width-$D$ projection into $h$ projections of width $D/h$ just partitions the same weight matrix into column blocks; the bias vectors partition the same way. The attention products also cost the same in total: $h$ heads each do $N\cdot M\cdot(d_k/h)$ multiply–adds for $QK^\top$, i.e. $NMd_k$ overall. The paper: "Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality."

### The course implementation

```python
class MultiheadAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_heads):
        super(MultiheadAttention, self).__init__()
        multi_embedding_dim = embedding_dim//n_heads   # value size per head
        multi_hidden_dim = hidden_dim//n_heads         # query/key size per head
        self.n_heads = n_heads
        self.Q_layers = nn.ModuleList([nn.Linear(hidden_dim, multi_hidden_dim) for i in range(n_heads)])
        self.K_layers = nn.ModuleList([nn.Linear(hidden_dim, multi_hidden_dim) for i in range(n_heads)])
        self.V_layers = nn.ModuleList([nn.Linear(embedding_dim, multi_embedding_dim) for i in range(n_heads)])
        self.attn_head = ScaledDotProductAttention()
        self.output_linear = nn.Linear(n_heads*multi_embedding_dim, embedding_dim)

    def forward(self, Q, K, V, mask=None):
        self_attn_results = torch.cat([self.attn_head(self.Q_layers[i](Q),
                                                      self.K_layers[i](K),
                                                      self.V_layers[i](V),
                                                      mask=mask) for i in range(self.n_heads)], dim=1)
        multi_attn_output = self.output_linear(self_attn_results)   # W_O
        return multi_attn_output
```

The heads run in a Python loop over an `nn.ModuleList` and are concatenated along `dim=1`, the feature dimension of a 2-D `(n_tokens, d)` tensor (the course processes one sentence at a time, with no batch dimension). `ScaledDotProductAttention` divides by `np.sqrt(K.size(-1))`, which is the per-head 16. The same `mask` goes to every head.

The encoder calls it as `self.multi_head_attention(self.fc_Q(z), self.fc_K(z), self.fc_V(z))`: queries already projected by `fc_Q` (`Linear(128, 64)`) are projected **again** by each head's `Linear(64, 16)`. Two linear maps in a row are one linear map,

$$
(zW_1+b_1)W_2+b_2 = z\,(W_1W_2) + (b_1W_2+b_2),
$$

so the extra layer adds parameters (`fc_Q`, `fc_K`, `fc_V` are 33,024 per block) but no expressive power; the paper projects the $d_{model}$-dimensional input directly inside each head. It is harmless, just redundant.

> [!note] Typo in the lecture
> The notebook writes $\mathrm{Concatenate}(\mathrm{Attention}_1, \mathrm{Attention}_2, \ldots, \mathrm{Attention}_2)W_O$; the last term should be $\mathrm{Attention}_h$.

## Do more heads help?

The paper varied $h$ on English→German (development set, base model, compute held fixed by shrinking $d_k = d_v = 512/h$):

| Heads $h$ | $d_k = d_v$ | BLEU (dev) |
| --- | --- | --- |
| 1 | 512 | 24.9 |
| 4 | 128 | 25.5 |
| **8** (base) | 64 | **25.8** |
| 16 | 32 | 25.8 |
| 32 | 16 | 25.4 |

A single head is 0.9 BLEU worse than the best; very small heads lose again, which matches the paper's other observation (Table 3, rows B) that shrinking $d_k$ hurts — comparing tokens needs enough dimensions per head. The course does not isolate the effect of heads: its [[7-transformers/transformer-architecture#results|Medium → Full]] step adds multiple heads **and** positional encoding together.

## In PyTorch

[`nn.MultiheadAttention(embed_dim, num_heads)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) implements the paper's block, with $d_k = d_v = $ `embed_dim // num_heads`:

```python
mha = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
x = torch.randn(2, 10, 128)                      # (batch, tokens, features)
causal = nn.Transformer.generate_square_subsequent_mask(10)
out, weights = mha(x, x, x, attn_mask=causal)    # self-attention
# out (2,10,128); weights (2,10,10), head-averaged
```

(Not lecture code.) Things to know:

- `batch_first=False` is the **default**: inputs are then `(tokens, batch, features)`.
- It projects Q, K, V internally (`in_proj_weight` of shape $(3E, E)$ plus `out_proj`): $4E^2 + 4E$ parameters, 66,048 for $E = 128$ and 1,050,624 for the paper's $E = 512$.
- `kdim` / `vdim` allow keys and values of a different width (cross-attention from another model).
- `need_weights=True` (default) also returns the attention weights, averaged over heads unless `average_attn_weights=False`.
- A float `attn_mask` is **added** to the scores (use $-\infty$ to block); a boolean mask uses `True` for **not allowed**. `key_padding_mask` hides padding tokens.

[`F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) is the fused single-head computation; it accepts extra leading dimensions, so heads can be a dimension: `q` of shape `(batch, heads, tokens, d_k)`.

> [!trap] Common mistakes
> - **Scaling by the full width.** Each head divides by the square root of its own key size ($\sqrt{16}$ in the course, $\sqrt{64}$ in the paper), not of the total query/key width (64 in the course).
> - **Concatenating along the wrong axis.** Heads are joined along the **feature** dimension; the token count stays $N$.
> - **Width not divisible by $h$.** The course's `//` silently drops dimensions: 128 with 3 heads gives value heads of 42, a 126-wide concatenation, and an `output_linear` of `Linear(126, 128)`. `nn.MultiheadAttention` raises an error instead.
> - **Expecting more heads to cost more parameters.** At fixed width the count does not change; the attention maps ($h$ of them) are what grows.
> - **Forgetting `batch_first`.** `nn.MultiheadAttention` defaults to `(L, N, E)`.

> [!question]- Check yourself
> 1. The paper's base model: what are the shapes of $W_j^Q$, $W^O$, and of one head's output for a 20-token sentence?
> 2. Count the parameters of the course's `MultiheadAttention(embedding_dim=256, hidden_dim=128, n_heads=8)`.
> 3. Why can't attention heads be merged into one head the way the stacked `fc_Q` and head `Linear` can be merged?
>
> > [!solution]- Answers
> > 1. $W_j^Q\in\mathbb R^{512\times64}$, $W^O\in\mathbb R^{512\times512}$ ($hd_v = 8\cdot64$), one head's output $20\times64$.
> > 2. Per head: Q and K `Linear(128, 16)` = 2,064 each, V `Linear(256, 32)` = 8,224; ×8 = 98,816; plus $W_O$ `Linear(256, 256)` = 65,792; total 164,608.
> > 3. Each head has its **own softmax**: the output is a sum of $h$ differently weighted averages, which a single softmax over one score matrix cannot reproduce in general. The two linear maps had no nonlinearity between them.

## PyTorch docs

- [`torch.nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) — `embed_dim`, `num_heads`, `batch_first`, `kdim`/`vdim`, `need_weights`, `attn_mask`, `key_padding_mask`.
- [`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — fused attention with `is_causal`.
- [`torch.nn.Transformer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html) — `generate_square_subsequent_mask` builds the causal mask.
- [`torch.nn.ModuleList`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) — how the course stores one `Linear` per head; [`torch.cat`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html) — the concatenation.

Related: [[7-transformers/attention]] · [[7-transformers/transformer-architecture]] · [[7-transformers/positional-encoding]] · [[7-transformers/attention-is-all-you-need]]
