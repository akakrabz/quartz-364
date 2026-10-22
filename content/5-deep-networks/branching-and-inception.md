---
title: "Branching and Inception blocks"
description: "Networks as graphs: forward() can split and merge paths; torch.cat needs the same H and W, addition needs the same C, H and W; the Inception block and its 1×1 bottlenecks (the GoogLeNet trick), DenseNet-style and multi-scale connections, two-head models, and how to fix shape mismatches."
tags: [topic, deep-nets]
aliases: ["Inception block", "Inception", "GoogLeNet trick", "1x1 convolution", "1×1 convolution", "bottleneck", "DenseNet", "two-head network", "multi-head network", "merging branches"]
---

*Lecture 16 · [[5-deep-networks/index|Deep networks]] · builds on [[5-deep-networks/convolutional-layers|convolutional layers]] and [[4-pytorch-training/nn-module|nn.Module]]*

> [!abstract] In one breath
> `forward` is ordinary Python, so a model does not have to be a chain: data can split into several paths and merge again, and the model becomes a graph. Paths merge in one of two ways: `torch.cat([a, b], dim=1)` stacks channels and needs the same **H and W**; `a + b` adds element-wise and needs the same **C, H and W**. The Inception block runs 1×1, 3×3, 5×5 convolutions and a max pool side by side on one input and concatenates the results; putting a cheap 1×1 convolution before each big kernel to cut the channels is what keeps it affordable.

## What it is

Until Lecture 15 every network was a single chain, written one of three ways: layers as attributes called one by one in `forward`, an [`nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html), or a function that returns a block and is stacked. `nn.Sequential` can only run its layers one after another. A model that **branches** needs your own `forward` (it can still use `nn.Sequential` blocks inside):

```python
def forward(self, x):
    a = self.path_a(x)        # both paths see x
    b = self.path_b(x)
    return torch.cat([a, b], dim=1)   # merge: stack channels
```

Autograd does not care about the shape of the graph: [[2-calculus-and-optimization/backpropagation|backpropagation]] sums the gradients that arrive at `x` from both paths, exactly as at any node with two children.

## What you should know

- The two merge rules (table below), and that `dim=1` is the channel axis of an $N\times C\times H\times W$ tensor.
- A $k\times k$ convolution with stride 1 keeps H and W when the padding is $p = (k-1)/2$: $p=0$ for $1\times1$, $1$ for $3\times3$, $2$ for $5\times5$. In general $H_\text{out} = \left\lfloor\frac{H_\text{in}+2p-k}{s}\right\rfloor + 1$.
- Parameters of `nn.Conv2d(C_in, C_out, k)`: $C_\text{out}C_\text{in}k^2 + C_\text{out}$. A $1\times1$ convolution costs $C_\text{out}C_\text{in} + C_\text{out}$.
- A $1\times 1$ convolution is a fully connected layer applied at every pixel: it mixes channels and changes their number, leaving H and W alone.
- The Inception block's output has $\sum$ (branch channels) channels; with 4 branches of 32, that is 128.
- The **GoogLeNet trick**: reduce channels with a $1\times1$ convolution before a $3\times3$ or $5\times5$ kernel; Lecture 16's 5×5 branch drops from 51,232 to 27,712 parameters.
- Fixing mismatches: channels with a $1\times1$ convolution; size with the right padding, or by resizing (upsample / pool).
- A two-head model shares a trunk and adds the losses of its heads.

## Merge rules

| merge | use it when | must match |
|---|---|---|
| `torch.cat([a, b], dim=1)` | both paths keep their own features side by side (Inception, DenseNet, [[5-deep-networks/fully-convolutional-networks\|U-Net]]) | **H** and **W**; channels may differ, they add up |
| `a + b` | one path refines the other ([[5-deep-networks/residual-connections\|ResNet]]) | **C**, **H** and **W** |

See [`torch.cat`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html) and [[1-foundations/tensors|tensors]] for `cat` vs `stack`.

## The Inception block

Look at the same input with several kernel sizes at once, then stack what each saw.

<!-- fig:deepb-inception -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 420" width="760" height="420" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>Inception block: four parallel branches of 1x1, 3x3, 5x5 convolutions and max pooling, concatenated</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><rect x="270" y="13" width="230" height="30" rx="6" fill="var(--muted)" fill-opacity="0.12" stroke="var(--muted)" stroke-width="1.4"/><text x="385" y="28" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">input: 64 × H × W</text><line x1="385" y1="43" x2="385" y2="62" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round"/><line x1="105" y1="62" x2="665" y2="62" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round"/><line x1="105" y1="62" x2="105" y2="81" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="19" y="81" width="172" height="28" rx="6" fill="var(--accent2)" fill-opacity="0.12" stroke="var(--accent2)" stroke-width="1.4"/><text x="105" y="95" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">1×1 conv, 64 → 32</text><line x1="105" y1="109" x2="105" y2="230" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><text x="105" y="243" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">32 × H × W</text><text x="105" y="261" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">branch1 · 2,080 params</text><line x1="105" y1="272" x2="105" y2="292" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round"/><line x1="285" y1="62" x2="285" y2="81" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="199" y="81" width="172" height="28" rx="6" fill="var(--accent2)" fill-opacity="0.12" stroke="var(--accent2)" stroke-width="1.4"/><text x="285" y="95" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">1×1 conv, 64 → 32</text><rect x="250" y="127" width="70" height="28" rx="6" fill="var(--muted)" fill-opacity="0.12" stroke="var(--muted)" stroke-width="1.4"/><text x="285" y="141" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">ReLU</text><line x1="285" y1="109" x2="285" y2="127" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="199" y="173" width="172" height="28" rx="6" fill="var(--accent)" fill-opacity="0.12" stroke="var(--accent)" stroke-width="1.4"/><text x="285" y="187" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">3×3 conv, pad 1, 32 → 32</text><line x1="285" y1="155" x2="285" y2="173" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><line x1="285" y1="201" x2="285" y2="230" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><text x="285" y="243" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">32 × H × W</text><text x="285" y="261" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">branch2 · 11,328 params</text><line x1="285" y1="272" x2="285" y2="292" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round"/><line x1="475" y1="62" x2="475" y2="81" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="389" y="81" width="172" height="28" rx="6" fill="var(--accent2)" fill-opacity="0.12" stroke="var(--accent2)" stroke-width="1.4"/><text x="475" y="95" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">1×1 conv, 64 → 32</text><rect x="440" y="127" width="70" height="28" rx="6" fill="var(--muted)" fill-opacity="0.12" stroke="var(--muted)" stroke-width="1.4"/><text x="475" y="141" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">ReLU</text><line x1="475" y1="109" x2="475" y2="127" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="389" y="173" width="172" height="28" rx="6" fill="var(--accent)" fill-opacity="0.12" stroke="var(--accent)" stroke-width="1.4"/><text x="475" y="187" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">5×5 conv, pad 2, 32 → 32</text><line x1="475" y1="155" x2="475" y2="173" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><line x1="475" y1="201" x2="475" y2="230" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><text x="475" y="243" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">32 × H × W</text><text x="475" y="261" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">branch3 · 27,712 params</text><line x1="475" y1="272" x2="475" y2="292" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round"/><line x1="665" y1="62" x2="665" y2="81" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="579" y="81" width="172" height="28" rx="6" fill="var(--hi)" fill-opacity="0.12" stroke="var(--hi)" stroke-width="1.4"/><text x="665" y="95" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">3×3 max pool, stride 1, pad 1</text><rect x="579" y="127" width="172" height="28" rx="6" fill="var(--accent2)" fill-opacity="0.12" stroke="var(--accent2)" stroke-width="1.4"/><text x="665" y="141" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">1×1 conv, 64 → 32</text><line x1="665" y1="109" x2="665" y2="127" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><line x1="665" y1="155" x2="665" y2="230" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><text x="665" y="243" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">32 × H × W</text><text x="665" y="261" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">branch4 · 2,080 params</text><line x1="665" y1="272" x2="665" y2="292" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round"/><line x1="105" y1="292" x2="665" y2="292" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round"/><line x1="385" y1="292" x2="385" y2="311" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="220" y="312" width="330" height="30" rx="6" fill="var(--muted)" fill-opacity="0.12" stroke="var(--muted)" stroke-width="1.4"/><text x="385" y="327" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">torch.cat(branches, dim=1): 128 × H × W</text><line x1="385" y1="342" x2="385" y2="362" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="345" y="363" width="80" height="28" rx="6" fill="var(--muted)" fill-opacity="0.12" stroke="var(--muted)" stroke-width="1.4"/><text x="385" y="377" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">ReLU</text><rect x="560" y="352" width="14" height="12" rx="2" fill="var(--accent2)" fill-opacity="0.25" stroke="var(--accent2)" stroke-width="1.4"/><text x="582" y="358" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">1×1 conv (mixes channels)</text><rect x="560" y="372" width="14" height="12" rx="2" fill="var(--accent)" fill-opacity="0.25" stroke="var(--accent)" stroke-width="1.4"/><text x="582" y="378" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">3×3 / 5×5 conv</text><rect x="560" y="392" width="14" height="12" rx="2" fill="var(--hi)" fill-opacity="0.25" stroke="var(--hi)" stroke-width="1.4"/><text x="582" y="398" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">max pool</text></svg><figcaption><strong>The Inception block from Lecture 16</strong> (64 input channels, 32 per branch). Every branch keeps H × W, so <code>torch.cat</code> along the channels works: 4 × 32 = 128 channels. Total 43,200 parameters; one 5×5 convolution from 64 to 128 channels would need 204,928.</figcaption></figure>

```python
class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_branch):
        super().__init__()
        self.branch1 = nn.Conv2d(c_in, c_branch, kernel_size=1)
        self.branch2 = nn.Sequential(nn.Conv2d(c_in, c_branch, 1), nn.ReLU(),
                                     nn.Conv2d(c_branch, c_branch, 3, padding=1))
        self.branch3 = nn.Sequential(nn.Conv2d(c_in, c_branch, 1), nn.ReLU(),
                                     nn.Conv2d(c_branch, c_branch, 5, padding=2))
        self.branch4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1),
                                     nn.Conv2d(c_in, c_branch, 1))

    def forward(self, x):
        branches = [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)]
        return torch.relu(torch.cat(branches, dim=1))   # stack along channels

block = InceptionBlock(c_in=64, c_branch=32)
x = torch.rand(1, 64, 16, 16)
print('input', tuple(x.shape), '-> output', tuple(block(x).shape))
for name, branch in block.named_children():
    print('  {:12s} {:>7,d} parameters'.format(name, n_params(branch)))
print('  {:12s} {:>7,d} parameters'.format('total', n_params(block)))
print('compare: one 5x5 conv 64 -> 128 channels has {:,d} parameters'.format(
    n_params(nn.Conv2d(64, 128, 5, padding=2))))
```

(`n_params(m)` is `sum(p.numel() for p in m.parameters())`.)

```
input (1, 64, 16, 16) -> output (1, 128, 16, 16)
  branch1        2,080 parameters
  branch2       11,328 parameters
  branch3       27,712 parameters
  branch4        2,080 parameters
  total         43,200 parameters
compare: one 5x5 conv 64 -> 128 channels has 204,928 parameters
```

> [!example] Counting the branches by hand
> - branch1: $64\cdot 32 + 32 = 2{,}080$.
> - branch2: $2{,}080 + (32\cdot32\cdot 9 + 32) = 2{,}080 + 9{,}248 = 11{,}328$.
> - branch3: $2{,}080 + (32\cdot32\cdot25 + 32) = 2{,}080 + 25{,}632 = 27{,}712$.
> - branch4: the max pool has no weights, so $2{,}080$.
> - One $5\times5$ convolution from 64 to 128 channels: $128\cdot 64\cdot 25 + 128 = 204{,}928$, almost five times the whole block.

Branch 4's max pool uses **stride 1, padding 1**: it takes the maximum over every $3\times3$ neighbourhood without shrinking the map, so it can be concatenated. Each branch is a different "zoom": $1\times1$ sees one pixel, $3\times3$ and $5\times5$ see neighbourhoods, the pool picks the strongest nearby response. The network learns how to weigh them. The full GoogLeNet stacks nine such blocks; see [[5-deep-networks/cnn-architectures|CNN architectures]].

## Why 1×1 convolutions first: the GoogLeNet trick

A $1\times 1$ convolution mixes the channels at every pixel and can change their number. A $k\times k$ convolution has $C_\text{out}\cdot C_\text{in}\cdot k^2$ weights, so halving $C_\text{in}$ before it halves its cost, while the $1\times1$ convolution itself is cheap.

<!-- fig:deepb-bottleneck -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 660 250" width="660" height="250" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>Parameter count of a 5x5 branch with and without a 1x1 bottleneck</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><text x="20" y="18" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">Directly</text><rect x="30" y="41.2" width="24" height="57.6" rx="3" fill="var(--muted)" fill-opacity="0.22" stroke="var(--muted)" stroke-width="1.4"/><text x="42" y="110.8" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">64 ch</text><line x1="60" y1="70" x2="228" y2="70" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><text x="144" y="58" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">5×5 conv</text><rect x="236" y="55.6" width="24" height="28.8" rx="3" fill="var(--accent)" fill-opacity="0.22" stroke="var(--accent)" stroke-width="1.4"/><text x="248" y="96.4" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">32 ch</text><text x="290" y="70" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:12.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">64·32·25 + 32 = 51,232 parameters</text><text x="20" y="134" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">With a 1×1 convolution first</text><rect x="30" y="161.2" width="24" height="57.6" rx="3" fill="var(--muted)" fill-opacity="0.22" stroke="var(--muted)" stroke-width="1.4"/><text x="42" y="230.8" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">64 ch</text><line x1="60" y1="190" x2="118" y2="190" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><text x="89" y="178" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">1×1</text><rect x="126" y="175.6" width="24" height="28.8" rx="3" fill="var(--accent2)" fill-opacity="0.22" stroke="var(--accent2)" stroke-width="1.4"/><text x="138" y="216.4" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">32 ch</text><line x1="156" y1="190" x2="228" y2="190" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><text x="192" y="178" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">5×5</text><rect x="236" y="175.6" width="24" height="28.8" rx="3" fill="var(--accent)" fill-opacity="0.22" stroke="var(--accent)" stroke-width="1.4"/><text x="248" y="216.4" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">32 ch</text><text x="290" y="182" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:12.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">(64·32 + 32) + (32·32·25 + 32)</text><text x="290" y="202" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:12.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">= 2,080 + 25,632 = 27,712 parameters (46% fewer)</text></svg><figcaption><strong>The GoogLeNet trick.</strong> A k×k convolution has C<sub>out</sub>·C<sub>in</sub>·k² weights, so shrinking the channels with a cheap 1×1 convolution first makes the expensive kernel cheaper. Slab height = number of channels. The 3×3 branch saves 39% the same way (18,464 → 11,328).</figcaption></figure>

```
3x3 branch: 18,464 parameters directly, 11,328 with a 1x1 convolution first (39% fewer)
5x5 branch: 51,232 parameters directly, 27,712 with a 1x1 convolution first (46% fewer)
```

The saving grows with the squeeze: GoogLeNet itself shrinks, for example, 192 channels to 16 before a $5\times5$ kernel. The same idea is called a **bottleneck** (ResNet-50 and deeper use $1\times1 \to 3\times3 \to 1\times1$ blocks). Fewer weights also means fewer multiply-adds per pixel, so the block is faster, not only smaller.

> [!intuition] 1×1 convolution = a linear layer at every pixel
> At pixel $(i,j)$, a $1\times 1$ convolution computes $W\,z_{ij} + b$, where $z_{ij}\in\mathbb R^{C_\text{in}}$ is the vector of channel values there and $W$ is $C_\text{out}\times C_\text{in}$. Same $W$ at every pixel. Lecture 16 checks this numerically by copying the weights into an `nn.Linear` and applying it to the permuted tensor; see [[5-deep-networks/fully-convolutional-networks#two-building-blocks|fully convolutional networks]].

## More graphs: DenseNet-style and multi-scale heads

Any connection pattern works if **(1)** shapes match wherever paths merge and **(2)** gradients have a short path back to every layer. Lecture 16's playground ("you are limited only by your imagination and gradient descent"):

- **ResNet-style**: add the input back (`+`); see [[5-deep-networks/residual-connections|residual connections]].
- **DenseNet-style**: every layer sees the concatenation of *all* previous outputs.
- **Multi-scale head**: global-average-pool every stage and feed all of them to the classifier.

```python
class DenseBlock(nn.Module):
    def __init__(self, c_in, growth, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c_in + k * growth, growth, 3, padding=1), nn.ReLU())
            for k in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)   # keep everything so far
        return x

class MultiScaleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1, self.stage2, self.stage3 = conv_block(1, 16), conv_block(16, 32), conv_block(32, 64)
        self.head = nn.Linear(16 + 32 + 64, 10)

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        pooled = [f.mean(dim=(2, 3)) for f in (f1, f2, f3)]   # global average pool
        return self.head(torch.cat(pooled, dim=1))
```

Channel bookkeeping in a dense block: layer $k$ receives $c_\text{in} + k\cdot\text{growth}$ channels, and the block outputs $c_\text{in} + n\cdot\text{growth}$. `DenseBlock(16, 16, 3)` turns 16 channels into $16 + 3\cdot 16 = 64$, and `DenseBlock(64, 16, 3)` turns 64 into 112, which is why the DenseNet-style model's head is `nn.Linear(112, 10)`. Use [`nn.ModuleList`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html), not a Python list, so the layers' parameters are registered. (`conv_block(c_in, c_out)` is two $3\times3$ conv + ReLU and a $2\times2$ max pool.)

Two epochs on 10,000 FashionMNIST images (treat small differences as noise):

| model | parameters | test accuracy |
|---|---|---|
| plain blocks | 72,442 | 71.7% |
| ResNet-style | 77,738 | 76.2% |
| DenseNet-style | 49,770 | 65.1% |
| multi-scale head | 72,922 | 73.9% |

## Branching at the output: two heads, one trunk

One shared feature extractor, two predictions: the garment class (10 classes) and "is it footwear?" (sandal, sneaker, ankle boot). The two losses are simply added, and one `backward()` sends both gradients into the shared trunk.

```python
FOOTWEAR = torch.tensor([5, 7, 9])   # sandal, sneaker, ankle boot

class TwoHeadCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(conv_block(1, 16), conv_block(16, 32),
                                   nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.class_head = nn.Linear(32, 10)
        self.footwear_head = nn.Linear(32, 1)

    def forward(self, x):
        features = self.trunk(x)
        return self.class_head(features), self.footwear_head(features).squeeze(1)

# inside the training loop
class_logits, footwear_logit = two_head(x)
is_footwear = torch.isin(y, FOOTWEAR).float()
loss = (F.cross_entropy(class_logits, y)
        + F.binary_cross_entropy_with_logits(footwear_logit, is_footwear))
```

```
class accuracy:    67.4%
footwear accuracy: 98.4%
```

Each head gets the loss that fits it: [`F.cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) on 10 logits ([[3-linear-models/multiclass-logistic-regression|multi-class]]) and [`F.binary_cross_entropy_with_logits`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html) on one logit ([[3-linear-models/binary-cross-entropy|binary]]). [`torch.isin`](https://docs.pytorch.org/docs/stable/generated/torch.isin.html) builds the binary label from the class label.

## When shapes do not match

### Case 1: the channels differ

Only `a + b` cares; `torch.cat` stacks any number of channels. **Fix:** a $1\times 1$ convolution on the smaller path; if the size also halves, give it the same stride (the [[5-deep-networks/residual-connections#projection-shortcuts|projection shortcut]]).

```python
x = torch.rand(1, 32, 16, 16)     # block input
fx = torch.rand(1, 64, 16, 16)    # body output: 64 channels
fx + x                            # RuntimeError
shortcut = nn.Conv2d(32, 64, kernel_size=1)
print(tuple((fx + shortcut(x)).shape))
fx2 = torch.rand(1, 64, 8, 8)     # body also halved H, W
projection = nn.Conv2d(32, 64, kernel_size=1, stride=2)
print(tuple((fx2 + projection(x)).shape))
```

```
RuntimeError: The size of tensor a (64) must match the size of tensor b (32) at non-singleton dimension 1
(1, 64, 16, 16)
(1, 64, 8, 8)
```

### Case 2: H and W differ

Both merges need the same H and W.

- **Fix 1, by design: the right padding.** With stride 1, $p=(k-1)/2$ keeps the size. That is how all four Inception branches end at $16\times16$.
- **Fix 2, by resizing:** upsample the smaller map ([`nn.Upsample`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Upsample.html), [`F.interpolate`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html), or a learned [`nn.ConvTranspose2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)), or pool the larger one (no new weights). U-Net does exactly this; see [[5-deep-networks/fully-convolutional-networks|fully convolutional networks]].

```python
a = torch.rand(1, 32, 16, 16)
c = torch.rand(1, 64, 8, 8)            # pooled once more
c_up = nn.Upsample(scale_factor=2)(c)  # now 64 x 16 x 16
print(tuple(torch.cat([a, c_up], dim=1).shape))
```

```
(1, 96, 16, 16)
```

> [!example] Exercise: will these two branches concatenate?
> Input $64\times16\times16$, `c_in = 64`, `c_branch = 32`:
> ```python
> branch2 = nn.Sequential(nn.Conv2d(c_in, c_branch, 1), nn.ReLU(), nn.Conv2d(c_branch, c_branch, 3, padding=1))
> branch3 = nn.Sequential(nn.Conv2d(c_in, c_branch, 1), nn.ReLU(), nn.Conv2d(c_branch, c_branch, 5, padding=1))
> ```
> Can `torch.cat([branch2(x), branch3(x)], dim=1)` combine them?

> [!solution]- Solution
> The $1\times1$ convolutions keep $16\times16$. branch2's $3\times3$ with $p=1$: $\lfloor(16+2-3)/1\rfloor+1 = 16$. branch3's $5\times5$ with $p=1$: $\lfloor(16+2-5)/1\rfloor + 1 = 14$. Shapes $32\times16\times16$ and $32\times14\times14$: **no**.
> ```
> RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 16 but got size 14 for tensor number 1 in the list.
> ```
> Fix: `padding=2` for the $5\times5$ kernel.

> [!example] Exercise (Lecture 16, "if you have free time"): a fifth branch
> Add a fifth branch with a $7\times 7$ kernel to `InceptionBlock(64, 32)`, built like branch3: a $1\times1$ convolution $64\to32$, ReLU, then a $7\times7$ convolution $32\to32$. What padding does it need, how many parameters does it add, and what is the output shape for a $64\times16\times16$ input?

> [!solution]- Solution
> Padding $p = (7-1)/2 = 3$. Parameters: $(64\cdot32 + 32) + (32\cdot32\cdot49 + 32) = 2{,}080 + 50{,}208 = 52{,}288$, so the block grows from 43,200 to 95,488. Output: $5\cdot32 = 160$ channels, $160\times16\times16$. (A $7\times7$ convolution straight from 64 to 32 channels would cost $32\cdot64\cdot49 + 32 = 100{,}384$.)

> [!trap] Common mistakes
> - `torch.cat(..., dim=0)` stacks along the **batch**, not the channels; for $N\times C\times H\times W$ the channel axis is `dim=1`.
> - Padding 1 is right only for $3\times3$; a $5\times5$ kernel needs 2, a $7\times7$ needs 3.
> - `a + b` **broadcasts**: a $1\times1\times16\times16$ map plus a $1\times64\times16\times16$ map gives no error, it silently adds the same map to every channel ([[1-foundations/broadcasting|broadcasting]]). The shape check only catches dimensions that differ and are not 1.
> - Layers kept in a plain Python list are invisible to `model.parameters()`; use `nn.ModuleList` or attributes.
> - In a two-head model, give each head its own loss and add them; do not run two separate `backward()` calls on graphs that share the trunk without `retain_graph`.

## PyTorch docs

- [`torch.cat`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html) — the channel merge; all non-`dim` sizes must match.
- [`nn.Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — output-size formula, `padding='same'` as an alternative to computing $p$.
- [`nn.MaxPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) — stride 1, padding 1 keeps the size (branch 4).
- [`nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html) and [`nn.ModuleList`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) — chains inside branches; lists of layers you loop over yourself.
- [`nn.AdaptiveAvgPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html) — global average pooling for heads.
- [`nn.Upsample`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Upsample.html), [`F.interpolate`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html) — resizing before a merge.
- [`torch.isin`](https://docs.pytorch.org/docs/stable/generated/torch.isin.html) — the footwear label.
- [`torchvision.models.googlenet`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.googlenet.html) — the full Inception network, pretrained.

Related: [[5-deep-networks/residual-connections]] · [[5-deep-networks/fully-convolutional-networks]] · [[5-deep-networks/convolutional-layers]] · [[5-deep-networks/cnn-architectures]] · [[4-pytorch-training/nn-module]]
