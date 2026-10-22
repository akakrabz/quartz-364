---
title: "Residual connections"
description: "Why very deep plain networks train worse than shallow ones, and how ResNet's residual block y = ReLU(F(x) + x) fixes it: identity by default, a direct gradient path through the '+', projection shortcuts when shapes change, parameter counts, and Lecture 16's 21-layer plain-vs-residual experiment."
tags: [topic, deep-nets]
aliases: ["residual connection", "residual block", "skip connection", "ResNet", "shortcut connection", "projection shortcut", "identity shortcut", "degradation problem"]
---

*Lectures 16–17 · [[5-deep-networks/index|Deep networks]] · builds on [[5-deep-networks/branching-and-inception|branching]] and [[2-calculus-and-optimization/backpropagation|backpropagation]]*

> [!abstract] In one breath
> Adding layers should never hurt, since extra layers could learn the identity, yet very deep *plain* networks train worse, and sometimes not at all. A residual block lets its layers learn only a **correction**: $y = \mathrm{ReLU}(F(x) + x)$. If $F$ learns nothing, the block passes $x$ through; and the gradient always has a direct path back through the "+". When the block changes the channels or the size, the shortcut becomes a $1\times1$ convolution with the same stride.

## What it is

![[l17-deeper-error.png|Training and test error of 20- and 56-layer plain networks on CIFAR-10]]
<p class="cs-src">From the ResNet paper (He et al., 2015), as shown in Lecture 17: the 56-layer plain network has higher <em>training</em> error than the 20-layer one, so this is not overfitting.</p>

An $(N+L)$-layer network should be at least as good as an $N$-layer one: let the extra $L$ layers compute the identity. He et al. argued that a stack of nonlinear layers, starting from a random initialization, finds the identity hard to learn. So they changed what the layers are asked to learn. With $\mathcal F(x)$ the output of a few weight layers, the **residual unit** computes

$$
\mathcal H(x) = \mathcal F(x) + x .
$$

The identity is now $\mathcal F(x) = 0$, which weights near zero produce easily. The "+ x" is a **skip connection** (or shortcut).

![[l17-residual-block.png|The residual block from the ResNet paper]]

## What you should know

- The block: $y = \mathrm{ReLU}(F(x) + x)$; with a zero body, $y = \mathrm{ReLU}(x)$.
- Why it trains: $\dfrac{\partial y}{\partial x} = \dfrac{\partial F}{\partial x} + I$ (before the ReLU), so the gradient reaches earlier layers even when $\partial F/\partial x$ is small.
- `+` needs equal C, H and W. Body changes channels or uses stride 2 → **projection shortcut**: `nn.Conv2d(c_in, c_out, 1, stride=stride)` (+ batch norm).
- ResNet's block: conv3×3 → BN → ReLU → conv3×3 → BN, add shortcut, ReLU. Convolutions use `bias=False` because [[5-deep-networks/batch-normalization|batch norm]]'s $\beta$ already shifts.
- Parameter counts: `ResidualBlock(16, 16)` has 4,672; `ResidualBlock(16, 32, stride=2)` has 14,528 (derivation below).
- Lecture 16's experiment: 21 conv layers, no batch norm: plain stays at chance (loss 2.303, 10.1%), residual reaches 56.9% after 2 epochs.
- The two rules for any custom graph: matching shapes at every merge, a short gradient path to every layer.

## Why the "+ x" helps the gradient

Write a deep network as blocks $x_{k+1} = x_k + F_k(x_k)$ (ignore the ReLU). By the [[2-calculus-and-optimization/backpropagation|chain rule]],

$$
\frac{\partial L}{\partial x_k} = \frac{\partial L}{\partial x_{k+1}}\Big(I + \frac{\partial F_k}{\partial x_k}\Big)
\quad\Longrightarrow\quad
\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_K}\prod_{k=0}^{K-1}\Big(I + \frac{\partial F_k}{\partial x_k}\Big).
$$

In a plain network each factor is only $\partial F_k/\partial x_k$; if those Jacobians are small, their product shrinks exponentially with depth (**vanishing gradients**), and the early layers barely move. With residuals, expanding the product always leaves the term $I$: one path from the loss to $x_0$ goes through no weight layer at all.

> [!intuition] The computational-graph view
> At the "+" node the local partial is 1 for both inputs, so the adjoint is copied unchanged to $x$ and to $F(x)$ ([[2-calculus-and-optimization/computational-graphs|computational graphs]]). Stacked blocks give the gradient an express lane.

## Plain vs residual: 21 layers, no batch norm

Same layers, same initialization; the only difference is the "+ x".

```python
class SimpleResidual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(c, c, 3, padding=1))   # F(x)

    def forward(self, x):
        return torch.relu(x + self.body(x))

def deep_net(kind, n_blocks=10, c=16):
    layers = [nn.Conv2d(1, c, 3, padding=1), nn.ReLU()]
    for _ in range(n_blocks):                 # two convs per block
        if kind == 'residual':
            layers.append(SimpleResidual(c))
        else:
            layers += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
                       nn.Conv2d(c, c, 3, padding=1), nn.ReLU()]
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 10))
```

$1 + 10\cdot 2 = 21$ convolutional layers either way. Lecture 16 plots the gradient norm reaching each layer at initialization (log scale): for the plain network it falls layer after layer toward the input; for the residual one it stays large. Training both for 2 epochs on 8,000 FashionMNIST images:

```
plain
  trained in 36s, final training loss 2.303
  test accuracy 10.1%
residual
  trained in 37s, final training loss 1.290
  test accuracy 56.9%
```

The plain network's loss of $2.303 = \ln 10$ is exactly the cross-entropy of a uniform guess over 10 classes: it has learned nothing, and 10.1% is chance. In practice [[5-deep-networks/batch-normalization|batch norm]] is the other half of the fix.

## The full ResNet block

### Projection shortcuts

ResNet's block adds two things to `SimpleResidual`: a `nn.BatchNorm2d` after every convolution, and a **projection shortcut** for when the body changes the channels or the size. The figure shows the structure; the code is Lecture 16's `ResidualBlock`.

<!-- fig:deepb-residual-block -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 430" width="560" height="430" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>Residual block: two 3x3 conv + batch norm layers, a shortcut, an addition, and a ReLU</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><rect x="125" y="13" width="150" height="30" rx="6" fill="var(--muted)" fill-opacity="0.12" stroke="var(--muted)" stroke-width="1.4"/><text x="200" y="28" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">x: C_in × H × W</text><line x1="200" y1="43" x2="200" y2="66" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round"/><line x1="200" y1="66" x2="420" y2="66" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round"/><line x1="200" y1="66" x2="200" y2="86" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="125" y="86" width="150" height="28" rx="6" fill="var(--accent)" fill-opacity="0.12" stroke="var(--accent)" stroke-width="1.4"/><text x="200" y="100" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">3×3 conv, stride s</text><rect x="125" y="130" width="150" height="28" rx="6" fill="var(--accent2)" fill-opacity="0.12" stroke="var(--accent2)" stroke-width="1.4"/><text x="200" y="144" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">BatchNorm2d</text><line x1="200" y1="114" x2="200" y2="130" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="165" y="174" width="70" height="28" rx="6" fill="var(--muted)" fill-opacity="0.12" stroke="var(--muted)" stroke-width="1.4"/><text x="200" y="188" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">ReLU</text><line x1="200" y1="158" x2="200" y2="174" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="125" y="218" width="150" height="28" rx="6" fill="var(--accent)" fill-opacity="0.12" stroke="var(--accent)" stroke-width="1.4"/><text x="200" y="232" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">3×3 conv</text><line x1="200" y1="202" x2="200" y2="218" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="125" y="262" width="150" height="28" rx="6" fill="var(--accent2)" fill-opacity="0.12" stroke="var(--accent2)" stroke-width="1.4"/><text x="200" y="276" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">BatchNorm2d</text><line x1="200" y1="246" x2="200" y2="262" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><text x="100" y="188" text-anchor="end" dominant-baseline="central" fill="currentColor" style="font-size:15px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-style:italic;">F(x)</text><rect x="112" y="78" width="176" height="220" rx="10" fill="none" stroke="var(--muted)" stroke-width="1" stroke-dasharray="4 4"/><line x1="420" y1="66" x2="420" y2="165" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="330" y="163" width="180" height="64" rx="6" fill="var(--muted)" fill-opacity="0.06" stroke="var(--muted)" stroke-width="1.4" stroke-dasharray="5 4"/><text x="420" y="177" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">shortcut: identity, or</text><text x="420" y="195" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">1×1 conv, stride s + BN</text><text x="420" y="213" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">(if channels or size change)</text><line x1="200" y1="290" x2="200" y2="310" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><line x1="420" y1="227" x2="420" y2="326" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round"/><line x1="420" y1="326" x2="216" y2="326" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><circle cx="200" cy="326" r="15" fill="var(--paper)" stroke="currentColor" stroke-width="1.6"/><text x="200" y="326" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:20px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">+</text><line x1="200" y1="341" x2="200" y2="362" stroke="var(--muted)" stroke-width="1.4" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="165" y="362" width="70" height="28" rx="6" fill="var(--muted)" fill-opacity="0.12" stroke="var(--muted)" stroke-width="1.4"/><text x="200" y="376" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">ReLU</text><text x="260" y="376" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-style:italic;">y = ReLU(F(x) + x)</text></svg><figcaption><strong>ResNet's residual block</strong>, as built in Lecture 16 (<code>ResidualBlock</code>). The body learns a correction F(x); the shortcut carries x around it. Adding needs equal C, H and W, so when the body changes the channels or uses stride 2, the shortcut becomes a 1×1 convolution with the same stride (projection shortcut).</figcaption></figure>

```python
class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out))
        if stride != 1 or c_in != c_out:      # shapes differ: project x
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(c_out))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return torch.relu(self.body(x) + self.shortcut(x))

x = torch.rand(8, 16, 28, 28)
for c_in, c_out, stride in [(16, 16, 1), (16, 32, 2)]:
    res = ResidualBlock(c_in, c_out, stride)
    print('ResidualBlock({}, {}, stride={}): {} -> {}, {:,d} parameters, shortcut = {}'.format(
        c_in, c_out, stride, tuple(x.shape[1:]), tuple(res(x).shape[1:]), n_params(res),
        res.shortcut.__class__.__name__))
```

```
ResidualBlock(16, 16, stride=1): (16, 28, 28) -> (16, 28, 28), 4,672 parameters, shortcut = Identity
ResidualBlock(16, 32, stride=2): (16, 28, 28) -> (32, 14, 14), 14,528 parameters, shortcut = Sequential
```

> [!example] Counting the parameters
> `nn.BatchNorm2d(C)` has $2C$ parameters ($\gamma,\beta$); the convolutions have no bias.
> - `ResidualBlock(16, 16)`: $16\cdot16\cdot9 + 16\cdot16\cdot9 = 4{,}608$ conv weights, plus $2\cdot 32 = 64$ for two BNs: **4,672**.
> - `ResidualBlock(16, 32, stride=2)`: $16\cdot32\cdot9 + 32\cdot32\cdot9 = 4{,}608 + 9{,}216 = 13{,}824$; two BNs $2\cdot64 = 128$; shortcut $16\cdot 32 = 512$ plus BN $64$: total **14,528**.
>
> Shapes for the second: $28\times28$ with stride 2, $p=1$, $k=3$: $\lfloor(28+2-3)/2\rfloor+1 = 14$; the $1\times1$ shortcut with stride 2: $\lfloor(28-1)/2\rfloor + 1 = 14$. Both $32\times14\times14$, so they add.

[`nn.Identity`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Identity.html) returns its input; using it keeps `forward` the same in both cases.

> [!example] Lecture 16 "Your turn 1": complete `forward`
> The body and shortcut are given; add them and apply ReLU. The check also zeroes the body's last batch-norm weight ($\gamma$), so the body outputs zeros, and verifies that the block returns `torch.relu(x)`.

> [!solution]- Solution
> ```python
> def forward(self, x):
>     return torch.relu(self.body(x) + self.shortcut(x))
> ```
> With `nn.init.zeros_(block.body[-1].weight)`, the last BN outputs $0\cdot\hat z + \beta = 0$ ($\beta$ starts at 0), so $y = \mathrm{ReLU}(0 + x)$.

## Residual networks in practice

In Lecture 16's playground (2 epochs, 10,000 images) the ResNet-style model, a $3\times3$ stem then `ResidualBlock(16, 16)`, `ResidualBlock(16, 32, 2)`, `ResidualBlock(32, 64, 2)` and a global-average-pool head, reached **76.2%** with 77,738 parameters, against 71.7% for plain conv blocks of similar size (full table on [[5-deep-networks/branching-and-inception#more-graphs-densenet-style-and-multi-scale-heads|branching]]).

Residual blocks let He et al. train ResNet-34, -50 and -101 in 2015, and they sit at the heart of most deep models since. For the history and the ResNet-34 layout, see [[5-deep-networks/cnn-architectures|CNN architectures]]; U-Net's skips are the concatenating cousin, see [[5-deep-networks/fully-convolutional-networks|fully convolutional networks]].

```python
import torchvision
resnet = torchvision.models.resnet18(weights='DEFAULT')   # pretrained ImageNet model
print(resnet.layer1[0])                                    # a BasicBlock
```

> [!trap] Common mistakes
> - Adding before the shapes match: when `stride != 1` **or** `c_in != c_out`, the shortcut must project. A $1\times1$ convolution without the stride fixes the channels but not the size.
> - Putting the final ReLU inside the body, before the add; ResNet applies it after the sum.
> - Residual = `+` (shapes must match exactly); U-Net/DenseNet skip = `torch.cat` (only H, W must match). Do not mix them up when counting channels.
> - A residual connection does not remove the need for batch norm in very deep nets; the published ResNet uses both.
> - "Deeper got worse" in the ResNet figure is about **training** error too, so it is an optimization problem, not overfitting.

## PyTorch docs

- [`nn.Identity`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Identity.html) — the identity shortcut.
- [`nn.Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — `bias=False`, `stride` for the projection shortcut.
- [`nn.BatchNorm2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) — after every convolution in ResNet.
- [`F.relu`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html) — the same function as `torch.relu`, applied after the sum.
- [`nn.init.zeros_`](https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.zeros_) — used by the exercise check to zero the body.
- [`torchvision.models.resnet18`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) — a pretrained ResNet; `layer1` … `layer4` are stacks of residual blocks.

Related: [[5-deep-networks/branching-and-inception]] · [[5-deep-networks/batch-normalization]] · [[5-deep-networks/cnn-architectures]] · [[2-calculus-and-optimization/backpropagation]] · [[5-deep-networks/fully-convolutional-networks]]
