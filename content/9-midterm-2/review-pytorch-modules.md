---
title: "Review: writing modules and data pipelines"
description: "Midterm 2 problem type 2 (15–20 points of code on every past exam): the nn.Module template, turning equations into __init__ and forward line by line, every past fill-in-the-code problem (a gated block, a recurrent layer, an attention module, a feed-forward layer and an encoder layer), the Dataset methods, DataLoader batch arithmetic with samplers, the arguments of torch.optim.SGD, and a find-the-bug checklist."
tags: [exam, midterm-2, training]
aliases: ["writing modules", "nn.Module review", "fill in the code review"]
---

*Midterm 2 · problem type 2 of 6 · HW4 Q1–Q2, HW6 Q1 · Lectures 10–11, 14, 17, 19–21 · code on all four past exams · topic pages: [[4-pytorch-training/nn-module|nn.Module]], [[4-pytorch-training/datasets-and-dataloaders|Datasets and DataLoaders]], [[4-pytorch-training/optimizers|optimizers]] · back to [[9-midterm-2/index|Midterm 2]]*

> [!abstract] In one breath
> Every past exam shows a block as **equations or a diagram** and asks for the `nn.Module`: what goes in `__init__`, and the `forward`. The translation is mechanical. Each "matrix times something (plus bias)" is one `nn.Linear(in, out)`; `forward` is the equations in order, with `*` for element-wise products and `@` for matrix products. The keys say syntax is not graded strictly, so the points are in the **sizes** and the **order of operations**. The other code questions are facts about `Dataset`, `DataLoader` and `SGD`.

## The module template

```python
class MyBlock(nn.Module):
    def __init__(self, ...):
        super().__init__()                 # always first
        self.layer = nn.Linear(n_in, n_out)   # one attribute per learnable layer
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.layer(x))
```

- **`__init__` creates** the layers; **`forward` uses** them. A layer created inside `forward` would get new random weights at every call.
- **`super().__init__()`** must run before any layer is assigned. The Spring keys leave it out; real PyTorch then raises `AttributeError: cannot assign module before Module.__init__() call` ([[9-midterm-2/errata|errata]]).
- A variable number of layers goes in an **`nn.ModuleList`**, never a Python list ([[7-transformers/pytorch-transformer-layers#why-an-nnmodulelist|why]]).
- Call the module, `block(x)`, not `block.forward(x)`.

## From equations to code

> [!recipe] Recipe
> 1. **Sizes.** For every matrix, find input and output length from the equation ([[9-midterm-2/review-sizes-and-parameters#counting-parameters|sizes first]]). Equal-size operations ($\odot$, $+$) fix them.
> 2. **`__init__`.** One `nn.Linear(in, out, bias=...)` per matrix. Argument order is **(in, out)**, although the matrix is out × in. Store the activation.
> 3. **`forward`.** One line per equation, same order, same names.
> 4. **Read it back** against the equations: is every symbol used, is the right variable passed on?

| Math | Code |
|---|---|
| $Wx + b$ | `self.W(x)` with `self.W = nn.Linear(in, out)` |
| $Wx$ (no bias) | `nn.Linear(in, out, bias=False)` |
| $a\odot b$ (element-wise) | `a * b` |
| $AB$ (matrix product) | `A @ B` |
| $K^\top$ for a batch `(B, N, d)` | `K.transpose(1, 2)` (not `.T`) |
| $\sigma$, $\tanh$, ReLU | `torch.sigmoid(z)`, `torch.tanh(z)`, `torch.relu(z)`, or the `nn` modules |
| softmax over each row | `softmax(z, dim=-1)` |
| $x + f(x)$ (skip connection) | `x = x + self.f(x)` |

## Every past instance

| Exam | Given | You write | Sizes |
|---|---|---|---|
| [[9-midterm-2/fall-2024#problem-3--a-custom-block-20-pts\|F24 3]] | $y_1=\sigma(W_1x)\odot x$, $y_2=\sigma(W_2y_1)\odot x$, $z=W_3y_2$ | 3 `nn.Linear`, the activation, `forward` | `(N,N)`, `(N,N)`, `(N,D)` |
| [[9-midterm-2/fall-2025#problem-3--a-new-recurrent-layer-15-pts\|F25 3]] | $z=\tanh(W_xx_t)$, $h_t=\tanh(W_h(z+h_{t-1}))$ and the code | the four sizes, the count, the bug | `(d,H)`, `(H,H)` |
| [[9-midterm-2/spring-2025#problem-7--attention-is-all-you-need-15-pts\|Sp25 7]] | three attention steps with projections | the whole module | four `(d, d)` layers |
| [[9-midterm-2/spring-2026#problem-7--attention-is-all-you-need-15-pts\|Sp26 7]] | $y=W_o\,\sigma(W_ix+b_i)+b_o$; the encoder-layer diagram | feed-forward module; encoder layer | two `(d, d)`; attention, 2 `LayerNorm(d)` |

The pattern of the two Fall problems is the same: a **new block the lectures never showed**, built from pieces they did. Fall 2024's is written out here because it is the cleanest example:

```python
class CustomBlock(nn.Module):
    def __init__(self, N, D, activation_fn):
        super().__init__()
        self.W1 = nn.Linear(N, N)
        self.W2 = nn.Linear(N, N)
        self.W3 = nn.Linear(N, D)
        self.activation = activation_fn

    def forward(self, x):
        y1 = self.activation(self.W1(x)) * x
        y2 = self.activation(self.W2(y1)) * x
        z = self.W3(y2)
        return z
```

The two Transformer modules (attention; encoder layer) are on [[9-midterm-2/review-attention-and-transformers#the-two-modules-to-be-able-to-write|the attention review]].

## Dataset, DataLoader, optimizer

**A `Dataset` needs three methods** ([[9-midterm-2/fall-2024#problem-2--datasets-dataloaders-and-sgd-20-pts|F24 2(a)]], [[9-midterm-2/fall-2025#problem-1--true-or-false-30-pts|F25 1(a)]]): `__init__` (load or build the data), `__len__` (number of examples), `__getitem__(idx)` (the example at an index). Not `forward`.

**A `DataLoader` turns a `Dataset` into batches.** The arithmetic:

$$
\text{batches per epoch}=\left\lceil\frac{\text{examples the loader draws from}}{\text{batch size}}\right\rceil
$$

- F24 2(b): 72 points split into two equal halves of 36; four batches each → batch size $36/4 = 9$.
- A split is made by giving each loader a **sampler** over its own index list: `DataLoader(my_dataset, batch_size=9, sampler=SubsetRandomSampler(training_indices))`. Both loaders wrap the same dataset.
- One **iteration** = one batch = one `optimizer.step()`. One **epoch** = one full pass = all the batches ([[9-midterm-2/fall-2024#problem-1--true-or-false-30-pts|F24 1(h)]]).

**`torch.optim.SGD(params, lr, momentum=0, weight_decay=0)`** ([[9-midterm-2/fall-2024#problem-2--datasets-dataloaders-and-sgd-20-pts|F24 2(c)]]):

| Argument | Status | Note |
|---|---|---|
| trainable model parameters | required | `model.parameters()` |
| learning rate | required on the F24 key | newer PyTorch gives it the default 0.001, as printed on the F25 formula sheet |
| momentum | optional | default 0 |
| weight decay | optional | default 0 |
| number of epochs | not used | belongs to your loop |
| loss function | not used | the optimizer only reads `.grad` |

## Find the bug

Checklist for "this code runs but does not learn" questions, from the past exams and the lectures:

| Symptom in the code | What goes wrong | Fix |
|---|---|---|
| a loop computes `h_next` but never assigns `h = h_next` ([[9-midterm-2/fall-2025#problem-3--a-new-recurrent-layer-15-pts\|F25 3(b)]]) | every step starts from the zero state: no memory | update the state inside the loop |
| a mask is **multiplied** into the scores ([[9-midterm-2/fall-2025#problem-4--attention-15-pts\|F25 4(a)]]) | negative scores become $+\infty$ | **add** a mask of 0 and $-\infty$ |
| layers kept in a Python list | their weights are not in `model.parameters()`: never trained | `nn.ModuleList` |
| no `super().__init__()` | error when the first layer is assigned | call it first |
| `nn.Linear(out, in)` | shape error, or silently wrong sizes when square | (in, out) |
| no `model.eval()` at evaluation | dropout still drops, batch norm still uses batch statistics | `model.eval()`, then `model.train()` |
| `optimizer.zero_grad()` missing | gradients accumulate across steps | zero them each iteration |
| softmax applied before `nn.CrossEntropyLoss` | the loss applies it again | pass logits |

> [!trap] Common mistakes
> - **Writing the matrix shape into `nn.Linear`.** $W_3\in\mathbb R^{D\times N}$ is `nn.Linear(N, D)`.
> - **Using the wrong variable on the next line**, e.g. `self.W2(x)` where the equation has $W_2y_1$.
> - **`@` where `*` is meant** (or the reverse). $\odot$ is `*`.
> - **Forgetting `return`.**
> - **Answering "required" for epochs or the loss** in the SGD question: neither is an argument of the optimizer.

> [!question]- Check yourself
> 1. A block computes $u = \mathrm{ReLU}(W_ax)$, $v = W_bu + x$ with $x\in\mathbb R^n$, $u\in\mathbb R^m$, biases on. Write `__init__` and `forward`, and count the parameters.
> 2. A dataset has 1,000 examples and the loader uses `batch_size=64`. How many `optimizer.step()` calls in 3 epochs?
>
> > [!solution]- Answers
> > 1. `self.Wa = nn.Linear(n, m)`; `self.Wb = nn.Linear(m, n)` (the sum with $x$ forces $v\in\mathbb R^n$). `u = torch.relu(self.Wa(x)); v = self.Wb(u) + x; return v`. Parameters: $(nm + m) + (mn + n) = 2nm + m + n$.
> > 2. $\lceil1000/64\rceil = 16$ batches per epoch (15 full ones and one of 40), so 48 steps.

Related: [[9-midterm-2/review-sizes-and-parameters]] · [[9-midterm-2/review-attention-and-transformers]] · [[9-midterm-2/review-recurrent-networks]] · [[6-midterm-1/review-autograd]]
