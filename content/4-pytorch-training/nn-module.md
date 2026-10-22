---
title: "nn.Module: models as classes"
description: "How to write a PyTorch model: subclass torch.nn.Module, call super().__init__(), register learnable parameters with nn.Parameter or nn.Linear, implement forward, and let model.parameters() collect everything for the optimizer — with the lecture's polynomial, logistic-regression and multi-class models."
tags: [topic, training]
aliases: ["nn.Module", "torch.nn.Module", "nn.Parameter", "nn.Linear", "nn.Sequential", "nn.ModuleList", "forward method", "model.parameters()"]
---

*Lecture 10 (and nn.Linear in Lecture 11) · [[4-pytorch-training/index|PyTorch training]] · builds on [[2-calculus-and-optimization/autograd|autograd]] and [[2-calculus-and-optimization/fitting-models-with-gradient-descent|fitting models by hand]]*

> [!abstract] In one breath
> A model is a Python class that inherits from [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html). Its `__init__` calls `super().__init__()` and creates the learnable parameters; its `forward(self, x)` computes the prediction. Calling `model(x)` runs `forward`, and `model.parameters()` hands every registered parameter to the optimizer, so you no longer update and clear each tensor by hand.

## What it is

Every supervised problem in the course has the same four ingredients (Lecture 10):

| Ingredient | Symbol | PyTorch object |
|---|---|---|
| Dataset | $\mathcal{D}=\{(x_i,y_i)\}_{i=1}^N$ | a [[4-pytorch-training/datasets-and-dataloaders\|`Dataset`]] + `DataLoader` |
| Model with learnable parameters $\theta$ | $f_\theta(x)$ | an `nn.Module` subclass |
| Loss function | $\ell(f_\theta(x_i), y_i)$ | a [[4-pytorch-training/loss-functions\|loss object]] such as `nn.MSELoss()` |
| Learning algorithm | e.g. gradient descent | an [[4-pytorch-training/optimizers\|optimizer]] from `torch.optim` |

`nn.Module` is the base class for the second row. Before Lecture 10 the parameters were loose tensors with `requires_grad=True`, and every [[2-calculus-and-optimization/fitting-models-with-gradient-descent|gradient-descent loop]] had to list them one by one for the update and for clearing `.grad`. A module keeps them together.

Two methods must be written:

- **`__init__(self, ...)`** — the constructor. Its **first line** is `super().__init__()`, which runs `nn.Module`'s own constructor (the bookkeeping that lets it track parameters). Then create the parameters and any sub-layers as attributes of `self`.
- **`forward(self, x)`** — the forward pass: input in, prediction out.

## What you should know

- The template: `class M(nn.Module)`, `super().__init__()` first in `__init__`, parameters as `self.<name>`, and `forward(self, x)` returns the output.
- [`nn.Parameter(tensor)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html) wraps a tensor, sets `requires_grad=True`, and **registers** it with the module when assigned to `self`. A plain tensor attribute is *not* registered.
- `model(x)` calls `forward(x)` (through `__call__`). Write `model(x)`, not `model.forward(x)`.
- [`model.parameters()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters) is a **generator** over every registered parameter, including those inside sub-modules; it is what you pass to `torch.optim.SGD(...)`.
- [`nn.Linear(d, K)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) computes $z = Wx + b$ with `weight` of shape $(K, d)$ and `bias` of shape $(K,)$: $dK + K$ parameters. It accepts a batch $(B, d)$ and returns $(B, K)$.
- Layers kept in a Python list are invisible to `parameters()`; use [`nn.ModuleList`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) or [`nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html).
- Count parameters with `sum(p.numel() for p in model.parameters())`.
- A model whose output feeds [`nn.CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) returns **logits**: no softmax in `forward`.

## The first module: a cubic polynomial

$f_\theta(x) = ax^3 + bx^2 + cx + d$ with $\theta = \{a, b, c, d\}$:

```python
import torch
import torch.nn as nn

class ThirdOrderPolynomial(nn.Module):
    def __init__(self):
        super().__init__()                  # always first
        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))
        self.c = nn.Parameter(torch.rand(1))
        self.d = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return self.a*x**3 + self.b*x**2 + self.c*x + self.d

model = ThirdOrderPolynomial()
print(model.a)
for p in model.parameters():                # all four, in order
    print(p)
```

```
Parameter containing:
tensor([0.2153], requires_grad=True)
Parameter containing:
tensor([0.2153], requires_grad=True)
Parameter containing:
tensor([0.0176], requires_grad=True)
Parameter containing:
tensor([0.7241], requires_grad=True)
Parameter containing:
tensor([0.6884], requires_grad=True)
```

(The values are random; these are the lecture's.) A prediction such as `model(torch.tensor([1]))` comes back as `tensor([1.6454], grad_fn=<AddBackward0>)`, which is $a+b+c+d$: it is a node of the [[2-calculus-and-optimization/computational-graphs|computational graph]], ready for `backward()`.

> [!note] What `self` and `__init__` are
> A **class** bundles attributes and methods; an **object** is one instance of it. `self` is the instance inside its own methods, so `self.a = ...` stores `a` on *this* model. `__init__` is the constructor Python runs when you write `ThirdOrderPolynomial()`.

## Logistic regression as a module

The Lecture 10 class exercise: $f_\theta(x) = \Pr\{y=1 \mid x\} = \sigma(w^\top x + b)$ for $x \in \mathbb{R}^N$ (see [[3-linear-models/logistic-regression|logistic regression]]).

```python
class LogisticRegression(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.w = nn.Parameter(torch.rand(N))
        self.b = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return 1/(1 + torch.exp(-(self.w @ x + self.b)))
        # same: torch.sigmoid(self.w @ x + self.b)
```

With hand-made `w` of shape $(N,)$, a single input $x$ of shape $(N,)$ works, but a **batch** of shape $(B, N)$ does not line up with `w @ x`. Lectures 10–11 therefore feed `x.unsqueeze(-1)`, shape $(B, N, 1)$: `w @ x` [broadcasts](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html) the matrix product over the batch and returns $(B, 1)$, and `.squeeze(-1)` turns the output into $(B,)$ to match the labels. Writing `x @ self.w + self.b` instead takes $(B, N)$ directly and returns $(B,)$ — this is the form the Spring 2026 exam's solution uses (below).

## nn.Linear: a weight matrix with a bias

For $K$ classes, multi-class logistic regression scores each class with its own row of a matrix (Lecture 11, [[3-linear-models/multiclass-logistic-regression|multi-class logistic regression]]):

$$
z = Ax + b,\qquad A = \begin{bmatrix} w_1^\top \\ \vdots \\ w_K^\top \end{bmatrix} \in \mathbb{R}^{K\times d},\quad b\in\mathbb{R}^K .
$$

`nn.Linear(d, K)` is exactly that layer, with the batch handled for you:

```python
class MulticlassLogisticRegression(nn.Module):
    def __init__(self, d, K):
        super().__init__()
        self.weight_matrix = nn.Linear(d, K, bias=True)

    def forward(self, x):                   # x: (B, d)
        return self.weight_matrix(x)        # logits: (B, K)

model = MulticlassLogisticRegression(d=28*28, K=10)
print(model.weight_matrix.weight.shape)     # torch.Size([10, 784])
print(model.weight_matrix.bias.shape)       # torch.Size([10])
print(sum(p.numel() for p in model.parameters()))   # 7850
```

`nn.Linear` stores the weight as $(K, d)$ (out × in) and computes `x @ weight.T + bias`. Its parameters are registered automatically because a module assigned to `self` is a **sub-module**, and `parameters()` recurses into sub-modules. It initializes them with small random values ([`nn.init`](https://docs.pytorch.org/docs/stable/nn.init.html)), not zeros.

## Stacking layers: nn.Sequential and nn.ModuleList

Deeper models ([[5-deep-networks/multilayer-perceptrons|MLPs]], Lecture 14) compose many layers. Three equivalent styles:

```python
# 1. named attributes, forward by hand
class TwoLayer(nn.Module):
    def __init__(self, N, L, M):
        super().__init__()
        self.fc1 = nn.Linear(N, L)
        self.fc2 = nn.Linear(L, M)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# 2. nn.Sequential: forward is implied
mlp = nn.Sequential(nn.Linear(784, 128), nn.ReLU(),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.Linear(64, 10))

# 3. nn.ModuleList for a variable depth
class MyMLP(nn.Module):
    def __init__(self, input_dim, H, output_dim):
        super().__init__()
        dims = [input_dim] + H + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.act = nn.ReLU()
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))          # apply the activation!
        return self.layers[-1](x)           # last layer: logits
```

The `nn.Sequential` above has $784\cdot128+128 + 128\cdot64+64 + 64\cdot10+10 = 109{,}386$ parameters.

## Other methods you get for free

| Call | What it does |
|---|---|
| [`model.parameters()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters) / [`named_parameters()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_parameters) | iterate over parameters (with names like `fc1.weight`) |
| [`model.to(device)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to), `model.cpu()` | move every parameter (see [[4-pytorch-training/gpu-training\|GPU training]]) |
| [`model.train()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train) / [`model.eval()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval) | switch [[5-deep-networks/dropout\|dropout]] and [[5-deep-networks/batch-normalization\|batch norm]] between training and evaluation behaviour |
| [`model.zero_grad()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.zero_grad) | clear `.grad` of every parameter (same effect as `optimizer.zero_grad()`) |
| [`model.state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) | a dict of all parameter tensors, for [`torch.save`](https://docs.pytorch.org/docs/stable/generated/torch.save.html) / `load_state_dict` |
| `print(model)` | lists the sub-modules and their shapes |

> [!trap] Common mistakes
> - **Forgetting `super().__init__()`**, or calling it after assigning a layer: PyTorch raises an error as soon as you assign a parameter or sub-module.
> - **Plain tensors as parameters.** `self.w = torch.rand(N, requires_grad=True)` is not registered, so `model.parameters()` skips it and the optimizer never updates it. Wrap it: `nn.Parameter(...)`.
> - **A Python list of layers.** `self.layers = [nn.Linear(...), ...]` hides them from `parameters()` and from `.to(device)`. Use `nn.ModuleList`.
> - **Storing an activation is not applying it.** The Lecture 14 template saves `self.activation` but its `forward` only calls the linear layers, so the "MLP" is one big linear map. Call the activation between layers.
> - **Softmax inside `forward` before `nn.CrossEntropyLoss`.** That loss applies log-softmax itself; return logits.
> - **Shapes.** `nn.Linear(d, K)` wants the feature dimension **last**: $(B, d)$, not $(d, B)$. Its weight is $(K, d)$, the transpose of what you might write on paper as $W\in\mathbb{R}^{d\times K}$.

> [!exam] On the exams
> Lectures 10–11 come after the Midterm 1 scope (Lectures 2–9), but both Spring exams had a 10–15 point code question on them — see [[6-midterm-1/spring-2026|Spring 2026]] Q8 and [[6-midterm-1/spring-2025|Spring 2025]] Q7. Spring 2026 Q8(ii) asked for a linear model on $B\times 3$ inputs returning $B$ predictions:
>
> ```python
> class HomePriceModel(nn.Module):
>     def __init__(self, num_features):
>         super().__init__()
>         self.w = nn.Parameter(torch.ones(num_features))
>         self.b = nn.Parameter(torch.zeros(1))
>     def forward(self, x):             # x: (B, 3)
>         return x @ self.w + self.b    # (B,)
> ```
>
> The `nn.Linear` alternative is `self.model = nn.Linear(num_features, 1)` with `return self.model(x).squeeze(-1)` — the `squeeze` turns $(B, 1)$ into the required $(B,)$.

## PyTorch docs

- [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) — every method a model inherits (`parameters`, `to`, `train`, `eval`, `state_dict`, …).
- [`torch.nn.Parameter`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html) — the tensor subclass that registers itself.
- [`torch.nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) — shapes, the formula $y = xA^\top + b$, initialization.
- [`torch.nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html) and [`torch.nn.ModuleList`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) — containers for layers.
- [`torch.nn`](https://docs.pytorch.org/docs/stable/nn.html) — the full list of layers, activations and losses.
- [`torch.nn.init`](https://docs.pytorch.org/docs/stable/nn.init.html) — how layers are initialized, and how to change it.
- Tutorial: [Build the neural network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html).

Related: [[4-pytorch-training/loss-functions]] · [[4-pytorch-training/optimizers]] · [[4-pytorch-training/training-loop]] · [[3-linear-models/logistic-regression]] · [[5-deep-networks/multilayer-perceptrons]]
