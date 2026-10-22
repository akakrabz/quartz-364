---
title: "Loss functions in PyTorch"
description: "The loss classes in torch.nn used in the course — MSELoss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss (and NLLLoss) — what each expects as input and target (probabilities vs logits, float vs class indices, shapes), the reduction argument, and hand-checked numbers."
tags: [topic, training]
aliases: ["loss function", "criterion", "nn.MSELoss", "nn.BCELoss", "nn.BCEWithLogitsLoss", "nn.CrossEntropyLoss", "reduction", "logits"]
---

*Lectures 10–11 · [[4-pytorch-training/index|PyTorch training]] · the math is on [[3-linear-models/linear-regression|linear regression]], [[3-linear-models/binary-cross-entropy|binary cross-entropy]] and [[3-linear-models/multiclass-logistic-regression|multi-class logistic regression]]*

> [!abstract] In one breath
> A loss is an object you build once (`criterion = nn.MSELoss()`) and call on every batch: `loss = criterion(predictions, targets)`. It returns a scalar tensor that is the seed of `loss.backward()`. The choices that matter are **what the model must output** (probabilities for `BCELoss`, raw scores/logits for `BCEWithLogitsLoss` and `CrossEntropyLoss`), **what the target looks like** (floats of the same shape, or integer class indices), and the **`reduction`** (`'mean'` by default).

## What it is

The loss turns the model's predictions on a batch into one number to minimise. The [`torch.nn` loss classes](https://docs.pytorch.org/docs/stable/nn.html#loss-functions) are thin wrappers around functions in [`torch.nn.functional`](https://docs.pytorch.org/docs/stable/nn.functional.html) (`F.mse_loss`, `F.cross_entropy`, …); the class form stores options such as `reduction` so the training loop stays short:

```python
predictions = model(inputs)
loss = criterion(predictions, targets)      # scalar tensor
loss.backward()                             # seed of backprop
```

A custom loss can be any function (or `nn.Module`) built from differentiable tensor operations; autograd handles it the same way.

## What you should know

- **Argument order**: `criterion(prediction, target)` — model output first.
- `nn.MSELoss`: the mean of $(\hat y_i - y_i)^2$ over all elements ($\frac1B\sum_i$ for a $(B,)$ batch); prediction and target float, **same shape**.
- `nn.BCELoss`: input = **probabilities** in $[0,1]$ (after a sigmoid), target = floats $0.0/1.0$ of the same shape.
- `nn.BCEWithLogitsLoss`: input = **logits** (no sigmoid); it applies the sigmoid internally in a numerically stable way. Prefer it.
- `nn.CrossEntropyLoss`: input = **logits** of shape $(B, K)$, target = **class indices** of shape $(B,)$, dtype `torch.long` (int64). It applies log-softmax internally. Never softmax first.
- `reduction='mean'` (default) averages (over all elements for MSE/BCE, over the $B$ examples for cross-entropy), `'sum'` adds, `'none'` returns one loss per element.
- `loss.item()` turns the 0-d result into a Python float for logging.

## The four losses side by side

| Loss | Model output | Target | Formula per example |
|---|---|---|---|
| [`nn.MSELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) | any real values $\hat y$ | float, same shape | $(\hat y - y)^2$ |
| [`nn.BCELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) | probability $p = \sigma(z)\in[0,1]$ | float 0/1, same shape | $-[y\log p + (1-y)\log(1-p)]$ |
| [`nn.BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) | logit $z\in\mathbb{R}$ | float 0/1, same shape | same, with $p=\sigma(z)$ computed inside |
| [`nn.CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) | logits $z\in\mathbb{R}^K$, shape $(B,K)$ | class index $y\in\{0,\dots,K-1\}$, shape $(B,)$, `long` | $-\log\dfrac{e^{z_y}}{\sum_{j} e^{z_j}}$ |

Related ones you may meet: [`nn.NLLLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) takes **log-probabilities** (`CrossEntropyLoss` = [`log_softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html) + `NLLLoss`), and [`nn.L1Loss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html) is mean absolute error.

## reduction: mean, sum or none

From Lecture 10, prediction all zeros and target all ones (four elements):

```python
criterion_mean = nn.MSELoss(reduction='mean')
criterion_sum = nn.MSELoss(reduction='sum')
criterion_none = nn.MSELoss(reduction='none')
x, y = torch.zeros(4), torch.ones(4)
print(criterion_mean(x, y))
print(criterion_sum(x, y))
print(criterion_none(x, y))
```

```
tensor(1.)
tensor(4.)
tensor([1., 1., 1., 1.])
```

`'none'` keeps one loss per element, useful for masking or re-weighting some examples before you reduce them yourself. With `'sum'` the gradient grows with the batch size, so the learning rate that works for `'mean'` is effectively multiplied by $B$.

## Numbers to check by hand

**MSE.** Predictions $(1, 2)$, targets $(0, 4)$: $\frac12[(1)^2 + (-2)^2] = 2.5$.

**BCE.** A batch of two: $p=0.9$ with $y=1$, $p=0.2$ with $y=0$:

$$
\tfrac12\big[-\log 0.9 - \log(1-0.2)\big] = \tfrac12(0.1054 + 0.2231) = 0.1643 .
$$

**BCE with logits.** Logit $z=0$ means $p=\sigma(0)=0.5$, so the loss is $\log 2 = 0.6931$ whatever the label. Logit $z=2$ with $y=1$: $\sigma(2)=0.8808$ and the loss is $\log(1+e^{-2}) = 0.1269$.

**Cross-entropy.** Logits $z=(2, 1, 0.1)$ give softmax probabilities $(0.659, 0.242, 0.099)$. With target class 0 the loss is $-\log 0.659 = 0.417$; with logits $(2, 0, 0)$ and target 0 it is $\log(1 + 2e^{-2}) = 0.2395$.

```python
criterion = nn.CrossEntropyLoss()
logits = torch.tensor([[2.0, 1.0, 0.1]])    # (B=1, K=3)
target = torch.tensor([0])                  # (B,), long
loss = criterion(logits, target)            # 0.417
```

## Which model output goes with which loss

| Task | Last layer | Loss | Prediction at test time |
|---|---|---|---|
| Regression | `nn.Linear(d, 1)` (squeeze to $(B,)$) | `MSELoss` | the output itself |
| Binary, lecture style | linear → `torch.sigmoid` | `BCELoss` | $p \ge 0.5$ |
| Binary, stable | linear (logit) | `BCEWithLogitsLoss` | logit $\ge 0$ |
| $K$ classes | `nn.Linear(d, K)` (logits) | `CrossEntropyLoss` | `torch.argmax(logits, dim=1)` |

The softmax and the sigmoid are monotone, so the class with the largest logit is the class with the largest probability: you never need the softmax just to predict.

> [!trap] Common mistakes
> - **Softmax (or sigmoid) before `CrossEntropyLoss` / `BCEWithLogitsLoss`.** They apply it again internally; training still runs but the gradients are squashed and the reported loss is wrong.
> - **Logits into `BCELoss`.** It needs values in $[0,1]$ and raises an error otherwise.
> - **Wrong target dtype for `CrossEntropyLoss`.** Class indices must be `long`; the Lecture 11 four-class dataset ends its label tensor with `.long()` for that reason. Float targets of shape $(B,)$ raise an error (float targets are only accepted as class *probabilities* of shape $(B,K)$).
> - **One-hot labels where indices are expected**, or the reverse.
> - **Shape mismatch for MSE/BCE.** A $(B,1)$ prediction against a $(B,)$ target broadcasts to $(B,B)$ in MSE (PyTorch warns) — the loss is then silently wrong. Squeeze the prediction.
> - **Arguments swapped**: `criterion(target, prediction)` gives a different (wrong) value for every loss except the symmetric MSE and L1.
> - **Pairing MSE with a sigmoid output for classification** gives tiny gradients when the sigmoid saturates — a Spring 2026 true/false item ([[6-midterm-1/spring-2026|Spring 2026]] T/F).

> [!exam] On the exams
> The losses appear in the math form on Midterm 1 (BCE as the negative log-likelihood of [[3-linear-models/logistic-regression|logistic regression]]; cross-entropy with softmax in [[3-linear-models/multiclass-logistic-regression|multi-class]]). In code, Spring 2025 Q7 and Spring 2026 Q8 build `nn.MSELoss()` and call it as `criterion(output, target)`.

## PyTorch docs

- [Loss functions in `torch.nn`](https://docs.pytorch.org/docs/stable/nn.html#loss-functions) — the complete list.
- [`nn.MSELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html), [`nn.L1Loss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html) — regression losses and `reduction`.
- [`nn.BCELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html), [`nn.BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) — binary; the second explains the log-sum-exp stability trick and `pos_weight`.
- [`nn.CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — shapes for input/target, class-index vs probability targets, `weight`, `label_smoothing`.
- [`nn.NLLLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html) and [`F.log_softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html) — the two halves of cross-entropy.
- [`torch.nn.functional`](https://docs.pytorch.org/docs/stable/nn.functional.html) — function versions such as [`F.cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) and [`F.mse_loss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html).

Related: [[4-pytorch-training/nn-module]] · [[4-pytorch-training/training-loop]] · [[3-linear-models/binary-cross-entropy]] · [[3-linear-models/multiclass-logistic-regression]] · [[3-linear-models/linear-regression]]
