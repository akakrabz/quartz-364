---
title: "Review: autograd in code and find-the-bug"
description: "Midterm 1 problem type 5 (5–6 pts on the Spring exams, plus T/F items everywhere): the manual gradient-descent loop from Lecture 6 line by line, what backward() does and does not do, gradient accumulation, and a table of the bugs past exams asked you to find — x = x − α·grad, the missing grad reset, updates outside no_grad, .numpy() on a graph tensor."
tags: [exam, midterm-1, calculus]
aliases: ["autograd review", "find the bug"]
---

*Midterm 1 · problem type 5 of 7 · Lectures 5, 6, 9 · 5–6 pts on the Spring exams and several T/F items · topic pages: [[2-calculus-and-optimization/autograd|autograd]], [[2-calculus-and-optimization/fitting-models-with-gradient-descent|fitting models with gradient descent]] · back to [[6-midterm-1/index|Midterm 1]]*

> [!abstract] In one breath
> The manual training loop has five moving parts — a leaf tensor with `requires_grad=True`, a forward pass that builds the graph, `loss.backward()` that **adds** gradients into `.grad`, an update inside `torch.no_grad()` that is **in place**, and a reset of `.grad`. Both Spring exams removed one of them and asked what breaks.

## The loop, line by line

```python
w = torch.randn((), requires_grad=True)    # autograd tracks ops on w
for k in range(n_iter):
    loss = ((y - model(x, w))**2).mean()   # builds graph; loss is the seed
    loss.backward()                        # fills w.grad (adds to it!)
    with torch.no_grad():                  # keep update out of graph
        w -= alpha * w.grad                # in place: same leaf tensor
        w.grad = None                      # clear before next backward
```

- `loss.backward()` only **computes** gradients. It does not change the weights ([[6-midterm-1/spring-2026#problem-1--true-or-false-12-pts|Spring 2026 T/F e]]).
- `.grad` **accumulates** across `backward()` calls, so it must be cleared every iteration: `w.grad = None`, `w.grad.zero_()` or `optimizer.zero_grad()`.
- `.item()` gives a Python number; `.detach()` (or `.data`) gives the values without the graph; use `.detach().numpy()` before plotting.
- `z.retain_grad()` keeps `.grad` on an intermediate node; Lecture 5 uses it to compare hand-computed adjoints with autograd.
- The gradient over a dataset is the sum (or mean) of the per-example gradients, because differentiation is linear (Lecture 6).

## Bugs the exams asked you to find

| bug | symptom | fix | exam |
|---|---|---|---|
| `x = x - alpha*x.grad` inside `no_grad` | makes a **new** tensor with `requires_grad=False`; the next `backward()` fails | `x -= alpha*x.grad` | [[6-midterm-1/spring-2025#problem-2--feeling-out-of-place-5-pts\|Spr 25 Q2]] |
| gradient never cleared | each step uses the running sum of all past gradients; the iterate overshoots and never settles | add `x.grad = None` after the update | [[6-midterm-1/spring-2026#problem-3--feeling-out-of-place-6-pts\|Spr 26 Q3]] |
| update outside `torch.no_grad()` | in-place change of a leaf that requires grad raises an error | wrap the update in `with torch.no_grad():` | L6 |
| `.numpy()` on a tensor in the graph | error | `.detach().numpy()` | L6 |
| reading `.grad` of an intermediate node | `None` | `z.retain_grad()` before `backward()` | L5 |

## Loss modules you should recognize

`nn.MSELoss` (`reduction` = `'mean'`, `'sum'` or `'none'`); `nn.BCELoss` (probabilities after a sigmoid, float 0/1 targets); `nn.CrossEntropyLoss` (raw logits, integer class labels). More on [[4-pytorch-training/loss-functions|loss functions]].

The optimizer version (Lecture 10, after the Midterm 1 scope) replaces the last three lines with `optimizer.zero_grad()` before the forward pass and `optimizer.step()` after `loss.backward()` — see [[4-pytorch-training/training-loop|the training loop]] and [[4-pytorch-training/optimizers|optimizers]].

Practice: [[6-midterm-1/practice-problems#p10--find-the-bug|P10]] · next type: [[6-midterm-1/review-linear-regression|linear regression]]
