---
title: "Autograd"
description: "PyTorch's automatic differentiation engine in practice: requires_grad and leaf tensors, backward() and where gradients land, why .grad accumulates and must be cleared, retain_grad, torch.no_grad for updates, detach/item/numpy, and the classic bugs in a hand-written gradient-descent loop."
tags: [topic, calculus, midterm-1]
aliases: ["torch.autograd", "requires_grad", "backward", "loss.backward()", "no_grad", "torch.no_grad", "detach", "retain_grad", "gradient accumulation", "leaf tensor"]
---

*Lectures 3, 5, 6 · [[2-calculus-and-optimization/index|Calculus and optimization]] · builds on [[2-calculus-and-optimization/backpropagation|backpropagation]] · used by every training loop from [[2-calculus-and-optimization/fitting-models-with-gradient-descent|Lecture 6]] on*

> [!abstract] In one breath
> Mark the tensors you want derivatives for with `requires_grad=True`. Every operation on them is recorded in a computational graph during the forward pass. Calling `.backward()` on a scalar result runs [[2-calculus-and-optimization/backpropagation|backpropagation]] and **adds** $\partial(\text{result})/\partial(\text{leaf})$ into each leaf's `.grad`. Then update parameters inside `torch.no_grad()`, in place, and clear `.grad` before the next backward pass.

## What it is

`torch.autograd` is PyTorch's reverse-mode automatic-differentiation engine. It does not produce formulas; it records the operations as they run (the graph is built on the fly, anew each forward pass) and, on request, walks them backwards using each operation's known local derivative.

```python
x = torch.tensor(2.0, requires_grad=True)  # leaf tensor
y = x**2 + 3*x                             # recorded: y.grad_fn exists
y.backward()                               # dy/dx = 2x + 3
print(x.grad)
```

```
tensor(7.)
```

## What you should know

- `requires_grad=True` must be set for PyTorch to track operations on a tensor (True, Spring 2026 T/F). Results of operations on such tensors require grad too and carry a `grad_fn` (their node in the graph).
- **Leaf tensors** are the ones you created (parameters, inputs). After `backward()`, only leaves that require grad get `.grad`; intermediate nodes need `.retain_grad()` first.
- `backward()` starts from a **scalar** seed (a loss). It only **computes** gradients — it does not change any weights (False: "`loss.backward()` updates the weights", Spring 2026).
- `.grad` **accumulates** (adds) across `backward()` calls. Clear it every iteration: `w.grad = None`, `w.grad.zero_()`, or `optimizer.zero_grad()`.
- Do the update inside `with torch.no_grad():` and **in place** (`w -= alpha * w.grad`), so `w` stays the same leaf tensor and the update is not recorded.
- `.item()` → Python number; `.detach()` (or the older `.data`) → same values without the graph; `.detach().numpy()` before plotting.
- Because differentiation is linear, the gradient of a mean (or sum) loss over a dataset is the mean (or sum) of per-example gradients — autograd gets it from one `backward()` on the averaged loss.

## The pieces

### `requires_grad` and leaves

Three equivalent ways to ask for gradients: `torch.tensor(1.0, requires_grad=True)`, `torch.randn((), requires_grad=True)`, or setting `x.requires_grad = True` / calling [`x.requires_grad_()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.requires_grad_.html) on a tensor you made. Inputs like the data $x_i$ usually do **not** require grad; parameters do. Check with [`x.is_leaf`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html) and `x.requires_grad`.

### `backward()` and `.grad`

[`f.backward()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html) seeds the backward pass with $\bar f=1$, so it needs `f` to be a single number; calling it on a non-scalar tensor raises an error unless you reduce first (`.sum()`, `.mean()`) or pass a `gradient` argument. Afterwards [`w.grad`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.grad.html) holds $\partial f/\partial w$, with the same shape as `w`.

### Gradients accumulate

```python
x = torch.tensor(3.0, requires_grad=True)
for _ in range(2):
    y = x**2          # new graph each time
    y.backward()      # adds dy/dx = 6
print(x.grad)
```

```
tensor(12.)
```

Accumulation is deliberate (it lets several losses or mini-batches add their gradients), but in a training loop it means stale gradients from the previous iteration get added to the new ones unless you clear them.

### Intermediate nodes: `retain_grad()`

Lecture 5 compares hand-computed adjoints with autograd for $f=x^2\cos(xy)+1$ at $(x,y)=(-1,\tfrac\pi3)$. Intermediate nodes only keep `.grad` if asked with [`retain_grad()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.retain_grad.html):

```python
x = torch.tensor([-1.0], requires_grad=True)
y = torch.tensor([np.pi/3], requires_grad=True)
z1, z2 = x, y
z3 = z1**2; z4 = z1*z2; z5 = torch.cos(z4)
z6 = z3*z5; z7 = z6 + 1
for z in (z3, z4, z5, z6, z7):
    z.retain_grad()               # else .grad is None
z7.backward()
print(z1.grad, z2.grad, z4.grad)
```

```
tensor([-0.0931]) tensor([-0.8660]) tensor([0.8660])
```

By hand: $z_3=1$, $z_4=-\pi/3$, $z_5=\tfrac12$; $\bar z_4=-z_3\sin z_4=\sin(\pi/3)\approx0.866$, $\bar z_2=-z_1z_3\sin z_4\approx-0.866$, $\bar z_1=2z_1z_5-z_2z_3\sin z_4=-1+\tfrac\pi3\cdot0.866\approx-0.0931$. ✓

### `torch.no_grad()`, `detach()`, `item()`

- [`torch.no_grad()`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html): operations inside are not recorded. Use it for parameter updates and for evaluation. It affects only new operations — `z3bar = z5` inside it is still the same tensor as `z5`, graph and all (visible in Lecture 5's printout as `grad_fn=<CosBackward0>`).
- [`t.detach()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html): a tensor sharing `t`'s data but cut from the graph. `t.data` is the older equivalent.
- [`t.item()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html): the single value as a Python number (for logging the loss).
- `t.numpy()` fails on a tensor that requires grad; use `t.detach().numpy()`.

## The manual training loop (Lecture 6)

Know what every line does — exam questions remove or break one of them.

```python
w = torch.randn((), requires_grad=True)   # parameter, tracked
for k in range(n_iter):
    loss = ((y - model(x, w))**2).mean()  # forward pass; loss is the seed
    loss.backward()                       # fills w.grad (adds!)
    with torch.no_grad():                 # update is not recorded
        w -= alpha * w.grad               # in place: same leaf
        w.grad = None                     # clear for next iteration
```

From [[4-pytorch-training/optimizers|Lecture 10]] on, an optimizer replaces the last three lines: `optimizer.zero_grad()` before the forward pass and `optimizer.step()` after `loss.backward()` ([[4-pytorch-training/training-loop|training loop]]).

| bug | symptom | fix | seen on |
|---|---|---|---|
| `x = x - alpha*x.grad` inside `no_grad` | new tensor with `requires_grad=False`; the next `backward()` fails | `x -= alpha*x.grad` | Spring 2025 Q2 |
| gradient never cleared | gradients pile up, steps grow, `x` runs off | `x.grad = None` after the update | Spring 2026 Q3 |
| update outside `torch.no_grad()` | in-place change of a leaf that requires grad raises an error | wrap the update in `with torch.no_grad():` | Lecture 6 |
| `.numpy()` on a tensor in the graph | error | `.detach().numpy()` | Lecture 6 |
| reading `.grad` of an intermediate node | `None` | `z.retain_grad()` before `backward()` | Lecture 5 |
| `requires_grad` never set | `backward()` errors: nothing to differentiate | `requires_grad=True` on the parameters | Spring 2026 T/F |

> [!trap] Common mistakes
> - Believing `backward()` updates parameters. It only writes `.grad`.
> - Forgetting that `.grad` adds up. A loop without clearing computes the gradient of the *sum* of all past losses.
> - Rebinding (`w = w - ...`) instead of updating in place: `w` stops being the leaf the next loss depends on.
> - Calling `backward()` on a vector loss (e.g. `reduction='none'`) without reducing it to a scalar.

> [!exam] On the exams
> Autograd appears as a 5–6 point "find the bug in this gradient-descent loop" question (Spring 2025 Q2, Spring 2026 Q3) and as T/F items ("`requires_grad` must be True…", "`loss.backward()` … immediately updates the weights"). See [[6-midterm-1/spring-2025|Spring 2025]], [[6-midterm-1/spring-2026|Spring 2026]], [[6-midterm-1/true-false|T/F bank]].

## PyTorch docs

- [`torch.autograd`](https://docs.pytorch.org/docs/stable/autograd.html) — package reference.
- [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) — how the graph is recorded, leaves, `requires_grad`, in-place operations.
- [`torch.Tensor.backward`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html) · [`torch.Tensor.grad`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.grad.html) · [`torch.Tensor.requires_grad_`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.requires_grad_.html) · [`torch.Tensor.is_leaf`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html) · [`torch.Tensor.retain_grad`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.retain_grad.html)
- [`torch.no_grad`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html) · [`torch.enable_grad`](https://docs.pytorch.org/docs/stable/generated/torch.enable_grad.html) · [`torch.Tensor.detach`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html) · [`torch.Tensor.item`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html) · [`torch.Tensor.zero_`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.zero_.html)
- [`torch.autograd.grad`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html) — gradients returned instead of accumulated into `.grad`.
- [`torch.autograd.functional.jacobian`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html) — whole Jacobians ([[2-calculus-and-optimization/matrix-calculus|matrix calculus]]).
- Tutorials: [Automatic differentiation with `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html), [A gentle introduction to `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html).

Related: [[2-calculus-and-optimization/backpropagation]] · [[2-calculus-and-optimization/fitting-models-with-gradient-descent]] · [[1-foundations/tensors]] · [[4-pytorch-training/training-loop]] · [[4-pytorch-training/optimizers]]
