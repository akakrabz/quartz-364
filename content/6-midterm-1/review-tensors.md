---
title: "Review: tensors, shapes, views and masks"
description: "Midterm 1 problem type 1 (10–25 pts on every past exam): predicting shapes, memory and dtypes, the view-vs-copy table, slicing rules, boolean-mask patterns and the syntax that costs points, broadcasting, and PyTorch linear algebra — with every past exam item and the HW1 answers."
tags: [exam, midterm-1, foundations]
aliases: ["tensor review", "slicing review"]
---

*Midterm 1 · problem type 1 of 7 · HW1 · Lecture 2 · 10–25 pts on every past exam · topic pages: [[1-foundations/tensors|tensors]], [[1-foundations/storage-and-views|storage and views]], [[1-foundations/indexing-and-slicing|indexing and slicing]], [[1-foundations/broadcasting|broadcasting]], [[1-foundations/linear-algebra-in-pytorch|linear algebra in PyTorch]] · back to [[6-midterm-1/index|Midterm 1]]*

> [!abstract] In one breath
> Every exam opens with a short code snippet and asks for a **shape**, the **contents** after an in-place change, whether something is a **view**, or the **one line** that builds a boolean mask. Four rules decide almost all of it: `stop` is excluded; basic slicing makes views while list/mask indexing copies; a mask always returns a 1-D tensor; and conditions combine with `&`, `|`, `~` with **every comparison in parentheses**.

## Shapes you must predict

| code | result |
|---|---|
| `torch.arange(N)` | shape `(N,)`, integers `0 … N−1` (stop excluded) |
| `torch.randint(low, high, size)` | integers in `[low, high)`; `high` excluded |
| `torch.rand(s)` / `torch.randn(s)` | uniform on $[0,1)$ / standard normal |
| `x.view(3, -1)` | `-1` is inferred: `arange(42).view(3, -1)` → `(3, 14)` |
| `torch.cat((a, a), dim=1)` | joins along an existing dim: `(4, 10)` → `(4, 20)` |
| `torch.stack((a, a, a), dim=-1)` | adds a new dim: `(4, 10)` → `(4, 10, 3)` |
| stack 4 copies of a `(2, 3)` at dim 0 / 1 / 2 | `(4, 2, 3)` / `(2, 4, 3)` / `(2, 3, 4)` |
| `x.unsqueeze(1)` on `(2, 3)` | `(2, 1, 3)`; `.squeeze(1)` removes it again |
| `A.T`, `A.transpose(0, 1)` | rows and columns swap |
| `a[2]`, `a[2, :]`, `a[:, 3]` | 1-D; `a[2:3, :]` keeps the 2-D shape `(1, n)` |
| `a[mask]` | always 1-D, elements in row-major order |

## Memory and dtypes

Bytes per element: `int8`/`uint8` 1, `int16` 2, `int32`/`float32` 4, `int64`/`float64` 8. Float tensors default to `float32`; integer tensors from `torch.tensor`, `arange` and `randint` default to `int64`. Memory $=$ `a.element_size() * a.nelement()`.

> [!example] HW1 Q2 — one million 256×256 RGB images, one byte per channel
> ```python
> images = torch.zeros((1_000_000, 3, 256, 256), dtype=torch.uint8)  # (image, channel, row, col)
> ```
> $10^6\cdot3\cdot256\cdot256=196{,}608{,}000{,}000$ bytes $\approx183.1$ GiB. `uint8` is the most efficient choice because intensities 0–255 fit in one byte. Lecture example: `torch.rand([8, 128, 224, 224], dtype=torch.float64)` takes $8\cdot128\cdot224\cdot224\cdot8$ B $=392$ MiB.

## View or new memory?

A **view** is a second window onto the same storage: writing through any window changes the original and every other view. Lecture 2 checks this with `a.storage().data_ptr()`; for a slice that starts mid-tensor, `a.data_ptr()` differs (it points at the slice's first element) even though the storage is shared.

| same storage (a view) | new storage (a copy) |
|---|---|
| `.view(…)` | arithmetic and math: `a + b`, `a * 2`, `torch.exp(a)`, `a.exp()` |
| basic slicing: `a[1:3]`, `a[::2, 1:8:3]`, `a[0, :]` | indexing with a list or tensor: `a[:, [0, 2, 3]]` |
| `.T`, `.transpose()`, `.unsqueeze()`, `.squeeze()`, `.expand()` | boolean masks: `a[a > 0]` |
| in-place ops ending in `_`: after `b = a.cos_()`, `b` *is* `a` | `torch.cat`, `torch.stack` |
| `.cpu()` on a tensor already on the CPU | `.clone()` |

**Past items** (full solutions on the exam pages):

- [[6-midterm-1/fall-2024#problem-2--list-indexing-views-and-a-matrix-product-15-pts|Fall 2024 Q2]] — `y = x[:, [0, 2, 3]]` is **not** a view; `z = y.T @ x` is `3×4`.
- [[6-midterm-1/spring-2025#problem-1--lost-in-the-layers-navigating-data-slices-24-pts|Spring 2025 Q1(b)]] — `d[0:2, 1:6:4]` is a view, `f.add_(1)` writes through to `e` and `d`, `h = g.t() @ g = 40`.
- [[6-midterm-1/spring-2026#problem-2--lost-in-the-layers-navigating-data-slices-14-pts|Spring 2026 Q2(b)]] — `a[:, ::5]` is a view, so after `b.add_(1)`, `a[:, 0] = [1, 15, 29]`.

> [!example] Lecture 12 review — in-place op, then addition
> ```python
> a = torch.zeros(3, 2)
> b = a.cos_()     # in-place: returns a itself
> c = a + b        # new tensor
> ```
> `a` and `b` share a pointer; `c` does not. `a` and `b` are all 1 ($\cos0$), `c` is all 2.

## Slicing rules

Zero-indexed, `[row, col]` order, `start:stop:step` with `stop` excluded, negative indices count from the end.

- `a[::2, ::4]` on an `8×8` takes rows 0, 2, 4, 6 and columns 0, 4: shape `(4, 2)` ([[6-midterm-1/fall-2024|Fall 2024 Q1a]]).
- `a[::2, 1:8:3]` takes columns 1, 4, 7. `a[[0, 1, 3, 4, 7], 1:5:3]` takes the listed rows and columns 1 and 4.

## Boolean-mask patterns

| goal | code | seen in |
|---|---|---|
| magnitude greater than ½ | `a[torch.abs(a) > 0.5]` | Fall 24 Q1b |
| positive multiples of 5 | `a[(a > 0) & (a % 5 == 0)]` | Fall 25 Q2c |
| negative multiples of 9 | `x[(x < 0) & (x % 9 == 0)]` | L2 |
| two more than a multiple of 5 | `x[x % 5 == 2]` | L2 |
| greater than the mean | `a[a > a.mean()]` | HW1 |
| multiples of 4 in column 3 | `e = a[:, 3]; h = e[e % 4 == 0]` | L12 |
| top-right quadrant of a `10×10` | `a[0:5, 5:]` | L12 |
| columns 1 and 3 | `a[:, [1, 3]]` | HW1 |
| every $n$th row and column (a view) | `a[::n, ::n]` | HW1 |
| every third element of the last row | `x[-1, ::3]` | L2 |

> [!trap] Syntax that costs points
> - Combine conditions with `&`, `|`, `~` — never `and`/`or`/`not` (those try to turn a whole tensor into one bool and raise an error).
> - **Parenthesize every comparison.** `&` binds tighter than `>` and `==`, so `a > 0 & a % 5 == 0` parses as `a > (0 & a % 5) == 0`, a chained comparison, and raises an error.
> - `%` follows Python: `-3 % 5 == 2`. So `x % 5 == 2` also picks −3, −8, −78 — which *are* two more than a multiple of 5.

## Broadcasting

Compare shapes from the **last** dimension backwards. Each pair must be equal, or one of them must be 1 or missing; that dimension is stretched to match.

| shapes | result |
|---|---|
| `(2, 3) + (3,)` | `(2, 3)`: the row is added to each row |
| `(2, 3) + (2, 1)` | `(2, 3)`: the column is added to each column |
| `(3, 2) + (3, 1)` | `(3, 2)` ([[6-midterm-1/spring-2026|Spring 2026 Q2a]]) |
| `(4, 3) * (1, 3)` | `(4, 3)` ([[6-midterm-1/spring-2025|Spring 2025 Q1a]]) |
| `(2, 3) + (2,)` | **error**: 3 vs 2 |

## Linear algebra in PyTorch

| code | meaning |
|---|---|
| `A @ B`, `torch.matmul`, `torch.mm`, `torch.mv` | matrix product $(m\times n)(n\times p)\to(m\times p)$; inner sizes must match |
| `A * B` | Hadamard (elementwise) product; same or broadcastable shapes |
| `torch.dot(u, v)` | 1-D vectors only; returns a scalar |
| `torch.outer(u, v)` | `(n,)` and `(m,)` → $n\times m$ matrix $uv^\top$ |
| `v @ v` for 1-D `v` | dot product; `v.t()` does nothing on a 1-D tensor |
| `torch.norm(v, p=2 / 1 / torch.inf)` | $\Vert v\Vert_2$, $\sum_i\lvert v_i\rvert$, $\max_i\lvert v_i\rvert$; needs a float tensor |
| `torch.inverse(B)` | square invertible matrices only; singular ones raise an error |

$u\cdot v=\sum_iu_iv_i$, and the cosine similarity $\cos\theta=\dfrac{u\cdot v}{\Vert u\Vert_2\Vert v\Vert_2}\in[-1,1]$ is 0 for perpendicular vectors.

## Other HW1 answers

> [!solution]- HW1 Q1, Q3, Q4, Q5
> **Q1** (supervised or unsupervised?) (a) Counting distinct authors from writing style: **unsupervised** (clustering, no labels). (b) Language of an audio clip from labelled clips: **supervised**. (c) Make or miss from shot videos with known outcomes: **supervised**. See [[1-foundations/what-is-machine-learning|what is machine learning]].
>
> **Q3** (a) `b = a[:, [1, 3]]`. (b) `b = a[a > a.mean()]`. (c) `a[::n, ::n]` keeps every $n$th row and column: a subsampled grid, and a view of `a`.
>
> **Q4** (soft-shrinkage) $S_\tau(x)=\operatorname{sgn}(x)\max\{0,\lvert x\rvert-\tau\}$. For $x=(1,3,-1,-4,5,-6)$ and $\tau=2$: $(0,1,0,-2,3,-4)$. Missing line: `result = x_sign * torch.where(x_abs - tau > 0, x_abs - tau, y)` (with `y` a tensor of zeros).
>
> **Q5** $\partial\Vert y-x\Vert_2^2/\partial x=2(x-y)\in\mathbb R^4$ and $\partial(x\odot y)/\partial x=\operatorname{diag}(y)\in\mathbb R^{4\times4}$ — see [[6-midterm-1/review-gradient-descent#matrix-calculus|the matrix-calculus table]].

Practice: [[6-midterm-1/practice-problems#p1--views-and-slicing|P1]], [[6-midterm-1/practice-problems#p2--shapes|P2]], [[6-midterm-1/practice-problems#p3--mask|P3]] · next type: [[6-midterm-1/review-gradient-descent|matrix calculus and gradient descent]]
