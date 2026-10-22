---
title: "Broadcasting"
description: "How PyTorch combines tensors of different shapes elementwise: align shapes from the right, each pair of sizes must be equal or 1 or missing, the size-1 side is stretched without copying; the Lecture 2 examples, the result-shape algorithm, uses in ML (bias, centering, normalizing), the (N,) vs (N, 1) bug, and the Spring 2025/2026 exam problems."
tags: [topic, foundations, midterm-1]
aliases: ["broadcastable", "broadcast rules", "broadcasting rules"]
---

*Lecture 2 · [[1-foundations/index|Foundations]] · builds on [[1-foundations/tensors|tensors]]*

> [!abstract] In one breath
> An elementwise operation on two tensors of different shapes still works when the shapes are **broadcastable**: line them up from the **last** dimension; in each position the sizes must be equal, or one of them must be 1, or one must be missing. A size-1 (or missing) dimension is stretched to match the other, virtually (stride 0, no copy), and the result has the larger size in every position. `A + 4`, adding a row vector to every row, and adding a column vector to every column are all broadcasting.

## What it is

<!-- fig:found-broadcasting -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 230" width="700" height="230" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>adding a row vector and a column vector to a 2 by 3 matrix</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><text x="20" y="18" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">(2, 3) + (3,)</text><rect x="20" y="34" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="35" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">1</text><rect x="50" y="34" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="65" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">2</text><rect x="80" y="34" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="95" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">3</text><rect x="20" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="35" y="79" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">0</text><rect x="50" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="65" y="79" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">1</text><rect x="80" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="95" y="79" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">0</text><text x="118" y="64" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:18px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">+</text><rect x="136" y="34" width="30" height="30" rx="0" fill="var(--accent)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="151" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">1</text><rect x="166" y="34" width="30" height="30" rx="0" fill="var(--accent)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="181" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">2</text><rect x="196" y="34" width="30" height="30" rx="0" fill="var(--accent)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="211" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">3</text><rect x="136" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3"/><text x="151" y="79" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">1</text><rect x="166" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3"/><text x="181" y="79" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">2</text><rect x="196" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3"/><text x="211" y="79" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">3</text><text x="234" y="64" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:18px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">=</text><rect x="252" y="34" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="267" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">2</text><rect x="282" y="34" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="297" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">4</text><rect x="312" y="34" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="327" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">6</text><rect x="252" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="267" y="79" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">1</text><rect x="282" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="297" y="79" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">3</text><rect x="312" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="327" y="79" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">3</text><text x="20" y="120" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">row copied down each row</text><text x="370" y="18" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">(2, 3) + (2, 1)</text><rect x="370" y="34" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="385" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">1</text><rect x="400" y="34" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="415" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">2</text><rect x="430" y="34" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="445" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">3</text><rect x="370" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="385" y="79" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">0</text><rect x="400" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="415" y="79" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">1</text><rect x="430" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="445" y="79" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">0</text><text x="468" y="64" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:18px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">+</text><rect x="486" y="34" width="30" height="30" rx="0" fill="var(--hi)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="501" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">2</text><rect x="486" y="64" width="30" height="30" rx="0" fill="var(--hi)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="501" y="79" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">4</text><rect x="516" y="34" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3"/><text x="531" y="49" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">2</text><rect x="546" y="34" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3"/><text x="561" y="49" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">2</text><rect x="516" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3"/><text x="531" y="79" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">4</text><rect x="546" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3"/><text x="561" y="79" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">4</text><text x="584" y="64" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:18px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">=</text><rect x="602" y="34" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="617" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">3</text><rect x="632" y="34" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="647" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">4</text><rect x="662" y="34" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="677" y="49" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">5</text><rect x="602" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="617" y="79" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">4</text><rect x="632" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="647" y="79" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">5</text><rect x="662" y="64" width="30" height="30" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="677" y="79" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">4</text><text x="370" y="120" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">column copied across each column</text><text x="20" y="160" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">Rule: line shapes up from the right; each pair of sizes must be equal, or one is 1, or one is missing.</text><text x="20" y="182" text-anchor="start" dominant-baseline="central" fill="var(--hi)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">(2, 3) + (2,) fails: trailing sizes 3 and 2 differ. Fix: torch.tensor([2, 4]).unsqueeze(1) → (2, 1).</text><text x="20" y="204" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">Dashed cells are never stored: broadcasting stretches with stride 0, no copy.</text></svg><figcaption><b>Broadcasting, Lecture 2.</b> The smaller tensor is stretched (dashed) along its size-1 or missing dimensions to match <code>A</code>, then the operation is elementwise.</figcaption></figure>

```python
A = torch.tensor([[1, 2, 3],
                  [0, 1, 0]])          # (2, 3)
B = A + 4                              # scalar
C = A + torch.tensor([1, 2, 3])        # (3,): each row
D = A + torch.tensor([2, 4]).unsqueeze(1)  # (2, 1): each column
```

```
B: tensor([[5, 6, 7],
           [4, 5, 4]])
C: tensor([[2, 4, 6],
           [1, 3, 3]])
D: tensor([[3, 4, 5],
           [4, 5, 4]])
```

Multiplication works the same way, giving a Hadamard ([[1-foundations/linear-algebra-in-pytorch#hadamard-elementwise-product|elementwise]]) product with a stretched vector:

```python
E = A * 4                              # [[4, 8, 12], [0, 4, 0]]
F = A * torch.tensor([2, 3, 2])        # [[2, 6, 6], [0, 3, 0]]
G = A * torch.tensor([2, 4]).unsqueeze(-1)  # [[2, 4, 6], [0, 4, 0]]
```

In math, each example adds or multiplies by a stretched copy:

$$
C = \begin{bmatrix}1&2&3\\0&1&0\end{bmatrix} + \begin{bmatrix}1&2&3\\1&2&3\end{bmatrix},\qquad
G = \begin{bmatrix}1&2&3\\0&1&0\end{bmatrix}\odot\begin{bmatrix}2&2&2\\4&4&4\end{bmatrix}.
$$

## What you should know

- **The rule** (Lecture 2): starting at the trailing (last) dimension, the sizes must be equal, or one of them is 1, or one of them does not exist.
- **Result shape**: pad the shorter shape with 1s on the left, then take the larger size in every position.
- `(n,)` lines up with the **last** dimension, so it acts as a **row**; to act on each **column** reshape it to `(n, 1)` with `unsqueeze(1)` (or `unsqueeze(-1)`, the same thing for a 1-D tensor).
- Broadcasting stretches without copying (like [[1-foundations/storage-and-views#other-views-expand-unsqueeze-transposes|`expand`]]), so it is as cheap as the scalar case.
- An **in-place** op cannot grow its left side: `a += b` needs the broadcast shape to equal `a`'s shape.
- PyTorch function docs say "supports broadcasting" when they do.

## Computing the result shape

Write the shapes right-aligned and compare column by column:

| `x` | `y` | result | why |
|---|---|---|---|
| `(2, 3)` | `()` scalar | `(2, 3)` | scalar stretches everywhere |
| `(2, 3)` | `(3,)` | `(2, 3)` | `3 = 3`, missing → 2 |
| `(2, 3)` | `(2, 1)` | `(2, 3)` | `3` vs `1` → 3; `2 = 2` |
| `(3, 2)` | `(3, 1)` | `(3, 2)` | Spring 2026 Q2(a) |
| `(4, 3)` | `(1, 3)` | `(4, 3)` | Spring 2025 Q1(a) |
| `(5, 1)` | `(1, 4)` | `(5, 4)` | both sides stretch: an "outer sum" |
| `(8, 1, 6, 1)` | `(7, 1, 5)` | `(8, 7, 6, 5)` | the docs' example |
| `(2, 3)` | `(2,)` | **error** | trailing sizes 3 and 2 differ |
| `(2, 3)` | `(3, 2)` | **error** | 3 vs 2 |

## Exam problems

> [!example] Spring 2025 Q1(a)
> ```python
> a = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])  # (4, 3)
> b = torch.arange(3).view(1, 3)
> c = a * b
> c += 1
> ```
> (i) `c` is $4\times3$: `(4, 3)` and `(1, 3)` broadcast to `(4, 3)`. (ii) `b` is `[[0, 1, 2]]`, shape $(1, 3)$: note the **two** brackets. Each row of `a` is multiplied by $(0, 1, 2)$ and then 1 is added, so $c = \begin{bmatrix}1&2&5\\1&5&11\\1&8&17\\1&11&23\end{bmatrix}$.

> [!example] Spring 2026 Q2(a)
> `torch.zeros((3, 2)) + torch.tensor([[10.], [20.], [30.]])` adds a $(3, 1)$ column to each column: rows of 10s, 20s and 30s, shape $(3, 2)$. The follow-up masked update is on [[1-foundations/indexing-and-slicing#writing-through-an-index-or-a-mask|indexing and slicing]].

## Where broadcasting shows up in ML

| Operation | Shapes |
|---|---|
| add a bias to every prediction, `X @ w + b` | `(N,)` + `()` |
| add a bias vector to a batch of layer outputs, `X @ W.T + b` | `(N, K)` + `(K,)` |
| center each feature, `X - X.mean(dim=0)` | `(N, d)` − `(d,)` |
| normalize each color channel of an image | `(3, H, W)` − `(3, 1, 1)` |
| all pairwise differences of two point sets | `(n, 1, d)` − `(1, m, d)` → `(n, m, d)` |

Every [`nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) layer relies on the second row; see [[4-pytorch-training/nn-module|nn.Module]].

> [!trap] The (N,) vs (N, 1) bug
> A model that outputs `y_pred` of shape `(N, 1)` compared with targets `y` of shape `(N,)`: `y_pred - y` broadcasts to `(N, N)`, every prediction minus every target. The loss is then a meaningless average of $N^2$ numbers, and no error is raised ([`nn.MSELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) only warns). Fix the shapes first: `y_pred.squeeze(1)` or `y.unsqueeze(1)`. See [[4-pytorch-training/loss-functions|loss functions]].

> [!trap] Common mistakes
> - Matching shapes from the **left**. Always start from the last dimension.
> - Expecting a `(2,)` vector to add to each column of a `(2, 3)` matrix: it lines up with the 3 and fails. Use `.unsqueeze(1)`.
> - Confusing broadcasting with matrix multiplication: `A * v` is elementwise; `A @ v` is a [[1-foundations/linear-algebra-in-pytorch#matrix-vector-product|matrix-vector product]].
> - In-place growth: `x = torch.zeros(1, 3); x += torch.ones(2, 3)` is an error, while `x = x + torch.ones(2, 3)` works and makes a new `(2, 3)` tensor.

> [!exam] On the exams
> Broadcasting hides inside the tensor problems: [[6-midterm-1/spring-2025|Spring 2025]] Q1(a) and [[6-midterm-1/spring-2026|Spring 2026]] Q2(a) above. Shape questions ("what is the shape of `c`?") are the usual form.

## PyTorch docs

- [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) — the rules, examples, and in-place restrictions (linked from the lecture).
- [`torch.broadcast_shapes`](https://docs.pytorch.org/docs/stable/generated/torch.broadcast_shapes.html) — compute a result shape without any data; good for checking your answer.
- [`torch.broadcast_tensors`](https://docs.pytorch.org/docs/stable/generated/torch.broadcast_tensors.html), [`Tensor.expand`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html) — make the stretching explicit.
- [`torch.unsqueeze`](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html) — the usual fix for a shape that does not line up.
- [`torch.add`](https://docs.pytorch.org/docs/stable/generated/torch.add.html), [`torch.mul`](https://docs.pytorch.org/docs/stable/generated/torch.mul.html) — the functions behind `+` and `*`, both broadcasting.

Related: [[1-foundations/tensors]] · [[1-foundations/linear-algebra-in-pytorch]] · [[1-foundations/indexing-and-slicing]] · [[3-linear-models/multiple-linear-regression]]
