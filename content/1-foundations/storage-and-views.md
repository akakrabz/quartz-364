---
title: "Storage and views"
description: "How a tensor sits in memory: one flat storage plus offset, shape and strides; which operations return a view of the same storage (view, basic slicing, .T, transpose, unsqueeze, squeeze, expand, in-place ops) and which allocate new memory (math, list and mask indexing, cat, stack, clone); data_ptr checks, contiguity, view vs reshape, and the exam's chains of views and in-place updates."
tags: [topic, foundations, midterm-1]
aliases: ["tensor view", "views", "storage", "data_ptr", "in-place operation", "in-place ops", "contiguous", "stride", "view vs reshape", "torch.Storage"]
---

*Lecture 2 · HW 1 · [[1-foundations/index|Foundations]] · builds on [[1-foundations/tensors|tensors]] · next: [[1-foundations/indexing-and-slicing|indexing and slicing]]*

> [!abstract] In one breath
> A tensor's numbers live in a flat block of memory, its **storage**; the tensor itself is only a recipe for reading that block (where to start, the shape, and how far to jump per dimension, the **strides**). A **view** is a second recipe for the same storage, so it costs no memory, and writing through any view changes every other one. Reshaping, transposing and basic slicing make views; arithmetic, list or mask indexing, `cat`, `stack` and `clone` make copies; methods ending in `_` modify a tensor **in place** and return that same tensor.

## What it is

<!-- fig:found-storage-views -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 260" width="720" height="260" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>flat storage 0 to 11 viewed as a 3 by 4 grid, column 1 highlighted</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><text x="20" y="20" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">storage (one flat block of 12 elements)</text><rect x="20" y="34" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="37" y="51" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">0</text><text x="37" y="80" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">0</text><rect x="54" y="34" width="34" height="34" rx="0" fill="var(--accent)" fill-opacity="0.3" stroke="currentColor" stroke-width="1.2"/><text x="71" y="51" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">1</text><text x="71" y="80" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">1</text><rect x="88" y="34" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="105" y="51" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">2</text><text x="105" y="80" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">2</text><rect x="122" y="34" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="139" y="51" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">3</text><text x="139" y="80" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">3</text><rect x="156" y="34" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="173" y="51" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">4</text><text x="173" y="80" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">4</text><rect x="190" y="34" width="34" height="34" rx="0" fill="var(--accent)" fill-opacity="0.3" stroke="currentColor" stroke-width="1.2"/><text x="207" y="51" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">5</text><text x="207" y="80" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">5</text><rect x="224" y="34" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="241" y="51" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">6</text><text x="241" y="80" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">6</text><rect x="258" y="34" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="275" y="51" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">7</text><text x="275" y="80" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">7</text><rect x="292" y="34" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="309" y="51" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">8</text><text x="309" y="80" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">8</text><rect x="326" y="34" width="34" height="34" rx="0" fill="var(--accent)" fill-opacity="0.3" stroke="currentColor" stroke-width="1.2"/><text x="343" y="51" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">9</text><text x="343" y="80" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">9</text><rect x="360" y="34" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="377" y="51" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">10</text><text x="377" y="80" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">10</text><rect x="394" y="34" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="411" y="51" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">11</text><text x="411" y="80" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">11</text><text x="436" y="51" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">value</text><text x="436" y="80" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">offset</text><rect x="20" y="120" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="37" y="137" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">0</text><rect x="54" y="120" width="34" height="34" rx="0" fill="var(--accent)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="71" y="137" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">1</text><rect x="88" y="120" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="105" y="137" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">2</text><rect x="122" y="120" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="139" y="137" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">3</text><rect x="20" y="154" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="37" y="171" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">4</text><rect x="54" y="154" width="34" height="34" rx="0" fill="var(--accent)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="71" y="171" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">5</text><rect x="88" y="154" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="105" y="171" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">6</text><rect x="122" y="154" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="139" y="171" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">7</text><rect x="20" y="188" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="37" y="205" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">8</text><rect x="54" y="188" width="34" height="34" rx="0" fill="var(--accent)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="71" y="205" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">9</text><rect x="88" y="188" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="105" y="205" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">10</text><rect x="122" y="188" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="139" y="205" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">11</text><text x="20" y="240" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">a = torch.arange(12).view(3, 4)</text><text x="180" y="125" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">shape (3, 4), stride (4, 1)</text><text x="180" y="145" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">next row: jump 4 slots; next col: 1 slot</text><text x="180" y="180" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">a[:, 1] → elements 1, 5, 9</text><text x="180" y="200" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">same storage: offset 1, stride (4,)</text><text x="445" y="125" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">a.T → shape (4, 3), stride (1, 4)</text><text x="445" y="145" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">same storage, not contiguous</text><text x="445" y="180" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">a + 0, a[:, [1]], a.clone()</text><text x="445" y="200" text-anchor="start" dominant-baseline="central" fill="var(--hi)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">new storage (copies)</text></svg><figcaption><b>One storage, many views.</b> A tensor is a window on a flat block of memory: an offset, a shape and a stride per dimension. Reshaping, transposing and basic slicing only change those numbers; the highlighted column <code>a[:, 1]</code> is the storage elements 1, 5, 9.</figcaption></figure>

The data of every tensor is held in a `torch.Storage`, a one-dimensional array of bytes. A tensor adds metadata on top: a **storage offset** (where its first element is), a **shape**, and a **stride** per dimension (how many elements to skip in storage to move one step along that dimension). For a freshly made `torch.arange(12).view(3, 4)` the strides are `(4, 1)`: one row down is 4 elements later in storage, one column right is 1 element later.

Any operation that only changes offset, shape or strides can return a new tensor object over the **same** storage. That is a **view**. Nothing is copied, which is why views are fast and memory-free, and why they are dangerous: there is only one copy of the numbers.

## What you should know

- **Views (same storage):** `.view()`, basic slicing with integers and `start:stop:step` (`a[1:3]`, `a[::2, 1:8:3]`, `a[0, :]`), `.T`, `.transpose()`, `.permute()`, `.unsqueeze()`, `.squeeze()`, `.expand()`, `.detach()`, and `.cpu()` on a tensor already on the CPU.
- **In-place ops** end in `_` (`a.exp_()`, `a.cos_()`, `b.add_(1)`) and so do augmented assignments (`a += 1`): they write into `a`'s storage and return `a` itself.
- **New storage (copies):** arithmetic and math functions (`a + b`, `a * 2`, `torch.exp(a)`, `a.exp()`), indexing with a list or tensor (`a[:, [0, 2, 3]]`), boolean masks (`a[a > 0]`), `torch.cat`, `torch.stack`, `.clone()`, and `.to()` a different dtype or device.
- A write through a view (`b[0, 0] = 5`, `b.add_(1)`) changes the original and every other view.
- Check sharing with `x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()`. `x.data_ptr()` is the address of `x`'s **first element**, so a slice that starts mid-tensor has a different `data_ptr()` even though it shares storage.
- `.view()` needs strides compatible with the new shape (in practice, a contiguous tensor); `.reshape()` returns a view when it can and copies when it must.

## Storage and data pointers

```python
a = torch.randn([4, 3])
print(a.is_contiguous())             # elements laid out in order?
print(a.storage().data_ptr())
b = a.cpu()                          # already on CPU: no copy
print(b.storage().data_ptr())
c = torch.exp(a)                     # math: new memory
print(c.storage().data_ptr())
a[1, 1] = 0
print(b)                             # b sees the change
```

```
True
838003584
838003584
832964928
tensor([[-0.9194, -1.4681,  1.5404],
        [ 1.1997,  0.0000, -1.3892],
        [ 0.0995,  0.6378,  0.3501],
        [-0.1592,  0.0283,  0.1203]])
```

`b = a.cpu()` returned `a` itself because `a` was already in CPU RAM; `.cuda()` (or `.to('cuda')`) would allocate a copy in GPU memory. The lecture's `a.storage()` prints a deprecation warning in current PyTorch; the replacement is [`a.untyped_storage()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.untyped_storage.html), which has the same `data_ptr()`.

### The view method

```python
t = torch.rand(4, 4)
b = t.view(1, 16)          # same 16 numbers, new shape
print(t.storage().data_ptr() == b.storage().data_ptr())
```

```
True
```

`view` never changes the number of elements; one size may be `-1` and is inferred (`torch.arange(42).view(3, -1)` is $(3, 14)$).

## In-place operations

> [!example] Lecture 2, Example 1 — `exp` vs `exp_`
> ```python
> a = torch.rand((3, 3))
> b = a.exp()      # new tensor
> c = a.exp_()     # overwrites a, returns a
> a[0, 0] = 0
> print(b[0, 0], c[0, 0])
> ```
> `a` and `c` share a pointer, `b` does not. After `a[0, 0] = 0`, `c[0, 0]` is 0 (it **is** `a`) while `b` keeps its old value.

> [!example] Lecture 12 review — in-place op, then an addition
> ```python
> a = torch.zeros(3, 2)
> b = a.cos_()     # in-place: returns a itself
> c = a + b        # new tensor
> print(a.data_ptr(), b.data_ptr(), c.data_ptr())
> ```
> ```
> 462740736 462740736 462737344
> ```
> (a) `b` shares `a`'s pointer; `c` does not share `b`'s. (b) `cos(0) = 1`, so `a` and `b` are all ones, and `c = a + b` is all twos, each of shape $(3, 2)$.

## Other views: expand, unsqueeze, transposes

**`expand`** repeats a tensor along new or size-1 dimensions without copying, by giving them stride 0:

```python
a = torch.rand([3, 3])
b = a.expand(3, -1, -1)       # (3, 3, 3), -1 keeps a size
print(a.storage().data_ptr() == b.storage().data_ptr())   # True
```

Every "copy" in `b` is the same memory, so writing into an expanded tensor is an error or a surprise; call `.clone()` first if you need independent copies. [[1-foundations/broadcasting|Broadcasting]] uses the same trick.

**`unsqueeze`** (Lecture 2, Example 3):

```python
a = torch.rand((3, 4))
b = a.unsqueeze(0)    # (1, 3, 4)
c = a.unsqueeze(-1)   # (3, 4, 1)
c[0, 2, 0] = 50       # writes a[0, 2]
```

After the last line `a[0, 2]`, `b[0, 0, 2]` and `c[0, 2, 0]` are all 50: three tensors, one storage.

**Transposes** swap strides. `a.T` of a $(3, 4)$ tensor has shape $(4, 3)$ and strides $(1, 4)$: still the same storage, but no longer **contiguous** (walking it row by row jumps around in memory).

## View vs reshape, and contiguity

```python
a = torch.arange(12).view(3, 4)
a.T.is_contiguous()      # False
a.T.reshape(-1)          # works: copies here
a.T.contiguous().view(-1)  # same result, explicit copy
a.T.view(-1)             # RuntimeError: use .reshape(...) instead
```

| | `x.view(shape)` | `x.reshape(shape)` |
|---|---|---|
| returns | always a view | a view when possible, else a copy |
| non-contiguous input | may raise an error | copies |
| use when | you need sharing guaranteed | you just need the shape |

`x.contiguous()` returns `x` itself if it is already contiguous, otherwise a contiguous copy.

## Views and in-place updates on the exams

> [!example] Spring 2025 Q1(b) — a chain of views and an in-place add
> ```python
> d = torch.arange(100).view(10, 10)
> e = d[0:2, 1:6:4]    # rows 0-1, columns 1 and 5
> f = e[0, :]          # view of e
> g = f.add_(1)[:2]    # in-place on f
> h = g.t() @ g
> ```
> `e` starts as $\begin{bmatrix}1&5\\11&15\end{bmatrix}$, a view of `d`. `f = [1, 5]` is a view of `e`. `f.add_(1)` makes `f = [2, 6]` **and** changes `e` and `d`, then returns `f`; `g` is `f[:2]`, the same two numbers. `g` is 1-D, so `.t()` does nothing and `@` is a dot product: $h = 2^2 + 6^2 = 40$. Finally $e = \begin{bmatrix}2&6\\11&15\end{bmatrix}$.

> [!example] Spring 2026 Q2(b) — slice with a step, then `add_`
> ```python
> a = torch.arange(42).view(3, -1)   # (3, 14)
> b = a[:, ::5]                      # columns 0, 5, 10
> b.add_(1)
> ```
> `b` has shape $(3, 3)$ and is a view, so after `add_` it is $\begin{bmatrix}1&6&11\\15&20&25\\29&34&39\end{bmatrix}$ and `a[:, 0]` is `[1, 15, 29]`.

> [!example] Fall 2024 Q2(b) — is a list-indexed tensor a view?
> `y = x[:, [0, 2, 3]]`: **False**. Indexing with a list gathers elements into a new tensor, so `x` and `y` have different data pointers ([[1-foundations/indexing-and-slicing#advanced-indexing-lists-and-tensors|advanced indexing]]).

> [!trap] Common mistakes
> - Thinking `b = a.exp()` changes `a`. Only the `_` version does.
> - Thinking `b = a[mask]` or `b = a[[0, 1]]` is a view. Reading with a mask or a list **copies**; but **assigning** through one (`a[a > 15] -= 5`, `a[[0, 1]] = 0`) does write into `a`.
> - Comparing `x.data_ptr()` of a slice with the original and concluding "different memory". Compare storages, or remember that a slice starting at element $k$ points $k$ elements later.
> - Expecting `.view()` to work after `.T` or `.permute()`; use `.reshape()` or `.contiguous()` first.
> - Forgetting that `b = a.cpu()` (on a CPU tensor) and `b = a.to(a.dtype)` are the same tensor, not copies.

> [!exam] On the exams
> HW 1 and three of the four past Midterm 1s asked "view or new memory?" or "what does this tensor contain after the in-place op?" ([[6-midterm-1/fall-2024|Fall 2024]] Q2, [[6-midterm-1/spring-2025|Spring 2025]] Q1, [[6-midterm-1/spring-2026|Spring 2026]] Q2). The review lecture's summary: tensor views share the same underlying memory; operations with an underscore operate in place and do not allocate new memory. A typical true/false item: *"A tensor view allocates new memory for its data"* — **false** (see the [[6-midterm-1/true-false|T/F bank]]).

## PyTorch docs

- [Tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html) — the official list of operations that return views (the lecture points here).
- [`torch.Storage`](https://docs.pytorch.org/docs/stable/storage.html) — typed and untyped storage.
- [`Tensor.view`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html), [`torch.reshape`](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html) — the two ways to change shape.
- [`Tensor.data_ptr`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html), [`Tensor.untyped_storage`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.untyped_storage.html) — checking whether two tensors share memory.
- [`Tensor.stride`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.stride.html), [`Tensor.is_contiguous`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html), [`Tensor.contiguous`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html) — the layout.
- [`Tensor.expand`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html), [`Tensor.clone`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.clone.html) — a free repeat vs a real copy.
- [`Tensor.cpu`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cpu.html), [`Tensor.cuda`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cuda.html) — moving between devices (a copy unless already there).
- [In-place semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html#in-place-semantics) — what in-place ops may and may not broadcast.

Related: [[1-foundations/tensors]] · [[1-foundations/indexing-and-slicing]] · [[1-foundations/broadcasting]] · [[2-calculus-and-optimization/autograd]] · [[6-midterm-1/index|Midterm 1]]
