---
title: "Foundations: ML and PyTorch tensors"
description: "Unit 1 (Lectures 1–2, HW 1): what machine learning is, and the PyTorch tensor toolkit every later unit uses — creating tensors, dtypes and memory, storage and views, indexing and boolean masks, broadcasting, and linear algebra."
tags: [unit, foundations, midterm-1]
aliases: ["foundations", "unit 1", "PyTorch basics"]
---

*Lectures 1–2 · HW 1 · Midterm 1 scope · next unit: [[2-calculus-and-optimization/index|Calculus and optimization]]*

Lecture 1 defines the problem the whole course solves: learn a function $f_\theta$ from examples so it generalizes to new ones. Lecture 2 introduces the data structure every model, dataset and gradient lives in, the PyTorch **tensor**, and reviews the linear algebra done with it. Most Midterm 1 points from this unit are bookkeeping you can practice: predict a shape, a value, a memory size, or whether two tensors share memory. Lecture 2 also reviews derivatives and the chain rule; that material is on [[2-calculus-and-optimization/gradients-and-partial-derivatives|gradients and partial derivatives]].

1. [[1-foundations/what-is-machine-learning|What is machine learning?]] — supervised vs unsupervised, naming $x_i$ and $y_i$, the four ingredients, and the pretrained ResNet demo.
2. [[1-foundations/tensors|Tensors]] — creation, default dtypes, bytes per element and memory size, `cat` vs `stack`, `unsqueeze`, NumPy conversion, elementwise math.
3. [[1-foundations/storage-and-views|Storage and views]] — which operations share memory and which copy; in-place `_` ops; `view` vs `reshape`.
4. [[1-foundations/indexing-and-slicing|Indexing and slicing]] — `start:stop:step`, list indexing, boolean masks with `&` and `|`, masked assignment, `torch.where`.
5. [[1-foundations/broadcasting|Broadcasting]] — the right-to-left shape rule, and the `(N,)` vs `(N, 1)` bug.
6. [[1-foundations/linear-algebra-in-pytorch|Linear algebra in PyTorch]] — dot product, norms, cosine similarity, outer, matrix and Hadamard products, the inverse.

> [!exam] On the exams
> Every past Midterm 1 opened with a 10–25 point tensor problem built from pages 2–5: shapes after `cat`/`stack`/slicing, a boolean mask to fill in, view-or-copy, and values after an in-place update. See the [[6-midterm-1/index|Midterm 1 hub]] and the [[6-midterm-1/formula-sheet|formula sheet]].

> [!tip] Check yourself
> Can you, without running anything: give the shape of `torch.stack((a, a, a), dim=-1)` for `a` of shape $(4, 10)$; the bytes in `torch.zeros(1000, 3, 64, 64)`; whether `a[:, [0, 2]]` is a view; the line selecting the positive multiples of 5; and the shape of `(2, 3) + (2, 1)`? Answers: $(4, 10, 3)$; $4\cdot 1000\cdot3\cdot64\cdot64 = 49{,}152{,}000$ bytes; no; `a[(a > 0) & (a % 5 == 0)]`; $(2, 3)$.
