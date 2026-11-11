---
title: "Typos and slips in the Midterm 2 sources"
description: "Known errors and loose ends in the official Midterm 2 solutions, exam statements and the Lecture 21–22 notebooks, each with the correct version: missing super().__init__() calls and a wrong keyword in the Spring keys, a renamed variable, an argument name that does not match, the SGD learning-rate default, a formula-sheet index, mislabelled axes, a positional encoding that gives every token the same vector, and a padding amount that is the remainder instead of its complement."
tags: [exam, midterm-2]
aliases: ["Midterm 2 errata", "Midterm 2 typos"]
---

*Midterm 2 · checked against the official solution PDFs and the lecture notebooks; the code items were confirmed by running PyTorch 2.15 (`tools/verify_mt2_torch.py`) · Midterm 1's list is [[6-midterm-1/errata|here]] · back to [[9-midterm-2/index|Midterm 2]]*

> [!abstract] In one breath
> **No official final answer is wrong.** Every true/false key, every number and every conclusion in the four solution sets was recomputed and holds. What follows are slips in the **code** of the Spring keys (they would not run as printed), a few mismatched names, and two quirks in the Lecture 21 notebook that are worth understanding because they are about shapes.

## Official exam solutions and statements

| where | the source says | correct | why it matters |
|---|---|---|---|
| [[9-midterm-2/spring-2025#problem-7--attention-is-all-you-need-15-pts\|Spring 2025 Q7]], skeleton and key | `__init__` starts with `self.d_model = d_model` | `super().__init__()` first | Without it, assigning the first `nn.Linear` raises `AttributeError: cannot assign module before Module.__init__() call`. |
| Spring 2025 Q7, key | `val_states = self.wv(V)`, later `attn_weights @ value_states` | one name for both | `NameError` as printed. |
| Spring 2025 Q7, statement | Step 3: "$\text{Attention}(Q, K, W)$" | $\text{Attention}(Q, K, V)$ | A typo in the problem text. |
| [[9-midterm-2/spring-2026#problem-7--attention-is-all-you-need-15-pts\|Spring 2026 Q7]], skeletons and keys | no `super().__init__()` in `FeedforwardLayer` or `TransformerEncoder` | add it | Same error as above. |
| Spring 2026 Q7(b), key | `MultiheadAttention(embed_dim=d, num_heads=num_heads)` | `MultiheadAttention(d, num_heads)` | The class given in the problem takes `(d, num_heads)`; `embed_dim` is the keyword of PyTorch's own `nn.MultiheadAttention`, and with the given class the call raises a `TypeError`. |
| [[9-midterm-2/spring-2026#problem-2--race-to-the-bottom-10-pts\|Spring 2026 Q2]], figure | axes labelled $\theta_1$ and $\theta_2$ | $\theta_0$ and $\theta_1$ | The model is $\hat y = \theta_0 + \theta_1x$. Nothing depends on it. |
| [[9-midterm-2/spring-2026#problem-3--lifes-a-pool-dive-in-10-pts\|Spring 2026 Q3]] | "the convolution" | cross-correlation, as `nn.Conv2d` computes | Not an error here: the given kernel is unchanged by a 180° rotation, so both give $[[0, 2], [1, -1]]$. With an unsymmetric kernel the two would differ. |
| [[9-midterm-2/fall-2024#problem-2--datasets-dataloaders-and-sgd-20-pts\|Fall 2024 Q2(b)]], statement and key | "`train_indices`" in the note and in the key's code | `training_indices` | The function's argument is named `training_indices`. The key says syntax is not graded strictly. |
| Fall 2024 Q2(c)(iii), key | learning rate: **Required** | required in older PyTorch; now optional | Current `torch.optim.SGD` has `lr=0.001` as a default (confirmed: `torch.optim.SGD(model.parameters())` runs), and the Fall 2025 formula sheet prints that signature. Answer from the signature you are given. |
| [[9-midterm-2/fall-2025#the-formula-sheet-that-came-with-the-exam\|Fall 2025 formula sheet]] | $\text{Concatenate}(\text{Attention}_1, \text{Attention}_2, \dots, \text{Attention}_2)$ | $\dots, \text{Attention}_h$ | The last index should be the number of heads. |
| [[9-midterm-2/fall-2025#problem-4--attention-15-pts\|Fall 2025 Q4(a)]], key | multiplying by $-\infty$ makes negative scores $+\infty$ "and thus have maximal weight" | right, and more can go wrong | A zero score times $-\infty$ is undefined, and in PyTorch a row containing $+\infty$ comes out of the softmax as NaN. The key's answer (it does not work) stands. |

## Course page

| where | it says | correct |
|---|---|---|
| schedule line for the exam | "11/17/2025 Midterm 2 (in class)" | **Tuesday, Nov 17, 2026**: the year is a leftover; this is the Fall 2026 offering, and the lecturer said on Nov 5 that the exam is "a week from Tuesday", ahead of fall break. |

## Lecture 21 and 22 notebooks

| where | the notebook does | effect | fix |
|---|---|---|---|
| L21 `TransformerTextClassifier`, through `PositionalEncoding('learned', …)` | indexes positions with `torch.arange(len(x))`; here `x` has shape `(1, L+1)`, so `len(x)` is 1 | every token gets the **same** position vector (row 0): the classifier has no word-order information. Confirmed by running the class: reversing the words of an input leaves the logits unchanged. | `torch.arange(x.size(1))`, as the lecture's `ViTPositionalEncoding` does. See [[7-transformers/text-classification-with-transformers#the-model\|the note on the model]]. |
| L21 `VisionTransformer.tokenize_image` | pads by `W % token_size` (the remainder $r$) | reaching the next multiple needs $P - r$; for remainders below $P/2$ the last real columns are dropped. No effect on the $160\times160$ training crops. | pad by `(-W) % token_size`. See [[7-transformers/vision-transformer#the-lectures-implementation\|the warning]]. |
| L21 text before the ImageWoof cells | "loads the ImageWoof dataset with 5 classes" | the code sets `n_classes = 4`, and the plot's baseline is $1/4$ | read 4. |
| L21 `class DoggieDataset(nn.Module)` | subclasses `nn.Module` | works, because a `DataLoader` only needs `__len__` and `__getitem__`; conventionally it would subclass `Dataset` | |
| L22 text | `tensorboard --logdir==logs` | doubled `=` | `tensorboard --logdir=logs` |
| L22 experiment cell as saved | `n_head = 1`, `dropout = 0.1` | the six shipped log folders all end in `dropout_00`: the runs on the site's plots used no dropout; the cell shows the settings of a last demonstration run that was stopped | |
