---
title: "Midterm 1"
description: "Everything for ECE 364 Midterm 1: logistics and scope, where the points have been on four past exams, a one-hour plan, the four past exams worked problem by problem, review pages by problem type, a true/false bank, fresh practice, the formula sheet, Spring-only topics and known errata."
tags: [exam, midterm-1]
aliases: ["Midterm 1", "MT1", "Exam 1", "exam 1 guide"]
---

*Fall 2026 Midterm 1 was **Tue Oct 6, 9:30–10:50 am** (80 min, in class) in **ECEB 3081** · scope **HW 1–3 and Lectures 2–9** · 100 pts, about 0.8 min per point · built from the four past Midterm 1s with solutions, the lecture notebooks (with the [[lectures|Lecture 12 review]]) and HW 1–3*

> [!abstract] In one breath
> Four past exams, two templates. The **Fall** exams (written by Prof. Snyder, who also writes the homework) go *tensor code → gradient descent by hand → computational graph → linear regression → (logistic regression)*, with a T/F block in Fall 2025. The **Spring** exams (another section) share the first three and add SVMs and `nn.Module`/`DataLoader` code. Every exam had a computational graph and a two-step gradient-descent problem: **30–40 points** from those two alone. The review lecture promised "less focus on code syntax, more on the concepts of what code is trying to accomplish or the mathematical concepts", in a mix of free response, fill-in-the-code, multiple choice and T/F.

## The past exams, worked

| exam | problems | page |
|---|---|---|
| Fall 2024 (Kamalabadi, Snyder) | slicing · list indexing and `y.T @ x` · GD on $(x+3y)^2+(x-2)^2$ · graph · weighted least squares | [[6-midterm-1/fall-2024\|Fall 2024]] |
| Fall 2025 (Snyder) | T/F · `cat`/`stack` and a mask · GD on $(x^\top x-3)^2$ · graph · linear-regression calls · logistic regression | [[6-midterm-1/fall-2025\|Fall 2025]] |
| Spring 2025 | slicing · a GD bug · $\partial x^\top x/\partial x$ · SVM · logic table · GD · gradient accumulation · graph | [[6-midterm-1/spring-2025\|Spring 2025]] |
| Spring 2026 | T/F · slicing · a GD bug · $\partial a^\top x/\partial x$ · SVM · logic table · GD · `Dataset`/`nn.Module` · log-softmax graph | [[6-midterm-1/spring-2026\|Spring 2026]] |

Every problem is restated with a folded worked solution that matches the official key (and says so where the key slips — see [[6-midterm-1/errata|errata]]).

## Where the points have been

| problem type | Fall 24 | Fall 25 | Spr 25 | Spr 26 | review page |
|---|---|---|---|---|---|
| computational graph: node values → local partials → adjoints | 25 | 25 | 21 | 18 | [[6-midterm-1/review-computational-graphs\|graphs]] |
| find the gradient, then two steps of gradient descent by hand | 15 | 15 | 15 | 12 | [[6-midterm-1/review-gradient-descent\|gradient descent]] |
| tensor indexing, shapes, views, boolean masks | 25 | 10 | 24 | 14 | [[6-midterm-1/review-tensors\|tensors]] |
| linear regression: gradient and closed form, build $\Phi$, choose $\lambda$ | 35 | 15 | – | – | [[6-midterm-1/review-linear-regression\|linear regression]] |
| logistic regression: which $w$ classifies, max likelihood, one-vs-all $W$ | – | 20 | – | – | [[6-midterm-1/review-logistic-regression\|logistic regression]] |
| true/false concepts | – | 15 | – | 12 | [[6-midterm-1/true-false\|T/F bank]] |
| matrix-calculus one-liner ($\partial x^\top x/\partial x$, $\partial a^\top x/\partial x$) | in Q3 | in Q3 | 5 | 6 | [[6-midterm-1/review-gradient-descent#matrix-calculus\|matrix calculus]] |
| find the bug in a PyTorch gradient-descent loop | – | – | 5 | 6 | [[6-midterm-1/review-autograd\|autograd]] |
| SVM margins, logic-table classifier, `nn.Module`/`DataLoader` code | – | – | 30 | 37 | [[6-midterm-1/spring-only-topics\|Spring-only]] |

> [!key] What this tells you
> - All four exams had a **computational-graph** problem and a **two-step gradient-descent** problem.
> - Both Fall exams follow one template: indexing code → gradient descent → computational graph → linear regression → (logistic regression).
> - The Spring exams stapled a formula sheet to the back; the Fall exams had none, so know the [[6-midterm-1/formula-sheet|formula sheet]] without looking.

## If you only have an hour

1. **20 min · graphs.** Redo [[6-midterm-1/fall-2025#problem-4--computational-graph-25-pts|Fall 2025 Q4]] from the figure without peeking, then the [[6-midterm-1/review-computational-graphs#practice-the-lecture-12-review-graph|Lecture 12 practice graph]].
2. **10 min · gradient descent.** [[6-midterm-1/fall-2024#problem-3--gradient-descent-by-hand-15-pts|Fall 2024 Q3]] and [[6-midterm-1/fall-2025#problem-3--a-gradient-and-two-steps-15-pts|Fall 2025 Q3]] on paper, with exact fractions.
3. **10 min · tensors.** The [[6-midterm-1/review-tensors#view-or-new-memory|view-vs-copy table]] and the [[6-midterm-1/review-tensors#boolean-mask-patterns|boolean-mask patterns]].
4. **10 min · linear regression.** Derive $w^*$ from the gradient, the [[6-midterm-1/review-linear-regression#the-linear-in-the-weights-test|"linear in the weights" test]], and reading $\lambda$ from a plot.
5. **10 min · logistic regression.** The sign test, the [[6-midterm-1/review-logistic-regression#the-margins-table-fall-2025-q6a-14-pts|margins table]], one-vs-all argmax.
6. **Leftover · T/F.** Run the [[6-midterm-1/true-false|T/F bank]].

## Review by problem type

1. [[6-midterm-1/review-tensors|Tensors, shapes, views and masks]] — HW1, Lecture 2
2. [[6-midterm-1/review-gradient-descent|Matrix calculus and gradient descent by hand]] — HW1 Q5, HW2 Q1, Lectures 3–4, 6
3. [[6-midterm-1/review-computational-graphs|Computational graphs and backpropagation]] — HW2 Q2–Q3, Lecture 5
4. [[6-midterm-1/review-autograd|Autograd in code and find-the-bug]] — Lectures 5, 6, 9
5. [[6-midterm-1/review-linear-regression|Linear regression, features and ridge]] — HW3 Q1–Q2, Lectures 7–8
6. [[6-midterm-1/review-logistic-regression|Logistic regression and multi-class]] — HW3 Q3–Q6, Lectures 8–9

Then: [[6-midterm-1/true-false|true/false bank]] · [[6-midterm-1/practice-problems|fresh practice problems]] · [[6-midterm-1/formula-sheet|formula sheet]] · [[6-midterm-1/spring-only-topics|Spring-only topics]] · [[6-midterm-1/errata|typos in the sources]].

> [!tip] Sources
> All in your ECE 364 folder (not copied into this site): `exam/ECE_364_Midterm_1_Fall_2024` (+ Solutions), `exam/ECE_364_Midterm_1_Fall_2025` (+ Solutions), `exam/ECE364_MT1_sp25` (+ sol), `exam/ECE364_MT1_sp26_vf` (+ sol), `exam/ECE364_Exam1_Guide.html` (the earlier guide this section is built from), `py-complete/Lecture1–12` including `L12_Midterm1_Review`, and the HW 1–3 problem PDFs.
