---
title: "Calculus and optimization"
description: "Unit 2 (Lectures 3–6): gradients and Jacobians, optimization problems and convexity, gradient descent, computational graphs, backpropagation, PyTorch autograd, and fitting a model by gradient descent."
tags: [unit, calculus, midterm-1]
aliases: ["unit 2", "calculus", "autodiff"]
---

*Lectures 3–6 · Homework 1 Q5 and Homework 2 · on Midterm 1 · after [[1-foundations/index|Foundations]], before [[3-linear-models/index|Linear models]]*

Every model in this course is trained the same way: write down an **objective function** that measures how bad the model is, then follow its **negative gradient** downhill. This unit builds that machine from the calculus up. Lecture 3 differentiates functions of vectors (gradients, Jacobians) and poses optimization problems. Lecture 4 replaces "set the gradient to zero" by **gradient descent**. Lecture 5 shows how a computer gets exact gradients of any composition of simple functions: draw the **computational graph** and run **backpropagation**. Lecture 6 lets PyTorch's **autograd** do the backward pass and fits a model to data — the first machine-learning problem of the course.

Together this unit carries roughly 40 of Midterm 1's 100 points: every past exam had a two-step gradient-descent problem and a computational-graph problem.

## Pages, in order

1. [[2-calculus-and-optimization/gradients-and-partial-derivatives|Gradients and partial derivatives]] — partial derivatives, the gradient vector, steepest ascent, and a first autograd check (L3).
2. [[2-calculus-and-optimization/matrix-calculus|Matrix calculus]] — derivatives with respect to vectors, the identities you must know, Jacobians in numerator and denominator layout, and chain rule with matrices (L3).
3. [[2-calculus-and-optimization/primal-optimization|Primal optimization]] — objective functions, critical points, the second-derivative test, convexity, and why "set ∇f = 0" is not enough (L3, L4).
4. [[2-calculus-and-optimization/gradient-descent|Gradient descent]] — the update rule, two steps by hand, how the step size decides convergence, local minima (L4).
5. [[2-calculus-and-optimization/computational-graphs|Computational graphs]] — breaking a function into nodes, the forward pass, local partials, and how to read the exam figures (L5).
6. [[2-calculus-and-optimization/backpropagation|Backpropagation]] — adjoints, the backward pass, summing over successors; numeric vs symbolic vs reverse-mode differentiation (L4, L5).
7. [[2-calculus-and-optimization/autograd|Autograd]] — `requires_grad`, `backward()`, `.grad`, `no_grad`, `detach`, and the bugs exam questions are built on (L3, L5, L6).
8. [[2-calculus-and-optimization/fitting-models-with-gradient-descent|Fitting models with gradient descent]] — MSE objectives, θ notation, the manual training loop in NumPy and in PyTorch (L6).

## The one-line version of each lecture

| Lecture | Title | Big idea |
|---|---|---|
| 3 | Matrix Calculus and Primal Optimization | $\nabla_x(a^\top x)=a$, $\nabla_x(x^\top Ax)=(A+A^\top)x$, $\nabla_x\tfrac12\Vert y-Ax\Vert_2^2=A^\top(Ax-y)$; optimization means finding where $\nabla f=0$ |
| 4 | Automatic Differentiation I: Gradient Descent | $x^{(k+1)}=x^{(k)}-\alpha\nabla f(x^{(k)})$; converges for small enough $\alpha$; global minimum only if convex |
| 5 | Automatic Differentiation II: Computational Graphs and Backpropagation | forward pass stores values; backward pass computes adjoints $\bar z_i=\sum_j \bar z_j\,\partial z_j/\partial z_i$ |
| 6 | Automatic Differentiation III: Combining Gradient Descent and Backpropagation | loss is the seed node; `loss.backward()` + an update under `torch.no_grad()` = learning |

> [!exam] On Midterm 1
> The gradient-descent problem (12–15 pts) and the computational-graph problem (18–25 pts) appeared on all four past exams; matrix-calculus one-liners and an autograd "find the bug" question appear on some. Worked versions: [[6-midterm-1/fall-2024|Fall 2024]], [[6-midterm-1/spring-2025|Spring 2025]], [[6-midterm-1/fall-2025|Fall 2025]], [[6-midterm-1/spring-2026|Spring 2026]]; the formulas are on the [[6-midterm-1/formula-sheet|formula sheet]].

Next unit: [[3-linear-models/index|Linear models]], where the same machinery fits lines, polynomials and classifiers.
