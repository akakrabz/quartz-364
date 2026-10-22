---
title: "Typos and slips in the sources"
description: "Known errors in the official Midterm 1 solutions and lecture notebooks, so they don't trip you up: local partials written as full derivatives (Spring 2025 Q8, Spring 2026 Q9), a wrong entry in the Fall 2025 score table, a mask that admits 0, a sign error in the multi-class objective, the L∞ norm, and several smaller slips — each with the correct version."
tags: [exam, midterm-1]
aliases: ["errata", "typos in the solutions"]
---

*Midterm 1 · checked against the official solution PDFs and the lecture notebooks · back to [[6-midterm-1/index|Midterm 1]]*

> [!abstract] In one breath
> The official keys are right where it counts (every final answer below still holds), but two of them write a **full derivative where a local partial is asked**, which is exactly the mistake graders take points for. Learn the corrected versions.

## Official exam solutions

| where | the key says | correct | why it matters |
|---|---|---|---|
| [[6-midterm-1/spring-2026#problem-9--the-computational-graph-that-ate-my-sanity-18-pts\|Spring 2026 Q9(b)]] | $\partial w_9/\partial w_1=(e^y+e^z)/S$ | $1$ | $w_9=w_1-w_8$, so the *local* partial is 1. $(e^y+e^z)/S$ is the full derivative $\partial g/\partial x=\bar w_1$; the key's final adjoint $1-e^x/S$ is right. |
| [[6-midterm-1/spring-2025#problem-8--the-computational-graph-that-ate-my-sanity-21-pts\|Spring 2025 Q8(b)]] | $\partial w_6/\partial w_2=1-12\sin^2x+1/\sin x$ | $1$ | Same slip: $w_6=w_2-w_4+w_5$, so the local partial is 1. The expression is $\bar w_2$; the key's adjoints in (c) are right. |
| [[6-midterm-1/fall-2025#problem-6--logistic-regression-20-pts\|Fall 2025 Q6(a)(ii)]] | score table row $w_2$: $1,1,1,-2,-2,-4$ | $1,1,3,-2,-2,-4$ | $w_2^\top x_3=(-1,-1)\cdot(0,-3)=3$. The conclusion ($w_3$ maximizes the likelihood) is unaffected. |
| [[6-midterm-1/fall-2025#problem-2--cat-stack-and-a-mask-10-pts\|Fall 2025 Q2(c)]] | accepts `(a >= 0) & (a % 5 == 0)` | `(a > 0) & (a % 5 == 0)` | `>=` also keeps 0, which is a multiple of 5 but not a positive one. |
| [[6-midterm-1/fall-2025#problem-6--logistic-regression-20-pts\|Fall 2025 Q6(a)]] | "Input $x\in\mathbb R^3$" | $x\in\mathbb R^2$ | Each input has two entries. |
| [[6-midterm-1/fall-2024#problem-3--gradient-descent-by-hand-15-pts\|Fall 2024 Q3]] | $(x^{(k+1)},y^{(k+1)})=(x^{(k+1)},y^{(k+1)})-\alpha\nabla f(x,y)$ | right side $(x^{(k)},y^{(k)})$, gradient at $(x^{(k)},y^{(k)})$ | The gradient is evaluated at the current iterate. |
| [[6-midterm-1/spring-2025#problem-2--feeling-out-of-place-5-pts\|Spring 2025 Q2]] | `x_gd -= alpha*w_gd` | `x_gd -= alpha*x_gd.grad` | There is no `w_gd`; the explanation around it is right. |
| [[6-midterm-1/spring-2025#problem-7--torchnn-models-dataloaders-and-optimizers-10-pts\|Spring 2025 Q7]] | `train_loader, _ = split_dataset(dataset)` | `DataLoader(dataset, batch_size=8, ...)` | `split_dataset` is never defined; the batch size of 8 is the point. |
| [[6-midterm-1/spring-2026#problem-8--torchnn-models-dataloaders-and-optimizers-15-pts\|Spring 2026 Q8(i)]] | `torch.tensor([area, bedrooms, age])` | add `dtype=torch.float32` | Without it `x` is `int64`, and `x @ self.w` with float parameters raises a dtype error. Also, `lr=0.01` on these raw features diverges within one epoch; real code would standardize first. |
| [[6-midterm-1/spring-2026#problem-3--feeling-out-of-place-6-pts\|Spring 2026 Q3]] (problem statement) | "`x_gd` keeps becoming infinite" | it oscillates | Simulated (float32 and float64), the un-cleared gradient acts like undamped momentum: $x$ swings between about 3.35 and 6.0 and never settles. The fix (`x_gd.grad = None`) is the same. |
| Spring 2026 point values | — | the listed problem points add to 105 | $12+14+6+6+12+10+12+15+18$. |

## Lecture notebooks

| where | the notebook says | correct |
|---|---|---|
| Lecture 9 and the Lecture 12 review, multi-class objective | $\min_{W,b}-\sum_i\ell_{\text{ce}}(f_W(x_i),y_i)$ | $\min_{W,b}+\sum_i\ell_{\text{ce}}$: $\ell_{\text{ce}}=-\sum_k\mathbb 1\{y=k\}\log f_k$ already contains the minus sign. |
| Lecture 2, norms | $\Vert v\Vert_\infty=\max_iv_i$ | $\max_i\lvert v_i\rvert$: for $(-5,1)$ the $L_\infty$ norm is 5, not 1. |

## The earlier Exam 1 guide

This section was built from that guide; where it and the PDFs differed, the PDFs won:

- The guide's Fall 2025 Q4 has a part "(d) Check". The exam has only (a)–(c) (8 + 5 + 12 points); the check is still worth doing, and it is kept here as a note.
- The two key errors in Spring 2025 Q8(b) and Fall 2025 Q6(a)(ii), the `>=` mask in Fall 2025 Q2(c), and the Spring 2026 Q3 and Q8 issues above were not in the guide.
