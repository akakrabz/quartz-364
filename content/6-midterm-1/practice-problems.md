---
title: "Practice problems in the exam's style"
description: "Ten new Midterm 1 problems with folded answers checked by computation: a view-and-slice trace, shape prediction, a negative-odd mask, two gradient-descent steps, a graph with fan-out, linear-in-the-weights calls, normal equations by hand, logistic predictions, softmax and cross-entropy, and a find-the-bug loop."
tags: [exam, midterm-1]
aliases: ["Midterm 1 practice", "fresh practice"]
---

*Midterm 1 · new problems in the format of the [[6-midterm-1/index|past exams]] · every answer checked by running it · the review pages explain the method: [[6-midterm-1/review-tensors|tensors]], [[6-midterm-1/review-gradient-descent|gradient descent]], [[6-midterm-1/review-computational-graphs|graphs]], [[6-midterm-1/review-linear-regression|linear regression]], [[6-midterm-1/review-logistic-regression|logistic regression]], [[6-midterm-1/review-autograd|autograd]]*

## P1 — Views and slicing

> [!question] P1
> ```python
> a = torch.arange(24).view(4, 6)
> b = a[1:, ::2]
> b[0, 0] = 100
> ```
> Shape of `b`? Contents of `b` before the assignment? Value of `a[1, 0]` afterwards? Is `b` a view?

> [!solution]- Answer
> `(3, 3)`; `[[6, 8, 10], [12, 14, 16], [18, 20, 22]]`; `a[1, 0] = 100`; yes — basic slicing makes a view, so the write goes through to `a`.

## P2 — Shapes

> [!question] P2
> `x = torch.rand(2, 5)`. Give the shapes of `torch.stack((x, x), dim=1)`, `torch.cat((x, x, x), dim=0)`, `x.unsqueeze(-1)`, `x.view(-1, 2)`, `x[:, 1]`, `x[x > 0.5]`.

> [!solution]- Answer
> `(2, 2, 5)`; `(6, 5)`; `(2, 5, 1)`; `(5, 2)`; `(2,)`; 1-D with a data-dependent length.

## P3 — Mask

> [!question] P3
> Fill in `d` so that `a[d]` holds every negative odd number in an integer tensor `a`.

> [!solution]- Answer
> `d = (a < 0) & (a % 2 == 1)`. Python's `%` makes `-3 % 2` equal 1, so this works; `(a < 0) & (a % 2 != 0)` also works.

## P4 — Two steps of gradient descent

> [!question] P4
> $f(x,y)=x^2+2y^2-2x$, start $(0,1)$, $\alpha=\tfrac14$. Find $\nabla f$ and the iterates at $k=1,2$.

> [!solution]- Answer
> $\nabla f=(2x-2,\;4y)$. $\nabla f(0,1)=(-2,4)\Rightarrow(\tfrac12,0)$. $\nabla f(\tfrac12,0)=(-1,0)\Rightarrow(\tfrac34,0)$. The minimizer is $(1,0)$; the $y$ coordinate got there in one step because $\alpha=\tfrac14=1/f_{yy}$.

## P5 — Graph with fan-out

> [!question] P5
> $z_1=x$, $z_2=z_1^2$, $z_3=e^{z_1}$, $z_4=z_2z_3$, $z_5=\ln z_2$, $z_6=z_4+z_5=f$. Find every adjoint and check $\bar z_1$.

> [!solution]- Answer
> $$
> \bar z_6=\bar z_5=\bar z_4=1,\qquad \bar z_3=\bar z_4z_2=z_2,\qquad \bar z_2=\bar z_4z_3+\bar z_5\frac1{z_2}=z_3+\frac1{z_2},
> $$
> $$
> \bar z_1=\bar z_2(2z_1)+\bar z_3e^{z_1}=2z_1\Big(z_3+\frac1{z_2}\Big)+z_2e^{z_1}=2xe^x+\frac2x+x^2e^x.
> $$
> Check: $f=x^2e^x+2\ln x$ (for $x>0$), so $f'=2xe^x+x^2e^x+2/x$ ✓. Both $z_1$ and $z_2$ fan out.

## P6 — Linear in the weights?

> [!question] P6
> Can each be fitted by linear regression? If so, give row $i$ of $\Phi$.
> (a) $y\approx w_1\cos x+w_2x^3+w_3$ (b) $y\approx\dfrac1{1+w_2x}$ (c) $y\approx(w_1+w_2)x$

> [!solution]- Answer
> (a) **Yes**, row $[\cos x_i,\ x_i^3,\ 1]$. (b) **No**: $w_2$ sits inside a nonlinear function. (c) Linear, but the two columns of $\Phi$ are identical, so $\Phi^\top\Phi$ is singular: only $w_1+w_2$ can be recovered.

## P7 — Normal equations by hand

> [!question] P7
> Fit $y=mx+b$ to $(0,1)$, $(1,2)$, $(2,4)$ with the normal equations.

> [!solution]- Answer
> $$
> X=\begin{bmatrix}0&1\\1&1\\2&1\end{bmatrix},\quad X^\top X=\begin{bmatrix}5&3\\3&3\end{bmatrix},\quad X^\top y=\begin{bmatrix}10\\7\end{bmatrix},\quad
> w^*=\frac16\begin{bmatrix}3&-3\\-3&5\end{bmatrix}\begin{bmatrix}10\\7\end{bmatrix}=\begin{bmatrix}3/2\\5/6\end{bmatrix}.
> $$
> So $m=\tfrac32$, $b=\tfrac56$.

## P8 — Logistic predictions

> [!question] P8
> $w=(2,-1,1)$ acting on $(x_1,x_2,1)$. Classify $(1,1)$, $(0,3)$, $(-1,-2)$, and give $\Pr\{y=+1\}$ for $(1,1)$.

> [!solution]- Answer
> $w^\top x=2,\,-2,\,1\to+1,\,-1,\,+1$. $\Pr=\sigma(2)\approx0.881$.

## P9 — Softmax and cross-entropy

> [!question] P9
> Logits $z=(2,0,-1)$. Probabilities? Cross-entropy if the true class is the second one?

> [!solution]- Answer
> $\operatorname{softmax}(z)\approx(0.844,\,0.114,\,0.042)$; $\ell_{\text{ce}}=-\ln0.114\approx2.17$ (equivalently $-z_2+\ln\sum_je^{z_j}=0+2.170$).

## P10 — Find the bug

> [!question] P10
> ```python
> w = torch.tensor(0.0, requires_grad=True)
> for k in range(100):
>     loss = (3*w - 6)**2
>     loss.backward()
>     w = w - 0.01 * w.grad
> ```
> What goes wrong, and how do you fix it?

> [!solution]- Answer
> Outside `no_grad`, `w = w - …` builds a new **non-leaf** tensor whose `.grad` stays `None`, so the second iteration fails on `None * 0.01`. (Inside `no_grad`, as in [[6-midterm-1/spring-2025#problem-2--feeling-out-of-place-5-pts|Spring 2025 Q2]], the new tensor has `requires_grad=False` and `backward()` fails instead.) The gradient is also never cleared. Fix:
> ```python
> with torch.no_grad():
>     w -= 0.01 * w.grad
>     w.grad = None
> ```
> Then $w\to2$.
