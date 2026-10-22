---
title: "Midterm 1 formula sheet"
description: "Everything Midterm 1 expects you to know without looking, on one page: tensor views vs copies, masks and byte sizes; matrix-calculus identities; the gradient-descent update and step-size rule; the backprop sum and local partials; normal equations, ridge and weighted least squares with costs; logistic regression, BCE, softmax, cross-entropy and the multi-class schemes."
tags: [exam, midterm-1]
aliases: ["formula sheet", "cheat sheet", "Midterm 1 cheat sheet"]
---

*Midterm 1 · the Fall exams had no formula sheet (the Spring ones stapled one to the back), so know this cold · back to [[6-midterm-1/index|Midterm 1]]*

## Tensors

- `cat` joins an existing dim; `stack` adds one.
- **Views:** `view`, basic slicing, `.T`, `unsqueeze`, in-place ops. **Copies:** math, list indexing, masks, `cat`, `clone`.
- Masks: `a[(c1) & (c2)]`; the result is 1-D.
- Bytes per element: `uint8` 1 · `int32`/`float32` 4 · `int64`/`float64` 8.

## Matrix calculus

$$
\frac{\partial a^\top x}{\partial x}=a,\qquad \frac{\partial x^\top x}{\partial x}=2x,\qquad \frac{\partial x^\top Ax}{\partial x}=(A+A^\top)x,\qquad \frac{\partial Ax}{\partial x}=A,\qquad \frac{\partial}{\partial x}\tfrac12\Vert Ax-y\Vert^2=A^\top(Ax-y).
$$
The gradient has the shape of $x$; the Jacobian of $f:\mathbb R^n\to\mathbb R^m$ is $m\times n$.

## Gradient descent

$$
x^{(k+1)}=x^{(k)}-\alpha\,\nabla f\big(x^{(k)}\big)
$$
Quadratic: the error is multiplied by $(1-\alpha f'')$ each step, so it converges iff $0<\alpha<2/f''$. A global minimum is guaranteed only if $f$ is convex ($f''\ge0$).

## Backpropagation

$$
\bar z_i=\sum_{j\in\text{succ}(i)}\bar z_j\,\frac{\partial z_j}{\partial z_i},\qquad \bar z_{\text{seed}}=1.
$$
Local partials: $uv\to v,\,u$ · $u/v\to\tfrac1v,\,-\tfrac{u}{v^2}$ · $e^u\to e^u$ · $\ln u\to\tfrac1u$ · $\sigma\to z(1-z)$ · $u-v\to1,\,-1$.

## Linear regression

$$
\nabla\tfrac12\Vert Xw-y\Vert^2=X^\top(Xw-y),\qquad w^*=(X^\top X)^{-1}X^\top y.
$$
Ridge: $(X^\top X+\lambda I)^{-1}X^\top y$ (for $\Vert Xw-y\Vert^2+\lambda\Vert w\Vert^2$) · weighted: $(X^\top\Sigma^\top\Sigma X)^{-1}X^\top\Sigma^\top\Sigma y$. Linear regression needs the model to be linear in the **weights**. Cost: $O(Nd^2+d^3)$ closed form vs $O(Nd)$ per gradient-descent step.

## Logistic regression

$\Pr\{y\mid x\}=\sigma(y\,w^\top x)$; predict $+1$ iff $w^\top x\ge0$. $J=\sum_i\ln\!\big(1+e^{-y_iw^\top x_i}\big)$; no closed form. BCE: $-\big[y\ln f+(1-y)\ln(1-f)\big]$ with $y\in\{0,1\}$. $\sigma'(z)=\sigma(z)(1-\sigma(z))$.

## Multi-class

$z=Wx+b$, predict $\arg\max_kz_k$; softmax $e^{z_k}/\sum_je^{z_j}$ keeps the order. Cross-entropy $=-\ln f_y=-z_y+\ln\sum_je^{z_j}$. One-vs-one: $K(K-1)/2$ classifiers; one-vs-all: $K$. `nn.CrossEntropyLoss` takes logits and integer labels; `nn.BCELoss` takes probabilities.
