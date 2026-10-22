---
title: "Matrix calculus"
description: "Derivatives of scalar and vector functions of vectors: the shape rule, the identities for aᵀx, xᵀAx and ½‖y − Ax‖², Jacobians in numerator and denominator layout, product and chain rules with matrices, and checking answers with torch.autograd.functional.jacobian."
tags: [topic, calculus, midterm-1]
aliases: ["Jacobian", "Jacobians", "Jacobian matrix", "numerator layout", "denominator layout", "vector calculus", "matrix derivatives"]
---

*Lecture 3 · Homework 1 Q5 · [[2-calculus-and-optimization/index|Calculus and optimization]] · builds on [[2-calculus-and-optimization/gradients-and-partial-derivatives|gradients]] · used in [[3-linear-models/linear-regression|linear regression]]*

> [!abstract] In one breath
> The derivative of a **scalar** with respect to a vector $x$ has the shape of $x$; the derivative of a **vector** $f(x)\in\mathbb{R}^m$ with respect to $x\in\mathbb{R}^n$ is an $m\times n$ **Jacobian** (numerator layout, the course default) or its $n\times m$ transpose (denominator layout). Three identities do most of the work — $\nabla_x a^\top x=a$, $\nabla_x x^\top Ax=(A+A^\top)x$, $\nabla_x\tfrac12\Vert y-Ax\Vert_2^2=A^\top(Ax-y)$ — and when in doubt, write the sum and differentiate one entry.

## What it is

Partial derivatives extend from scalar variables to vectors and matrices by organizing **one partial derivative per entry**:

- scalar $f$, vector $x\in\mathbb{R}^n$: $\dfrac{\partial f}{\partial x}=\Bigl[\dfrac{\partial f}{\partial x_1},\ldots,\dfrac{\partial f}{\partial x_n}\Bigr]^\top\in\mathbb{R}^n$ — same shape as $x$, even if $x$ appears transposed in $f$;
- vector $f(x)\in\mathbb{R}^m$, vector $x\in\mathbb{R}^n$: an $m\times n$ table of $\partial f_i/\partial x_j$, the **Jacobian**.

The first example from lecture: for the dot product $f(x,y)=x^\top y=\sum_i x_iy_i$ with $x,y\in\mathbb{R}^5$, $\partial f/\partial x_i=y_i$, so $\partial f/\partial x=y$ and likewise $\partial f/\partial y=x$.

> [!definition] Definition — Jacobian
> For $f:\mathbb{R}^n\to\mathbb{R}^m$, the **Jacobian** collects the partial derivative of every output entry with respect to every input entry. In **numerator layout** (rows follow the numerator $f$):
> $$
> \frac{\partial f}{\partial x}=\begin{bmatrix}
> \frac{\partial f_1}{\partial x_1}&\cdots&\frac{\partial f_1}{\partial x_n}\\
> \vdots&\ddots&\vdots\\
> \frac{\partial f_m}{\partial x_1}&\cdots&\frac{\partial f_m}{\partial x_n}
> \end{bmatrix}\in\mathbb{R}^{m\times n}.
> $$
> **Denominator layout** is the transpose, $n\times m$ (rows follow the denominator $x$). Both hold the same numbers and both are correct; in a chain-rule product you pick the layout whose shapes multiply.

## What you should know

- **Shape rule:** $\partial(\text{scalar})/\partial x$ has the shape of $x$. $\partial(\mathbb{R}^m)/\partial(\mathbb{R}^n)$ is $m\times n$ (numerator) or $n\times m$ (denominator).
- The identity table below, especially $a^\top x\to a$, $x^\top x\to 2x$, $x^\top Ax\to(A+A^\top)x$, $Ax\to A$, $\tfrac12\Vert y-Ax\Vert_2^2\to A^\top(Ax-y)$.
- The **entrywise method**: write $f$ as a sum, differentiate with respect to $x_k$, recognize the result as a vector. This is also the "explain why" answer exams accept.
- Product rule $\dfrac{\partial(a^\top b)}{\partial x}=\Bigl(\dfrac{\partial a}{\partial x}\Bigr)^{\!\top}b+\Bigl(\dfrac{\partial b}{\partial x}\Bigr)^{\!\top}a$ (Jacobians in numerator layout).
- Chain rule for a scalar at the end: $\nabla_x L=J^\top\nabla_a L$ where $a=a(x)$ and $J=\partial a/\partial x$ is $m\times n$ — the Jacobian-transpose times the downstream gradient. This is what [[2-calculus-and-optimization/backpropagation|backpropagation]] computes node by node.
- A vector-valued function of a vector has a matrix (Jacobian) derivative, not a vector one: $y=x/\Vert x\Vert_2$ with $x\in\mathbb{R}^d$ has a $d\times d$ Jacobian.
- [`torch.autograd.functional.jacobian`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html) checks any of these numerically.

## The identity table

$x\in\mathbb{R}^n$; $a,y$ constant vectors; $A$ a constant matrix of compatible size; $g$ a scalar function.

| $f(x)$ | $\partial f/\partial x$ | note |
|---|---|---|
| $a^\top x$ or $x^\top a$ | $a$ | Spring 2026 Q4 |
| $x^\top x=\Vert x\Vert_2^2$ | $2x$ | Spring 2025 Q3 |
| $\mathrm{Tr}(xx^\top)$ | $2x$ | it equals $x^\top x$ (Lecture 12) |
| $x^\top Ax$ | $(A+A^\top)x$ | $=2Ax$ if $A$ is symmetric |
| $y^\top Ax$ | $A^\top y$ | and $\partial/\partial y=Ax$ |
| $Ax$ (vector) | $A$ | $m\times n$ Jacobian |
| $\Vert y-x\Vert_2^2$ | $2(x-y)$ | HW 1 Q5 |
| $\tfrac12\Vert y-Ax\Vert_2^2$ | $A^\top(Ax-y)$ | the [[3-linear-models/linear-regression\|least-squares]] gradient |
| $\tfrac12\Vert Ax\Vert_2^2$ | $A^\top Ax$ | Lecture 12 |
| $\lambda\Vert w\Vert_2^2$ | $2\lambda w$ | [[3-linear-models/ridge-regression\|ridge]] penalty |
| $\Vert x\Vert_2$ | $x/\Vert x\Vert_2$ | |
| $g(x^\top x)$ | $2g'(x^\top x)\,x$ | Fall 2025: $\nabla(x^\top x-3)^2=4(x^\top x-3)x$ |
| $x\odot y$ (vector) | $\mathrm{diag}(y)$ | elementwise product, HW 1 Q5 |
| $x/\Vert x\Vert_2$ (vector) | $\dfrac{1}{\Vert x\Vert_2}\Bigl(I-\dfrac{xx^\top}{\Vert x\Vert_2^2}\Bigr)$ | a $d\times d$ Jacobian |

### The entrywise method, on $x^\top Ax$

$$
x^\top Ax=\sum_{i}\sum_{j}x_iA_{ij}x_j
\quad\Longrightarrow\quad
\frac{\partial}{\partial x_k}=\sum_j A_{kj}x_j+\sum_i x_iA_{ik}=(Ax)_k+(A^\top x)_k ,
$$

so $\nabla_x(x^\top Ax)=(A+A^\top)x$. The same three lines prove every row of the table; for $a^\top x=\sum_i a_ix_i$ the derivative with respect to $x_k$ is $a_k$, so the gradient is $a$.

## Jacobians, worked

**Example 1 — $f(x)=Ax$** with $A\in\mathbb{R}^{2\times4}$, $x\in\mathbb{R}^4$. Writing out $f_1=a_{11}x_1+\cdots+a_{14}x_4$ and $f_2=a_{21}x_1+\cdots+a_{24}x_4$, the entry $\partial f_i/\partial x_j$ is $a_{ij}$:

$$
\frac{\partial f}{\partial x}=\begin{bmatrix}a_{11}&a_{12}&a_{13}&a_{14}\\a_{21}&a_{22}&a_{23}&a_{24}\end{bmatrix}=A\quad(\text{numerator layout, }2\times4).
$$

**Examples 2 and 3 — product rule.** With $a(x)=x$ (Jacobian $I$) and $b(x)=Ax$ (Jacobian $A$):

$$
\frac{\partial\,x^\top x}{\partial x}=I^\top x+I^\top x=2x,
\qquad
\frac{\partial\,x^\top Ax}{\partial x}=I^\top(Ax)+A^\top x=(A+A^\top)x .
$$

The second term of the product rule is written as $\bigl(\partial b/\partial x\bigr)^\top a$ rather than $a^\top\,\partial b/\partial x$ so that both terms are column vectors.

**Example 4 — chain rule.** $h(x)=\tfrac12\Vert y-Ax\Vert_2^2$ with $x\in\mathbb{R}^n$, $A\in\mathbb{R}^{m\times n}$. Decompose $a=Ax$, $b=y-a$, $c=\Vert b\Vert_2^2$, $d=\tfrac12c$:

$$
\frac{\partial a}{\partial x}=A,\quad\frac{\partial b}{\partial a}=-I_m,\quad\frac{\partial c}{\partial b}=2b,\quad\frac{\partial d}{\partial c}=\tfrac12 .
$$

Multiplying in the order $\frac{\partial a}{\partial x}\frac{\partial b}{\partial a}\frac{\partial c}{\partial b}\frac{\partial d}{\partial c}$ only works if $\partial a/\partial x$ is in **denominator layout** ($A^\top$, $n\times m$). That is not a hack: $A^\top$ times the $m$-vector $\partial h/\partial a$ adds up, for each $x_j$, the contributions $\sum_i\frac{\partial a_i}{\partial x_j}\frac{\partial h}{\partial a_i}$ — exactly the chain rule.

$$
\frac{\partial h}{\partial x}=A^\top(-I)\,2(y-Ax)\,\tfrac12=A^\top(Ax-y).
$$

Setting this to zero gives the normal equations of [[3-linear-models/multiple-linear-regression|least squares]], $A^\top Ax=A^\top y$.

> [!intuition] Which layout?
> Numerator layout is the natural way to *write* a Jacobian ($f(x)=Ax$ has Jacobian $A$). When the function ends in a scalar loss, the gradient you want is a column the shape of $x$, and every chain-rule factor gets transposed: $\nabla_xL=J_1^\top J_2^\top\cdots\nabla L$. If the shapes do not multiply, you have the wrong layout somewhere.

## Checking with PyTorch

[`torch.autograd.functional.jacobian(func, inputs)`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html) evaluates the Jacobian of a Python function at a point. Lecture 3's check of the four examples, with $x=(0,1,2,3)$:

```python
A = torch.tensor([[1, 2, 3, 4],
                  [-1, 2, -3, 4]]).float()
x = torch.arange(4).float()
J = torch.autograd.functional.jacobian

print(J(lambda x: A @ x, x))           # Example 1: equals A
print(J(lambda x: torch.dot(x, x), x)) # Example 2: equals 2x
```

```
tensor([[ 1.,  2.,  3.,  4.],
        [-1.,  2., -3.,  4.]])
tensor([0., 2., 4., 6.])
```

With a $4\times4$ $A$ for Example 3 and a $6\times4$ $A$ and $y\in\mathbb{R}^6$ for Example 4, the hand formulas $(A+A^\top)x$ and $A^\top(Ax-y)$ matched autograd exactly (`[14, 40, 34, 60]` and `[51, 510, 65, 736]`). For a scalar function, `jacobian` returns a vector the shape of $x$ — the gradient.

> [!trap] Common mistakes
> - $\nabla_x(x^\top Ax)=2Ax$ only when $A$ is symmetric. In general it is $(A+A^\top)x$.
> - $\nabla_x(y^\top Ax)$ is $A^\top y$, not $Ay$ or $y^\top A$ (a row). Check the shape: it must be $n\times1$ like $x$.
> - A vector-valued function has a matrix derivative. "$\partial(x/\Vert x\Vert_2)/\partial x\in\mathbb{R}^d$" is False (Fall 2025 T/F).
> - Dropping the $-I$ from $b=y-a$: the answer is $A^\top(Ax-y)$, not $A^\top(y-Ax)$.

> [!exam] On the exams
> Matrix calculus is worth 5–6 points directly (a one-liner such as "find $\nabla_x a^\top x$ and explain") and feeds the gradient-descent and linear-regression problems: Fall 2025 Q3 needs $\nabla(x^\top x-3)^2=4(x^\top x-3)x$, and every linear-regression derivation needs $A^\top(Ax-y)$. See [[6-midterm-1/fall-2025|Fall 2025]], [[6-midterm-1/spring-2025|Spring 2025]], [[6-midterm-1/spring-2026|Spring 2026]] and the [[6-midterm-1/formula-sheet|formula sheet]].

> [!question]- Practice (Lecture 12 review): $\mathrm{Tr}(xx^\top)$ and $\tfrac12\Vert Ax\Vert_2^2$
> (a) $f(x)=\mathrm{Tr}(xx^\top)$. The diagonal of $xx^\top$ is $x_1^2,\ldots,x_n^2$, so $f=\sum_ix_i^2=x^\top x$ and $\partial f/\partial x=2x$.
>
> (b) $f(x)=\tfrac12\Vert Ax\Vert_2^2=\tfrac12x^\top A^\top Ax$. With the symmetric matrix $A^\top A$ in the $x^\top Mx$ rule, $\partial f/\partial x=\tfrac12\cdot2A^\top Ax=A^\top Ax$.

## PyTorch docs

- [`torch.autograd.functional.jacobian`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html) — Jacobian of a function at a point (returns numerator layout: output shape then input shape).
- [`torch.autograd.functional.hessian`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.functional.hessian.html) — second derivatives of a scalar function.
- [`torch.dot`](https://docs.pytorch.org/docs/stable/generated/torch.dot.html), [`torch.matmul`](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html) (the `@` operator), [`torch.linalg.vector_norm`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html), [`torch.trace`](https://docs.pytorch.org/docs/stable/generated/torch.trace.html), [`torch.outer`](https://docs.pytorch.org/docs/stable/generated/torch.outer.html) — the operations in the identities.
- [`torch.autograd`](https://docs.pytorch.org/docs/stable/autograd.html) — the package overview, including the functional API.

Related: [[2-calculus-and-optimization/gradients-and-partial-derivatives]] · [[2-calculus-and-optimization/backpropagation]] · [[1-foundations/linear-algebra-in-pytorch]] · [[3-linear-models/multiple-linear-regression]] · [[3-linear-models/ridge-regression]]
