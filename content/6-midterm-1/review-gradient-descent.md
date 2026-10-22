---
title: "Review: matrix calculus and gradient descent by hand"
description: "Midterm 1 problem types 2 and 3 (about 20 pts on every past exam): the matrix-calculus identities with layouts and the explain-why answer, scalar derivatives and optimization facts, the two-step gradient-descent recipe with all four past exam problems, the HW2 step-size table and why it behaves that way, and the limits of gradient descent."
tags: [exam, midterm-1, calculus]
aliases: ["gradient descent review", "matrix calculus review"]
---

*Midterm 1 · problem types 2–3 of 7 · HW1 Q5, HW2 Q1 · Lectures 3, 4, 6 · matrix calculus 5–6 pts, gradient descent 12–15 pts on every past exam · topic pages: [[2-calculus-and-optimization/matrix-calculus|matrix calculus]], [[2-calculus-and-optimization/primal-optimization|primal optimization]], [[2-calculus-and-optimization/gradient-descent|gradient descent]] · back to [[6-midterm-1/index|Midterm 1]]*

> [!abstract] In one breath
> Every exam has a "find $\nabla f$, then take two steps by hand" problem. Write the gradient symbolically (it is its own ~6-point part), evaluate it at the **current** iterate, update every coordinate with **exact fractions**, and repeat. The step sizes are chosen so the arithmetic collapses — ugly numbers usually mean a slip. Matrix calculus enters as a one-line identity ($\partial x^\top x/\partial x=2x$, $\partial a^\top x/\partial x=a$) that you must *justify* by writing the sum.

## Matrix calculus

The gradient of a scalar with respect to a vector has **the same shape as that vector**. The Jacobian of $f:\mathbb R^n\to\mathbb R^m$ is $m\times n$ in numerator layout (the course default): row $i$ holds $\partial f_i/\partial x$. Denominator layout is its transpose. Both are correct; pick the one whose shapes multiply.

When unsure, write the sum and differentiate one entry: $a^\top x=\sum_ia_ix_i\Rightarrow\partial/\partial x_j=a_j$, so the stacked result is $a$. That is the "explain why" answer the Spring solutions wanted ([[6-midterm-1/spring-2025#problem-3--jacobians-hessians-and-why-my-brain-hurts-5-pts|Spring 2025 Q3]], [[6-midterm-1/spring-2026#problem-4--matrix-calculus-6-pts|Spring 2026 Q4]]).

| expression | derivative w.r.t. $x$ | note |
|---|---|---|
| $a^\top x$ or $x^\top a$ | $a$ | Spr 26 Q4 |
| $x^\top x=\Vert x\Vert_2^2$ | $2x$ | Spr 25 Q3 |
| $x^\top Ax$ | $(A+A^\top)x$ | $2Ax$ if $A$ is symmetric |
| $Ax$ | $A$ | $m\times n$ Jacobian |
| $y^\top Ax$ | $A^\top y$ | and $\partial/\partial y=Ax$ |
| $\tfrac12\Vert y-Ax\Vert_2^2$ | $A^\top(Ax-y)$ | the linear-regression gradient |
| $\tfrac12\Vert Ax\Vert_2^2$ | $A^\top Ax$ | L12 |
| $\Vert y-x\Vert_2^2$ | $2(x-y)$ | HW1 5(b) |
| $x\odot y$ | $\operatorname{diag}(y)$ | HW1 5(d), $4\times4$ |
| $\operatorname{Tr}(xx^\top)$ | $2x$ | it equals $x^\top x$ (L12) |
| $\Vert x\Vert_2$ | $x/\Vert x\Vert_2$ | |
| $g(x^\top x)$ | $2g'(x^\top x)\,x$ | Fall 25: $\nabla(x^\top x-3)^2=4(x^\top x-3)x$ |
| $\lambda\Vert w\Vert_2^2$ | $2\lambda w$ | ridge penalty |
| $x/\Vert x\Vert_2$ (a vector) | a $d\times d$ Jacobian | so "$\in\mathbb R^d$" is False (Fall 25 T/F) |

### Chain rule with vectors

Lecture 3 builds $h(x)=\tfrac12\Vert y-Ax\Vert_2^2$ from $a=Ax$, $b=y-a$, $c=\Vert b\Vert^2$, $d=c/2$ and multiplies the pieces (denominator layout) so the shapes line up:
$$
\frac{\partial h}{\partial x}=\underbrace{\frac{\partial a}{\partial x}}_{A^\top}\;\underbrace{\frac{\partial b}{\partial a}}_{-I}\;\underbrace{\frac{\partial c}{\partial b}}_{2(y-Ax)}\;\underbrace{\frac{\partial d}{\partial c}}_{1/2}=A^\top(Ax-y).
$$
Product rule for $a(x)^\top b(x)$: $\dfrac{\partial(a^\top b)}{\partial x}=\Big(\dfrac{\partial a}{\partial x}\Big)^{\!\top}b+\Big(\dfrac{\partial b}{\partial x}\Big)^{\!\top}a$; e.g. $a=x$, $b=Ax$ gives $Ax+A^\top x$.

### Scalar derivatives you will need

$(x^p)'=px^{p-1}$ · $(\ln x)'=1/x$ · $(e^u)'=u'e^u$ · $(\sin x)'=\cos x$ · $(\cos x)'=-\sin x$ · $(x\ln x)'=\ln x+1$ · $\sigma'(z)=\sigma(z)(1-\sigma(z))$ · $\dfrac{d}{dz}\ln(1+e^{-z})=-\sigma(-z)$.

### Optimization facts

- Stationary (critical) points have $\nabla f=0$. Second-derivative test: $f''>0$ minimum, $f''<0$ maximum, $f''=0$ inconclusive (think $x^3$).
- Convex: $f(\lambda x_1+(1-\lambda)x_2)\le\lambda f(x_1)+(1-\lambda)f(x_2)$ for $0\le\lambda\le1$, or $f''(x)\ge0$ everywhere. For a convex function every stationary point is a global minimum.
- Why not always set $\nabla f=0$? The roots can be intractable (Lecture 3: $2(x-1)^5-5(x-2)^4+20x^3=0$), and real models have millions of parameters. That motivates gradient descent.

## Gradient descent by hand

$$
x^{(k+1)}=x^{(k)}-\alpha\,\nabla f\big(x^{(k)}\big)
$$

The gradient points in the direction of steepest **ascent**, so stepping against it lowers $f$. $\alpha$ is the step size (learning rate). Every coordinate gets the same update. To maximize, use gradient ascent ($+\alpha\nabla f$).

> [!recipe] Exam recipe
> 1. Write $\nabla f$ symbolically. It is usually its own part, worth about 6 points.
> 2. Evaluate it at the current iterate $x^{(k)}$, not the new one.
> 3. Update every coordinate, keeping exact fractions or $e$.
> 4. Repeat once more for $k=2$ and box both iterates.
> 5. If you could not do part (a), use the fallback gradient the question offers (Fall 2025 gave $\nabla f=-x$).

### All four past problems

| exam | $f$ | start, $\alpha$ | $\nabla f$ | $x^{(1)}$ | $x^{(2)}$ |
|---|---|---|---|---|---|
| [[6-midterm-1/fall-2024#problem-3--gradient-descent-by-hand-15-pts\|Fall 24 Q3]] | $(x+3y)^2+(x-2)^2$ | $(0,0)$, $\tfrac18$ | $\big(2(x+3y)+2(x-2),\;6(x+3y)\big)$ | $(\tfrac12,0)$ | $(\tfrac34,-\tfrac38)$ |
| [[6-midterm-1/fall-2025#problem-3--a-gradient-and-two-steps-15-pts\|Fall 25 Q3]] | $(x^\top x-3)^2$, $x\in\mathbb R^3$ | $(1,0,0)$, $\tfrac1{16}$ | $4(x^\top x-3)\,x$ | $(\tfrac32,0,0)$ | $(\tfrac{57}{32},0,0)$ |
| [[6-midterm-1/spring-2025#problem-6--stumbling-down-the-gradient-15-pts\|Spr 25 Q6]] | $-x\ln x-y\ln y$ | $(1,1)$, $e-1$ | $(-1-\ln x,\;-1-\ln y)$ | $(e,e)$ | $(3e-2,\,3e-2)$ |
| [[6-midterm-1/spring-2026#problem-7--stumbling-down-the-gradient-12-pts\|Spr 26 Q7]] | $\tfrac1{2e}(x^2+y^2)-(\tfrac12+\tfrac1e)(x+y)$ | $(1,1)$, $2(e-1)$ | $\tfrac{x}{e}-\tfrac12-\tfrac1e$ (each coord) | $(e,e)$ | $(3-\tfrac2e,\,3-\tfrac2e)$ |

The Spring exams picked $\alpha$ so the logs and $e$'s cancel; the Fall exams picked it so the fractions stay small. Warm-up from Lecture 4: $f=(x-1)^2$, $x^{(0)}=4$, $\alpha=\tfrac14$ gives $4\to\tfrac52\to\tfrac74\to\tfrac{11}8\to\dots\to1$, halving the distance to 1 each step.

## Step size decides everything (HW2 Q1)

$f(x)=2x^2-6x+5$, $f'(x)=4x-6$, $x^*=\tfrac32$, start at 0.

| $\alpha$ | $x^{(1)},x^{(2)},x^{(3)}$ | what happens |
|---|---|---|
| $\tfrac14$ | $\tfrac32,\ \tfrac32,\ \tfrac32$ | lands on $x^*$ in one step |
| $\tfrac13$ | $2,\ \tfrac43,\ \tfrac{14}9$ | zig-zags inward, converges to $\tfrac32$ |
| $\tfrac12$ | $3,\ 0,\ 3$ | bounces between 0 and 3 forever |
| $1$ | $6,\ -12,\ 42$ | diverges |

**Why:** for a quadratic, the distance to the minimum is multiplied by the same factor every step,
$$
x^{(k+1)}-x^*=(1-\alpha f'')\,(x^{(k)}-x^*),\qquad f''=4\text{ here}.
$$
So it converges exactly when $\lvert1-4\alpha\rvert<1$, i.e. $0<\alpha<\tfrac12$: monotone for $\alpha<\tfrac14$, zig-zag for $\tfrac14<\alpha<\tfrac12$, stuck oscillating at $\tfrac12$, divergent beyond. In general: converges iff $0<\alpha<2/f''$.

## Limits of gradient descent

- It finds the global minimum (with a suitable step) only for **convex** functions. On $f=x^2+2\cos(\pi x)+x+2$ it settles in whichever local minimum lies downhill from the start (Lecture 4).
- It is **not** guaranteed to converge at all: too large a step oscillates or diverges. "Always guaranteed to converge to a locally optimal point" is False ([[6-midterm-1/fall-2025#problem-1--true-or-false-15-pts|Fall 2025 T/F d]]).
- Too small a step converges slowly.

### Ways to get derivatives (Lecture 4)

| method | pros | cons |
|---|---|---|
| numeric, $\dfrac{f(x+h)-f(x)}{h}$ | simple, only evaluates $f$ | approximate; one extra evaluation per parameter |
| symbolic | exact; derive once, evaluate many times | needs a closed form; expressions grow very large |
| backpropagation (reverse-mode autodiff) | exact; reuses earlier computation (dynamic programming); scales to millions of parameters | must store forward-pass values, which costs memory |

Practice: [[6-midterm-1/practice-problems#p4--two-steps-of-gradient-descent|P4]] · next type: [[6-midterm-1/review-computational-graphs|computational graphs]]
