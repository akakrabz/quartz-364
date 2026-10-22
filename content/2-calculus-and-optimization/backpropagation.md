---
title: "Backpropagation"
description: "Reverse-mode automatic differentiation on a computational graph: the adjoint z̄ᵢ = ∂f/∂zᵢ, the rule that sums one term per successor, a complete worked example with numbers, exam-style graphs, vanishing gradients through stacked sigmoids, and how backpropagation compares with numeric and symbolic differentiation."
tags: [topic, calculus, midterm-1]
aliases: ["backpropagation", "backprop", "backward pass", "adjoint", "adjoints", "reverse-mode automatic differentiation", "reverse accumulation", "automatic differentiation", "vanishing gradient", "vanishing gradients"]
---

*Lectures 4–5 · Homework 2 Q2–Q3 · [[2-calculus-and-optimization/index|Calculus and optimization]] · builds on [[2-calculus-and-optimization/computational-graphs|computational graphs]] · implemented by [[2-calculus-and-optimization/autograd|autograd]]*

> [!abstract] In one breath
> After the [[2-calculus-and-optimization/computational-graphs|forward pass]] has stored every node's value, the **backward pass** walks the graph from the seed node to the inputs and computes each node's **adjoint** $\bar z_i=\partial f/\partial z_i$: start with $\bar z_{\text{seed}}=1$, and at every node add up, over its successors, (successor's adjoint) × (local partial). The adjoints of the input nodes are the gradient. One forward pass plus one backward pass gives the exact derivative with respect to **every** input, which is why all deep-learning frameworks use it.

## What it is

> [!definition] Definition — adjoint
> For a computational graph with output (seed) $f$, the **adjoint** of node $z_i$ is
> $$
> \bar z_i=\frac{\partial f}{\partial z_i}=\sum_{z_j\in\text{successors}(z_i)}\bar z_j\,\frac{\partial z_j}{\partial z_i},\qquad \bar z_{\text{seed}}=\frac{\partial f}{\partial f}=1 .
> $$
> Each term is the chain rule along one outgoing arrow; the sum is the **law of total derivatives** — $f$ depends on $z_i$ through every path leaving it.

**Backpropagation** = one forward pass (compute and store all values) + one backward pass (compute adjoints in reverse topological order). This is **reverse accumulation**: derivatives are accumulated backwards from the seed. Because every adjoint reuses its successors' adjoints, it is a form of **dynamic programming** — nothing is recomputed.

## What you should know

- The adjoint formula, and that $\bar z_{\text{seed}}=1$.
- **Fan-out rule:** a node with $k$ outgoing arrows gets $k$ terms in its adjoint. Missing one is the most common lost point.
- Local partials are evaluated at the **stored forward values**; that is why the forward pass must come first.
- The adjoints of the input nodes equal the partial derivatives of $f$ — always check them against a direct derivative.
- Adjoints of nodes are full derivatives $\partial f/\partial z_i$; arrow labels in part (b) are local partials $\partial z_j/\partial z_i$. Do not mix them.
- Numeric differentiation is approximate and needs one extra evaluation per parameter; symbolic differentiation blows up in size; backpropagation is exact and cheap but stores every intermediate value (memory).
- Repeated multiplication by small local partials (sigmoid: $\sigma'\le\tfrac14$) makes gradients **vanish**; a skip connection adds a path that bypasses them.

> [!recipe] Recipe — the four-part exam problem
> 1. **Node values.** Read each node off its outgoing label; substitute back to the original variables.
> 2. **Local partials.** One per arrow, $\partial(\text{child})/\partial(\text{parent})$, other inputs held fixed, written in node variables.
> 3. **Adjoints.** Seed $=1$, then right to left. At a node with several outgoing arrows, sum one term per arrow.
> 4. **Check.** The input's adjoint must equal $df/dx$ computed directly.

## Worked: $f(x,y)=x^2\cos(xy)+1$

The graph from [[2-calculus-and-optimization/computational-graphs|computational graphs]]: $z_3=z_1^2$, $z_4=z_1z_2$, $z_5=\cos z_4$, $z_6=z_3z_5$, $z_7=z_6+1$.

$$
\begin{aligned}
\bar z_7&=1, & \bar z_6&=\bar z_7\cdot1=1,\\
\bar z_5&=\bar z_6\,\frac{\partial z_6}{\partial z_5}=z_3, & \bar z_3&=\bar z_6\,\frac{\partial z_6}{\partial z_3}=z_5,\\
\bar z_4&=\bar z_5\,\frac{\partial z_5}{\partial z_4}=-z_3\sin z_4, & \bar z_2&=\bar z_4\,\frac{\partial z_4}{\partial z_2}=-z_1z_3\sin z_4,\\
\bar z_1&=\bar z_3\,\frac{\partial z_3}{\partial z_1}+\bar z_4\,\frac{\partial z_4}{\partial z_1}=2z_1z_5-z_2z_3\sin z_4 . && 
\end{aligned}
$$

$z_1$ has two successors ($z_3$ and $z_4$), so $\bar z_1$ has two terms. Substituting back: $\bar z_1=2x\cos(xy)-x^2y\sin(xy)=\partial f/\partial x$ and $\bar z_2=-x^3\sin(xy)=\partial f/\partial y$. ✓

![[l05-backprop-adjoint.png|Adjoints (green) accumulated backwards through the graph]]

**With numbers**, at the forward-pass point $(2,\tfrac\pi2)$ where $z=(2,\tfrac\pi2,4,\pi,-1,-4,-3)$:

$$
\bar z_7=\bar z_6=1,\quad \bar z_5=z_3=4,\quad \bar z_3=z_5=-1,\quad \bar z_4=-4\sin\pi=0,\quad \bar z_2=0,\quad \bar z_1=2(2)(-1)-0=-4 .
$$

Direct check: $\partial f/\partial x=2x\cos(xy)-x^2y\sin(xy)=4(-1)-0=-4$ and $\partial f/\partial y=-x^3\sin(xy)=0$. ✓

![[l05-backprop-full.png|The full backward pass: local partials (red), adjoints (green), and the final gradient (blue)]]

## Worked exercise with numbers

Lecture 5's exercise $g(x,y)=xye^{xy}$, graph $z_3=z_1z_2$, $z_4=e^{z_3}$, $z_5=z_3z_4$ (figure on [[2-calculus-and-optimization/computational-graphs#exercise-a-graph-with-fan-out|computational graphs]]):

$$
\bar z_5=1,\qquad \bar z_4=\bar z_5\,z_3=z_3,\qquad
\bar z_3=\underbrace{\bar z_4\,e^{z_3}}_{\text{via }z_4}+\underbrace{\bar z_5\,z_4}_{\text{via }z_5}=z_3e^{z_3}+e^{z_3},
$$

$$
\bar z_1=\bar z_3\,z_2=y e^{xy}(xy+1)=ye^{xy}+xy^2e^{xy},\qquad \bar z_2=\bar z_3\,z_1=xe^{xy}+x^2ye^{xy},
$$

matching the hand-computed $\partial g/\partial x$ and $\partial g/\partial y$. At $(x,y)=(1,2)$: $z_3=2$, $z_4=e^2$, $\bar z_4=2$, $\bar z_3=3e^2\approx22.17$, $\bar z_1=6e^2\approx44.33$, $\bar z_2=3e^2\approx22.17$ — exactly what autograd printed in lecture:

```python
x = torch.tensor([1.], requires_grad=True)
y = torch.tensor([2.], requires_grad=True)
z1, z2 = x, y
z3 = z1*z2
z4 = torch.exp(z3)
z5 = z3*z4
z3.retain_grad(); z4.retain_grad()   # keep intermediate .grad
z5.backward()
print(z1.grad, z2.grad, z3.grad, z4.grad)
```

```
tensor([44.3343]) tensor([22.1672]) tensor([22.1672]) tensor([2.])
```

## Worked: Fall 2025 Q4 (three-way fan-out)

Graph: $a_1=x$, $a_2=a_1^2$, $a_3=e^{a_1}$, $a_4=a_1a_3$, $a_5=a_2a_3$, $a_6=a_3+a_4+a_5=f$, so $f(x)=x^2e^x+xe^x+e^x$.

$$
\bar a_6=1,\quad \bar a_5=\bar a_4=1,\quad
\bar a_3=\bar a_4a_1+\bar a_5a_2+\bar a_6\cdot1=a_1+a_2+1,\quad \bar a_2=\bar a_5a_3=a_3,
$$

$$
\bar a_1=\bar a_2(2a_1)+\bar a_3e^{a_1}+\bar a_4a_3=2a_1a_3+e^{a_1}(a_1+a_2+1)+a_3 .
$$

Check: $\bar a_1=2xe^x+e^x(x+x^2+1)+e^x=e^x(x^2+3x+2)=\frac{d}{dx}(x^2e^x+xe^x+e^x)$. ✓ Both $a_1$ (three successors) and $a_3$ (three successors) fan out. Full problem: [[6-midterm-1/fall-2025|Fall 2025]].

## Practice: the Lecture 12 review graph

$f(x,y)=2\cos^2(xy)+\ln(\cos(xy))$ with the unlabeled graph from [[2-calculus-and-optimization/computational-graphs#reading-exam-figures|reading exam figures]]. Find (a) $\partial f/\partial x$, $\partial f/\partial y$ directly; (b) every node value; (c) every local partial; (d) every adjoint, and check (a).

> [!solution]- Solution
> (a) $\dfrac{\partial f}{\partial x}=-4y\sin(xy)\cos(xy)-y\tan(xy)$, and symmetrically $\dfrac{\partial f}{\partial y}=-4x\sin(xy)\cos(xy)-x\tan(xy)$.
>
> (b) $z_1=x$, $z_2=y$, $z_3=z_1z_2=xy$, $z_4=\cos z_3$, $z_5=z_4^2$, $z_6=2z_5$, $z_7=\ln z_4$, $z_8=z_6+z_7=f$.
>
> (c) $\dfrac{\partial z_8}{\partial z_6}=\dfrac{\partial z_8}{\partial z_7}=1$, $\dfrac{\partial z_6}{\partial z_5}=2$, $\dfrac{\partial z_5}{\partial z_4}=2z_4$, $\dfrac{\partial z_7}{\partial z_4}=\dfrac1{z_4}$, $\dfrac{\partial z_4}{\partial z_3}=-\sin z_3$, $\dfrac{\partial z_3}{\partial z_1}=z_2$, $\dfrac{\partial z_3}{\partial z_2}=z_1$.
>
> (d) $\bar z_8=\bar z_7=\bar z_6=1$, $\bar z_5=2$, $\bar z_4=\bar z_5(2z_4)+\bar z_7\dfrac1{z_4}=4z_4+\dfrac1{z_4}$ (two successors), $\bar z_3=-\sin(z_3)\Bigl(4z_4+\dfrac1{z_4}\Bigr)$,
> $$
> \bar z_1=z_2\bar z_3=-y\sin(xy)\Bigl(4\cos(xy)+\frac1{\cos(xy)}\Bigr),\qquad \bar z_2=z_1\bar z_3=-x\sin(xy)\Bigl(4\cos(xy)+\frac1{\cos(xy)}\Bigr),
> $$
> which expand to (a) since $\sin/\cos=\tan$. (Numerically at $(0.5,1)$: $\bar z_1\approx-2.2292$, matching a finite-difference check.)

More: [[6-midterm-1/practice-problems|practice problems]] (a graph with fan-out into a log), [[6-midterm-1/fall-2024|Fall 2024 Q4]] ($\sin x+e^{\sin x}+\sin^2x+364$), [[6-midterm-1/spring-2025|Spring 2025]] (a subtraction edge, local partial $-1$), [[6-midterm-1/spring-2026|Spring 2026]] (log-softmax).

## Vanishing gradients through stacked sigmoids

Homework 2 Q3 chains three sigmoids: $z_2=z_1-a$, $z_3=\sigma(z_2)$, $z_4=z_3-b$, $z_5=\sigma(z_4)$, $z_6=z_5-c$, $z_7=\sigma(z_6)=f$. Each sigmoid node has local partial $\sigma'=z(1-z)\le\tfrac14$, so

$$
\frac{\partial f}{\partial c}=-z_7(1-z_7),\quad
\frac{\partial f}{\partial b}=-z_7(1-z_7)\,z_5(1-z_5),\quad
\frac{\partial f}{\partial a}=-z_7(1-z_7)\,z_5(1-z_5)\,z_3(1-z_3).
$$

With $x=a=1$, $b=c=\tfrac12$: $z=(1,0,\tfrac12,0,\tfrac12,0,\tfrac12)$, so $\partial f/\partial c=-\tfrac14$, $\partial f/\partial b=-\tfrac1{16}$, $\partial f/\partial a=-\tfrac1{64}$. The further a parameter is from the output, the smaller its gradient — a **vanishing gradient**, and the reason deep sigmoid networks train slowly.

A **skip connection** $g=\sigma(z_6)+z_2$ adds an arrow $z_2\to g$ with local partial 1. Now $\partial g/\partial a=-(1+\tfrac1{64})=-\tfrac{65}{64}$: the gradient flows straight back without the $\tfrac14$ factors. This is the idea behind [[5-deep-networks/residual-connections|residual connections]] and one reason [[5-deep-networks/activation-functions|ReLU]] replaced sigmoids.

## Three ways to compute derivatives

Lecture 4's comparison of how a computer can get $\nabla f$:

| method | how | pros | cons |
|---|---|---|---|
| numeric (finite differences) | $\dfrac{f(x+h)-f(x)}{h}$ for small $h$ | simple; only evaluates $f$ | approximate; one extra evaluation of $f$ per parameter |
| symbolic | apply derivative rules to the formula, evaluate the result | exact; derive once, evaluate many times | needs a closed form; expressions grow exponentially |
| backpropagation (reverse-mode autodiff) | computational graph, forward pass, backward pass | exact; simple local rules; reuses computation (dynamic programming); all partials in one backward pass | must store every intermediate value: memory |

There is also **forward-mode** accumulation (e.g. with dual numbers), which pushes derivatives from the inputs toward the output; it needs one pass per input, so reverse mode wins when there are many parameters and one scalar loss — the ML situation.

> [!trap] Common mistakes
> - **Missing a fan-out term.** Fall 2025 $a_1$ and $a_3$, Fall 2024 $w_2$, Spring 2026 $w_1$ all have several successors.
> - **Dropping a minus** on a subtraction edge: $w_6=w_2-w_4\Rightarrow\partial w_6/\partial w_4=-1$.
> - Writing the full derivative where the local partial is asked. The Spring 2026 official key itself lists $\partial w_9/\partial w_1=(e^y+e^z)/S$ for $w_9=w_1-w_8$; the local partial is $1$, and $(e^y+e^z)/S$ is the adjoint $\bar w_1$.
> - Answering in node variables when the question says "in terms of $x$" (or the reverse).

> [!exam] On the exams
> 18–25 points on every Midterm 1, always the same four parts. True/false items: "backpropagation involves a forward pass followed by a backward pass" (True, Fall 2025), "backpropagation computes exact gradients" (True), "its main cost is memory for intermediate values" (True). See [[6-midterm-1/true-false|the T/F bank]] and the [[6-midterm-1/formula-sheet|formula sheet]].

## PyTorch docs

- [`torch.Tensor.backward`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html) — runs the backward pass from a (scalar) seed.
- [`torch.Tensor.retain_grad`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.retain_grad.html) — keep the adjoint of an intermediate (non-leaf) node, as in the code above.
- [`torch.autograd.grad`](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html) — returns gradients directly instead of storing them in `.grad`.
- [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) — how PyTorch saves forward values for the backward pass.
- [A gentle introduction to `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — tutorial with the DAG picture.

Related: [[2-calculus-and-optimization/computational-graphs]] · [[2-calculus-and-optimization/autograd]] · [[2-calculus-and-optimization/matrix-calculus]] · [[2-calculus-and-optimization/fitting-models-with-gradient-descent]] · [[5-deep-networks/residual-connections]]
