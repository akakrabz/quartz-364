---
title: "True/false bank"
description: "Thirty-four true/false statements for Midterm 1 with folded one-line answers: every T/F item from Fall 2025 and Spring 2026 (marked ★) plus in-scope lecture points on tensors, gradient descent, backpropagation, autograd, linear and ridge regression, and logistic and multi-class classification."
tags: [exam, midterm-1]
aliases: ["T/F bank", "true false bank"]
---

*Midterm 1 · 12–15 pts on two of four past exams ([[6-midterm-1/fall-2025#problem-1--true-or-false-15-pts|Fall 2025]], [[6-midterm-1/spring-2026#problem-1--true-or-false-12-pts|Spring 2026]]) · ★ = a real past exam item; the rest restate lecture points in scope · back to [[6-midterm-1/index|Midterm 1]]*

> [!abstract] How to use this page
> Say True or False out loud, then open the answer. A "False" needs the one fact that breaks it; the answers give it. On the exam there is no partial credit, so a statement is True only if it is **always** true.

## Past exam items ★

**1.** ★ Backpropagation involves a forward pass followed by a backward pass through a computational graph. *(Fall 25)*
> [!solution]- Answer
> **True.**

**2.** ★ As $N$ grows with $d$ fixed, gradient descent becomes more computationally efficient than the closed-form linear-regression solution. *(Fall 25)*
> [!solution]- Answer
> **False.** Both costs grow linearly in $N$: $O(Nd^2+d^3)$ vs $O(Ndk)$. The closed form becomes expensive as $d$ grows. → [[3-linear-models/computational-cost|computational cost]]

**3.** ★ For $y=x/\Vert x\Vert_2$ with $x\in\mathbb R^d$, the partial derivative $\partial y/\partial x$ is in $\mathbb R^d$. *(Fall 25)*
> [!solution]- Answer
> **False.** $y$ is a vector, so $\partial y/\partial x$ is a $d\times d$ Jacobian. → [[2-calculus-and-optimization/matrix-calculus|matrix calculus]]

**4.** ★ Gradient descent is always guaranteed to converge to a locally optimal point for a minimization problem. *(Fall 25)*
> [!solution]- Answer
> **False.** Too large a step size oscillates or diverges (HW2 Q1 with $\alpha=\tfrac12$ and $\alpha=1$). → [[6-midterm-1/review-gradient-descent#step-size-decides-everything-hw2-q1|step sizes]]

**5.** ★ Predicting the next day's highest temperature from daily weather data can be done with logistic regression. *(Fall 25)*
> [!solution]- Answer
> **False.** That is a regression problem; logistic regression is for classification.

**6.** ★ Linear regression assumes $y$ can be represented as a linear combination of the input features. *(Spr 26)*
> [!solution]- Answer
> **True.**

**7.** ★ Logistic regression is mainly used to predict continuous values such as the exact price of a house. *(Spr 26)*
> [!solution]- Answer
> **False.** It is a classifier.

**8.** ★ A tensor's `requires_grad` must be `True` for PyTorch to track operations for automatic differentiation. *(Spr 26)*
> [!solution]- Answer
> **True.** → [[2-calculus-and-optimization/autograd|autograd]]

**9.** ★ ReLU is defined by $f(x)=\max(0,x)$. *(Spr 26)*
> [!solution]- Answer
> **True.** → [[5-deep-networks/activation-functions|activation functions]]

**10.** ★ `loss.backward()` computes the gradients and immediately updates the model's weights. *(Spr 26)*
> [!solution]- Answer
> **False.** It only fills `.grad`. The update is a separate step, by hand or with `optimizer.step()`.

**11.** ★ It is not advisable to pair MSE loss with a sigmoid output because gradients are small for large inputs. *(Spr 26)*
> [!solution]- Answer
> **True.** The sigmoid saturates, so $\sigma'$ is tiny far from 0, and the MSE gradient carries that factor.

**12.** ★ Swish, $f(x)=x\cdot\sigma(\beta x)$, is strictly increasing for every $\beta>0$. *(Spr 26)*
> [!solution]- Answer
> **False.** It dips below zero for negative $x$ and has a minimum there (near $x\approx-1.28/\beta$).

**13.** ★ A ReLU unit whose output is zero for every training input ends up with zero gradient permanently. *(Spr 26)*
> [!solution]- Answer
> **True.** This is the "dead ReLU" problem.

## Tensors

**14.** A tensor view allocates new memory for its data. *(L2)*
> [!solution]- Answer
> **False.** It reuses the original storage. → [[1-foundations/storage-and-views|storage and views]]

**15.** `torch.cat` adds a new dimension. *(L2)*
> [!solution]- Answer
> **False.** `torch.stack` adds one; `cat` joins along an existing dimension.

**16.** `a[:, [0, 2]]` returns a view of `a`. *(Fall 24 Q2)*
> [!solution]- Answer
> **False.** Indexing with a list copies.

**17.** `a[a > 0]` returns a tensor with the same shape as `a`. *(L2)*
> [!solution]- Answer
> **False.** Boolean-mask indexing returns a 1-D tensor.

## Gradients and backpropagation

**18.** The gradient points in the direction of steepest descent. *(L3–L4)*
> [!solution]- Answer
> **False.** Steepest **ascent**. Gradient descent steps along $-\nabla f$.

**19.** With a suitable step size, gradient descent finds the global minimum of a convex function. *(L4)*
> [!solution]- Answer
> **True.**

**20.** Backpropagation computes exact gradients. *(L4)*
> [!solution]- Answer
> **True.** Finite differences are the approximate method.

**21.** The main drawback of backpropagation is the memory needed to store intermediate values. *(L4)*
> [!solution]- Answer
> **True.**

**22.** The adjoint of a node with two successors adds one contribution from each successor. *(L5)*
> [!solution]- Answer
> **True.** $\bar z_i=\sum_{j\in\text{succ}(i)}\bar z_j\,\partial z_j/\partial z_i$.

**23.** `x.grad` must be cleared each iteration because PyTorch accumulates gradients. *(L6)*
> [!solution]- Answer
> **True.** → [[6-midterm-1/review-autograd|autograd review]]

## Linear regression

**24.** The closed-form least-squares solution always exists. *(L8)*
> [!solution]- Answer
> **False.** $X^\top X$ is singular when $d>N$ or when columns are dependent. Ridge regression fixes this.

**25.** Increasing $\lambda$ in ridge regression tends to shrink the weights. *(L8)*
> [!solution]- Answer
> **True.**

**26.** Linear regression can only fit straight lines. *(L8)*
> [!solution]- Answer
> **False.** It only needs to be linear in the **weights**, so polynomial, sinusoidal and other feature maps work. → [[3-linear-models/basis-function-regression|basis functions]]

**27.** Squaring the errors in MSE treats positive and negative errors equally. *(L6–L7)*
> [!solution]- Answer
> **True.**

**28.** Linear regression with $\pm1$ labels classifies reliably even with imbalanced, skewed data. *(L8)*
> [!solution]- Answer
> **False.** Far-away points drag the line and cause misclassifications.

## Logistic and multi-class

**29.** Logistic regression has a closed-form solution. *(L9)*
> [!solution]- Answer
> **False.** Train it with gradient descent.

**30.** Logistic regression's decision boundary is linear in $x$. *(L9)*
> [!solution]- Answer
> **True.** It is the set where $w^\top x=0$.

**31.** One-vs-one classification with $K$ classes trains $K$ classifiers. *(L9)*
> [!solution]- Answer
> **False.** $K(K-1)/2$. One-vs-all trains $K$.

**32.** Applying softmax can change which class has the highest score. *(L9)*
> [!solution]- Answer
> **False.** Softmax preserves the order.

**33.** `nn.CrossEntropyLoss` expects softmax probabilities as its input. *(L9)*
> [!solution]- Answer
> **False.** It expects logits. → [[4-pytorch-training/loss-functions|loss functions]]

**34.** Maximizing the likelihood $\prod_i\sigma(y_iw^\top x_i)$ and minimizing $\sum_i\ln(1+e^{-y_iw^\top x_i})$ give the same $w$. *(L9)*
> [!solution]- Answer
> **True.** The second is $-\ln$ of the first, and $\ln$ is monotone.
