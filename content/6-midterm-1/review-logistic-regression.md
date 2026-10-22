---
title: "Review: logistic regression and multi-class classification"
description: "Midterm 1 problem type 7 (20 pts on Fall 2025): regression vs classification, the logistic model and the sign test, maximum likelihood as minimum NLL, binary cross-entropy, the margins table for 'which w classifies / maximizes likelihood', one-vs-one vs one-vs-all, softmax and cross-entropy, and the HW3 answers."
tags: [exam, midterm-1, linear-models]
aliases: ["logistic regression review", "classification review"]
---

*Midterm 1 · problem type 7 of 7 · HW3 Q3–Q6 · Lectures 8–9 · 20 pts on Fall 2025 · topic pages: [[3-linear-models/logistic-regression|logistic regression]], [[3-linear-models/binary-cross-entropy|binary cross-entropy]], [[3-linear-models/multiclass-logistic-regression|multi-class logistic regression]] · back to [[6-midterm-1/index|Midterm 1]]*

> [!abstract] In one breath
> Logistic regression predicts $\Pr\{y=+1\mid x\}=\sigma(w^\top x)$ and classifies with a **sign test**: $+1$ iff $w^\top x\ge0$. "Which $w$ classifies the data?" means "are all margins $y_iw^\top x_i$ positive?", and "which maximizes the likelihood?" means "whose margins are largest", because $\sigma$ is increasing. Training minimizes the negative log-likelihood (= binary cross-entropy) by gradient descent — there is no closed form. Multi-class: one weight row per class, predict the argmax of $Wx$.

## Regression or classification?

| task | type |
|---|---|
| predict tomorrow's high temperature (Fall 25 T/F e) | regression, so logistic regression is the wrong tool |
| predict the exact price of a house (Spr 26 T/F b) | regression |
| will a patient develop Type-II diabetes next year? (HW3 3a) | classification (binary) |
| speaker's age from an audio clip (HW3 3b) | regression |
| vehicle speed from a video clip (HW3 3c) | regression |

## The model and the decision rule

$$
f_w(x)=\sigma(w^\top x)=\frac1{1+e^{-w^\top x}}=\Pr\{y=+1\mid x\},\qquad 1-f_w(x)=\frac1{1+e^{w^\top x}}=\Pr\{y=-1\mid x\}.
$$
Compactly, $\Pr\{y=y_i\mid x_i\}=\sigma(y_iw^\top x_i)$ for $y_i\in\{-1,+1\}$. A bias is handled by appending a 1 to $x$.

> [!key] Decision rule
> Predict $+1\iff f_w(x)\ge0.5\iff w^\top x\ge0$. Every "which class" or "which $w$ classifies the data" question is a **sign check**; you never need to evaluate the sigmoid. The decision boundary $w^\top x=0$ is a line (a hyperplane in higher dimensions).

## Training: maximum likelihood = minimum negative log-likelihood

$$
\max_w\prod_{i=1}^N\sigma(y_iw^\top x_i)\iff\min_wJ(w)=\sum_{i=1}^N\ln\!\left(1+e^{-y_iw^\top x_i}\right),\qquad
\nabla_wJ=-\sum_{i=1}^N\frac{y_ix_i}{1+e^{y_iw^\top x_i}}.
$$
Taking $\ln$ turns the product into a sum without moving the maximizer ($\ln$ is monotone), and the minus sign makes it a minimization. There is no closed form, so train with gradient descent.

**Binary cross-entropy** with labels $y\in\{0,1\}$: $\ell_{\text{bce}}(f,y)=-\big[y\ln f+(1-y)\ln(1-f)\big]$. With $\pm1$ labels it is $-\big[\tfrac12(1-y)\ln(1-f)+\tfrac12(1+y)\ln f\big]$, which sums to exactly the $J(w)$ above. `nn.BCELoss` takes probabilities (after the sigmoid) and float 0/1 targets.

## The margins table (Fall 2025 Q6a, 14 pts)

Data: $(0,-1),(1,-2),(0,-3)$ labelled $+1$; $(1,1),(0,2),(1,3)$ labelled $-1$. Margins $m_i=y_iw^\top x_i$; a $w$ classifies everything when all six are positive.

| $w$ | margins $m_1\dots m_6$ | all $>0$? | $\prod\sigma(m_i)$ |
|---|---|---|---|
| $w_1=(1,1)$ | $-1,-1,-3,-2,-2,-4$ | no | — |
| $w_2=(-1,-1)$ | $1,1,3,2,2,4$ | yes | $0.388$ |
| $w_3=(0,-2)$ | $2,4,6,2,4,6$ | yes | $0.744$ |
| $w_4=(0,2)$ | $-2,-4,-6,-2,-4,-6$ | no | — |
| $w_5=(1,-2)$ | $2,5,6,1,4,5$ | yes | $0.622$ |

(i) $w_2$, $w_3$, $w_5$ classify the data. (ii) $w_3$ maximizes the likelihood: its smallest margin is 2, while $w_5$ has only 1 on $(1,1)$. Full solution: [[6-midterm-1/fall-2025#problem-6--logistic-regression-20-pts|Fall 2025 Q6]].

> [!solution]- HW3 Q4 answers
> (a) $w=-3$: positive class when $-3x\ge0$, i.e. $x\le0$; negative when $x>0$.
>
> (b) $w=(1,-2,2,-1)$: $w^\top x_1=-1\to$ negative; $w^\top x_2=-6+12-4=2\to$ positive; $w^\top x_3=-3-2-4=-9\to$ negative.
>
> (c) Positive cluster near $(2,-4)$, negative near $(-3,3)$: $w_2=(1,-1)$ and $w_4=(2,-3)$ work. $w_1=(1,1)$ sends $(2,-4)$ to the negative side; $w_3=(-2,3)$ flips both classes.

## Multi-class

**One vs. one.** A binary classifier for every pair of classes: $\binom K2=\tfrac{K(K-1)}2$ classifiers. Each class is in $K-1$ of them; the class that wins most often is predicted.

**One vs. all (one vs. rest).** $K$ classifiers, one weight vector per class, stacked as the rows of $W$. Each treats its class as positive and all others as negative. Predict the class with the highest score. This is the scheme multi-class logistic regression uses.
$$
z=Wx+b,\qquad \operatorname{Softmax}(z)_k=\Pr\{y=k\mid x\}=\frac{e^{z_k}}{\sum_{j=1}^Ke^{z_j}},\qquad \hat y=\arg\max_kz_k.
$$
Softmax keeps the order of the scores, so the argmax of the probabilities equals the argmax of the logits.
$$
\ell_{\text{ce}}(f(x),y)=-\sum_{k=1}^K\mathbb 1\{y=k\}\ln f_k(x)=-\ln f_y(x)=-z_y+\ln\sum_je^{z_j}.
$$
Minimize $\sum_i\ell_{\text{ce}}$ over $W$ and $b$. `nn.CrossEntropyLoss` takes raw **logits** (it applies the log-softmax itself) and integer (long) class labels.

**Which $W$ classifies the marked points?** ([[6-midterm-1/fall-2025#problem-6--logistic-regression-20-pts|Fall 2025 Q6b]]) Multiply each point by $W$ and take the largest score. Shortcut: each row $w_k$ should point toward its own class's cluster. Answer there: $W_1$.

> [!solution]- HW3 Q5 — four-class one-vs-all and softmax
> $W=\begin{bmatrix}1&3\\2&-1\\-3&2\\-2&-1\end{bmatrix}$.
>
> - $x_1=(1,-1)$: $z=(-2,3,-5,-1)\to$ class 2.
> - $x_2=(-1,4)$: $z=(11,-6,11,-2)\to$ a **tie** between classes 1 and 3.
> - $x_3=(0,5)$: $z=(15,-5,10,-5)\to$ class 1.
>
> (b) $x=(1,1)$: $z=(4,1,-1,-3)$, softmax $\approx(0.946,\,0.047,\,0.006,\,0.001)$.

Spring variant (logic-table classifiers, outside the Fall scope): [[6-midterm-1/spring-only-topics#linear-classifiers-for-logic-tables|Spring-only topics]].

Practice: [[6-midterm-1/practice-problems#p8--logistic-predictions|P8]], [[6-midterm-1/practice-problems#p9--softmax-and-cross-entropy|P9]] · next: [[6-midterm-1/true-false|the T/F bank]]
