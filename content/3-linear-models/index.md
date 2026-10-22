---
title: "Linear models"
description: "Unit 3 (Lectures 7–9, plus multi-class classification from Lecture 11): linear regression and its closed form, multiple and basis-function regression, ridge regularization, the cost of each solver, and logistic regression for binary and multi-class classification."
tags: [unit, linear-models, midterm-1]
aliases: ["unit 3", "linear models", "linear and logistic regression"]
---

*Lectures 7, 8, 9 (and 11) · HW 3 · Midterm 1 §6–§7*

Linear models are the first complete machine-learning methods in the course. Each one is a model that is linear in its weights, an objective that measures fit, and a way to minimize it. For regression the objective is the mean squared error, which has an exact solution (the normal equations) as well as a gradient-descent one. For classification the objective is the negative log-likelihood (cross-entropy), which has no closed form, so you train it by gradient descent, first by hand in NumPy and then with PyTorch autograd. Everything here comes back in the deep-network units: an `nn.Linear` layer is a linear model, and cross-entropy is the loss you will use for every classifier.

## Topics, in order

1. [[3-linear-models/linear-regression|Linear regression]] — the line of best fit, the MSE objective, and three ways to solve it: partials set to zero, gradient descent, the normal equations. *(Lecture 7)*
2. [[3-linear-models/multiple-linear-regression|Multiple linear regression]] — $d$ features, the $N\times(d+1)$ design matrix, when the solution is unique, and weighted least squares. *(Lecture 7)*
3. [[3-linear-models/basis-function-regression|Basis-function (kernelized) regression]] — nonlinear features in a $\Phi$ matrix and the "linear in the weights" test. *(Lecture 8)*
4. [[3-linear-models/computational-cost|Computational cost]] — $\mathcal{O}(Nd^2+d^3)$ for the closed form vs $\mathcal{O}(Nd)$ per gradient step. *(Lecture 8)*
5. [[3-linear-models/ridge-regression|Ridge regression]] — $\lambda\lVert w\rVert^2$, the always-invertible $(X^\top X+\lambda I)$, outliers, overfitting, and reading $\lambda$ off a plot. *(Lecture 8)*
6. [[3-linear-models/logistic-regression|Logistic regression]] — why least squares is a poor classifier, the sigmoid, the sign test, maximum likelihood and margins. *(Lectures 8–9)*
7. [[3-linear-models/binary-cross-entropy|Binary cross-entropy]] — the logistic-regression loss in PyTorch's form, its gradient $\sigma(z)-y$, `BCELoss` vs `BCEWithLogitsLoss`. *(Lecture 9)*
8. [[3-linear-models/multiclass-logistic-regression|Multi-class logistic regression]] — one vs. all, $z = Wx+b$, softmax, cross-entropy on logits. *(Lectures 9, 11)*

## The whole unit on one screen

| | model | objective | solution |
|---|---|---|---|
| linear regression | $\hat y = Xw$ | $\frac1N\lVert Xw-y\rVert_2^2$ | $w^* = (X^\top X)^{-1}X^\top y$, or GD with $\frac2NX^\top(Xw-y)$ |
| basis functions | $\hat y = \Phi w$ | $\frac1N\lVert\Phi w-y\rVert_2^2$ | $w^* = (\Phi^\top\Phi)^{-1}\Phi^\top y$ |
| ridge | $\hat y = Xw$ | $\lVert Xw-y\rVert_2^2 + \lambda\lVert w\rVert_2^2$ | $w^* = (X^\top X+\lambda I)^{-1}X^\top y$ |
| logistic | $\Pr\{y=+1\mid x\} = \sigma(w^\top x)$ | $\sum_i\ln(1+e^{-y_iw^\top x_i})$ | gradient descent only |
| multi-class | $\Pr\{y=k\mid x\} = \mathrm{Softmax}(Wx+b)_k$ | $\sum_i -\log\mathrm{Softmax}(z_i)_{y_i}$ | gradient descent only |

## On Midterm 1

Linear regression was 35 points on Fall 2024 and 15 on Fall 2025; logistic and multi-class regression were 20 points on Fall 2025. The recurring questions are: derive the gradient and closed form (with whatever constants the problem uses), write $\Phi$ and decide whether a model is linear in its weights, choose $\lambda$ from a plot, classify points by the sign of $w^\top x$, pick the $W$ whose argmax classifies the data, and count one-vs-one classifiers. See [[6-midterm-1/index|Midterm 1]] and the [[6-midterm-1/formula-sheet|formula sheet]].

Before this unit: [[2-calculus-and-optimization/index|Calculus and optimization]]. After it: [[4-pytorch-training/index|PyTorch training]], which turns these hand-written loops into `nn.Module`, `torch.optim` and `DataLoader`.
