---
title: "Multiple linear regression"
description: "Linear regression with d input features: the N×(d+1) design matrix, the unchanged gradient and normal equations, when the solution is unique, the diabetes and California-housing examples from Lecture 7, and weighted least squares (Fall 2024 Q5)."
tags: [topic, linear-models, midterm-1]
aliases: ["multivariate linear regression", "normal equations", "design matrix", "plane of best fit", "weighted least squares"]
---

*Lecture 7 · [[3-linear-models/index|Linear models]] · builds on [[3-linear-models/linear-regression|linear regression]] and [[1-foundations/linear-algebra-in-pytorch|linear algebra in PyTorch]]*

> [!abstract] In one breath
> With $d$ covariates per example, the model is $\hat y = m_1x_1 + \dots + m_dx_d + b$: a plane (or hyperplane) of best fit with $d+1$ unknowns. Stack the examples as the rows of an $N\times(d+1)$ matrix $X$ whose last column is all ones, and the objective is still $\frac1N\lVert Xw-y\rVert_2^2$. Nothing else changes: the gradient is $\frac2N X^\top(Xw-y)$ and the exact solution is $w^* = (X^\top X)^{-1}X^\top y$, provided $X^\top X$ is invertible.

## What it is

Each data point now has $d$ covariates, $x_i = (x_{i,1}, \dots, x_{i,d})\in\mathbb{R}^d$, and one target $y_i\in\mathbb{R}$. With three numbers $(x, y, z)$ and two inputs, the model fits a **plane of best fit** to predict $z$ from $(x, y)$; in general it fits a hyperplane. In scalar form:
$$
\min_{m_1,\dots,m_d,\,b}\ \frac1N\sum_{i=1}^{N}\Big(\sum_{j=1}^{d} m_jx_{i,j} + b - y_i\Big)^2 .
$$
Deriving $d+1$ partial derivatives one at a time would be miserable. The matrix form does it in one line.

## What you should know

- The **design matrix** $X\in\mathbb{R}^{N\times(d+1)}$: row $i$ is $[x_{i,1},\ \dots,\ x_{i,d},\ 1]$; $w = [m_1,\dots,m_d,\ b]^\top$ has $d+1$ entries.
- The objective, gradient and solution are **the same formulas** as in one dimension: $J(w)=\frac1N\lVert Xw-y\rVert_2^2$, $\nabla_wJ = \frac2NX^\top(Xw-y)$, $w^* = (X^\top X)^{-1}X^\top y$.
- The shapes: $X^\top X$ is $(d+1)\times(d+1)$ and $X^\top y$ has length $d+1$, whatever $N$ is.
- A unique solution needs $X^\top X$ invertible: at least $d+1$ data points and linearly independent columns. Otherwise use [[3-linear-models/ridge-regression|ridge regression]].
- Adding a feature can never increase the *training* MSE of least squares (the old solution is still available with the new weight set to 0). Whether it helps on new data is a separate question, see [[4-pytorch-training/experimental-setup|experimental setup]].
- Weighted least squares reduces to ordinary least squares by rescaling the rows (Fall 2024 Q5, below).

## The matrix form

$$
X = \begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,d} & 1\\
x_{2,1} & x_{2,2} & \cdots & x_{2,d} & 1\\
\vdots & \vdots & & \vdots & \vdots\\
x_{N,1} & x_{N,2} & \cdots & x_{N,d} & 1
\end{bmatrix},\qquad
w = \begin{bmatrix}m_1\\ m_2\\ \vdots\\ m_d\\ b\end{bmatrix},\qquad
y = \begin{bmatrix}y_1\\ y_2\\ \vdots\\ y_N\end{bmatrix}.
$$

Row $i$ of $Xw$ is $\sum_j m_jx_{i,j} + b$, the prediction for example $i$, so $\frac1N\lVert Xw-y\rVert_2^2$ is the scalar objective above. The only differences from one dimension are that $X$ has $d+1$ columns instead of 2 and $w$ has $d+1$ entries:
$$
\nabla_wJ(w) = \frac2N X^\top(Xw-y),\qquad w^{(k+1)} = w^{(k)} - \alpha\nabla_wJ(w^{(k)}),\qquad w^* = \left(X^\top X\right)^{-1}X^\top y .
$$

| object | shape | note |
|---|---|---|
| $X$ | $N\times(d+1)$ | one row per example, last column all ones |
| $w$ | $d+1$ | $d$ slopes, then the bias |
| $Xw - y$ | $N$ | the residuals |
| $X^\top X$ | $(d+1)\times(d+1)$ | must be invertible for the closed form |
| $X^\top y$, $\nabla_w J$ | $d+1$ | same shape as $w$ |

```python
# Lecture 7: append the ones column, solve
def multiple_linear_regression_solution(x_data, y_data):
    X = np.concatenate((x_data, np.ones((x_data.shape[0], 1))), axis=1)
    w = np.linalg.inv(X.T @ X) @ X.T @ y_data
    return w        # [m_1, ..., m_d, b]
```

## When is the solution unique?

$w^* = (X^\top X)^{-1}X^\top y$ needs $X^\top X$ to be invertible, which happens exactly when the $d+1$ columns of $X$ are linearly independent. Two ways that fails:

- **Too few points.** If $N < d+1$, $X$ has rank at most $N$, so the $(d+1)\times(d+1)$ matrix $X^\top X$ has rank at most $N < d+1$ and is singular. Infinitely many $w$ fit the data exactly. You need at least as many data points as unknowns.
- **Redundant features.** Two columns that are copies (or multiples) of each other, or a feature that is constant (it duplicates the ones column), make the columns dependent even when $N$ is large. Only the combination of their weights can be recovered.

[[3-linear-models/ridge-regression|Ridge regression]] fixes both: $X^\top X + \lambda I$ is invertible for every $\lambda > 0$.

## Example: disease progression (Lecture 7)

The scikit-learn [diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) has 442 patients with 10 features (age, sex, BMI, blood pressure, six blood measurements) and a disease-progression score a year later. The lecture fits a line on each feature alone, then one model on all ten.

```python
# baseline: always predict the mean target
print(np.mean((targets - np.mean(targets))**2))
# all ten features at once
w = multiple_linear_regression_solution(covariates, targets)
```

```
MSE when using mean target as prediction = 5929.8849
MSE with feature sex is 5918.8889
MSE with feature bmi is 3890.4566
Solution MSE for all features= 2859.6963
```

<!-- fig:lin-diabetes-mse -->
<figure class="cs-fig"><svg role="img" aria-label="Bar chart of single-feature MSEs with two horizontal reference lines" style="max-width:621px;width:100%;height:auto" stroke-linejoin="round" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 466.16 213.08" xmlns="http://www.w3.org/2000/svg" version="1.1"><defs></defs><g><g><path d="M 0 213.08  L 466.16 213.08  L 466.16 0  L 0 0  L 0 213.08  z " style="fill: none"/></g><g><g><path d="M 57.2 187.88  L 458.96 187.88  L 458.96 21.56  L 57.2 21.56  L 57.2 187.88  z " style="fill: none"/></g><g><path d="M 75.46 187.88  L 100.06 187.88  L 100.06 62.69  L 75.46 62.69  z " clip-path="url(#fc009a2)" style="fill: var(--accent)"/></g><g><path d="M 113.31 187.88  L 137.91 187.88  L 137.91 58.35  L 113.31 58.35  z " clip-path="url(#fc009a2)" style="fill: var(--accent)"/></g><g><path d="M 151.15 187.88  L 175.76 187.88  L 175.76 102.74  L 151.15 102.74  z " clip-path="url(#fc009a2)" style="fill: var(--accent)"/></g><g><path d="M 189 187.88  L 213.6 187.88  L 213.6 83.4  L 189 83.4  z " clip-path="url(#fc009a2)" style="fill: var(--accent)"/></g><g><path d="M 226.85 187.88  L 251.45 187.88  L 251.45 63.94  L 226.85 63.94  z " clip-path="url(#fc009a2)" style="fill: var(--accent)"/></g><g><path d="M 264.7 187.88  L 289.3 187.88  L 289.3 62.04  L 264.7 62.04  z " clip-path="url(#fc009a2)" style="fill: var(--accent)"/></g><g><path d="M 302.55 187.88  L 327.15 187.88  L 327.15 78.33  L 302.55 78.33  z " clip-path="url(#fc009a2)" style="fill: var(--accent)"/></g><g><path d="M 340.4 187.88  L 365 187.88  L 365 82.15  L 340.4 82.15  z " clip-path="url(#fc009a2)" style="fill: var(--accent)"/></g><g><path d="M 378.24 187.88  L 402.85 187.88  L 402.85 99.66  L 378.24 99.66  z " clip-path="url(#fc009a2)" style="fill: var(--accent)"/></g><g><path d="M 416.09 187.88  L 440.69 187.88  L 440.69 77.09  L 416.09 77.09  z " clip-path="url(#fc009a2)" style="fill: var(--accent)"/></g><g><g><g><defs><path id="fc009a0" d="M 0 0  L 0 3.5  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#fc009a0" x="87.76" y="187.88" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="87.76" y="203.24" transform="rotate(-0 87.76 203.24)">age</text></g></g><g><g><g><use xlink:href="#fc009a0" x="125.61" y="187.88" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="125.61" y="203.24" transform="rotate(-0 125.61 203.24)">sex</text></g></g><g><g><g><use xlink:href="#fc009a0" x="163.46" y="187.88" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="163.46" y="203.24" transform="rotate(-0 163.46 203.24)">bmi</text></g></g><g><g><g><use xlink:href="#fc009a0" x="201.3" y="187.88" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="201.3" y="203.24" transform="rotate(-0 201.3 203.24)">bp</text></g></g><g><g><g><use xlink:href="#fc009a0" x="239.15" y="187.88" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="239.15" y="203.24" transform="rotate(-0 239.15 203.24)">s1</text></g></g><g><g><g><use xlink:href="#fc009a0" x="277" y="187.88" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="277" y="203.24" transform="rotate(-0 277 203.24)">s2</text></g></g><g><g><g><use xlink:href="#fc009a0" x="314.85" y="187.88" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="314.85" y="203.24" transform="rotate(-0 314.85 203.24)">s3</text></g></g><g><g><g><use xlink:href="#fc009a0" x="352.7" y="187.88" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="352.7" y="203.24" transform="rotate(-0 352.7 203.24)">s4</text></g></g><g><g><g><use xlink:href="#fc009a0" x="390.55" y="187.88" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="390.55" y="203.24" transform="rotate(-0 390.55 203.24)">s5</text></g></g><g><g><g><use xlink:href="#fc009a0" x="428.39" y="187.88" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="428.39" y="203.24" transform="rotate(-0 428.39 203.24)">s6</text></g></g></g><g><g><g><defs><path id="fc009a1" d="M 0 0  L -3.5 0  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#fc009a1" x="57.2" y="187.88" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="50.2" y="192.06" transform="rotate(-0 50.2 192.06)">0</text></g></g><g><g><g><use xlink:href="#fc009a1" x="57.2" y="144.11" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="50.2" y="148.29" transform="rotate(-0 50.2 148.29)">2000</text></g></g><g><g><g><use xlink:href="#fc009a1" x="57.2" y="100.34" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="50.2" y="104.52" transform="rotate(-0 50.2 104.52)">4000</text></g></g><g><g><g><use xlink:href="#fc009a1" x="57.2" y="56.57" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="50.2" y="60.75" transform="rotate(-0 50.2 60.75)">6000</text></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="15.56" y="104.72" transform="rotate(-90 15.56 104.72)">training MSE</text></g></g><g><path d="M 57.2 58.11  L 458.96 58.11  " clip-path="url(#fc009a2)" style="fill: none; stroke-dasharray: 5.18,2.24; stroke-dashoffset: 0; stroke: var(--muted); stroke-width: 1.4"/></g><g><path d="M 57.2 125.3  L 458.96 125.3  " clip-path="url(#fc009a2)" style="fill: none; stroke: var(--hi); stroke-width: 1.6; stroke-linecap: square"/></g><g><path d="M 57.2 187.88  L 57.2 21.56  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><path d="M 57.2 187.88  L 458.96 187.88  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="258.08" y="15.56" transform="rotate(-0 258.08 15.56)">one feature at a time vs. all ten together</text></g><g><g><path d="M 123.91 33.35  L 132.91 33.35  L 141.91 33.35  " style="fill: none; stroke-dasharray: 5.18,2.24; stroke-dashoffset: 0; stroke: var(--muted); stroke-width: 1.4"/></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="149.11" y="36.5" transform="rotate(-0 149.11 36.5)">predict the mean: 5930</text></g><g><path d="M 273.37 33.35  L 282.37 33.35  L 291.37 33.35  " style="fill: none; stroke: var(--hi); stroke-width: 1.6; stroke-linecap: square"/></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="298.57" y="36.5" transform="rotate(-0 298.57 36.5)">all 10 features: 2860</text></g></g></g></g><defs><clipPath id="fc009a2"><rect x="57.2" y="21.56" width="401.76" height="166.32"/></clipPath></defs></svg> <figcaption>Lecture 7's diabetes data (442 patients, 10 features): the training MSE of a line fit on each single feature (bars), against the constant mean prediction and multiple linear regression on all ten features. Numbers are the notebook's printed output. BMI is the best single feature; sex alone is almost useless.</figcaption></figure>

Using all ten features halves the error of the mean-only baseline and beats the best single feature by a quarter. The individual slopes are harder to read: with the scaled features, `s1` gets $-792$ and `s2` gets $+477$ even though neither is useful alone. Large weights of opposite sign on two features usually mean the features are correlated and the model is trading one off against the other, which is one motivation for [[3-linear-models/ridge-regression|regularization]].

The second exercise, the California housing data (20,640 houses, 8 features, price in units of \$100k), tells the same story: predicting the mean gives MSE $1.3316$, the best single feature (median income) gives $0.7011$, and all eight give $0.5243$. Latitude and longitude are nearly useless *as linear* features (MSE $\approx 1.30$ and $1.33$) even though location obviously matters for house prices; the relationship is not a straight line in either coordinate. Capturing that needs [[3-linear-models/basis-function-regression|nonlinear features]] or a [[5-deep-networks/multilayer-perceptrons|neural network]].

## Weighted least squares (Fall 2024 Q5)

Give each data point its own weight $\sigma_i > 0$:
$$
\min_{w_1,\,w_0}\ f = \frac12\sum_{i=1}^{N}\big[\sigma_i\,(y_i - w_1x_i - w_0)\big]^2 .
$$

> [!example] Partial derivatives, matrix form, closed form
> **Partials.**
> $$
> \frac{\partial f}{\partial w_1} = -\sum_i \sigma_i^2\,x_i\,(y_i - w_1x_i - w_0),\qquad
> \frac{\partial f}{\partial w_0} = -\sum_i \sigma_i^2\,(y_i - w_1x_i - w_0).
> $$
> **Matrix form.** With $X$ the usual $N\times 2$ matrix (rows $[x_i,\ 1]$), $w = [w_1,\ w_0]^\top$ and $\Sigma = \mathrm{diag}(\sigma_1,\dots,\sigma_N)$,
> $$
> f(w) = \frac12\lVert \Sigma(Xw - y)\rVert_2^2 .
> $$
> **Closed form.** Substitute $Z = \Sigma X$ and $v = \Sigma y$; the problem becomes ordinary least squares $\frac12\lVert Zw - v\rVert_2^2$, so
> $$
> \nabla_w f = Z^\top(Zw - v) = X^\top\Sigma^\top\Sigma Xw - X^\top\Sigma^\top\Sigma y = 0
> \quad\Longrightarrow\quad
> w^* = \left(X^\top\Sigma^\top\Sigma X\right)^{-1}X^\top\Sigma^\top\Sigma\,y .
> $$
> Since $\Sigma$ is diagonal, $\Sigma^\top\Sigma = \mathrm{diag}(\sigma_1^2,\dots,\sigma_N^2)$. **Check:** with every $\sigma_i = 1$, $\Sigma = I$ and this is the ordinary $w^* = (X^\top X)^{-1}X^\top y$.

The substitution trick works for any reweighting: scale each row of $X$ and each entry of $y$, then apply the formula you already know.

## In PyTorch

```python
# X_feat: (N, d) features, y: (N,) targets
X = torch.cat((X_feat, torch.ones(X_feat.shape[0], 1)), dim=1)  # (N, d+1)
w = torch.linalg.solve(X.T @ X, X.T @ y)                       # (d+1,)
w_ls = torch.linalg.lstsq(X, y.unsqueeze(1)).solution          # same, (d+1, 1)
y_hat = X @ w                                                 # predictions
```

The same model as a module is a single [`nn.Linear(d, 1)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html): its `weight` holds the $d$ slopes and its `bias` holds $b$, so you never append the ones column yourself. See [[4-pytorch-training/nn-module|nn.Module]].

> [!trap] Common mistakes
> - **Counting unknowns.** $d$ features means $d+1$ unknowns (the bias), so you need at least $d+1$ points.
> - **Reading slopes as importance.** A slope's size depends on the feature's units and on the other features in the model. Compare features by the MSE they achieve, as the lecture does.
> - **Concatenating along the wrong axis.** The ones column is appended along `dim=1` (a new column), not `dim=0` (a new row).
> - **Expecting lower training MSE to mean a better model.** More features always fit the training set at least as well. Judge on held-out data.

> [!exam] On the exams
> Fall 2024 Q5 (35 points) was weighted least squares: partials, matrix form with $\Sigma$, and the closed form, as above. Fall 2025 Q5 asked whether two feature models are linear in their weights, for the $\Phi$ matrix of one, and whether to change $\lambda$ for a plotted fit; that is [[3-linear-models/basis-function-regression|basis-function regression]]. See [[6-midterm-1/fall-2024|Fall 2024]] and [[6-midterm-1/fall-2025|Fall 2025]].

## PyTorch docs

- [`torch.cat`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html) — append the column of ones (`dim=1`).
- [`torch.linalg.solve`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve.html) — solve $X^\top Xw = X^\top y$.
- [`torch.linalg.lstsq`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html) — least squares from $X$ and $y$ directly.
- [`torch.linalg.matrix_rank`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html) — check whether the columns of $X$ are independent.
- [`torch.diag`](https://docs.pytorch.org/docs/stable/generated/torch.diag.html) — build $\Sigma$ for weighted least squares.
- [`nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) — the same model as a layer, bias included.

Related: [[3-linear-models/linear-regression]] · [[3-linear-models/basis-function-regression]] · [[3-linear-models/ridge-regression]] · [[3-linear-models/computational-cost]] · [[2-calculus-and-optimization/matrix-calculus]]
