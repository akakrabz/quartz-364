---
title: "Multiple linear regression"
description: "Linear regression with d input features: the N×(d+1) design matrix, the unchanged gradient and normal equations, when the solution is unique, the diabetes and California-housing examples from Lecture 7, and weighted least squares (Fall 2024 Q5)."
tags: [topic, linear-models, midterm-1]
aliases: ["multiple linear regression", "multivariate linear regression", "normal equations", "design matrix", "plane of best fit", "weighted least squares"]
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
<figure class="cs-fig"><svg role="img" aria-label="Bar chart of single-feature MSEs with two horizontal reference lines" style="max-width:621px;width:100%;height:auto" stroke-linejoin="round" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 466.155859 213.079141" xmlns="http://www.w3.org/2000/svg" version="1.1"><defs></defs><g id="lin-diabetes-mse-figure_1"><g id="lin-diabetes-mse-patch_1"><path d="M 0 213.079141  L 466.155859 213.079141  L 466.155859 0  L 0 0  L 0 213.079141  z " style="fill: none"/></g><g id="lin-diabetes-mse-axes_1"><g id="lin-diabetes-mse-patch_2"><path d="M 57.195859 187.878281  L 458.955859 187.878281  L 458.955859 21.558281  L 57.195859 21.558281  L 57.195859 187.878281  z " style="fill: none"/></g><g id="lin-diabetes-mse-patch_3"><path d="M 75.457678 187.878281  L 100.059091 187.878281  L 100.059091 62.688626  L 75.457678 62.688626  z " clip-path="url(#lin-diabetes-mse-pb0e953b0a7)" style="fill: var(--accent)"/></g><g id="lin-diabetes-mse-patch_4"><path d="M 113.306005 187.878281  L 137.907418 187.878281  L 137.907418 58.34807  L 113.306005 58.34807  z " clip-path="url(#lin-diabetes-mse-pb0e953b0a7)" style="fill: var(--accent)"/></g><g id="lin-diabetes-mse-patch_5"><path d="M 151.154333 187.878281  L 175.755746 187.878281  L 175.755746 102.73871  L 151.154333 102.73871  z " clip-path="url(#lin-diabetes-mse-pb0e953b0a7)" style="fill: var(--accent)"/></g><g id="lin-diabetes-mse-patch_6"><path d="M 189.002661 187.878281  L 213.604074 187.878281  L 213.604074 83.400568  L 189.002661 83.400568  z " clip-path="url(#lin-diabetes-mse-pb0e953b0a7)" style="fill: var(--accent)"/></g><g id="lin-diabetes-mse-patch_7"><path d="M 226.850989 187.878281  L 251.452402 187.878281  L 251.452402 63.94109  L 226.850989 63.94109  z " clip-path="url(#lin-diabetes-mse-pb0e953b0a7)" style="fill: var(--accent)"/></g><g id="lin-diabetes-mse-patch_8"><path d="M 264.699317 187.878281  L 289.30073 187.878281  L 289.30073 62.038794  L 264.699317 62.038794  z " clip-path="url(#lin-diabetes-mse-pb0e953b0a7)" style="fill: var(--accent)"/></g><g id="lin-diabetes-mse-patch_9"><path d="M 302.547645 187.878281  L 327.149058 187.878281  L 327.149058 78.333329  L 302.547645 78.333329  z " clip-path="url(#lin-diabetes-mse-pb0e953b0a7)" style="fill: var(--accent)"/></g><g id="lin-diabetes-mse-patch_10"><path d="M 340.395972 187.878281  L 364.997386 187.878281  L 364.997386 82.152631  L 340.395972 82.152631  z " clip-path="url(#lin-diabetes-mse-pb0e953b0a7)" style="fill: var(--accent)"/></g><g id="lin-diabetes-mse-patch_11"><path d="M 378.2443 187.878281  L 402.845713 187.878281  L 402.845713 99.663057  L 378.2443 99.663057  z " clip-path="url(#lin-diabetes-mse-pb0e953b0a7)" style="fill: var(--accent)"/></g><g id="lin-diabetes-mse-patch_12"><path d="M 416.092628 187.878281  L 440.694041 187.878281  L 440.694041 77.092078  L 416.092628 77.092078  z " clip-path="url(#lin-diabetes-mse-pb0e953b0a7)" style="fill: var(--accent)"/></g><g id="lin-diabetes-mse-matplotlib.axis_1"><g id="lin-diabetes-mse-xtick_1"><g id="lin-diabetes-mse-line2d_1"><defs><path id="lin-diabetes-mse-m64cf51c136" d="M 0 0  L 0 3.5  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#lin-diabetes-mse-m64cf51c136" x="87.758384" y="187.878281" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="lin-diabetes-mse-text_1"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="87.758384" y="203.235703" transform="rotate(-0 87.758384 203.235703)">age</text></g></g><g id="lin-diabetes-mse-xtick_2"><g id="lin-diabetes-mse-line2d_2"><g><use xlink:href="#lin-diabetes-mse-m64cf51c136" x="125.606712" y="187.878281" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="lin-diabetes-mse-text_2"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="125.606712" y="203.235703" transform="rotate(-0 125.606712 203.235703)">sex</text></g></g><g id="lin-diabetes-mse-xtick_3"><g id="lin-diabetes-mse-line2d_3"><g><use xlink:href="#lin-diabetes-mse-m64cf51c136" x="163.45504" y="187.878281" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="lin-diabetes-mse-text_3"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="163.45504" y="203.236563" transform="rotate(-0 163.45504 203.236563)">bmi</text></g></g><g id="lin-diabetes-mse-xtick_4"><g id="lin-diabetes-mse-line2d_4"><g><use xlink:href="#lin-diabetes-mse-m64cf51c136" x="201.303368" y="187.878281" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="lin-diabetes-mse-text_4"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="201.303368" y="203.236563" transform="rotate(-0 201.303368 203.236563)">bp</text></g></g><g id="lin-diabetes-mse-xtick_5"><g id="lin-diabetes-mse-line2d_5"><g><use xlink:href="#lin-diabetes-mse-m64cf51c136" x="239.151695" y="187.878281" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="lin-diabetes-mse-text_5"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="239.151695" y="203.235703" transform="rotate(-0 239.151695 203.235703)">s1</text></g></g><g id="lin-diabetes-mse-xtick_6"><g id="lin-diabetes-mse-line2d_6"><g><use xlink:href="#lin-diabetes-mse-m64cf51c136" x="277.000023" y="187.878281" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="lin-diabetes-mse-text_6"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="277.000023" y="203.235703" transform="rotate(-0 277.000023 203.235703)">s2</text></g></g><g id="lin-diabetes-mse-xtick_7"><g id="lin-diabetes-mse-line2d_7"><g><use xlink:href="#lin-diabetes-mse-m64cf51c136" x="314.848351" y="187.878281" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="lin-diabetes-mse-text_7"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="314.848351" y="203.235703" transform="rotate(-0 314.848351 203.235703)">s3</text></g></g><g id="lin-diabetes-mse-xtick_8"><g id="lin-diabetes-mse-line2d_8"><g><use xlink:href="#lin-diabetes-mse-m64cf51c136" x="352.696679" y="187.878281" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="lin-diabetes-mse-text_8"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="352.696679" y="203.235703" transform="rotate(-0 352.696679 203.235703)">s4</text></g></g><g id="lin-diabetes-mse-xtick_9"><g id="lin-diabetes-mse-line2d_9"><g><use xlink:href="#lin-diabetes-mse-m64cf51c136" x="390.545007" y="187.878281" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="lin-diabetes-mse-text_9"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="390.545007" y="203.235703" transform="rotate(-0 390.545007 203.235703)">s5</text></g></g><g id="lin-diabetes-mse-xtick_10"><g id="lin-diabetes-mse-line2d_10"><g><use xlink:href="#lin-diabetes-mse-m64cf51c136" x="428.393335" y="187.878281" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="lin-diabetes-mse-text_10"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="428.393335" y="203.235703" transform="rotate(-0 428.393335 203.235703)">s6</text></g></g></g><g id="lin-diabetes-mse-matplotlib.axis_2"><g id="lin-diabetes-mse-ytick_1"><g id="lin-diabetes-mse-line2d_11"><defs><path id="lin-diabetes-mse-maad9d876d3" d="M 0 0  L -3.5 0  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#lin-diabetes-mse-maad9d876d3" x="57.195859" y="187.878281" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="lin-diabetes-mse-text_11"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="50.195859" y="192.056992" transform="rotate(-0 50.195859 192.056992)">0</text></g></g><g id="lin-diabetes-mse-ytick_2"><g id="lin-diabetes-mse-line2d_12"><g><use xlink:href="#lin-diabetes-mse-maad9d876d3" x="57.195859" y="144.10986" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="lin-diabetes-mse-text_12"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="50.195859" y="148.288571" transform="rotate(-0 50.195859 148.288571)">2000</text></g></g><g id="lin-diabetes-mse-ytick_3"><g id="lin-diabetes-mse-line2d_13"><g><use xlink:href="#lin-diabetes-mse-maad9d876d3" x="57.195859" y="100.341439" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="lin-diabetes-mse-text_13"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="50.195859" y="104.52015" transform="rotate(-0 50.195859 104.52015)">4000</text></g></g><g id="lin-diabetes-mse-ytick_4"><g id="lin-diabetes-mse-line2d_14"><g><use xlink:href="#lin-diabetes-mse-maad9d876d3" x="57.195859" y="56.573018" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="lin-diabetes-mse-text_14"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="50.195859" y="60.751729" transform="rotate(-0 50.195859 60.751729)">6000</text></g></g><g id="lin-diabetes-mse-text_15"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="15.558281" y="104.718281" transform="rotate(-90 15.558281 104.718281)">training MSE</text></g></g><g id="lin-diabetes-mse-line2d_15"><path d="M 57.195859 58.107432  L 458.955859 58.107432  " clip-path="url(#lin-diabetes-mse-pb0e953b0a7)" style="fill: none; stroke-dasharray: 5.18,2.24; stroke-dashoffset: 0; stroke: var(--muted); stroke-width: 1.4"/></g><g id="lin-diabetes-mse-line2d_16"><path d="M 57.195859 125.296085  L 458.955859 125.296085  " clip-path="url(#lin-diabetes-mse-pb0e953b0a7)" style="fill: none; stroke: var(--hi); stroke-width: 1.6; stroke-linecap: square"/></g><g id="lin-diabetes-mse-patch_13"><path d="M 57.195859 187.878281  L 57.195859 21.558281  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g id="lin-diabetes-mse-patch_14"><path d="M 57.195859 187.878281  L 458.955859 187.878281  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g id="lin-diabetes-mse-text_16"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="258.075859" y="15.558281" transform="rotate(-0 258.075859 15.558281)">one feature at a time vs. all ten together</text></g><g id="lin-diabetes-mse-legend_1"><g id="lin-diabetes-mse-line2d_17"><path d="M 123.914687 33.346875  L 132.914687 33.346875  L 141.914687 33.346875  " style="fill: none; stroke-dasharray: 5.18,2.24; stroke-dashoffset: 0; stroke: var(--muted); stroke-width: 1.4"/></g><g id="lin-diabetes-mse-text_17"><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="149.114687" y="36.496875" transform="rotate(-0 149.114687 36.496875)">predict the mean: 5930</text></g><g id="lin-diabetes-mse-line2d_18"><path d="M 273.369531 33.346875  L 282.369531 33.346875  L 291.369531 33.346875  " style="fill: none; stroke: var(--hi); stroke-width: 1.6; stroke-linecap: square"/></g><g id="lin-diabetes-mse-text_18"><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="298.569531" y="36.496875" transform="rotate(-0 298.569531 36.496875)">all 10 features: 2860</text></g></g></g></g><defs><clipPath id="lin-diabetes-mse-pb0e953b0a7"><rect x="57.195859" y="21.558281" width="401.76" height="166.32"/></clipPath></defs></svg> <figcaption>Lecture 7's diabetes data (442 patients, 10 features): the training MSE of a line fit on each single feature (bars), against the constant mean prediction and multiple linear regression on all ten features. Numbers are the notebook's printed output. BMI is the best single feature; sex alone is almost useless.</figcaption></figure>

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
