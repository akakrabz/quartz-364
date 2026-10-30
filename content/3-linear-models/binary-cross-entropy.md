---
title: "Binary cross-entropy"
description: "The loss that trains logistic regression: BCE with 0/1 and ±1 labels, the proof that it equals the negative log-likelihood, its gradient σ(z) − y, why MSE on a sigmoid is a bad idea, and nn.BCELoss vs nn.BCEWithLogitsLoss."
tags: [topic, linear-models, midterm-1]
aliases: ["BCE", "BCELoss", "BCEWithLogitsLoss", "log loss", "empirical risk minimization", "ERM"]
---

*Lecture 9 · [[3-linear-models/index|Linear models]] · builds on [[3-linear-models/logistic-regression|logistic regression]]*

> [!abstract] In one breath
> For a predicted probability $f = \Pr\{y=1\mid x\}$ and a label $y\in\{0,1\}$, the binary cross-entropy is $\ell_{\text{bce}}(f, y) = -\left[y\log f + (1-y)\log(1-f)\right]$: $-\log f$ when the label is 1, $-\log(1-f)$ when it is 0. Summed over the data, it is **exactly** the negative log-likelihood of logistic regression, so minimizing BCE is maximum likelihood. With respect to the score $z$ (where $f = \sigma(z)$) its gradient is simply $\sigma(z) - y$. In PyTorch: `nn.BCELoss` takes probabilities, `nn.BCEWithLogitsLoss` takes raw scores.

## What it is

Lecture 9 frames training as **empirical risk minimization** (ERM): pick a model $f_w$, pick a loss $\ell$ that measures how wrong one prediction is, and minimize the total loss over the data,
$$
\min_w\ \sum_{i=1}^{N}\ell\big(f_w(x_i),\ y_i\big).
$$
[[3-linear-models/linear-regression|Linear regression]] is ERM with the squared error. For binary classification the matching loss is the **binary cross-entropy**. With labels $y\in\{-1, +1\}$, as in the logistic-regression derivation:
$$
\ell_{\text{bce}}(f_w(x), y) = -\left[\tfrac12(1-y)\log\left(1-f_w(x)\right) + \tfrac12(1+y)\log f_w(x)\right].
$$
Only one term is ever active: $\frac12(1+y)$ is 1 when $y = +1$ and 0 when $y=-1$, and $\frac12(1-y)$ the reverse.

- $y = +1$: the loss is $-\log f_w(x)$, small when $f_w(x)\to 1$.
- $y = -1$: the loss is $-\log\left(1-f_w(x)\right)$, small when $f_w(x)\to 0$.

The **standard form**, which PyTorch uses and which extends to many classes, relabels the negative class as 0:
$$
\ell_{\text{bce}}(f_w(x), y) = -\left[y\log f_w(x) + (1-y)\log\left(1-f_w(x)\right)\right],\qquad y\in\{0,1\}.
$$

<!-- fig:lin-bce -->
<figure class="cs-fig"><svg role="img" aria-label="Two loss curves of binary cross-entropy" style="max-width:444px;width:100%;height:auto" stroke-linejoin="round" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 333.56 218.12" xmlns="http://www.w3.org/2000/svg" version="1.1"><defs></defs><g><g><path d="M 0 218.12  L 333.56 218.12  L 333.56 0  L 0 0  L 0 218.12  z " style="fill: none"/></g><g><g><path d="M 36.2 177.7  L 326.36 177.7  L 326.36 11.38  L 36.2 11.38  L 36.2 177.7  z " style="fill: none"/></g><g><g><g><path d="M 48.06 177.7  L 48.06 11.38  " clip-path="url(#fceafb2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><defs><path id="fceafb0" d="M 0 0  L 0 3.5  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#fceafb0" x="48.06" y="177.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="48.06" y="193.06" transform="rotate(-0 48.06 193.06)">0.0</text></g></g><g><g><path d="M 101.35 177.7  L 101.35 11.38  " clip-path="url(#fceafb2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fceafb0" x="101.35" y="177.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="101.35" y="193.06" transform="rotate(-0 101.35 193.06)">0.2</text></g></g><g><g><path d="M 154.63 177.7  L 154.63 11.38  " clip-path="url(#fceafb2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fceafb0" x="154.63" y="177.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="154.63" y="193.06" transform="rotate(-0 154.63 193.06)">0.4</text></g></g><g><g><path d="M 207.92 177.7  L 207.92 11.38  " clip-path="url(#fceafb2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fceafb0" x="207.92" y="177.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="207.92" y="193.06" transform="rotate(-0 207.92 193.06)">0.6</text></g></g><g><g><path d="M 261.21 177.7  L 261.21 11.38  " clip-path="url(#fceafb2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fceafb0" x="261.21" y="177.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="261.21" y="193.06" transform="rotate(-0 261.21 193.06)">0.8</text></g></g><g><g><path d="M 314.5 177.7  L 314.5 11.38  " clip-path="url(#fceafb2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fceafb0" x="314.5" y="177.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="314.5" y="193.06" transform="rotate(-0 314.5 193.06)">1.0</text></g></g><g><g style="fill: currentColor" transform="translate(109.72 208.28)"><text><tspan x="0" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">p</tspan><tspan x="6.98" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">r</tspan><tspan x="11.5" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">e</tspan><tspan x="18.27" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">d</tspan><tspan x="25.25" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">i</tspan><tspan x="28.31" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">c</tspan><tspan x="34.36" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">t</tspan><tspan x="38.67" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">e</tspan><tspan x="45.44" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">d</tspan><tspan x="52.42" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor"></tspan><tspan x="61.93" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">=</tspan><tspan x="73.29" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">P</tspan><tspan x="79.93" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">r</tspan><tspan x="86.24" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">{</tspan><tspan x="101.89" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">=</tspan><tspan x="113.25" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">1</tspan><tspan x="122.39" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">∣</tspan><tspan x="136.54" y="-0.52" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">}</tspan><tspan x="55.92" y="-0.52" style="font-style: oblique; font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">f</tspan><tspan x="93.23" y="-0.52" style="font-style: oblique; font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">y</tspan><tspan x="130.03" y="-0.52" style="font-style: oblique; font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">x</tspan></text></g></g></g><g><g><g><path d="M 36.2 177.7  L 326.36 177.7  " clip-path="url(#fceafb2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><defs><path id="fceafb1" d="M 0 0  L -3.5 0  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#fceafb1" x="36.2" y="177.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="29.2" y="181.88" transform="rotate(-0 29.2 181.88)">0</text></g></g><g><g><path d="M 36.2 144.43  L 326.36 144.43  " clip-path="url(#fceafb2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fceafb1" x="36.2" y="144.43" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="29.2" y="148.61" transform="rotate(-0 29.2 148.61)">1</text></g></g><g><g><path d="M 36.2 111.17  L 326.36 111.17  " clip-path="url(#fceafb2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fceafb1" x="36.2" y="111.17" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="29.2" y="115.35" transform="rotate(-0 29.2 115.35)">2</text></g></g><g><g><path d="M 36.2 77.91  L 326.36 77.91  " clip-path="url(#fceafb2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fceafb1" x="36.2" y="77.91" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="29.2" y="82.09" transform="rotate(-0 29.2 82.09)">3</text></g></g><g><g><path d="M 36.2 44.64  L 326.36 44.64  " clip-path="url(#fceafb2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fceafb1" x="36.2" y="44.64" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="29.2" y="48.82" transform="rotate(-0 29.2 48.82)">4</text></g></g><g><g><path d="M 36.2 11.38  L 326.36 11.38  " clip-path="url(#fceafb2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fceafb1" x="36.2" y="11.38" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="29.2" y="15.56" transform="rotate(-0 29.2 15.56)">5</text></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="15.56" y="94.54" transform="rotate(-90 15.56 94.54)">loss</text></g></g><g><path d="M 49.39 1.46  L 50.27 18.36  L 51.15 29.51  L 52.04 37.85  L 52.92 44.51  L 53.8 50.06  L 55.56 58.97  L 57.33 65.99  L 59.09 71.79  L 60.86 76.72  L 62.62 81.02  L 64.39 84.82  L 66.15 88.23  L 68.8 92.77  L 71.44 96.77  L 74.09 100.33  L 76.74 103.56  L 79.38 106.49  L 82.91 110.04  L 86.44 113.25  L 89.97 116.18  L 94.38 119.5  L 98.79 122.53  L 103.2 125.3  L 108.5 128.35  L 113.79 131.14  L 119.97 134.13  L 126.14 136.87  L 133.2 139.75  L 140.26 142.4  L 148.2 145.15  L 157.02 147.96  L 166.72 150.79  L 177.31 153.64  L 188.78 156.46  L 201.13 159.26  L 214.36 162.02  L 228.48 164.73  L 243.48 167.39  L 260.24 170.12  L 277.88 172.78  L 297.29 175.48  L 313.17 177.53  L 313.17 177.53  " clip-path="url(#fceafb2)" style="fill: none; stroke: var(--hi); stroke-width: 2; stroke-linecap: square"/></g><g><path d="M 49.39 177.53  L 70.56 174.76  L 89.97 172.01  L 107.61 169.28  L 124.38 166.47  L 139.37 163.74  L 153.49 160.94  L 166.72 158.09  L 178.19 155.4  L 188.78 152.71  L 198.48 150.04  L 208.19 147.14  L 217.01 144.26  L 224.95 141.43  L 232.01 138.7  L 239.06 135.72  L 245.24 132.88  L 251.42 129.78  L 256.71 126.86  L 262 123.67  L 266.41 120.75  L 270.82 117.55  L 274.35 114.74  L 277.88 111.68  L 281.41 108.31  L 284.94 104.56  L 287.59 101.44  L 290.23 98  L 292.88 94.16  L 295.53 89.82  L 297.29 86.57  L 299.06 82.97  L 300.82 78.94  L 302.58 74.35  L 304.35 69.02  L 306.11 62.67  L 307.88 54.81  L 308.76 50.06  L 309.64 44.51  L 310.52 37.85  L 311.41 29.51  L 312.29 18.36  L 313.17 1.46  L 313.17 1.46  " clip-path="url(#fceafb2)" style="fill: none; stroke: var(--accent); stroke-width: 2; stroke-linecap: square"/></g><g><path d="M 36.2 177.7  L 36.2 11.38  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><path d="M 36.2 177.7  L 326.36 177.7  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><g><path d="M 119.29 25.79  L 130.29 25.79  L 141.29 25.79  " style="fill: none; stroke: var(--hi); stroke-width: 2; stroke-linecap: square"/></g><g><g style="fill: currentColor" transform="translate(150.09 29.64)"><text><tspan x="0" y="-0.64" style="font-style: oblique; font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">y</tspan><tspan x="61.99" y="-0.64" style="font-style: oblique; font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">f</tspan><tspan x="8.65" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">=</tspan><tspan x="20.01" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">1</tspan><tspan x="27.01" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">:</tspan><tspan x="30.72" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor"></tspan><tspan x="34.21" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">−</tspan><tspan x="43.43" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">l</tspan><tspan x="46.49" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">o</tspan><tspan x="53.22" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">g</tspan></text></g></g><g><path d="M 119.29 42.29  L 130.29 42.29  L 141.29 42.29  " style="fill: none; stroke: var(--accent); stroke-width: 2; stroke-linecap: square"/></g><g><g style="fill: currentColor" transform="translate(150.09 46.14)"><text><tspan x="0" y="-0.64" style="font-style: oblique; font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">y</tspan><tspan x="84.99" y="-0.64" style="font-style: oblique; font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">f</tspan><tspan x="8.65" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">=</tspan><tspan x="20.01" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">0</tspan><tspan x="27.01" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">:</tspan><tspan x="30.72" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor"></tspan><tspan x="34.21" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">−</tspan><tspan x="43.43" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">l</tspan><tspan x="46.49" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">o</tspan><tspan x="53.22" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">g</tspan><tspan x="60.2" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">(</tspan><tspan x="64.49" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">1</tspan><tspan x="73.63" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">−</tspan><tspan x="88.86" y="-0.64" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">)</tspan></text></g></g></g></g></g><defs><clipPath id="fceafb2"><rect x="36.2" y="11.38" width="290.16" height="166.32"/></clipPath></defs></svg> <figcaption>Binary cross-entropy: only one of its two terms is active for each example. A confident wrong prediction costs a lot (−log <i>f</i> → ∞ as <i>f</i> → 0 when <i>y</i> = 1); a confident right one costs almost nothing.</figcaption></figure>

## What you should know

- Both forms of the loss, and which label convention each uses.
- That summed BCE equals the logistic-regression NLL $\sum_i\ln(1+e^{-y_iw^\top x_i})$ (derivation below), so the two objectives have the same minimizer.
- A confident wrong prediction is punished without bound: $-\log f\to\infty$ as $f\to 0$ when $y=1$.
- The gradient with respect to the score: $\dfrac{\partial\ell_{\text{bce}}}{\partial z} = \sigma(z) - y$ for $y\in\{0,1\}$.
- Why a sigmoid output trained with MSE learns slowly (Spring 2026 T/F).
- `nn.BCELoss(probabilities, float_targets)` vs `nn.BCEWithLogitsLoss(logits, float_targets)`; both average over the batch by default.

## BCE is the negative log-likelihood

> [!derivation]- Derivation (Lecture 9)
> With $f_w(x_i) = \frac1{1+e^{-w^\top x_i}}$ and $1-f_w(x_i) = \frac1{1+e^{w^\top x_i}}$:
> $$
> \begin{aligned}
> \sum_{i=1}^{N}\ell_{\text{bce}}(f_w(x_i), y_i)
> &= -\sum_{i=1}^{N}\tfrac12(1-y_i)\log\frac1{1+e^{w^\top x_i}} + \tfrac12(1+y_i)\log\frac1{1+e^{-w^\top x_i}}\\
> &= \sum_{i=1}^{N}\tfrac12(1-y_i)\log\left(1+e^{w^\top x_i}\right) + \tfrac12(1+y_i)\log\left(1+e^{-w^\top x_i}\right)\\
> &= \sum_{i=1}^{N}\log\left(1+e^{-y_iw^\top x_i}\right).
> \end{aligned}
> $$
> The last step checks both cases: $y_i = +1$ keeps only $\log(1+e^{-w^\top x_i})$, and $y_i = -1$ keeps only $\log(1+e^{w^\top x_i})$, both equal to $\log(1+e^{-y_iw^\top x_i})$. That is the NLL from [[3-linear-models/logistic-regression|logistic regression]].

## The gradient is "prediction minus label"

Let $z = w^\top x$ and $f = \sigma(z)$, with $y\in\{0,1\}$. Using $\sigma'(z) = \sigma(z)(1-\sigma(z))$:
$$
\frac{\partial\ell_{\text{bce}}}{\partial z} = -\left[y\,\frac{\sigma(1-\sigma)}{\sigma} - (1-y)\,\frac{\sigma(1-\sigma)}{1-\sigma}\right] = -\left[y(1-\sigma) - (1-y)\sigma\right] = \sigma(z) - y .
$$
By the chain rule $\nabla_w\ell = (\sigma(w^\top x) - y)\,x$, so a whole batch has gradient $X^\top(\sigma(Xw) - y)$. The error signal is just how far the predicted probability is from the label.

> [!intuition] Why not MSE on a sigmoid?
> With $\ell = (\sigma(z) - y)^2$, the gradient is $\frac{\partial\ell}{\partial z} = 2(\sigma(z)-y)\,\sigma(z)(1-\sigma(z))$. When the model is confidently **wrong** (say $y = 1$, $z = -10$), $\sigma(z)(1-\sigma(z))\approx 4.5\times10^{-5}$ and the gradient almost vanishes, so learning stalls exactly when it is needed most. BCE's gradient $\sigma(z) - y\approx -1$ stays large. This is the Spring 2026 true/false item: pairing MSE with a sigmoid output is not advisable because the gradients are small for large inputs (**true**).

## Numbers

| label $y$ | predicted $f$ | loss $-[y\log f + (1-y)\log(1-f)]$ |
|---|---|---|
| 1 | 0.9 | $-\log 0.9 = 0.105$ |
| 1 | 0.1 | $-\log 0.1 = 2.303$ |
| 0 | 0.2 | $-\log 0.8 = 0.223$ |
| 0 | 0.9 | $-\log 0.1 = 2.303$ |
| any | 0.5 | $\log 2 = 0.693$ |

The last row is a useful sanity check: an untrained model that outputs 0.5 everywhere starts at a mean BCE of about $0.693$.

## In PyTorch

```python
import torch.nn as nn
f = torch.tensor([0.9, 0.2])   # probabilities
y = torch.tensor([1., 0.])     # float labels
nn.BCELoss()(f, y)             # mean of 0.1054 and 0.2231
```

```
tensor(0.1643)
```

```python
z = torch.tensor([2.0, -1.0])         # raw scores (logits)
nn.BCEWithLogitsLoss()(z, y)          # sigmoid + BCE in one
nn.BCELoss()(torch.sigmoid(z), y)     # same value, less stable
```

Both lines give the mean of $-\log\sigma(2) = 0.127$ and $-\log(1-\sigma(-1)) = 0.313$, about $0.220$.

[`nn.BCELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) expects inputs already in $(0,1)$, so the model must end with a sigmoid. [`nn.BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) applies the sigmoid inside the loss using a numerically stable formula, so the model outputs raw scores; it is the usual choice in practice. The functional forms are [`F.binary_cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html) and [`F.binary_cross_entropy_with_logits`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html). All of them default to `reduction='mean'`; see [[4-pytorch-training/loss-functions|loss functions]].

> [!trap] Common mistakes
> - **Wrong label type.** Targets must be floats ($0.$ and $1.$), the same shape as the predictions. Integer or $\pm1$ targets are wrong here.
> - **Shape mismatch.** A model ending in `nn.Linear(d, 1)` outputs shape `(N, 1)`; the labels are usually `(N,)`. Squeeze the output (or unsqueeze the labels) so the shapes match.
> - **Sigmoid twice.** `BCEWithLogitsLoss` already applies the sigmoid. Feeding it probabilities is a silent bug.
> - **Mixing conventions on paper.** The $\frac12(1\pm y)$ form is for $y\in\{-1,+1\}$; the standard form is for $y\in\{0,1\}$.

> [!exam] On the exams
> Expect to state BCE, to know it equals the NLL, and to know that `nn.BCELoss` takes probabilities and float 0/1 targets. Spring 2026 had the MSE-with-sigmoid true/false item. See [[6-midterm-1/spring-2026|Spring 2026]] and the [[6-midterm-1/true-false|true/false bank]].

## PyTorch docs

- [`nn.BCELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html) — BCE on probabilities.
- [`nn.BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) — sigmoid and BCE combined, numerically stable; `pos_weight` for imbalanced classes.
- [`torch.nn.functional.binary_cross_entropy`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html) and [`binary_cross_entropy_with_logits`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html) — the functional versions.
- [`torch.sigmoid`](https://docs.pytorch.org/docs/stable/generated/torch.sigmoid.html) — probabilities from scores.
- [`torch.Tensor.squeeze`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.squeeze.html) — fix the `(N, 1)` vs `(N,)` shape mismatch.

Related: [[3-linear-models/logistic-regression]] · [[3-linear-models/multiclass-logistic-regression]] (cross-entropy for $K$ classes) · [[4-pytorch-training/loss-functions]] · [[5-deep-networks/activation-functions]]
