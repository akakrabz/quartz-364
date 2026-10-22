---
title: "Linear algebra in PyTorch"
description: "The vector and matrix operations the course runs on, by hand and in PyTorch: transpose, column vs row vectors, dot product, L1/L2/L∞ norms and distances, angle and cosine similarity, outer product, matrix-vector and matrix-matrix products with their shape rules, the Hadamard product, the inverse, and the Lecture 2 practice problems with answers."
tags: [topic, foundations, midterm-1]
aliases: ["linear algebra", "dot product", "norm", "vector norm", "cosine similarity", "outer product", "matrix multiplication", "matrix product", "Hadamard product", "matrix inverse", "transpose", "torch.matmul"]
---

*Lecture 2 · HW 1 · [[1-foundations/index|Foundations]] · next: [[2-calculus-and-optimization/matrix-calculus|matrix calculus]]*

> [!abstract] In one breath
> Data are vectors and matrices, and models are built from a handful of operations on them. The **dot product** $u^\top v=\sum_i u_iv_i$ measures alignment; the **norm** $\Vert v\Vert$ measures length; together they give the **angle** (cosine similarity). A **matrix product** $(m\times n)(n\times p)\to(m\times p)$ needs matching inner sizes, and each entry is a row-times-column dot product. In PyTorch: `torch.dot`, `torch.norm`, `torch.outer`, `@` (`torch.matmul`), `*` (elementwise, Hadamard), `.T`, `torch.inverse`.

## What it is

Notation (Lecture 2): scalars $a\in\mathbb{R}$; vectors $v\in\mathbb{R}^n$ are **column** vectors; matrices $A\in\mathbb{R}^{m\times n}$ list rows first (row-major), so $A_{2,4}$ is row 2, column 4 (math is 1-indexed; PyTorch is 0-indexed). Lower case for scalars and vectors, upper case for matrices, bold for tensors, but **always state your notation**.

| Operation | Math | Shapes | PyTorch |
|---|---|---|---|
| transpose | $A^\top$ | $(m, n)\to(n, m)$ | `A.T`, `A.transpose(0, 1)`, `torch.transpose(A, 0, 1)` |
| dot product | $u\cdot v = u^\top v$ | $(n), (n)\to$ scalar | `torch.dot(u, v)`, `u @ v` |
| norm | $\Vert v\Vert_p$ | $(n)\to$ scalar | `torch.norm(v, p=2)`, `torch.linalg.vector_norm` |
| outer product | $uv^\top$ | $(n), (m)\to(n, m)$ | `torch.outer(u, v)` |
| matrix-vector | $Av$ | $(m, n), (n)\to(m)$ | `A @ v`, `torch.mv`, `torch.matmul` |
| matrix-matrix | $AB$ | $(m, n), (n, p)\to(m, p)$ | `A @ B`, `torch.mm`, `torch.matmul` |
| Hadamard | $A\odot B$ | same (or broadcastable) shapes | `A * B` |
| inverse | $B^{-1}$ | $(n, n)\to(n, n)$ | `torch.inverse(B)`, `torch.linalg.inv(B)` |
| identity | $I_n$ | $(n, n)$ | `torch.eye(n)` |

## What you should know

- Every entry of a product: $c_{ij} = \sum_k a_{ik}b_{kj}$ = (row $i$ of $A$) · (column $j$ of $B$). Inner sizes must match; the result takes the outer sizes.
- $AB\ne BA$ in general (often one of them is not even defined); $(AB)^\top = B^\top A^\top$.
- $u\cdot v>0$: similar directions; $=0$: perpendicular; $<0$: opposed. $\cos\theta = \dfrac{u\cdot v}{\Vert u\Vert_2\Vert v\Vert_2}\in[-1, 1]$.
- Norms: $\Vert v\Vert_2=\sqrt{\sum v_i^2}$, $\Vert v\Vert_1=\sum\lvert v_i\rvert$, $\Vert v\Vert_\infty=\max_i\lvert v_i\rvert$; distance $= \Vert u-v\Vert$. Note $\Vert v\Vert_2^2 = v^\top v$.
- `*` is elementwise; `@` is matrix multiplication. For 1-D tensors `u @ v` is a dot product and `.t()` does nothing.
- `torch.dot` takes two 1-D tensors only; `torch.norm` and `torch.inverse` need float tensors; a singular matrix has no inverse (runtime error).
- Predict the **shape** of an expression before computing it.

## Transpose, column and row vectors

$$
A=\begin{bmatrix}1&3&2&4\\2&4&3&5\\0&2&0&7\end{bmatrix},\qquad
A^{\top}=\begin{bmatrix}1&2&0\\3&4&2\\2&3&0\\4&5&7\end{bmatrix}
$$

The 5 at row 2, column 4 of $A$ is at row 4, column 2 of $A^\top$. In PyTorch `torch.transpose(A, 0, 1)`, `A.transpose(0, 1)` and `A.T` all give this (and all are [[1-foundations/storage-and-views|views]]).

A column vector $v\in\mathbb{R}^{4\times1}$ transposes to a row vector $v^\top\in\mathbb{R}^{1\times4}$; we often write column vectors flat as $v = [2\ \ 3\ \ {-1}\ \ 5]^\top$. A **1-D tensor has no orientation**: `v.t()` returns it unchanged. To get an explicit column or row use `v.unsqueeze(1)` → $(4, 1)$ or `v.unsqueeze(0)` → $(1, 4)$.

## Dot product, norm and angle

### Dot product

$$
u\cdot v = \langle u, v\rangle = v^\top u = \sum_{i=1}^{n}u_iv_i
$$

It is commutative and always a scalar.

```python
u = torch.tensor([1, 2, 3, 4])
v = torch.tensor([2, 0, -1, 1])
torch.dot(u, v)   # 1*2 + 2*0 + 3*(-1) + 4*1
```

```
tensor(3)
```

### Norm

$$
\Vert v\Vert_2=\sqrt{\sum_{i=1}^{n}v_i^2},\qquad
\Vert v\Vert_1=\sum_{i=1}^{n}\lvert v_i\rvert,\qquad
\Vert v\Vert_\infty=\max_{i}\lvert v_i\rvert
$$

```python
v = torch.tensor([2, 0, -1, 1]).float()   # norm needs floats
torch.norm(v, p=2)          # sqrt(4 + 0 + 1 + 1)
torch.norm(v, p=1)          # 2 + 0 + 1 + 1
torch.norm(v, p=torch.inf)  # largest magnitude
```

```
tensor(2.4495)
tensor(4.)
tensor(2.)
```

<!-- fig:found-norm-balls -->
<figure class="cs-fig"><svg role="img" aria-label="unit circle, diamond and square" style="max-width:360px;width:100%;height:auto" stroke-linejoin="round" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 245.938836 259.6704" xmlns="http://www.w3.org/2000/svg" version="1.1"><defs></defs><g id="found-norm-balls-figure_1"><g id="found-norm-balls-patch_1"><path d="M 0 259.6704  L 245.938836 259.6704  L 245.938836 -0  L 0 -0  L 0 259.6704  z " style="fill: none"/></g><g id="found-norm-balls-axes_1"><g id="found-norm-balls-patch_2"><path d="M 55.911094 219.4704  L 238.326578 219.4704  L 238.326578 30.9744  L 55.911094 30.9744  L 55.911094 219.4704  z " style="fill: none"/></g><g id="found-norm-balls-matplotlib.axis_1"><g id="found-norm-balls-xtick_1"><g id="found-norm-balls-line2d_1"><defs><path id="found-norm-balls-m9cea1034a6" d="M 0 0  L 0 3.5  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#found-norm-balls-m9cea1034a6" x="86.313674" y="219.4704" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="found-norm-balls-text_1"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="86.313674" y="234.827822" transform="rotate(-0 86.313674 234.827822)">−1</text></g></g><g id="found-norm-balls-xtick_2"><g id="found-norm-balls-line2d_2"><g><use xlink:href="#found-norm-balls-m9cea1034a6" x="147.118836" y="219.4704" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="found-norm-balls-text_2"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="147.118836" y="234.827822" transform="rotate(-0 147.118836 234.827822)">0</text></g></g><g id="found-norm-balls-xtick_3"><g id="found-norm-balls-line2d_3"><g><use xlink:href="#found-norm-balls-m9cea1034a6" x="207.923997" y="219.4704" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="found-norm-balls-text_3"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="207.923997" y="234.827822" transform="rotate(-0 207.923997 234.827822)">1</text></g></g><g id="found-norm-balls-text_4"><g style="fill: currentColor" transform="translate(141.233836 249.827822)"><text><tspan x="0" y="-0.984375" style="font-style: oblique; font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">v</tspan><tspan x="6.509766" y="0.66562" style="font-size: 7.7px; font-family: 'DejaVu Sans'; fill: currentColor">1</tspan></text></g></g></g><g id="found-norm-balls-matplotlib.axis_2"><g id="found-norm-balls-ytick_1"><g id="found-norm-balls-line2d_4"><defs><path id="found-norm-balls-m460f145c16" d="M 0 0  L -3.5 0  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#found-norm-balls-m460f145c16" x="55.911094" y="219.4704" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="found-norm-balls-text_5"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="48.911094" y="223.649111" transform="rotate(-0 48.911094 223.649111)">−1.5</text></g></g><g id="found-norm-balls-ytick_2"><g id="found-norm-balls-line2d_5"><g><use xlink:href="#found-norm-balls-m460f145c16" x="55.911094" y="189.067819" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="found-norm-balls-text_6"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="48.911094" y="193.24653" transform="rotate(-0 48.911094 193.24653)">−1.0</text></g></g><g id="found-norm-balls-ytick_3"><g id="found-norm-balls-line2d_6"><g><use xlink:href="#found-norm-balls-m460f145c16" x="55.911094" y="158.665239" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="found-norm-balls-text_7"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="48.911094" y="162.84395" transform="rotate(-0 48.911094 162.84395)">−0.5</text></g></g><g id="found-norm-balls-ytick_4"><g id="found-norm-balls-line2d_7"><g><use xlink:href="#found-norm-balls-m460f145c16" x="55.911094" y="128.262658" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="found-norm-balls-text_8"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="48.911094" y="132.441369" transform="rotate(-0 48.911094 132.441369)">0.0</text></g></g><g id="found-norm-balls-ytick_5"><g id="found-norm-balls-line2d_8"><g><use xlink:href="#found-norm-balls-m460f145c16" x="55.911094" y="97.860077" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="found-norm-balls-text_9"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="48.911094" y="102.038788" transform="rotate(-0 48.911094 102.038788)">0.5</text></g></g><g id="found-norm-balls-ytick_6"><g id="found-norm-balls-line2d_9"><g><use xlink:href="#found-norm-balls-m460f145c16" x="55.911094" y="67.457497" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="found-norm-balls-text_10"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="48.911094" y="71.636208" transform="rotate(-0 48.911094 71.636208)">1.0</text></g></g><g id="found-norm-balls-ytick_7"><g id="found-norm-balls-line2d_10"><g><use xlink:href="#found-norm-balls-m460f145c16" x="55.911094" y="37.054916" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="found-norm-balls-text_11"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="48.911094" y="41.233627" transform="rotate(-0 48.911094 41.233627)">1.5</text></g></g><g id="found-norm-balls-text_12"><g style="fill: currentColor" transform="translate(15.557422 131.1074) rotate(-90)"><text><tspan x="0" y="-0.984375" style="font-style: oblique; font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">v</tspan><tspan x="6.509766" y="0.66562" style="font-size: 7.7px; font-family: 'DejaVu Sans'; fill: currentColor">2</tspan></text></g></g></g><g id="found-norm-balls-line2d_11"><path d="M 207.923997 128.262658  L 207.80341 124.435114  L 207.442127 120.622752  L 206.841581 116.840692  L 206.004154 113.103935  L 204.933168 109.427303  L 203.63287 105.825379  L 202.108419 102.312448  L 200.365859 98.902445  L 198.412104 95.608895  L 196.254902 92.44486  L 193.902809 89.422892  L 191.365156 86.554975  L 188.652006 83.852485  L 185.774121 81.326142  L 182.742916 78.985964  L 179.570413 76.841235  L 176.269197 74.900461  L 172.85236 73.17134  L 169.333456 71.66073  L 165.72644 70.374622  L 162.04562 69.318119  L 158.305596 68.495409  L 154.521201 67.909757  L 150.707446 67.563486  L 146.879457 67.457968  L 143.052417 67.593622  L 139.241506 67.969911  L 135.46184 68.585342  L 131.728409 69.437473  L 128.056022 70.522925  L 124.459244 71.837393  L 120.952343 73.375663  L 117.549226 75.131633  L 114.263393 77.098339  L 111.107876 79.267981  L 108.09519 81.631952  L 105.237286 84.180877  L 102.545498 86.904645  L 100.030504 89.792453  L 97.702278 92.832848  L 95.570055 96.013769  L 93.642292 99.3226  L 91.926636 102.746218  L 90.429891 106.271042  L 89.157994 109.883093  L 88.115989 113.568043  L 87.30801 117.311278  L 86.737261 121.097949  L 86.406006 124.913038  L 86.315559 128.741413  L 86.466279 132.567888  L 86.857568 136.377288  L 87.487873 140.154503  L 88.354696 143.88455  L 89.454597 147.552635  L 90.783214 151.14421  L 92.335278 154.645029  L 94.104633 158.041206  L 96.084259 161.319271  L 98.266307 164.466222  L 100.64212 167.469578  L 103.202276 170.317425  L 105.93662 172.998469  L 108.834307 175.502075  L 111.883844 177.818314  L 115.073135 179.937997  L 118.38953 181.852719  L 121.819875 183.554884  L 125.350564 185.037741  L 128.967594 186.295408  L 132.656619 187.322898  L 136.403005 188.116134  L 140.191894 188.671971  L 144.008257 188.988204  L 147.836958 189.063579  L 151.662811 188.897796  L 155.470641 188.491513  L 159.245345 187.846342  L 162.97195 186.964842  L 166.635677 185.850509  L 170.221994 184.507762  L 173.716675 182.941928  L 177.10586 181.159217  L 180.376107 179.166701  L 183.514443 176.972281  L 186.508422 174.584663  L 189.346169 172.013315  L 192.016427 169.268437  L 194.508606 166.360916  L 196.812821 163.302285  L 198.919933 160.104674  L 200.821583 156.780767  L 202.51023 153.343747  L 203.979176 149.807247  L 205.222594 146.185294  L 206.235553 142.492253  L 207.014034 138.742773  L 207.554951 134.951726  L 207.856157 131.134147  L 207.923997 128.262658  L 207.923997 128.262658  " clip-path="url(#found-norm-balls-pb5506a8008)" style="fill: none; stroke: var(--accent); stroke-width: 2; stroke-linecap: square"/></g><g id="found-norm-balls-line2d_12"><path d="M 207.923997 128.262658  L 147.118836 67.457497  L 86.313674 128.262658  L 147.118836 189.067819  L 207.923997 128.262658  " clip-path="url(#found-norm-balls-pb5506a8008)" style="fill: none; stroke: var(--hi); stroke-width: 2; stroke-linecap: square"/></g><g id="found-norm-balls-line2d_13"><path d="M 207.923997 67.457497  L 86.313674 67.457497  L 86.313674 189.067819  L 207.923997 189.067819  L 207.923997 67.457497  " clip-path="url(#found-norm-balls-pb5506a8008)" style="fill: none; stroke: var(--accent2); stroke-width: 2; stroke-linecap: square"/></g><g id="found-norm-balls-line2d_14"><path d="M 55.911094 128.262658  L 238.326578 128.262658  " clip-path="url(#found-norm-balls-pb5506a8008)" style="fill: none; stroke: var(--muted); stroke-width: 0.8; stroke-linecap: square"/></g><g id="found-norm-balls-line2d_15"><path d="M 147.118836 219.4704  L 147.118836 30.9744  " clip-path="url(#found-norm-balls-pb5506a8008)" style="fill: none; stroke: var(--muted); stroke-width: 0.8; stroke-linecap: square"/></g><g id="found-norm-balls-patch_3"><path d="M 55.911094 219.4704  L 55.911094 30.9744  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g id="found-norm-balls-patch_4"><path d="M 55.911094 219.4704  L 238.326578 219.4704  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g id="found-norm-balls-legend_1"><g id="found-norm-balls-line2d_16"><path d="M 59.098836 14.58  L 64.498836 14.58  L 69.898836 14.58  " style="fill: none; stroke: var(--accent); stroke-width: 2; stroke-linecap: square"/></g><g id="found-norm-balls-text_13"><g style="fill: currentColor" transform="translate(77.098836 17.73)"><text><tspan x="0" y="-0.125" style="font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">‖</tspan><tspan x="9.826172" y="-0.125" style="font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">‖</tspan><tspan x="20.420068" y="-0.125" style="font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">=</tspan><tspan x="29.714502" y="-0.125" style="font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">1</tspan><tspan x="4.5" y="-0.125" style="font-style: oblique; font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">v</tspan><tspan x="14.412305" y="1.224996" style="font-size: 6.3px; font-family: 'DejaVu Sans'; fill: currentColor">2</tspan></text></g></g><g id="found-norm-balls-line2d_17"><path d="M 119.758836 14.58  L 125.158836 14.58  L 130.558836 14.58  " style="fill: none; stroke: var(--hi); stroke-width: 2; stroke-linecap: square"/></g><g id="found-norm-balls-text_14"><g style="fill: currentColor" transform="translate(137.758836 17.73)"><text><tspan x="0" y="-0.125" style="font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">‖</tspan><tspan x="9.826172" y="-0.125" style="font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">‖</tspan><tspan x="20.420068" y="-0.125" style="font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">=</tspan><tspan x="29.714502" y="-0.125" style="font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">1</tspan><tspan x="4.5" y="-0.125" style="font-style: oblique; font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">v</tspan><tspan x="14.412305" y="1.224996" style="font-size: 6.3px; font-family: 'DejaVu Sans'; fill: currentColor">1</tspan></text></g></g><g id="found-norm-balls-line2d_18"><path d="M 180.418836 14.58  L 185.818836 14.58  L 191.218836 14.58  " style="fill: none; stroke: var(--accent2); stroke-width: 2; stroke-linecap: square"/></g><g id="found-norm-balls-text_15"><g style="fill: currentColor" transform="translate(198.418836 17.73)"><text><tspan x="0" y="-0.125" style="font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">‖</tspan><tspan x="9.826172" y="-0.125" style="font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">‖</tspan><tspan x="21.659766" y="-0.125" style="font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">=</tspan><tspan x="30.954199" y="-0.125" style="font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">1</tspan><tspan x="4.5" y="-0.125" style="font-style: oblique; font-size: 9px; font-family: 'DejaVu Sans'; fill: currentColor">v</tspan><tspan x="14.412305" y="1.224996" style="font-size: 6.3px; font-family: 'DejaVu Sans'; fill: currentColor">∞</tspan></text></g></g></g></g></g><defs><clipPath id="found-norm-balls-pb5506a8008"><rect x="55.911094" y="30.9744" width="182.415484" height="188.496"/></clipPath></defs></svg> <figcaption><b>Unit balls in 2-D.</b> All vectors of norm 1: a circle for <i>L</i><sub>2</sub>, a diamond for <i>L</i><sub>1</sub>, a square for <i>L</i><sub>∞</sub>.</figcaption></figure>

> [!warning] Typo in the lecture notebook
> Lecture 2 writes $\Vert v\Vert_\infty = \max_i v_i$. The $L_\infty$ norm is $\max_i\lvert v_i\rvert$: for $(-5, 1)$ it is 5, not 1. (The lecture's example $v = (2, 0, -1, 1)$ gives 2 either way.)

A norm defines a **distance**: $\Vert u - v\Vert_2$ is the Euclidean distance. `torch.norm(u - v)` computes one distance; [`torch.cdist`](https://docs.pytorch.org/docs/stable/generated/torch.cdist.html) computes all pairwise distances between two sets of points. Current PyTorch recommends [`torch.linalg.vector_norm`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html) over the older `torch.norm`; the arguments are the same idea (`ord=2`, `ord=1`, `ord=torch.inf`).

### Angle and cosine similarity

$$
\theta = \cos^{-1}\left(\frac{u\cdot v}{\Vert u\Vert_2\,\Vert v\Vert_2}\right),\qquad
\cos\theta = \frac{u\cdot v}{\Vert u\Vert_2\,\Vert v\Vert_2}\in[-1, 1]
$$

![[l02-vector-angle.png|The angle θ between vectors u and v]]

Cosine similarity near 1: well aligned; near $-1$: anti-parallel (aligned, but opposite); 0: perpendicular, unrelated. For the dot-product example above, $\cos\theta = 3/(\sqrt{30}\sqrt{6}) = 3/\sqrt{180}\approx 0.224$, so $\theta\approx 77°$. In PyTorch: [`torch.nn.functional.cosine_similarity`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html).

## Outer product

For $u\in\mathbb{R}^n$ and $v\in\mathbb{R}^m$ (lengths may differ), $C = uv^\top\in\mathbb{R}^{n\times m}$ with $C_{ij}=u_iv_j$:

```python
u = torch.tensor([1, 2, 3])
v = torch.tensor([-1, 3, 0, -2])
torch.outer(u, v)
```

```
tensor([[-1,  3,  0, -2],
        [-2,  6,  0, -4],
        [-3,  9,  0, -6]])
```

Every row is a multiple of $v$: an outer product has rank 1. The same result by [[1-foundations/broadcasting|broadcasting]]: `u.unsqueeze(1) * v`.

## Matrix multiplication

### Matrix-vector product

$A\in\mathbb{R}^{m\times n}$, $v\in\mathbb{R}^{n}$: $y = Av\in\mathbb{R}^m$ with $y_i = \sum_j a_{ij}v_j$, the dot product of row $i$ with $v$. The vector's length must match the number of **columns**; $vA$ is not a valid product.

```python
A = torch.tensor([[1, 3, 2, 4],
                  [2, 4, 3, 5],
                  [0, 2, 0, 7]])
v = torch.tensor([-1, 0, 3, 2])
A @ v          # also torch.matmul(A, v), torch.mv(A, v)
```

```
tensor([13, 17, 14])
```

Row 1: $-1 + 0 + 6 + 8 = 13$.

### Matrix product

<!-- fig:found-matmul -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 210" width="620" height="210" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>2 by 3 times 3 by 2 equals 2 by 2, entry 17 highlighted</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><rect x="20" y="50" width="34" height="34" rx="0" fill="var(--accent)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="37" y="67" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">3</text><rect x="54" y="50" width="34" height="34" rx="0" fill="var(--accent)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="71" y="67" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">0</text><rect x="88" y="50" width="34" height="34" rx="0" fill="var(--accent)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="105" y="67" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">1</text><rect x="20" y="84" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="37" y="101" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">0</text><rect x="54" y="84" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="71" y="101" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">2</text><rect x="88" y="84" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="105" y="101" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">2</text><text x="71" y="34" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">A: (2 × 3)</text><text x="140" y="84" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:18px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">@</text><rect x="160" y="33" width="34" height="34" rx="0" fill="var(--hi)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="177" y="50" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">4</text><rect x="194" y="33" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="211" y="50" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">0</text><rect x="160" y="67" width="34" height="34" rx="0" fill="var(--hi)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="177" y="84" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">1</text><rect x="194" y="67" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="211" y="84" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">3</text><rect x="160" y="101" width="34" height="34" rx="0" fill="var(--hi)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="177" y="118" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">5</text><rect x="194" y="101" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="211" y="118" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">1</text><text x="194" y="20" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">B: (3 × 2)</text><text x="250" y="84" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:18px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">=</text><rect x="270" y="50" width="34" height="34" rx="0" fill="var(--accent2)" fill-opacity="0.28" stroke="currentColor" stroke-width="1.2"/><text x="287" y="67" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">17</text><rect x="304" y="50" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="321" y="67" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">1</text><rect x="270" y="84" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="287" y="101" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">12</text><rect x="304" y="84" width="34" height="34" rx="0" fill="none" stroke="currentColor" stroke-width="1.2"/><text x="321" y="101" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">8</text><text x="304" y="34" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">C: (2 × 2)</text><text x="360" y="60" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">c₁₁ = row 1 of A · column 1 of B</text><text x="360" y="80" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">= 3·4 + 0·1 + 1·5 = 17</text><text x="360" y="115" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">(m × n) @ (n × p) → (m × p)</text><text x="360" y="135" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">inner sizes must match</text><text x="20" y="175" text-anchor="start" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">B @ A is (3 × 2)(2 × 3) = (3 × 3): a different matrix. AB ≠ BA in general.</text></svg><figcaption><b>Matrix product, Lecture 2 example.</b> Entry <i>c<sub>ij</sub></i> is the dot product of row <i>i</i> of <i>A</i> with column <i>j</i> of <i>B</i>.</figcaption></figure>

$$
\begin{bmatrix}3&0&1\\0&2&2\end{bmatrix}\begin{bmatrix}4&0\\1&3\\5&1\end{bmatrix}=\begin{bmatrix}3\cdot4+0\cdot1+1\cdot5 & 3\cdot0+0\cdot3+1\cdot1\\0\cdot4+2\cdot1+2\cdot5 & 0\cdot0+2\cdot3+2\cdot1\end{bmatrix}=\begin{bmatrix}17&1\\12&8\end{bmatrix}
$$

```python
A = torch.tensor([[3, 0, 1], [0, 2, 2]])
B = torch.tensor([[4, 0], [1, 3], [5, 1]])
A @ B          # torch.mm(A, B), torch.matmul(A, B)
```

> [!key] `torch.matmul` (`@`) by dimension
> | operands | result |
> |---|---|
> | 1-D @ 1-D, both $(n)$ | 0-d scalar: the dot product |
> | 2-D @ 1-D, $(m, n)$ @ $(n)$ | $(m)$ |
> | 1-D @ 2-D, $(n)$ @ $(n, p)$ | $(p)$ (the vector acts as a row) |
> | 2-D @ 2-D, $(m, n)$ @ $(n, p)$ | $(m, p)$ |
> | batched, $(b, m, n)$ @ $(b, n, p)$ | $(b, m, p)$; batch dims broadcast |
>
> `torch.mm` is 2-D only and `torch.mv` is matrix-vector only; neither broadcasts.

### Hadamard (elementwise) product

$D = A\odot B$, $d_{ij}=a_{ij}b_{ij}$, for matrices of the same shape. In PyTorch it is `*`:

```python
A = torch.tensor([[1, 2, 3], [0, 1, 0]])
B = torch.tensor([[2, -3, 0], [7, 5, 2]])
A * B          # [[2, -6, 0], [0, 5, 0]]
```

## Matrix inverse

For a square $B\in\mathbb{R}^{n\times n}$, $B^{-1}$ satisfies $BB^{-1}=B^{-1}B=I$. Not every square matrix has one: **singular** matrices (determinant 0, or a zero eigenvalue) do not, and `torch.inverse` raises a runtime error on them.

```python
B = torch.tensor([[1, 2, 3],
                  [-2, 2, 0],
                  [-1, 0, 4]]).float()
B_inv = torch.inverse(B)
print(B_inv)
print(B_inv @ B)      # identity, up to rounding
```

```
tensor([[ 0.2667, -0.2667, -0.2000],
        [ 0.2667,  0.2333, -0.2000],
        [ 0.0667, -0.0667,  0.2000]])
tensor([[ 1.0000e+00,  5.9605e-08,  5.9605e-08],
        [ 1.4901e-08,  1.0000e+00,  5.9605e-08],
        [-1.4901e-08,  1.4901e-08,  1.0000e+00]])
```

Here $\det B = 30$ and $B^{-1} = \tfrac{1}{30}\begin{bmatrix}8&-8&-6\\8&7&-6\\2&-2&6\end{bmatrix}$. `B_inv @ B` prints entries like `5.9605e-08` off the diagonal: floating-point rounding, not an error. The inverse is how the review lecture solves systems of equations and how the [[3-linear-models/multiple-linear-regression|normal equations]] $w^*=(X^\top X)^{-1}X^\top y$ are written; in code, [`torch.linalg.solve`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve.html) solves $Ax=b$ more accurately than forming $A^{-1}$.

## Practice: Lecture 2 exercise

> [!question] Compute, and state each result's shape first
> $u = (1, -2, 0, 4)^\top\in\mathbb{R}^4$, $v = (3, 0, 6, 0, 4)^\top\in\mathbb{R}^5$,
> $A=\begin{bmatrix}6&2&3&0\\-2&2&1&3\\9&0&4&5\\7&7&7&1\\0&1&3&0\end{bmatrix}\in\mathbb{R}^{5\times4}$.
> (a) $vu^\top$ (b) $v^\top Au$ (c) $\Vert Au-v\Vert_2^2$ (d) $A^\top A + (u^\top u)I$

> [!solution]- Answers
> ```python
> torch.outer(v, u)                     # (a) (5, 4)
> torch.dot(v, A @ u)                   # (b) scalar
> torch.norm(A @ u - v) ** 2            # (c) scalar
> A.T @ A + torch.dot(u, u) * torch.eye(4)  # (d) (4, 4)
> ```
> (a) $5\times4$, rows are $3u^\top, 0, 6u^\top, 0, 4u^\top$: first row $(3, -6, 0, 12)$.
> (b) $Au = (2, 6, 29, -3, -2)^\top$, so $v^\top Au = 6 + 0 + 174 + 0 - 8 = 172$.
> (c) $Au - v = (-1, 6, 23, -3, -6)$, so $1 + 36 + 529 + 9 + 36 = 611$.
> (d) $u^\top u = 21$; the result is $4\times4$: $\begin{bmatrix}191&57&101&46\\57&79&60&13\\101&60&105&30\\46&13&30&56\end{bmatrix}$.
> The notebook's answer mixes `np.dot` and `np.eye` into (d) and gets a `float64` result with deprecation warnings; stay in PyTorch.

> [!trap] Common mistakes
> - `A * B` when you meant `A @ B`, or the reverse. `*` needs equal (or broadcastable) shapes; `@` needs matching inner sizes.
> - Writing $vA$ for a column vector $v$; check that inner dimensions meet.
> - `torch.dot` on 2-D tensors or on two different dtypes: an error. Use `@`, and `.float()` both.
> - Calling `torch.norm` on an integer tensor (error: it needs floats).
> - Forgetting the square in $\Vert x\Vert_2^2$: `torch.norm(x) ** 2`, or `x @ x`.
> - Using `max_i v_i` for the $L_\infty$ norm (see the typo above).

> [!exam] On the exams
> Linear algebra appears inside other problems: [[6-midterm-1/fall-2024|Fall 2024]] Q2(c) asks for `z = y.T @ x` by hand (worked on [[1-foundations/indexing-and-slicing#advanced-indexing-lists-and-tensors|indexing and slicing]]), [[6-midterm-1/spring-2025|Spring 2025]] Q1(b) hinges on `@` of two 1-D tensors being a dot product ([[1-foundations/storage-and-views#views-and-in-place-updates-on-the-exams|storage and views]]), and every [[3-linear-models/multiple-linear-regression|linear regression]] problem is matrix shapes. Norms and $x^\top Ax$ return in [[2-calculus-and-optimization/matrix-calculus|matrix calculus]].

## PyTorch docs

- [`torch.matmul`](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html) — the behaviour table for every combination of dimensions.
- [`torch.mm`](https://docs.pytorch.org/docs/stable/generated/torch.mm.html), [`torch.mv`](https://docs.pytorch.org/docs/stable/generated/torch.mv.html), [`torch.dot`](https://docs.pytorch.org/docs/stable/generated/torch.dot.html), [`torch.outer`](https://docs.pytorch.org/docs/stable/generated/torch.outer.html), [`torch.mul`](https://docs.pytorch.org/docs/stable/generated/torch.mul.html) — the specific products.
- [`torch.transpose`](https://docs.pytorch.org/docs/stable/generated/torch.transpose.html), [`Tensor.transpose`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.transpose.html), [`Tensor.T`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor.T), [`Tensor.t`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.t.html) — transposes.
- [`torch.norm`](https://docs.pytorch.org/docs/stable/generated/torch.norm.html), [`torch.linalg.vector_norm`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html), [`torch.linalg.norm`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.norm.html), [`torch.cdist`](https://docs.pytorch.org/docs/stable/generated/torch.cdist.html) — norms and distances.
- [`torch.nn.functional.cosine_similarity`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html) — the angle as a similarity score.
- [`torch.inverse`](https://docs.pytorch.org/docs/stable/generated/torch.inverse.html), [`torch.linalg.inv`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.inv.html), [`torch.linalg.solve`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve.html), [`torch.eye`](https://docs.pytorch.org/docs/stable/generated/torch.eye.html) — inverses and linear systems.
- [`torch.linalg`](https://docs.pytorch.org/docs/stable/linalg.html) — the whole linear-algebra module.

Related: [[1-foundations/tensors]] · [[1-foundations/broadcasting]] · [[2-calculus-and-optimization/matrix-calculus]] · [[3-linear-models/multiple-linear-regression]] · [[6-midterm-1/formula-sheet|Formula sheet]]
