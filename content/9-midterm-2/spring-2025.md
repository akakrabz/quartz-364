---
title: "Spring 2025 Midterm 2 — worked"
description: "ECE 364 Midterm 2, Spring 2025 (the other section), every in-scope problem restated with a folded worked solution: ten true/false statements on CNNs, activations, RNNs and attention, an underfitting diagnosis from a loss plot, a 3×3 max pool with stride 2, shapes and parameters of a two-layer MLP, a two-unit ReLU network that equals mx + b, and a single-head attention module written as an nn.Module. The parts the course staff excluded this semester (PCA, K-Means, GMM, GANs, BERT pre-training, SSD) are kept in place and marked as skipped."
tags: [exam, midterm-2]
aliases: ["Spring 2025 Midterm 2", "MT2 Spring 2025", "Midterm 2 Spring 2025"]
---

*Spring 2025 · May 1, 2025 · 75 minutes · the other section · only resources: the cheatsheet and scratch paper stapled to the back · 7 problems, 100 points · the other exams: [[9-midterm-2/fall-2024|Fall 2024]], [[9-midterm-2/fall-2025|Fall 2025]], [[9-midterm-2/spring-2026|Spring 2026]] · back to [[9-midterm-2/index|Midterm 2]]*

> [!abstract] In one breath
> Fifteen true/false items (30), then short problems worth 10 each: read a **loss plot**, do a **max pool** by hand, give **shapes and parameter counts** of a small MLP, and **design a ReLU network** by hand. The last problem (15) is code: a complete **attention module**. The Spring section covered clustering, PCA, GANs and object detection before its Midterm 2; this semester's exam does not, so 25 points here are off the table.

> [!note] Skipped parts
> The course staff's list for Fall 2026: skip **1(a), (b), (d), (k), (n) and all of Problem 6** (PCA, K-Means, GMMs, GANs, BERT pre-training, object detection). They are **not on our Midterm 2**. Those parts are kept in their place below and marked **[skipped]**, with the official answer and a one-line reason; they are also listed on [[9-midterm-2/skipped-topics|skipped topics]]. In scope: 20 of the 30 true/false points and Problems 2, 3, 4, 5 and 7 (75 points).

Solutions are folded: try each problem first. Every number was rechecked by script (`tools/verify_mt2.py`, and with PyTorch in `tools/verify_mt2_torch.py`) and agrees with the official solutions.

## Problem 1 — Fact or cap? (30 pts)

> [!question] Problem 1
> For each statement, circle whether it is true or false.
>
> **(a)** **[skipped]** The first principal component captures the lowest variance of the dataset.
>
> **(b)** **[skipped]** In each iteration of K-Means, centroids are updated by moving them toward the point with the smallest distance in their cluster.
>
> **(c)** In a convolutional layer, each weight (or parameter) is used exactly once while computing the output of the layer.
>
> **(d)** **[skipped]** A strong discriminator too early in GAN training would make it difficult for the generator to improve.
>
> **(e)** The sigmoid activation function can cause vanishing gradients in deep networks.
>
> **(f)** The derivative of ReLU is always 1.
>
> **(g)** Standard RNNs are designed to handle sequential data by maintaining a hidden state across time steps.
>
> **(h)** LSTMs suffer more from the vanishing gradient problem than standard RNNs do.
>
> **(i)** Self-attention compares every token to every other token in the sequence.
>
> **(j)** A convolutional layer applies different kernels (filters) to different regions of the input.
>
> **(k)** **[skipped]** Masked language modeling is used to pre-train BERT by randomly masking input tokens.
>
> **(l)** In decoder self-attention, a causal mask prevents tokens from attending to future positions.
>
> **(m)** Decoder-only transformer models have no cross-attention mechanism.
>
> **(n)** **[skipped]** Single shot multibox detection uses the same default-box scale for every feature map level.
>
> **(o)** In deep NNs, having multiple linear layers back to back without an activation function in between is mathematically no different than having one linear layer.

> [!solution]- Solution
> **(a) False.** **[skipped]** It captures the **highest** variance.
>
> **(b) False.** **[skipped]** Each centroid moves to the mean of the points assigned to it.
>
> **(c) False.** The kernel slides: every kernel weight is used once **per output position**. A $3\times3$ kernel on an $8\times8$ input produces $6\times6 = 36$ outputs, and each of its 9 weights enters all 36. That reuse is weight sharing. → [[5-deep-networks/convolution#one-filter-applied-everywhere|one filter, applied everywhere]]
>
> **(d) True.** **[skipped]** A discriminator that rejects every generated sample with confidence leaves the generator little to learn from.
>
> **(e) True.** $\sigma'(z) = \sigma(z)(1-\sigma(z)) \le \tfrac14$, and it is near 0 for large $\lvert z\rvert$. Backpropagation multiplies one such factor per layer, so the gradient shrinks with depth. → [[5-deep-networks/activation-functions#vanishing-gradients-and-dead-units|vanishing gradients]]
>
> **(f) False.** It is 1 for $x > 0$ and **0 for $x < 0$**.
>
> **(g) True.** $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$: the hidden state is the memory. → [[5-deep-networks/recurrent-neural-networks|RNNs]]
>
> **(h) False.** The other way round: the LSTM's cell state, updated additively through gates, was designed to reduce vanishing gradients. → [[5-deep-networks/lstm-and-gru|LSTM and GRU]]
>
> **(i) True.** $QK^\top$ is an $N\times N$ table with one score for every (query token, key token) pair. → [[7-transformers/attention|attention]]
>
> **(j) False.** Each filter is applied to **every** region. A layer has several filters (one per output channel), but none of them is tied to a location.
>
> **(k) True.** **[skipped]** BERT is pre-trained by hiding some input tokens and predicting them.
>
> **(l) True.** $-\infty$ is added to the scores above the diagonal, so those weights become 0. → [[7-transformers/attention#masked-self-attention|masked self-attention]]
>
> **(m) True.** Cross-attention reads an encoder's output. With no encoder there is nothing to cross-attend to; a decoder-only model has masked self-attention and feed-forward blocks only.
>
> **(n) False.** **[skipped]** The default-box scale changes from one feature-map level to the next.
>
> **(o) True.** $W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$: one linear layer. → [[5-deep-networks/activation-functions|why activations are needed]]

Topics: [[5-deep-networks/convolution|convolution]] · [[5-deep-networks/activation-functions|activations]] · [[5-deep-networks/lstm-and-gru|LSTM]] · [[7-transformers/attention|attention]] · drill: [[9-midterm-2/true-false|T/F bank]]

## Problem 2 — It trained all night and still flopped (10 pts)

> [!question] Problem 2
> In your machine learning course, you have been assigned to build a classification model to detect whether emails are spam or not. After dedicating several days to the project, you have designed a promising model. But during training you observe the following behavior in the training and validation losses.
>
> **(a)** You know your classmates have been able to achieve lower losses than what you have plateaued at. Which of the following is a possible issue with the training? Underfitting / Overfitting / No issue
>
> **(b)** Justify your answer. If you selected "Underfitting" or "Overfitting", suggest a possible remedy to mitigate the issue. Use no more than four sentences in total.

<!-- fig:exam2-sp25-loss -->
<figure class="cs-fig"><svg role="img" aria-label="Training and validation loss curves that fall together and plateau at about 0.28" style="max-width:520px;width:100%;height:auto" stroke-linejoin="round" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 365.36 217.19" xmlns="http://www.w3.org/2000/svg" version="1.1"><defs></defs><g><g><path d="M 0 217.19  L 365.36 217.19  L 365.36 0  L 0 0  L 0 217.19  z " style="fill: none"/></g><g><g><path d="M 46.34 177.7  L 347.66 177.7  L 347.66 11.38  L 46.34 11.38  L 46.34 177.7  z " style="fill: none"/></g><g><g><g><path d="M 46.34 177.7  L 46.34 11.38  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><defs><path id="f7ecdd0" d="M 0 0  L 0 3.5  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#f7ecdd0" x="46.34" y="177.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="46.34" y="193.06" transform="rotate(-0 46.34 193.06)">0</text></g></g><g><g><path d="M 106.6 177.7  L 106.6 11.38  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#f7ecdd0" x="106.6" y="177.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="106.6" y="193.06" transform="rotate(-0 106.6 193.06)">100</text></g></g><g><g><path d="M 166.87 177.7  L 166.87 11.38  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#f7ecdd0" x="166.87" y="177.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="166.87" y="193.06" transform="rotate(-0 166.87 193.06)">200</text></g></g><g><g><path d="M 227.13 177.7  L 227.13 11.38  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#f7ecdd0" x="227.13" y="177.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="227.13" y="193.06" transform="rotate(-0 227.13 193.06)">300</text></g></g><g><g><path d="M 287.4 177.7  L 287.4 11.38  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#f7ecdd0" x="287.4" y="177.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="287.4" y="193.06" transform="rotate(-0 287.4 193.06)">400</text></g></g><g><g><path d="M 347.66 177.7  L 347.66 11.38  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#f7ecdd0" x="347.66" y="177.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="347.66" y="193.06" transform="rotate(-0 347.66 193.06)">500</text></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="197" y="207.7" transform="rotate(-0 197 207.7)">epoch</text></g></g><g><g><g><path d="M 46.34 177.7  L 347.66 177.7  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><defs><path id="f7ecdd1" d="M 0 0  L -3.5 0  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#f7ecdd1" x="46.34" y="177.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="39.34" y="181.88" transform="rotate(-0 39.34 181.88)">0.0</text></g></g><g><g><path d="M 46.34 153.94  L 347.66 153.94  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#f7ecdd1" x="46.34" y="153.94" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="39.34" y="158.12" transform="rotate(-0 39.34 158.12)">0.1</text></g></g><g><g><path d="M 46.34 130.18  L 347.66 130.18  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#f7ecdd1" x="46.34" y="130.18" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="39.34" y="134.36" transform="rotate(-0 39.34 134.36)">0.2</text></g></g><g><g><path d="M 46.34 106.42  L 347.66 106.42  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#f7ecdd1" x="46.34" y="106.42" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="39.34" y="110.6" transform="rotate(-0 39.34 110.6)">0.3</text></g></g><g><g><path d="M 46.34 82.66  L 347.66 82.66  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#f7ecdd1" x="46.34" y="82.66" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="39.34" y="86.84" transform="rotate(-0 39.34 86.84)">0.4</text></g></g><g><g><path d="M 46.34 58.9  L 347.66 58.9  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#f7ecdd1" x="46.34" y="58.9" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="39.34" y="63.08" transform="rotate(-0 39.34 63.08)">0.5</text></g></g><g><g><path d="M 46.34 35.14  L 347.66 35.14  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#f7ecdd1" x="46.34" y="35.14" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="39.34" y="39.32" transform="rotate(-0 39.34 39.32)">0.6</text></g></g><g><g><path d="M 46.34 11.38  L 347.66 11.38  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#f7ecdd1" x="46.34" y="11.38" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="39.34" y="15.56" transform="rotate(-0 39.34 15.56)">0.7</text></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="15.56" y="94.54" transform="rotate(-90 15.56 94.54)">loss</text></g></g><g><path d="M 46.34 51.77  L 48.75 57.71  L 51.16 63.05  L 53.57 67.86  L 55.98 72.18  L 58.39 76.08  L 60.8 79.59  L 63.21 82.74  L 65.62 85.58  L 68.03 88.14  L 71.05 90.98  L 74.06 93.47  L 77.07 95.65  L 80.09 97.56  L 83.1 99.24  L 86.72 100.98  L 90.33 102.47  L 94.55 103.94  L 98.77 105.15  L 103.59 106.3  L 109.01 107.32  L 115.04 108.21  L 121.67 108.96  L 129.5 109.6  L 138.54 110.11  L 149.99 110.53  L 165.06 110.84  L 186.75 111.04  L 225.32 111.15  L 347.66 111.17  L 347.66 111.17  " clip-path="url(#f7ecdd2)" style="fill: none; stroke: var(--accent); stroke-width: 1.8; stroke-linecap: square"/></g><g><path d="M 46.34 51.77  L 48.75 57.54  L 51.16 62.73  L 53.57 67.41  L 55.98 71.61  L 58.39 75.4  L 60.8 78.81  L 63.21 81.87  L 65.62 84.63  L 68.03 87.12  L 71.05 89.88  L 74.06 92.3  L 77.07 94.42  L 80.09 96.28  L 83.1 97.91  L 86.72 99.61  L 90.33 101.05  L 94.55 102.47  L 98.77 103.66  L 103.59 104.77  L 109.01 105.77  L 115.04 106.63  L 121.67 107.36  L 129.5 107.98  L 139.15 108.5  L 151.2 108.92  L 166.87 109.21  L 190.37 109.4  L 233.76 109.49  L 347.66 109.51  L 347.66 109.51  " clip-path="url(#f7ecdd2)" style="fill: none; stroke-dasharray: 6.66,2.88; stroke-dashoffset: 0; stroke: var(--hi); stroke-width: 1.8"/></g><g><path d="M 46.34 177.7  L 46.34 11.38  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><path d="M 46.34 177.7  L 347.66 177.7  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><g><path d="M 249.8 23.17  L 258.8 23.17  L 267.8 23.17  " style="fill: none; stroke: var(--accent); stroke-width: 1.8; stroke-linecap: square"/></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="275" y="26.32" transform="rotate(-0 275 26.32)">training loss</text></g><g><path d="M 249.8 36.38  L 258.8 36.38  L 267.8 36.38  " style="fill: none; stroke-dasharray: 6.66,2.88; stroke-dashoffset: 0; stroke: var(--hi); stroke-width: 1.8"/></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="275" y="39.53" transform="rotate(-0 275 39.53)">validation loss</text></g></g></g></g><defs><clipPath id="f7ecdd2"><rect x="46.34" y="11.38" width="301.32" height="166.32"/></clipPath></defs></svg> <figcaption>Spring 2025 Problem 2, re-drawn from the exam's figure: training and validation loss fall together from about 0.53 and flatten near 0.28 by epoch 200, almost on top of each other.</figcaption></figure>

> [!solution]- Solution
> **(a) Underfitting.**
>
> **(b)** Both losses fall together and flatten at a **high** value (about 0.28, while classmates get lower), and they lie on top of each other: there is no gap between training and validation. The model is not capturing even the training data, so it is too simple for the task. Remedy: increase the model's capacity (more parameters: more or wider layers); replacing linear activations by non-linear ones would also help.
>
> How to rule out the other two answers:
> - **Overfitting** would show the training loss still falling while the validation loss turns around and **rises**. Here the two curves never separate.
> - **No issue** is excluded by the problem text: lower losses are known to be reachable.
>
> "Train longer" is not a good remedy here, because the curves have been flat for 300 epochs. The checklist from Lecture 22 would add one more thing to try: a loss that flattens early can also mean the learning rate is too high. → [[4-pytorch-training/underfitting-and-overfitting#the-three-cases|the three cases]]

Topics: [[4-pytorch-training/underfitting-and-overfitting|underfitting and overfitting]] · [[4-pytorch-training/experimental-setup#overfitting|validation data]] · drill: [[9-midterm-2/review-training-and-activations|training behaviour and activations]]

## Problem 3 — Life's a pool, dive in (10 pts)

> [!question] Problem 3
> Consider the $5\times5$ input
> $$
> \begin{bmatrix}0&1&2&3&4\\4&3&2&1&0\\1&2&3&4&5\\5&4&3&2&1\\0&1&0&1&0\end{bmatrix}
> $$
> Compute the output of `MaxPool2d(3, stride=2)`.

> [!solution]- Solution
> Output size: $\lfloor(5-3)/2\rfloor + 1 = 2$ in each direction. The $3\times3$ windows start at rows 0 and 2 and at columns 0 and 2, so neighbouring windows **overlap** in row 2 and column 2.
>
> | window | entries | max |
> |---|---|---|
> | rows 0–2, cols 0–2 | 0 1 2 / 4 3 2 / 1 2 3 | 4 |
> | rows 0–2, cols 2–4 | 2 3 4 / 2 1 0 / 3 4 5 | 5 |
> | rows 2–4, cols 0–2 | 1 2 3 / 5 4 3 / 0 1 0 | 5 |
> | rows 2–4, cols 2–4 | 3 4 5 / 3 2 1 / 0 1 0 | 5 |
>
> $$
> \begin{bmatrix}4&5\\5&5\end{bmatrix}
> $$
> The 5 at row 2, column 4 is the maximum of the top-right and the bottom-right window: with stride smaller than the kernel, one input entry can be counted more than once. (Compare [[9-midterm-2/fall-2024#problem-4--a-convolutional-layer-and-a-max-pool-15-pts|Fall 2024 Problem 4(c)]], where the stride is larger than the kernel and some entries are never read.)

Topics: [[5-deep-networks/pooling#worked-example|pooling by hand]] · drill: [[9-midterm-2/review-sizes-and-parameters|sizes and parameter counts]]

## Problem 4 — Who wants to be a parameterionaire? (10 pts)

> [!question] Problem 4
> Consider the following network:
> ```python
> model = nn.Sequential(
>     nn.Linear(8, 16, bias=True),
>     nn.ReLU(),
>     nn.Linear(16, 4, bias=True)
> )
> ```
> **(a)** Suppose the input is of shape `(1, 8)`. What are the shapes after the first Linear, after ReLU, and after the second Linear?
>
> **(b)** Determine the total learnable parameters in the model.

> [!solution]- Solution
> **(a)** `nn.Linear(in, out)` replaces the **last** dimension; an activation changes no shape.
>
> | after | shape |
> |---|---|
> | first `Linear(8, 16)` | `(1, 16)` |
> | `ReLU` | `(1, 16)` |
> | second `Linear(16, 4)` | `(1, 4)` |
>
> **(b)** A linear layer with bias has $\text{in}\cdot\text{out} + \text{out}$ parameters; ReLU has none.
> $$
> (8\cdot16+16)+(16\cdot4+4)=144+68=212 .
> $$

Topics: [[5-deep-networks/multilayer-perceptrons#counting-parameters|counting parameters]] · [[4-pytorch-training/nn-module#nnlinear-a-weight-matrix-with-a-bias|nn.Linear]] · drill: [[9-midterm-2/review-sizes-and-parameters|sizes and parameter counts]]

## Problem 5 — Layer by layer: a neural network love story (10 pts)

> [!question] Problem 5
> In this problem, you will be hand designing a simple neural network to model a specific function. Assume $x\in\mathbb R$ and provide appropriate weights $w_0, w_1, b_0\in\mathbb R^2$. In other words, the neural network has one input neuron, 2 hidden neurons, and one output neuron.
>
> Find $w_0, w_1, b_0\in\mathbb R^2$ such that $f(x) = w_1^\top\sigma(w_0x + b_0) = mx + b$ for all $x\in\mathbb R$, where $\sigma = \mathrm{ReLU}$. (Here $w_0x$ is a vector times a scalar, e.g. $[0, 1]^\top x = [0, x]^\top$.) Show why your answer is correct.

> [!solution]- Solution
> **The idea.** ReLU keeps the positive part of a number and throws the rest away. Two units can share the work: one keeps the positive part of $u$, the other keeps the positive part of $-u$, and their difference gives $u$ back:
> $$
> \mathrm{ReLU}(u)-\mathrm{ReLU}(-u)=u\quad\text{for every }u .
> $$
> (If $u > 0$ this is $u - 0$; if $u < 0$ it is $0 - (-u)$; if $u = 0$ it is $0$.)
>
> **The answer.** Take $u = mx + b$:
> $$
> w_0=\begin{bmatrix}m\\-m\end{bmatrix},\qquad b_0=\begin{bmatrix}b\\-b\end{bmatrix},\qquad w_1=\begin{bmatrix}1\\-1\end{bmatrix}.
> $$
> **Check.** $w_0x + b_0 = [\,mx + b,\ -(mx + b)\,]^\top$, so
> $$
> f(x)=\mathrm{ReLU}(mx+b)-\mathrm{ReLU}(-(mx+b))=mx+b .
> $$
> Other answers exist (the key says so): scale a hidden unit by any $a > 0$ and divide its output weight by $a$, for example $w_0 = [2m, -m]^\top$, $b_0 = [2b, -b]^\top$, $w_1 = [\tfrac12, -1]^\top$.
>
> *Checked numerically for several $(m, b)$ on a grid of $x$, including $m = 0$.* The Spring 2026 version asks for $\lvert x\rvert$ with the same network: [[9-midterm-2/spring-2026#problem-5--layer-by-layer-the-kink-awakens-10-pts|Spring 2026 Problem 5]].

Topics: [[5-deep-networks/activation-functions|ReLU]] · [[5-deep-networks/multilayer-perceptrons|MLPs]] · drill: [[9-midterm-2/review-training-and-activations#relu-networks-by-hand|ReLU networks by hand]]

## Problem 6 — Two clusters walk into a bar… (15 pts)

> [!question] Problem 6 **[skipped]**
> **(a)** For each figure below, indicate whether it is possible for K-Means, GMM, neither, or both algorithms to produce the clustering assignments shown (as indicated by the two colors, which are there for visualization purposes only). In 1–2 sentences, briefly explain your reasoning.
>
> *Figure (a):* two well-separated, elongated clouds of points, a blue one at the lower left and a red one at the upper right. *Figure (b):* two concentric rings of points, a small blue ring inside a larger red ring. (The figures are described here, not redrawn.)
>
> **(b)** Which algorithm is more sensitive to outliers: K-Means or GMM? Explain.
>
> **(c)** Which of the following is **not** minimized (or maximized) by K-Means or GMM?
>
> - **A.** $\sum_{i=1}^n\min_k\Vert x_i-\mu_k\Vert^2$
> - **B.** $-\sum_{i=1}^n\log\big(\sum_{k=1}^K\pi_k\,\mathcal N(x_i\mid\mu_k,\Sigma_k)\big)$
> - **C.** $\sum_{i=1}^n\min_k\Vert x_i-\mu_k\Vert$
> - **D.** $\sum_{i=1}^n\sum_{k=1}^K\gamma_{ik}\Vert x_i-\mu_k\Vert^2$

> [!solution]- Solution
> **[skipped]** Official answers.
>
> **(a)** Figure (a): **both**. Figure (b): **neither**. (The key gives the two words only.)
>
> **(b)** **K-Means** is more sensitive, because it uses means, which shift due to outliers. A GMM can be more robust if the outlier is far enough to get very low probability under all components.
>
> **(c)** **C.** K-Means uses squared distances and a GMM optimizes a likelihood; neither optimizes the sum of unsquared distances.

## Problem 7 — Attention is all you need! (15 pts)

> [!question] Problem 7
> The Transformer architecture integrates attention mechanisms with multi-layer perceptrons (MLPs). The attention mechanism takes three inputs: queries ($Q$), keys ($K$) and values ($V$). Each has a shape of $B\times N\times d_{model}$, where $B$, $N$ and $d_{model}$ are the batch size, sequence length and model dimension (hidden size). In this question, you will implement a simple attention mechanism without using multiple heads. The attention outputs are computed as follows.
>
> **Step 1.** Apply learned projection matrices:
> $$
> Q_w=QW^Q+b^Q,\qquad K_w=KW^K+b^K,\qquad V_w=VW^V+b^V,
> $$
> where $W^Q, W^K, W^V\in\mathbb R^{d_{model}\times d_{model}}$ and $b^Q, b^K, b^V\in\mathbb R^{d_{model}}$.
>
> **Step 2.** Compute the attention outputs:
> $$
> A=\text{softmax}\!\left(\frac{Q_wK_w^\top}{\sqrt{d_{model}}}\right)V_w
> $$
> **Step 3.** Apply another linear projection on $A$:
> $$
> \text{Attention}(Q,K,V)=AW^O+b^O,
> $$
> where $W^O\in\mathbb R^{d_{model}\times d_{model}}$ and $b^O\in\mathbb R^{d_{model}}$.
>
> Complete the code below to implement attention. You can ignore attention masks, multi-head attention, dropout and optimized implementations.
> ```python
> import torch
> import torch.nn as nn
> from torch.nn.functional import softmax
>
> class Attention(nn.Module):
>     def __init__(self, d_model: int) -> None:
>         self.d_model = d_model
>         # Space to add more class variables as required
>
>     def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
>         """
>         Q : B x N x d_model
>         K : B x N x d_model
>         V : B x N x d_model
>         """
>         # Complete this
> ```

> [!solution]- Solution
> Each "matrix plus bias" in the steps is an `nn.Linear(d_model, d_model)`: four of them. The forward pass is the three steps in order.
> ```python
> class Attention(nn.Module):
>     def __init__(self, d_model: int) -> None:
>         super().__init__()                      # see the note below
>         self.d_model = d_model
>         self.wq = nn.Linear(d_model, d_model, bias=True)
>         self.wk = nn.Linear(d_model, d_model, bias=True)
>         self.wv = nn.Linear(d_model, d_model, bias=True)
>         self.wo = nn.Linear(d_model, d_model, bias=True)
>         self.scaling = d_model ** -0.5
>
>     def forward(self, Q, K, V):
>         query_states = self.wq(Q) * self.scaling    # B x N x d_model
>         key_states = self.wk(K)                     # B x N x d_model
>         value_states = self.wv(V)                   # B x N x d_model
>
>         attn_weights = query_states @ key_states.transpose(1, 2)   # B x N x N
>         attn_weights = softmax(attn_weights, dim=-1)               # rows sum to 1
>
>         attn_outputs = attn_weights @ value_states  # B x N x d_model
>         attn_outputs = self.wo(attn_outputs)        # B x N x d_model
>         return attn_outputs
> ```
> Line by line:
> - **`transpose(1, 2)`**, not `.T`: the keys are a batch of matrices `(B, N, d)`, and only the last two axes are swapped, giving `(B, d, N)`. `@` then multiplies matrix by matrix inside the batch: `(B, N, d) @ (B, d, N) → (B, N, N)`.
> - **Scaling.** The key multiplies the queries by $d_{model}^{-1/2}$ before the product. That equals dividing the scores afterwards, since the factor is a scalar.
> - **`dim=-1`**: the softmax runs along each **row** of the $N\times N$ table, so every query's weights over the keys sum to 1.
> - **Parameters:** $4(d_{model}^2 + d_{model})$.
>
> **Two slips in the official key** ([[9-midterm-2/errata|errata]]). It stores the projected values as `val_states` and then uses `value_states` (one name is needed, as above). And neither the exam's skeleton nor the key calls `super().__init__()`; in real PyTorch the first `self.wq = nn.Linear(...)` then raises `AttributeError: cannot assign module before Module.__init__() call`. Write the `super().__init__()` line whenever you write a module.
>
> *Run with PyTorch:* for inputs of shape `(2, 5, 8)` the output is `(2, 5, 8)` and equals the three steps computed directly.

Topics: [[7-transformers/attention#scaled-dot-product-attention-step-by-step|scaled dot-product attention]] · [[7-transformers/attention#the-lectures-code|the lecture's attention code]] · [[4-pytorch-training/nn-module|nn.Module]] · drill: [[9-midterm-2/review-attention-and-transformers|attention and Transformers]] · [[9-midterm-2/review-pytorch-modules|writing modules]]

## The cheatsheet that came with the exam

The Spring exams staple a "Deep Learning Cheatsheet" to the back. The in-scope part of it lists: the signatures of `nn.Linear`, `F.sigmoid`, `F.softmax`, `nn.Conv2d`, `nn.MaxPool2d`, `nn.AvgPool2d` and `nn.BatchNorm2d` (with $y = \frac{x - \mathrm E[x]}{\sqrt{\mathrm{Var}[x]+\epsilon}}\gamma + \beta$); the sinusoidal position encoding in the paper's form; self-, cross- and masked attention; multi-head attention; and three code samples (a logistic-regression training script with `Dataset`, `SubsetRandomSampler` and `DataLoader`, a three-layer MLP module, and a small CNN with global average pooling). The rest covers PCA, K-Means, GMMs, GANs and detection metrics. It does **not** print the convolution output-size formula, which the Fall sheet does: see [[9-midterm-2/fall-2025#the-formula-sheet-that-came-with-the-exam|the Fall 2025 sheet]] and the [[9-midterm-2/formula-sheet|formula sheet]] for this semester.

> [!exam] What this exam rewards
> Short, mechanical problems done without slips: a pooling window, a parameter count, a shape after each layer. The two that need an idea are Problem 5 (split a number into its positive and negative parts) and Problem 7 (attention is four linear layers, one matrix product, a softmax along rows, and another matrix product). Both ideas return on [[9-midterm-2/spring-2026|Spring 2026]].
