---
title: "Convolution"
description: "Why an MLP ignores the spatial structure of images, convolution as sliding template matching, 1-D convolution vs cross-correlation worked by hand (and what F.conv1d / nn.Conv2d really compute), 2-D convolution, locality and weight sharing, translation equivariance, and learning a filter with gradient descent."
tags: [topic, deep-nets]
aliases: ["cross-correlation", "correlation", "template matching", "filter", "kernel", "weight sharing", "locality", "translation equivariance", "equivariance", "conv1d"]
---

*Lecture 15 · [[5-deep-networks/index|Deep networks]] · builds on [[5-deep-networks/multilayer-perceptrons#what-an-mlp-cannot-see|what an MLP cannot see]] · continues in [[5-deep-networks/convolutional-layers|convolutional layers]] and [[5-deep-networks/pooling|pooling]]*

> [!abstract] In one breath
> A convolution slides a small **kernel** (filter, template) along a signal or image and takes a dot product at every position, so each output says how well that neighborhood matches the pattern. The textbook convolution $y[n] = \sum_k x[k]\,h[n-k]$ flips the kernel first; PyTorch's `F.conv1d`, `F.conv2d` and `nn.Conv2d` do **not** flip (they compute cross-correlation), which does not matter when the kernel is learned. Two ideas make it the right layer for images: **locality** (each output looks at a small neighborhood) and **weight sharing** (the same kernel everywhere), which together make it **translation equivariant**: shift the input and the output shifts the same way.

## What it is

To the computer an image is a grid of numbers (a $28\times28$ FashionMNIST image is 784 values from 0 to 255) and nothing else: the **semantic gap**. Recognition is hard because the same object gives very different numbers under changes of viewpoint, lighting, deformation, occlusion and clutter, and because the object can be **anywhere**. An [[5-deep-networks/multilayer-perceptrons|MLP]] cannot help here: it treats the image as an unordered list (a fixed scramble of the pixels costs it nothing) and needs separate weights for every position.

**1-D convolution.** For a signal $x$ and a kernel $h$,

$$
y[n] = (x * h)[n] = \sum_{k} x[k]\,h[n-k].
$$

Read it as: **flip** the template $h$, **slide** it along $x$, and at each shift $n$ take the **dot product** of the overlapping entries. A large output means the local piece of $x$ matches the (flipped) template. With $x$ of length $N$ and $h$ of length $K$, the *full* output (every shift with any overlap) has length $N + K - 1$.

**Cross-correlation** skips the flip:

$$
y[n] = (x \star h)[n] = \sum_{k} x[n+k]\,h[k].
$$

Convolution with $h$ equals cross-correlation with the flipped kernel $h[-k]$. Deep learning libraries implement cross-correlation and call it convolution; since the kernel weights are *learned*, the network simply learns the flipped kernel, and nothing changes.

![[l16-1d-convolution-example.png|1-D convolution: the flipped kernel slides along x]]

The figure is the lecture's example: $x = [2, 1, -3, 0, 4]$, $h = [1, 0, -1]$ (a difference filter: it responds to changes, i.e. edges). Each row places the flipped kernel $h[-(k-n)] = [-1, 0, 1]$ at shift $n$; the green circle is the dot product with the overlapping part of $x$.

## What you should know

- The convolution sum $y[n] = \sum_k x[k]\,h[n-k]$: flip, slide, dot product. Full output length $N + K - 1$; with no padding (*valid*) the length is $N - K + 1$.
- Compute a short 1-D convolution and the matching cross-correlation **by hand** (worked example below).
- `F.conv1d`, `F.conv2d`, `nn.Conv1d`, `nn.Conv2d` compute **cross-correlation** (no flip). To get true convolution, flip the kernel yourself with `h.flip(0)`.
- Input shapes: `(batch, channels, length)` for 1-D and `(batch, channels, H, W)` for 2-D; kernel shape `(out_channels, in_channels, K)` or `(..., K, K)`. A single signal needs `x.view(1, 1, -1)`.
- 2-D: a $K\times K$ kernel on an $H\times W$ image with no padding and stride 1 gives $(H-K+1)\times(W-K+1)$; padding and stride: [[5-deep-networks/convolutional-layers#output-size|output-size formula]].
- **Locality** and **weight sharing**; **equivariance** (conv of a shifted input = shifted conv) vs **invariance** (output unchanged), which needs [[5-deep-networks/pooling|global pooling]].
- A filter plus global max pooling detects a pattern **anywhere**; making the kernel weights parameters lets gradient descent **learn** the filter.
- The built-in assumption, *neighboring pixels belong together*, is what makes CNNs data-efficient; when it is false (scrambled pixels) it hurts.

## Worked example: convolution vs cross-correlation

$x = [2, 1, -3, 0, 4]$ (indices $0..4$), $h = [1, 0, -1]$ (indices $0..2$). Full convolution, $y[n] = \sum_k x[k]\,h[n-k]$, keeping only terms where both indices exist:

| $n$ | terms | $y[n]$ |
|---|---|---|
| 0 | $x[0]h[0] = 2\cdot1$ | $2$ |
| 1 | $x[0]h[1] + x[1]h[0] = 0 + 1$ | $1$ |
| 2 | $x[0]h[2] + x[1]h[1] + x[2]h[0] = -2 + 0 - 3$ | $-5$ |
| 3 | $x[1]h[2] + x[2]h[1] + x[3]h[0] = -1 + 0 + 0$ | $-1$ |
| 4 | $x[2]h[2] + x[3]h[1] + x[4]h[0] = 3 + 0 + 4$ | $7$ |
| 5 | $x[3]h[2] + x[4]h[1] = 0 + 0$ | $0$ |
| 6 | $x[4]h[2] = 4\cdot(-1)$ | $-4$ |

Cross-correlation with the same zero padding slides $h$ *unflipped*, giving $[-2, -1, 5, 1, -7, 0, 4]$. Here that is exactly $-1$ times the convolution, because this $h$ is antisymmetric: flipping it gives $[-1, 0, 1] = -h$.

<!-- fig:deepa-conv-vs-corr -->
<figure class="cs-fig"><svg role="img" aria-label="Stem plots of x, h, their convolution and cross-correlation" style="max-width:733px;width:100%;height:auto" stroke-linejoin="round" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 550.08 221.78" xmlns="http://www.w3.org/2000/svg" version="1.1"><defs></defs><g><g><path d="M 0 221.78  L 550.08 221.78  L 550.08 0  L 0 0  L 0 221.78  z " style="fill: none"/></g><g><g><path d="M 7.2 181.58  L 128.11 181.58  L 128.11 20.8  L 7.2 20.8  L 7.2 181.58  z " style="fill: none"/></g><g><g><g><defs><path id="fb10000" d="M 0 0  L 0 3.5  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#fb10000" x="38.56" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><g style="fill: currentColor" transform="translate(35.26 196.93)"><text><tspan x="0" y="-0.98" style="font-style: oblique; font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">x</tspan></text></g></g></g><g><g><g><use xlink:href="#fb10000" x="109.68" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><g style="fill: currentColor" transform="translate(106.16 196.94)"><text><tspan x="0" y="-0.64" style="font-style: oblique; font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">h</tspan></text></g></g></g></g><g/><g><path d="M 12.7 101.19  L 12.7 82.27  " clip-path="url(#fb10005)" style="fill: none; stroke: var(--muted); stroke-width: 2"/><path d="M 25.63 101.19  L 25.63 91.73  " clip-path="url(#fb10005)" style="fill: none; stroke: var(--muted); stroke-width: 2"/><path d="M 38.56 101.19  L 38.56 129.56  " clip-path="url(#fb10005)" style="fill: none; stroke: var(--muted); stroke-width: 2"/><path d="M 51.49 101.19  L 51.49 101.19  " clip-path="url(#fb10005)" style="fill: none; stroke: var(--muted); stroke-width: 2"/><path d="M 64.42 101.19  L 64.42 63.36  " clip-path="url(#fb10005)" style="fill: none; stroke: var(--muted); stroke-width: 2"/></g><g><defs><path id="fb10001" d="M 0 3  C 0.8 3 1.56 2.68 2.12 2.12  C 2.68 1.56 3 0.8 3 0  C 3 -0.8 2.68 -1.56 2.12 -2.12  C 1.56 -2.68 0.8 -3 0 -3  C -0.8 -3 -1.56 -2.68 -2.12 -2.12  C -2.68 -1.56 -3 -0.8 -3 0  C -3 0.8 -2.68 1.56 -2.12 2.12  C -1.56 2.68 -0.8 3 0 3  z " style="stroke: var(--muted)"/></defs><g clip-path="url(#fb10005)"><use xlink:href="#fb10001" x="12.7" y="82.27" style="fill: var(--muted); stroke: var(--muted)"/><use xlink:href="#fb10001" x="25.63" y="91.73" style="fill: var(--muted); stroke: var(--muted)"/><use xlink:href="#fb10001" x="38.56" y="129.56" style="fill: var(--muted); stroke: var(--muted)"/><use xlink:href="#fb10001" x="51.49" y="101.19" style="fill: var(--muted); stroke: var(--muted)"/><use xlink:href="#fb10001" x="64.42" y="63.36" style="fill: var(--muted); stroke: var(--muted)"/></g></g><g/><g><path d="M 96.75 101.19  L 96.75 91.73  " clip-path="url(#fb10005)" style="fill: none; stroke: var(--muted); stroke-width: 2"/><path d="M 109.68 101.19  L 109.68 101.19  " clip-path="url(#fb10005)" style="fill: none; stroke: var(--muted); stroke-width: 2"/><path d="M 122.62 101.19  L 122.62 110.65  " clip-path="url(#fb10005)" style="fill: none; stroke: var(--muted); stroke-width: 2"/></g><g><defs><path id="fb10002" d="M -3 3  L 3 3  L 3 -3  L -3 -3  z " style="stroke: var(--muted); stroke-linejoin: miter"/></defs><g clip-path="url(#fb10005)"><use xlink:href="#fb10002" x="96.75" y="91.73" style="fill: var(--muted); stroke: var(--muted); stroke-linejoin: miter"/><use xlink:href="#fb10002" x="109.68" y="101.19" style="fill: var(--muted); stroke: var(--muted); stroke-linejoin: miter"/><use xlink:href="#fb10002" x="122.62" y="110.65" style="fill: var(--muted); stroke: var(--muted); stroke-linejoin: miter"/></g></g><g/><g><path d="M 7.2 101.19  L 128.11 101.19  " clip-path="url(#fb10005)" style="fill: none; stroke: var(--muted); stroke-width: 0.8; stroke-linecap: square"/></g><g><path d="M 7.2 181.58  L 128.11 181.58  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="12.7" y="77.54" transform="rotate(-0 12.7 77.54)">2</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="25.63" y="87" transform="rotate(-0 25.63 87)">1</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="38.56" y="138.07" transform="rotate(-0 38.56 138.07)">-3</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="51.49" y="96.46" transform="rotate(-0 51.49 96.46)">0</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="64.42" y="58.63" transform="rotate(-0 64.42 58.63)">4</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="96.75" y="87" transform="rotate(-0 96.75 87)">1</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="109.68" y="96.46" transform="rotate(-0 109.68 96.46)">0</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="122.62" y="119.16" transform="rotate(-0 122.62 119.16)">-1</text></g><g><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="7.2" y="14.8" transform="rotate(-0 7.2 14.8)">input and kernel</text></g></g><g><g><path d="M 159.62 181.58  L 335.5 181.58  L 335.5 20.8  L 159.62 20.8  L 159.62 181.58  z " style="fill: none"/></g><g><g><g><g><use xlink:href="#fb10000" x="167.62" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="167.62" y="196.93" transform="rotate(-0 167.62 196.93)">0</text></g></g><g><g><g><use xlink:href="#fb10000" x="194.26" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="194.26" y="196.93" transform="rotate(-0 194.26 196.93)">1</text></g></g><g><g><g><use xlink:href="#fb10000" x="220.91" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="220.91" y="196.93" transform="rotate(-0 220.91 196.93)">2</text></g></g><g><g><g><use xlink:href="#fb10000" x="247.56" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="247.56" y="196.93" transform="rotate(-0 247.56 196.93)">3</text></g></g><g><g><g><use xlink:href="#fb10000" x="274.21" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="274.21" y="196.93" transform="rotate(-0 274.21 196.93)">4</text></g></g><g><g><g><use xlink:href="#fb10000" x="300.85" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="300.85" y="196.93" transform="rotate(-0 300.85 196.93)">5</text></g></g><g><g><g><use xlink:href="#fb10000" x="327.5" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="327.5" y="196.93" transform="rotate(-0 327.5 196.93)">6</text></g></g><g><g style="fill: currentColor" transform="translate(244.04 211.93)"><text><tspan x="0" y="-0.84" style="font-style: oblique; font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">n</tspan></text></g></g></g><g/><g><path d="M 167.62 101.19  L 167.62 82.27  " clip-path="url(#fb10006)" style="fill: none; stroke: var(--accent); stroke-width: 2"/><path d="M 194.26 101.19  L 194.26 91.73  " clip-path="url(#fb10006)" style="fill: none; stroke: var(--accent); stroke-width: 2"/><path d="M 220.91 101.19  L 220.91 148.48  " clip-path="url(#fb10006)" style="fill: none; stroke: var(--accent); stroke-width: 2"/><path d="M 247.56 101.19  L 247.56 110.65  " clip-path="url(#fb10006)" style="fill: none; stroke: var(--accent); stroke-width: 2"/><path d="M 274.21 101.19  L 274.21 34.99  " clip-path="url(#fb10006)" style="fill: none; stroke: var(--accent); stroke-width: 2"/><path d="M 300.85 101.19  L 300.85 101.19  " clip-path="url(#fb10006)" style="fill: none; stroke: var(--accent); stroke-width: 2"/><path d="M 327.5 101.19  L 327.5 139.02  " clip-path="url(#fb10006)" style="fill: none; stroke: var(--accent); stroke-width: 2"/></g><g><defs><path id="fb10003" d="M 0 3  C 0.8 3 1.56 2.68 2.12 2.12  C 2.68 1.56 3 0.8 3 0  C 3 -0.8 2.68 -1.56 2.12 -2.12  C 1.56 -2.68 0.8 -3 0 -3  C -0.8 -3 -1.56 -2.68 -2.12 -2.12  C -2.68 -1.56 -3 -0.8 -3 0  C -3 0.8 -2.68 1.56 -2.12 2.12  C -1.56 2.68 -0.8 3 0 3  z " style="stroke: var(--accent)"/></defs><g clip-path="url(#fb10006)"><use xlink:href="#fb10003" x="167.62" y="82.27" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#fb10003" x="194.26" y="91.73" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#fb10003" x="220.91" y="148.48" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#fb10003" x="247.56" y="110.65" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#fb10003" x="274.21" y="34.99" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#fb10003" x="300.85" y="101.19" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#fb10003" x="327.5" y="139.02" style="fill: var(--accent); stroke: var(--accent)"/></g></g><g/><g><path d="M 159.62 101.19  L 335.5 101.19  " clip-path="url(#fb10006)" style="fill: none; stroke: var(--muted); stroke-width: 0.8; stroke-linecap: square"/></g><g><path d="M 159.62 181.58  L 335.5 181.58  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="167.62" y="75.65" transform="rotate(-0 167.62 75.65)">2</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="194.26" y="85.11" transform="rotate(-0 194.26 85.11)">1</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="220.91" y="161.72" transform="rotate(-0 220.91 161.72)">-5</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="247.56" y="123.89" transform="rotate(-0 247.56 123.89)">-1</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="274.21" y="28.37" transform="rotate(-0 274.21 28.37)">7</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="300.85" y="94.57" transform="rotate(-0 300.85 94.57)">0</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="327.5" y="152.26" transform="rotate(-0 327.5 152.26)">-4</text></g><g><g style="fill: currentColor" transform="translate(159.62 14.8)"><text><tspan x="0" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">c</tspan><tspan x="5.5" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">o</tspan><tspan x="11.62" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">n</tspan><tspan x="17.95" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">v</tspan><tspan x="23.87" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">o</tspan><tspan x="29.99" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">l</tspan><tspan x="32.77" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">u</tspan><tspan x="39.11" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">t</tspan><tspan x="43.03" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">i</tspan><tspan x="45.81" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">o</tspan><tspan x="51.92" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">n</tspan><tspan x="58.26" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor"></tspan><tspan x="69.31" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">*</tspan><tspan x="82.59" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor"></tspan><tspan x="85.77" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">(</tspan><tspan x="89.67" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">k</tspan><tspan x="95.46" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">e</tspan><tspan x="101.62" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">r</tspan><tspan x="105.73" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">n</tspan><tspan x="112.07" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">e</tspan><tspan x="118.22" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">l</tspan><tspan x="121" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor"></tspan><tspan x="124.17" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">f</tspan><tspan x="127.7" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">l</tspan><tspan x="130.47" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">i</tspan><tspan x="133.25" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">p</tspan><tspan x="139.6" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">p</tspan><tspan x="145.95" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">e</tspan><tspan x="152.1" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">d</tspan><tspan x="158.45" y="-0.41" style="font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">)</tspan><tspan x="61.44" y="-0.41" style="font-style: oblique; font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">x</tspan><tspan x="76.25" y="-0.41" style="font-style: oblique; font-size: 10px; font-family: 'DejaVu Sans'; fill: currentColor">h</tspan></text></g></g></g><g><g><path d="M 367.01 181.58  L 542.88 181.58  L 542.88 20.8  L 367.01 20.8  L 367.01 181.58  z " style="fill: none"/></g><g><g><g><g><use xlink:href="#fb10000" x="375" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="375" y="196.93" transform="rotate(-0 375 196.93)">0</text></g></g><g><g><g><use xlink:href="#fb10000" x="401.65" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="401.65" y="196.93" transform="rotate(-0 401.65 196.93)">1</text></g></g><g><g><g><use xlink:href="#fb10000" x="428.3" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="428.3" y="196.93" transform="rotate(-0 428.3 196.93)">2</text></g></g><g><g><g><use xlink:href="#fb10000" x="454.94" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="454.94" y="196.93" transform="rotate(-0 454.94 196.93)">3</text></g></g><g><g><g><use xlink:href="#fb10000" x="481.59" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="481.59" y="196.93" transform="rotate(-0 481.59 196.93)">4</text></g></g><g><g><g><use xlink:href="#fb10000" x="508.24" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="508.24" y="196.93" transform="rotate(-0 508.24 196.93)">5</text></g></g><g><g><g><use xlink:href="#fb10000" x="534.89" y="181.58" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="534.89" y="196.93" transform="rotate(-0 534.89 196.93)">6</text></g></g><g><g style="fill: currentColor" transform="translate(451.42 211.93)"><text><tspan x="0" y="-0.84" style="font-style: oblique; font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">n</tspan></text></g></g></g><g/><g><path d="M 375 101.19  L 375 120.1  " clip-path="url(#fb10007)" style="fill: none; stroke: var(--hi); stroke-width: 2"/><path d="M 401.65 101.19  L 401.65 110.65  " clip-path="url(#fb10007)" style="fill: none; stroke: var(--hi); stroke-width: 2"/><path d="M 428.3 101.19  L 428.3 53.9  " clip-path="url(#fb10007)" style="fill: none; stroke: var(--hi); stroke-width: 2"/><path d="M 454.94 101.19  L 454.94 91.73  " clip-path="url(#fb10007)" style="fill: none; stroke: var(--hi); stroke-width: 2"/><path d="M 481.59 101.19  L 481.59 167.39  " clip-path="url(#fb10007)" style="fill: none; stroke: var(--hi); stroke-width: 2"/><path d="M 508.24 101.19  L 508.24 101.19  " clip-path="url(#fb10007)" style="fill: none; stroke: var(--hi); stroke-width: 2"/><path d="M 534.89 101.19  L 534.89 63.36  " clip-path="url(#fb10007)" style="fill: none; stroke: var(--hi); stroke-width: 2"/></g><g><defs><path id="fb10004" d="M 0 3  C 0.8 3 1.56 2.68 2.12 2.12  C 2.68 1.56 3 0.8 3 0  C 3 -0.8 2.68 -1.56 2.12 -2.12  C 1.56 -2.68 0.8 -3 0 -3  C -0.8 -3 -1.56 -2.68 -2.12 -2.12  C -2.68 -1.56 -3 -0.8 -3 0  C -3 0.8 -2.68 1.56 -2.12 2.12  C -1.56 2.68 -0.8 3 0 3  z " style="stroke: var(--hi)"/></defs><g clip-path="url(#fb10007)"><use xlink:href="#fb10004" x="375" y="120.1" style="fill: var(--hi); stroke: var(--hi)"/><use xlink:href="#fb10004" x="401.65" y="110.65" style="fill: var(--hi); stroke: var(--hi)"/><use xlink:href="#fb10004" x="428.3" y="53.9" style="fill: var(--hi); stroke: var(--hi)"/><use xlink:href="#fb10004" x="454.94" y="91.73" style="fill: var(--hi); stroke: var(--hi)"/><use xlink:href="#fb10004" x="481.59" y="167.39" style="fill: var(--hi); stroke: var(--hi)"/><use xlink:href="#fb10004" x="508.24" y="101.19" style="fill: var(--hi); stroke: var(--hi)"/><use xlink:href="#fb10004" x="534.89" y="63.36" style="fill: var(--hi); stroke: var(--hi)"/></g></g><g/><g><path d="M 367.01 101.19  L 542.88 101.19  " clip-path="url(#fb10007)" style="fill: none; stroke: var(--muted); stroke-width: 0.8; stroke-linecap: square"/></g><g><path d="M 367.01 181.58  L 542.88 181.58  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="375" y="133.34" transform="rotate(-0 375 133.34)">-2</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="401.65" y="123.89" transform="rotate(-0 401.65 123.89)">-1</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="428.3" y="47.28" transform="rotate(-0 428.3 47.28)">5</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="454.94" y="85.11" transform="rotate(-0 454.94 85.11)">1</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="481.59" y="180.63" transform="rotate(-0 481.59 180.63)">-7</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="508.24" y="94.57" transform="rotate(-0 508.24 94.57)">0</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="534.89" y="56.74" transform="rotate(-0 534.89 56.74)">4</text></g><g><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="367.01" y="14.8" transform="rotate(-0 367.01 14.8)">cross-correlation (F.conv1d)</text></g></g></g><defs><clipPath id="fb10005"><rect x="7.2" y="20.8" width="120.91" height="160.78"/></clipPath><clipPath id="fb10006"><rect x="159.62" y="20.8" width="175.87" height="160.78"/></clipPath><clipPath id="fb10007"><rect x="367.01" y="20.8" width="175.87" height="160.78"/></clipPath></defs></svg> <figcaption><b>The Lecture 15 example.</b> <i>x</i> = [2, 1, −3, 0, 4], <i>h</i> = [1, 0, −1], full output of length 5 + 3 − 1 = 7. Convolution gives [2, 1, −5, −1, 7, 0, −4]; cross-correlation (what <code>F.conv1d</code> computes, no flip) gives the negative here, because flipping this antisymmetric <i>h</i> negates it.</figcaption></figure>

In PyTorch (Lecture 15). `F.conv1d` wants `(batch, channels, length)` for the input and `(out_channels, in_channels, K)` for the kernel; `padding=2` (that is, $K-1$) gives the full output:

```python
import torch
import torch.nn.functional as F

x = torch.tensor([2., 1., -3., 0., 4.])
h = torch.tensor([1., 0., -1.])  # difference filter: finds edges
y_conv = F.conv1d(x.view(1, 1, -1), h.flip(0).view(1, 1, -1), padding=2).flatten()
y_corr = F.conv1d(x.view(1, 1, -1), h.view(1, 1, -1), padding=2).flatten()
print('convolution: ', y_conv.tolist())
print('correlation:', y_corr.tolist())
```

```
convolution:  [2.0, 1.0, -5.0, -1.0, 7.0, 0.0, -4.0]
correlation: [-2.0, -1.0, 5.0, 1.0, -7.0, 0.0, 4.0]
```

The same sum with plain loops ("Your turn 1"):

```python
def my_convolve1d(x, h):
    y = [0.0] * (len(x) + len(h) - 1)  # full output length
    for n in range(len(y)):
        for k in range(len(x)):
            if 0 <= n - k < len(h):     # h index in range
                y[n] += x[k] * h[n - k]
    return y

my_convolve1d([2, 1, -3, 0, 4], [1, 0, -1])  # [2, 1, -5, -1, 7, 0, -4]
```

## Convolution in 2-D

In 2-D a small $K\times K$ kernel slides over the rows and columns. What PyTorch computes, for one input channel, is

$$
y[i, j] = \sum_{a=0}^{K-1}\sum_{b=0}^{K-1} x[i + a,\; j + b]\; w[a, b] \;+\; \textrm{bias},
$$

so with no padding and stride 1 an $H\times W$ image becomes an $(H-K+1)\times(W-K+1)$ **feature map** (also called an activation or response map).

![[l16-pytorch-conv-2d-example.gif|A 2-D convolution sliding over an image]]

**Small example.** $x = \begin{bmatrix}1&2&0\\0&1&3\\2&1&0\end{bmatrix}$, $w = \begin{bmatrix}1&0\\0&-1\end{bmatrix}$ (no bias). Each output is the top-left pixel of the window minus the bottom-right one:

$$
y = \begin{bmatrix} 1-1 & 2-3 \\ 0-1 & 1-0 \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ -1 & 1 \end{bmatrix}, \qquad 3\times3 \;\rightarrow\; (3-2+1)\times(3-2+1) = 2\times 2.
$$

Real layers have many input channels (RGB, or the previous layer's feature maps) and many kernels; channels, padding, stride and parameter counts are on [[5-deep-networks/convolutional-layers|convolutional layers]].

## Template matching: where is the pattern?

Lecture 15's toy problem: each $20\times20$ image contains either a **blob** (a disc of radius 2, label 1) or a **vertical bar** (7 pixels tall, 1 wide, label 0) at a random position, plus noise. Which one, when it can be anywhere?

A hand-designed classifier in three steps:

$$
\textrm{filter (convolution)} \;\rightarrow\; \textrm{global max pooling} \;\rightarrow\; \textrm{threshold}.
$$

The filter is the $3\times3$ box kernel with every weight $\tfrac19$ (the average brightness of each window); global max pooling asks "how strong is the best match **anywhere**?"

- A blob's peak is near $1$: a $3\times3$ window fits entirely inside the disc.
- A bar's peak is near $\tfrac13$: any $3\times3$ window covers at most 3 lit pixels of a 1-pixel-wide bar.

Measured mean peaks: blobs $0.98$, bars $0.35$, so a threshold of $0.5$ separates them. A $3\times3$ **receptive field** is enough because the feature that separates the classes (width) fits inside it; a $1\times1$ filter would not work, since both shapes contain bright pixels.

```python
box = torch.full((1, 1, 3, 3), 1/9)            # 3x3 average filter
responses = F.conv2d(X_shapes, box)            # (N,1,20,20) -> (N,1,18,18)
peaks = responses.amax(dim=(1, 2, 3))          # best match anywhere
predictions = (peaks > 0.5).float()            # blob = 1, bar = 0
```

## Learning the filter

Hand-designed filters (Sobel, Prewitt, HOG, …) ruled computer vision for decades. The key idea of a CNN: make the kernel weights **parameters** and let gradient descent find them. The lecture's model is one `nn.Conv2d` with a single $3\times 3$ filter (**9 parameters**, no bias), global max pooling, and an MSE loss between the peak and the label, trained one image at a time:

```python
model = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, bias=False),  # 9 weights
                      nn.AdaptiveMaxPool2d(1),                      # global max
                      nn.Flatten())
nn.init.normal_(model[0].weight, std=0.01)  # small random start
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(2):
    for i in torch.randperm(len(X)):
        optimizer.zero_grad()
        loss = (model(X[i:i+1]).squeeze() - y[i])**2  # MSE on peak
        loss.backward()
        optimizer.step()
```

Four random seeds, 160 training images:

| seed | test accuracy (40 images) | fresh test set (1000) | mean peak: blobs | bars |
|---|---|---|---|---|
| 0 | 100.0% | 100.0% | 1.10 | 0.41 |
| 1 | 100.0% | 100.0% | 0.94 | 0.13 |
| 2 | 100.0% | 100.0% | 0.84 | 0.35 |
| 3 | 37.5% | 54.3% | 1.01 | 0.53 |

The learned kernels differ from each other and from the box: gradient descent finds **any** weights whose peak response separates the classes, not a "template of a blob". Learned features are chosen for being useful, not for being interpretable. Seed 3 looks like a failure, but its filter separates the classes cleanly too: its bar responses sit right on the $0.5$ threshold, so every single-image update pushes them back and forth across it. The filter is fine; the threshold is not. (One reason real layers learn many filters at once, and a reminder to plot the quantities behind a decision.)

> [!note] Gradient through a max
> Only the winning position of a max receives gradient, so each update of the kernel is driven by one $3\times3$ patch. On ties `torch.amax` splits the gradient evenly while `nn.AdaptiveMaxPool2d` sends all of it to one position:
> ```python
> x = torch.tensor([[[[1., 3.], [3., 0.]]]], requires_grad=True)
> x.amax().backward()                      # grad [0, 0.5, 0.5, 0]
> x.grad = None
> nn.AdaptiveMaxPool2d(1)(x).sum().backward()  # grad [0, 1, 0, 0]
> ```

## One filter, applied everywhere

- **Locality**: each output looks at a small neighborhood, because nearby pixels belong together.
- **Weight sharing**: the *same* filter is used at every position, because a pattern useful in one place is useful everywhere. The number of weights does not depend on the image size.
- **Translation equivariance** follows: if the input shifts by some pixels, the feature map shifts by the same amount (away from the borders). Equivariance is not yet invariance: the class decision becomes shift-invariant only after [[5-deep-networks/pooling#the-shift-experiment|global pooling]].
- History: Hubel and Wiesel (1959–62) found neurons in the visual cortex that respond to oriented edges in small regions; Fukushima's Neocognitron (1980) and LeCun's LeNet (1998) turned this into networks ([[5-deep-networks/cnn-architectures|CNN architectures]]).

The first-layer filters of a pretrained ResNet-18 (64 filters of shape $3\times7\times7$, trained on 1.2 million ImageNet photos; [`torchvision.models.resnet18`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)) look like oriented edge and color detectors: learned, but close to what engineers used to design by hand.

## When the assumption is wrong: scrambled pixels

Back to the opening question of Lecture 15: train on images whose pixels were shuffled by one fixed permutation. FashionMNIST test accuracy:

| model | original | scrambled |
|---|---|---|
| MLP | 84.7% | 84.0% |
| CNN + flatten + FC | 88.8% | 85.3% |
| CNN + GAP + FC | 84.2% | 79.4% |
| CNN 5×5 + GAP + FC | 87.6% | 80.9% |

The MLP does not care. The flatten CNN loses its advantage and falls back to about MLP level, because its fully-connected layer can still learn position-specific weights. The global-average-pooling CNNs have no position-specific weights at all; they rely only on local patterns, which scrambling destroys, so they drop the most. And data efficiency (1,500 training steps, varying training-set size):

| test accuracy | 600 images | 2,000 | 6,000 | 20,000 |
|---|---|---|---|---|
| MLP | 79.0% | 82.7% | 85.0% | 86.5% |
| CNN | 81.1% | 85.9% | 89.0% | 89.5% |
| CNN on scrambled pixels | 75.8% | 80.8% | 84.4% | 84.8% |

The CNN trained on 6,000 images beats the MLP trained on 20,000: the right assumption **is worth data**. On scrambled pixels the same CNN is below the MLP at every size: the wrong assumption costs data. CNNs win wherever neighbors really belong together (images, audio spectrograms, time series), not on tables of unrelated features.

> [!trap] Common mistakes
> - **Expecting PyTorch to flip the kernel.** `F.conv1d`, `F.conv2d` and `nn.Conv2d` compute cross-correlation. Flip with `h.flip(0)` (1-D) or `w.flip(-1, -2)` (2-D) for true convolution.
> - **Forgetting the batch and channel dimensions.** A single 1-D signal must be `x.view(1, 1, -1)`; a single grayscale image `x.view(1, 1, H, W)`.
> - **Output length.** Full: $N + K - 1$. Valid (no padding): $N - K + 1$. "Same" (padding $(K-1)/2$, stride 1): $N$.
> - **Index direction.** Convolution pairs $x[k]$ with $h[n-k]$ (opposite directions); cross-correlation pairs $x[n+k]$ with $h[k]$ (same direction).
> - **Equivariance is not invariance.** Convolution moves features with the object; only global pooling discards *where*.

## PyTorch docs

- [`torch.nn.functional.conv1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html) and [`torch.nn.functional.conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html): the functional forms with explicit weight tensors, as used above.
- [`torch.nn.Conv1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) and [`torch.nn.Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): the layer forms, with learnable weights; the docs state the cross-correlation formula and the output-size formula.
- [Convolution layers in `torch.nn`](https://docs.pytorch.org/docs/stable/nn.html#convolution-layers): the whole family (transposed, 3-D, ...).
- [`torch.flip`](https://docs.pytorch.org/docs/stable/generated/torch.flip.html): flip a kernel for true convolution.
- [`torch.amax`](https://docs.pytorch.org/docs/stable/generated/torch.amax.html) and [`torch.nn.AdaptiveMaxPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool2d.html): global max pooling.
- [`torch.nn.init`](https://docs.pytorch.org/docs/stable/nn.init.html): `normal_` and other initializers.
- [`torchvision.models.resnet18`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html): the pretrained network whose first-layer filters the lecture shows.

Related: [[5-deep-networks/convolutional-layers]] · [[5-deep-networks/pooling]] · [[5-deep-networks/multilayer-perceptrons]] · [[5-deep-networks/cnn-architectures]] · [[2-calculus-and-optimization/gradient-descent]]
