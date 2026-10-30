---
title: "CNN architectures: LeNet to ResNet"
description: "The landmark image-classification CNNs and the one idea each contributed: LeNet-5 (1998), ImageNet and the ILSVRC challenge, AlexNet (2012: depth, ReLU, GPUs), VGG (2014: stacks of 3×3 kernels), GoogLeNet/Inception (2014: parallel branches and 1×1 bottlenecks) and ResNet (2015: residual connections), with layer-by-layer shapes and parameter counts worked by hand."
tags: [topic, deep-nets]
aliases: ["CNN history", "LeNet", "LeNet-5", "AlexNet", "VGG", "VGG-16", "VGG-19", "GoogLeNet", "ResNet-18", "ResNet-34", "ImageNet", "ILSVRC", "top-5 error"]
---

*Lectures 16–17 · [[5-deep-networks/index|Deep networks]] · builds on [[5-deep-networks/convolutional-layers|convolutional layers]] and [[5-deep-networks/pooling|pooling]] · details in [[5-deep-networks/branching-and-inception|branching and Inception]] and [[5-deep-networks/residual-connections|residual connections]]*

> [!abstract] In one breath
> Image classification drove the rise of CNNs. **LeNet-5** (1998) read handwritten digits with two conv layers. **ImageNet** (1,000 classes, 1.28 M training images) gave the field a hard benchmark, and **AlexNet** won its 2012 challenge with an 8-layer CNN trained on GPUs, cutting top-5 error from about 26% to 15.3%. **VGG** (2014) went deeper with nothing but $3\times3$ kernels, **GoogLeNet** (2014) ran kernels of several sizes in parallel behind cheap $1\times1$ bottlenecks, and **ResNet** (2015) made 100+ layers trainable by having each block learn a correction $F(x)$ added to its input $x$. Each step is one idea you can state in a sentence and check with a parameter count.

## What it is

**Image classification** assigns one of a fixed set of labels to an image: one of {cat, dog, horse, frog, giraffe, cow} for a 6-class problem, one of 1,000 for ImageNet. Every model on this page is a [[5-deep-networks/convolutional-layers|convolutional]] feature extractor followed by a classifier head, trained with [[3-linear-models/multiclass-logistic-regression|softmax and cross-entropy]]. What changed between 1998 and 2015 is depth, and the tricks that made depth affordable and trainable.

<!-- fig:deepc-imagenet-timeline -->
<figure class="cs-fig"><svg role="img" aria-label="Bar charts: ImageNet top-5 error dropping from 28.2% to 3.57%, and depth growing from 5 to 152 layers" style="max-width:945px;width:100%;height:auto" stroke-linejoin="round" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 708.91 249.84" xmlns="http://www.w3.org/2000/svg" version="1.1"><defs></defs><g><g><path d="M 0 249.84  L 708.91 249.84  L 708.91 0  L 0 0  L 0 249.84  z " style="fill: none"/></g><g><g><path d="M 43.2 205.03  L 389.97 205.03  L 389.97 21.56  L 43.2 21.56  L 43.2 205.03  z " style="fill: none"/></g><g><path d="M 58.96 205.03  L 93.74 205.03  L 93.74 43.35  L 58.96 43.35  z " clip-path="url(#f13ecd2)" style="fill: var(--muted)"/></g><g><path d="M 115.05 205.03  L 149.83 205.03  L 149.83 57.11  L 115.05 57.11  z " clip-path="url(#f13ecd2)" style="fill: var(--muted)"/></g><g><path d="M 171.15 205.03  L 205.93 205.03  L 205.93 117.31  L 171.15 117.31  z " clip-path="url(#f13ecd2)" style="fill: var(--accent)"/></g><g><path d="M 227.24 205.03  L 262.02 205.03  L 262.02 163.18  L 227.24 163.18  z " clip-path="url(#f13ecd2)" style="fill: var(--accent)"/></g><g><path d="M 283.34 205.03  L 318.11 205.03  L 318.11 166.62  L 283.34 166.62  z " clip-path="url(#f13ecd2)" style="fill: var(--accent)"/></g><g><path d="M 339.43 205.03  L 374.21 205.03  L 374.21 184.57  L 339.43 184.57  z " clip-path="url(#f13ecd2)" style="fill: var(--hi)"/></g><g><g><g><defs><path id="f13ecd0" d="M 0 0  L 0 3.5  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#f13ecd0" x="76.35" y="205.03" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(65.53 219.34)">2010</text><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(55.54 229.55)">NEC-UIUC</text><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(57.06 239.75)">(no CNN)</text></g></g><g><g><g><use xlink:href="#f13ecd0" x="132.44" y="205.03" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(121.63 219.34)">2011</text><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(121.14 229.55)">XRCE</text><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(113.16 239.75)">(no CNN)</text></g></g><g><g><g><use xlink:href="#f13ecd0" x="188.54" y="205.03" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(177.72 219.34)">2012</text><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(171.93 229.55)">AlexNet</text></g></g><g><g><g><use xlink:href="#f13ecd0" x="244.63" y="205.03" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(233.81 219.34)">2014</text><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(235.14 229.55)">VGG</text><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(220.43 239.75)">(2nd place)</text></g></g><g><g><g><use xlink:href="#f13ecd0" x="300.72" y="205.03" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(289.91 219.34)">2014</text><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(277.16 229.55)">GoogLeNet</text></g></g><g><g><g><use xlink:href="#f13ecd0" x="356.82" y="205.03" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(346 219.34)">2015</text><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(341.77 229.55)">ResNet</text></g></g></g><g><g><g><defs><path id="f13ecd1" d="M 0 0  L -3.5 0  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#f13ecd1" x="43.2" y="205.03" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="36.2" y="209.21" transform="rotate(-0 36.2 209.21)">0</text></g></g><g><g><g><use xlink:href="#f13ecd1" x="43.2" y="176.37" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="36.2" y="180.55" transform="rotate(-0 36.2 180.55)">5</text></g></g><g><g><g><use xlink:href="#f13ecd1" x="43.2" y="147.7" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="36.2" y="151.88" transform="rotate(-0 36.2 151.88)">10</text></g></g><g><g><g><use xlink:href="#f13ecd1" x="43.2" y="119.03" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="36.2" y="123.21" transform="rotate(-0 36.2 123.21)">15</text></g></g><g><g><g><use xlink:href="#f13ecd1" x="43.2" y="90.36" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="36.2" y="94.54" transform="rotate(-0 36.2 94.54)">20</text></g></g><g><g><g><use xlink:href="#f13ecd1" x="43.2" y="61.69" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="36.2" y="65.87" transform="rotate(-0 36.2 65.87)">25</text></g></g><g><g><g><use xlink:href="#f13ecd1" x="43.2" y="33.03" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="36.2" y="37.2" transform="rotate(-0 36.2 37.2)">30</text></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="15.56" y="113.3" transform="rotate(-90 15.56 113.3)">ILSVRC top-5 error (%)</text></g></g><g><path d="M 43.2 175.79  L 389.97 175.79  " clip-path="url(#f13ecd2)" style="fill: none; stroke-dasharray: 3.7,1.6; stroke-dashoffset: 0; stroke: currentColor"/></g><g><path d="M 43.2 205.03  L 43.2 21.56  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><path d="M 43.2 205.03  L 389.97 205.03  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="76.35" y="36.93" transform="rotate(-0 76.35 36.93)">28.2%</text></g><g><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="132.44" y="50.69" transform="rotate(-0 132.44 50.69)">25.8%</text></g><g><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="188.54" y="110.89" transform="rotate(-0 188.54 110.89)">15.3%</text></g><g><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="244.63" y="156.76" transform="rotate(-0 244.63 156.76)">7.3%</text></g><g><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="300.72" y="160.2" transform="rotate(-0 300.72 160.2)">6.7%</text></g><g><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="356.82" y="178.15" transform="rotate(-0 356.82 178.15)">3.57%</text></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="51.11" y="170.76" transform="rotate(-0 51.11 170.76)">human ≈ 5.1%</text></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="43.2" y="15.56" transform="rotate(-0 43.2 15.56)">Error fell once CNNs arrived</text></g></g><g><g><path d="M 444.84 205.03  L 701.71 205.03  L 701.71 21.56  L 444.84 21.56  L 444.84 205.03  z " style="fill: none"/></g><g><path d="M 456.52 86590.49  L 487.86 86590.49  L 487.86 185.88  L 456.52 185.88  z " clip-path="url(#f13ecd3)" style="fill: var(--muted)"/></g><g><path d="M 507.06 86590.49  L 538.4 86590.49  L 538.4 168.25  L 507.06 168.25  z " clip-path="url(#f13ecd3)" style="fill: var(--accent)"/></g><g><path d="M 557.61 86590.49  L 588.95 86590.49  L 588.95 135.82  L 557.61 135.82  z " clip-path="url(#f13ecd3)" style="fill: var(--accent)"/></g><g><path d="M 608.15 86590.49  L 639.49 86590.49  L 639.49 130.32  L 608.15 130.32  z " clip-path="url(#f13ecd3)" style="fill: var(--accent)"/></g><g><path d="M 658.7 86590.49  L 690.04 86590.49  L 690.04 57.84  L 658.7 57.84  z " clip-path="url(#f13ecd3)" style="fill: var(--hi)"/></g><g><g><g><g><use xlink:href="#f13ecd0" x="472.19" y="205.03" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(455.58 219.34)">LeNet-5</text><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(461.37 229.55)">1998</text></g></g><g><g><g><use xlink:href="#f13ecd0" x="522.73" y="205.03" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(506.13 219.34)">AlexNet</text><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(511.92 229.55)">2012</text></g></g><g><g><g><use xlink:href="#f13ecd0" x="573.28" y="205.03" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(556.84 219.34)">VGG-19</text><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(562.46 229.55)">2014</text></g></g><g><g><g><use xlink:href="#f13ecd0" x="623.82" y="205.03" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(600.26 219.34)">GoogLeNet</text><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(613.01 229.55)">2014</text></g></g><g><g><g><use xlink:href="#f13ecd0" x="674.37" y="205.03" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(649.67 219.34)">ResNet-152</text><text style="font-size: 8.5px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; fill: currentColor" transform="translate(663.55 229.55)">2015</text></g></g></g><g><g><g><g><use xlink:href="#f13ecd1" x="444.84" y="185.88" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="437.84" y="190.06" transform="rotate(-0 437.84 190.06)">5</text></g></g><g><g><g><use xlink:href="#f13ecd1" x="444.84" y="159.89" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="437.84" y="164.07" transform="rotate(-0 437.84 164.07)">10</text></g></g><g><g><g><use xlink:href="#f13ecd1" x="444.84" y="133.89" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="437.84" y="138.07" transform="rotate(-0 437.84 138.07)">20</text></g></g><g><g><g><use xlink:href="#f13ecd1" x="444.84" y="99.53" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="437.84" y="103.71" transform="rotate(-0 437.84 103.71)">50</text></g></g><g><g><g><use xlink:href="#f13ecd1" x="444.84" y="73.54" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="437.84" y="77.72" transform="rotate(-0 437.84 77.72)">100</text></g></g><g><g><g><use xlink:href="#f13ecd1" x="444.84" y="47.55" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="437.84" y="51.73" transform="rotate(-0 437.84 51.73)">200</text></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="410.2" y="113.3" transform="rotate(-90 410.2 113.3)">weight layers (log scale)</text></g></g><g><path d="M 444.84 205.03  L 444.84 21.56  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><path d="M 444.84 205.03  L 701.71 205.03  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="472.19" y="179.23" transform="rotate(-0 472.19 179.23)">5</text></g><g><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="522.73" y="161.6" transform="rotate(-0 522.73 161.6)">8</text></g><g><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="573.28" y="129.17" transform="rotate(-0 573.28 129.17)">19</text></g><g><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="623.82" y="123.67" transform="rotate(-0 623.82 123.67)">22</text></g><g><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="674.37" y="51.19" transform="rotate(-0 674.37 51.19)">152</text></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="444.84" y="15.56" transform="rotate(-0 444.84 15.56)">…while the networks got deeper</text></g></g></g><defs><clipPath id="f13ecd2"><rect x="43.2" y="21.56" width="346.77" height="183.48"/></clipPath><clipPath id="f13ecd3"><rect x="444.84" y="21.56" width="256.87" height="183.48"/></clipPath></defs></svg> <figcaption><b>ImageNet (ILSVRC) classification, 2010–2015.</b> Left: top-5 error of the winning entry in each year shown (2013 is left out; VGG placed 2nd in 2014; challenge entries were often ensembles of several networks). The 15.3% for AlexNet is from the lecture; the other values are the published challenge results (Russakovsky et al., 2015) and the original papers; the human estimate is the one trained annotator in Russakovsky et al. Right: depth counted in weight layers (conv + fully connected); ResNet-152 is 7× deeper than GoogLeNet.</figcaption></figure>

**Top-5 error** is the fraction of test images whose correct label is *not* among the model's 5 highest-scoring classes; it is the headline metric of the ImageNet challenge because many images contain several plausible objects.

## What you should know

- The timeline and the **one idea per model**: LeNet (conv + pool + FC for digits), AlexNet (deep CNN + ReLU + GPUs on ImageNet), VGG (only $3\times3$ kernels), GoogLeNet (Inception: parallel branches, $1\times1$ bottlenecks), ResNet (skip connections).
- ImageNet-1k: 1,000 classes, 1,281,167 training images; AlexNet's 2012 top-5 error 15.3%, more than 10.8 points ahead of the runner-up.
- How to trace shapes through a classic network with the [[5-deep-networks/convolutional-layers|output-size formula]] $\lfloor (n + 2p - k)/s\rfloor + 1$, and count its parameters ($C_{\text{out}}C_{\text{in}}k^2 + C_{\text{out}}$ per conv).
- VGG's argument: three $3\times3$ layers have a $7\times7$ receptive field with $27C^2$ instead of $49C^2$ weights and two extra nonlinearities.
- Where the parameters of VGG/AlexNet live: mostly in the **fully connected** layers.
- The **degradation problem**: a deeper plain network can have higher *training* error than a shallower one, so it is an optimization failure, not overfitting. ResNet's fix: $\mathcal{H}(x) = \mathcal{F}(x) + x$.
- Loading a pretrained model from [`torchvision.models`](https://docs.pytorch.org/vision/stable/models.html).

## LeNet-5 (1998)

The first major practical success of CNNs: Yann LeCun and colleagues at Bell Labs built LeNet-5 for handwritten digits (MNIST), and it was used to read zip codes on mail and amounts on checks.

![[l17-lenet-5.png|LeNet-5: two 5×5 conv layers with 2×2 average pooling, then three fully connected layers]]

Tracing a $1\times28\times28$ digit through the figure's version:

| Layer | Output shape | Parameters |
| --- | --- | --- |
| $5\times5$ conv, 6 channels, pad 2 | $6\times28\times28$ | $6\cdot1\cdot25 + 6 = 156$ |
| $2\times2$ avg pool, stride 2 | $6\times14\times14$ | 0 |
| $5\times5$ conv, 16 channels | $16\times10\times10$ | $16\cdot6\cdot25 + 16 = 2{,}416$ |
| $2\times2$ avg pool, stride 2 | $16\times5\times5$ | 0 |
| flatten | 400 | 0 |
| FC 120 | 120 | $400\cdot120 + 120 = 48{,}120$ |
| FC 84 | 84 | $120\cdot84 + 84 = 10{,}164$ |
| FC 10 | 10 | $84\cdot10 + 10 = 850$ |
| **total** | | **61,706** |

```python
lenet = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Tanh(),  # 6x28x28
    nn.AvgPool2d(kernel_size=2, stride=2),                  # 6x14x14
    nn.Conv2d(6, 16, kernel_size=5), nn.Tanh(),            # 16x10x10
    nn.AvgPool2d(kernel_size=2, stride=2),                  # 16x5x5
    nn.Flatten(),                                           # 400 features
    nn.Linear(400, 120), nn.Tanh(),
    nn.Linear(120, 84), nn.Tanh(),
    nn.Linear(84, 10))                                      # class logits
```

LeNet used saturating activations (tanh/sigmoid style) and average pooling; ReLU and max pooling came later. Already the conv layers are cheap (2,572 parameters) and the FC layers are most of the model (59,134), a pattern that grows with every model until global pooling replaces the big FC head ([[5-deep-networks/pooling|pooling]]).

## ImageNet and the ILSVRC (2009–)

The [ImageNet dataset](https://en.wikipedia.org/wiki/ImageNet) (Fei-Fei Li and colleagues, introduced in 2009) is credited with accelerating image classification and CNNs. Its best-known use is the **ImageNet Large Scale Visual Recognition Challenge** (ILSVRC); the version usually meant by "ImageNet" is **ImageNet-1k**: 1,000 classes and 1,281,167 training images. ([Sample images per class](https://github.com/EliSchwartz/imagenet-sample-images).) The pretrained ResNet-18 in [[1-foundations/what-is-machine-learning|Lecture 1]] outputs 1,000 scores, one per ImageNet class, for exactly this reason.

## AlexNet (2012)

Until 2012, CNNs were not popular in computer vision despite LeNet: compute and hardware were limiting, and classical models such as SVMs dominated. **AlexNet** (Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton) won ILSVRC 2012 with a top-5 error of **15.3%**, more than **10.8 points** better than the runner-up, and turned the field towards deep learning.

Key contributions:
1. **Deeper and larger**: 8 weight layers (5 conv + 3 FC), about 61 M parameters.
2. **ReLU** instead of saturating activations, so gradients do not vanish in the positive range and training is faster ([[5-deep-networks/activation-functions|activation functions]]).
3. **GPUs** to make training such a model on such a dataset feasible ([[4-pytorch-training/gpu-training|GPU training]]).
4. [[5-deep-networks/dropout|Dropout]] in the large fully connected layers (from the AlexNet paper) and [[5-deep-networks/data-augmentation|data augmentation]] against overfitting.

![[l17-alexnet.png|LeNet (left) and AlexNet (right) side by side]]

Tracing the figure's version on a $3\times224\times224$ image (output size $\lfloor(n + 2p - k)/s\rfloor + 1$):

| Layer | Output | Size computation |
| --- | --- | --- |
| $11\times11$ conv, 96, stride 4 | $96\times54\times54$ | $\lfloor(224 - 11)/4\rfloor + 1 = 54$ |
| $3\times3$ max pool, stride 2 | $96\times26\times26$ | $\lfloor(54 - 3)/2\rfloor + 1 = 26$ |
| $5\times5$ conv, 256, pad 2 | $256\times26\times26$ | size kept |
| $3\times3$ max pool, stride 2 | $256\times12\times12$ | $\lfloor(26-3)/2\rfloor+1 = 12$ |
| 3 × ($3\times3$ conv, pad 1): 384, 384, 256 | $256\times12\times12$ | size kept |
| $3\times3$ max pool, stride 2 | $256\times5\times5$ | $\lfloor(12-3)/2\rfloor+1 = 5$ → 6,400 features |
| FC 4096, FC 4096, FC 1000 | 1000 | |

The five conv layers hold 3.75 M parameters; the three FC layers hold 47.1 M, **93%** of this version's 50.8 M. (The original paper's arithmetic works out for $227\times227$ inputs (the paper itself says 224), giving $55\times55$ after the first layer and $256\times6\times6$ before the FC layers, hence its roughly 61 M parameters.)

## VGG (2014): just use small kernels

After AlexNet, the question was how to go deeper without the cost exploding. Karen Simonyan and Andrew Zisserman (Visual Geometry Group, Oxford) answered with **VGG-16 and VGG-19**, which use **only $3\times3$ convolutions**.

A small kernel alone has a small **receptive field**, but stacking restores it. With $C$ input and output channels everywhere:

| | Receptive field | Weights |
| --- | --- | --- |
| one $7\times7$ layer | $7\times7$ | $7\cdot7\cdot C\cdot C = 49C^2$ |
| three $3\times3$ layers (ReLU between) | $3\to5\to7$ | $3\cdot(3\cdot3\cdot C\cdot C) = 27C^2$ |

The stack is $27/49 \approx 55\%$ of the cost, and adds two more nonlinearities. Lecture 15 checks this with $C = 64$ (biases included): one $7\times7$ layer has 200,768 parameters, three $3\times3$ layers 110,784, and the receptive field of the stack grows $3\times3$, $5\times5$, $7\times7$ ([[5-deep-networks/convolutional-layers|receptive field]]).

![[l17-vgg.png|VGG configurations A–E from the original paper, and their parameter counts in millions]]

The table from the [VGG paper](https://arxiv.org/abs/1409.1556) shows configurations A–E: blocks of $3\times3$ convs (64, 128, 256, 512, 512 channels) separated by max pooling, then FC-4096, FC-4096, FC-1000. Column D is VGG-16 (138 M parameters), E is VGG-19 (144 M).

> [!key] Where VGG-16's 138 M parameters are
> The 13 conv layers hold 14.7 M. The first FC layer alone maps $512\cdot7\cdot7 = 25{,}088$ features to 4,096: $25{,}088\cdot4{,}096 + 4{,}096 = 102{,}764{,}544$ parameters, **74%** of the model; the three FC layers together are 123.6 M (89%). Total $14{,}714{,}688 + 123{,}642{,}856 = 138{,}357{,}544$. This is why later models replace the flattened FC head with [[5-deep-networks/pooling|global average pooling]].

## GoogLeNet / Inception (2014)

GoogLeNet (Szegedy et al., winner of ILSVRC 2014 with 6.7% top-5 error, 22 layers) is built from **Inception blocks**: several branches look at the same input with different kernel sizes ($1\times1$, $3\times3$, $5\times5$, and a pooling branch), and their outputs are concatenated along the channel dimension. **$1\times1$ convolutions** before the expensive kernels reduce the number of channels first (the "GoogLeNet trick").

Lecture 16's block on a 64-channel input, four branches of 32 channels each: 43,200 parameters for a 128-channel output, versus 204,928 for a single $5\times5$ conv from 64 to 128 channels. Building and checking the block is on [[5-deep-networks/branching-and-inception|branching and Inception]].

## ResNet (2015): learn the correction

As networks got deeper, a surprising **degradation** appeared: beyond some depth, accuracy saturated and then got *worse*.

![[l17-deeper-error.png|From the ResNet paper: on CIFAR-10, a 56-layer plain network has higher training and test error than a 20-layer one]]

The 56-layer network is worse on the **training** set too, so this is not overfitting. It is also counter-intuitive: an $(N+L)$-layer model could copy an $N$-layer model by making the extra $L$ layers the identity. He et al. argued that plain layers find it hard to learn that identity mapping, and proposed the **residual unit**
$$
\mathcal{H}(x) = \mathcal{F}(x) + x,
$$
where a **skip connection** adds the block's input to the output of its weight layers. The identity is now $\mathcal{F}(x) = 0$, easy to reach, and the gradient always has a direct path back through the "+".

![[l17-resnet-34.png|VGG-19, a 34-layer plain network, and ResNet-34 from the original paper]]

From left to right: VGG-19, a 34-layer plain network, and the same network with shortcuts every two layers (**ResNet-34**). Dotted shortcuts are where the block halves the size and doubles the channels, so $x$ must be resized to match $\mathcal{F}(x)$: the paper either zero-pads the extra channels or projects $x$ with a $1\times1$ convolution of stride 2 (torchvision and Lecture 16 use the projection). ResNets end with global average pooling and a single FC-1000 layer, so ResNet-34 has about 21.8 M parameters despite its depth, a sixth of VGG-16. Residual blocks made ResNet-50, -101 and -152 trainable, won ILSVRC 2015 (3.57% top-5 with an ensemble of ResNets), and are now inside most large deep models.

In Lecture 16, 21-layer networks on FashionMNIST made the point directly: after 2 epochs the plain network stayed at **10.1%** test accuracy (chance; its early layers received almost no gradient), while the identical network with "+ x" reached **56.9%**. Code, projection shortcuts and batch norm in [[5-deep-networks/residual-connections|residual connections]].

## Summary table

| Model | Year | Weight layers | Parameters | ImageNet top-5 | Idea |
| --- | --- | --- | --- | --- | --- |
| LeNet-5 | 1998 | 5 | ≈ 62 K | (MNIST digits) | conv + pool + FC |
| AlexNet | 2012 | 8 | ≈ 61 M | 15.3% | deep CNN, ReLU, GPUs, dropout |
| VGG-16 / 19 | 2014 | 16 / 19 | 138 M / 144 M | 7.3% (VGG team, 2nd) | only $3\times3$ kernels |
| GoogLeNet | 2014 | 22 | ≈ 6.6 M | 6.7% | Inception blocks, $1\times1$ bottlenecks |
| ResNet-34 / 152 | 2015 | 34 / 152 | ≈ 21.8 M / 60.2 M | 3.57% (ResNet ensemble) | residual connections |

Parameter counts other than LeNet's and VGG's (worked above) are those of the [torchvision implementations](https://docs.pytorch.org/vision/stable/models.html); error rates as in the figure at the top.

## Using them in PyTorch

`torchvision.models` has all of these with ImageNet-pretrained weights. Lecture 1 classifies photos with a pretrained ResNet-18; Lecture 15 looks at its first-layer filters:

```python
import torchvision
from torchvision.models import ResNet18_Weights

model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.eval()                                  # inference mode
filters = model.conv1.weight.detach()         # first-layer filters
print(filters.shape)
```

```
torch.Size([64, 3, 7, 7])
```

64 filters, each $3\times7\times7$ (RGB): the "7x7 conv, 64, /2" at the top of ResNet-34 in the figure above. Without `weights=...` the same call builds the architecture with **random** weights, which is why the Lecture 1 demo first gives nonsense predictions.

> [!trap] Common mistakes
> - **Counting pooling or activation layers as depth.** "VGG-16" and "ResNet-34" count weight layers (conv + FC) only.
> - **Calling degradation overfitting.** Overfitting means low training error and high test error. The deep plain network has *higher training error*: it fails to optimize.
> - **Thinking small kernels mean a small receptive field.** Each stacked $3\times3$ layer adds 2 to the receptive field; depth buys back what the kernel size gives up.
> - **Mixing top-1 and top-5 error.** AlexNet's 15.3% is top-5.
> - **Assuming more layers means more parameters.** ResNet-34 (21.8 M) has far fewer than VGG-16 (138 M): the FC head dominates.

## Check yourself

> [!question] 1. Receptive field and cost
> How many stacked $3\times3$ conv layers (stride 1) give the receptive field of one $11\times11$ layer, and what fraction of the weights do they need (equal channels $C$)?

> [!solution]- Answer
> Each layer adds 2: $3, 5, 7, 9, 11$, so **5 layers**. Weights $5\cdot9C^2 = 45C^2$ versus $121C^2$, about 37%.

> [!question] 2. LeNet without padding
> If the first LeNet conv had no padding, what would the shapes be after each layer up to the flatten, and how big is the first FC layer's input?

> [!solution]- Answer
> $28 - 5 + 1 = 24$: $6\times24\times24$ → pool $6\times12\times12$ → conv $16\times8\times8$ → pool $16\times4\times4$ → flatten **256** features (instead of 400).

> [!question] 3. Why the residual form?
> Give the two reasons, from the lectures, why $\mathcal{H}(x) = \mathcal{F}(x) + x$ trains better than learning $\mathcal{H}$ directly.

> [!solution]- Answer
> (1) The identity, which the deeper model needs to be at least as good as the shallow one, is just $\mathcal{F} = 0$, which is easy to reach. (2) The "+ x" gives the gradient a direct path back to every earlier layer, so it does not vanish through the stack (the Lecture 16 plain-vs-residual experiment).

## PyTorch docs

- [`torchvision.models`](https://docs.pytorch.org/vision/stable/models.html) — every architecture on this page, pretrained weights, and the table of parameter counts and accuracies.
- [`torchvision.models.resnet18`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html), [`alexnet`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html), [`vgg16`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html), [`googlenet`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.googlenet.html) — builders and their `weights=` enums.
- [`torch.nn.Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — the output-size formula used in the shape traces.
- [`torch.nn.MaxPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html), [`torch.nn.AvgPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html), [`torch.nn.AdaptiveAvgPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html) — LeNet/AlexNet/VGG pooling and ResNet's global average pool.
- [`torch.nn.Flatten`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.flatten.Flatten.html) and [`torch.nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html) — the LeNet code.
- Papers: [VGG (Simonyan & Zisserman)](https://arxiv.org/abs/1409.1556), [GoogLeNet (Szegedy et al.)](https://arxiv.org/abs/1409.4842), [ResNet (He et al.)](https://arxiv.org/abs/1512.03385), [ILSVRC (Russakovsky et al.)](https://arxiv.org/abs/1409.0575); background on [LeNet](https://en.wikipedia.org/wiki/LeNet) and [AlexNet](https://en.wikipedia.org/wiki/AlexNet).

Related: [[5-deep-networks/residual-connections]] · [[5-deep-networks/branching-and-inception]] · [[5-deep-networks/convolutional-layers]] · [[5-deep-networks/pooling]] · [[5-deep-networks/dropout]] · [[5-deep-networks/batch-normalization]]
