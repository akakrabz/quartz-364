---
title: "Review: training behaviour and activations"
description: "Midterm 2 problem type 5 (10–25 points of short explanations on every past exam): why networks need activation functions, the activation facts asked in true/false items, small ReLU networks designed by hand, whether ReLU commutes with pooling, reading underfitting and overfitting off a loss plot, gradient descent versus SGD and epochs versus iterations, momentum, and what dropout, batch norm and augmentation do in training and in evaluation."
tags: [exam, midterm-2, training]
aliases: ["training behaviour review", "activations review", "explain-why review"]
---

*Midterm 2 · problem type 5 of 6 · HW4 Q1, Q3–Q4, HW5 Q3 · Lectures 10–11, 14, 16 · explain-why questions on all four past exams · topic pages: [[5-deep-networks/activation-functions|activation functions]], [[4-pytorch-training/optimizers|optimizers]], [[5-deep-networks/dropout|dropout]], [[5-deep-networks/batch-normalization|batch norm]], [[5-deep-networks/data-augmentation|augmentation]] · back to [[9-midterm-2/index|Midterm 2]]*

> [!abstract] In one breath
> These are the questions answered in sentences. They reward one habit: **name the mechanism, then its consequence**. "Linear maps compose to a linear map, so without activations depth adds nothing." "SGD's gradient comes from a few examples, so it is a noisy estimate and the path jitters." "Both losses are flat, high and together, so the model underfits; make it larger." Know the dozen mechanisms below and each answer is two lines.

## Why activations

The learnable operations (matrix products, convolutions) are linear, and linear maps collapse:

$$
W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2).
$$

A stack of linear layers is **one** linear layer ([[9-midterm-2/spring-2025#problem-1--fact-or-cap-30-pts|Sp25 1(o)]], True). A non-linear activation between the layers stops the collapse, so each layer can build on the features of the one before: that is what makes depth worth having ([[9-midterm-2/fall-2024#problem-5--activations-15-pts|F24 5(a)]], the model answer).

| Activation | Range | Derivative | What exams asked |
|---|---|---|---|
| sigmoid | $(0, 1)$ | $\sigma(1-\sigma)\le\tfrac14$ | can cause vanishing gradients in deep networks (Sp25 1(e), True) |
| tanh | $(-1, 1)$ | $1-\tanh^2\le1$ | the RNN's default non-linearity |
| ReLU | $[0, \infty)$ | 1 for $x>0$, 0 for $x<0$ | "always 1" is False (Sp25 1(f)) |

Pooling is not where the non-linearity has to come from: average pooling is linear, max pooling is not ([[9-midterm-2/fall-2024#problem-1--true-or-false-30-pts|F24 1(c)]]).

## ReLU networks by hand

Both Spring exams ask for a network $f(x) = w_1^\top\mathrm{ReLU}(w_0x + b_0)$ with **two hidden units** that equals a given function. A hidden unit $\mathrm{ReLU}(ax + c)$ is a "hinge": zero on one side of $x = -c/a$, a straight line on the other. Two hinges and two output weights give these building blocks:

| Target | Identity | $w_0$ | $b_0$ | $w_1$ |
|---|---|---|---|---|
| $mx + b$ ([[9-midterm-2/spring-2025#problem-5--layer-by-layer-a-neural-network-love-story-10-pts\|Sp25 5]]) | $\mathrm{ReLU}(u)-\mathrm{ReLU}(-u)=u$ with $u=mx+b$ | $(m, -m)$ | $(b, -b)$ | $(1, -1)$ |
| $\lvert x\rvert$ ([[9-midterm-2/spring-2026#problem-5--layer-by-layer-the-kink-awakens-10-pts\|Sp26 5]]) | $\mathrm{ReLU}(x)+\mathrm{ReLU}(-x)=\lvert x\rvert$ | $(1, -1)$ | $(0, 0)$ | $(1, 1)$ |
| $\max(x, 0)$ clipped at 1 | $\mathrm{ReLU}(x)-\mathrm{ReLU}(x-1)$ | $(1, 1)$ | $(0, -1)$ | $(1, -1)$ |
| $\max(x, c)$ for $c\ge0$ | $\mathrm{ReLU}(x-c)+\mathrm{ReLU}(0\cdot x+c)$: a unit with input weight 0 is a constant | $(1, 0)$ | $(-c, c)$ | $(1, 1)$ |

**How to find one:** sketch the target, mark its kinks (each kink is where one hidden unit switches on, at $x = -c/a$), then choose output weights so the slopes add up correctly on each piece. **How to show it is correct:** check by cases, one case per piece ($x > 0$, $x < 0$, and so on).

## Does ReLU commute with pooling?

[[9-midterm-2/fall-2024#problem-5--activations-15-pts|F24 5(b)]]: is $\mathrm{ReLU}(g(x)) = g(\mathrm{ReLU}(x))$ for mean pooling $g$? **No**: one counterexample settles it. With $x = (-2, 3, -1, 0, -3, 6)$ and blocks of three, the left side is $(0, 1)$ and the right side $(1, 2)$. Averaging first lets negatives cancel positives; clipping first removes them. (With max pooling the two sides **are** equal, because ReLU never changes which entry is largest.) The method generalizes: to show two operations do not commute, try an input with mixed signs.

## Reading a loss plot

| | Training loss / metric | Validation | Verdict | First remedy |
|---|---|---|---|---|
| both flat, **high**, together | poor | about the same | **underfitting** | larger model, non-linear activations, train longer if still falling |
| training keeps improving, validation **turns worse** | near perfect | peaks, then degrades | **overfitting** | regularize (dropout, weight decay, augmentation), stop earlier, more data |
| training strong, validation lower but **flat** | strong | plateaus | proper fit | none needed |

[[9-midterm-2/spring-2025#problem-2--it-trained-all-night-and-still-flopped-10-pts|Sp25 2]] is the first row. A gap alone is not overfitting; look at the direction of the validation curve. Lecture 22 is not on the exam, but its page says all of this at length: [[4-pytorch-training/underfitting-and-overfitting|underfitting and overfitting]].

## Gradient descent, SGD, epochs

| Term | Meaning |
|---|---|
| gradient descent (GD) | each step uses the gradient over the **whole** dataset: smooth path |
| stochastic gradient descent (SGD) | each step uses one example or a mini-batch: a **noisy estimate**, jagged path ([[9-midterm-2/spring-2026#problem-2--race-to-the-bottom-10-pts\|Sp26 2]]) |
| iteration | one batch, one `optimizer.step()` |
| epoch | one full pass through the training data: $\lceil N/B\rceil$ iterations ([[9-midterm-2/fall-2024#problem-1--true-or-false-30-pts\|F24 1(h)]]) |
| momentum | the step uses $v_t = \mu v_{t-1} + g_t$, a weighted combination of the current and earlier gradients ([[9-midterm-2/fall-2025#problem-1--true-or-false-30-pts\|F25 1(c)]]) |

SGD with a batch equal to the dataset is GD. Smaller batches: cheaper steps, more of them per epoch, noisier. → [[4-pytorch-training/optimizers#stochastic-mini-batch-gradient-descent|mini-batch SGD]]

## Dropout, batch norm, augmentation

| | During training | In `model.eval()` | Purpose | Asked |
|---|---|---|---|---|
| dropout | zeroes each activation with probability $p$, rescales the rest | does nothing | against overfitting | F24 1(f) True; Sp26 1(f) |
| batch norm | normalizes each channel with the **batch's** mean and variance, then scale and shift; updates running statistics | still normalizes, with the **running** statistics | stable, faster training | F24 1(g) False; Sp26 1(f) False, 1(j) |
| augmentation | random label-preserving changes to the inputs | not applied | more varied training data, against overfitting | F25 2(b) |

- Batch norm acts on **feature maps inside the network**, not on the colour channels of the input image (that is preprocessing).
- "Both are turned off in eval mode" is False: only dropout is.
- **The augmentation test: does the label survive?** Flipping is fine for fruit, not for letters (p/q). Colour-channel shuffling is fine for letters, not for fruit told apart by colour. Resizing with the aspect ratio kept is fine for both ([[9-midterm-2/fall-2025#problem-2--layer-shapes-and-augmentations-17-pts|F25 2(b)]]).

> [!tip] Shortcut — a two-line answer
> Line 1: the mechanism, in the problem's own terms. Line 2: the consequence that answers the question. Example for F25 5(d): "A convolution's first-layer output depends only on a $K\times K$ window. The two corners are never in the same window, so a CNN cannot relate them in its first layer."

> [!trap] Common mistakes
> - **"Activations add parameters / make the model bigger."** They add none; they add non-linearity.
> - **Calling any gap overfitting**, or calling a high flat loss "converged, no issue".
> - **"SGD is wrong because it is noisy."** It follows the true gradient on average and is far cheaper per step.
> - **Proving a general claim with one example** (fine for "False", not for "True").
> - **Naming an augmentation without checking the labels** of the dataset in question.

> [!question]- Check yourself
> 1. Give $w_0, b_0, w_1$ so that $w_1^\top\mathrm{ReLU}(w_0x + b_0) = 2x - 6$ for all $x$.
> 2. Training accuracy 62%, validation 61%, both flat for 100 epochs on a 4-class problem. Verdict and one remedy?
> 3. A dataset of 1,000 examples, batch size 100. How many gradient steps does GD take per epoch? SGD?
>
> > [!solution]- Answers
> > 1. $w_0 = (2, -2)$, $b_0 = (-6, 6)$, $w_1 = (1, -1)$: $\mathrm{ReLU}(2x-6) - \mathrm{ReLU}(-(2x-6)) = 2x - 6$.
> > 2. Underfitting: above the 25% baseline but far from fitting even the training data, with no gap. Increase the model's capacity (or lower the learning rate if the loss flattened very early).
> > 3. GD: 1 (the whole dataset per step). Mini-batch SGD: $1000/100 = 10$.

Related: [[9-midterm-2/review-choosing-models]] · [[9-midterm-2/review-sizes-and-parameters]] · [[9-midterm-2/true-false]] · [[4-pytorch-training/experimental-setup]]
