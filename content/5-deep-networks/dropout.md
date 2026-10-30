---
title: "Dropout"
description: "Inverted dropout: zero each activation with probability p during training and scale the survivors by 1/(1-p), do nothing at evaluation; nn.Dropout vs F.dropout (the eval-mode bug), nn.Dropout2d for feature maps, and Lecture 16's experiments showing dropout matters when a big model can memorize noisy labels."
tags: [topic, deep-nets]
aliases: ["inverted dropout", "nn.Dropout", "Dropout2d", "F.dropout", "spatial dropout"]
---

*Lecture 16 · [[5-deep-networks/index|Deep networks]] · builds on [[5-deep-networks/multilayer-perceptrons|multilayer perceptrons]] and [[4-pytorch-training/experimental-setup|experimental setup]]*

> [!abstract] In one breath
> During training, dropout sets each activation to 0 with probability $p$ and multiplies the survivors by $\frac{1}{1-p}$, so the expected value is unchanged and the network cannot rely on any single feature. At evaluation it does nothing. It is a **regularizer**: it makes memorizing individual training examples harder, which matters most when a big model meets a small or noisy dataset. Use `nn.Dropout` (or `F.dropout(..., training=self.training)`), and always switch modes with `model.train()` / `model.eval()`.

## What it is

![[l16-dropout-example.png|Dropout: in each training step a random subset of units is removed]]

With a random mask $m_i \sim \text{Bernoulli}(1-p)$ drawn fresh for every activation and every forward pass,

$$
\text{train: } y_i = \frac{m_i\,x_i}{1-p}, \qquad \text{eval: } y_i = x_i, \qquad
\mathbb E[y_i] = \frac{(1-p)\,x_i}{1-p} = x_i .
$$

Scaling during training ("inverted" dropout) is what lets evaluation do nothing at all. Every step trains a different random sub-network; the full network at test time behaves like an average of them.

## What you should know

- $p$ is the probability of **dropping** (zeroing) a unit. `nn.Dropout(p=0.5)` turns a vector of ones into 0s and 2s in training mode.
- Train mode: random zeros and $\frac{1}{1-p}$ scaling; eval mode: identity. Deterministic outputs at test time require `model.eval()`.
- `F.dropout(x, p)` defaults to `training=True` and ignores `model.eval()`; use `nn.Dropout`, or pass `training=self.training`.
- `nn.Dropout2d` zeroes **whole channels** of a conv feature map; neighbouring pixels are so correlated that dropping single pixels would hardly matter.
- Dropout has **no parameters**.
- Lecture 16: with clean labels, dropout changes little; with 30% wrong labels, it keeps the final test accuracy about 10 points higher (77.2% vs 67.4% for the MLP; 77.8% vs 68.8% for the CNN when placed before the fully connected layer).
- Related regularizers: [[4-pytorch-training/optimizers|weight decay]] ([[3-linear-models/ridge-regression|ridge]]), [[5-deep-networks/data-augmentation|augmentation]], early stopping.

## Train mode vs eval mode

```python
drop = nn.Dropout(p=0.5)
x = torch.ones(1, 10)
drop.train()
print('nn.Dropout, train mode:', drop(x))
drop.eval()
print('nn.Dropout, eval mode: ', drop(x))
```

```
nn.Dropout, train mode: tensor([[2., 2., 0., 0., 0., 0., 0., 0., 2., 0.]])
nn.Dropout, eval mode:  tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
```

(Seven of ten zeroed here; with $p=0.5$ the count is random, about half on average.)

### The `F.dropout` bug

```python
class BuggyNet(nn.Module):        # pattern from older course code
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return F.dropout(self.fc(x), p=0.5)    # training=True by default!

net = BuggyNet().eval()
print(torch.equal(net(x), net(x)))
```

```
False
```

In eval mode, with the same input every time, three runs even predicted different classes (6, 6, 0). [`F.dropout`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html) is a plain function and does not know about `.train()`/`.eval()`. Fix: `self.drop = nn.Dropout(0.5)` in `__init__`, or `F.dropout(h, p=0.5, training=self.training)`.

> [!example] Lecture 16 "Your turn 5": inverted dropout by hand
> Complete `my_dropout(x, p, training)`.

> [!solution]- Solution
> ```python
> def my_dropout(x, p, training):
>     if not training or p == 0:
>         return x
>     keep = (torch.rand_like(x) > p).float()   # keep with prob 1 - p
>     return x * keep / (1 - p)
> ```
> On 20,000 ones with $p=0.5$, about half come out 0 and the rest exactly 2.

## Does dropout help?

An MLP far too big for its data, $784\to512\to512\to10$ ($401{,}920 + 262{,}656 + 5{,}130 = 669{,}706$ parameters), trained 100 epochs with Adam on **2,000** images; dropout 0.2 on the input and 0.5 after each hidden layer. In the second experiment 30% of the training labels were replaced at random.

```python
def dropout_mlp(p_hidden=0.0, p_input=0.0):
    return nn.Sequential(nn.Flatten(), nn.Dropout(p_input),
                         nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p_hidden),
                         nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p_hidden),
                         nn.Linear(512, 10))
```

**fit** = accuracy on the training images against their (possibly wrong) training labels; **best** = highest test accuracy at any epoch; **final** = test accuracy after the last epoch.

| after 100 epochs (%) | fit | best | final |
|---|---|---|---|
| clean labels, no dropout | 100.0 | 84.5 | 84.1 |
| clean labels, dropout | 97.6 | 85.6 | 84.1 |
| 30% wrong labels, no dropout | 100.0 | 79.4 | 67.4 |
| 30% wrong labels, dropout | 84.9 | 81.6 | 77.2 |

<!-- fig:deepb-dropout-results -->
<figure class="cs-fig"><svg role="img" aria-label="Bar chart of final test accuracy with and without dropout" style="max-width:620px;width:100%;height:auto" stroke-linejoin="round" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 537.84 206.64" xmlns="http://www.w3.org/2000/svg" version="1.1"><defs></defs><g><g><path d="M 0 206.64  L 537.84 206.64  L 537.84 0  L 0 0  L 0 206.64  z " style="fill: none"/></g><g><g><path d="M 43.2 181.44  L 279.23 181.44  L 279.23 21.56  L 43.2 21.56  L 43.2 181.44  z " style="fill: none"/></g><g><path d="M 53.93 371.77  L 130.07 371.77  L 130.07 51.63  L 53.93 51.63  z " clip-path="url(#ffc3452)" style="fill: var(--muted)"/></g><g><path d="M 192.36 371.77  L 268.5 371.77  L 268.5 51.63  L 192.36 51.63  z " clip-path="url(#ffc3452)" style="fill: var(--accent)"/></g><g><g><g><defs><path id="ffc3450" d="M 0 0  L 0 3.5  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#ffc3450" x="92" y="181.44" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="92" y="196.8" transform="rotate(-0 92 196.8)">no dropout</text></g></g><g><g><g><use xlink:href="#ffc3450" x="230.43" y="181.44" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="230.43" y="196.8" transform="rotate(-0 230.43 196.8)">dropout</text></g></g></g><g><g><g><defs><path id="ffc3451" d="M 0 0  L -3.5 0  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#ffc3451" x="43.2" y="181.44" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="36.2" y="185.62" transform="rotate(-0 36.2 185.62)">50</text></g></g><g><g><g><use xlink:href="#ffc3451" x="43.2" y="143.37" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="36.2" y="147.55" transform="rotate(-0 36.2 147.55)">60</text></g></g><g><g><g><use xlink:href="#ffc3451" x="43.2" y="105.3" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="36.2" y="109.48" transform="rotate(-0 36.2 109.48)">70</text></g></g><g><g><g><use xlink:href="#ffc3451" x="43.2" y="67.24" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="36.2" y="71.42" transform="rotate(-0 36.2 71.42)">80</text></g></g><g><g><g><use xlink:href="#ffc3451" x="43.2" y="29.17" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="36.2" y="33.35" transform="rotate(-0 36.2 33.35)">90</text></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="15.56" y="101.5" transform="rotate(-90 15.56 101.5)">final test accuracy (%)</text></g></g><g><path d="M 43.2 181.44  L 43.2 21.56  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><path d="M 43.2 181.44  L 279.23 181.44  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="92" y="45.92" transform="rotate(-0 92 45.92)">84.1%</text></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="230.43" y="45.92" transform="rotate(-0 230.43 45.92)">84.1%</text></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="43.2" y="15.56" transform="rotate(-0 43.2 15.56)">clean labels</text></g></g><g><g><path d="M 294.61 181.44  L 530.64 181.44  L 530.64 21.56  L 294.61 21.56  L 294.61 181.44  z " style="fill: none"/></g><g><path d="M 305.34 371.77  L 381.48 371.77  L 381.48 115.2  L 305.34 115.2  z " clip-path="url(#ffc3453)" style="fill: var(--muted)"/></g><g><path d="M 443.77 371.77  L 519.91 371.77  L 519.91 77.9  L 443.77 77.9  z " clip-path="url(#ffc3453)" style="fill: var(--accent)"/></g><g><g><g><g><use xlink:href="#ffc3450" x="343.41" y="181.44" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="343.41" y="196.8" transform="rotate(-0 343.41 196.8)">no dropout</text></g></g><g><g><g><use xlink:href="#ffc3450" x="481.84" y="181.44" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="481.84" y="196.8" transform="rotate(-0 481.84 196.8)">dropout</text></g></g></g><g><g><g><g><use xlink:href="#ffc3451" x="294.61" y="181.44" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g></g><g><g><g><use xlink:href="#ffc3451" x="294.61" y="143.37" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g></g><g><g><g><use xlink:href="#ffc3451" x="294.61" y="105.3" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g></g><g><g><g><use xlink:href="#ffc3451" x="294.61" y="67.24" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g></g><g><g><g><use xlink:href="#ffc3451" x="294.61" y="29.17" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g></g></g><g><path d="M 294.61 181.44  L 294.61 21.56  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><path d="M 294.61 181.44  L 530.64 181.44  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="343.41" y="109.49" transform="rotate(-0 343.41 109.49)">67.4%</text></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="481.84" y="72.19" transform="rotate(-0 481.84 72.19)">77.2%</text></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="294.61" y="15.56" transform="rotate(-0 294.61 15.56)">30% wrong labels</text></g></g></g><defs><clipPath id="ffc3452"><rect x="43.2" y="21.56" width="236.03" height="159.88"/></clipPath><clipPath id="ffc3453"><rect x="294.61" y="21.56" width="236.03" height="159.88"/></clipPath></defs></svg> <figcaption><strong>Dropout matters when a big model can memorize bad data</strong>: a 784→512→512→10 MLP (≈670k parameters) trained 100 epochs on 2,000 images. With 30% wrong labels, the model without dropout fits 100% of the training labels and its test accuracy falls from a peak of 79.4% to 67.4%.</figcaption></figure>

- **Clean labels:** both end at the same test accuracy; dropout's curve is just smoother.
- **30% wrong labels:** without dropout the network agrees with *every* training label, wrong ones included (fit 100%), and its test accuracy falls far below its early peak. With dropout, fit drops to 84.9%. About 73% of the labels are correct (70% untouched, plus a tenth of the replaced ones by chance), so it still memorizes some wrong labels, but far fewer, and it ends about 10 points higher.
- Early stopping would also help for the same reason: the model without dropout was at its best early. Choosing the epoch by the **test** set (the "best" column) is optimistic; in practice use a validation set ([[4-pytorch-training/experimental-setup|experimental setup]]).

## Dropout in a CNN

[`nn.Dropout2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html) zeroes whole feature maps; before the fully connected layer the ordinary [`nn.Dropout`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html) is used.

```python
def dropout_cnn(p_conv=0.0, p_fc=0.0):
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(p_conv),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(p_conv),
        nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(p_fc), nn.Linear(256, 10))
```

Same 2,000 images and 30% wrong labels, 60 epochs:

| CNN after 60 epochs (%) | fit | best | final |
|---|---|---|---|
| clean, no dropout | 100.0 | 87.6 | 87.3 |
| clean, Dropout2d only | 100.0 | 87.7 | 87.7 |
| clean, Dropout2d + Dropout | 99.7 | 87.8 | 86.5 |
| 30% wrong, no dropout | 100.0 | 82.5 | 68.8 |
| 30% wrong, Dropout2d only | 99.4 | 82.0 | 69.6 |
| 30% wrong, Dropout2d + Dropout | 97.0 | 82.8 | 77.8 |

`Dropout2d(0.2)` on the feature maps alone barely helps; adding `Dropout(0.5)` before the fully connected layer ends about 9 points higher. That is where the parameters are: `nn.Linear(3136, 256)` has $3136\cdot256 + 256 = 803{,}072$ of the model's $824{,}458$ (the convolutions have $320 + 18{,}496$, the last layer $2{,}570$).

> [!trap] Common mistakes
> - Using `F.dropout` in `forward` without `training=self.training`: dropout stays on at test time.
> - Forgetting `model.eval()` before evaluating (and `model.train()` before the next epoch); the same applies to [[5-deep-networks/batch-normalization|batch norm]].
> - Reading $p$ as the keep probability. In PyTorch it is the **drop** probability.
> - Scaling by $1-p$ at test time as well: PyTorch already scaled during training (inverted dropout); evaluation must do nothing.
> - Expecting dropout to help on every problem: with clean labels and enough data it changed nothing here; it is a regularizer for overfitting.
> - Putting `nn.Dropout` (element-wise) on conv feature maps and expecting much effect; `nn.Dropout2d` drops whole channels.

## PyTorch docs

- [`nn.Dropout`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html) — element-wise, scales by $\frac{1}{1-p}$ in training.
- [`nn.Dropout2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html) — zeroes entire channels of $N\times C\times H\times W$ inputs.
- [`F.dropout`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.dropout.html) — the functional form; note `training=True` default.
- [`nn.Module.train` / `eval`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval) — set `self.training` for every submodule.
- [`torch.rand_like`](https://docs.pytorch.org/docs/stable/generated/torch.rand_like.html) — the random mask in `my_dropout`.

Related: [[5-deep-networks/batch-normalization]] · [[5-deep-networks/data-augmentation]] · [[5-deep-networks/multilayer-perceptrons]] · [[3-linear-models/ridge-regression]] · [[4-pytorch-training/experimental-setup]] · [[4-pytorch-training/training-loop]]
