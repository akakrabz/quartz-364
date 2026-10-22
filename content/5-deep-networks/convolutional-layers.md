---
title: "Convolutional layers (nn.Conv2d)"
description: "Channels and kernel shapes, every nn.Conv2d argument, the output-size formula with padding, stride and dilation, parameter counting for conv, linear and pooling layers, receptive fields with the jump recursion, and three ways to build a CNN in PyTorch, all with worked examples from Lectures 15 and 16."
tags: [topic, deep-nets]
aliases: ["Conv2d", "nn.Conv2d", "convolutional layer", "conv layer", "output size formula", "padding", "stride", "dilation", "groups", "depthwise convolution", "receptive field", "parameter count", "channels", "feature map", "CNN", "convolutional neural network"]
---

*Lectures 15 and 16 · [[5-deep-networks/index|Deep networks]] · builds on [[5-deep-networks/convolution|convolution]] · see also [[5-deep-networks/pooling|pooling and CNN heads]]*

> [!abstract] In one breath
> A convolutional layer with $C_\textrm{in}$ input channels and $C_\textrm{out}$ output channels holds $C_\textrm{out}$ kernels, each of shape $C_\textrm{in}\times K\times K$; each kernel sums over **all** input channels and produces one output feature map. Three formulas do almost all the work: the output size $H_\textrm{out} = \lfloor (H_\textrm{in} + 2p - K)/s \rfloor + 1$, the parameter count $C_\textrm{out} C_\textrm{in} K^2 + C_\textrm{out}$ (independent of the image size), and the receptive field recursion $r_l = r_{l-1} + (k_l - 1)\,j_{l-1}$, $j_l = j_{l-1} s_l$.

## What it is

A CNN is built from four kinds of components:

1. **Convolutional layers**: many filters per layer, each producing one feature map.
2. **Non-linear [[5-deep-networks/activation-functions|activations]]** after each convolution (without them, stacked convolutions collapse into one).
3. **[[5-deep-networks/pooling|Pooling]]** (or strided convolution) to shrink feature maps.
4. A **head** that turns feature maps into class scores (flatten + linear, or global pooling + linear).

![[l16-five-layer-cnn.png|A five-layer CNN: two 3x3 convs and a max pool, two 5x5 convs and a max pool, a 3x3 conv, global average pooling, a fully-connected layer]]

The figure is the lecture's `FiveLayerCNN`. (Its last $3\times3$ layer is labeled "pad=2"; the notebook's code uses `padding=1`, which is the value that keeps the size.)

**Channels.** An input with $C_\textrm{in}$ channels (1 for grayscale, 3 for RGB, or the previous layer's $C_\textrm{out}$) is convolved with $C_\textrm{out}$ kernels. Kernel $c$ has shape $C_\textrm{in}\times K\times K$, and output map $c$ is

$$
y_c[i, j] = \sum_{c'=1}^{C_\textrm{in}} \sum_{a,b} x_{c'}[i\cdot s + a,\; j\cdot s + b]\; w_{c, c'}[a, b] \;+\; \textrm{bias}_c .
$$

So the layer's `weight` has shape `(C_out, C_in, K, K)` and its `bias` has shape `(C_out,)`: one bias per output channel, not per pixel.

![[l16-convolution-with-multiple-filters2.png|A 6x6x3 input convolved with two 3x3x3 filters gives a 4x4x2 output]]

In the figure, a $6\times6\times3$ input meets two $3\times3\times3$ filters; each filter covers all three channels and produces one $4\times4$ map ($6 - 3 + 1 = 4$), and the two maps stack into a $4\times4\times2$ output. That layer has $2\cdot3\cdot3\cdot3 + 2 = 56$ parameters with bias.

## What you should know

- Weight shape `(C_out, C_in, K, K)`, bias shape `(C_out,)`; each kernel spans all input channels.
- Output size per spatial dimension: $H_\textrm{out} = \left\lfloor \frac{H_\textrm{in} + 2p - K}{s} \right\rfloor + 1$ (with dilation $d$: replace $K$ by $d(K-1)+1$). "Same" padding $p = (K-1)/2$ at stride 1 keeps the size; stride 2 halves it.
- Parameters: conv $C_\textrm{out}\,(C_\textrm{in}/\textrm{groups})\,K^2 + C_\textrm{out}$; linear $n_\textrm{out} n_\textrm{in} + n_\textrm{out}$; batch norm $2C$; pooling, activations, flatten $0$. Parameter counts do **not** depend on $H\times W$ (a flatten head's linear layer does).
- Receptive field: each layer adds $(k-1)\times$ jump; each stride or pool multiplies the jump. Three $3\times3$ layers see $7\times7$, like one $7\times7$ layer, with fewer parameters.
- Build CNNs with attributes + `forward`, `nn.Sequential`, or block functions; keep sub-modules in `nn.Sequential` / `nn.ModuleList`, never a plain list.
- Do the [[#worked-example-the-lecture-16-warm-up|warm-up]] (output shape, parameters, receptive field) without a computer.

## Output size

$$
H_\textrm{out} = \left\lfloor\frac{H_\textrm{in}+2p-K}{s}\right\rfloor + 1
\qquad\text{(general: } H_\textrm{out} = \left\lfloor\frac{H_\textrm{in} + 2p - d\,(K-1) - 1}{s}\right\rfloor + 1\text{)}
$$

The same formula holds for the width and for pooling layers. Why: padding gives a row of $H_\textrm{in} + 2p$ cells, a window of $K$ cells can start at positions $0, s, 2s, \ldots$ as long as it fits, and there are $\lfloor (H_\textrm{in} + 2p - K)/s\rfloor + 1$ such positions.

<!-- fig:deepa-padding-stride -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 490 250" width="490" height="250" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>A padded row of 9 cells with four kernel windows of width 3 at stride 2</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><text x="30" y="16" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">H_in = 7, padding p = 1, kernel K = 3, stride s = 2</text><rect x="30" y="46" width="40" height="40" rx="0" fill="none" stroke="var(--muted)" stroke-width="1.4" stroke-dasharray="4 3"/><text x="50" y="66" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">0</text><rect x="70" y="46" width="40" height="40" rx="0" fill="none" stroke="currentColor" stroke-width="1.4"/><text x="90" y="66" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">3</text><rect x="110" y="46" width="40" height="40" rx="0" fill="none" stroke="currentColor" stroke-width="1.4"/><text x="130" y="66" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">1</text><rect x="150" y="46" width="40" height="40" rx="0" fill="none" stroke="currentColor" stroke-width="1.4"/><text x="170" y="66" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">4</text><rect x="190" y="46" width="40" height="40" rx="0" fill="none" stroke="currentColor" stroke-width="1.4"/><text x="210" y="66" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">1</text><rect x="230" y="46" width="40" height="40" rx="0" fill="none" stroke="currentColor" stroke-width="1.4"/><text x="250" y="66" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">5</text><rect x="270" y="46" width="40" height="40" rx="0" fill="none" stroke="currentColor" stroke-width="1.4"/><text x="290" y="66" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">9</text><rect x="310" y="46" width="40" height="40" rx="0" fill="none" stroke="currentColor" stroke-width="1.4"/><text x="330" y="66" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">2</text><rect x="350" y="46" width="40" height="40" rx="0" fill="none" stroke="var(--muted)" stroke-width="1.4" stroke-dasharray="4 3"/><text x="370" y="66" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">0</text><rect x="33" y="104" width="114" height="18" rx="4" fill="var(--accent)" fill-opacity="0.15" stroke="var(--accent)" stroke-width="1.6"/><text x="158" y="113" text-anchor="start" dominant-baseline="central" fill="var(--accent)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">output 1</text><rect x="113" y="130" width="114" height="18" rx="4" fill="var(--hi)" fill-opacity="0.15" stroke="var(--hi)" stroke-width="1.6"/><text x="238" y="139" text-anchor="start" dominant-baseline="central" fill="var(--hi)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">output 2</text><rect x="193" y="156" width="114" height="18" rx="4" fill="var(--accent2)" fill-opacity="0.15" stroke="var(--accent2)" stroke-width="1.6"/><text x="318" y="165" text-anchor="start" dominant-baseline="central" fill="var(--accent2)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">output 3</text><rect x="273" y="182" width="114" height="18" rx="4" fill="var(--accent)" fill-opacity="0.15" stroke="var(--accent)" stroke-width="1.6"/><text x="398" y="191" text-anchor="start" dominant-baseline="central" fill="var(--accent)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">output 4</text><text x="30" y="238" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">floor((7 + 2·1 − 3) / 2) + 1 = floor(6/2) + 1 = 4 outputs</text></svg><figcaption><b>Where the output-size formula comes from.</b> Padding adds <i>p</i> zeros on each side (dashed), so the padded row has <i>H</i> + 2<i>p</i> = 9 cells. A window of <i>K</i> = 3 cells fits at starting positions 0, 2, 4, 6 when it moves <i>s</i> = 2 cells at a time: ⌊(9 − 3)/2⌋ + 1 = 4 outputs.</figcaption></figure>

**Lecture 15's examples** on a $3\times28\times28$ input with 16 filters:

| $K$ | $s$ | $p$ | $H_\textrm{out}$ | output shape | weight shape |
|---|---|---|---|---|---|
| 3 | 1 | 1 | $\lfloor (28+2-3)/1\rfloor + 1 = 28$ | `(1, 16, 28, 28)` | `(16, 3, 3, 3)` |
| 3 | 1 | 0 | $\lfloor (28-3)/1\rfloor + 1 = 26$ | `(1, 16, 26, 26)` | `(16, 3, 3, 3)` |
| 5 | 2 | 2 | $\lfloor (28+4-5)/2\rfloor + 1 = 13 + 1 = 14$ | `(1, 16, 14, 14)` | `(16, 3, 5, 5)` |
| 7 | 2 | 3 | $\lfloor (28+6-7)/2\rfloor + 1 = 13 + 1 = 14$ | `(1, 16, 14, 14)` | `(16, 3, 7, 7)` |

The floor matters: in the last two rows $27/2 = 13.5$ rounds down. A $2\times2$ max pool with stride 2 maps $28\rightarrow14\rightarrow7\rightarrow3$ (since $\lfloor (7-2)/2\rfloor + 1 = 3$).

```python
def conv_output_size(h_in, k, s=1, p=0, d=1):
    return (h_in + 2 * p - d * (k - 1) - 1) // s + 1  # floor division

conv_output_size(28, k=5, s=2, p=2)  # 14
```

## Every argument of `nn.Conv2d`

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
          groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
```

| argument | what it does | typical values |
|---|---|---|
| `in_channels` | channels of the input: 1 grayscale, 3 RGB, or the previous layer's `out_channels` | 1, 3, 16, 64 |
| `out_channels` | number of filters = number of output feature maps | 16 to 512 |
| `kernel_size` | height and width of each filter (one number means square) | 3; sometimes 1, 5, 7 |
| `stride` | step between neighboring filter positions; 2 halves the output size | 1, 2 |
| `padding` | pixels added around the border: `(K-1)/2` or `'same'` keeps the size (`'same'` needs stride 1); `0` or `'valid'` shrinks it | 0, 1, `'same'` |
| `dilation` | spacing between taps: $3\times3$ with dilation 2 covers $5\times5$ pixels with the same 9 weights | 1, 2 |
| `groups` | splits channels into independent groups; `groups=in_channels` is one filter per channel (depthwise) | 1, `in_channels` |
| `bias` | one learnable constant per output channel; often `False` when [[5-deep-networks/batch-normalization|batch norm]] follows | `True` |
| `padding_mode` | what fills the padding: `'zeros'`, `'reflect'`, `'replicate'`, `'circular'` | `'zeros'` |

The same 16-channel $28\times28$ input through different layers (Lecture 15 output):

| `nn.Conv2d(16, ...)` | output shape | weight shape | parameters |
|---|---|---|---|
| `32, 3, padding=1` (size kept) | `(32, 28, 28)` | `(32, 16, 3, 3)` | $32\cdot16\cdot9 + 32 = 4{,}640$ |
| `32, 5, padding='same'` | `(32, 28, 28)` | `(32, 16, 5, 5)` | $32\cdot16\cdot25 + 32 = 12{,}832$ |
| `32, 3` (no padding, shrinks) | `(32, 26, 26)` | `(32, 16, 3, 3)` | $4{,}640$ |
| `32, 3, stride=2, padding=1` (halved) | `(32, 14, 14)` | `(32, 16, 3, 3)` | $4{,}640$ |
| `32, 3, dilation=2, padding=2` (covers $5\times5$) | `(32, 28, 28)` | `(32, 16, 3, 3)` | $4{,}640$ |
| `16, 3, padding=1, groups=16` (depthwise) | `(16, 28, 28)` | `(16, 1, 3, 3)` | $16\cdot1\cdot9 + 16 = 160$ |
| `32, 3, padding=1, bias=False` | `(32, 28, 28)` | `(32, 16, 3, 3)` | $4{,}608$ |

Stride, padding and dilation change the output shape but **not** the number of parameters.

## Counting parameters

| layer | parameters |
|---|---|
| `nn.Conv2d(C_in, C_out, K)` | $C_\textrm{out}\cdot C_\textrm{in}\cdot K^2 + C_\textrm{out}$ (general: $C_\textrm{out}\cdot \frac{C_\textrm{in}}{\textrm{groups}}\cdot K_hK_w + C_\textrm{out}$) |
| `nn.Linear(n_in, n_out)` | $n_\textrm{out}\cdot n_\textrm{in} + n_\textrm{out}$ |
| `nn.BatchNorm2d(C)` | $2C$ (plus $2C$ running statistics, which are not trained) |
| pooling, activations, flatten | $0$ |

**Lecture 15's four models** on a $1\times64\times64$ canvas, 10 classes, every conv and linear layer with a bias. Conv layers: $3\times3$, padding 1, channels $1\rightarrow16\rightarrow32\rightarrow64$, a $2\times2$ max pool after the first two, so the last feature maps are $64\times16\times16$.

| layer | output shape | parameters |
|---|---|---|
| `Conv2d(1, 16, 3, padding=1)` | `(16, 64, 64)` | $16\cdot1\cdot9 + 16 = 160$ |
| `MaxPool2d(2)` | `(16, 32, 32)` | $0$ |
| `Conv2d(16, 32, 3, padding=1)` | `(32, 32, 32)` | $32\cdot16\cdot9 + 32 = 4{,}640$ |
| `MaxPool2d(2)` | `(32, 16, 16)` | $0$ |
| `Conv2d(32, 64, 3, padding=1)` | `(64, 16, 16)` | $64\cdot32\cdot9 + 64 = 18{,}496$ |
| `Flatten` | `(16384,)` | $0$ |
| `Linear(16384, 10)` | `(10,)` | $16384\cdot10 + 10 = 163{,}850$ |
| **CNN + flatten + FC, total** | | **187,146** |

| model | head | total |
|---|---|---|
| MLP $4096\rightarrow256\rightarrow10$ | – | $1{,}048{,}832 + 2{,}570 = 1{,}051{,}402$ |
| CNN + flatten + FC | `Linear(16384, 10)` | $23{,}296 + 163{,}850 = 187{,}146$ |
| CNN + GAP + FC | global avg pool, `Linear(64, 10)` | $23{,}296 + 650 = 23{,}946$ |
| CNN $5\times5$ + GAP + FC | same, $5\times5$ kernels, padding 2 | $416 + 12{,}832 + 51{,}264 + 650 = 65{,}162$ |

Most of the flatten model's parameters live in its last linear layer. The GAP model has the same 23,946 parameters for any image size, while the MLP's first layer alone grows to 12,845,312 for a $224\times224$ input.

**VGG's insight: stack small kernels.** With 64 channels in and out:

| | receptive field | parameters |
|---|---|---|
| one $7\times7$ layer | $7\times7$ | $64\cdot64\cdot49 + 64 = 200{,}768$ |
| three $3\times3$ layers (ReLU between) | $3\rightarrow5\rightarrow7$ | $3\,(64\cdot64\cdot9 + 64) = 110{,}784$ |

Same view, about 45% fewer parameters, and two extra non-linearities ([[5-deep-networks/cnn-architectures|CNN architectures]]).

```python
def conv_params(c_in, c_out, k, bias=True, groups=1):
    return c_out * (c_in // groups) * k * k + (c_out if bias else 0)

n_params = lambda m: sum(p.numel() for p in m.parameters())
n_params(nn.Conv2d(16, 32, 3)) == conv_params(16, 32, 3)  # True: 4640
```

## Receptive field

The **receptive field** of a unit is the patch of the input image that can influence it.

- One $3\times3$ layer: each unit sees $3\times3$.
- A second $3\times3$ layer on top: it sees $3\times3$ units whose patches overlap, shifted by one pixel, so $5\times5$. Each extra $3\times3$ layer adds one pixel on each side: 3, 5, 7, 9, …
- After a $2\times2$ pool (stride 2), neighboring units are 2 pixels apart in the image (the **jump**), so the next $3\times3$ kernel adds $2\cdot2 = 4$ pixels instead of 2.

> [!key] The rule
> Each layer adds $(k-1)\times$ jump pixels; each stride or pooling multiplies the jump. With kernel $k_l$ and stride $s_l$ at layer $l$ (convolutions and pooling alike):
> $$
> r_l = r_{l-1} + (k_l - 1)\,j_{l-1}, \qquad j_l = j_{l-1}\, s_l, \qquad r_0 = j_0 = 1.
> $$

<!-- fig:deepa-receptive-field -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 612 268" width="612" height="268" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>Receptive-field cone through conv, pool, conv layers covering 8 input pixels</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><line x1="252" y1="154" x2="218" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="252" y1="154" x2="252" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="252" y1="154" x2="286" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="286" y1="154" x2="252" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="286" y1="154" x2="286" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="286" y1="154" x2="320" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="320" y1="154" x2="286" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="320" y1="154" x2="320" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="320" y1="154" x2="354" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="354" y1="154" x2="320" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="354" y1="154" x2="354" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="354" y1="154" x2="388" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="388" y1="154" x2="354" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="388" y1="154" x2="388" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="388" y1="154" x2="422" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="422" y1="154" x2="388" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="422" y1="154" x2="422" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="422" y1="154" x2="456" y2="216" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="269" y1="92" x2="252" y2="154" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="269" y1="92" x2="286" y2="154" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="337" y1="92" x2="320" y2="154" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="337" y1="92" x2="354" y2="154" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="405" y1="92" x2="388" y2="154" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="405" y1="92" x2="422" y2="154" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="337" y1="30" x2="269" y2="92" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="337" y1="30" x2="337" y2="92" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><line x1="337" y1="30" x2="405" y2="92" stroke="var(--accent)" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/><rect x="140" y="206" width="20" height="20" rx="3" fill="none" stroke="var(--muted)" stroke-width="1.4"/><rect x="174" y="206" width="20" height="20" rx="3" fill="none" stroke="var(--muted)" stroke-width="1.4"/><rect x="208" y="206" width="20" height="20" rx="3" fill="var(--accent)" fill-opacity="0.35" stroke="var(--accent)" stroke-width="1.4"/><rect x="242" y="206" width="20" height="20" rx="3" fill="var(--accent)" fill-opacity="0.35" stroke="var(--accent)" stroke-width="1.4"/><rect x="276" y="206" width="20" height="20" rx="3" fill="var(--accent)" fill-opacity="0.35" stroke="var(--accent)" stroke-width="1.4"/><rect x="310" y="206" width="20" height="20" rx="3" fill="var(--accent)" fill-opacity="0.35" stroke="var(--accent)" stroke-width="1.4"/><rect x="344" y="206" width="20" height="20" rx="3" fill="var(--accent)" fill-opacity="0.35" stroke="var(--accent)" stroke-width="1.4"/><rect x="378" y="206" width="20" height="20" rx="3" fill="var(--accent)" fill-opacity="0.35" stroke="var(--accent)" stroke-width="1.4"/><rect x="412" y="206" width="20" height="20" rx="3" fill="var(--accent)" fill-opacity="0.35" stroke="var(--accent)" stroke-width="1.4"/><rect x="446" y="206" width="20" height="20" rx="3" fill="var(--accent)" fill-opacity="0.35" stroke="var(--accent)" stroke-width="1.4"/><rect x="480" y="206" width="20" height="20" rx="3" fill="none" stroke="var(--muted)" stroke-width="1.4"/><rect x="514" y="206" width="20" height="20" rx="3" fill="none" stroke="var(--muted)" stroke-width="1.4"/><rect x="548" y="206" width="20" height="20" rx="3" fill="none" stroke="var(--muted)" stroke-width="1.4"/><text x="128" y="216" text-anchor="end" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">input pixels</text><circle cx="184" cy="154" r="8" fill="var(--paper)" stroke="var(--muted)" stroke-width="1.6"/><circle cx="218" cy="154" r="8" fill="var(--paper)" stroke="var(--muted)" stroke-width="1.6"/><circle cx="252" cy="154" r="8" fill="var(--accent)" fill-opacity="0.5" stroke="var(--accent)" stroke-width="1.6"/><circle cx="286" cy="154" r="8" fill="var(--accent)" fill-opacity="0.5" stroke="var(--accent)" stroke-width="1.6"/><circle cx="320" cy="154" r="8" fill="var(--accent)" fill-opacity="0.5" stroke="var(--accent)" stroke-width="1.6"/><circle cx="354" cy="154" r="8" fill="var(--accent)" fill-opacity="0.5" stroke="var(--accent)" stroke-width="1.6"/><circle cx="388" cy="154" r="8" fill="var(--accent)" fill-opacity="0.5" stroke="var(--accent)" stroke-width="1.6"/><circle cx="422" cy="154" r="8" fill="var(--accent)" fill-opacity="0.5" stroke="var(--accent)" stroke-width="1.6"/><circle cx="456" cy="154" r="8" fill="var(--paper)" stroke="var(--muted)" stroke-width="1.6"/><circle cx="490" cy="154" r="8" fill="var(--paper)" stroke="var(--muted)" stroke-width="1.6"/><circle cx="524" cy="154" r="8" fill="var(--paper)" stroke="var(--muted)" stroke-width="1.6"/><text x="128" y="154" text-anchor="end" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">3-wide conv</text><circle cx="201" cy="92" r="8" fill="var(--paper)" stroke="var(--muted)" stroke-width="1.6"/><circle cx="269" cy="92" r="8" fill="var(--accent)" fill-opacity="0.5" stroke="var(--accent)" stroke-width="1.6"/><circle cx="337" cy="92" r="8" fill="var(--accent)" fill-opacity="0.5" stroke="var(--accent)" stroke-width="1.6"/><circle cx="405" cy="92" r="8" fill="var(--accent)" fill-opacity="0.5" stroke="var(--accent)" stroke-width="1.6"/><circle cx="473" cy="92" r="8" fill="var(--paper)" stroke="var(--muted)" stroke-width="1.6"/><text x="128" y="92" text-anchor="end" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">2×2 pool, stride 2</text><circle cx="269" cy="30" r="8" fill="var(--paper)" stroke="var(--muted)" stroke-width="1.6"/><circle cx="337" cy="30" r="8" fill="var(--accent)" fill-opacity="0.5" stroke="var(--accent)" stroke-width="1.6"/><circle cx="405" cy="30" r="8" fill="var(--paper)" stroke="var(--muted)" stroke-width="1.6"/><text x="128" y="30" text-anchor="end" dominant-baseline="central" fill="currentColor" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">3-wide conv</text></svg><figcaption><b>Receptive field, one row of pixels.</b> Conv 3 → pool 2 (stride 2) → conv 3. The top unit depends on the 8 highlighted input pixels: <i>r</i> = 1 + (3−1)·1 + (2−1)·1 + (3−1)·2 = 8. After the pool the jump is 2, so the last 3-wide kernel adds 2·2 = 4 pixels instead of 2.</figcaption></figure>

| layer | $k$ | $s$ | $r$ | $j$ |
|---|---|---|---|---|
| input pixels | – | – | 1 | 1 |
| 3-wide conv | 3 | 1 | $1 + 2\cdot1 = 3$ | 1 |
| pool, stride 2 | 2 | 2 | $3 + 1\cdot1 = 4$ | 2 |
| 3-wide conv | 3 | 1 | $4 + 2\cdot2 = 8$ | 2 |

For the lecture's `FiveLayerCNN` ($3\times3$, $3\times3$, pool, $5\times5$, $5\times5$, pool, $3\times3$):

```
layer         kernel  stride    jump   receptive field
Conv2d             3       1       1             3 x 3
Conv2d             3       1       1             5 x 5
MaxPool2d          2       2       2             6 x 6
Conv2d             5       1       2           14 x 14
Conv2d             5       1       2           22 x 22
MaxPool2d          2       2       4           24 x 24
Conv2d             3       1       4           32 x 32
```

Once two poolings have made the jump 4, the last $3\times3$ layer alone adds $2\cdot4 = 8$ pixels. Backpropagating from one unit of that last layer to a $64\times64$ input gives non-zero gradient on exactly a $32\times32$ region, strongest in the center (central pixels reach the unit through more paths), so the **effective** receptive field is smaller than the theoretical one. Global pooling makes the receptive field the whole image.

**Stride grows it faster.** A stack of $3\times3$, stride-2, padding-1 layers with 16 channels on a $28\times28$ image (the convolution explorer):

| layer | output shape | parameters | receptive field |
|---|---|---|---|
| conv 1 | $16\times14\times14$ | $16\cdot1\cdot9 + 16 = 160$ | $3\times3$ |
| conv 2 | $16\times7\times7$ | $16\cdot16\cdot9 + 16 = 2{,}320$ | $7\times7$ |
| conv 3 | $16\times4\times4$ | $2{,}320$ | $15\times15$ |
| conv 4 | $16\times2\times2$ | $2{,}320$ | $31\times31$ |

Receptive fields 3, 7, 15, 31 (vs 3, 5, 7, 9 at stride 1), at the same parameter cost, total 7,120. A $1\times1$ kernel never grows the receptive field; it mixes channels at each pixel, which is what $1\times1$ convolutions in [[5-deep-networks/branching-and-inception|Inception blocks]] are for.

```python
def rf_from_layers(layers):          # [(kernel, stride), ...]
    r, j = 1, 1
    for k, s in layers:
        r = r + (k - 1) * j          # add (k-1) x jump
        j = j * s                    # stride multiplies jump
    return r

rf_from_layers([(3, 1), (3, 1), (2, 2), (5, 1), (5, 1), (2, 2), (3, 1)])  # 32
```

## Worked example: the Lecture 16 warm-up

> [!question] Warm-up
> For a $3\times32\times32$ input and `nn.Sequential(nn.Conv2d(3, 8, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2))`, find (a) the output shape, (b) the number of parameters, (c) the receptive field of one output unit.

**(a)** The conv keeps the size ($\lfloor (32 + 4 - 5)/1\rfloor + 1 = 32$), the pool halves it: `(8, 16, 16)`. **(b)** $8\cdot3\cdot25 + 8 = 608$ (ReLU and pooling have none). **(c)** $r = 1 + 4\cdot1 = 5$ after the conv, then $5 + 1\cdot1 = 6$ after the pool: $6\times6$. All three match the notebook's `summary` and `receptive_field` output.

> [!question] Take-home: extend the warm-up
> Append `nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(16, 32, kernel_size=3, padding=1)`. Same three questions.

> [!solution]- Solution (worked by hand)
> | layer | output shape | parameters | $r$ | $j$ |
> |---|---|---|---|---|
> | `Conv2d(3, 8, 5, padding=2)` | $(8, 32, 32)$ | $8\cdot3\cdot25 + 8 = 608$ | 5 | 1 |
> | `MaxPool2d(2)` | $(8, 16, 16)$ | 0 | 6 | 2 |
> | `Conv2d(8, 16, 3, stride=2, padding=1)` | $(16, 8, 8)$: $\lfloor (16+2-3)/2\rfloor + 1 = 8$ | $16\cdot8\cdot9 + 16 = 1{,}168$ | $6 + 2\cdot2 = 10$ | 4 |
> | `Conv2d(16, 32, 3, padding=1)` | $(32, 8, 8)$ | $32\cdot16\cdot9 + 32 = 4{,}640$ | $10 + 2\cdot4 = 18$ | 4 |
>
> Output `(32, 8, 8)`, $608 + 1{,}168 + 4{,}640 = 6{,}416$ parameters, receptive field $18\times18$. Check with the notebook's `summary(warmup, torch.rand(1, 3, 32, 32))` and `receptive_field(...)`.

## Building a CNN in PyTorch

Lecture 15 writes one network three ways: three stages of two $3\times3$ convolutions (padding 1) and a $2\times2$ max pool, with 16, 32 and 64 channels, then global average pooling and a linear layer to 10 classes, for an RGB input.

**1. Attributes.** Full control in `forward` (skip connections, several outputs); every layer appears twice.

```python
class AttributeCNN(nn.Module):
    def __init__(self, in_channels=3, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        # ... conv3 to conv6: 32, 64 channels
        self.relu, self.pool = nn.ReLU(), nn.MaxPool2d(2)  # reusable: no params
        self.gap, self.fc = nn.AdaptiveAvgPool2d(1), nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv2(self.relu(self.conv1(x)))))  # stage 1
        # ... stages 2 and 3
        return self.fc(self.gap(x).flatten(1))
```

**2. `nn.Sequential`.** No `forward` to write; cannot branch or merge.

**3. Blocks.** A function returns the repeated pattern; the model stacks them.

```python
def conv_block(c_in, c_out):
    # two 3x3 convs, then halve the size
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(),
                         nn.MaxPool2d(2))

class BlockCNN(nn.Module):
    def __init__(self, in_channels=1, widths=(16, 32, 64), n_classes=10):
        super().__init__()
        channels = (in_channels,) + tuple(widths)
        self.blocks = nn.Sequential(*[conv_block(a, b) for a, b in zip(channels[:-1], channels[1:])])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(widths[-1], n_classes))

    def forward(self, x):
        return self.head(self.blocks(x))
```

All three have **72,730** parameters for an RGB input: blocks of $2{,}768$ ($448 + 2{,}320$), $13{,}888$ ($4{,}640 + 9{,}248$) and $55{,}424$ ($18{,}496 + 36{,}928$), plus a $650$-parameter head. Given the same weights they produce identical outputs. Because nothing in `BlockCNN` depends on the image size (the head pools globally), the same model accepts $28\times28$, $32\times32$, $64\times64$ or $97\times97$ inputs and always returns `(1, 10)`.

> [!warning] A classic bug: layers in a Python list
> `self.blocks = [conv_block(a, b) for ...]` stores the blocks in a plain list, which `nn.Module` does not register. `model.parameters()` then sees only the head (**650** parameters instead of **72,730**): the optimizer trains only the head, `model.to('cuda')` leaves the conv layers on the CPU, `state_dict()` does not save them, and no error is raised. Use `nn.ModuleList(...)` or `nn.Sequential(...)`.

To inspect a model layer by layer, the lecture's `summary` helper registers [forward hooks](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) that record each leaf module's output shape and parameter count.

> [!trap] Common mistakes
> - **Bias per pixel.** A conv layer has one bias per output channel: $C_\textrm{out}$ biases, whatever the image size.
> - **Forgetting $C_\textrm{in}$** in the parameter count. Each kernel is $C_\textrm{in}\times K\times K$, not $K\times K$.
> - **Rounding up.** The output-size formula floors: $\lfloor 27/2\rfloor + 1 = 14$, not 15.
> - **`padding='same'` with stride 2** is not allowed in PyTorch; use `padding=(K-1)//2`.
> - **Channel mismatch.** Each layer's `in_channels` must equal the previous layer's `out_channels`.
> - **Counting pooling or ReLU parameters.** They have none.
> - **Receptive field vs output size.** They are different quantities: stride shrinks the output and *grows* the receptive field.
> - **Ignoring the jump** after a pool or strided layer when computing receptive fields.

## PyTorch docs

- [`torch.nn.Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html): arguments, the cross-correlation formula, and the output-shape formula with dilation.
- [`torch.nn.Conv1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) and [`torch.nn.functional.conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html).
- [`torch.nn.MaxPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) and [`torch.nn.AdaptiveAvgPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html): the pooling layers used above.
- [`torch.nn.BatchNorm2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html): its $2C$ learnable parameters.
- [`torch.nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html) and [`torch.nn.ModuleList`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html): registering sub-modules.
- [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html): `parameters()`, `named_modules()`, `apply()`, `register_forward_hook()`.
- [`torch.Tensor.numel`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numel.html): counting parameters.
- [Convolution layers in `torch.nn`](https://docs.pytorch.org/docs/stable/nn.html#convolution-layers).
- Tutorial: [Training a classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html), a small CNN on CIFAR-10.

Related: [[5-deep-networks/convolution]] · [[5-deep-networks/pooling]] · [[5-deep-networks/cnn-architectures]] · [[5-deep-networks/branching-and-inception]] · [[5-deep-networks/batch-normalization]] · [[4-pytorch-training/nn-module]]
