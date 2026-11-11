---
title: "Review: choosing a model"
description: "Midterm 2 problem type 6, the skill the lecturer stressed: which of MLP, CNN, RNN and Transformer is appropriate for a task and why. One comparison table (what each assumes, what its first layer can see, whether the input size may change, where its parameters are shared), a task-to-model table, the past exam questions built on these comparisons, and a template for answering which-model-and-why questions."
tags: [exam, midterm-2, deep-nets]
aliases: ["choosing a model", "model comparison", "MLP vs CNN vs RNN vs Transformer", "which model for which task"]
---

*Midterm 2 · problem type 6 of 6 · HW5 Q3, HW6 Q1(d) and Q3 · Lectures 14–21 · 23 points on Fall 2025, true/false items on the others · back to [[9-midterm-2/index|Midterm 2]]*

> [!abstract] In one breath
> The four model families differ in **what they assume about the data**. An MLP assumes nothing: every input is connected to everything, at the price of a fixed input size and many parameters. A CNN assumes nearby pixels belong together and a pattern means the same wherever it is. An RNN assumes the data comes in order and reads it step by step. A Transformer assumes only that the data can be cut into tokens, and lets every token look at every other. Most exam questions here are one row of the table below, asked as "which models can…, and why".

## The comparison

| | MLP | CNN | RNN | Transformer |
|---|---|---|---|---|
| Input | one fixed-length vector | a grid (image), any size | a sequence, any length | a set of tokens with positions added, any number |
| Core operation | matrix product with **all** inputs | a small kernel slid over the grid | the same cell applied step after step, state carried along | self-attention: every token with every token |
| Weights shared across | nothing | positions | time steps | tokens |
| Parameters depend on input size? | **yes** (first layer) | no | no | no |
| What one first-layer output sees | every input | a $K\times K$ window | inputs up to this step | every token |
| Far-apart inputs interact | in layer 1 | only after enough layers (receptive field) | through the hidden state, fading with distance | in layer 1, directly |
| Order / position information | fixed by the input slots | built into the grid | built into the step-by-step order | **must be added** (positional encoding) |
| Trains in parallel over the input? | yes | yes | no: step $t$ needs step $t-1$ | yes |
| Cost grows with | input size × width | image area | sequence length | (number of tokens)$^2$ |

Evidence from the course for the rows:

- **Scrambling the pixels** (the same permutation for every image) leaves an MLP unchanged (84.7% → 84.0% in Lecture 15) and costs the CNNs 3 to 7 points ([[5-deep-networks/convolution#when-the-assumption-is-wrong-scrambled-pixels|experiment]]; [[9-midterm-2/spring-2026#problem-1--fact-or-cap-30-pts|Sp26 1(b)]]).
- **Shifting test images** by 2 pixels takes the MLP from 86.8% to 46.1% and a CNN with global pooling from 78.0% to 74.4% ([[5-deep-networks/pooling#the-shift-experiment|experiment]]; [[9-midterm-2/spring-2026#problem-1--fact-or-cap-30-pts|Sp26 1(c)]]).
- **The sequence-to-sequence RNN's three limits** (no parallel training, a single context vector, limited encoder–decoder interaction) are what attention removes ([[7-transformers/transformer-architecture#how-the-transformer-fixes-the-s2s-rnns-limits|how]]).

## Which model for which task

*A summary assembled from the topic pages, not a table shown in class.*

| Data / task | First choice | Why | Watch out for |
|---|---|---|---|
| Fixed-length feature vectors (tabular) | [[5-deep-networks/multilayer-perceptrons\|MLP]] | no spatial or temporal structure to exploit | parameter count grows with input size |
| Images: classify | [[5-deep-networks/convolutional-layers\|CNN]] with [[5-deep-networks/pooling\|pooling]] / global pooling, [[5-deep-networks/residual-connections\|residual blocks]] when deep; or a [[7-transformers/vision-transformer\|Vision Transformer]] | local patterns, weight sharing; a ViT relates distant patches at once | a flatten + `Linear` head ties a CNN to one input size; a ViT needs more data or care in training |
| Images: per-pixel output | [[5-deep-networks/fully-convolutional-networks\|FCN / U-Net]] | the output is an image; skip connections keep detail | judge with [[5-deep-networks/semantic-segmentation\|IoU]], not pixel accuracy |
| A sequence → one label | [[5-deep-networks/recurrent-neural-networks\|RNN]] / [[5-deep-networks/lstm-and-gru\|LSTM, GRU]] reading the last state, [[5-deep-networks/bidirectional-and-stacked-rnns\|bidirectional]] if the whole sequence is available; or a [[7-transformers/text-classification-with-transformers\|Transformer encoder with `<CLS>`]] | both handle variable length | long-range memory in plain RNNs; compare with the majority-class baseline |
| A sequence → a sequence of another length | [[7-transformers/sequence-to-sequence-rnns\|seq2seq RNN]] → [[7-transformers/transformer-architecture\|encoder–decoder Transformer]] | the encoder reads, the decoder writes until `<EOS>` | the single context vector of the RNN version |
| A sequence of images (video) | a recurrent layer whose matrix products are replaced by convolutions | keeps the image structure inside the recurrence (the lecturer's example on Nov 10, from Homework 6) | the hidden state is then a feature map, not a vector |

## How past exams asked

| Question | Answer | Row used |
|---|---|---|
| An MLP needs inputs of one fixed size ([[9-midterm-2/fall-2024#problem-1--true-or-false-30-pts\|F24 1(b)]]) | True | parameters depend on input size |
| New images are $128\times128$ instead of $64\times64$: which of MLP, CNN, ViT still work? ([[9-midterm-2/fall-2025#problem-5--mlp-cnn-or-vision-transformer-23-pts\|F25 5(c)]]) | CNN and ViT | input size |
| Which can relate the top-left and bottom-right pixels in the first layer? ([[9-midterm-2/fall-2025#problem-5--mlp-cnn-or-vision-transformer-23-pts\|F25 5(d)]]) | MLP and ViT | what one first-layer output sees |
| Permuted pixels make an MLP noticeably worse ([[9-midterm-2/spring-2026#problem-1--fact-or-cap-30-pts\|Sp26 1(b)]]) | False | weights shared across nothing |
| MLPs and CNNs drop equally under a 2-pixel shift ([[9-midterm-2/spring-2026#problem-1--fact-or-cap-30-pts\|Sp26 1(c)]]) | False | weights shared across positions |
| RNNs are feed-forward models ([[9-midterm-2/fall-2024#problem-1--true-or-false-30-pts\|F24 1(d)]]) | False | core operation |
| Each conv weight is used once; different kernels for different regions ([[9-midterm-2/spring-2025#problem-1--fact-or-cap-30-pts\|Sp25 1(c), 1(j)]]) | False, False | weights shared across positions |

## Answering "which model, and why"

> [!recipe] Template
> 1. **Say what the data is**: a fixed vector, a grid, an ordered sequence, a set of tokens. One length or many?
> 2. **Name the property of the model that fits or fails**: fixed input size, local windows, step-by-step state, all-pairs attention, shared weights.
> 3. **Draw the consequence for this task** in one sentence.
>
> Example ([[9-midterm-2/fall-2025#problem-5--mlp-cnn-or-vision-transformer-23-pts|F25 5(c)]], MLP): "The image is flattened to a vector of $3\cdot128\cdot128$ numbers. The MLP's first layer is a matrix with $3\cdot64\cdot64$ columns. The product is not defined, so a new model is needed."

A "can" answer and an "appropriate" answer are different. An MLP **can** classify images; a CNN is **appropriate** because local kernels and shared weights use what is known about images (Lecture 15, FashionMNIST: 88.8% for a small CNN against 84.7% for the MLP). A Transformer **can** read a short fixed vector; nothing is gained by it.

> [!exam] Said in class
> The stress on *appropriate* is his: know not just which models you can use for a task but which are appropriate, and why. On Nov 10, 2026 he repeated it for the final project: for a given task, for example with audio, there may be reason to consider an MLP, an RNN, a CNN, a Transformer, or a combination, and he pointed to the homework question that puts a convolution inside an RNN for sequences of images. The same day he listed the exam's main topics as exactly these four families. See [[9-midterm-2/index|Midterm 2]].

> [!trap] Common mistakes
> - **"CNNs work on any image size"** without the condition: only with global pooling (or another size-independent head) before the classifier.
> - **"Transformers understand order."** Only through the positional encoding you add.
> - **"MLPs cannot relate distant pixels."** They relate all pixels at once; what they lack is weight sharing and size flexibility.
> - **"RNNs process the sequence in parallel."** Training is sequential in time; that is a main reason Transformers replaced them.
> - **Answering with the newest model** instead of the property the question is about.

> [!question]- Check yourself
> 1. You get sensor readings as 200 numbers per example, in no meaningful order. Which family, and why not a CNN?
> 2. A CNN ends with `nn.Flatten()` and `nn.Linear(64*8*8, 10)`. Can it classify images twice as large? What one change makes it able to?
> 3. Name one thing an RNN has built in that a Transformer has to be given, and one thing a Transformer can do in training that an RNN cannot.
>
> > [!solution]- Answers
> > 1. An MLP. A CNN's kernel assumes neighbouring entries are related and that a pattern means the same at every position; with no meaningful order, neither holds.
> > 2. No: the feature map would be $16\times16$ and the flattened vector 4 times too long for the linear layer. Replace the flatten by global average pooling (`nn.AdaptiveAvgPool2d((1, 1))`, then `nn.Linear(64, 10)`).
> > 3. Built in: the order of the tokens (a Transformer needs positional encoding). In training: process all positions of a sequence in parallel.

Related: [[9-midterm-2/review-attention-and-transformers]] · [[9-midterm-2/review-recurrent-networks]] · [[9-midterm-2/review-training-and-activations]] · [[5-deep-networks/cnn-architectures]]
