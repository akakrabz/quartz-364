# ECE 364 site — content spec

ECE 364 · Programming Methods for Machine Learning · Fall 2026 · Prof. Snyder (Illinois).
Sources: lecture notebooks in `~/Nextcloud/Notes/ECE 364/py-complete/LectureN/` (Lecture 13 was the midterm),
past Midterm 1 PDFs + solutions in `exam/`, and the earlier Exam 1 guide (`exam/ECE364_Exam1_Guide.html`).
Lecture numbers on the site are the **folder** numbers (Lecture 15 = CNNs part 1, 16 = CNNs part 2, 17 = CNN history + RNNs),
even where a notebook's own header says otherwise.

The site is organized **by topic, not by lecture**: one page per ML topic, grouped into units (folders).
`lectures.md` maps each lecture to its topic pages.

## Layout

```
content/
  index.md                      home: what the site is, course map
  lectures.md                   lecture -> topics table
  images/lNN-*.png              lecture figures (copied from the notebooks' folders)
  1-foundations/                L1, L2
  2-calculus-and-optimization/  L3-L6
  3-linear-models/              L7-L9 (+ multi-class from L11)
  4-pytorch-training/           L10, L11 (+ GPU from L14)
  5-deep-networks/              L14-L17
  6-midterm-1/                  past exams, worked; formula sheet; T/F bank
  7-transformers/               L18 onward (embeddings, tokenization, then attention/transformers)
  8-diffusion-models/           beyond the lectures: diffusion / score-based image generation and newer methods
```

Each unit folder has an `index.md` (title, one-paragraph overview, ordered list of its topic pages with a one-line hook each,
which lectures it covers).

## Topic pages (slugs are fixed: link to these)

1-foundations/
- what-is-machine-learning — supervised vs unsupervised, ingredients (data, model, objective, learning algorithm), pretrained ResNet demo (L1)
- tensors — creating tensors, lists vs NumPy vs tensors, attributes (shape, dtype, device), dtypes and memory size, cat vs stack, unsqueeze/squeeze, NumPy conversion, elementwise math (L2)
- storage-and-views — storage, strides, view vs copy, contiguous, data_ptr, in-place ops (L2)
- indexing-and-slicing — basic slicing, list/tensor indexing, boolean masks, shapes they return (L2, HW1)
- broadcasting — rules, examples (L2)
- linear-algebra-in-pytorch — dot product, norms, angle between vectors, matmul `@`, transpose, mat-vec shapes (L2)

2-calculus-and-optimization/
- gradients-and-partial-derivatives (L3)
- matrix-calculus — gradient w.r.t. vectors, identities like ∇(a^T x), ∇(x^T A x), ∇‖Ax−b‖², Jacobians, checking with PyTorch (L3)
- primal-optimization — critical points, setting derivative to zero, convexity, min vs max (L3)
- gradient-descent — update rule, learning rate, convergence/divergence, by-hand steps (L4)
- computational-graphs — nodes, forward pass, local partials (L5)
- backpropagation — adjoints, backward pass, gradient accumulation over paths (L5)
- autograd — requires_grad, backward(), .grad, zero_(), no_grad, detach, leaf tensors; comparing manual vs autograd (L3, L5, L6)
- fitting-models-with-gradient-descent — objective functions, the GD+autograd loop that fits parameters (quadratic, sigmoid fit) (L6)

3-linear-models/
- linear-regression — 1-D model, MSE objective, closed form from partials, GD solution (L7)
- multiple-linear-regression — matrix form, normal equations, design matrix, many features, disease-progression example (L7)
- basis-function-regression — "kernelized" regression with feature map Φ, linear in the weights, polynomial fits (L8)
- computational-cost — closed form vs gradient descent cost (L8)
- ridge-regression — regularization, λ, outliers, erratic solutions, choosing λ (L8)
- logistic-regression — regression vs classification, why linear regression fails, sigmoid, NLL / max likelihood, decision boundary (L8, L9)
- binary-cross-entropy — BCE = NLL, BCELoss vs BCEWithLogitsLoss (L9)
- multiclass-logistic-regression — softmax, cross-entropy, one-vs-all, CrossEntropyLoss takes logits (L11)

4-pytorch-training/
- nn-module — subclassing, __init__/forward, parameters, nn.Linear, nn.Sequential (L10)
- loss-functions — MSELoss, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss; reduction (L10, L11)
- optimizers — torch.optim, SGD, momentum, weight decay, Adam; zero_grad/step (L10)
- learning-rate-schedulers — StepLR etc., lr adjustments (L10)
- training-loop — the canonical loop, train/eval modes, accuracy, logging (L10, L11)
- datasets-and-dataloaders — Dataset (__len__, __getitem__), TensorDataset, DataLoader, batch_size, shuffle, Subset/random_split, torchvision datasets, transforms (L11)
- experimental-setup — train/validation/test splits, overfitting, what each split is for (L11)
- gpu-training — devices, .to(device), cuda availability, moving model and data (L14)

5-deep-networks/
- multilayer-perceptrons — limits of logistic regression (two moons), hidden layers, universal approximation intuition, MLP in PyTorch (L14)
- activation-functions — sigmoid, tanh, ReLU, (leaky ReLU etc. as covered) and why nonlinearity is needed (L14)
- convolution — template matching, 1-D convolution vs cross-correlation, what F.conv1d/conv2d compute (L15)
- convolutional-layers — nn.Conv2d arguments, channels, output-size formula, parameter counting, receptive field (L15, L16)
- pooling — max/avg pooling, global pooling, flatten vs global-pool heads, translation (in)variance experiment (L15)
- cnn-architectures — LeNet-5, AlexNet, VGG, GoogLeNet/Inception, ResNet history; ImageNet (L16, L17)
- branching-and-inception — models as graphs, Inception block, 1x1 convolutions, multi-head outputs, concatenation shape rules (L16)
- residual-connections — residual block, why deep plain nets fail, projection shortcuts (L16, L17)
- fully-convolutional-networks — upsampling, transposed convolution, encoder-decoder, U-Net (L16)
- semantic-segmentation — per-pixel classification, intersection over union (L16)
- data-augmentation — torchvision transforms, augmenting images and masks, when it helps (L16)
- batch-normalization — what it computes, train vs eval, parameters (L16)
- dropout — what it does, train vs eval, when it helps (L16)
- recurrent-neural-networks — sequences, hidden state, unrolling, nn.RNN shapes, surname-classification example (L17)
- bidirectional-and-stacked-rnns (L17)
- lstm-and-gru — gates, cell state, nn.LSTM/nn.GRU, Elman RNN (L17)

7-transformers/
- text-encodings-and-embeddings — one-hot, binary, custom 2-D, nn.Embedding, embedding-dimension sweep (L18)
- tokenization — tokens, vocabulary, token size vs vocabulary size, sentiment-analysis set-up, min_freq (L18)
- sequence-to-sequence-rnns — sequence transduction, Eng->Fr dataset, encoder-decoder GRU, context vector, teacher forcing, greedy decoding, BLEU, the three limitations (L19)
- attention — queries/keys/values, scaled dot-product self-attention, masked (causal) self-attention, cross-attention, shapes (L19)
- transformer-architecture — encoder/decoder, training vs inference, Skinny -> Medium -> Full progression and results, stacking layers (L19, L20)
- multi-head-attention — heads, per-head dimensions, concatenation and W_O, parameter cost (L20)
- positional-encoding — permutation equivariance, sinusoidal and learned encodings, ablation (L20)
- layer-norm-and-feed-forward — position-wise FFN, residual Add & Norm, LayerNorm vs BatchNorm (L20)
- attention-is-all-you-need — the Vaswani et al. (2017) paper explained section by section, and how the course implementation differs

8-diffusion-models/  (NOT from the lecture notebooks: sourced from the original papers via Exa; every page says so in its meta line)
- brownian-motion-and-forward-diffusion — Brownian motion / Wiener process, the forward noising process (DDPM discrete chain, closed form q(x_t|x_0)), SDE view (VP/VE), noise schedules
- reverse-time-sde-and-score-matching — the score function, Anderson's reverse-time SDE, Langevin dynamics, denoising score matching, the probability-flow ODE
- ddpm — Ho et al. (2020): ε-prediction, the simple loss, training and sampling algorithms, U-Net, a minimal PyTorch implementation
- fast-sampling-ddim-and-ode-solvers — DDIM, probability-flow ODE solvers (DPM-Solver, EDM), distillation and consistency models
- guidance-and-latent-diffusion — classifier and classifier-free guidance, conditioning by cross-attention, latent diffusion / Stable Diffusion, the VAE latent space
- flow-matching-and-rectified-flow — flow matching, rectified flow, the link to diffusion, why recent models use it
- modern-image-generators — newer implementations: diffusion transformers (DiT, MMDiT), Stable Diffusion 3, FLUX and other recent systems, few-step generation (as of the research date, with sources)

6-midterm-1/  (see the exam agent's brief)
- index, fall-2024, spring-2025, fall-2025, spring-2026, formula-sheet, true-false, practice-problems (+ others as needed)

## Page format

```markdown
---
title: "Gradient descent"
description: "One specific sentence: what the page covers (shown in previews and search)."
tags: [topic, calculus, midterm-1]          # topic + unit tag (foundations | calculus | linear-models | training | deep-nets | transformers | diffusion); midterm-1 if in Lectures 2-9 / HW 1-3
aliases: ["gradient descent", "GD", "learning rate"]
---

*Lecture 4 · [[2-calculus-and-optimization/index|Calculus and optimization]] · builds on [[2-calculus-and-optimization/gradients-and-partial-derivatives|gradients]]*

> [!abstract] In one breath
> 2-4 sentences: the whole idea.

## What it is
Description with intuition, then the precise definitions/formulas (KaTeX).

## What you should know
- Checklist of 5-10 concrete, testable things (formulas, shapes, behaviours, API facts).

## ...topic sections...
Derivations, worked examples with numbers, diagrams, code.

> [!trap] Common mistakes
> - ...

> [!exam] On the exams        # only where past exams tested it; link to 6-midterm-1 pages
> ...

## PyTorch docs
- [`torch.optim.SGD`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) — one line on why you'd open it.

Related: [[...]] · [[...]]
```

### Rules
- **Content first.** Concrete numbers, shapes, formulas and worked examples beat prose. Write for review before an exam.
- **Math** in KaTeX: `$...$` inline, `$$...$$` display (display blocks on their own lines). No macros. Inside tables, never a bare `|` in math: use `\lvert`, `\rvert`, `\mid`, `\|` → `\Vert`.
- **Callouts** allowed: definition, key, recipe, trap, exam, intuition, derivation (fold with `-`), example, abstract, tip, note, warning, question, solution, hint, easy, medium, hard. Folded: `> [!solution]-`.
- **Links**: wikilinks with full slugs `[[3-linear-models/ridge-regression|ridge regression]]`; anchors must match headings exactly (github slug rules). Link generously between topics.
- **PyTorch docs**: many links, inline at first mention of an API and in the closing "PyTorch docs" list. Only real APIs, URL patterns:
  - classes/functions/methods: `https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html`, `.../generated/torch.Tensor.view.html`, `.../generated/torch.cat.html`, `.../generated/torch.nn.functional.conv2d.html`, `.../generated/torch.optim.SGD.html`, `.../generated/torch.optim.lr_scheduler.StepLR.html`
  - topic pages: `.../stable/tensors.html`, `tensor_attributes.html`, `tensor_view.html`, `autograd.html`, `notes/autograd.html`, `notes/broadcasting.html`, `nn.html`, `nn.functional.html`, `nn.init.html`, `optim.html`, `data.html`, `notes/cuda.html`
  - `torch.utils.data` classes live on `https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader`
  - torchvision: `https://docs.pytorch.org/vision/stable/transforms.html`, `.../vision/stable/datasets.html`, `.../vision/stable/models.html`, `.../vision/stable/generated/torchvision.transforms.v2.RandomHorizontalFlip.html`
  - tutorials: `https://docs.pytorch.org/tutorials/beginner/basics/intro.html` and siblings (`tensorqs_tutorial`, `data_tutorial`, `buildmodel_tutorial`, `autogradqs_tutorial`, `optimization_tutorial`)
- **Code samples**: `python` fenced blocks, taken from (or closely following) the notebooks, trimmed to the essential lines. Show printed output in a following plain fenced block when it teaches something (copy it from the notebook output; torch is not installed here, so never invent outputs you cannot derive by hand). Code comments ≤ 7 words.
- **Lecture images**: `![[l05-forward-pass.png|Forward pass on the computational graph]]` (files in `content/images/`, see list below). Add a short line of explanation around each.
- **Generated figures** (plots, diagrams): write `tools/figs_<unit>.py` using `tools/mplfig.py` (matplotlib, theme-aware) or `tools/figlib.py` (SVG boxes/arrows/graphs); each figure → `tools/figs/<name>.html`. Put `<!-- fig:<name> -->` on its own line in the page and run `python3 tools/assemble.py content/<page>.md` to inline it. Prefix figure names with the unit, e.g. `calc-gd-steps`. Use numpy for the math; never fabricate data that should come from a trained model.
- Images, figures and `<figure>` blocks: no blank lines inside HTML blocks.
- Headings: `##` for sections, `###` for subsections. No `#` (the title renders from frontmatter).
- Before finishing: `python3 tools/check.py --only <your-folder>` must report no errors on your pages.

## Lecture images available (`content/images/`)
l01-function.png, l01-supervised-learning.png, l02-memory-size-vs-latency.png, l02-vector-angle.png,
l05-backprop-adjoint.png, l05-backprop-full.png, l05-backprop-partial.png, l05-computational-graph.png, l05-forward-pass.png,
l06-quadratic-regression-graph.png, l12-computational-graph.png, l14-mlp-figure.jpg,
l16-1d-convolution-example.png, l16-augmentation-examples.jpg, l16-autoencoder-example.png, l16-convolution-with-multiple-filters2.png,
l16-dropout-example.png, l16-five-layer-cnn.png, l16-l17-center-vs-shifted.png, l16-maxpoolsample2.png, l16-pytorch-conv-2d-example.gif,
l17-alexnet.png, l17-deeper-error.png, l17-gru-overview.png, l17-lenet-5.png, l17-lstm-cell.png, l17-residual-block.png,
l17-resnet-34.png, l17-rnn-bidirectional.png, l17-rnn-unfolded.png, l17-semantic-segmentation-example.png, l17-vgg.png
(The l16-* images come from the folder Lecture16/images; the L15 notebook uses 1d-convolution, maxpool, conv-2d gif, multiple-filters and five-layer-cnn.)

## Notebook text
Plain-text extracts (markdown + code + truncated outputs) are in `/home/claude/work/nb/*.md`
(use `L15_Solution.md` and `L16_Solution.md`, the answer-key versions). Originals: `/mnt/user-data/uploads/ECE 364/py-complete/`.
