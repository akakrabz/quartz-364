---
title: "ECE 364 · Programming Methods for Machine Learning"
description: "Review notes for ECE 364 (Fall 2026, Illinois), organized by topic: tensors, matrix calculus, gradient descent and backpropagation, linear and logistic regression, PyTorch training, MLPs, CNNs and RNNs, embeddings and tokenization — each page with the must-know summary, diagrams, code from the lecture notebooks and links to the PyTorch docs — plus a Midterm 1 section with every past exam worked."
tags: [unit]
---

Notes for **ECE 364 — Programming Methods for Machine Learning** (University of Illinois, Fall 2026), built from the lecture notebooks. They are organized **by topic, not by lecture**: each page explains one idea, lists what you should know about it, and shows the math, a diagram and the PyTorch code. Every API mentioned links to the [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html).

> [!tip] How to use this site
> - Start a topic from its **unit page** below, or go from a lecture to its topics with [[lectures|Lectures → topics]].
> - Every topic page opens with **In one breath** (the whole idea) and **What you should know** (the checklist to review), and ends with the **PyTorch docs** it uses.
> - The **[[6-midterm-1/index|Midterm 1]]** section has every past Midterm 1 worked problem by problem, the formula sheet and a true/false bank.
> - Hover a link for a preview; the graph view (top right) shows how topics connect.

## Course map

### [[1-foundations/index|1 · Foundations]] — Lectures 1–2
What machine learning is, and PyTorch tensors: [[1-foundations/what-is-machine-learning|What is ML]] · [[1-foundations/tensors|Tensors]] · [[1-foundations/storage-and-views|Storage and views]] · [[1-foundations/indexing-and-slicing|Indexing and slicing]] · [[1-foundations/broadcasting|Broadcasting]] · [[1-foundations/linear-algebra-in-pytorch|Linear algebra in PyTorch]]

### [[2-calculus-and-optimization/index|2 · Calculus and optimization]] — Lectures 3–6
Derivatives of vectors and matrices, and how PyTorch computes them: [[2-calculus-and-optimization/gradients-and-partial-derivatives|Gradients]] · [[2-calculus-and-optimization/matrix-calculus|Matrix calculus]] · [[2-calculus-and-optimization/primal-optimization|Primal optimization]] · [[2-calculus-and-optimization/gradient-descent|Gradient descent]] · [[2-calculus-and-optimization/computational-graphs|Computational graphs]] · [[2-calculus-and-optimization/backpropagation|Backpropagation]] · [[2-calculus-and-optimization/autograd|Autograd]] · [[2-calculus-and-optimization/fitting-models-with-gradient-descent|Fitting models with GD]]

### [[3-linear-models/index|3 · Linear models]] — Lectures 7–9, 11
Regression and classification with models linear in their weights: [[3-linear-models/linear-regression|Linear regression]] · [[3-linear-models/multiple-linear-regression|Multiple linear regression]] · [[3-linear-models/basis-function-regression|Basis-function regression]] · [[3-linear-models/computational-cost|Computational cost]] · [[3-linear-models/ridge-regression|Ridge regression]] · [[3-linear-models/logistic-regression|Logistic regression]] · [[3-linear-models/binary-cross-entropy|Binary cross-entropy]] · [[3-linear-models/multiclass-logistic-regression|Multi-class logistic regression]]

### [[4-pytorch-training/index|4 · Training with PyTorch]] — Lectures 10–11, 14
The machinery every model uses: [[4-pytorch-training/nn-module|nn.Module]] · [[4-pytorch-training/loss-functions|Loss functions]] · [[4-pytorch-training/optimizers|Optimizers]] · [[4-pytorch-training/learning-rate-schedulers|LR schedulers]] · [[4-pytorch-training/training-loop|The training loop]] · [[4-pytorch-training/datasets-and-dataloaders|Datasets and DataLoaders]] · [[4-pytorch-training/experimental-setup|Experimental setup]] · [[4-pytorch-training/gpu-training|GPU training]]

### [[5-deep-networks/index|5 · Deep networks]] — Lectures 14–17
- **MLPs:** [[5-deep-networks/multilayer-perceptrons|Multilayer perceptrons]] · [[5-deep-networks/activation-functions|Activation functions]]
- **CNNs:** [[5-deep-networks/convolution|Convolution]] · [[5-deep-networks/convolutional-layers|Convolutional layers]] · [[5-deep-networks/pooling|Pooling]] · [[5-deep-networks/cnn-architectures|CNN architectures]] · [[5-deep-networks/branching-and-inception|Branching and Inception]] · [[5-deep-networks/residual-connections|Residual connections]]
- **Dense prediction and training tricks:** [[5-deep-networks/fully-convolutional-networks|Fully convolutional networks]] · [[5-deep-networks/semantic-segmentation|Semantic segmentation]] · [[5-deep-networks/data-augmentation|Data augmentation]] · [[5-deep-networks/batch-normalization|Batch normalization]] · [[5-deep-networks/dropout|Dropout]]
- **Sequences:** [[5-deep-networks/recurrent-neural-networks|Recurrent neural networks]] · [[5-deep-networks/bidirectional-and-stacked-rnns|Bidirectional and stacked RNNs]] · [[5-deep-networks/lstm-and-gru|LSTM and GRU]]

### [[7-transformers/index|7 · Transformers]] — Lecture 18 onward
How text becomes model input: [[7-transformers/text-encodings-and-embeddings|Text encodings and learned embeddings]] · [[7-transformers/tokenization|Tokenization and vocabulary]]

### [[6-midterm-1/index|6 · Midterm 1]] — past exams, worked
Scope: Lectures 2–9 and HW 1–3.
- **Past exams, worked:** [[6-midterm-1/fall-2024|Fall 2024]] · [[6-midterm-1/spring-2025|Spring 2025]] · [[6-midterm-1/fall-2025|Fall 2025]] · [[6-midterm-1/spring-2026|Spring 2026]]
- **By problem type:** [[6-midterm-1/review-tensors|Tensors and views]] · [[6-midterm-1/review-gradient-descent|Matrix calculus and GD by hand]] · [[6-midterm-1/review-computational-graphs|Computational graphs]] · [[6-midterm-1/review-autograd|Autograd and find-the-bug]] · [[6-midterm-1/review-linear-regression|Linear regression]] · [[6-midterm-1/review-logistic-regression|Logistic regression]]
- **Drill:** [[6-midterm-1/formula-sheet|Formula sheet]] · [[6-midterm-1/true-false|True/false bank]] · [[6-midterm-1/practice-problems|Practice problems]] · [[6-midterm-1/spring-only-topics|Spring-only topics]] · [[6-midterm-1/errata|Typos in the sources]]
