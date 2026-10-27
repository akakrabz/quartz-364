---
title: "Deep networks"
description: "Unit 5 (Lectures 14–17): multilayer perceptrons and activation functions, convolutional networks from the convolution operation to residual and fully convolutional architectures, segmentation, augmentation, batch norm and dropout, the history of CNNs, and recurrent networks (RNN, bidirectional and stacked RNNs, LSTM, GRU)."
tags: [unit, deep-nets]
aliases: ["deep nets", "deep learning", "neural networks", "unit 5"]
---

*Lectures 14–17 · after Midterm 1 · builds on [[3-linear-models/index|linear models]] and [[4-pytorch-training/index|PyTorch training]]*

Logistic regression draws one straight boundary; this unit stacks layers until the model can represent almost anything, and then adds structure that matches the data. An **MLP** composes linear layers with nonlinear activations. A **CNN** exploits images: small filters, shared across positions, pooled into features that tolerate shifts, and later grown into deep branching and residual networks and into fully convolutional networks that label every pixel. An **RNN** exploits sequences: one cell applied step after step, carrying a hidden state, with gates (LSTM, GRU) to remember over long ranges. Every model is still trained with the same loop, loss and optimizer from [[4-pytorch-training/training-loop|Unit 4]].

## Feed-forward networks (Lecture 14)

1. [[5-deep-networks/multilayer-perceptrons|Multilayer perceptrons]] — why logistic regression fails on two moons, hidden layers, an MLP in PyTorch, training on the GPU.
2. [[5-deep-networks/activation-functions|Activation functions]] — sigmoid, tanh, ReLU and friends, and why a stack of linear layers without them is still linear.

## Convolutional networks (Lecture 15)

3. [[5-deep-networks/convolution|Convolution]] — template matching; convolution vs cross-correlation; what `F.conv1d`/`F.conv2d` actually compute.
4. [[5-deep-networks/convolutional-layers|Convolutional layers]] — every `nn.Conv2d` argument, the output-size formula, parameter counts, receptive fields.
5. [[5-deep-networks/pooling|Pooling]] — max and average pooling, flatten vs global-pooling heads, and what happens when the object moves.

## Beyond the chain (Lecture 16)

6. [[5-deep-networks/branching-and-inception|Branching and Inception]] — models as graphs, the Inception block, $1\times1$ bottlenecks, merge rules for `torch.cat` and `+`.
7. [[5-deep-networks/residual-connections|Residual connections]] — why deep plain networks stall, $\textrm{ReLU}(F(x) + x)$, projection shortcuts.
8. [[5-deep-networks/fully-convolutional-networks|Fully convolutional networks]] — upsampling, transposed convolution, encoder–decoder and U-Net.
9. [[5-deep-networks/semantic-segmentation|Semantic segmentation]] — a class for every pixel, and intersection over union.
10. [[5-deep-networks/data-augmentation|Data augmentation]] — torchvision transforms, augmenting images and masks together, when it helps.
11. [[5-deep-networks/batch-normalization|Batch normalization]] — what it computes, its parameters, train vs eval mode.
12. [[5-deep-networks/dropout|Dropout]] — randomly silencing units in training, scaling, train vs eval mode.

## Landmark CNNs (Lectures 16–17)

13. [[5-deep-networks/cnn-architectures|CNN architectures: LeNet to ResNet]] — LeNet-5, ImageNet, AlexNet, VGG, GoogLeNet and ResNet: one idea each, with shapes and parameter counts.

## Recurrent networks (Lecture 17)

14. [[5-deep-networks/recurrent-neural-networks|Recurrent neural networks]] — hidden state, weights shared over time, one-hot letters, the surname classifier, `nn.RNN` shapes.
15. [[5-deep-networks/bidirectional-and-stacked-rnns|Bidirectional and stacked RNNs]] — reading both ways, layers on layers, and the layout of `output` and `h_n`.
16. [[5-deep-networks/lstm-and-gru|LSTM and GRU]] — gates, the cell state, `nn.LSTM`/`nn.GRU`, and parameter counts of all three cells.

> [!tip] Shapes and parameter counts are the through-line
> Almost every page here has the same two skills: trace a tensor's shape through the model, and count its learnable parameters by hand ($C_{\text{out}}C_{\text{in}}k^2 + C_{\text{out}}$ for a conv, $n_{\text{out}}n_{\text{in}} + n_{\text{out}}$ for a linear layer, $G\cdot H(d+H+2)$ for a recurrent layer). Practise them on [[5-deep-networks/convolutional-layers|convolutional layers]], [[5-deep-networks/cnn-architectures|CNN architectures]] and [[5-deep-networks/lstm-and-gru#counting-parameters-of-rnn-gru-and-lstm|recurrent layers]].

Next: [[7-transformers/index|Transformers]] — Lecture 18 replaces one-hot letters with learned embeddings and moves from letters to word tokens.
