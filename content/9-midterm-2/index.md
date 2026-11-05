---
title: "Midterm 2"
description: "What the lecturer said Midterm 2 covers (the Transformer lectures plus a cumulative review of the deep-network models, with weight on choosing the right model for a task), every exam hint given in class with links to the topic pages, a model-choice table, and a checklist of the calculations to practise."
tags: [exam, midterm-2]
aliases: ["midterm two", "exam 2", "second midterm"]
---

*Built from what was said in the class recordings · no past Midterm 2 papers yet · see also [[6-midterm-1/index|Midterm 1]]*

> [!note] Logistics, as announced in class
> Midterm 2 is **in class, on the Tuesday before Thanksgiving break**. Before it come a lecture on model-training best practices and a **review session on the Thursday** before the exam. Check the course website for the exact date and room; this page only records what was said aloud.

## Scope

- **New material ends with the Transformers.** The lecturer described Transformers as the last material he wanted to develop for the homework and for the second midterm; topics after them are "additional good content", nothing to be rigorous about.
- **Plus a cumulative review of the deep-network models:** [[5-deep-networks/multilayer-perceptrons|MLPs]], [[5-deep-networks/convolutional-layers|CNNs]] and their extensions, [[5-deep-networks/recurrent-neural-networks|RNNs]] / [[5-deep-networks/lstm-and-gru|LSTM and GRU]], [[7-transformers/sequence-to-sequence-rnns|sequence-to-sequence RNNs]] and the [[7-transformers/transformer-architecture|Transformer]].
- **The skill he stressed:** knowing not just which models you *can* use for a task but which ones are *appropriate*, and why. Homework 7 is a summary comparison of all the models, meant as preparation.
- **Homeworks as the calibration.** For CNNs he said the depth of detail is calibrated to the homeworks. Homework 5 covers conv output sizes, parameter counts and receptive fields. Homework 6 covers RNNs (parameters and a forward pass), Transformer questions including interpreting attention, and an RNN-vs-Transformer comparison.

## Everything flagged in class

| Topic | What was said | Page |
| --- | --- | --- |
| Conv layer output shape | "probably would be on the exam": know the formula; also parameter count and receptive field | [[5-deep-networks/convolutional-layers#from-the-lecture\|Convolutional layers]] · [[5-deep-networks/branching-and-inception#from-the-lecture\|Branching]] |
| CNN architectures | detail calibrated to the homeworks | [[5-deep-networks/cnn-architectures#from-the-lecture\|CNN architectures]] |
| Transposed convolution | high-level idea only (learned upsampling), not detailed computation | [[5-deep-networks/fully-convolutional-networks#from-the-lecture\|Fully convolutional networks]] |
| Batch norm | short homework exercise; normalizations compared again with Transformers | [[5-deep-networks/batch-normalization#from-the-lecture\|Batch normalization]] · [[7-transformers/layer-norm-and-feed-forward#from-the-lecture\|LayerNorm]] |
| RNNs | Homework 6: parameters and forward pass | [[5-deep-networks/recurrent-neural-networks#from-the-lecture\|RNNs]] |
| BLEU | "I will never ask you to calculate the BLEU metric by hand" — know what it measures and how to read it | [[7-transformers/sequence-to-sequence-rnns#from-the-lecture\|Seq2seq RNNs]] |
| Self-attention | the ingredient every Transformer variant has; Homework 6 practises interpreting attention | [[7-transformers/attention#from-the-lecture\|Attention]] |
| Positional encoding | the notebook section supports Homework 6 | [[7-transformers/positional-encoding#from-the-lecture\|Positional encoding]] |
| Midterm 2 itself | scope and logistics above | [[7-transformers/transformer-architecture#from-the-lecture\|Transformer architecture]] |

## Which model for which task

*A summary for review, assembled from the topic pages — not a table shown in class.*

| Data / task | First choice | Why | Watch out for |
| --- | --- | --- | --- |
| Fixed-length feature vectors (tabular) | [[5-deep-networks/multilayer-perceptrons\|MLP]] | no spatial or temporal structure to exploit | parameter count grows with input size |
| Images: classify | [[5-deep-networks/convolutional-layers\|CNN]] with [[5-deep-networks/pooling\|pooling]] / global pooling, [[5-deep-networks/residual-connections\|residual blocks]] when deep | local patterns, weight sharing, translation equivariance | flattening an MLP head ties it to one input size and position |
| Images: per-pixel output | [[5-deep-networks/fully-convolutional-networks\|FCN / U-Net]] | output is an image; skip connections keep detail | judge with [[5-deep-networks/semantic-segmentation\|IoU]], not pixel accuracy |
| Sequences, streaming or order-dependent, one label | [[5-deep-networks/recurrent-neural-networks\|RNN]], [[5-deep-networks/lstm-and-gru\|LSTM/GRU]] for long dependencies, [[5-deep-networks/bidirectional-and-stacked-rnns\|bidirectional]] if the whole sequence is available | processes one step at a time with a memory | vanishing gradients; sequential (slow) training |
| Sequence → sequence of different length | [[7-transformers/sequence-to-sequence-rnns\|seq2seq RNN]] → [[7-transformers/transformer-architecture\|Transformer]] | encoder reads, decoder writes until `<EOS>` | single context-vector bottleneck in the RNN version |
| Long-range interactions, large data, parallel training | [[7-transformers/transformer-architecture\|Transformer]] | every token attends to every token; all positions trained at once | needs [[7-transformers/positional-encoding\|positional encoding]]; attention cost grows as $N^2$ |

## Calculations to have at your fingertips

- Conv output size $\lfloor (H + 2p - K)/s\rfloor + 1$, parameters $C_\text{out}(C_\text{in}K^2 + 1)$, receptive field layer by layer — [[5-deep-networks/convolutional-layers|convolutional layers]].
- Linear layer $n_\text{out}(n_\text{in}+1)$ and MLP totals — [[5-deep-networks/multilayer-perceptrons|MLPs]].
- RNN $H(d+H+2)$ per direction and layer; GRU $3\times$, LSTM $4\times$ — [[5-deep-networks/lstm-and-gru|LSTM and GRU]].
- Embedding table $V\cdot d$ — [[7-transformers/text-encodings-and-embeddings|embeddings]].
- Attention shapes: $(N\times H)(H\times N)\to N\times N$ weights, times $N\times d$ values; cross-attention $N_\text{dec}\times N_\text{enc}$; the causal mask — [[7-transformers/attention|attention]].
- Multi-head per-head widths and the cost equal to one full-width head — [[7-transformers/multi-head-attention|multi-head attention]].
