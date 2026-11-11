---
title: "Midterm 2 true/false bank"
description: "50 true/false statements for Midterm 2 with folded one-line answers: every in-scope true/false item from the four past Midterm 2 exams (36, marked ★, grouped by topic) plus 14 more statements on points from Lectures 14–21."
tags: [exam, midterm-2]
aliases: ["Midterm 2 T/F bank", "Midterm 2 true false"]
---

*Midterm 2 · true/false was 30 points on every past exam ([[9-midterm-2/fall-2024#problem-1--true-or-false-30-pts|Fall 2024]], [[9-midterm-2/fall-2025#problem-1--true-or-false-30-pts|Fall 2025]], [[9-midterm-2/spring-2025#problem-1--fact-or-cap-30-pts|Spring 2025]], [[9-midterm-2/spring-2026#problem-1--fact-or-cap-30-pts|Spring 2026]]) · ★ = a real past exam item, in scope this semester · the parts the course staff excluded are on [[9-midterm-2/skipped-topics|skipped topics]] · back to [[9-midterm-2/index|Midterm 2]]*

> [!abstract] How to use this page
> Say True or False out loud, then open the answer. There is no partial credit, so a statement is True only if it is **always** true: watch for "always", "must", "only", "regardless", "equal". Of the 36 real items, 18 are False.

## Data, optimizers and training ★

**1.** ★ PyTorch `Dataset` classes have three required methods that must be implemented: `__init__`, `__len__` and `forward`. *(Fall 25)*
> [!solution]- Answer
> **False.** The third is `__getitem__`; `forward` belongs to `nn.Module`. → [[4-pytorch-training/datasets-and-dataloaders|Datasets and DataLoaders]]

**2.** ★ PyTorch `DataLoader` objects allow us to load batches of multiple examples at a time during a training epoch. *(Fall 25)*
> [!solution]- Answer
> **True.**

**3.** ★ A training epoch is one full pass of model updates through the training dataset. *(Fall 24)*
> [!solution]- Answer
> **True.** $\lceil N/\text{batch size}\rceil$ iterations make one epoch.

**4.** ★ Momentum in stochastic gradient descent optimizers takes a weighted average of current and previous gradients during update steps. *(Fall 25)*
> [!solution]- Answer
> **True.** $v_t = \mu v_{t-1} + g_t$ keeps every earlier gradient with weight $\mu^k$. → [[4-pytorch-training/optimizers#sgd-with-momentum|momentum]]

**5.** ★ Dropout is a technique for preventing overfitting in deep net models. *(Fall 24)*
> [!solution]- Answer
> **True.** → [[5-deep-networks/dropout|dropout]]

**6.** ★ During evaluation (`model.eval()`), both Dropout and Batch Normalization are effectively turned off, meaning neither layer has any effect on the forward pass. *(Spr 26)*
> [!solution]- Answer
> **False.** Dropout does nothing in eval mode, but batch norm still normalizes, using its stored running mean and variance.

**7.** ★ Batch norm is applied to normalize the color channels of input images to convolutional neural networks. *(Fall 24)*
> [!solution]- Answer
> **False.** It normalizes feature maps **inside** the network, per channel over the batch. Normalizing the input image is preprocessing. → [[5-deep-networks/batch-normalization|batch norm]]

**8.** ★ In deep learning training, batch normalization can address the problem caused by internal covariate shift. *(Spr 26)*
> [!solution]- Answer
> **True** (key). That is the batch-norm paper's own motivation: the inputs of a layer keep changing distribution as earlier layers are updated.

## MLPs and activations ★

**9.** ★ A given multi-layer perceptron model requires inputs of a specific size, i.e. the size of the data cannot vary in the dataset during training or testing. *(Fall 24)*
> [!solution]- Answer
> **True.** The first `nn.Linear(in_features, …)` fixes the input length. → [[9-midterm-2/review-choosing-models|choosing a model]]

**10.** ★ In deep NNs, having multiple linear layers back to back without an activation function in between is mathematically no different than having one linear layer. *(Spr 25)*
> [!solution]- Answer
> **True.** $W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2)$.

**11.** ★ The sigmoid activation function can cause vanishing gradients in deep networks. *(Spr 25)*
> [!solution]- Answer
> **True.** $\sigma'\le\tfrac14$, and near 0 when the unit saturates. → [[5-deep-networks/activation-functions#vanishing-gradients-and-dead-units|vanishing gradients]]

**12.** ★ The derivative of ReLU is always 1. *(Spr 25)*
> [!solution]- Answer
> **False.** 1 for $x>0$, 0 for $x<0$.

**13.** ★ A multi-layer perceptron trained on FashionMNIST will achieve noticeably worse accuracy if you first apply a fixed random permutation to every image's pixels (the same permutation for all images) before training. *(Spr 26)*
> [!solution]- Answer
> **False.** An MLP has no notion of neighbouring pixels (Lecture 15: 84.7% vs 84.0%). → [[5-deep-networks/convolution#when-the-assumption-is-wrong-scrambled-pixels|scrambled pixels]]

## Convolutional networks ★

**14.** ★ Pooling functions in convolutional neural networks must be non-linear functions. *(Fall 24)*
> [!solution]- Answer
> **False.** Average pooling is linear.

**15.** ★ In a convolutional layer, each weight (or parameter) is used exactly once while computing the output of the layer. *(Spr 25)*
> [!solution]- Answer
> **False.** Each kernel weight is used at every output position (weight sharing).

**16.** ★ A convolutional layer applies different kernels (filters) to different regions of the input. *(Spr 25)*
> [!solution]- Answer
> **False.** Every filter is applied to every region.

**17.** ★ In a 2D convolutional layer, setting padding $P = (K-1)/2$ with stride 1 always guarantees that the output feature map has the same height and width as the input, regardless of the kernel size $K$. *(Spr 26)*
> [!solution]- Answer
> **False.** Only for odd $K$; for even $K$ the padding is not a whole number.

**18.** ★ Global average pooling in image classification CNN models transforms a shape $(C, H, W)$ collection of feature maps into a length-$C$ vector. *(Fall 25)*
> [!solution]- Answer
> **True.** One average per channel. → [[5-deep-networks/pooling#the-head-flatten-or-global-pooling|global pooling]]

**19.** ★ When test images are shifted 2 pixels to the right, MLPs and CNNs suffer an equal drop in accuracy, because neither architecture was explicitly trained on shifted data. *(Spr 26)*
> [!solution]- Answer
> **False.** Lecture 15: MLP 86.8% → 46.1%; CNN with global pooling 78.0% → 74.4%. → [[5-deep-networks/pooling#the-shift-experiment|the shift experiment]]

## Recurrent networks and sequences ★

**20.** ★ Recurrent neural networks are not an example of feed-forward deep net models. *(Fall 24)*
> [!solution]- Answer
> **True.** The hidden state feeds back into the layer at the next time step.

**21.** ★ Standard RNNs are designed to handle sequential data by maintaining a hidden state across time steps. *(Spr 25)*
> [!solution]- Answer
> **True.**

**22.** ★ Long short-term memory networks achieve long-term and short-term time-dependencies by applying short and long skip connections between nearby and far away model layers, respectively. *(Fall 24)*
> [!solution]- Answer
> **False.** By a gated **cell state** inside the recurrent layer. Skip connections between layers are ResNets. → [[5-deep-networks/lstm-and-gru|LSTM]]

**23.** ★ LSTMs suffer more from the vanishing gradient problem than standard RNNs do. *(Spr 25)*
> [!solution]- Answer
> **False.** Less; that is what the gates are for.

**24.** ★ Bi-directional RNN layers use the same learnable parameters to process data forwards and backwards in a sequence. *(Fall 25)*
> [!solution]- Answer
> **False.** Separate weights per direction; the count doubles.

**25.** ★ In sequence transduction tasks, an input sequence is transformed into an output sequence. The lengths of the input and output sequences must be the same. *(Fall 25)*
> [!solution]- Answer
> **False.** They may differ; the decoder stops at `<EOS>`.

## Text, attention and Transformers ★

**26.** ★ Suppose we have a vocabulary of 64 different words for a text classification model and we encode each word with one-hot encoding. Each encoded vector will be length 8. *(Fall 25)*
> [!solution]- Answer
> **False.** Length 64, one entry per word.

**27.** ★ Consider a learnable embedding module for text data, i.e. `nn.Embedding`. As the vocabulary size increases, we must also increase the dimension of embedded vectors. *(Fall 25)*
> [!solution]- Answer
> **False.** Rows and width are independent arguments.

**28.** ★ Tokenization can segment language by letters, words or something in between. *(Spr 26)*
> [!solution]- Answer
> **True.** → [[7-transformers/tokenization|tokenization]]

**29.** ★ Self-attention compares every token to every other token in the sequence. *(Spr 25)*
> [!solution]- Answer
> **True.** $QK^\top$ is $N\times N$.

**30.** ★ In the attention mechanism, queries, keys, and values are all derived from the same input in self-attention. *(Spr 26)*
> [!solution]- Answer
> **True.**

**31.** ★ In decoder self-attention, a causal mask prevents tokens from attending to future positions. *(Spr 25)*
> [!solution]- Answer
> **True.** $-\infty$ is added above the diagonal of the scores.

**32.** ★ In decoder of Transformer, queries, and keys used in cross-attention are derived from encoder. *(Spr 26)*
> [!solution]- Answer
> **False.** Queries from the decoder; keys **and values** from the encoder.

**33.** ★ Decoder-only transformer models have no cross-attention mechanism. *(Spr 25)*
> [!solution]- Answer
> **True.** There is no encoder output to attend to.

**34.** ★ Multi-head attention uses multiple attention heads to allow the model to capture different types of relationships in parallel. *(Spr 26)*
> [!solution]- Answer
> **True.**

**35.** ★ The "Add & Norm" layers in the original Transformer model use batch norm to normalize activations within the model. *(Fall 25)*
> [!solution]- Answer
> **False.** Layer norm. → [[7-transformers/layer-norm-and-feed-forward#layer-norm-versus-batch-norm|layer norm vs batch norm]]

**36.** ★ Positional encoding helps Transformer models understand the order and relative positions of tokens from an input. *(Fall 25)*
> [!solution]- Answer
> **True.** → [[7-transformers/positional-encoding|positional encoding]]

## More statements in scope

**37.** `nn.Conv2d(3, 16, 5)` has more parameters when it is applied to $224\times224$ images than to $32\times32$ images.
> [!solution]- Answer
> **False.** $16(3\cdot25+1) = 1{,}216$ either way; the image size is not in the count.

**38.** A max pooling layer has no learnable parameters.
> [!solution]- Answer
> **True.**

**39.** Two stacked $3\times3$ convolutions with stride 1 have a $5\times5$ receptive field.
> [!solution]- Answer
> **True.** $3 + (3-1)\cdot1 = 5$. → [[9-midterm-2/review-sizes-and-parameters#receptive-field|receptive field]]

**40.** A CNN that ends in global average pooling followed by a linear layer can classify images of different sizes with the same weights.
> [!solution]- Answer
> **True.** The pooled vector always has length $C$.

**41.** `nn.RNN(10, 20)` has more parameters when the input sequences are longer.
> [!solution]- Answer
> **False.** 640 for any length: the weights are reused at every step.

**42.** An LSTM keeps two states, a hidden state and a cell state.
> [!solution]- Answer
> **True.**

**43.** Doubling `nhead` in `nn.MultiheadAttention(128, nhead)` doubles its number of parameters.
> [!solution]- Answer
> **False.** $4\cdot128^2 + 4\cdot128$ for any valid number of heads.

**44.** Without positional encoding, shuffling the input tokens of a Transformer encoder shuffles its outputs in the same way and changes nothing else.
> [!solution]- Answer
> **True.** Self-attention is permutation-equivariant. → [[7-transformers/positional-encoding#attention-does-not-see-order|attention does not see order]]

**45.** In a Transformer classifier with a `<CLS>` token, the class scores are computed from the average of all output tokens.
> [!solution]- Answer
> **False.** From the output at the `<CLS>` position only. → [[7-transformers/text-classification-with-transformers|text classification]]

**46.** A $64\times64$ RGB image cut into $16\times16$ patches gives a Vision Transformer 16 patch tokens of length 768.
> [!solution]- Answer
> **True.** $(64/16)^2 = 16$ and $3\cdot16\cdot16 = 768$.

**47.** With teacher forcing, the decoder is fed its own previous predictions during training.
> [!solution]- Answer
> **False.** It is fed the true previous target tokens. → [[7-transformers/sequence-to-sequence-rnns#teacher-forcing-and-lining-up-the-loss|teacher forcing]]

**48.** The causal mask of a Transformer decoder is needed only at inference time.
> [!solution]- Answer
> **False.** It is essential in training, where the whole target is fed in at once.

**49.** `nn.CrossEntropyLoss` expects the model to output probabilities, so the model should end with a softmax.
> [!solution]- Answer
> **False.** It expects logits and applies log-softmax itself.

**50.** A residual (skip) connection adds a block's input to the block's output.
> [!solution]- Answer
> **True.** → [[5-deep-networks/residual-connections|residual connections]]

Related: [[9-midterm-2/review-training-and-activations]] · [[9-midterm-2/review-choosing-models]] · [[9-midterm-2/review-attention-and-transformers]] · [[9-midterm-2/practice-problems]] · [[6-midterm-1/true-false]]
