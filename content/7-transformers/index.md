---
title: "Transformers"
description: "Unit 7 (Lectures 18–20): how text becomes model input (embeddings, tokens), sequence-to-sequence RNNs for translation, attention, and the Transformer built up from a Skinny to a Full model, plus a guide to the Attention Is All You Need paper."
tags: [unit, transformers]
aliases: ["unit 7"]
---

*Lectures 18–20 · after [[5-deep-networks/index|Deep networks]]*

The transformer lectures start where the RNN lectures stopped: with the question of how a model should **read** its input. Lecture 18 ("Transformers I: Words, Tokenization, and Vocabulary") replaces hand-made one-hot letters with **learned embeddings**, then moves from letters to **tokens** and a **vocabulary**, and applies the same `EmbeddingRNN` to sentiment analysis of Amazon review titles. Lecture 19 ("Transformers II") sets up English→French translation with sequence-to-sequence RNNs, shows their three limitations, and introduces **attention** and a stripped-down "Skinny" Transformer. Lecture 20 ("Transformers III") adds the remaining pieces — feed-forward networks, residual Add & Norm, positional encoding and multi-head attention — to reach the full model of the paper.

1. [[7-transformers/text-encodings-and-embeddings|Text encodings and learned embeddings]] — one-hot vs binary vs a hand-designed 2-D code vs `nn.Embedding` on the surname task, why a one-hot input times a matrix is already an embedding, gradients only for used rows, and the embedding-dimension sweep (validation peaks at 62.7% for $d = 64$).
2. [[7-transformers/tokenization|Tokenization and vocabulary]] — character, n-gram and word tokens; information per token against vocabulary size; text → index → vector; the sentiment set-up with `min_freq` (vocabulary sizes computed from the course data) and the notebook's preprocessing quirks.
3. [[7-transformers/sequence-to-sequence-rnns|Sequence-to-sequence RNNs]] — sequence transduction, the English→French data, encoder–decoder GRUs with a context vector, teacher forcing, greedy decoding, the BLEU score, and the three limitations that motivate attention.
4. [[7-transformers/attention|Attention]] — queries, keys and values; scaled dot-product self-attention; the causal mask; cross-attention between decoder and encoder.
5. [[7-transformers/transformer-architecture|The Transformer architecture]] — encoder and decoder, parallel training vs autoregressive inference, and the Skinny → Medium → Full progression with its results.
6. [[7-transformers/layer-norm-and-feed-forward|LayerNorm, residuals and feed-forward networks]] — Add & Norm, the position-wise FFN, LayerNorm vs BatchNorm.
7. [[7-transformers/positional-encoding|Positional encoding]] — why attention needs positions, sinusoidal and learned encodings, and the ablation.
8. [[7-transformers/multi-head-attention|Multi-head attention]] — several smaller heads in parallel, concatenation and $W_O$, cost and shapes.
9. [[7-transformers/attention-is-all-you-need|Paper guide: Attention Is All You Need]] — Vaswani et al. (2017) section by section, its training recipe and results, and how the course implementation differs.

**Lecture 18 at a glance.** Lecture 17's surname data cut from 18 languages to eight, and its `ElmanRNN` with 128 hidden units (Lecture 17 used 64). Validation mean category accuracy: one-hot 53.1%, binary 49.9%, custom 2-D 37.1%, learned 8-d embedding 58.9% (single runs). Then word tokens for review titles, vocabulary of about 1,700 words at `min_freq = 100`, bidirectional `EmbeddingRNN` with $d = 64$.

**Lectures 19–20 at a glance** (validation BLEU, mean of the last five evaluations of the saved runs; BLEU on a 0–1 scale). Lecture 19, 9,976 sentence pairs (`min_freq = 200`): sequence-to-sequence GRU 0.27, with teacher forcing 0.33, bidirectional encoder 0.37, both 0.36; Skinny Transformer 0.35 (0.37 when trained twice as long). Lecture 20, 37,756 pairs (`min_freq = 50`), 1M training steps: Skinny 0.26, Medium 0.38, Full 0.45; after 3M steps 0.34, 0.46 and 0.51. Positional-encoding ablation of the Full model: sinusoidal 0.45, learned 0.44, none 0.40. The two lectures use different data, so compare within a lecture, not across.

Next: [[8-diffusion-models/index|Diffusion models]] (beyond the lectures) — image generators whose denoiser is often a Transformer over image patches, conditioned on text through cross-attention.
