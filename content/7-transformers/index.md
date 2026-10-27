---
title: "Transformers"
description: "Unit 7 (Lecture 18 onward): how text becomes model input — encodings, learned embeddings with nn.Embedding, tokens and vocabularies — the groundwork for the transformer lectures."
tags: [unit, transformers]
aliases: ["unit 7"]
---

*Lecture 18 onward · after [[5-deep-networks/index|Deep networks]]*

The transformer lectures start where the RNN lectures stopped: with the question of how a model should **read** its input. Lecture 18 ("Transformers I: Words, Tokenization, and Vocabulary") replaces hand-made one-hot letters with **learned embeddings**, then moves from letters to **tokens** and a **vocabulary**, and applies the same `EmbeddingRNN` to sentiment analysis of Amazon review titles. Attention and the transformer architecture itself come in the following lectures; pages for them will be added here.

1. [[7-transformers/text-encodings-and-embeddings|Text encodings and learned embeddings]] — one-hot vs binary vs a hand-designed 2-D code vs `nn.Embedding` on the surname task, why a one-hot input times a matrix is already an embedding, gradients only for used rows, and the embedding-dimension sweep (validation peaks at 62.7% for $d = 64$).
2. [[7-transformers/tokenization|Tokenization and vocabulary]] — character, n-gram and word tokens; information per token against vocabulary size; text → index → vector; the sentiment set-up with `min_freq` (vocabulary sizes computed from the course data) and the notebook's preprocessing quirks.

**Lecture 18 at a glance.** Lecture 17's surname data cut from 18 languages to eight, and its `ElmanRNN` with 128 hidden units (Lecture 17 used 64). Validation mean category accuracy: one-hot 53.1%, binary 49.9%, custom 2-D 37.1%, learned 8-d embedding 58.9% (single runs). Then word tokens for review titles, vocabulary of about 1,700 words at `min_freq = 100`, bidirectional `EmbeddingRNN` with $d = 64$.
