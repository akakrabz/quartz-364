---
title: "Tokenization and vocabulary"
description: "Splitting text into tokens and numbering them: character, n-gram and word tokens, the trade-off between information per token and vocabulary size, the token → index → embedding pipeline, and the Lecture 18 sentiment-analysis set-up (Amazon review titles, word tokens, a min_freq vocabulary cut-off, a bidirectional EmbeddingRNN), with vocabulary sizes computed from the course data and the notebook's preprocessing quirks."
tags: [topic, transformers]
aliases: ["tokens", "token", "vocabulary", "vocab", "tokenizer", "sentiment analysis", "min_freq", "Amazon Review Polarity"]
---

*Lecture 18 · [[7-transformers/index|Transformers]] · builds on [[7-transformers/text-encodings-and-embeddings|text encodings and embeddings]]*

> [!abstract] In one breath
> **Tokenization** splits an input into pieces, **tokens**; the set of distinct tokens the model knows is its **vocabulary**, and each token gets an index that selects a row of an [[7-transformers/text-encodings-and-embeddings|embedding]]. Small tokens (characters) give a tiny vocabulary but long sequences of nearly meaningless symbols; big tokens (words) carry meaning but the vocabulary explodes. Lecture 18 uses whole words for **sentiment analysis** of Amazon review titles, keeps only words that occur at least `min_freq` times, and trains the same `EmbeddingRNN` as for surnames. The same idea carries over to images in later lectures, where a token is a patch.

## What it is

> [!definition] Definitions
> - **Token**: one unit of the input sequence the model reads per step — a character, a group of characters, a word, or (for images) a patch.
> - **Tokenization**: the rule that splits raw input into tokens.
> - **Vocabulary**: the set of distinct tokens, each assigned an integer index $0,\dots,V-1$; $V$ is the vocabulary size and the number of rows of the embedding matrix.
> - **Sentiment analysis**: classifying the attitude of a text; here binary, positive or negative.

The surname task already used tokens: each letter was one, from a vocabulary of 52. Sentences change the picture because they are long and because single letters say little: after reading `a`, `w` the model cannot yet tell "awful" from "awesome".

## What you should know

- The definitions of token, tokenization and vocabulary, and the pipeline text → tokens → indices → embedding vectors → model.
- The **trade-off**: larger tokens carry more information each and give shorter sequences, but the vocabulary grows fast — at most $26$, $26^2 = 676$ and $26^3 = 17{,}576$ tokens for lowercase 1-, 2- and 3-letter tokens.
- Each vocabulary entry costs $d$ embedding parameters, so $V\cdot d$ often dominates the model's size.
- What `min_freq` does: it drops rare words from the vocabulary, and here also drops every title that contains one. Raising it shrinks $V$ and the embedding, but also the set of usable examples, and it biases the data toward short titles.
- Out-of-vocabulary tokens have no row in the embedding; something must handle them (drop the example, as here, or map them to a reserved "unknown" index).
- How to read the bidirectional head `(fc(outputs[-1]) + fc(outputs[0]))/2`.
- Tokens are not only for text: an image can be cut into non-overlapping patches, which is where the transformer lectures are headed.

## Token size against vocabulary size

![[l18-tokenization-schemes.png|Character, n-gram and word tokens for "studious" and "student"]]

The lecture's figure tokenizes "studious" and "student" three ways. Character tokens give long sequences over a small alphabet (the blue numbers are 1-based alphabet positions; the last `s` of "studious" should read 19, not 21). N-gram tokens such as `stu` + `di` + `ous` share pieces between related words (`stu` appears in both). Word tokens give one token per word but need an index for every word.

| Token | Vocabulary (lowercase English) | Sequence for "the studious student graduated" | Meaning per token |
| --- | --- | --- | --- |
| character | 26 | 27 letters (plus spaces) | almost none |
| 2-letter | at most $26^2 = 676$ | 15 | a little |
| 3-letter | at most $26^3 = 17{,}576$ | 10 | syllable-like |
| word | tens of thousands (81,305 distinct "words" in the course's review titles) | 4 | a whole word |

Many letter combinations never occur ("zx"), so real vocabularies stay below the $26^k$ bound, but the growth is still exponential in token length.

## From tokens to vectors

![[l18-input-embedding.png|A sentence mapped to vocabulary indices and then to columns of a 512-dimensional embedding]]

Each token is looked up in the vocabulary to get its index, and the index selects a learned vector — here of size 512 — from the embedding table, a matrix with one column (in PyTorch's [`nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html), one row) per vocabulary entry. The model never sees the strings, only these vectors. The lecture's dataset keeps the two directions of the mapping as dictionaries:

```python
def build_vocabulary(self):
    word_to_idx = {}
    idx_to_word = {}
    idx_ctr = 0
    for title in self.titles:
        for word in title:
            if word not in word_to_idx:
                word_to_idx[word] = idx_ctr
                idx_to_word[idx_ctr] = word
                idx_ctr += 1
    return word_to_idx, idx_to_word

def embed_titles(self):
    title_indices = []
    for t in self.titles:
        title_tensor = torch.tensor([self.word_to_idx[word] for word in t]).long()
        title_indices.append(title_tensor.unsqueeze(1))   # (L, 1)
    return title_indices
```

Each title becomes a `long` tensor of shape $(L, 1)$, exactly the shape the surname indices had, so the model code does not change.

## The sentiment-analysis set-up

**Data.** The [Amazon Review Polarity dataset](https://arxiv.org/abs/1509.01626) (Zhang, Zhao and LeCun, 2015): product reviews, each a title and a text, labelled 1 (negative) or 2 (positive). The course uses the 400,000-review test split in `AmazonReviewSentiment/reviews.csv`, and only the **titles**, to keep sequences short.

Running the notebook's own counting code on that file gives:

| Quantity | Value |
| --- | --- |
| rows read by `pd.read_csv(data_path)` | 399,999 (200,000 negative, 199,999 positive — see the trap below) |
| title length, `len(title_split)` (punctuation and the split's empty strings included) | median 5, mean 5.7 |
| review length, words | median 66, mean 74 |
| distinct lowercase title "words" | 81,305 |
| most frequent | `.` (115,089), `!` (110,878), `the`, `a`, `,`, `not`, `great`, `of`, `for`, `this`, … |

Character tokens for a 74-word review would mean sequences of several hundred steps; word tokens on titles mean about five.

**Tokenizing a title.** The notebook's rule, in order: put spaces around `.` `,` `!` `?` so they become their own tokens, split on spaces, lowercase, and delete `'` and `"` (so "don't" becomes `dont`, the 25th most common token).

```python
title_str = re.sub('([.,!?])', r' \1 ', str(title))   # "Great CD!" -> "Great CD ! "
title_split = title_str.split(' ')
words = [w.lower().replace("'", "").replace('"', '') for w in title_split if w != '']
```

**Building the vocabulary.** Count every word over all 399,999 titles read, keep the words with count $\ge$ `min_freq`, keep only titles whose words are **all** kept, shuffle, take the first $N = 10{,}000$ such titles, and index the words that occur in them:

```python
min_freq = 100   # try changing this!
N = 10000
sentiment_dataset = SentimentDataset(min_freq, N)
print('Dataset has vocabulary size of {}.'.format(len(sentiment_dataset.word_to_idx)))
```

The notebook prints no output for this cell, so the table below comes from running its preprocessing on the course file (sampling seeded with `np.random.seed(0)`; the last three columns vary slightly with the seed):

| `min_freq` | words with count $\ge$ `min_freq` | vocabulary $V$ of the 10,000 titles | mean title length | titles scanned to find 10,000 |
| --- | --- | --- | --- | --- |
| 1 | 81,305 | 8,310 | 5.03 | 10,008 |
| 10 | 9,620 | 5,041 | 4.78 | 13,382 |
| 50 | 3,001 | 2,617 | 4.49 | 18,100 |
| **100** (default) | 1,750 | 1,673 | 4.32 | 21,730 |
| 500 | 412 | 411 | 3.70 | 41,681 |
| 1,000 | 222 | 222 | 3.39 | 60,921 |
| 5,000 | 49 | 49 | 2.97 | 276,496 |

<!-- fig:tf-vocab -->
<figure class="cs-fig"><svg role="img" aria-label="Log-log plot of vocabulary size falling as min_freq grows" style="max-width:490px;width:100%;height:auto" stroke-linejoin="round" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 368.24 220.93" xmlns="http://www.w3.org/2000/svg" version="1.1"><defs></defs><g><g><path d="M 0 220.93  L 368.24 220.93  L 368.24 0  L 0 0  L 0 220.93  z " style="fill: none"/></g><g><g><path d="M 48.56 180.73  L 361.04 180.73  L 361.04 8.86  L 48.56 8.86  L 48.56 180.73  z " style="fill: none"/></g><g><g><g><path d="M 62.76 180.73  L 62.76 8.86  " clip-path="url(#fa83064)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><defs><path id="fa83060" d="M 0 0  L 0 3.5  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#fa83060" x="62.76" y="180.73" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="62.76" y="196.09" transform="rotate(-0 62.76 196.09)">1</text></g></g><g><g><path d="M 139.56 180.73  L 139.56 8.86  " clip-path="url(#fa83064)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fa83060" x="139.56" y="180.73" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="139.56" y="196.09" transform="rotate(-0 139.56 196.09)">10</text></g></g><g><g><path d="M 216.36 180.73  L 216.36 8.86  " clip-path="url(#fa83064)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fa83060" x="216.36" y="180.73" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="216.36" y="196.09" transform="rotate(-0 216.36 196.09)">100</text></g></g><g><g><path d="M 293.16 180.73  L 293.16 8.86  " clip-path="url(#fa83064)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fa83060" x="293.16" y="180.73" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="293.16" y="196.09" transform="rotate(-0 293.16 196.09)">1000</text></g></g><g><g><path d="M 346.84 180.73  L 346.84 8.86  " clip-path="url(#fa83064)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fa83060" x="346.84" y="180.73" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="346.84" y="196.09" transform="rotate(-0 346.84 196.09)">5000</text></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="204.8" y="211.09" transform="rotate(-0 204.8 211.09)">min_freq</text></g></g><g><g><g><path d="M 48.56 157.88  L 361.04 157.88  " clip-path="url(#fa83064)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><defs><path id="fa83061" d="M 0 0  L -3.5 0  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#fa83061" x="48.56" y="157.88" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><g style="fill: currentColor" transform="translate(22.2 163.05)"><text><tspan x="0" y="-0.74" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">1</tspan><tspan x="7" y="-0.74" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">0</tspan><tspan x="14.1" y="-5.28" style="font-size: 7.7px; font-family: 'DejaVu Sans'; fill: currentColor">2</tspan></text></g></g></g><g><g><path d="M 48.56 109.36  L 361.04 109.36  " clip-path="url(#fa83064)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fa83061" x="48.56" y="109.36" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><g style="fill: currentColor" transform="translate(22.2 114.53)"><text><tspan x="0" y="-0.74" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">1</tspan><tspan x="7" y="-0.74" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">0</tspan><tspan x="14.1" y="-5.28" style="font-size: 7.7px; font-family: 'DejaVu Sans'; fill: currentColor">3</tspan></text></g></g></g><g><g><path d="M 48.56 60.84  L 361.04 60.84  " clip-path="url(#fa83064)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fa83061" x="48.56" y="60.84" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><g style="fill: currentColor" transform="translate(22.2 65.95)"><text><tspan x="0" y="-0.85" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">1</tspan><tspan x="7" y="-0.85" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">0</tspan><tspan x="14.1" y="-5.39" style="font-size: 7.7px; font-family: 'DejaVu Sans'; fill: currentColor">4</tspan></text></g></g></g><g><g><path d="M 48.56 12.31  L 361.04 12.31  " clip-path="url(#fa83064)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g><g><use xlink:href="#fa83061" x="48.56" y="12.31" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g><g style="fill: currentColor" transform="translate(22.2 17.43)"><text><tspan x="0" y="-0.85" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">1</tspan><tspan x="7" y="-0.85" style="font-size: 11px; font-family: 'DejaVu Sans'; fill: currentColor">0</tspan><tspan x="14.1" y="-5.39" style="font-size: 7.7px; font-family: 'DejaVu Sans'; fill: currentColor">5</tspan></text></g></g></g><g><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="15.56" y="94.8" transform="rotate(-90 15.56 94.8)">vocabulary size</text></g></g><g><path d="M 62.76 16.68  L 139.56 61.65  L 193.24 86.2  L 216.36 97.57  L 270.04 128.05  L 293.16 141.08  L 346.84 172.92  " clip-path="url(#fa83064)" style="fill: none; stroke: var(--muted); stroke-width: 2; stroke-linecap: square"/><defs><path id="fa83062" d="M 0 2  C 0.53 2 1.04 1.79 1.41 1.41  C 1.79 1.04 2 0.53 2 0  C 2 -0.53 1.79 -1.04 1.41 -1.41  C 1.04 -1.79 0.53 -2 0 -2  C -0.53 -2 -1.04 -1.79 -1.41 -1.41  C -1.79 -1.04 -2 -0.53 -2 0  C -2 0.53 -1.79 1.04 -1.41 1.41  C -1.04 1.79 -0.53 2 0 2  z " style="stroke: var(--muted)"/></defs><g clip-path="url(#fa83064)"><use xlink:href="#fa83062" x="62.76" y="16.68" style="fill: var(--muted); stroke: var(--muted)"/><use xlink:href="#fa83062" x="139.56" y="61.65" style="fill: var(--muted); stroke: var(--muted)"/><use xlink:href="#fa83062" x="193.24" y="86.2" style="fill: var(--muted); stroke: var(--muted)"/><use xlink:href="#fa83062" x="216.36" y="97.57" style="fill: var(--muted); stroke: var(--muted)"/><use xlink:href="#fa83062" x="270.04" y="128.05" style="fill: var(--muted); stroke: var(--muted)"/><use xlink:href="#fa83062" x="293.16" y="141.08" style="fill: var(--muted); stroke: var(--muted)"/><use xlink:href="#fa83062" x="346.84" y="172.92" style="fill: var(--muted); stroke: var(--muted)"/></g></g><g><path d="M 62.76 64.74  L 139.56 75.27  L 193.24 89.09  L 216.36 98.52  L 270.04 128.1  L 293.16 141.08  L 346.84 172.92  " clip-path="url(#fa83064)" style="fill: none; stroke: var(--accent); stroke-width: 2; stroke-linecap: square"/><defs><path id="fa83063" d="M 0 2  C 0.53 2 1.04 1.79 1.41 1.41  C 1.79 1.04 2 0.53 2 0  C 2 -0.53 1.79 -1.04 1.41 -1.41  C 1.04 -1.79 0.53 -2 0 -2  C -0.53 -2 -1.04 -1.79 -1.41 -1.41  C -1.79 -1.04 -2 -0.53 -2 0  C -2 0.53 -1.79 1.04 -1.41 1.41  C -1.04 1.79 -0.53 2 0 2  z " style="stroke: var(--accent)"/></defs><g clip-path="url(#fa83064)"><use xlink:href="#fa83063" x="62.76" y="64.74" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#fa83063" x="139.56" y="75.27" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#fa83063" x="193.24" y="89.09" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#fa83063" x="216.36" y="98.52" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#fa83063" x="270.04" y="128.1" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#fa83063" x="293.16" y="141.08" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#fa83063" x="346.84" y="172.92" style="fill: var(--accent); stroke: var(--accent)"/></g></g><g><path d="M 48.56 180.73  L 48.56 8.86  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><path d="M 48.56 180.73  L 361.04 180.73  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g><g><path d="M 56.66 153.82  L 65.66 153.82  L 74.66 153.82  " style="fill: none; stroke: var(--muted); stroke-width: 2; stroke-linecap: square"/><g><use xlink:href="#fa83062" x="65.66" y="153.82" style="fill: var(--muted); stroke: var(--muted)"/></g></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="81.86" y="156.97" transform="rotate(-0 81.86 156.97)">words with count ≥ min_freq (all titles)</text></g><g><path d="M 56.66 167.32  L 65.66 167.32  L 74.66 167.32  " style="fill: none; stroke: var(--accent); stroke-width: 2; stroke-linecap: square"/><g><use xlink:href="#fa83063" x="65.66" y="167.32" style="fill: var(--accent); stroke: var(--accent)"/></g></g><g><text style="font-size: 9px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="81.86" y="170.47" transform="rotate(-0 81.86 170.47)">vocabulary of the 10,000 sampled titles</text></g></g></g></g><defs><clipPath id="fa83064"><rect x="48.56" y="8.86" width="312.48" height="171.86"/></clipPath></defs></svg> <figcaption><b>Vocabulary size against <code>min_freq</code></b> for the Lecture 18 <code>SentimentDataset</code> (word tokens of review titles, punctuation split off). Computed here by running the notebook's preprocessing on <code>reviews.csv</code>, with the sampling seeded (<code>np.random.seed(0)</code>), so the lower curve varies a little from run to run. The default <code>min_freq = 100</code> keeps 1,750 words and gives a vocabulary of about 1,700.</figcaption></figure>

Two things to read off. The vocabulary shrinks by orders of magnitude, and with it the embedding ($V\cdot d$ parameters). But the filter "every word must be in the vocabulary" also changes the **data**: at `min_freq = 5000` only titles built from the 49 most common tokens survive, they average under three tokens (titles such as "great !" or "not good ."), and over a quarter of a million titles must be scanned to find 10,000 of them. A small vocabulary makes the task easier to fit and less representative of real reviews.

**Model and training.** The `EmbeddingRNN` from [[7-transformers/text-encodings-and-embeddings#learning-the-embedding|the embedding page]], now bidirectional, with `num_embeddings = sentiment_dataset.vocab_size`, `embedding_dim = 64`, 128 hidden units, 2 output classes, SGD (`lr=3e-3`, `momentum=0.99`) for 25 epochs on a 50/50 split. With $V = 1{,}673$:

| Part | Parameters |
| --- | --- |
| `nn.Embedding(1673, 64)` | 107,072 |
| bidirectional `nn.RNN(64, 128)` | $2\,(128\cdot64 + 128\cdot128 + 2\cdot128) = 49{,}664$ |
| `nn.Linear(256, 2)` | 514 |
| total | 157,250 — the embedding is 68% of it |

> [!intuition] The bidirectional head
> `outputs[t]` concatenates the forward state after reading tokens $0..t$ and the backward state after reading tokens $t..L-1$. So `outputs[-1]` has the **full** title in its forward half but only the last token in its backward half, and `outputs[0]` is the mirror image. The model applies the same `fc` to both and averages, so the full-title forward state (from `outputs[-1]`) and the full-title backward state (from `outputs[0]`) each reach the prediction once, averaged through the same weights with the single-token states from the other end. (See [[5-deep-networks/bidirectional-and-stacked-rnns|bidirectional RNNs]] for the layout.)

**The exercise.** (a) Vary `min_freq` and see how accuracy changes with vocabulary size. (b) Fix the vocabulary and vary `embedding_dim`: does accuracy saturate as in the [[7-transformers/text-encodings-and-embeddings#choosing-the-embedding-dimension|surname sweep]]? (c) Bonus: list the 10 nearest embeddings of a word with [`torch.cdist`](https://docs.pytorch.org/docs/stable/generated/torch.cdist.html) and judge whether they are meaningful. The notebook leaves these cells without saved results, so this page gives none; when you run them, compare **validation** accuracy, and remember that changing `min_freq` changes the examples as well as $V$, so the runs are not on the same data.

## Beyond words

Tokenization is not specific to text. In the upcoming transformer lectures an image is split into non-overlapping **patches**, each patch is a token, and the same trade-off appears: big patches give short sequences of informative tokens, small patches give long sequences. Real language models also use **subword** tokenizers that learn a vocabulary of frequent character groups, the n-gram column of the figure; the lecture does not cover how those are built.

> [!trap] Common mistakes and notebook quirks
> - **Tokenizing two ways.** Counting (`load_reviews`) splits with `r' \1'`, filtering (`filter_reviews`) with `r' \1 '`. Words glued to punctuation come out differently: in "print carriage jam...arrghh!!" counting sees `.arrghh` but filtering looks up `arrghh`, and "101,000" is counted as `101` + `,000` but filtered as `000`. That is why even `min_freq = 1` rejects titles — 8 of the first 10,008 in the run above, every one of them for this reason. Use one tokenizer function everywhere.
> - **Losing the first row.** `pd.read_csv(path)` treats the first line as a header, so one review becomes the column names: 399,999 rows instead of 400,000. Use `header=None` for a file without a header.
> - **A global instead of an attribute.** `trim_words` compares counts with the global `min_freq`, not `self.min_freq`; it only works because the variable is defined in the notebook before the dataset is built.
> - **No-op string methods.** The first exploration cell calls `lower_word.replace("'", "")` without assigning the result; Python strings are immutable, so nothing is removed there (the `SentimentDataset` version assigns it correctly).
> - **Empty tokens.** Splitting on single spaces turns double spaces into `''` tokens (6,752 of them in the counts); the dataset drops them with `if w != ''`, but only after the all-words-kept check, which looks up `''` too. And `w != ''` tests the token **before** quotes are stripped, so a lone `"` or `'` token passes, becomes `''`, and enters the vocabulary as a word: at `min_freq = 1000` and `5000` the vocabulary equals the full kept set, `''` included (checked on the course data).
> - **Very large `min_freq`.** The filter's `r' \1 '` padding puts a `''` next to almost every `.` `,` `!` `?`, so once `min_freq` exceeds the count of `''` (6,752 here) nearly every title with punctuation is rejected. If fewer than `N` titles qualify, `__len__` still returns `N`, and indexing past the end fails.
> - **Out-of-vocabulary words at test time.** A word without an index cannot be embedded; here such titles are removed, which is only possible because we choose the data.
> - **Comparing runs across `min_freq`.** Different `min_freq` values select different titles, so accuracy differences mix vocabulary size with data difficulty.

> [!question]- Check yourself
> 1. A model uses word tokens with $V = 50{,}000$ and $d = 256$. How many parameters are in its embedding, and how many would a 3-letter-token embedding with the same $d$ need at most (lowercase letters only)?
> 2. Tokenize the title `Don't buy this!!` with the lecture's rule (the `filter_reviews` version, which pads punctuation on both sides).
> 3. Why does raising `min_freq` make the average title shorter?
>
> > [!solution]- Answers
> > 1. $50{,}000\cdot256 = 12{,}800{,}000$; at most $17{,}576\cdot256 = 4{,}499{,}456$.
> > 2. `["dont", "buy", "this", "!", "!"]`: each `!` is padded and split off, the apostrophe is deleted, and the empty strings from the double spaces are dropped.
> > 3. A title survives only if every one of its words is frequent; each extra word is another chance to contain a rare one, so long titles are filtered out more often.

## PyTorch docs

- [`torch.nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) — the vocabulary-sized lookup table; `padding_idx` reserves an index whose vector stays zero.
- [`torch.utils.data.Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset), [`Subset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Subset) and [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) — the custom `collate_fn` returns a plain list because titles have different lengths.
- [`torch.nn.utils.rnn.pad_sequence`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html) — the usual alternative to one-sequence-at-a-time loops for variable-length batches.
- [`torch.nn.RNN`](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html) — `bidirectional=True` doubles the output features.
- [`torch.nn.CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — two-class sentiment with logits of shape $(1, 2)$.
- Tutorials: [Word Embeddings: Encoding Lexical Semantics](https://docs.pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html), [NLP from scratch: classifying names with a character-level RNN](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html).

Related: [[7-transformers/text-encodings-and-embeddings]] · [[5-deep-networks/recurrent-neural-networks]] · [[5-deep-networks/bidirectional-and-stacked-rnns]] · [[4-pytorch-training/datasets-and-dataloaders]] · [[4-pytorch-training/experimental-setup]]
