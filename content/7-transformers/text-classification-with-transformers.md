---
title: "Text classification with a Transformer encoder"
description: "Lecture 21's encoder-only Transformer for sentiment analysis: why classification needs no decoder, three ways to turn a variable number of token vectors into one prediction, the <CLS> token borrowed from BERT, the full-review Amazon data set and its 60% majority baseline, the lecture's model and training code, a positional-encoding quirk in the notebook, parameter counts and results."
tags: [topic, transformers]
aliases: ["CLS token", "class token", "encoder-only Transformer", "Transformer text classifier", "sentiment analysis with a Transformer"]
---

*Lecture 21 · [[7-transformers/index|Transformers]] · builds on [[7-transformers/pytorch-transformer-layers|PyTorch's Transformer layers]] and [[7-transformers/tokenization|tokenization]]*

> [!abstract] In one breath
> Translation needs an encoder to read and a decoder to write. Classification only needs to read, so the model is an **encoder alone** followed by a small prediction layer. The catch is that an encoder returns one vector **per token**, and reviews have different lengths. The standard fix, taken from BERT, is to put a reserved **`<CLS>` token** at the front of every input and predict from the encoder's output at that one position. Self-attention lets that position collect information from the whole text, and training teaches it to.

## What it is

A [[7-transformers/transformer-architecture|sequence-transduction]] Transformer maps a sequence to another sequence. Many tasks want something smaller from a text: one label (sentiment, topic), one number (a star rating), or a summary vector. For those, the decoder has no job. What is left is

$$
\text{tokens}\;\xrightarrow{\ \text{embedding + positions}\ }\;\text{encoder layers}\;\longrightarrow\;\text{one vector}\;\xrightarrow{\ \texttt{nn.Linear}\ }\;\text{class scores}.
$$

The only design question is the arrow in the middle: how to get **one** vector out of $L$ token vectors when $L$ changes from input to input.

<!-- fig:tf-d-cls-token -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 330" width="700" height="330" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>Five tokens starting with CLS pass through an encoder; only the first output feeds a linear layer</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><rect x="68" y="282" width="84" height="30" rx="6" fill="var(--hi)" fill-opacity="0.14" stroke="var(--hi)" stroke-width="1.4"/><text x="110" y="297" text-anchor="middle" dominant-baseline="central" fill="var(--hi)" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">&lt;CLS&gt;</text><text x="110" y="322" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">position 0</text><line x1="110" y1="280" x2="110" y2="243" stroke="var(--hi)" stroke-width="1.5" stroke-linecap="round"/><path d="M110.0,243.0 L113.5,250.0 L106.5,250.0 Z" fill="var(--hi)"/><line x1="110" y1="158" x2="110" y2="130" stroke="var(--hi)" stroke-width="1.5" stroke-linecap="round"/><path d="M110.0,130.0 L113.5,137.0 L106.5,137.0 Z" fill="var(--hi)"/><rect x="80" y="100" width="60" height="28" rx="6" fill="var(--hi)" fill-opacity="0.14" stroke="var(--hi)" stroke-width="1.4"/><text x="110" y="114" text-anchor="middle" dominant-baseline="central" fill="var(--hi)" style="font-size:12.5px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">z[0]</text><rect x="188" y="282" width="84" height="30" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="230" y="297" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">love</text><text x="230" y="322" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">position 1</text><line x1="230" y1="280" x2="230" y2="243" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M230.0,243.0 L233.5,250.0 L226.5,250.0 Z" fill="currentColor"/><line x1="230" y1="158" x2="230" y2="130" stroke="var(--muted)" stroke-width="1.5" stroke-linecap="round"/><path d="M230.0,130.0 L233.5,137.0 L226.5,137.0 Z" fill="var(--muted)"/><rect x="200" y="100" width="60" height="28" rx="6" fill="var(--paper)" stroke="var(--muted)" stroke-width="1.4"/><text x="230" y="114" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12.5px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">z[1]</text><rect x="308" y="282" width="84" height="30" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="350" y="297" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">these</text><text x="350" y="322" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">position 2</text><line x1="350" y1="280" x2="350" y2="243" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M350.0,243.0 L353.5,250.0 L346.5,250.0 Z" fill="currentColor"/><line x1="350" y1="158" x2="350" y2="130" stroke="var(--muted)" stroke-width="1.5" stroke-linecap="round"/><path d="M350.0,130.0 L353.5,137.0 L346.5,137.0 Z" fill="var(--muted)"/><rect x="320" y="100" width="60" height="28" rx="6" fill="var(--paper)" stroke="var(--muted)" stroke-width="1.4"/><text x="350" y="114" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12.5px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">z[2]</text><rect x="428" y="282" width="84" height="30" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="470" y="297" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">shoes</text><text x="470" y="322" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">position 3</text><line x1="470" y1="280" x2="470" y2="243" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M470.0,243.0 L473.5,250.0 L466.5,250.0 Z" fill="currentColor"/><line x1="470" y1="158" x2="470" y2="130" stroke="var(--muted)" stroke-width="1.5" stroke-linecap="round"/><path d="M470.0,130.0 L473.5,137.0 L466.5,137.0 Z" fill="var(--muted)"/><rect x="440" y="100" width="60" height="28" rx="6" fill="var(--paper)" stroke="var(--muted)" stroke-width="1.4"/><text x="470" y="114" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12.5px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">z[3]</text><rect x="548" y="282" width="84" height="30" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="590" y="297" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">!</text><text x="590" y="322" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:10.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">position 4</text><line x1="590" y1="280" x2="590" y2="243" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M590.0,243.0 L593.5,250.0 L586.5,250.0 Z" fill="currentColor"/><line x1="590" y1="158" x2="590" y2="130" stroke="var(--muted)" stroke-width="1.5" stroke-linecap="round"/><path d="M590.0,130.0 L593.5,137.0 L586.5,137.0 Z" fill="var(--muted)"/><rect x="560" y="100" width="60" height="28" rx="6" fill="var(--paper)" stroke="var(--muted)" stroke-width="1.4"/><text x="590" y="114" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12.5px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">z[4]</text><rect x="50" y="160" width="600" height="82" rx="10" fill="var(--accent)" fill-opacity="0.06" stroke="var(--accent)" stroke-width="1.4"/><text x="350" y="185" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">embedding + positional encoding</text><text x="350" y="207" text-anchor="middle" dominant-baseline="central" fill="var(--accent)" style="font-size:12.5px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">n_layers × nn.TransformerEncoderLayer</text><text x="350" y="227" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">self-attention: every token mixes with every other, so z[0] can collect the whole review</text><line x1="110" y1="98" x2="110" y2="66" stroke="var(--hi)" stroke-width="1.5" stroke-linecap="round"/><path d="M110.0,66.0 L113.5,73.0 L106.5,73.0 Z" fill="var(--hi)"/><rect x="25" y="34" width="170" height="30" rx="6" fill="var(--paper)" stroke="var(--hi)" stroke-width="1.4"/><text x="110" y="49" text-anchor="middle" dominant-baseline="central" fill="var(--hi)" style="font-size:12.5px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">nn.Linear(d_emb, 2)</text><line x1="197" y1="49" x2="240" y2="49" stroke="var(--hi)" stroke-width="1.5" stroke-linecap="round"/><path d="M240.0,49.0 L233.0,52.5 L233.0,45.5 Z" fill="var(--hi)"/><text x="246" y="49" text-anchor="start" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">logits: negative, positive</text><text x="470" y="82" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11.5px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">the other outputs are not used for the prediction</text></svg><figcaption><b>Classifying with a <code>&lt;CLS&gt;</code> token.</b> A reserved token (vocabulary index 0) is put in front of every review, so a review of <i>L</i> words enters the encoder as <i>L</i> + 1 tokens. After the encoder layers, only the output at position 0 goes to the prediction layer (<code>z_enc[:, 0]</code> in the lecture's code). Training teaches the model to gather what the label needs into that one position.</figcaption></figure>

## What you should know

- A classifier needs only the encoder. No decoder means no causal mask and no cross-attention; every token may attend to every other token.
- `<CLS>` is an ordinary vocabulary entry (index 0 in the lecture) with its own learned embedding. An input of $L$ tokens becomes $L + 1$ tokens.
- The prediction uses only the encoder output at position 0: `self.fc_predict(z_enc[:, 0])`. The other $L$ outputs are computed but not fed to the loss.
- Nothing forces position 0 to summarize the text. It does because the loss is attached there and self-attention gives it access to all tokens.
- The prediction layer can be one `nn.Linear(d_emb, n_classes)` (text classifier) or a small MLP ([[7-transformers/vision-transformer|Vision Transformer]]).
- Alternatives: pad to a fixed length and flatten; average the token vectors.
- The same trick does regression (output size 1) and other "one answer per input" tasks. The name "class token" is a label for humans.
- Always compare accuracy with the **majority-class baseline**: 60.2% here, not 50%.

## From the lecture

*From the class recording, paraphrased.*

- For sentiment there is "no reason for us to have a decoder anymore": encode the text, then predict from the encoded features.
- The problem to solve is variable length. A 10-token review ends as 10 vectors, a 20-token review as 20, and the classifier must give one answer for each.
- He listed three options: force a fixed length (pad short reviews with blank tokens the model learns to ignore) and flatten everything; average the token representations; or reserve a placeholder token whose output carries the prediction. The lecture takes the third, which he called the standard practice popularized by the BERT language model.
- The name is only for readability: "call it whatever you want" (Nov 5). The same reserved token works for regression, for example predicting a star rating as a number, or as the starting point of a summary.
- During training the model learns to funnel what the task needs into that token. Nothing else tells it to.
- Know your trivial baseline. With 60% positive reviews, always answering "positive" scores 60%, so a model at 60% has learned nothing, even though there are two classes. If a model sits at the baseline, suspect a broken computational graph (gradients not reaching some parameters), the learning rate, or the data formatting.
- The data is messy on purpose: full reviews contain typos and merged words. Cleaning text can help, can be unnecessary, and can even hurt, because a little noise sometimes makes a model more robust.
- An observation from his own experiments: making the task "easier" (a smaller, cleaner vocabulary) produced a **larger** gap between training and validation accuracy; a bigger vocabulary with more examples generalized better.

> [!tip] Shortcut — the baseline to beat
> Majority-class accuracy is the share of the largest class. Here $60.19\%$ of 37,647 reviews are positive (22,660 reviews), so "always positive" is right 60.19% of the time. For $K$ balanced classes the baseline is $1/K$: 25% for the four dog breeds of the [[7-transformers/vision-transformer|Vision Transformer]] example.

> [!intuition] Analogy — a marker like `<SOS>`
> In translation every sentence was wrapped in `<SOS>` and `<EOS>` so the model knew where text starts and ends. `<CLS>` is the same kind of reserved marker with a different job: it is the seat where the answer is collected.

> [!exam] Said in class
> Nov 10, 2026: this is the model behind the Homework 6 text-classification exercise (Yahoo Answers), and Lecture 21 is inside the Midterm 2 scope ([[9-midterm-2/index|Midterm 2]]).

## Three ways to get one vector

| Option | How | Cost |
| --- | --- | --- |
| Pad and flatten | pad every input to a fixed length $L_{max}$, concatenate all $L_{max}$ outputs, `Linear(L_max * d_emb, n_classes)` | a large first prediction layer; a hard length limit; padding tokens must be masked or learned around |
| Average | mean of the $L$ output vectors, then `Linear(d_emb, n_classes)` | none in parameters; every token counts equally, whether informative or not |
| **`<CLS>` token** | prepend a reserved token; `Linear(d_emb, n_classes)` on output 0 | one extra token per input and one extra embedding row; attention decides what matters |

Averaging is the Transformer version of [[5-deep-networks/pooling#global-max-pooling-best-match-anywhere|global pooling]] in a CNN, and of using the last hidden state of an [[5-deep-networks/recurrent-neural-networks#many-to-one-and-many-to-many|RNN]] for a many-to-one task. All three are answers to the same question: how to go from many feature vectors to one.

## The data: full Amazon reviews

[[7-transformers/tokenization#the-sentiment-analysis-set-up|Lecture 18]] classified review **titles** with a `min_freq` filter on words. Lecture 21 uses the review **text**, keeps every review shorter than 25 words, and keeps every word:

```python
max_length = 25
N = 50000                      # cap on sampled reviews
sentiment_dataset = SentimentDataset(max_length, N=N)
```

```
Dataset has vocabulary size of 64432.
Dataset has 37647 examples.

Example from dataset:
Review: ['anyone', 'that', 'likes', 'a', 'good', 'romance', 'book', 'would', 'enjoy', 'this', 'book.', 'could', 'almost', 'be', "anyone's", 'family.', 'will', 'look', 'for', 'more', 'by', 'this', 'author.']
Label (0 = negative, 1 = positive): tensor([1])
Review words embedded as indices: tensor([[   0,  997,   51, 1278,   39,   36, 2048,   33,  533,  197,  100,  117,
          276,   65,   64, 7754, 4471,  144, 1536,    5,  364,   90,  100, 3721]])
39.8093% negative reviews, 60.1907% positive reviews.
```

Things to read off this output:

- The index tensor starts with **0**: the data set prepends `<CLS>` (`word_to_idx = {'<CLS>': 0}`) to every review. The 23 words become 24 indices, shape `(1, 24)`.
- The vocabulary has 64,432 entries for 37,647 reviews, about 1.7 new "words" per review. One reason is visible in the example: `'book'` and `'book.'` are different tokens. The data set computes a punctuation-separated string but then splits the original text on spaces, so punctuation stays glued to words, on top of the typos the lecturer mentioned.
- The classes are unbalanced: 60.2% positive. That is the baseline.
- Half of the reviews (18,823) are used for training and the rest for validation.

## The model

```python
class TransformerTextClassifier(nn.Module):
    def __init__(self, num_embeddings, d_emb, n_layers, n_head, n_classes, dropout=0.1):
        super(TransformerTextClassifier, self).__init__()
        d_ff = 2*d_emb
        self.positional_encoding = PositionalEncoding('learned', d_emb)
        self.encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_emb, n_head, d_ff,
                                                                        dropout=dropout,
                                                                        batch_first=True)
                                             for n in range(n_layers)])
        self.embedding = nn.Embedding(num_embeddings, d_emb)
        self.fc_predict = nn.Linear(d_emb, n_classes)

    def forward(self, x):
        z_enc = self.embedding(x).squeeze(-2)
        z_enc = z_enc + self.positional_encoding(x)   # positional embedding
        for layer in self.encoder_layers:
            z_enc = layer(z_enc)
        preds = self.fc_predict(z_enc[:, 0])          # token 0 is <CLS>
        return preds
```

It is the [[7-transformers/pytorch-transformer-layers#the-encoder-rewritten|encoder of the translation model]] with two changes: `batch_first=True`, and a prediction layer applied to one position. Shapes for the example review, with $d_{emb} = 64$:

| Step | Shape |
| --- | --- |
| `x` (indices, `<CLS>` first) | `(1, 24)` |
| `self.embedding(x)` | `(1, 24, 64)` |
| after the encoder layers | `(1, 24, 64)` |
| `z_enc[:, 0]` | `(1, 64)` |
| `preds` | `(1, 2)`: logits for negative, positive |

`z_enc[:, 0]` reads "every example in the batch, token 0". The model has no softmax: [`nn.CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) takes the logits ([[3-linear-models/multiclass-logistic-regression|multi-class logistic regression]] on top of learned features).

> [!note] A quirk in the notebook: every token gets the same position vector
> `PositionalEncoding` in `'learned'` mode builds its indices with `torch.arange(len(x))`. In the translation model `x` has shape `(N, 1)`, so `len(x)` is the number of tokens, as intended. Here `x` has shape `(1, L+1)`, so `len(x)` is 1: the module returns the single vector for position 0, and broadcasting adds that same vector to every token. As written, this classifier therefore gets **no** information about word order (only `<CLS>` is recognizable, by its own embedding). It still reaches about 80% because sentiment depends mostly on which words appear. Indexing with `torch.arange(x.size(1))`, as the lecture's [[7-transformers/vision-transformer|Vision Transformer]] code does, would give each position its own vector. Confirmed by running the notebook's class definitions (PyTorch 2.15, untrained weights, `tools/verify_mt2_torch.py`): `positional_encoding(x)` has shape `(1, 64)`, and reversing the order of a review's words leaves the two logits unchanged.

## Training and results

```python
cls_transformer = TransformerTextClassifier(num_embeddings, d_emb=64, n_layers=2, n_head=4,
                                            n_classes=2, dropout=0.2).to(DEVICE)
optimizer = torch.optim.SGD(cls_transformer.parameters(), lr=3e-3, momentum=0.99)
n_epochs = 10
```

The training function (`sentiment_train` in the lecture's `train_eval.py`) again simulates a batch of 64 by summing 64 single-review losses before one `backward()`. It switches to `model.eval()` to measure accuracy on 1,000 random training and 1,000 random validation reviews after each epoch, and back to `model.train()` afterwards, because the encoder layers now contain dropout.

Read off the notebook's plot: after the first epoch the accuracy is already about 0.67 (training) and 0.70 (validation), above the 0.60 baseline; after 10 epochs it is about 0.84 and 0.80. The loss is still falling, and the lecturer called the model under-trained. Ten epochs took about 12.5 minutes.

### Where the parameters are

| Part | Parameters |
| --- | --- |
| word embedding, $64{,}432 \times 64$ | 4,123,648 |
| positional table, $1{,}000 \times 64$ | 64,000 |
| 2 encoder layers, $8\cdot64^2 + 11\cdot64 = 33{,}472$ each | 66,944 |
| `fc_predict`, $64\cdot2 + 2$ | 130 |
| **total** | **4,254,722** |

About 97% of the model is the embedding table, and most of its rows belong to words that occur once or twice. This is why [[7-transformers/tokenization#token-size-against-vocabulary-size|vocabulary size]] is the first thing to control in a text model. (Counts derived from the layer definitions; see `tools/verify_mt2.py`.)

## Beyond sentiment

- **Regression:** the same model with `nn.Linear(d_emb, 1)` and an [[4-pytorch-training/loss-functions|MSE loss]] predicts a number, such as a star rating.
- **Other inputs:** anything that can be cut into tokens can be classified this way. Images are next: [[7-transformers/vision-transformer|the Vision Transformer]].
- **Model families:** encoder-only models (like BERT) read; decoder-only models (like the GPT family) generate text with masked self-attention and, having no encoder, **no cross-attention**; encoder–decoder models (the original Transformer) translate.

> [!trap] Common mistakes
> - **Comparing with 50%.** Two classes do not mean a 50% baseline; check the class balance.
> - **Predicting from all positions.** `fc_predict(z_enc)` returns one prediction per token; index position 0 first.
> - **Forgetting `<CLS>` at inference.** The token must be prepended the same way at training and test time.
> - **Indexing the wrong axis.** With `batch_first=True` the `<CLS>` output is `z[:, 0]`; with the default layout it is `z[0]`.
> - **Adding a causal mask.** That belongs to decoders that generate. A classifier should see the whole input.

> [!exam] On the exams
> True/false items on this material: [[9-midterm-2/spring-2025#problem-1--fact-or-cap-30-pts|Spring 2025 1(i), 1(m)]] (self-attention compares every pair of tokens; decoder-only models have no cross-attention). The lecture goals name "the significance of `<CLS>` tokens" explicitly. Drill: [[9-midterm-2/review-attention-and-transformers|attention and Transformers review]] and [[9-midterm-2/review-choosing-models|choosing a model]].

> [!question]- Check yourself
> 1. A review has 17 words. What shape does the encoder output have with $d_{emb} = 64$, and which slice goes to the prediction layer?
> 2. You switch the same data set to five star-rating classes. Which layer changes, and by how many parameters?
> 3. Why can the output at position 0 depend on the last word of the review?
>
> > [!solution]- Answers
> > 1. `(1, 18, 64)` (17 words plus `<CLS>`); `z_enc[:, 0]`, shape `(1, 64)`.
> > 2. Only `fc_predict`: `Linear(64, 5)` has $64\cdot5 + 5 = 325$ parameters instead of 130, so 195 more.
> > 3. In every encoder layer, self-attention lets position 0 take a weighted average over the values of **all** positions, including the last one. There is no mask in an encoder.

## PyTorch docs

- [`torch.nn.TransformerEncoderLayer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) — the encoder layer; `batch_first=True` here.
- [`torch.nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) — the word table (row 0 is `<CLS>`) and the positional table.
- [`torch.nn.ModuleList`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html) — holds the layers.
- [`torch.nn.CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — takes the two logits and the integer label.
- [`torch.nn.Module.eval`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval) and [`train`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train) — switch dropout off and on.

Related: [[7-transformers/pytorch-transformer-layers]] · [[7-transformers/vision-transformer]] · [[7-transformers/tokenization]] · [[7-transformers/attention]] · [[4-pytorch-training/underfitting-and-overfitting]]
