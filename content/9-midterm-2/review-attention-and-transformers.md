---
title: "Review: attention and Transformers"
description: "Midterm 2 problem type 4 (15 points on three of four past exams, plus true/false): attention scores by hand and reading who attends to whom, the additive causal mask and why a multiplicative one fails, where queries, keys and values come from in self-, masked and cross-attention, the order of operations in an encoder layer, positional encoding, multi-head facts, encoder-only and decoder-only models, the <CLS> token, and the two modules to be able to write."
tags: [exam, midterm-2, transformers]
aliases: ["attention review", "Transformer review"]
---

*Midterm 2 · problem type 4 of 6 · HW6 Q2 · Lectures 19–21 · 15 pts on Fall 2025, Spring 2025 and Spring 2026 · topic pages: [[7-transformers/attention|attention]], [[7-transformers/transformer-architecture|the Transformer]], [[7-transformers/pytorch-transformer-layers|PyTorch's Transformer layers]] · back to [[9-midterm-2/index|Midterm 2]]*

> [!abstract] In one breath
> Attention is one formula, $\text{softmax}(QK^\top/\sqrt{d_k})\,V$, and the questions take it apart. **$QK^\top$** is a table of scores, row = the token asking, column = the token being looked at. **Softmax along each row** turns scores into weights. **Times $V$** mixes the value vectors with those weights. A **mask** is added to the scores before the softmax. A Transformer layer wraps this in a skip connection and a layer norm, then does the same with a small feed-forward network. Know which tokens supply $Q$, $K$ and $V$ in each kind of attention, and you can answer almost everything asked so far.

## Attention by hand

> [!recipe] Who attends to whom
> 1. Compute $S = QK^\top$: entry $(i, j)$ is (row $i$ of $Q$) · (row $j$ of $K$).
> 2. Token $i$'s scores are **row $i$**. Its largest entry is the token it attends to most.
> 3. Only if weights are asked: divide the row by $\sqrt{d_k}$ and apply softmax. The order of the entries does not change.
> 4. Only if the output is asked: weights × rows of $V$.

| Where | $QK^\top$ | Question | Answer |
|---|---|---|---|
| [[9-midterm-2/fall-2025#problem-4--attention-15-pts\|F25 4(b)]] | rows $(2, -1, 7)$, $(3, -4, 3)$, $(-1, 1, -2)$ | "dog" (token 1) attends most to? | token 3, "cute" (score 7) |

$V$ is given and **not needed**: it decides what is passed along, not who is listened to. Two more complete cases: check-yourself 1 below and [[9-midterm-2/practice-problems#p5--attention-scores-and-a-causal-mask|practice P5]]; a full worked example with weights and outputs is on the [[7-transformers/attention#a-worked-example|attention page]].

## Masks

Masked (causal) self-attention **adds** a matrix to the scores:

$$
\text{softmax}\!\left(\frac{QK^\top+M}{\sqrt{d_k}}\right)V,\qquad M_{ij}=\begin{cases}0&i\ge j\\-\infty&i<j\end{cases}
$$

- $-\infty$ sits **above** the diagonal: token $i$ may look at tokens $1,\dots,i$ only. $e^{-\infty} = 0$, so those weights are exactly 0 and each row still sums to 1.
- It is used in the **decoder's self-attention**, during training too (the whole shifted target is fed in at once, and the mask keeps each position from seeing its own answer).
- **Multiplying** by a mask of 1 and $-\infty$ fails: a negative score times $-\infty$ becomes $+\infty$ and takes all the weight ([[9-midterm-2/fall-2025#problem-4--attention-15-pts|F25 4(a)]]).
- The encoder uses **no** causal mask: every token sees the whole input.

## Who supplies Q, K and V

| Attention | Queries from | Keys and values from | Score table |
|---|---|---|---|
| encoder self-attention | encoder tokens | the same encoder tokens | $N_{enc}\times N_{enc}$ |
| decoder masked self-attention | decoder tokens | the same decoder tokens | $N_{dec}\times N_{dec}$, causal |
| cross-attention (in the decoder) | **decoder** tokens | **encoder** output | $N_{dec}\times N_{enc}$ |

- Self-attention: all three from the same input ([[9-midterm-2/spring-2026#problem-1--fact-or-cap-30-pts|Sp26 1(m)]], True).
- Cross-attention: "queries and keys from the encoder" is **False** ([[9-midterm-2/spring-2026#problem-1--fact-or-cap-30-pts|Sp26 1(n)]]): the decoder asks, the encoder answers.
- Self-attention compares every token with every other ([[9-midterm-2/spring-2025#problem-1--fact-or-cap-30-pts|Sp25 1(i)]], True): that is the $N\times N$ table, and the reason its cost grows as $N^2$.

## Inside a layer

**Encoder layer** (the dashed box of the diagram; [[9-midterm-2/spring-2026#problem-7--attention-is-all-you-need-15-pts|Sp26 7]]):

$$
x\;\to\;x'=\text{Norm}\big(x+\text{SelfAttention}(x)\big)\;\to\;\text{Norm}\big(x'+\text{FeedForward}(x')\big)
$$

**Decoder layer**: masked self-attention, then cross-attention on the encoder output, then feed-forward; each followed by add and norm (three norms).

- "Add & Norm" is a skip connection and a **layer norm**, not batch norm ([[9-midterm-2/fall-2025#problem-1--true-or-false-30-pts|F25 1(f)]]). Layer norm normalizes each token's vector on its own.
- The feed-forward network is two linear layers with a non-linearity, applied to **each token separately** with the same weights.
- Shapes never change inside a layer: $(N, d)$ in, $(N, d)$ out. That is what lets layers be stacked.

## Positional encoding

Self-attention gives the same scores whatever order the tokens come in, so order must be put into the vectors: a position vector is **added** to each token embedding before the first layer ([[9-midterm-2/fall-2025#problem-1--true-or-false-30-pts|F25 1(g)]], True). It can be sinusoidal (fixed; the formula is on the sheet) or a learned embedding indexed by position. An RNN needs none: it reads the tokens one after another, so order is built into the computation. That contrast is Homework 6, Problem 2(c), and it can be seen by running the layers: shuffle the input tokens, and a Transformer encoder layer returns the same output vectors in shuffled order, while an RNN's final state changes (both run in `tools/verify_mt2_torch.py`). → [[7-transformers/positional-encoding|positional encoding]]

## Multi-head attention

Several attention computations ("heads") run in parallel, each with its own projections of $Q$, $K$, $V$ to a smaller width; their outputs are concatenated and passed through $W_O$. Heads can pick up different relationships ([[9-midterm-2/spring-2026#problem-1--fact-or-cap-30-pts|Sp26 1(o)]], True). In PyTorch the number of heads does not change the parameter count, and `d_model` must be divisible by it. → [[7-transformers/multi-head-attention|multi-head attention]]

## Model families and the class token

| Family | Parts | Attention used | Typical task |
|---|---|---|---|
| encoder–decoder | both | self, masked self, cross | translation (sequence transduction) |
| encoder-only | encoder | self (no mask) | classification from a `<CLS>` token; Vision Transformer |
| decoder-only | decoder without an encoder | masked self; **no cross-attention** ([[9-midterm-2/spring-2025#problem-1--fact-or-cap-30-pts\|Sp25 1(m)]]) | text generation |

A classifier needs one vector per input. The standard way: prepend a reserved `<CLS>` token and predict from the encoder's output at that position ([[7-transformers/text-classification-with-transformers|text classification]]). A [[7-transformers/vision-transformer|Vision Transformer]] does the same with image patches as tokens ([[9-midterm-2/review-sizes-and-parameters#vision-transformer-tokens|token counts]]).

## The two modules to be able to write

**Single-head attention with projections** ([[9-midterm-2/spring-2025#problem-7--attention-is-all-you-need-15-pts|Sp25 7]]):

```python
class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, Q, K, V):                       # each (B, N, d_model)
        scores = self.wq(Q) @ self.wk(K).transpose(1, 2) / self.d_model ** 0.5   # (B, N, N)
        weights = torch.softmax(scores, dim=-1)       # each row sums to 1
        return self.wo(weights @ self.wv(V))          # (B, N, d_model)
```

**One encoder layer** from a given attention module ([[9-midterm-2/spring-2026#problem-7--attention-is-all-you-need-15-pts|Sp26 7]]):

```python
def forward(self, x):
    x = self.norm1(x + self.attn(key=x, value=x, query=x))
    x = self.norm2(x + self.ff(x))
    return x
```

with `self.attn`, two separate `nn.LayerNorm(d)` and a feed-forward module created in `__init__`. Lecture 21's [`nn.TransformerEncoderLayer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) is this layer ready-made ([[7-transformers/pytorch-transformer-layers|PyTorch's Transformer layers]]).

> [!trap] Common mistakes
> - **Reading a column instead of a row** of $QK^\top$. Rows are the tokens that ask.
> - **Computing $KQ^\top$.** That is the transpose: rows and columns swap roles.
> - **Softmax over the wrong axis**, or over the whole table.
> - **Mask below the diagonal**, or multiplied in, or applied after the softmax.
> - **Batch norm in Add & Norm**; **one** shared layer norm for both places.
> - **Forgetting the skip connections**, or adding the un-normalized input the second time.
> - **Cross-attention with the roles reversed.**

> [!exam] Said in class
> Self-attention is the ingredient he called the most important, "every Transformer variant has it"; Homework 6 practised interpreting attention; and on Nov 10 he placed "the end of transformers", Lecture 21, inside the Midterm 2 scope. BLEU is never computed by hand. See [[9-midterm-2/index#everything-flagged-in-class|everything flagged in class]].

> [!question]- Check yourself
> 1. $Q = \begin{bmatrix}1&0\\0&2\\1&1\end{bmatrix}$, $K = \begin{bmatrix}2&1\\0&1\\-1&3\end{bmatrix}$. Which token does each token attend to most? With a causal mask and no scaling, what are token 2's weights?
> 2. A decoder has 5 target tokens and the encoder produced 9 vectors. What is the shape of the cross-attention weight matrix?
> 3. Why does the encoder not use a causal mask?
>
> > [!solution]- Answers
> > 1. $QK^\top$ has rows $(2, 0, -1)$, $(2, 2, 6)$, $(3, 1, 2)$: tokens 1, 3, 1. Masked, token 2 keeps $(2, 2)$ and drops the 6: weights $(0.5, 0.5, 0)$.
> > 2. $5\times9$: one row per decoder query, one column per encoder key.
> > 3. The whole source sentence is known before anything is generated, so every source token may use all the others. The mask exists to stop a generator from seeing tokens it has not produced yet.

Related: [[9-midterm-2/review-pytorch-modules]] · [[9-midterm-2/review-choosing-models]] · [[9-midterm-2/formula-sheet]] · [[7-transformers/attention-is-all-you-need]]
