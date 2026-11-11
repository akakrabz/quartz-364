---
title: "Midterm 2"
description: "Everything for ECE 364 Midterm 2 (Tue Nov 17, 2026, in class): the scope announced in lecture (Homeworks 4–6, Lectures 10–11 and 14–21: MLPs, CNNs, RNNs, Transformers), the format and formula sheet, where the points have been on four past exams, a one-hour plan, the four practice exams worked problem by problem with the excluded parts marked, six review pages by problem type, a true/false bank, new practice problems, a formula sheet, and every exam hint given in class."
tags: [exam, midterm-2]
aliases: ["midterm two", "exam 2", "second midterm", "MT2", "Midterm 2"]
---

*Fall 2026 Midterm 2 is **Tue Nov 17, 2026, in class during the regular lecture time** · scope **HW 4–6 and Lectures 10–11, 14–21** · a formula sheet is provided · review lecture Thu Nov 12 · built from the four practice Midterm 2 exams with solutions, the class recordings up to Nov 10, the lecture notebooks and HW 4–6 · see also [[6-midterm-1/index|Midterm 1]]*

> [!abstract] In one breath
> Four practice exams, two templates, one constant: **30 points of true/false** every time. The **Fall** exams (Prof. Snyder, who writes this semester's exam) take one model family per problem and ask you to reason about it: what shape comes out, how many parameters, what is wrong with this code, which model can do this and why. The **Spring** exams (the other section) are shorter drills (a pool or convolution by hand, a parameter count, a two-unit ReLU network) and one 15-point module to write (attention, then an encoder layer). Almost nothing needs heavy arithmetic. [[9-midterm-2/fall-2025|Fall 2025]] is the closest match: same instructor, same lectures, nothing skipped.

## Logistics, as announced

- **When:** Tuesday, Nov 17, 2026, in class, during the regular lecture time.
- **Review lecture:** Thursday, Nov 12. Hints from it will be added to this page once its recording is available.
- **No new homework before the exam.** Homework 6 was the last one; a Homework 7, if there is one, comes after the break (said after class on Nov 10).

## Scope

Announced in lecture on Nov 10, 2026:

- **Homeworks 4, 5 and 6**, "to the end of transformers back through lecture 14 on multilayer perceptron".
- **Main topics: multilayer perceptrons, convolutional neural networks, recurrent neural networks and Transformers.**
- Asked after class whether that means Lectures 10 to 21, he first said 14 and then agreed: Lectures 10 and 11, skip the next two, then onward, "technically optimizers, datasets, data loaders". So: **Lectures 10–11 and 14–21** (Lecture 12 was the Midterm 1 review, 13 the exam).
- **Lecture 22** (underfitting and overfitting, learning-rate schedulers, TensorBoard) "won't be on your midterm". It is there for the final project.

| Lectures | Unit on this site | Homework |
|---|---|---|
| 10–11 | [[4-pytorch-training/index\|PyTorch training]]: `nn.Module`, optimizers, `Dataset` and `DataLoader`, the training loop | HW4 Q1–Q2 |
| 14 | [[5-deep-networks/multilayer-perceptrons\|MLPs]], [[5-deep-networks/activation-functions\|activations]], [[4-pytorch-training/gpu-training\|GPU training]] | HW4 Q2–Q4 |
| 15–16 | [[5-deep-networks/index\|CNNs]]: convolution, layers, pooling, architectures, segmentation, augmentation, batch norm, dropout | HW5 |
| 17 | [[5-deep-networks/recurrent-neural-networks\|RNNs]], [[5-deep-networks/lstm-and-gru\|LSTM and GRU]] | HW6 Q1 |
| 18–21 | [[7-transformers/index\|Transformers]]: embeddings, tokenization, seq2seq, attention, the Transformer, PyTorch's layers, text classification, Vision Transformer | HW6 Q2–Q3 |

**Not on it:** Lecture 22; unit 8 of this site (diffusion models, beyond the lectures); and the parts of the practice exams on the staff's skip list (PCA, K-Means, GMMs, GANs, object detection, BERT pre-training): see [[9-midterm-2/skipped-topics|skipped topics]].

What he said earlier in the term still stands: Transformers are the last material developed for the homework and this exam; for CNNs the depth of detail is "calibrated to the homeworks"; and the skill he stressed is knowing not just which models you *can* use for a task but which are *appropriate*, and why.

## Format and formula sheet

- **A formula sheet is provided**, "more extensive" than Midterm 1's (Nov 10). His example: the output size of a convolutional layer will be on it, because memorizing that is "a waste of your brain space". His hint: use the sheet as evidence of what you can trust to be given, as opposed to what you should study and make sense of.
- This semester's sheet is not in the course folder. The [[9-midterm-2/fall-2025#the-formula-sheet-that-came-with-the-exam|Fall 2025 sheet]] (same instructor) had PyTorch signatures, the conv output-size formula, the LSTM and GRU equations, all the attention formulas with the mask, the sinusoidal encoding and the Transformer diagram. What it leaves to you is on the [[9-midterm-2/formula-sheet|formula sheet page]].
- Past Fall exams: 80 minutes, 5 problems, 100 points, true/false first. Problem types in the Fall papers: true/false, circle-all-that-apply with a brief justification, fill in missing code, short calculations, short explanations.

## The practice exams, worked

| Exam | Problems | Skip (staff's list) | Page |
|---|---|---|---|
| Fall 2024 (Kamalabadi, Snyder) | T/F · `Dataset`, `DataLoader`, SGD arguments · a custom block as an `nn.Module` · a conv layer and a max pool · activations | 1(a), (i), (j), 5(c) | [[9-midterm-2/fall-2024\|Fall 2024]] |
| Spring 2025 | T/F · underfitting from a loss plot · max pool · MLP shapes and count · ReLU net for $mx+b$ · an attention module | 1(a), (b), (d), (k), (n), 6 | [[9-midterm-2/spring-2025\|Spring 2025]] |
| Fall 2025 (Snyder) | T/F · conv/pool shapes and augmentations · a new RNN layer with a bug · a broken mask and $QK^\top$ by hand · MLP vs CNN vs ViT | nothing | [[9-midterm-2/fall-2025\|Fall 2025]] |
| Spring 2026 | T/F · SGD vs GD paths · a strided convolution · MLP shapes and count · ReLU net for $\lvert x\rvert$ · an encoder layer | 1(d), (e), (g), (h), (i), (k), 6 | [[9-midterm-2/spring-2026\|Spring 2026]] |

Every in-scope problem is restated with a folded worked solution that matches the official key (and says so where a key slips: [[9-midterm-2/errata|errata]]). **The skipped parts are kept too**, each in its place and marked **[skipped]**, with its statement, the official answer and a one-line reason. They are not on our Midterm 2 and are left out of the review pages and the true/false bank.

## Where the points have been

In-scope parts only; points in parentheses.

| Problem type | Fall 24 | Fall 25 | Spr 25 | Spr 26 | Review page |
|---|---|---|---|---|---|
| true/false | Q1 (21 of 30) | Q1 (30) | Q1 (20 of 30) | Q1 (18 of 30) | [[9-midterm-2/true-false\|T/F bank]] |
| sizes, shapes, parameter counts; a pool or conv by hand | Q3(a), Q4 (15) | Q2(a) (5), Q3(a) (8), Q5(a)–(b) (11) | Q3 (10), Q4 (10) | Q3 (10), Q4 (10) | [[9-midterm-2/review-sizes-and-parameters\|sizes and parameters]] |
| write or debug a module; `Dataset`, `DataLoader`, SGD | Q2 (20), Q3(b)–(c) | Q3(b) (7) | Q7 (15) | Q7 (15) | [[9-midterm-2/review-pytorch-modules\|modules and data]] |
| recurrent networks | T/F | Q3 (15) | T/F | – | [[9-midterm-2/review-recurrent-networks\|recurrent networks]] |
| attention and Transformers | – | Q4 (15), T/F | Q7 (15), T/F | Q7 (15), T/F | [[9-midterm-2/review-attention-and-transformers\|attention and Transformers]] |
| explain: activations, loss curves, optimizers, regularizers | Q5(a)–(b) | Q2(b) (12) | Q2 (10), Q5 (10) | Q2 (10), Q5 (10) | [[9-midterm-2/review-training-and-activations\|training and activations]] |
| which model, and why | T/F | Q5(c)–(d) (12) | T/F | T/F | [[9-midterm-2/review-choosing-models\|choosing a model]] |

> [!key] What this tells you
> - **True/false is 30 points on every exam.** Thirty-six real in-scope items are in the [[9-midterm-2/true-false|bank]]; half of them are False.
> - **Every exam has a size-and-count problem and a write-the-module problem.** Both are mechanical once practised.
> - **The Fall exams add reasoning about a new block or a comparison**: a layer the lectures never showed (Fall 2024 Q3, Fall 2025 Q3), a formula that is subtly wrong (Fall 2025 Q4(a)), three models side by side (Fall 2025 Q5).
> - **Transformers arrived on the exams with Fall 2025.** Fall 2024 has none, because its Midterm 2 came before them.

## If you only have an hour

1. **15 min · true/false.** Run the ★ items of the [[9-midterm-2/true-false|T/F bank]] out loud.
2. **15 min · Fall 2025.** [[9-midterm-2/fall-2025#problem-3--a-new-recurrent-layer-15-pts|Q3]], [[9-midterm-2/fall-2025#problem-4--attention-15-pts|Q4]] and [[9-midterm-2/fall-2025#problem-5--mlp-cnn-or-vision-transformer-23-pts|Q5]] on paper, without opening the solutions.
3. **10 min · sizes and counts.** The [[9-midterm-2/review-sizes-and-parameters#counting-parameters|parameter table]], then [[9-midterm-2/practice-problems#p1--a-cnn-layer-by-layer|P1]] and [[9-midterm-2/practice-problems#p4--count-the-parameters|P4]].
4. **10 min · one module.** Write the encoder layer of [[9-midterm-2/spring-2026#problem-7--attention-is-all-you-need-15-pts|Spring 2026 Q7]] from the figure.
5. **10 min · the comparison.** The table on [[9-midterm-2/review-choosing-models#the-comparison|choosing a model]], covering one column at a time.

With more time: the other three exams in the order Fall 2024, Spring 2026, Spring 2025, then the [[9-midterm-2/practice-problems|practice problems]].

## Review by problem type

1. [[9-midterm-2/review-sizes-and-parameters|Sizes, shapes and parameter counts]] — HW4 Q2, HW5 Q1–Q2; Lectures 14–17, 20–21
2. [[9-midterm-2/review-pytorch-modules|Writing modules and data pipelines]] — HW4 Q1–Q2, HW6 Q1; Lectures 10–11 and on
3. [[9-midterm-2/review-recurrent-networks|Recurrent networks]] — HW6 Q1; Lecture 17
4. [[9-midterm-2/review-attention-and-transformers|Attention and Transformers]] — HW6 Q2; Lectures 19–21
5. [[9-midterm-2/review-training-and-activations|Training behaviour and activations]] — HW4, HW5 Q3; Lectures 10–11, 14, 16
6. [[9-midterm-2/review-choosing-models|Choosing a model]] — HW5 Q3, HW6 Q3; all of it

Then: [[9-midterm-2/true-false|true/false bank]] · [[9-midterm-2/practice-problems|practice problems]] · [[9-midterm-2/formula-sheet|formula sheet]] · [[9-midterm-2/skipped-topics|skipped topics]] · [[9-midterm-2/errata|typos in the sources]].

## Everything flagged in class

| Topic | What was said | Page |
|---|---|---|
| Scope | HW 4–6; MLPs, CNNs, RNNs, Transformers; Lectures 10–11 and 14–21 (Nov 10) | [[#scope\|above]] |
| Formula sheet | provided, more extensive than Midterm 1's; conv output size will be on it (Nov 10) | [[9-midterm-2/formula-sheet\|formula sheet]] |
| Lecture 22 | "won't be on your midterm" (Nov 10) | [[4-pytorch-training/underfitting-and-overfitting#from-the-lecture\|Underfitting and overfitting]] |
| Conv layer output shape | "probably would be on the exam": know how to use the formula; also parameter count and receptive field | [[5-deep-networks/convolutional-layers#from-the-lecture\|Convolutional layers]] · [[5-deep-networks/branching-and-inception#from-the-lecture\|Branching]] |
| CNN architectures | detail calibrated to the homeworks | [[5-deep-networks/cnn-architectures#from-the-lecture\|CNN architectures]] |
| Transposed convolution | high-level idea only (learned upsampling), not detailed computation | [[5-deep-networks/fully-convolutional-networks#from-the-lecture\|Fully convolutional networks]] |
| Batch norm | short homework exercise; normalizations compared again with Transformers | [[5-deep-networks/batch-normalization#from-the-lecture\|Batch normalization]] · [[7-transformers/layer-norm-and-feed-forward#from-the-lecture\|LayerNorm]] |
| RNNs | Homework 6: parameters and forward pass | [[5-deep-networks/recurrent-neural-networks#from-the-lecture\|RNNs]] |
| BLEU | "I will never ask you to calculate the BLEU metric by hand": know what it measures and how to read it | [[7-transformers/sequence-to-sequence-rnns#from-the-lecture\|Seq2seq RNNs]] |
| Self-attention | the ingredient every Transformer variant has; Homework 6 practises interpreting attention | [[7-transformers/attention#from-the-lecture\|Attention]] |
| Positional encoding | the notebook section supports Homework 6 | [[7-transformers/positional-encoding#from-the-lecture\|Positional encoding]] |
| Lecture 21 | PyTorch's Transformer layers, the `<CLS>` token and Vision Transformers are the end of the scope; the ViT appeared "a little bit" in Homework 6 | [[7-transformers/pytorch-transformer-layers#from-the-lecture\|Transformer layers]] · [[7-transformers/text-classification-with-transformers#from-the-lecture\|Text classification]] · [[7-transformers/vision-transformer#from-the-lecture\|Vision Transformer]] |
| Model choice | which models are *appropriate* for a task, and why | [[9-midterm-2/review-choosing-models\|Choosing a model]] |

> [!tip] Sources
> All in your ECE 364 folder (not copied into this site): `exam/MT2/ECE_364_Midterm_2_Fall_2024` and `…_Fall_2025` (+ Solutions), `exam/MT2/ECE364_MT2_sp25_final` and `…_sp26_vf` (+ sol), `transcript/` (the recordings of Nov 5 and Nov 10 for the scope and format), `py-complete/Lecture10–22`, and the HW 4–6 problem PDFs. The exam answers were recomputed with `tools/verify_mt2.py` and `tools/verify_mt2_torch.py`.
