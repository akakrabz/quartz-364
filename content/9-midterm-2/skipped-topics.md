---
title: "Skipped topics: what is not on our Midterm 2"
description: "The parts of the four practice Midterm 2 exams that the course staff told this semester's students to skip, listed by exam with the topic and the official answer in one line: one-vs-one classification, PCA, K-Means and Gaussian mixtures, GANs, BERT pre-training and single-shot detection. They are kept in place on the exam pages, marked as skipped; this page collects them so nothing in scope gets confused with them."
tags: [exam, midterm-2]
aliases: ["not on Midterm 2", "skip list", "Midterm 2 skip list"]
---

*Midterm 2 · the course staff's skip list for Fall 2026, as posted with the practice exams · back to [[9-midterm-2/index|Midterm 2]]*

> [!abstract] In one breath
> Three of the four practice exams were written for semesters whose Midterm 2 came later in the term or followed a different syllabus (unsupervised learning, GANs, object detection). The staff's list removes those parts: **31 points of true/false items, one part of Fall 2024's Problem 5, and two whole 15-point problems**, all outside Lectures 10–11 and 14–21. Nothing is skipped on Fall 2025. On this site the skipped parts stay **in their place on each exam page, marked **[skipped]****, with their full statement, the official answer and a one-line reason. They are left out of the review pages, the true/false bank and the practice problems.

## The list

| Exam | Skip | In scope that remains |
|---|---|---|
| [[9-midterm-2/fall-2024\|Fall 2024]] | 1(a), 1(i), 1(j), 5(c) | 1(b)–(h), Problems 2–4, 5(a)–(b) |
| [[9-midterm-2/spring-2025\|Spring 2025]] | 1(a), 1(b), 1(d), 1(k), 1(n), all of Problem 6 | 1(c), (e)–(j), (l), (m), (o); Problems 2–5 and 7 (75 points) |
| [[9-midterm-2/fall-2025\|Fall 2025]] | nothing | all 100 points |
| [[9-midterm-2/spring-2026\|Spring 2026]] | 1(d), 1(e), 1(g), 1(h), 1(i), 1(k), all of Problem 6 | 1(a)–(c), (f), (j), (l)–(o); Problems 2–5 and 7 (73 points) |

## What the skipped parts were about

| Topic | Parts | The answer, in one line |
|---|---|---|
| One-vs-one multi-class classification | F24 1(a) | False: multi-class logistic regression scores every class and takes the largest. (This was [[3-linear-models/multiclass-logistic-regression\|Midterm 1 material]].) |
| Principal component analysis | F24 1(i); Sp25 1(a); Sp26 1(d) | Principal components are orthogonal; the first one has the **largest** variance / eigenvalue. All three statements are False. |
| K-Means and Gaussian mixture models | F24 1(j); Sp25 1(b), Problem 6; Sp26 1(e), 1(g), Problem 6 | K-Means: hard assignments, centroids move to the mean of their points, clusters are round. GMM: soft responsibilities that sum to 1 per point, clusters can be ellipsoids. |
| Generative adversarial networks | F24 5(c); Sp25 1(d); Sp26 1(h), 1(i) | The discriminator is trained to tell real from generated; the generator is trained to fool it; they are trained together. |
| BERT pre-training | Sp25 1(k) | True: masked language modeling. (BERT itself appears in Lecture 21 only as the source of the [[7-transformers/text-classification-with-transformers\|`<CLS>` token]].) |
| Single-shot object detection (SSD) | Sp25 1(n); Sp26 1(k) | Default boxes of several scales and aspect ratios at each feature-map location; the scale differs between feature-map levels. |

> [!warning] Two look-alikes that **are** in scope
> - **Batch norm and "internal covariate shift"** (Spring 2026 1(j)) is not on the skip list, although the phrase is not in this semester's notebooks.
> - **The convolution output-size, pooling and parameter-count questions** on the Spring exams (Problems 3 and 4) are in scope and are exactly Homework 5 material.

Related: [[9-midterm-2/index]] · [[9-midterm-2/true-false]] · [[6-midterm-1/spring-only-topics]]
