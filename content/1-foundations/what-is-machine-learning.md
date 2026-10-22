---
title: "What is machine learning?"
description: "Learning a function from data so it generalizes to unseen inputs: supervised vs unsupervised learning, the four ingredients (dataset, model with parameters θ, loss, learning algorithm), how to classify a problem and name its xᵢ and yᵢ, AI vs ML, and the Lecture 1 ResNet-18 demo (random vs pretrained weights)."
tags: [topic, foundations, midterm-1]
aliases: ["machine learning", "supervised learning", "unsupervised learning", "self-supervised learning", "ingredients of a machine learning problem", "learnable parameters"]
---

*Lecture 1 (with the "ingredients" recap from Lecture 10) · [[1-foundations/index|Foundations]] · next: [[1-foundations/tensors|tensors]]*

> [!abstract] In one breath
> Machine learning fits a model $f_\theta$ to examples so that it **generalizes** to inputs it has never seen. In **supervised** learning every input $x_i$ comes with a label $y_i$, and training repeatedly compares predictions $\hat y_i = f_\theta(x_i)$ with $y_i$ and updates the learnable parameters $\theta$; in **unsupervised** learning there are no labels and assumptions about the data take their place. Every problem in the course is the same four ingredients: a dataset, a model with parameters, a loss, and a learning algorithm (usually gradient descent).

## What it is

> [!definition] Definition — machine learning
> The study of algorithms that **learn patterns and functions from data** such that the learned models **generalize to new, unseen data**. Most problems fit the picture of an unknown process $f$ that maps inputs $x_i$ to outputs $y_i$; the goal is to model that process.

![[l01-function.png|An unknown function f maps inputs x to outputs y]]

The goal is never to do well on the data you already have; it is to do well on data you have not seen. That is why datasets are split into **training**, **validation** and **testing** data (see [[4-pytorch-training/experimental-setup|experimental setup]]).

## What you should know

- The definition above, and the word **generalize**.
- **Supervised**: a dataset of known pairs $\mathcal{D}=\{(x_i, y_i)\}_{i=1}^{N}$; the label $y_i$ is also called the **ground truth** or **annotation**.
- **Unsupervised**: only the $x_i$; prior information or simplifying assumptions guide the model (clustering, generation, self-supervised proxy tasks).
- For any problem statement, say whether it is supervised or unsupervised and name **what $x_i$ is and what $y_i$ is**; that is the exam's version of this topic (HW 1 Q1).
- The four ingredients: dataset $\mathcal D$, model $f_\theta(x)$ with **learnable parameters** $\theta$, loss $\ell(f_\theta(x_i), y_i)$, learning algorithm.
- The subscript $\theta$ marks the parameters that training updates. The course uses **parametric** models (a fixed set of numbers to learn).
- AI is broader than ML: rule-based game NPCs, early chatbots, maze solvers and classical chess engines are AI without learning from data.

## Supervised vs unsupervised

<!-- fig:found-ml-taxonomy -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 230" width="700" height="230" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>machine learning splits into supervised and unsupervised, with examples</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><rect x="270" y="10" width="160" height="36" rx="6" fill="none" stroke="currentColor" stroke-width="1.6"/><text x="350" y="28" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">Machine learning</text><rect x="40" y="90" width="270" height="120" rx="6" fill="none" stroke="var(--accent)" stroke-width="1.6"/><text x="175" y="108" text-anchor="middle" dominant-baseline="central" fill="var(--accent)" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">Supervised: labels yᵢ known</text><text x="175" y="128" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">classification (cat vs dog)</text><text x="175" y="145" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">regression (price, rating)</text><text x="175" y="162" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">speech-to-text, recommendation</text><rect x="390" y="90" width="270" height="120" rx="6" fill="none" stroke="var(--hi)" stroke-width="1.6"/><text x="525" y="108" text-anchor="middle" dominant-baseline="central" fill="var(--hi)" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">Unsupervised: no labels</text><text x="525" y="128" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">clustering (how many speakers?)</text><text x="525" y="145" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">generation (new faces)</text><text x="525" y="162" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">self-supervised proxy tasks</text><path d="M350 46 L175 88" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#ARR)"/><path d="M350 46 L525 88" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#ARR)"/></svg><figcaption><b>The two settings of Lecture 1.</b> Supervised problems learn from pairs (<i>x<sub>i</sub></i>, <i>y<sub>i</sub></i>); unsupervised ones only see <i>x<sub>i</sub></i> and rely on assumptions about its structure. This course is mostly supervised.</figcaption></figure>

### Supervised learning

Each input $x_i$ has a known label $y_i$, and the collection of pairs is the dataset

$$
\mathcal{D}=\{(x_i, y_i)\}_{i=1}^{N}.
$$

Examples from the lecture: **image classification** (cat or dog), **speech-to-text** (audio in, transcript out), **recommendation** (viewing history in, product out). A model is **trained** (or **fit**) on a training set by iteratively updating it to reduce an error measure, with the real goal of doing well on held-out validation or test data.

![[l01-supervised-learning.png|Supervised learning: inputs go through the model, predictions are compared with labels, and the error updates the parameters θ]]

Read the diagram as the training loop of the whole course: each $x_i$ goes through the same model $f_\theta$, the predictions $\hat y_i$ and the labels $y_i$ meet in the **error measure** (the loss), and the error is turned into **updates to $\theta$**. Later units fill in each box: the loss ([[3-linear-models/linear-regression|MSE]], [[3-linear-models/binary-cross-entropy|cross-entropy]]), the update ([[2-calculus-and-optimization/gradient-descent|gradient descent]]) and how its gradients are computed ([[2-calculus-and-optimization/backpropagation|backpropagation]]).

Two kinds of supervised output appear all semester:

| Problem | $y_i$ is | Example | Course model |
|---|---|---|---|
| **Regression** | a real number (or vector) | price, rating, disease progression | [[3-linear-models/linear-regression\|linear regression]] |
| **Classification** | one of finitely many classes | cat vs dog, digit 0–9 | [[3-linear-models/logistic-regression\|logistic regression]] |

### Unsupervised learning

No labels. Examples from the lecture:

- **Clustering**: group examples into a (possibly unknown) number of groups, for example "how many different people speak across these audio clips?", assuming clips of the same speaker sound alike.
- **Generation**: create new images of faces; there is no single correct answer, but a model can learn what faces have in common from many examples.
- **Self-supervised learning**: train on a **proxy task** whose labels come for free, such as predicting whether an image was rotated by 90°, or filling in deleted words in a sentence (how language models are trained).

The lecture goals also mention **discriminative** vs **generative** models: a discriminative model maps an input to an output (a label or a probability of each label given $x$); a generative model learns what the data itself looks like, well enough to produce new examples.

## Worked: classify the problem, name $x_i$ and $y_i$

> [!example] Lecture 1 exercise
> 1. *Friends send photos of their coffee with the price they paid and a 0–10 rating; you want to guess price and quality from photos posted online.* **Supervised regression.** $x_i$ = a photo, $y_i$ = (price, rating), two real numbers.
> 2. *Your phone groups photos into albums of the same friends or pets, with nobody tagged.* **Unsupervised (clustering).** $x_i$ = a photo (or a face in it); there is no $y_i$.
> 3. *Friends record their pets and tell you whether each is a cat or a dog; you want to classify new clips.* **Supervised classification.** $x_i$ = an audio clip, $y_i \in \{\text{cat}, \text{dog}\}$.

> [!example] HW 1 Q1
> (a) Count how many distinct authors wrote a pile of texts, from writing style: **unsupervised** (clustering, no labels). (b) Identify the language of an audio clip from labeled clips: **supervised**. (c) Predict make or miss from videos of shots with known outcomes: **supervised**.

> [!tip] The test
> Ask: *does the dataset contain the answer I want the model to produce?* If yes, supervised, and that answer is $y_i$. If the "answer" (the number of speakers, the albums) has to be discovered, unsupervised.

## The four ingredients

Lecture 10 names the parts every problem is built from:

| Ingredient | Symbol | Logistic regression example |
|---|---|---|
| Dataset | $\mathcal{D}=\{(x_i, y_i)\}_{i=1}^{N}$ | $x_i\in\mathbb{R}^3$, $y_i\in\{0,1\}$ |
| Model with learnable parameters | $f_\theta(x)$ | $f_\theta(x) = \dfrac{1}{1+e^{-(w^\top x+b)}}$, $\theta=\{w, b\}\in\mathbb{R}^4$ |
| Loss (objective) function | $\ell(f_\theta(x_i), y_i)$ | binary cross-entropy |
| Learning algorithm | | gradient descent with learning rate $\alpha$, gradients by backpropagation |

Minimizing the loss over the dataset with respect to $\theta$ should produce the behaviour you want (or you maximize a reward instead). Keep this table in mind: [[3-linear-models/linear-regression|linear regression]], [[3-linear-models/logistic-regression|logistic regression]], [[5-deep-networks/multilayer-perceptrons|MLPs]] and [[5-deep-networks/convolutional-layers|CNNs]] change the model and the loss, never the recipe.

## The Lecture 1 demo: a pretrained ResNet-18

The lecture ends by classifying a photo with ResNet-18, a [[5-deep-networks/cnn-architectures|convolutional network]] that `torchvision` ships ready-made. Five steps, each of which returns later in the course:

```python
import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

model = torch.hub.load('pytorch/vision:v0.13.0', 'resnet18',
                       weights=ResNet18_Weights.DEFAULT)  # trained weights
model.eval()  # inference mode

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
x = preprocess(Image.open('image5.jpg')).unsqueeze(0)  # batch of 1

with torch.no_grad():                      # no gradients needed
    logits = model(x)                      # shape (1, 1000)
probs = torch.nn.functional.softmax(logits[0], dim=0)
top5_prob, top5_id = torch.topk(probs, 5)
for p, i in zip(top5_prob, top5_id):
    print(categories[i], p.item())     # categories: ImageNet class names
```

```
Pomeranian 0.9978806972503662
Chihuahua 0.00034563057124614716
Pekinese 0.0003118149470537901
chow 0.00020339860930107534
Norwich terrier 0.00019351467199157923
```

- **Preprocessing** turns the image into a float tensor of shape $(3, 224, 224)$ (channels, height, width) normalized with the ImageNet mean and standard deviation; `unsqueeze(0)` adds the batch dimension, giving $(1, 3, 224, 224)$. See [[1-foundations/tensors#adding-and-removing-dimensions|adding dimensions]] and [[5-deep-networks/data-augmentation|transforms]].
- The model outputs **1000 unnormalized scores** (logits), one per ImageNet class; `softmax` turns them into probabilities that sum to 1 ([[3-linear-models/multiclass-logistic-regression|multi-class logistic regression]]).
- `torch.no_grad()` and `model.eval()` switch off what is only needed for training ([[2-calculus-and-optimization/autograd|autograd]], [[5-deep-networks/batch-normalization|batch norm]]).
- Moving `model` and `x` to `'cuda'` when available is the [[4-pytorch-training/gpu-training|GPU]] pattern.

> [!intuition] Parameters are the model
> Loaded **without** `weights=`, the same architecture has randomly initialized parameters. Its top five guesses for the lecture's first photo were orangutan, lacewing, Shih-Tzu, African crocodile and submarine, each with probability about 0.013–0.025: nearly flat scores, effectively random guessing. With trained weights (shown on a second photo, `image5.jpg`) the top class gets 0.998. Same code, same architecture; only $\theta$ differs. Learning is the process of finding good $\theta$.

> [!trap] Common mistakes
> - Calling a problem unsupervised because the labels are "noisy" or self-reported (ratings, prices). If the dataset contains the target, it is supervised.
> - Naming only $x_i$. Exam answers need both $x_i$ and $y_i$, with $y_i$'s type (real number, class).
> - Forgetting that the real objective is performance on **unseen** data, not the training set.

## PyTorch docs

- [PyTorch tutorials: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) — the official walk through tensors, data, models, autograd and training, in course order.
- [`torch.hub.load`](https://docs.pytorch.org/docs/stable/hub.html) — loading published models such as `pytorch/vision`.
- [`torchvision.models.resnet18`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) — the demo model and its `ResNet18_Weights`.
- [`torchvision.models`](https://docs.pytorch.org/vision/stable/models.html) — every pretrained model and its expected preprocessing.
- [`transforms.Compose`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html), [`transforms.Normalize`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html), [`transforms.ToTensor`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html) — the preprocessing pipeline.
- [`torch.nn.functional.softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html), [`torch.topk`](https://docs.pytorch.org/docs/stable/generated/torch.topk.html) — scores to probabilities, and the $k$ largest.
- [`torch.no_grad`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html) — inference without building a graph.

Related: [[1-foundations/tensors]] · [[4-pytorch-training/experimental-setup]] · [[3-linear-models/logistic-regression]] · [[5-deep-networks/cnn-architectures]] · [[lectures|Lecture map]]
