---
title: "PyTorch training"
description: "Unit 4 (Lectures 10, 11 and the GPU part of 14): the PyTorch machinery every model after Midterm 1 is trained with — nn.Module, loss functions, torch.optim optimizers and schedulers, Datasets and DataLoaders, train/validation/test splits, the training loop, and GPUs."
tags: [unit, training]
aliases: ["unit 4", "PyTorch training", "training machinery"]
---

Up to Lecture 9, every model was a handful of loose tensors updated by a hand-written gradient-descent loop. Lectures 10 and 11 replace each piece with the PyTorch object built for it — the **model** becomes an `nn.Module`, the **loss** a `torch.nn` loss object, the **learning algorithm** a `torch.optim` optimizer, and the **dataset** a `Dataset` served in mini-batches by a `DataLoader` — and assemble them into the training loop used for every deep network in the rest of the course. Lecture 14 adds the one extra line that makes it run on a GPU.

These lectures come after the Midterm 1 scope (Lectures 2–9), but the Spring Midterm 1s each had a 10–15 point code question on exactly this material ([[6-midterm-1/spring-2025|Spring 2025]] Q7, [[6-midterm-1/spring-2026|Spring 2026]] Q8).

## Pages, in order

1. [[4-pytorch-training/nn-module|nn.Module]] — a model is a class: `super().__init__()`, `nn.Parameter`, `nn.Linear`, `forward`, `model.parameters()`.
2. [[4-pytorch-training/loss-functions|Loss functions]] — `MSELoss`, `BCELoss`, `BCEWithLogitsLoss`, `CrossEntropyLoss`: what each wants as input and target, and `reduction`.
3. [[4-pytorch-training/optimizers|Optimizers]] — `optimizer.zero_grad()` / `step()`, stochastic gradient descent, momentum and weight decay worked by hand, Adam.
4. [[4-pytorch-training/learning-rate-schedulers|Learning-rate schedulers]] — lowering the learning rate on a plateau or a schedule.
5. [[4-pytorch-training/datasets-and-dataloaders|Datasets and DataLoaders]] — `__len__`, `__getitem__`, batches and their shapes, splits with samplers, torchvision's Fashion-MNIST.
6. [[4-pytorch-training/experimental-setup|Experimental setup]] — training, validation and test sets; hyperparameters; overfitting.
7. [[4-pytorch-training/training-loop|The training loop]] — zero_grad → forward → loss → backward → step, evaluation, logging, gradient accumulation.
8. [[4-pytorch-training/gpu-training|GPU training]] — devices, `.to(device)`, `.cpu()`.

## The whole unit in one block

```python
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

model = nn.Linear(784, 10).to(DEVICE)       # any nn.Module
criterion = nn.CrossEntropyLoss()           # takes logits
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,
                            momentum=0.99, weight_decay=1e-4)

for epoch in range(n_epoch):
    model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        correct = sum((model(x.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item()
                      for x, y in val_loader)
```

## Lectures covered

| Lecture | Topics |
|---|---|
| 10 — `nn.Module`, optimizers, datasets (part 1) | [[4-pytorch-training/nn-module\|nn.Module]], [[4-pytorch-training/loss-functions\|losses]], [[4-pytorch-training/optimizers\|optimizers]], [[4-pytorch-training/learning-rate-schedulers\|schedulers]], [[4-pytorch-training/training-loop\|training loop]] |
| 11 — datasets and dataloaders (part 2) | [[4-pytorch-training/experimental-setup\|splits]], [[4-pytorch-training/datasets-and-dataloaders\|Dataset/DataLoader]], the improved [[4-pytorch-training/training-loop\|loop]], `nn.Linear` for [[3-linear-models/multiclass-logistic-regression\|multi-class logistic regression]] |
| 14 — multilayer perceptrons (device part) | [[4-pytorch-training/gpu-training\|GPU training]] |

Before: [[3-linear-models/index|Linear models]] · Next: [[5-deep-networks/index|Deep networks]]
