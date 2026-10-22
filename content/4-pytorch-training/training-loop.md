---
title: "The PyTorch training loop"
description: "The five-step cycle every PyTorch model is trained with (zero_grad, forward, loss, backward, step), the epoch/batch structure with DataLoaders, logging loss and accuracy, evaluating under torch.no_grad with model.eval, gradient accumulation (Spring 2025 Q7), and the bugs that break a loop."
tags: [topic, training]
aliases: ["training loop", "train loop", "epoch", "iteration", "gradient accumulation", "model.eval", "torch.no_grad", "validation loop", "accuracy"]
---

*Lectures 10–11 (and 14 for devices) · [[4-pytorch-training/index|PyTorch training]] · uses [[4-pytorch-training/nn-module|nn.Module]], [[4-pytorch-training/loss-functions|losses]], [[4-pytorch-training/optimizers|optimizers]] and [[4-pytorch-training/datasets-and-dataloaders|DataLoaders]]*

> [!abstract] In one breath
> Before the loop: build the model, the data loaders, the loss and the optimizer. Inside: for every epoch, for every batch, **zero the gradients, run the model, compute the loss, backpropagate, step**. After each epoch, evaluate on the validation set without gradients. That loop, with logging added, is the core of all the training code in the rest of the course.

## What it is

<!-- fig:train-loop-cycle -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 660 400" width="660" height="400" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>Training loop cycle: zero_grad, forward, loss, backward, step</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><path d="M 409.0 66.4 L 413.3 67.6 L 417.6 68.9 L 421.9 70.3 L 426.1 71.7 L 430.3 73.2 L 434.4 74.7 L 438.5 76.3 L 442.5 78.0 L 446.5 79.7 L 450.4 81.5 L 454.2 83.3 L 458.0 85.2 L 461.8 87.2 L 465.4 89.2 L 469.0 91.3 L 472.6 93.4 L 476.0 95.6 L 479.4 97.8 L 482.8 100.1 L 486.0 102.4 L 489.2 104.8 L 492.3 107.2 L 495.4 109.7" fill="none" stroke="var(--muted)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#ARR)"/><path d="M 544.2 213.3 L 543.8 216.5 L 543.4 219.7 L 542.8 223.0 L 542.2 226.2 L 541.4 229.3 L 540.6 232.5 L 539.6 235.7 L 538.6 238.8 L 537.4 242.0 L 536.2 245.1 L 534.8 248.2 L 533.4 251.3 L 531.8 254.3 L 530.2 257.4 L 528.4 260.4 L 526.6 263.3 L 524.7 266.3 L 522.7 269.2 L 520.5 272.1 L 518.3 274.9 L 516.0 277.8 L 513.7 280.6 L 511.2 283.3" fill="none" stroke="var(--muted)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#ARR)"/><path d="M 383.4 347.4 L 378.8 348.1 L 374.2 348.8 L 369.6 349.5 L 365.0 350.0 L 360.4 350.5 L 355.7 350.9 L 351.1 351.3 L 346.4 351.6 L 341.7 351.8 L 337.0 351.9 L 332.3 352.0 L 327.7 352.0 L 323.0 351.9 L 318.3 351.8 L 313.6 351.6 L 308.9 351.3 L 304.3 350.9 L 299.6 350.5 L 295.0 350.0 L 290.4 349.5 L 285.8 348.8 L 281.2 348.1 L 276.6 347.4" fill="none" stroke="var(--muted)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#ARR)"/><path d="M 148.8 283.3 L 146.3 280.6 L 144.0 277.8 L 141.7 274.9 L 139.5 272.1 L 137.3 269.2 L 135.3 266.3 L 133.4 263.3 L 131.6 260.4 L 129.8 257.4 L 128.2 254.3 L 126.6 251.3 L 125.2 248.2 L 123.8 245.1 L 122.6 242.0 L 121.4 238.8 L 120.4 235.7 L 119.4 232.5 L 118.6 229.3 L 117.8 226.2 L 117.2 223.0 L 116.6 219.7 L 116.2 216.5 L 115.8 213.3" fill="none" stroke="var(--muted)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#ARR)"/><path d="M 164.6 109.7 L 167.7 107.2 L 170.8 104.8 L 174.0 102.4 L 177.2 100.1 L 180.6 97.8 L 184.0 95.6 L 187.4 93.4 L 191.0 91.3 L 194.6 89.2 L 198.2 87.2 L 202.0 85.2 L 205.8 83.3 L 209.6 81.5 L 213.5 79.7 L 217.5 78.0 L 221.5 76.3 L 225.6 74.7 L 229.7 73.2 L 233.9 71.7 L 238.1 70.3 L 242.4 68.9 L 246.7 67.6 L 251.0 66.4" fill="none" stroke="var(--muted)" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#ARR)"/><rect x="222" y="33" width="216" height="46" rx="8" fill="var(--paper)" stroke="currentColor" stroke-width="1.6"/><text x="330" y="47.5" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;font-weight:600;">optimizer.zero_grad()</text><text x="330" y="64.5" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">clear old .grad</text><rect x="426.1" y="135.3" width="216" height="46" rx="8" fill="var(--paper)" stroke="currentColor" stroke-width="1.6"/><text x="534.1" y="149.8" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;font-weight:600;">pred = model(x)</text><text x="534.1" y="166.8" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">forward pass</text><rect x="348.1" y="300.7" width="216" height="46" rx="8" fill="var(--paper)" stroke="currentColor" stroke-width="1.6"/><text x="456.1" y="315.2" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;font-weight:600;">loss = criterion(pred, y)</text><text x="456.1" y="332.2" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">scalar loss</text><rect x="95.9" y="300.7" width="216" height="46" rx="8" fill="var(--paper)" stroke="var(--accent)" stroke-width="1.6"/><text x="203.9" y="315.2" text-anchor="middle" dominant-baseline="central" fill="var(--accent)" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;font-weight:600;">loss.backward()</text><text x="203.9" y="332.2" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">fill every .grad</text><rect x="17.9" y="135.3" width="216" height="46" rx="8" fill="var(--paper)" stroke="currentColor" stroke-width="1.6"/><text x="125.9" y="149.8" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;font-weight:600;">optimizer.step()</text><text x="125.9" y="166.8" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">θ ← θ − α·g</text><text x="330" y="216" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">for x, y in train_loader:</text><text x="330" y="238" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">one iteration = one batch</text><text x="330" y="256" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">one epoch = one pass over the data</text></svg><figcaption><b>The training loop.</b> Every batch goes once around the cycle. <code>backward()</code> only <i>computes</i> gradients (adding them to <code>.grad</code>); <code>step()</code> is what changes the parameters, and <code>zero_grad()</code> stops last batch's gradient from being added to this one.</figcaption></figure>

Setup first (Lecture 10): the model is instantiated, the data is prepared, the loss and optimizer exist. Then each iteration:

1. `optimizer.zero_grad()` — clear gradients left from the previous batch;
2. `predictions = model(x_batch)` — forward pass;
3. `loss = criterion(predictions, y_batch)` — a scalar;
4. `loss.backward()` then `optimizer.step()` — compute gradients, then update;
5. (optional) log `loss.item()`, accuracy, …

**Iteration** = one batch, one `step()`. **Epoch** = one pass over the training set = `len(train_loader)` iterations.

## What you should know

- The order **zero_grad → forward → loss → backward → step**. `zero_grad()` may also go right after `step()`; what matters is that it happens once between consecutive `backward()` calls (unless you accumulate on purpose).
- Log with `loss.item()` (a float), not the tensor.
- Evaluate inside [`with torch.no_grad():`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html) (no graph, less memory) and, for models with dropout or batch norm, after [`model.eval()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval); switch back with `model.train()`.
- Multi-class accuracy: `(torch.argmax(logits, dim=1) == y).sum()`; binary with probabilities: threshold at 0.5.
- Average the epoch loss over batches: `epoch_loss / len(train_loader)`; average accuracy over examples: `correct / N_train`.
- Gradient accumulation: `backward()` every mini-batch, `step()` and `zero_grad()` every $k$ batches, gives an effective batch of $k\times B$.

## The Lecture 11 loop, annotated

Logistic regression on the toy two-class dataset with train / validation loaders (Lecture 11):

```python
criterion = nn.BCELoss()
model = LogisticRegression(2)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2,
                            momentum=0.99, weight_decay=1e-3)

n_epoch = 50
loss_values, train_acc, val_acc = [], [], []
for n in range(n_epoch):
    epoch_loss, epoch_correct = 0, 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(x_batch.unsqueeze(-1)).squeeze(-1)   # (B,)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_correct += model_accuracy(model, x_batch, y_batch)
    loss_values.append(epoch_loss / len(train_loader))
    train_acc.append(epoch_correct / N_train)

    correct = 0
    with torch.no_grad():                   # evaluation only
        for x_batch, y_batch in val_loader:
            correct += model_accuracy(model, x_batch, y_batch)
    val_acc.append(correct / N_val)
```

The `unsqueeze(-1)` / `squeeze(-1)` pair is there only because the hand-made `LogisticRegression` computes `w @ x` (see [[4-pytorch-training/nn-module#logistic-regression-as-a-module|nn.Module]]). With an `nn.Linear` model it disappears:

```python
def multiclass_model_accuracy(model, x, y):
    predictions = model(x)                  # logits (B, K)
    predicted = torch.argmax(predictions, dim=1)
    return torch.sum(predicted == y)        # count, not fraction
```

Lecture 11 wraps the whole thing in a `training_loop(model, train_loader, validation_loader, criterion, optimizer, n_epoch)` function that returns the model and the three curves, and counts `N_train += x_batch.size(0)` inside the loop so the accuracy is right even when the last batch is partial. On Fashion-MNIST with a [[3-linear-models/multiclass-logistic-regression|multi-class logistic regression]] (784 → 10, `lr=1e-3`, momentum 0.99, batch 64) 100 epochs took about 45 s on the lecture machine.

## A reusable template

```python
def train(model, train_loader, val_loader, criterion, optimizer,
          n_epoch, device='cpu'):
    model.to(device)
    history = {'loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(n_epoch):
        model.train()
        total, correct, running = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        history['loss'].append(running / len(train_loader))
        history['train_acc'].append(correct / total)
        history['val_acc'].append(evaluate(model, val_loader, device))
    return history

def evaluate(model, loader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total
```

(Device handling is explained on [[4-pytorch-training/gpu-training|GPU training]].) The plots to make afterwards are the lecture's three panels: training loss (log scale), training accuracy, validation accuracy, all against epoch. A training curve that keeps improving while validation stalls or gets worse is [[4-pytorch-training/experimental-setup#overfitting|overfitting]].

## Gradient accumulation

If a batch of 128 does not fit in memory but a batch of 8 does, skip `zero_grad()` on purpose: `backward()` **adds** into `.grad`, so after 16 backward calls the gradient holds the sum over 128 examples. Step and clear only then. This was [[6-midterm-1/spring-2025|Spring 2025]] Q7 (10 points):

```python
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
num_accumulation_steps = 128 // 8           # 16
batches_processed = 0
optimizer.zero_grad()
for epoch in range(10):
    for batch in train_loader:
        inp, tgt = batch["input"], batch["target"]
        output = model(inp)
        loss = criterion(output, tgt)
        loss.backward()                     # adds into .grad
        batches_processed += 1
        if batches_processed % num_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

With `MSELoss()`'s default mean, the accumulated gradient is the *sum* of 16 batch means, i.e. 16 times the gradient of the mean over 128. To match a true batch of 128 exactly, use `loss = criterion(output, tgt) / num_accumulation_steps` (or a 16× smaller learning rate). The exam solution does not divide; mentioning it shows you understand what is accumulated.

> [!trap] Common mistakes
> - **Missing `zero_grad()`**: gradients pile up across iterations, the steps keep growing and the loss explodes.
> - **`loss.backward()` and `optimizer.step()` swapped**, or `step()` forgotten — nothing (or the wrong thing) is learned.
> - **`loss_values.append(loss)`** keeps every batch's computational graph alive; memory grows each iteration. Use `loss.item()`.
> - **Validation without `torch.no_grad()`**: correct numbers, but it builds graphs and wastes memory; calling `backward()` or `step()` on validation data is a real bug (that is training on it).
> - **Forgetting `model.train()` after `model.eval()`**: dropout and batch norm stay in evaluation mode for the rest of training.
> - **Accuracy as a count vs a fraction**: the lecture helper returns the number correct; divide by the number of examples, not the number of batches.
> - **Shape mismatch between prediction and target**: $(B,1)$ vs $(B,)$ for MSE/BCE — squeeze first ([[4-pytorch-training/loss-functions|loss functions]]).

> [!exam] On the exams
> Spring 2026 Q8(iii) (part of 15 points): train `HomePriceModel` on `HomePriceDataset` for 10 epochs with SGD (`lr=0.01`), MSE and batch size 2. The expected answer is exactly the five-step loop:
>
> ```python
> dataset = HomePriceDataset()
> dataloader = DataLoader(dataset, batch_size=2)
> model = HomePriceModel(3)
> loss_fn = nn.MSELoss()
> optim = SGD(model.parameters(), lr=0.01)
> for epoch in range(10):
>     for batch in dataloader:
>         optim.zero_grad()
>         output = model(batch[0])
>         loss = loss_fn(output, batch[1])
>         loss.backward()
>         optim.step()
> ```
>
> Spring 2026 T/F (e) tests the first fact behind the loop: `loss.backward()` does **not** update the weights ("computes the gradients and immediately updates the weights" is **false**). Spring 2025 Q7 (above) rests on the second: `.grad` accumulates until cleared. See [[6-midterm-1/spring-2026|Spring 2026]] and [[6-midterm-1/spring-2025|Spring 2025]].

## PyTorch docs

- [`torch.no_grad`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html) — the context manager for evaluation.
- [`Module.train`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train) / [`Module.eval`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval) — the mode switch.
- [`Tensor.backward`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html) and [`Tensor.item`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html).
- [`torch.argmax`](https://docs.pytorch.org/docs/stable/generated/torch.argmax.html) — predicted class from logits.
- [`Optimizer.zero_grad`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html), [`Optimizer.step`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html).
- Tutorial: [Optimizing model parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) — the official train/test loop.

Related: [[4-pytorch-training/datasets-and-dataloaders]] · [[4-pytorch-training/experimental-setup]] · [[4-pytorch-training/optimizers]] · [[4-pytorch-training/gpu-training]] · [[2-calculus-and-optimization/autograd]]
