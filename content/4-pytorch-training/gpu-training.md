---
title: "Training on a GPU: devices and .to()"
description: "Why GPUs speed up deep learning, how PyTorch names devices (cpu, cuda:0), choosing a device with torch.cuda.is_available(), moving the model and every batch with .to(device), moving results back with .cpu() for plotting and NumPy, and the device-mismatch errors you will see."
tags: [topic, training]
aliases: ["GPU", "CUDA", "device", "cuda:0", ".to(device)", "torch.cuda.is_available", ".cpu()", "MPS"]
---

*Lecture 14 · [[4-pytorch-training/index|PyTorch training]] · applied in the [[4-pytorch-training/training-loop|training loop]] for [[5-deep-networks/multilayer-perceptrons|MLPs]] and every deep model after*

> [!abstract] In one breath
> A GPU runs the matrix multiplications that dominate deep learning many times faster than a CPU (the lecture quotes 10–100× for large models). PyTorch calls it `'cuda:0'` (first GPU) and the CPU `'cpu'`. Pick one with `DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'`, move the model once with `model = model.to(DEVICE)`, move **every batch** inside the loop with `x, y = x.to(DEVICE), y.to(DEVICE)`, and bring things back with `.cpu()` before plotting. Model and data must be on the same device.

## What it is

Every tensor lives on a **device** — its [`.device`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) attribute (see [[1-foundations/tensors|tensors]]). Operations run on the device of their inputs, and all inputs of one operation must be on the same device. A GPU is specialised hardware with very fast implementations of a few operations, above all large matrix products; deep networks are mostly matrix products, so training and evaluation speed up — little or not at all for a tiny model on tiny data, a lot for larger and deeper models such as [[5-deep-networks/convolutional-layers|CNNs]].

Devices are named by strings: `'cpu'`, `'cuda:0'`, `'cuda:1'`, … (one per GPU; a one-GPU machine such as the course's ICRN servers at NCSA only has `cuda:0`). On Apple-silicon Macs the GPU is `'mps'`.

## What you should know

- The device line every notebook from Lecture 14 starts with:
  `DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'`.
- `model.to(DEVICE)` moves all parameters (and buffers) of a module; for modules it works in place and also returns the model.
- `x.to(DEVICE)` on a **tensor** returns a **new** tensor and leaves `x` where it was — you must assign the result.
- Move each batch inside the loop, right after it comes out of the loader. Do not move the whole dataset to the GPU unless it fits in GPU memory.
- `.cpu()` brings a tensor or model back; `.numpy()` and matplotlib need CPU tensors (and `.detach()` if they require grad). `.item()` works from any device.
- If no GPU exists, every `.to('cpu')` is a no-op, so the same code runs everywhere.
- Mixing devices raises `RuntimeError: Expected all tensors to be on the same device ...`.

## The pattern in code

From the Lecture 14 two-moons MLP:

```python
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using device: {}'.format(DEVICE))

model = TwoLayerModel(N=2, L=1024, M=2)
model = model.to(DEVICE)                    # parameters to GPU
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2,
                            momentum=0.99, weight_decay=1e-5)

for epoch in range(n_epoch):
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()

model.cpu()                                 # back for plotting
```

```
Using device: cuda:0
```

Metrics computed on the GPU are GPU tensors too; the lecture's accuracy helper ends with `.cpu()` so they can be accumulated and plotted:

```python
def multiclass_model_accuracy(model, input_data, labels):
    predictions = model(input_data)
    predicted_classes = torch.argmax(predictions, dim=1)
    return torch.sum(predicted_classes == labels).cpu()
```

Move the model **before** constructing the optimizer (the PyTorch docs recommend it), so the optimizer is built from the parameters on their final device.

## Useful calls

| Call | Result |
|---|---|
| [`torch.cuda.is_available()`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html) | `True` if a usable NVIDIA GPU and driver exist |
| [`torch.cuda.device_count()`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.device_count.html) | number of GPUs |
| [`torch.device('cuda:0')`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | a device object (a string works too) |
| `x.device`, `next(model.parameters()).device` | where a tensor / model is |
| [`x.to(device)`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html), [`x.cpu()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cpu.html), [`x.cuda()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cuda.html) | copy of `x` on that device |
| [`model.to(device)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to) | moves the module in place |
| `torch.zeros(3, device=DEVICE)` | create directly on the device (no copy) |

> [!trap] Common mistakes
> - **`x_batch.to(DEVICE)` without assignment.** Tensors are not moved in place; the next line still uses the CPU copy and fails with the device-mismatch error.
> - **Moving the model but not the data** (or the labels): same error, often from inside the loss.
> - **Calling `.numpy()` on a CUDA tensor**: raises an error; use `x.detach().cpu().numpy()`.
> - **Plotting with a model left on the GPU** while the grid of plot inputs is on the CPU — the reason Lecture 14 calls `model.cpu()` before `plot_model_probs`.
> - **Expecting a speed-up for a tiny model**: copying small batches to the GPU can cost more than the compute it saves.

## PyTorch docs

- [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html) — devices, memory, best practices (device-agnostic code).
- [`torch.cuda.is_available`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html), [`torch.cuda.device_count`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.device_count.html).
- [`torch.device`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) — the device attribute and how devices are named.
- [`Tensor.to`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html), [`Tensor.cpu`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cpu.html), [`Tensor.cuda`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.cuda.html); [`Module.to`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to).
- [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html) — the Apple-silicon GPU.

Related: [[4-pytorch-training/training-loop]] · [[4-pytorch-training/nn-module]] · [[1-foundations/tensors]] · [[5-deep-networks/multilayer-perceptrons]]
