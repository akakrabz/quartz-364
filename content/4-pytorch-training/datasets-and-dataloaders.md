---
title: "Datasets and DataLoaders"
description: "torch.utils.data: writing a Dataset (__init__, __len__, __getitem__), wrapping tensors with TensorDataset, splitting with SubsetRandomSampler, Subset or random_split, and serving shuffled mini-batches with DataLoader (batch_size, shuffle, sampler, drop_last) — with the batch shapes, the batch count, torchvision's Fashion-MNIST and the Spring 2026 exam dataset."
tags: [topic, training]
aliases: ["Dataset", "DataLoader", "torch.utils.data", "__getitem__", "__len__", "SubsetRandomSampler", "random_split", "TensorDataset", "batch size", "torchvision datasets", "FashionMNIST", "transforms.ToTensor"]
---

*Lecture 11 · [[4-pytorch-training/index|PyTorch training]] · feeds the [[4-pytorch-training/training-loop|training loop]]; the splits are explained on [[4-pytorch-training/experimental-setup|experimental setup]]*

> [!abstract] In one breath
> A [`Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) answers two questions: *how many examples?* (`__len__`) and *what is example `i`?* (`__getitem__`, returning an `(x, y)` pair). A [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) wraps a dataset and, each time you loop over it, hands out batches: it picks indices (in order, shuffled, or from a sampler), fetches each example and **stacks** them along a new first dimension. One full loop over the loader is one epoch.

## What it is

Lecture 10's loop sampled batches by hand with `np.random.choice` and indexed tensors directly. `torch.utils.data` replaces that with two pieces:

- **`Dataset`** (you write it, or torchvision provides it): an indexable collection. Subclass `torch.utils.data.Dataset` and implement
  - `__init__(self, ...)`: load or generate the data, store it on `self`. Unlike `nn.Module`, calling `super().__init__()` is not required.
  - `__len__(self)`: makes `len(dataset)` work.
  - `__getitem__(self, idx)`: makes `dataset[idx]` work; return the input and the label.
- **`DataLoader`** (you only configure it): `DataLoader(dataset, batch_size=..., shuffle=..., sampler=...)`. It is an *iterable*: `for x_batch, y_batch in loader:`.

<!-- fig:train-dataflow -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 250" width="640" height="250" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>Dataset items flowing through a DataLoader into batches of shape (8, 2)</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><text x="80" y="18" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:14px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;font-weight:600;">Dataset</text><text x="80" y="36" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">len(ds) = 20</text><rect x="20" y="50" width="120" height="22" rx="4" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="80" y="61" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">ds[0] → (x, y)</text><rect x="20" y="76" width="120" height="22" rx="4" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="80" y="87" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">ds[1] → (x, y)</text><rect x="20" y="102" width="120" height="22" rx="4" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="80" y="113" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">ds[2] → (x, y)</text><rect x="20" y="128" width="120" height="22" rx="4" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="80" y="139" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">ds[3] → (x, y)</text><rect x="20" y="154" width="120" height="22" rx="4" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="80" y="165" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">ds[4] → (x, y)</text><rect x="20" y="180" width="120" height="22" rx="4" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="80" y="191" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">⋮</text><text x="80" y="222" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">x: (2,)   y: ()</text><line x1="150" y1="120" x2="222" y2="120" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="230" y="70" width="160" height="100" rx="10" fill="var(--paper)" stroke="var(--accent)" stroke-width="1.8"/><text x="310" y="92" text-anchor="middle" dominant-baseline="central" fill="var(--accent)" style="font-size:14px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;font-weight:600;">DataLoader</text><text x="310" y="114" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">batch_size=8</text><text x="310" y="132" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">shuffle=True</text><text x="310" y="152" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">stacks 8 items</text><line x1="398" y1="120" x2="446" y2="120" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" marker-end="url(#ARR)"/><rect x="455" y="52" width="170" height="40" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="540" y="65" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">batch 1</text><text x="540" y="80" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">x: (8, 2)   y: (8,)</text><rect x="455" y="102" width="170" height="40" rx="6" fill="var(--paper)" stroke="currentColor" stroke-width="1.4"/><text x="540" y="115" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">batch 2</text><text x="540" y="130" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">x: (8, 2)   y: (8,)</text><rect x="455" y="152" width="170" height="40" rx="6" fill="var(--paper)" stroke="var(--hi)" stroke-width="1.4"/><text x="540" y="165" text-anchor="middle" dominant-baseline="central" fill="var(--hi)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">batch 3  (last, partial)</text><text x="540" y="180" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:11px;font-family:'JetBrains Mono','IBM Plex Mono',ui-monospace,monospace;">x: (4, 2)   y: (4,)</text><text x="540" y="222" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">3 batches = len(loader)</text></svg><figcaption><b>Dataset → DataLoader → batches.</b> The <code>Dataset</code> hands out one example at a time; the <code>DataLoader</code> draws indices, calls <code>ds[i]</code> for each and stacks the results along a new dimension 0. With 20 examples and <code>batch_size=8</code> there are ⌈20/8⌉ = 3 batches per epoch, the last one holding only 4 (unless <code>drop_last=True</code>).</figcaption></figure>

## What you should know

- The three methods of a `Dataset` and what each overrides: `len(ds)` → `__len__`, `ds[i]` → `__getitem__`.
- If `ds[i]` returns `(x, y)` with `x` of shape $(d,)$ and `y` a scalar, a batch is `x_batch` $(B, d)$ and `y_batch` $(B,)$. Images $(1, 28, 28)$ become $(B, 1, 28, 28)$.
- `len(loader)` $= \lceil N/B\rceil$ batches; when $B$ does not divide $N$ the last one holds only $N \bmod B$ examples (or is dropped with `drop_last=True`).
- `shuffle=True` reshuffles every epoch; default is `False` (in order). Training loaders shuffle; evaluation loaders need not.
- `sampler=SubsetRandomSampler(indices)` draws only from `indices`, in random order — the lecture's way to make train / val / test loaders from one dataset. `shuffle` and `sampler` cannot both be given.
- [`TensorDataset(X, y)`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset) and [`random_split`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.dataset.random_split) are the shortcuts when the data is already in tensors.
- `__getitem__` should return tensors of the dtype the model and loss expect: float inputs, float targets for MSE/BCE, `long` targets for cross-entropy.

## Writing a Dataset: the toy two-class data

Lecture 11 turns Lecture 10's two Gaussian clusters into a dataset:

```python
import torch
from torch.utils.data import Dataset

class TwoClassDataset(Dataset):
    def __init__(self, N, sigma):
        self.N = N                          # points per class
        self.sigma = sigma
        self.plus_class = sigma*torch.randn(N, 2) + torch.tensor([-1, 1])
        self.negative_class = sigma*torch.randn(N, 2) + torch.tensor([1, -1])
        self.data = torch.cat((self.plus_class, self.negative_class), dim=0)
        self.labels = torch.cat((torch.ones(N), torch.zeros(N)))

    def __len__(self):
        return len(self.labels)             # 2N

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = TwoClassDataset(N=50, sigma=1)
print(len(dataset))
x, y = dataset[2]
```

```
100
```

`dataset[2]` is one point, `x` of shape $(2,)$ and `y` a 0-d float tensor (here `1.0`, since the first 50 points are the positive class).

For the four-class version the labels end with `.long()` — integer class indices for [[4-pytorch-training/loss-functions|`nn.CrossEntropyLoss`]]:

```python
self.labels = torch.cat((torch.zeros(N), torch.ones(N),
                         2*torch.ones(N), 3*torch.ones(N))).long()
```

## DataLoaders and the train / val / test split

```python
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

N_train, N_val, N_test = 60, 20, 20
indices = np.arange(len(dataset))
np.random.shuffle(indices)                  # shuffle once
train_indices = indices[:N_train]
val_indices = indices[N_train:N_train+N_val]
test_indices = indices[N_train+N_val:]

batch_size = 8
train_loader = DataLoader(dataset, batch_size=batch_size,
                          sampler=SubsetRandomSampler(train_indices))
val_loader = DataLoader(dataset, batch_size=batch_size,
                        sampler=SubsetRandomSampler(val_indices))
test_loader = DataLoader(dataset, batch_size=batch_size,
                         sampler=SubsetRandomSampler(test_indices))

for x_batch, y_batch in val_loader:         # loaders are iterable
    print(x_batch.shape, y_batch.shape)
```

```
torch.Size([8, 2]) torch.Size([8])
torch.Size([8, 2]) torch.Size([8])
torch.Size([4, 2]) torch.Size([4])
```

(The lecture prints the tensors themselves; these are their shapes.) The validation split has 20 points, so with batch size 8 there are $\lceil 20/8\rceil = 3$ batches of 8, 8 and 4. The training loader has $\lceil 60/8\rceil = 8$ batches per epoch. All three loaders share the one dataset object; only the indices differ, and they are disjoint because they are slices of one shuffled permutation.

The same split without numpy:

```python
from torch.utils.data import random_split, TensorDataset

ds = TensorDataset(X, y)                    # X: (100, 2), y: (100,)
train_ds, val_ds, test_ds = random_split(ds, [60, 20, 20])
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)
```

`random_split` returns [`Subset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Subset) objects; recent PyTorch versions also accept fractions such as `[0.6, 0.2, 0.2]`.

## The DataLoader arguments that matter

| Argument | Default | Meaning |
|---|---|---|
| `batch_size` | 1 | examples per batch |
| `shuffle` | `False` | new random order every epoch |
| `sampler` | `None` | an object that yields indices, e.g. [`SubsetRandomSampler`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.SubsetRandomSampler); excludes `shuffle` |
| `drop_last` | `False` | drop the final, smaller batch |
| `num_workers` | 0 | background processes that load batches in parallel |
| `collate_fn` | stack | how a list of examples becomes a batch |

The default collate function stacks tensors, turns Python numbers into a tensor, and batches each field of a tuple or dict separately — which is why a dataset returning `{"input": x, "target": y}` gives a batch dict with `batch["input"]` of shape $(B, M)$ (Spring 2025 Q7).

## torchvision datasets: Fashion-MNIST

[torchvision](https://docs.pytorch.org/vision/stable/datasets.html) ships many datasets that already implement the `Dataset` interface. Fashion-MNIST: 60,000 training and 10,000 test grayscale images, $28\times 28$, ten clothing classes.

```python
import torchvision
from torchvision import transforms

class FashionMNIST(torch.utils.data.Dataset):     # Lecture 14 version
    def __init__(self):
        self.data = torchvision.datasets.FashionMNIST(
            root='./', train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
        self.class_names = self.data.classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]       # image: (1, 28, 28)
        return image.reshape(-1), label     # flatten to (784,)
```

[`transforms.ToTensor()`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html) turns a PIL image (or $H\times W\times C$ uint8 array) into a float tensor of shape $(C, H, W)$ with values in $[0,1]$. [`transforms.Compose`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html) chains several transforms; [[5-deep-networks/data-augmentation|data augmentation]] adds random ones here. With `batch_size=64`, a loader over this dataset gives `x` of shape $(64, 784)$ and $\lceil 60000/64\rceil = 938$ batches per epoch.

The Lecture 11 version builds its tensors from `dataset.data` instead — the raw uint8 images, which **skip the transform** — so its pixels are floats in $[0, 255]$, not $[0, 1]$. Both train, but the scale of the inputs changes which learning rate works.

> [!trap] Common mistakes
> - **`__len__` returning the wrong number** (e.g. `self.N` when there are $2N$ points): the loader silently ignores the rest of the data.
> - **Integer tensors from Python ints.** `torch.tensor([1500, 2, 5])` is `int64`; `x @ self.w` with a float parameter then fails with a dtype error, and `MSELoss` with `long` targets raises a dtype error (typically at `backward()`, "Found dtype Long but expected Float"). Use `torch.tensor([...], dtype=torch.float32)` (the Spring 2026 dataset below needs this to actually run).
> - **Overlapping splits**: drawing train and test indices independently at random lets the same example land in both. Shuffle once and slice.
> - **`indices[N_train+N_val]` vs `indices[N_train+N_val:]`**: without the colon you get one index, not the rest of the array (a typo in the Lecture 14 starter code).
> - **`shuffle=True` together with `sampler=`** raises an error; the sampler already decides the order.
> - **Counting examples with `len(loader)`** — that is the number of batches.

> [!exam] On the exams
> [[6-midterm-1/spring-2026|Spring 2026]] Q8(i): complete a `HomePriceDataset` whose `dataset[i]` returns `(x, y)` with `x` a 1-D tensor (area, bedrooms, age) and `y` a 0-D tensor (price). The solution:
>
> ```python
> class HomePriceDataset(Dataset):
>     def __init__(self):
>         super().__init__()
>         self.area_list = [1500, 1800, 2200, 2500, 3000]
>         self.bedrooms_list = [2, 3, 3, 4, 5]
>         self.age_list = [5, 6, 4, 2, 5]
>         self.price_list = [250000, 310000, 360000, 420000, 510000]
>     def __len__(self):
>         return len(self.price_list)
>     def __getitem__(self, index):
>         x = torch.tensor([self.area_list[index],
>                           self.bedrooms_list[index],
>                           self.age_list[index]])
>         y = torch.tensor(self.price_list[index])
>         return x, y
> ```
>
> Write `dtype=torch.float32` in both `torch.tensor` calls so it works with the float model and MSE in part (iii). [[6-midterm-1/spring-2025|Spring 2025]] Q7 needs a `train_loader` with batch size 8, e.g. `DataLoader(dataset, batch_size=8, shuffle=True)` (the official key writes `split_dataset(dataset)`, a helper the problem never defines). The Spring formula sheets list `batch_size`, `shuffle` and `sampler` with the Lecture 11 descriptions.

## PyTorch docs

- [`torch.utils.data`](https://docs.pytorch.org/docs/stable/data.html) — the whole data-loading story: map-style datasets, samplers, batching, collate, workers.
- [`Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset), [`TensorDataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset), [`Subset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Subset), [`random_split`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.dataset.random_split).
- [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) — every argument.
- [`SubsetRandomSampler`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.SubsetRandomSampler) — the lecture's split tool.
- [torchvision datasets](https://docs.pytorch.org/vision/stable/datasets.html) and [`FashionMNIST`](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html).
- [torchvision transforms](https://docs.pytorch.org/vision/stable/transforms.html), [`ToTensor`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html), [`Compose`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html).
- Tutorial: [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html).

Related: [[4-pytorch-training/experimental-setup]] · [[4-pytorch-training/training-loop]] · [[4-pytorch-training/nn-module]] · [[5-deep-networks/data-augmentation]] · [[1-foundations/tensors]]
