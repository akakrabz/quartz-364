---
title: "Data augmentation"
description: "Random label-preserving transformations of the training images with torchvision transforms.v2: pipelines, where they go in a Dataset, augmenting images and masks together, Lecture 16's four experiments on when augmentation helps and when it hurts, and adversarial examples (FGSM) as augmentation turned against the model."
tags: [topic, deep-nets]
aliases: ["augmentation", "image augmentation", "transforms.v2", "torchvision transforms", "tv_tensors", "adversarial examples", "FGSM", "fast gradient sign method", "adversarial training"]
---

*Lecture 16 · [[5-deep-networks/index|Deep networks]] · builds on [[4-pytorch-training/datasets-and-dataloaders|datasets and dataloaders]] and [[4-pytorch-training/experimental-setup|experimental setup]]*

> [!abstract] In one breath
> Augmentation applies random transformations that **do not change the label** (flips, small rotations and shifts, brightness and contrast, blur) to every training image, fresh each epoch, so the model sees more variation than the dataset holds. It helps when it simulates variation the model **will meet at test time** and cannot learn from the data it has; on clean test data it can cost a little accuracy at the same training budget. For segmentation, geometric changes must move the image and its mask together. Adversarial examples turn the idea around: a tiny change in the *worst* direction fools the model.

## What it is

![[l16-augmentation-examples.jpg|One image, many label-preserving variations]]

```python
from torchvision.transforms import v2

augment = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.3, contrast=0.3),
])
augmented = augment(image)       # new random draw every call
```

Each call draws new random parameters. [`v2.Compose`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.Compose.html) applies the transforms in order; the input here is a $1\times28\times28$ float tensor.

## What you should know

- Augmentation is applied to **training** data only, inside the `Dataset`'s `__getitem__` (or the batch loop), so every epoch sees new versions. Validation and test data stay as they are.
- Common transforms: [`RandomHorizontalFlip`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomHorizontalFlip.html), [`RandomCrop(28, padding=3)`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomCrop.html) (shift by up to 3 px), [`RandomRotation`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomRotation.html), [`RandomAffine`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomAffine.html), [`ColorJitter`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ColorJitter.html), [`GaussianBlur`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.GaussianBlur.html) inside [`RandomApply`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomApply.html).
- A transform is only safe if it **preserves the label** (a 6 rotated 180° is a 9) and is **realistic** (an upside-down boot never appears at test time).
- Segmentation: wrap as `tv_tensors.Image` and `tv_tensors.Mask` and call the pipeline on both at once; geometry applies to both, colour only to the image; masks stay exactly 0/1.
- Lecture 16's results: augmentation helps on **shifted/varied** test sets, most with **little data**; on clean test images at a fixed number of steps it gains at most about half a point and can cost up to 11. 600 augmented images beat 6,000 plain ones on the varied test set.
- Augmentation puts robustness into the **data**; a CNN with global pooling gets shift robustness from its **architecture** ([[5-deep-networks/pooling|pooling]]).
- FGSM: $x_\text{adv} = \mathrm{clip}_{[0,1]}\big(x + \epsilon\,\mathrm{sign}(\nabla_x L)\big)$. $\epsilon = 0.05$ drops a CNN from 88.3% to 45.9%; random noise of the same size barely matters.

## Where augmentation goes

```python
class FashionDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X, self.y, self.transform = X.unsqueeze(1), y, transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        return (self.transform(x) if self.transform else x), self.y[idx]

random_shift = v2.RandomCrop(28, padding=3)       # pad 3, crop back randomly
train_data = FashionDataset(X_train[:600], y_train[:600], transform=random_shift)
loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
```

Because `__getitem__` runs every time a sample is loaded, the [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) hands the model a different random version of each image in every epoch. The dataset itself is never enlarged or changed ([[4-pytorch-training/datasets-and-dataloaders|datasets and dataloaders]]).

## Augmenting images and masks

For [[5-deep-networks/semantic-segmentation|segmentation]], a flip or rotation of the image must apply to its mask with the **same** random numbers, while brightness changes only the image. `transforms.v2` does this when the inputs are wrapped as [`tv_tensors`](https://docs.pytorch.org/vision/stable/tv_tensors.html):

```python
from torchvision import tv_tensors

seg_augment = v2.Compose([v2.RandomHorizontalFlip(p=0.5),
                          v2.RandomAffine(degrees=25, translate=(0.15, 0.15)),
                          v2.ColorJitter(brightness=0.5)])
image, mask = tv_tensors.Image(scenes_val[0]), tv_tensors.Mask(masks_val[0])
aug_image, aug_mask = seg_augment(image, mask)    # one call, same parameters
print(aug_mask.unique().tolist(), type(aug_image).__name__, type(aug_mask).__name__)
```

```
[0.0, 1.0] Image Mask
```

The mask keeps exactly the values 0 and 1: masks are resampled with nearest-neighbour interpolation, so augmentation never invents half-labels.

> [!example] Lecture 16 "Your turn 3": a segmentation pipeline
> Build `my_seg_augment`: a random horizontal flip, a random rotation of up to 15°, a brightness jitter of 0.3. It must keep image and mask aligned and the mask 0/1.

> [!solution]- Solution
> ```python
> my_seg_augment = v2.Compose([v2.RandomHorizontalFlip(), v2.RandomRotation(15),
>                              v2.ColorJitter(brightness=0.3)])
> image, mask = my_seg_augment(tv_tensors.Image(scenes_val[0]), tv_tensors.Mask(masks_val[0]))
> ```

### The augmentation explorer

The separate notebook `L16_Augmentation_Explorer.ipynb` applies a flip, rotation, shift and brightness with sliders, using the functional API so the image and mask get the same numbers:

```python
F2 = v2.functional
image, mask = F2.horizontal_flip(image), F2.horizontal_flip(mask)
args = dict(angle=30.0, translate=[10, -6], scale=1.0, shear=[0.0, 0.0])
image, mask = F2.affine(image, **args), F2.affine(mask, **args)   # same geometry
image = F2.adjust_brightness(image, 1.6)                          # image only
```

The outline of the mask still fits the clothing exactly after this flip, $30°$ rotation, shift and 60% brightening.

> [!question] Things to try (from the explorer)
> 1. Rotate by 45°: what fills the corners that come from outside the image? Does the mask agree?
> 2. Shift by 16 pixels: part of a garment leaves the image. What happens to its mask pixels?
> 3. Brightness 2.0: the image saturates. Does the mask change? Should it?
> 4. Why must the rotation and shift use the *same* numbers for image and mask?

> [!solution]- Answers
> 1. Zeros (the default fill) in both: black corners in the image, "not clothing" in the mask, so they agree.
> 2. Those pixels are cut off in both the image and the mask; the label simply loses that part of the object.
> 3. No, and it should not: brightness is a colour transform and the clothing is still where it was.
> 4. Otherwise the mask labels pixels that are no longer the object, and the model is trained on wrong per-pixel labels.

## When does augmentation help?

### 1. Position

Lecture 15 trained an MLP with every garment at the **centre** of a $64\times64$ canvas; it failed as soon as the object moved. Pasting each training image at a **random** position (shift augmentation) keeps the MLP at roughly the same accuracy (about 65–70%) at every shift, where the centre-trained one drops from about 85% to chance, and now every input pixel's weights get trained instead of only the centre $28\times28$. Two costs: at the centre the augmented MLP is *less* accurate (the task got harder), and the CNN with global pooling got the same robustness from its architecture with more than 40 times fewer parameters.

![[l16-l17-center-vs-shifted.png|Lecture 15: trained at the centre, tested shifted]]

### 2. Data size and test-time variation

A small CNN, 1,200 training steps, 600 or 6,000 images, with or without `RandomCrop(28, padding=3)`. Tested on centred test images and on test images shifted by up to 3 pixels:

| training images | augmentation | centred test | shifted test |
|---|---|---|---|
| 600 | no | 81.9% | 53.0% |
| 600 | yes | 81.4% | 77.4% |
| 6,000 | no | 88.3% | 62.2% |
| 6,000 | yes | 83.3% | 78.9% |

- **Centred** test set: augmentation does not help, and at the same number of steps it can cost accuracy: the variation it simulates never shows up.
- **Shifted** test set: worth about 17–24 points, and it matters most with **little data**.

### 3. Every augmentation, three data sizes

Each augmentation alone, all realistic ones combined, and a vertical flip as an unrealistic control; 800 steps per run (the lecture also ran 2,000 images; 600 and 6,000 are shown). The **varied** test set changes every test image once by a random mix of the realistic augmentations.

<!-- fig:deepb-aug-grid -->
<figure class="cs-fig"><svg role="img" aria-label="Dot plot of clean and varied test accuracy for each augmentation" style="max-width:640px;width:100%;height:auto" stroke-linejoin="round" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 508.963906 271.440312" xmlns="http://www.w3.org/2000/svg" version="1.1"><defs></defs><g id="deepb-aug-grid-figure_1"><g id="deepb-aug-grid-patch_1"><path d="M 0 271.440312  L 508.963906 271.440312  L 508.963906 0  L 0 0  L 0 271.440312  z " style="fill: none"/></g><g id="deepb-aug-grid-axes_1"><g id="deepb-aug-grid-patch_2"><path d="M 144.747656 231.239453  L 501.763906 231.239453  L 501.763906 30.200781  L 144.747656 30.200781  L 144.747656 231.239453  z " style="fill: none"/></g><g id="deepb-aug-grid-matplotlib.axis_1"><g id="deepb-aug-grid-xtick_1"><g id="deepb-aug-grid-line2d_1"><path d="M 174.49901 231.239453  L 174.49901 30.200781  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g id="deepb-aug-grid-line2d_2"><defs><path id="deepb-aug-grid-m0ff6d609e4" d="M 0 0  L 0 3.5  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#deepb-aug-grid-m0ff6d609e4" x="174.49901" y="231.239453" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_1"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="174.49901" y="246.596875" transform="rotate(-0 174.49901 246.596875)">40</text></g></g><g id="deepb-aug-grid-xtick_2"><g id="deepb-aug-grid-line2d_3"><path d="M 234.001719 231.239453  L 234.001719 30.200781  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g id="deepb-aug-grid-line2d_4"><g><use xlink:href="#deepb-aug-grid-m0ff6d609e4" x="234.001719" y="231.239453" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_2"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="234.001719" y="246.596875" transform="rotate(-0 234.001719 246.596875)">50</text></g></g><g id="deepb-aug-grid-xtick_3"><g id="deepb-aug-grid-line2d_5"><path d="M 293.504427 231.239453  L 293.504427 30.200781  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g id="deepb-aug-grid-line2d_6"><g><use xlink:href="#deepb-aug-grid-m0ff6d609e4" x="293.504427" y="231.239453" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_3"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="293.504427" y="246.596875" transform="rotate(-0 293.504427 246.596875)">60</text></g></g><g id="deepb-aug-grid-xtick_4"><g id="deepb-aug-grid-line2d_7"><path d="M 353.007135 231.239453  L 353.007135 30.200781  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g id="deepb-aug-grid-line2d_8"><g><use xlink:href="#deepb-aug-grid-m0ff6d609e4" x="353.007135" y="231.239453" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_4"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="353.007135" y="246.596875" transform="rotate(-0 353.007135 246.596875)">70</text></g></g><g id="deepb-aug-grid-xtick_5"><g id="deepb-aug-grid-line2d_9"><path d="M 412.509844 231.239453  L 412.509844 30.200781  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g id="deepb-aug-grid-line2d_10"><g><use xlink:href="#deepb-aug-grid-m0ff6d609e4" x="412.509844" y="231.239453" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_5"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="412.509844" y="246.596875" transform="rotate(-0 412.509844 246.596875)">80</text></g></g><g id="deepb-aug-grid-xtick_6"><g id="deepb-aug-grid-line2d_11"><path d="M 472.012552 231.239453  L 472.012552 30.200781  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-opacity: 0.35; stroke-width: 0.8; stroke-linecap: square"/></g><g id="deepb-aug-grid-line2d_12"><g><use xlink:href="#deepb-aug-grid-m0ff6d609e4" x="472.012552" y="231.239453" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_6"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="472.012552" y="246.596875" transform="rotate(-0 472.012552 246.596875)">90</text></g></g><g id="deepb-aug-grid-text_7"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: middle; fill: currentColor" x="323.255781" y="261.597734" transform="rotate(-0 323.255781 261.597734)">test accuracy (%), small CNN, 6,000 training images, 800 steps</text></g></g><g id="deepb-aug-grid-matplotlib.axis_2"><g id="deepb-aug-grid-ytick_1"><g id="deepb-aug-grid-line2d_13"><defs><path id="deepb-aug-grid-m8855423305" d="M 0 0  L -3.5 0  " style="stroke: currentColor; stroke-width: 0.8"/></defs><g><use xlink:href="#deepb-aug-grid-m8855423305" x="144.747656" y="39.338903" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_8"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="137.747656" y="43.517614" transform="rotate(-0 137.747656 43.517614)">none</text></g></g><g id="deepb-aug-grid-ytick_2"><g id="deepb-aug-grid-line2d_14"><g><use xlink:href="#deepb-aug-grid-m8855423305" x="144.747656" y="62.184206" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_9"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="137.747656" y="66.363347" transform="rotate(-0 137.747656 66.363347)">horizontal flip</text></g></g><g id="deepb-aug-grid-ytick_3"><g id="deepb-aug-grid-line2d_15"><g><use xlink:href="#deepb-aug-grid-m8855423305" x="144.747656" y="85.02951" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_10"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="137.747656" y="89.208651" transform="rotate(-0 137.747656 89.208651)">random crop (shift)</text></g></g><g id="deepb-aug-grid-ytick_4"><g id="deepb-aug-grid-line2d_16"><g><use xlink:href="#deepb-aug-grid-m8855423305" x="144.747656" y="107.874814" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_11"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="137.747656" y="112.053954" transform="rotate(-0 137.747656 112.053954)">rotation ±15°</text></g></g><g id="deepb-aug-grid-ytick_5"><g id="deepb-aug-grid-line2d_17"><g><use xlink:href="#deepb-aug-grid-m8855423305" x="144.747656" y="130.720117" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_12"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="137.747656" y="134.899258" transform="rotate(-0 137.747656 134.899258)">shear + scale</text></g></g><g id="deepb-aug-grid-ytick_6"><g id="deepb-aug-grid-line2d_18"><g><use xlink:href="#deepb-aug-grid-m8855423305" x="144.747656" y="153.565421" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_13"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="137.747656" y="157.744561" transform="rotate(-0 137.747656 157.744561)">blur</text></g></g><g id="deepb-aug-grid-ytick_7"><g id="deepb-aug-grid-line2d_19"><g><use xlink:href="#deepb-aug-grid-m8855423305" x="144.747656" y="176.410724" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_14"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="137.747656" y="180.589865" transform="rotate(-0 137.747656 180.589865)">brightness + contrast</text></g></g><g id="deepb-aug-grid-ytick_8"><g id="deepb-aug-grid-line2d_20"><g><use xlink:href="#deepb-aug-grid-m8855423305" x="144.747656" y="199.256028" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_15"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="137.747656" y="203.435169" transform="rotate(-0 137.747656 203.435169)">vertical flip (unrealistic)</text></g></g><g id="deepb-aug-grid-ytick_9"><g id="deepb-aug-grid-line2d_21"><g><use xlink:href="#deepb-aug-grid-m8855423305" x="144.747656" y="222.101332" style="fill: currentColor; stroke: currentColor; stroke-width: 0.8"/></g></g><g id="deepb-aug-grid-text_16"><text style="font-size: 11px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: end; fill: currentColor" x="137.747656" y="226.280472" transform="rotate(-0 137.747656 226.280472)">all realistic combined</text></g></g></g><g id="deepb-aug-grid-line2d_22"><path d="M 194.729931 39.338903  L 461.302065 39.338903  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-width: 1.2; stroke-linecap: square"/></g><g id="deepb-aug-grid-line2d_23"><path d="M 270.298371 62.184206  L 441.666171 62.184206  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-width: 1.2; stroke-linecap: square"/></g><g id="deepb-aug-grid-line2d_24"><path d="M 281.008858 85.02951  L 411.31979 85.02951  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-width: 1.2; stroke-linecap: square"/></g><g id="deepb-aug-grid-line2d_25"><path d="M 231.026583 107.874814  L 430.955683 107.874814  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-width: 1.2; stroke-linecap: square"/></g><g id="deepb-aug-grid-line2d_26"><path d="M 209.010581 130.720117  L 436.905954 130.720117  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-width: 1.2; stroke-linecap: square"/></g><g id="deepb-aug-grid-line2d_27"><path d="M 187.589606 153.565421  L 444.046279 153.565421  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-width: 1.2; stroke-linecap: square"/></g><g id="deepb-aug-grid-line2d_28"><path d="M 214.365825 176.410724  L 449.99655 176.410724  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-width: 1.2; stroke-linecap: square"/></g><g id="deepb-aug-grid-line2d_29"><path d="M 195.919985 199.256028  L 436.310927 199.256028  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-width: 1.2; stroke-linecap: square"/></g><g id="deepb-aug-grid-line2d_30"><path d="M 367.882812 222.101332  L 397.03914 222.101332  " clip-path="url(#deepb-aug-grid-pfbf1a04183)" style="fill: none; stroke: var(--muted); stroke-width: 1.2; stroke-linecap: square"/></g><g id="deepb-aug-grid-patch_3"><path d="M 144.747656 231.239453  L 144.747656 30.200781  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g id="deepb-aug-grid-patch_4"><path d="M 144.747656 231.239453  L 501.763906 231.239453  " style="fill: none; stroke: currentColor; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/></g><g id="deepb-aug-grid-PathCollection_1"><defs><path id="deepb-aug-grid-m955a02f900" d="M 0 3  C 0.795609 3 1.55874 2.683901 2.12132 2.12132  C 2.683901 1.55874 3 0.795609 3 0  C 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132  C 1.55874 -2.683901 0.795609 -3 0 -3  C -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132  C -2.683901 -1.55874 -3 -0.795609 -3 0  C -3 0.795609 -2.683901 1.55874 -2.12132 2.12132  C -1.55874 2.683901 -0.795609 3 0 3  z " style="stroke: var(--accent)"/></defs><g clip-path="url(#deepb-aug-grid-pfbf1a04183)"><use xlink:href="#deepb-aug-grid-m955a02f900" x="461.302065" y="39.338903" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#deepb-aug-grid-m955a02f900" x="441.666171" y="62.184206" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#deepb-aug-grid-m955a02f900" x="411.31979" y="85.02951" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#deepb-aug-grid-m955a02f900" x="430.955683" y="107.874814" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#deepb-aug-grid-m955a02f900" x="436.905954" y="130.720117" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#deepb-aug-grid-m955a02f900" x="444.046279" y="153.565421" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#deepb-aug-grid-m955a02f900" x="449.99655" y="176.410724" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#deepb-aug-grid-m955a02f900" x="436.310927" y="199.256028" style="fill: var(--accent); stroke: var(--accent)"/><use xlink:href="#deepb-aug-grid-m955a02f900" x="397.03914" y="222.101332" style="fill: var(--accent); stroke: var(--accent)"/></g></g><g id="deepb-aug-grid-PathCollection_2"><defs><path id="deepb-aug-grid-mbe38ee535d" d="M 0 3  C 0.795609 3 1.55874 2.683901 2.12132 2.12132  C 2.683901 1.55874 3 0.795609 3 0  C 3 -0.795609 2.683901 -1.55874 2.12132 -2.12132  C 1.55874 -2.683901 0.795609 -3 0 -3  C -0.795609 -3 -1.55874 -2.683901 -2.12132 -2.12132  C -2.683901 -1.55874 -3 -0.795609 -3 0  C -3 0.795609 -2.683901 1.55874 -2.12132 2.12132  C -1.55874 2.683901 -0.795609 3 0 3  z " style="stroke: var(--hi)"/></defs><g clip-path="url(#deepb-aug-grid-pfbf1a04183)"><use xlink:href="#deepb-aug-grid-mbe38ee535d" x="194.729931" y="39.338903" style="fill: var(--hi); stroke: var(--hi)"/><use xlink:href="#deepb-aug-grid-mbe38ee535d" x="270.298371" y="62.184206" style="fill: var(--hi); stroke: var(--hi)"/><use xlink:href="#deepb-aug-grid-mbe38ee535d" x="281.008858" y="85.02951" style="fill: var(--hi); stroke: var(--hi)"/><use xlink:href="#deepb-aug-grid-mbe38ee535d" x="231.026583" y="107.874814" style="fill: var(--hi); stroke: var(--hi)"/><use xlink:href="#deepb-aug-grid-mbe38ee535d" x="209.010581" y="130.720117" style="fill: var(--hi); stroke: var(--hi)"/><use xlink:href="#deepb-aug-grid-mbe38ee535d" x="187.589606" y="153.565421" style="fill: var(--hi); stroke: var(--hi)"/><use xlink:href="#deepb-aug-grid-mbe38ee535d" x="214.365825" y="176.410724" style="fill: var(--hi); stroke: var(--hi)"/><use xlink:href="#deepb-aug-grid-mbe38ee535d" x="195.919985" y="199.256028" style="fill: var(--hi); stroke: var(--hi)"/><use xlink:href="#deepb-aug-grid-mbe38ee535d" x="367.882812" y="222.101332" style="fill: var(--hi); stroke: var(--hi)"/></g></g><g id="deepb-aug-grid-legend_1"><g id="deepb-aug-grid-PathCollection_3"><g><use xlink:href="#deepb-aug-grid-m955a02f900" x="168.470562" y="16.173437" style="fill: var(--accent); stroke: var(--accent)"/></g></g><g id="deepb-aug-grid-text_17"><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="186.470562" y="18.798437" transform="rotate(-0 186.470562 18.798437)">clean test images</text></g><g id="deepb-aug-grid-PathCollection_4"><g><use xlink:href="#deepb-aug-grid-mbe38ee535d" x="305.283062" y="16.173437" style="fill: var(--hi); stroke: var(--hi)"/></g></g><g id="deepb-aug-grid-text_18"><text style="font-size: 10px; font-family: 'Source Sans 3', 'Source Sans Pro', 'DejaVu Sans'; text-anchor: start; fill: currentColor" x="323.283062" y="18.798437" transform="rotate(-0 323.283062 18.798437)">varied test images</text></g></g></g></g><defs><clipPath id="deepb-aug-grid-pfbf1a04183"><rect x="144.747656" y="30.200781" width="357.01625" height="201.038672"/></clipPath></defs></svg> <figcaption><strong>Which augmentation helps?</strong> Each one alone, all realistic ones combined, and an unrealistic vertical flip, at the same 800 training steps. On clean test images every augmentation costs 2 to 11 points; on the <em>varied</em> test set (each image changed once by a random mix of realistic augmentations), shifts and flips help most and the combination adds about 30 points. Numbers from the Lecture 16 notebook.</figcaption></figure>

| augmentation | clean, 600 | clean, 6,000 | varied, 600 | varied, 6,000 |
|---|---|---|---|---|
| none | 81.7 | 88.2 | 38.5 | 43.4 |
| horizontal flip | 79.5 | 84.9 | 44.9 | 56.1 |
| random crop (shift) | 81.3 | 79.8 | 53.2 | 57.9 |
| rotation ±15° | 80.3 | 83.1 | 45.6 | 49.5 |
| shear + scale | 81.1 | 84.1 | 41.5 | 45.8 |
| blur | 81.7 | 85.3 | 40.3 | 42.2 |
| brightness + contrast | 82.3 | 86.3 | 39.8 | 46.7 |
| vertical flip (unrealistic) | 80.0 | 84.0 | 34.3 | 43.6 |
| all realistic combined | 78.1 | 77.4 | 73.7 | 72.5 |

- **Clean** test images: no augmentation helps by more than half a point; most cost accuracy, because with the same 800 steps the augmented models have not finished learning their harder task.
- **Varied** test images: shifts help most, then flips and rotations; all realistic augmentations together add about 30 points.
- **More data does not buy robustness**: 10× more clean images barely move the varied-test accuracy (38.5 → 43.4), while 600 images *with* augmentation reach 73.7.
- The unrealistic **vertical flip never helps**.

### 4. Segmentation

The U-Net trained 12 epochs with and without `seg_augment` (image and mask together). Validation IoU on clean scenes and on a varied copy:

| training scenes | augmentation | clean IoU | varied IoU |
|---|---|---|---|
| 100 | no | 0.828 | 0.679 |
| 100 | yes | 0.805 | 0.758 |
| 500 | no | 0.868 | 0.620 |
| 500 | yes | 0.825 | 0.816 |

Without augmentation the U-Net marks 8% of the background as clothing on the varied scenes: brightened background noise reaches 0.45, brighter than any background it saw in training (at most 0.33). With augmentation, 0.6%.

> [!key] The rule
> **Augment for the variation you expect at test time, and budget more training when you do.**

## When augmentation hurts

A transformation is only safe if it **preserves the label**:

- rotating a **6** by 180° makes a **9**;
- flipping text produces non-letters;
- colour jitter destroys the signal when colour *is* the label (ripe vs unripe fruit, some medical stains).

An upside-down boot keeps its label, but no test image will look like that; such augmentation mostly wastes model capacity. Horizontal flips, on the other hand, are realistic for clothing.

## Adversarial examples: augmentation turned against the model

Augmentation changes pixels a little to make a model *more* robust. Turned around: change the pixels a little in the one direction that increases the loss the most. The **fast gradient sign method** (FGSM, Goodfellow et al., 2015) takes the gradient of the loss with respect to the **input image** (not the weights) via [[2-calculus-and-optimization/autograd|autograd]]:

$$
x_\text{adv} = \mathrm{clip}_{[0,1]}\Big(x + \epsilon\,\mathrm{sign}\big(\nabla_x L(f(x), y)\big)\Big).
$$

```python
def fgsm(model, x, y, epsilon):
    x = x.clone().to(device).requires_grad_(True)
    F.cross_entropy(model(x), y.to(device)).backward()   # gradient w.r.t. image
    return (x + epsilon * x.grad.sign()).clamp(0, 1).detach().cpu()
```

On the small CNN trained on 6,000 clean images, FGSM against random $\pm\epsilon$ changes of exactly the same size:

| $\epsilon$ (pixel values in [0, 1]) | 0 | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|
| FGSM (adversarial direction) | 88.3 | 71.7 | 45.9 | 17.8 | 3.4 |
| random direction, same size | 88.3 | 88.2 | 87.4 | 85.1 | 67.9 |

- It is the **direction**, not the size: $\epsilon = 0.05$ is about 13 of 255 grey levels.
- The gradient that trains the network also tells an attacker which way to push every pixel; this is a security and trust issue (adversarial glasses can fool face recognition).
- **Adversarial training** turns it back into augmentation: add FGSM images, with their correct labels, to every training batch.

> [!example] Lecture 16 "Your turn 4": one FGSM step
> Given `x` in $[0,1]$ and `grad` $=\nabla_x L$, return the adversarial image.

> [!solution]- Solution
> ```python
> def my_fgsm_step(x, grad, epsilon):
>     return (x + epsilon * grad.sign()).clamp(0, 1)
> ```
> Every pixel moves by exactly $\epsilon$ (up or down by the sign of its gradient), then is clipped back into the valid range with [`torch.clamp`](https://docs.pytorch.org/docs/stable/generated/torch.clamp.html).

> [!trap] Common mistakes
> - Augmenting the validation or test set during evaluation: you then measure a different task. (Lecture 16 builds a *fixed* "varied" test copy on purpose, to measure robustness.)
> - Applying the image pipeline and the mask pipeline in two separate calls: they draw **different** random parameters. Call one pipeline on `(image, mask)`.
> - Colour transforms on masks or on images where colour is the label.
> - Judging augmentation on clean data with the same number of steps: the augmented model needs more training to catch up.
> - `RandomCrop(28, padding=3)` on a 28×28 image does not crop away content permanently: it pads to 34×34 and crops a random 28×28 window, i.e. a random shift of up to 3 pixels.
> - FGSM uses `sign(grad)`, not the gradient itself, and must `clamp` to the valid pixel range.

## PyTorch docs

- [Transforms v2 overview](https://docs.pytorch.org/vision/stable/transforms.html) — every transform, and how v2 handles images, masks and boxes.
- [Getting started with transforms v2](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html) — the joint image + mask pattern.
- [`v2.Compose`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.Compose.html), [`v2.RandomHorizontalFlip`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomHorizontalFlip.html), [`v2.RandomVerticalFlip`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomVerticalFlip.html), [`v2.RandomCrop`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomCrop.html), [`v2.RandomRotation`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomRotation.html), [`v2.RandomAffine`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomAffine.html), [`v2.ColorJitter`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.ColorJitter.html), [`v2.GaussianBlur`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.GaussianBlur.html), [`v2.RandomApply`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomApply.html).
- [`tv_tensors`](https://docs.pytorch.org/vision/stable/tv_tensors.html): [`Image`](https://docs.pytorch.org/vision/stable/generated/torchvision.tv_tensors.Image.html), [`Mask`](https://docs.pytorch.org/vision/stable/generated/torchvision.tv_tensors.Mask.html) — tell the transforms which input is a label.
- [`torch.utils.data.Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) — where `transform` is applied.
- [Adversarial example generation (FGSM tutorial)](https://docs.pytorch.org/tutorials/beginner/fgsm_tutorial.html) — the attack in full.
- [`Tensor.requires_grad_`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.requires_grad_.html), [`torch.sign`](https://docs.pytorch.org/docs/stable/generated/torch.sign.html), [`torch.clamp`](https://docs.pytorch.org/docs/stable/generated/torch.clamp.html) — the three pieces of FGSM.

Related: [[5-deep-networks/semantic-segmentation]] · [[5-deep-networks/pooling]] · [[5-deep-networks/dropout]] · [[4-pytorch-training/datasets-and-dataloaders]] · [[4-pytorch-training/experimental-setup]] · [[2-calculus-and-optimization/autograd]]
