---
title: "Modern image generators"
description: "The architectures behind current text-to-image systems, as of October 2026, from primary sources: diffusion transformers (DiT: patchify, adaLN-Zero, Gflops scaling), MMDiT and Stable Diffusion 3, the FLUX family from FLUX.1 to FLUX 3 Image Klein, Imagen, DALL·E 3's synthetic captions, SDXL, PixArt-α, SD3.5, Qwen-Image, ERNIE-Image and i1, few-step distilled generators (ADD, LADD, LCM, guidance distillation), autoregressive competitors (GPT-4o image generation, HunyuanImage 3.0), a sourced comparison table, and a step-by-step walkthrough of how a prompt becomes an image."
tags: [topic, diffusion]
aliases: ["diffusion transformer", "DiT", "adaLN-Zero", "MMDiT", "MM-DiT", "Stable Diffusion 3", "SD3", "SDXL", "FLUX", "FLUX.1", "text-to-image models"]
---

*Beyond the lectures · [[8-diffusion-models/index|Diffusion models]] · sourced from the original papers, official model cards and release posts, as of October 2026 · builds on [[8-diffusion-models/guidance-and-latent-diffusion|guidance and latent diffusion]], [[8-diffusion-models/flow-matching-and-rectified-flow|flow matching]] and [[7-transformers/transformer-architecture|the Transformer]]*

> [!abstract] In one breath
> Between 2022 and 2026 text-to-image generation converged on one recipe: a frozen text encoder, the latent space of a pretrained autoencoder, a **transformer** that denoises latent patches, training by **flow matching** (rectified flow), and sampling with an ODE solver plus classifier-free guidance — or a distilled model that needs one to a few steps and no guidance. DiT showed that transformer denoisers improve predictably with compute; MMDiT (Stable Diffusion 3) let text and image tokens attend to each other in one sequence; FLUX, Qwen-Image and others scaled the design and replaced CLIP and T5 by large language or vision-language models. Autoregressive generators (GPT-4o image generation, HunyuanImage 3.0) are the main alternative. Only what the developers published is reported here.

## What it is

The [[8-diffusion-models/guidance-and-latent-diffusion|latent diffusion]] recipe of 2022 had four parts: an autoencoder, a text encoder, a convolutional [[5-deep-networks/fully-convolutional-networks|U-Net]] denoiser with cross-attention, and a diffusion objective with guidance. Later systems changed each part:

| part | Stable Diffusion v1 (2022) | typical 2024–2026 open model |
| --- | --- | --- |
| autoencoder | $f=8$, 4 latent channels | $f=8$ with more channels (SD3: 16), retrained VAEs (FLUX.2) |
| text encoder | CLIP ViT-L/14 | T5-XXL alongside CLIP, then LLMs / VLMs (Mistral, Qwen) |
| denoiser | U-Net + cross-attention | transformer over latent patches (DiT, MMDiT, single-stream) |
| objective | $\varepsilon$-prediction diffusion | flow matching / rectified flow, logit-normal $t$ |
| sampling | 50 steps, CFG 7.5 | about 50 steps with CFG, or 1–16 steps when distilled |

> [!definition] Definition — diffusion transformer
> A **diffusion transformer** (DiT) cuts the noisy latent into $p\times p$ patches, embeds each patch as a token, runs a stack of [[7-transformers/transformer-architecture|Transformer]] blocks over the tokens, and maps each output token back to a patch of predicted noise or velocity. The timestep and global conditioning enter through **adaptive layer norm**; the text enters through cross-attention, or as extra tokens in the same sequence.

## What you should know

- **Patchify:** an $I\times I\times C$ latent with patch size $p$ becomes $T=(I/p)^2$ tokens of $p^2C$ numbers; halving $p$ quadruples $T$ and the compute, not the parameter count.
- **adaLN-Zero** (Peebles & Xie): LayerNorm's scale and shift, plus a gate $\alpha$ before each residual addition, are regressed from the timestep and class embeddings; $\alpha$ starts at zero, so every block starts as the identity.
- **DiT scaling:** FID tracks **Gflops**, not parameters; DiT-XL/2 (118.6 Gflops) reached FID 2.27 on ImageNet 256×256 with guidance.
- **MMDiT** (Esser et al., SD3): text and image tokens in one sequence with **separate weights per modality**, joined only inside attention; both streams are updated in every block. SD3's largest model: depth 38, 8B parameters.
- **FLUX.1:** 12B rectified-flow transformer of double-stream (MMDiT) and single-stream blocks; CLIP ViT-L/14 + T5-XXL. [dev] is guidance-distilled; [schnell] is adversarially distilled to 1–4 steps.
- **Text encoders grew:** CLIP → T5-XXL (Imagen, PixArt-α) → CLIP + T5 (SD3, FLUX.1) → VLMs (Qwen2.5-VL in Qwen-Image, Mistral Small 3.1 in FLUX.2 [dev], Qwen3-VL in FLUX 3 Image Klein).
- **Captions matter as much as encoders:** DALL·E 3 trained on 95% descriptive synthetic captions; PixArt-α on dense pseudo-captions.
- **Few-step models** are distilled: ADD and LADD (adversarial), LCM (consistency); they run without CFG.
- **Autoregressive alternative:** GPT-4o image generation is autoregressive according to its system card; HunyuanImage 3.0 is an autoregressive MoE LLM that models image latents with diffusion inside the same transformer.

## Diffusion transformers (DiT)

Peebles & Xie (2023) replaced the U-Net of a latent diffusion model by a Vision-Transformer-like network and studied how it scales.

**Patchify.** For 256×256 images the Stable Diffusion autoencoder gives a $32\times32\times4$ latent. Cutting it into $p\times p$ patches gives $T=(32/p)^2$ tokens — 256 for $p=2$, 64 for $p=4$, 16 for $p=8$ — each linearly embedded to the hidden width, with fixed sine–cosine [[7-transformers/positional-encoding|positional embeddings]]. A [`nn.Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) with `kernel_size = stride = p` is exactly "cut into patches, then apply one linear map to each". After the last block a linear decoder maps each token back to a $p\times p\times C$ patch (DiT predicts noise and a diagonal covariance), and the patches are put back in place.

**Conditioning.** DiT compared four ways to feed in the timestep $t$ and class $c$: as two extra tokens (*in-context*), by *cross-attention*, by *adaptive layer norm* (adaLN), and by **adaLN-Zero**. In adaLN the [[7-transformers/layer-norm-and-feed-forward|LayerNorm]] has no learned affine parameters of its own; its scale $\gamma$ and shift $\beta$ are regressed from the sum of the embeddings of $t$ and $c$. adaLN-Zero also regresses a per-channel gate $\alpha$ applied just before each [[5-deep-networks/residual-connections|residual]] addition,

$$
h\leftarrow h+\alpha_1\odot\operatorname{Attn}\big(\gamma_1\odot\operatorname{LN}(h)+\beta_1\big),\qquad
h\leftarrow h+\alpha_2\odot\operatorname{MLP}\big(\gamma_2\odot\operatorname{LN}(h)+\beta_2\big),
$$

and initializes the regressor so that $\alpha=0$: every block starts as the identity. At 400K steps the adaLN-Zero model's FID was nearly half that of the in-context model, while costing the least (118.6 Gflops, against 137.6 for cross-attention). A sketch in PyTorch (the $1+\gamma$ form keeps the initial scale at 1):

```python
class AdaLNZeroBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)      # gates start at zero
        nn.init.zeros_(self.ada[1].bias)        # so block = identity

    def forward(self, x, c):                    # x: (B, T, d), c: (B, d)
        b1, g1, a1, b2, g2, a2 = self.ada(c)[:, None].chunk(6, dim=-1)
        h = (1 + g1) * self.norm1(x) + b1
        x = x + a1 * self.attn(h, h, h, need_weights=False)[0]
        h = (1 + g2) * self.norm2(x) + b2
        return x + a2 * self.mlp(h)
```

**Scaling.** Across DiT-S, B, L, XL and patch sizes 8, 4, 2, the FID after 400K steps correlates strongly and negatively with the model's Gflops; configurations with similar Gflops (DiT-S/2 and DiT-B/4) reach similar FID. Parameters alone do not decide: DiT-S keeps its 33M parameters from $p=8$ to $p=2$ while its compute grows from 0.36 to 6.06 Gflops and its FID (no guidance) falls from 153.6 to 68.4. Trained for 7M steps and sampled with classifier-free guidance, DiT-XL/2 reached FID 2.27 on class-conditional ImageNet 256×256 (previous best 3.60, by LDM), at 118.6 Gflops against 103.6 for LDM-4 and 1,120 for the pixel-space ADM.

PixArt-α (Chen et al., 2023) turned DiT into a text-to-image model: cross-attention between self-attention and feed-forward in each block, and **adaLN-single** — one set of shifts and scales computed from $t$ alone and shared across blocks, with a learned per-block offset — because the per-block adaLN regressors held 27% of DiT's parameters.

## MMDiT and Stable Diffusion 3

Esser et al. (2024) kept DiT's modulation (from the timestep and a pooled text vector $c_{\text{vec}}$) but changed how the token sequence of the text, $c_{\text{ctxt}}$, enters. Image latents are cut into $2\times2$ patches, giving $\tfrac h2\cdot\tfrac w2$ tokens; text and image tokens are embedded to a common width and **concatenated into one sequence**. Because the two modalities are so different, each has its own weights for the projections, MLPs and modulation: the block is "equivalent to having two independent transformers for each modality, but joining the sequences of the two modalities for the attention operation".

$$
\big[O_x;\,O_c\big]=\operatorname{softmax}\!\Big(\frac{[Q_x;Q_c]\,[K_x;K_c]^\top}{\sqrt{d_h}}\Big)[V_x;V_c],
$$

with $Q_x,K_x,V_x$ computed from the image tokens by image weights and $Q_c,K_c,V_c$ from the text tokens by text weights. Unlike [[8-diffusion-models/guidance-and-latent-diffusion#text-conditioning-through-cross-attention|cross-attention]], where the text is a fixed set of keys and values, the text tokens here also attend to the image and are **updated in every block**. The cost is attention over $n+m$ tokens instead of $n$.

```python
def joint_attention(x, c, img, txt, heads):
    """MMDiT-style: separate weights, one attention."""
    n = x.shape[1]                                  # number of image tokens
    q, k, v = torch.cat([img.qkv(x), txt.qkv(c)], dim=1).chunk(3, dim=-1)
    q, k, v = (u.unflatten(-1, (heads, -1)).transpose(1, 2) for u in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
    return img.out(o[:, :n]), txt.out(o[:, n:])     # split back per modality
```

(Modulation, QK-normalization and the MLPs are left out; `img.qkv` and `txt.qkv` are separate [`nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) layers.)

What else SD3 fixed, from the paper:

- **Size by depth.** A model of depth $d$ has hidden width $64d$, MLP width $4\cdot64d$ and $d$ attention heads; the largest, depth 38, has 8B parameters. Validation loss falls smoothly with model size and training steps.
- **Training objective:** rectified flow with logit-normal timesteps and a resolution-dependent shift ($\alpha=3$ at 1024²) — derived on [[8-diffusion-models/flow-matching-and-rectified-flow#why-stable-diffusion-3-switched-and-how-it-samples-t|the flow-matching page]].
- **A 16-channel autoencoder.** Reconstruction FID 2.41, 1.56 and 1.06 for 4, 8 and 16 latent channels; the 16-channel latent is harder to model but scales better.
- **Three frozen text encoders:** CLIP-L/14, OpenCLIP bigG/14 and T5-XXL, each dropped independently 46.3% of the time in training, so any subset works at inference. Leaving out T5 saves memory and stays competitive for most prompts, but hurts complex prompts and spelled-out text.
- **QK-normalization** (RMSNorm on queries and keys) to stop attention logits from growing and training from diverging at high resolution.

**Stable Diffusion 3.5** (October 2024) followed with Large (8.1B parameters), Large Turbo (distilled, 4 steps) and Medium (2.5B, "MMDiT-X") under the Stability AI Community License; text encoders, VAE and noise schedule are those of SD3 Medium. According to the model cards, both models use QK-normalization, and Medium's MMDiT-X adds extra self-attention ("dual attention") blocks in its first transformer layers.

## FLUX

Black Forest Labs released **FLUX.1** on 1 August 2024: "a hybrid architecture of multimodal and parallel diffusion transformer blocks and scaled to 12B parameters", trained with flow matching, with rotary positional embeddings and parallel attention layers. The double-stream blocks are MMDiT blocks; the single-stream ("parallel") blocks run the concatenated text and image tokens through **shared** weights — 19 double- and 38 single-stream blocks in FLUX.1. The text encoders are CLIP ViT-L/14 and T5 v1.1-XXL. Three variants: [pro] (API only), [dev] (open weights, non-commercial license, **guidance-distilled** from [pro]: the guidance strength is an input, about 50 steps), and [schnell] (Apache 2.0, distilled with latent adversarial diffusion distillation to **1–4 steps**, run with guidance 0).

**FLUX.2** (25 November 2025) is a new model trained from scratch: a rectified-flow transformer coupled with a 24B vision-language model (Mistral Small 3.1, whose intermediate-layer outputs are stacked as the prompt embedding), a retrained VAE released under Apache 2.0, and up to 10 reference images for editing. Its [dev] checkpoint has 32B parameters, is guidance-distilled and non-commercial; the balance shifts towards single-stream blocks (8 double, 48 single), with shared modulation across blocks and no bias parameters. **FLUX.2 [klein]** (15 January 2026) comes in 4B (Apache 2.0) and 9B (non-commercial; 8B Qwen3 text embedder) sizes, as 4-step distilled models and 50-step base models.

**FLUX 3 Image Klein** (15 October 2026) is a 5.6B open-weights image transformer — the count excludes its Qwen3-VL-4B-Instruct text encoder — using the FLUX.2 autoencoder, for generation and editing with up to 10 reference images. A step-distilled version runs 16 steps at CFG 1, the base model 50 steps at CFG 3.5; both are under the FLUX Kommunity License (non-commercial use, plus commercial use of outputs for users under US$5M annual revenue). Its model card says not to use it without a **prompt upsampler**: it expects full scene captions, produced by BFL's upsampler or another LLM.

## Other systems that shaped the field

- **Imagen** (Google, 2022): a frozen **T5-XXL** text encoder and a pixel-space cascade — a 64×64 diffusion model and two text-conditional super-resolution models to 256×256 and 1024×1024, all with text cross-attention. Its key finding: embeddings from large language models pretrained on text only are remarkably effective for text-to-image. It also introduced sampling techniques that tolerate large guidance weights.
- **SDXL** (Stability AI, 2023): a U-Net three times larger (2.6B) with most attention moved to lower resolutions, two text encoders (CLIP ViT-L and OpenCLIP ViT-bigG, outputs concatenated to a 2,048-wide context, plus the pooled bigG embedding), multi-aspect training and a separate refinement model applied image-to-image; weights released.
- **DALL·E 3** (OpenAI, 2023): the contribution is the **data**. A captioner fine-tuned to write long, descriptive captions relabels the training images; DALL·E 3 is trained on 95% synthetic and 5% original captions, after ablations (with T5-conditioned diffusion models) showed higher synthetic blends always improved CLIP score.
- **PixArt-α** (2023): DiT with cross-attention and adaLN-single, Flan-T5-XXL (4.3B) with 120 text tokens, the frozen LDM VAE, training split into pixel-dependency, text-alignment and aesthetics stages, and dense captions from a vision-language model — about 753 A100 GPU-days, 12% of Stable Diffusion v1.5's 6,250.
- **Qwen-Image** (Alibaba, August 2025): a 20B MMDiT conditioned on the last hidden layer of **Qwen2.5-VL**, a VAE with the Wan-2.1 encoder (frozen) and a fine-tuned image decoder, trained with flow matching and logit-normal $t$, with a curriculum aimed at text rendering, especially Chinese; Apache 2.0. Updates followed (Qwen-Image-2512 in December 2025, Qwen-Image-2.0 announced in February 2026).
- **ERNIE-Image** (Baidu, 2026): a latent model with an 8B **single-stream** DiT, the FLUX.2 VAE and a small Ministral-3 (3B) text encoder, plus a 3B prompt enhancer; about 50 steps, and an 8-step Turbo variant distilled with DMD and reinforcement learning.
- **i1** (Princeton, June 2026): a fully open 3B model (weights, code and data pipeline, public data only) built from 300+ controlled experiments: an MMDiT with long skip connections, the FLUX.2 VAE, T5Gemma-2B, and no AdaLN at all. Its survey sentence summarizes the field: despite alternative paradigms, most leading text-to-image systems use DiTs trained with flow matching, with text fed by cross-attention, single-stream or dual-stream (MMDiT) designs.

## Few-step generators

Sampling cost is (steps) × (network evaluations per step) × (cost per evaluation), and CFG doubles the middle factor. Distillation attacks the first two; the methods are on the [[8-diffusion-models/fast-sampling-ddim-and-ode-solvers|fast-sampling page]], and the image-generation versions are:

- **Adversarial Diffusion Distillation** (ADD; Sauer et al., 2023): a student trained with an adversarial loss (discriminator on pretrained image features) plus score distillation from a frozen teacher; **1–4 steps**, no CFG at inference. SDXL-Turbo is ADD applied to SDXL; with four steps ADD-XL beat its SDXL teacher at 512².
- **Latent ADD** (LADD; Sauer et al., 2024): the discriminator uses the generative features of a pretrained latent diffusion model instead of a DINOv2 network applied to decoded pixels, which simplifies training at high resolution; SD3-Turbo matches its 50-step teacher's image quality in **four unguided steps**. FLUX.1 [schnell] was trained with it.
- **Latent Consistency Models** (LCM; Luo et al., 2023): consistency distillation in latent space, learning to jump to the solution of the guided probability-flow ODE; **2–4 steps**, distilled in about 32 A100 GPU-hours.
- **Guidance distillation** (FLUX.1 [dev], FLUX.2 [dev]): the guidance strength becomes an input, halving the evaluations per step without cutting the number of steps.

By 2026 releases typically pair a **base** model (about 50 steps, CFG, for fine-tuning) with a **distilled** one: SD3.5 Large / Large Turbo (4 steps), FLUX.2 [klein] Base / distilled (50 / 4), FLUX 3 Image Klein Base / distilled (50 / 16), ERNIE-Image / Turbo (50 / 8).

## Autoregressive competitors

Not every leading system is a diffusion transformer. OpenAI's system card for **4o image generation** (March 2025) states: "Unlike DALL·E, which operates as a diffusion model, 4o image generation is an autoregressive model natively embedded within ChatGPT"; no further architecture is published. **HunyuanImage 3.0** (Tencent, September 2025) builds image generation into an LLM: a decoder-only Mixture-of-Experts transformer (Hunyuan-A13B, over 80B parameters in total, 64 experts of which 8 are active, about 13B active per token) with a vision encoder and a VAE attached; text is predicted token by token, and images are modelled by diffusion on VAE latents inside the same transformer, as in Transfusion. Its authors call it the largest open-source image generation model to date. The boundary is blurring: an "autoregressive" multimodal model may still use diffusion for the pixels.

## Comparison

Only published facts are listed; "—" means not stated in the sources used here. "Objective" is the training objective of the image generator; "steps" are the numbers given in official documentation, not tuned optimum values.

| system (year) | backbone | latent? | objective | text encoder(s) | steps | open weights? |
| --- | --- | --- | --- | --- | --- | --- |
| Stable Diffusion v1 (2022) | U-Net + cross-attention, 860M | yes, $f=8$, 4 ch | diffusion, $\varepsilon$-prediction | CLIP ViT-L/14 | 50 (reference script) | yes |
| Imagen (2022) | U-Net cascade 64→256→1024 | no (pixels) | diffusion | T5-XXL | — | no |
| SDXL (2023) | U-Net 2.6B + refiner | yes | diffusion | CLIP ViT-L + OpenCLIP ViT-bigG | — | yes |
| DALL·E 3 (2023) | — | yes (latent decoder) | diffusion | T5 in the paper's ablations | — | no |
| PixArt-α (2023) | DiT + cross-attention | yes (LDM VAE) | diffusion | Flan-T5-XXL | — | yes |
| SD3 / SD3.5 (2024) | MMDiT; 3.5 Large 8.1B, Medium 2.5B | yes, 16 ch | rectified flow | CLIP-L + OpenCLIP bigG + T5-XXL | Large Turbo: 4 | SD3.5: yes |
| FLUX.1 (2024) | 12B, double- + single-stream blocks | yes | flow matching | CLIP ViT-L/14 + T5 v1.1-XXL | [dev] ≈50; [schnell] 1–4 | [dev], [schnell] |
| 4o image generation (2025) | autoregressive | — | — | — | — | no |
| Qwen-Image (2025) | MMDiT 20B | yes (Wan-2.1-based VAE) | flow matching | Qwen2.5-VL | — | yes |
| HunyuanImage 3.0 (2025) | autoregressive MoE, 80B+ total, 13B active | yes (VAE) | next-token + diffusion | the LLM itself | — | yes |
| FLUX.2 (2025–26) | [dev] 32B; [klein] 4B, 9B | yes (new VAE) | flow matching | Mistral Small 3.1 ([dev]); Qwen3 8B ([klein] 9B) | [klein] 4; Base 50 | yes |
| ERNIE-Image (2026) | single-stream DiT 8B | yes (FLUX.2 VAE) | — | Ministral-3 3B | 50; Turbo 8 | yes |
| i1 (2026) | MMDiT 3B, no AdaLN | yes (FLUX.2 VAE) | flow matching | T5Gemma-2B | — | yes, with code and data pipeline |
| FLUX 3 Image Klein (2026) | transformer 5.6B | yes (FLUX.2 autoencoder) | — | Qwen3-VL-4B-Instruct | 16 at CFG 1; Base 50 at CFG 3.5 | yes |

Three trends are visible: U-Nets gave way to transformers over latent patches; diffusion objectives gave way to flow matching; and text encoders grew from CLIP to multimodal LLMs, often with an LLM rewriting the prompt first.

## How a prompt becomes an image

Here is the whole unit in one pass, for an SD3-style model at 1024×1024 (SD3's time convention: data at $t=0$, noise at $t=1$).

<!-- fig:diff-c-pipeline -->
<figure class="cs-fig"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 300" width="740" height="300" role="img" style="font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-size:13px"><title>Pipeline: prompt to text encoders, noise latent to patchify, diffusion transformer loop, VAE decoder</title><defs><marker id="ARR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs><rect x="10" y="30" width="120" height="52" rx="8" fill="var(--paper)" stroke="var(--hi)" stroke-width="1.6"/><text x="70" y="48" text-anchor="middle" dominant-baseline="central" fill="var(--hi)" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">prompt</text><text x="70" y="64" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">text</text><line x1="132" y1="56" x2="162" y2="56" stroke="var(--hi)" stroke-width="1.5" stroke-linecap="round"/><path d="M162.0,56.0 L155.0,59.5 L155.0,52.5 Z" fill="var(--hi)"/><rect x="164" y="30" width="150" height="52" rx="8" fill="var(--paper)" stroke="var(--hi)" stroke-width="1.6"/><text x="239" y="48" text-anchor="middle" dominant-baseline="central" fill="var(--hi)" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">text encoder(s)</text><text x="239" y="64" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">CLIP / T5 / VLM</text><line x1="316" y1="56" x2="360" y2="100" stroke="var(--hi)" stroke-width="1.5" stroke-linecap="round"/><path d="M360.0,100.0 L352.6,97.5 L357.5,92.6 Z" fill="var(--hi)"/><text x="350" y="70" text-anchor="start" dominant-baseline="central" fill="var(--hi)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">text tokens</text><rect x="10" y="128" width="120" height="52" rx="8" fill="var(--paper)" stroke="currentColor" stroke-width="1.6"/><text x="70" y="146" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">noise</text><text x="70" y="162" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">latent, t = 1</text><line x1="132" y1="154" x2="162" y2="154" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M162.0,154.0 L155.0,157.5 L155.0,150.5 Z" fill="currentColor"/><rect x="164" y="128" width="150" height="52" rx="8" fill="var(--paper)" stroke="var(--accent2)" stroke-width="1.6"/><text x="239" y="146" text-anchor="middle" dominant-baseline="central" fill="var(--accent2)" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">patchify</text><text x="239" y="162" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">2×2 latent patches</text><line x1="316" y1="154" x2="360" y2="154" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M360.0,154.0 L353.0,157.5 L353.0,150.5 Z" fill="currentColor"/><rect x="362" y="92" width="196" height="124" rx="10" fill="var(--paper)" stroke="var(--accent)" stroke-width="1.8"/><text x="460" y="114" text-anchor="middle" dominant-baseline="central" fill="var(--accent)" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">diffusion transformer</text><text x="460" y="136" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">joint attention over</text><text x="460" y="152" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">text + image tokens (MMDiT)</text><text x="460" y="172" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">AdaLN: timestep, pooled text</text><text x="460" y="194" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">predicts velocity (or ε)</text><path d="M 430 218 C 430 270, 540 270, 548 222" fill="none" stroke="var(--accent)" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="5 4"/><line x1="549" y1="230" x2="551" y2="219" stroke="var(--accent)" stroke-width="1.5" stroke-linecap="round"/><path d="M551.0,219.0 L553.2,226.5 L546.3,225.3 Z" fill="var(--accent)"/><text x="590" y="246" text-anchor="start" dominant-baseline="central" fill="var(--accent)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">sampler loop: N steps</text><text x="590" y="262" text-anchor="start" dominant-baseline="central" fill="var(--accent)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">(Euler, DPM-Solver, …;</text><text x="590" y="278" text-anchor="start" dominant-baseline="central" fill="var(--accent)" style="font-size:11px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">classifier-free guidance)</text><line x1="560" y1="120" x2="596" y2="90" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M596.0,90.0 L592.9,97.2 L588.4,91.8 Z" fill="currentColor"/><rect x="598" y="30" width="130" height="52" rx="8" fill="var(--paper)" stroke="var(--accent2)" stroke-width="1.6"/><text x="663" y="48" text-anchor="middle" dominant-baseline="central" fill="var(--accent2)" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">VAE decoder</text><text x="663" y="64" text-anchor="middle" dominant-baseline="central" fill="var(--muted)" style="font-size:12px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;">latent → pixels</text><line x1="663" y1="84" x2="663" y2="104" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M663.0,104.0 L659.5,97.0 L666.5,97.0 Z" fill="currentColor"/><text x="663" y="120" text-anchor="middle" dominant-baseline="central" fill="currentColor" style="font-size:13px;font-family:'Source Sans 3','Source Sans Pro',system-ui,sans-serif;font-weight:600;">image</text></svg><figcaption><b>How a prompt becomes an image in a current latent flow/diffusion transformer</b> (the SD3 / FLUX family). Frozen text encoders turn the prompt into tokens; a noise latent is cut into patches; the transformer predicts a velocity (or noise) for every patch at the current timestep; a sampler repeats this for N steps, applying classifier-free guidance unless the model was guidance-distilled; the VAE decoder turns the final latent into pixels.</figcaption></figure>

1. **Prompt to tokens.** Optionally an LLM expands the prompt into a detailed caption (required by FLUX 3 Image Klein; built into ERNIE-Image). [[7-transformers/tokenization|Tokenizers]] and frozen text encoders turn it into a sequence of token vectors and a pooled vector ([[7-transformers/text-encodings-and-embeddings|embeddings]]). The encoders run once per image, not once per step.
2. **Pure noise.** Draw $z_1\sim\mathcal N(0,\mathbf I)$ in latent shape: with an 8× autoencoder and 16 channels, $128\times128\times16$. This is the end point of the [[8-diffusion-models/brownian-motion-and-forward-diffusion|forward process]], where no information about any image is left.
3. **Patches.** $2\times2$ patches turn the latent into $64\cdot64=4{,}096$ tokens of $2\cdot2\cdot16=64$ numbers each, embedded to the model width and given positional information.
4. **Time grid.** Choose $N$ times from 1 down to 0, shifted towards noise for high resolution ($t\mapsto\alpha t/(1+(\alpha-1)t)$, $\alpha=3$) — [[8-diffusion-models/flow-matching-and-rectified-flow|flow matching]].
5. **One step.** The transformer takes the image tokens, the text tokens and $t$ (through adaLN) and returns a velocity for every patch. For CFG it runs again with the empty prompt (in the same batch) and combines $v_\varnothing+s\,(v_c-v_\varnothing)$; because velocity is an affine function of the score with the same coefficients for both, this is exactly [[8-diffusion-models/guidance-and-latent-diffusion|classifier-free guidance]]. A guidance-distilled model makes one pass instead.
6. **Integrate.** An ODE step moves the latent along the velocity, $z\leftarrow z+(t_{i+1}-t_i)\,v$ with $t_{i+1}<t_i$ — the [[8-diffusion-models/reverse-time-sde-and-score-matching|probability-flow ODE]] in flow coordinates, solved by Euler or a [[8-diffusion-models/fast-sampling-ddim-and-ode-solvers|faster solver]]. Steps 5–6 repeat $N$ times (about 50 for base models, 1–16 for distilled ones).
7. **Decode.** At $t=0$ the latent is a sample from the learned data distribution in latent space; the VAE decoder maps it to $1024\times1024\times3$ pixels, once.

What a model learned is all in step 5, and it was trained by the simplest possible loss — the regression of [[8-diffusion-models/ddpm|DDPM]] or of flow matching on noisy latents. Everything else is the sampler.

```python
z = torch.randn(1, 16, 128, 128)                 # noise latent, t = 1
ts = torch.linspace(1, 0, N + 1)
ts = alpha * ts / (1 + (alpha - 1) * ts)          # resolution shift
for i in range(N):
    v_c = model(z, ts[i], text)                   # conditional velocity
    v_u = model(z, ts[i], empty)                  # unconditional velocity
    v = v_u + s * (v_c - v_u)                     # CFG, s = 1 + w
    z = z + (ts[i + 1] - ts[i]) * v               # Euler, time decreasing
image = vae.decode(z)
```

(A sketch: `model`, `text`, `empty` and `vae` stand for the real components, and real pipelines also rescale latents and patchify inside the model. [`torch.randn`](https://docs.pytorch.org/docs/stable/generated/torch.randn.html) and [`torch.linspace`](https://docs.pytorch.org/docs/stable/generated/torch.linspace.html) build the noise and the time grid.)

> [!trap] Common mistakes
> - **Comparing parameter counts across model cards.** They count different things: FLUX 3 Image Klein's 5.6B excludes its 4B text encoder and its 9B prompt upsampler; HunyuanImage 3.0 has over 80B parameters but about 13B active per token.
> - **Equating parameters with compute.** In DiT, a smaller patch size keeps the parameters and multiplies the tokens and Gflops; FID follows Gflops.
> - **Running CFG on a guidance-distilled or step-distilled model.** The guidance input of FLUX.1 [dev] is not true CFG, and turbo/schnell models are run at guidance 1 or 0 as documented.
> - **Treating MMDiT as cross-attention.** In MMDiT the text tokens are updated in every block and attend to the image; in cross-attention they are fixed keys and values.
> - **Assuming one time convention.** SD3 puts data at $t=0$; Liu et al., Lipman et al. and the Qwen-Image report put data at $t=1$.
> - **Inferring architectures of closed systems.** For 4o image generation the published fact is "autoregressive"; anything more is speculation.

> [!question]- Check yourself
> 1. A DiT works on the $32\times32\times4$ latent of a 256×256 image. How many tokens does it see with $p=2$, $4$, $8$? Why do DiT-S/2 and DiT-B/4 reach similar FID?
> 2. Why does adaLN-Zero initialize the gates $\alpha$ to zero, and what does the network compute at initialization?
> 3. SD3 at 1024×1024 with an 8× autoencoder, 16 channels and $2\times2$ patches: how many image tokens, and how many numbers per token?
> 4. In which blocks do text tokens change: cross-attention (SD v1, PixArt-α) or MMDiT (SD3)? Why?
> 5. The FLUX.1 [schnell] model card runs 4 steps with guidance 0. Why no classifier-free guidance?
> 6. Why can CFG be applied to velocities $v_\varnothing+s(v_c-v_\varnothing)$ just as to noise predictions?
>
> > [!solution]- Answers
> > 1. $(32/p)^2$: 256, 64, 16 tokens. They have similar Gflops (more tokens at a smaller width versus fewer tokens at a larger width), and DiT's FID tracks Gflops.
> > 2. With $\alpha=0$ every residual branch is multiplied by zero, so each block is the identity and the whole network starts as a simple map from input patches to output patches; training then opens the gates gradually, which DiT found trains much better than vanilla adaLN.
> > 3. Latent $128\times128\times16$; $64\times64=4{,}096$ tokens of $2\cdot2\cdot16=64$ numbers each (before embedding to the model width).
> > 4. Only in MMDiT: the text tokens are part of the joint sequence and have their own projections and MLPs in every block. In cross-attention they are only read, as keys and values.
> > 5. [schnell] was distilled (LADD) to produce good samples in 1–4 steps without guidance; CFG would double the cost and push it off the distribution it was trained to produce.
> > 6. For a Gaussian path, $v=a(t)\,x+b(t)\,\nabla\log p_t$ with coefficients that do not depend on the condition. So $v_\varnothing+s(v_c-v_\varnothing)=a\,x+b\,[\nabla\log p_t(x)+s(\nabla\log p_t(x\mid c)-\nabla\log p_t(x))]$: the velocity of the guided score.

## References

Papers:

- Peebles, Xie (2023). *Scalable Diffusion Models with Transformers.* ICCV 2023. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) — patchify, adaLN-Zero, Gflops scaling, FID 2.27.
- Esser et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3). ICML 2024. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206) — MM-DiT (Sec. 4), autoencoder channels (Tab. 3), QK-normalization, text-encoder dropout, scaling.
- Saharia et al. (2022). *Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding* (Imagen). [arXiv:2205.11487](https://arxiv.org/abs/2205.11487)
- Podell et al. (2023). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.* ICLR 2024. [arXiv:2307.01952](https://arxiv.org/abs/2307.01952)
- Betker et al. (2023). *Improving Image Generation with Better Captions* (DALL·E 3). [cdn.openai.com/papers/dall-e-3.pdf](https://cdn.openai.com/papers/dall-e-3.pdf)
- Chen et al. (2023). *PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis.* ICLR 2024. [arXiv:2310.00426](https://arxiv.org/abs/2310.00426)
- Qwen Team (2025). *Qwen-Image Technical Report.* [arXiv:2508.02324](https://arxiv.org/abs/2508.02324)
- Tencent Hunyuan (2025). *HunyuanImage 3.0 Technical Report.* [arXiv:2509.23951](https://arxiv.org/abs/2509.23951)
- ERNIE Team, Baidu (2026). *ERNIE-Image Technical Report.* [arXiv:2605.25347](https://arxiv.org/abs/2605.25347)
- Zeng et al. (2026). *i1: A Simple and Fully Open Recipe for Strong Text-to-Image Models.* [arXiv:2606.11289](https://arxiv.org/abs/2606.11289)
- Sauer, Lorenz, Blattmann, Rombach (2023). *Adversarial Diffusion Distillation.* ECCV 2024. [arXiv:2311.17042](https://arxiv.org/abs/2311.17042)
- Sauer et al. (2024). *Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation.* [arXiv:2403.12015](https://arxiv.org/abs/2403.12015)
- Luo, Tan, Huang, Li, Zhao (2023). *Latent Consistency Models.* [arXiv:2310.04378](https://arxiv.org/abs/2310.04378)

Model cards, release posts and documentation:

- Black Forest Labs: [Announcing Black Forest Labs (FLUX.1)](https://bfl.ai/blog/24-08-01-bfl), [FLUX.1 [schnell] model card](https://huggingface.co/black-forest-labs/FLUX.1-schnell), [FLUX.2](https://bfl.ai/blog/flux-2), [FLUX.2 [dev] model card](https://huggingface.co/black-forest-labs/FLUX.2-dev), [flux2 repository (klein variants)](https://github.com/black-forest-labs/flux2), [FLUX 3 Image Klein](https://bfl.ai/blog/flux-3-image-klein), [FLUX 3 Image Klein model card](https://huggingface.co/black-forest-labs/flux-3-image-klein), [release notes](https://docs.bfl.ml/release-notes).
- Hugging Face: [Diffusers welcomes FLUX.2](https://huggingface.co/blog/flux-2) (block counts, text encoder), [Flux pipeline docs](https://huggingface.co/docs/diffusers/api/pipelines/flux) (text encoders, guidance distillation), [FLUX 3 Image Klein pipeline docs](https://huggingface.co/docs/diffusers/api/pipelines/flux3), [Diffusers welcomes SD3.5](https://huggingface.co/blog/sd3-5).
- Stability AI: [Introducing Stable Diffusion 3.5](https://stability.ai/news-updates/introducing-stable-diffusion-3-5); [sd3.5 reference code](https://github.com/Stability-AI/sd3.5) (text encoders, 16-channel VAE); model cards for [SD3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) (QK-normalization) and [SD3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) (MMDiT-X).
- OpenAI: [Addendum to GPT-4o System Card: 4o image generation](https://openai.com/index/gpt-4o-image-generation-system-card-addendum/) (March 2025).
- Qwen: [Qwen-Image repository](https://github.com/QwenLM/Qwen-Image) (license, later releases). Baidu: [ERNIE-Image model card](https://huggingface.co/baidu/ERNIE-Image). PixArt-α: [model card](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS).

## PyTorch docs

- [`torch.nn.Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — `kernel_size = stride = p` is patchify plus a linear patch embedding.
- [`torch.nn.LayerNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) — `elementwise_affine=False` for adaptive layer norm, whose scale and shift come from the conditioning.
- [`torch.nn.init`](https://docs.pytorch.org/docs/stable/nn.init.html) — `zeros_` for the adaLN-Zero gates.
- [`torch.nn.MultiheadAttention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) and [`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — self- and joint attention.
- [`torch.Tensor.unflatten`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unflatten.html) — split the width into heads.
- [`torch.linspace`](https://docs.pytorch.org/docs/stable/generated/torch.linspace.html) and [`torch.randn`](https://docs.pytorch.org/docs/stable/generated/torch.randn.html) — time grid and starting noise of a sampler.

Related: [[8-diffusion-models/brownian-motion-and-forward-diffusion]] · [[8-diffusion-models/reverse-time-sde-and-score-matching]] · [[8-diffusion-models/ddpm]] · [[8-diffusion-models/fast-sampling-ddim-and-ode-solvers]] · [[8-diffusion-models/guidance-and-latent-diffusion]] · [[8-diffusion-models/flow-matching-and-rectified-flow]] · [[7-transformers/transformer-architecture]] · [[7-transformers/attention]]
