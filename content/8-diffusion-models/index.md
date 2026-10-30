---
title: "Diffusion models"
description: "Unit 8 (beyond the lectures): how image generators turn noise into pictures by reversing a Brownian-motion-like noising process — the forward process, the reverse-time SDE and score matching, DDPM, fast samplers, guidance and latent diffusion, flow matching, and the diffusion transformers behind current systems — with runnable 2-D toy examples."
tags: [unit, diffusion]
aliases: ["unit 8", "generative models", "image generation"]
---

*Beyond the lectures · not from the ECE 364 notebooks · sourced from the original papers · builds on [[5-deep-networks/index|deep networks]] and [[7-transformers/index|Transformers]]*

> [!note] Not lecture material
> This unit was written on request and goes beyond what Lectures 1–20 cover. Every formula and number comes from the cited papers or official model documentation, and the worked examples are small 1-D and 2-D toys that were actually run (in NumPy). If diffusion models come up in a later lecture, the lecture's notation and scope win.

**The idea in one paragraph.** Take an image and add a little Gaussian noise, again and again, until nothing but noise is left: a **forward process** that behaves like Brownian motion pulling the data toward a standard Gaussian. That direction is easy and needs no learning. The remarkable fact (Anderson, 1982) is that this process has an exact **time reversal** — another stochastic differential equation that runs from noise back to data — and its only unknown ingredient is the **score** $\nabla_x \log p_t(x)$ of the noisy data distribution at each noise level. A neural network trained to **denoise** (predict the added noise) is exactly an estimator of that score. Sampling an image is then: start from pure noise and integrate the reverse process with the learned score. Everything else in this unit — DDPM, DDIM, guidance, latent diffusion, flow matching, diffusion transformers — is a refinement of how to parameterize, train or integrate that reverse process.

## Pages, in reading order

1. [[8-diffusion-models/brownian-motion-and-forward-diffusion|Brownian motion and the forward diffusion]] — the Wiener process, the DDPM noising chain and its closed form $q(x_t\mid x_0)$, noise schedules, and the SDE view (VP and VE).
2. [[8-diffusion-models/reverse-time-sde-and-score-matching|The reverse-time SDE and score matching]] — the score, Langevin dynamics, Anderson's reverse-time SDE ("reverse Brownian motion"), the probability-flow ODE, and denoising score matching, with an exact-score 2-D demonstration.
3. [[8-diffusion-models/ddpm|DDPM]] — Ho et al. (2020): the noise-prediction objective, the training and sampling algorithms, the U-Net, and a toy model trained from scratch.
4. [[8-diffusion-models/fast-sampling-ddim-and-ode-solvers|Fast sampling]] — DDIM, ODE solvers, and distillation / consistency models: from 1,000 network evaluations to a handful.
5. [[8-diffusion-models/guidance-and-latent-diffusion|Guidance and latent diffusion]] — classifier and classifier-free guidance, text conditioning through cross-attention, and diffusion in an autoencoder's latent space (Stable Diffusion).
6. [[8-diffusion-models/flow-matching-and-rectified-flow|Flow matching and rectified flow]] — learning a velocity field along straight paths from noise to data, and why recent models switched to it.
7. [[8-diffusion-models/modern-image-generators|Modern image generators]] — diffusion transformers (DiT, MMDiT), Stable Diffusion 3, FLUX and other current systems, few-step generators, and how a prompt becomes an image.

**Where the course connects.** The denoiser is a [[5-deep-networks/fully-convolutional-networks|U-Net]] with [[5-deep-networks/residual-connections|residual blocks]] or, in newer systems, a [[7-transformers/transformer-architecture|Transformer]]; the noise level is fed in with the same sinusoidal embedding as [[7-transformers/positional-encoding|positional encoding]]; text prompts enter through [[7-transformers/attention|cross-attention]]; and training is plain [[2-calculus-and-optimization/gradient-descent|gradient descent]] on a mean-squared error ([[4-pytorch-training/loss-functions|`MSELoss`]]).
