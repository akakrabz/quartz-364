import { loadQuartzConfig, loadQuartzLayout } from "./quartz/plugins/loader/config-loader"
import * as ExternalPlugin from "./.quartz/plugins"

// Order the Explorer ("Course map"): topic folders by name (numeric prefixes fix the
// unit order), exam folders last, and pages inside a unit in course order (list below).
//
// NOTE: this function is serialized with .toString() and re-run in the browser,
// so it must stay self-contained (no imports, no outer variables).
ExternalPlugin.Explorer({
  sortFn: (a, b) => {
    if (a.isFolder !== b.isFolder) {
      return a.isFolder ? -1 : 1
    }
    // exam folders (6-midterm-1, …) after all topic units
    const ea = (a.slugSegment ?? "").includes("midterm") ? 1 : 0
    const eb = (b.slugSegment ?? "").includes("midterm") ? 1 : 0
    if (ea !== eb) {
      return ea - eb
    }
    const order = ["what-is-machine-learning", "tensors", "storage-and-views", "indexing-and-slicing", "broadcasting", "linear-algebra-in-pytorch", "gradients-and-partial-derivatives", "matrix-calculus", "primal-optimization", "gradient-descent", "computational-graphs", "backpropagation", "autograd", "fitting-models-with-gradient-descent", "linear-regression", "multiple-linear-regression", "basis-function-regression", "computational-cost", "ridge-regression", "logistic-regression", "binary-cross-entropy", "multiclass-logistic-regression", "nn-module", "loss-functions", "optimizers", "learning-rate-schedulers", "training-loop", "datasets-and-dataloaders", "experimental-setup", "gpu-training", "multilayer-perceptrons", "activation-functions", "convolution", "convolutional-layers", "pooling", "cnn-architectures", "branching-and-inception", "residual-connections", "fully-convolutional-networks", "semantic-segmentation", "data-augmentation", "batch-normalization", "dropout", "recurrent-neural-networks", "bidirectional-and-stacked-rnns", "lstm-and-gru", "text-encodings-and-embeddings", "tokenization", "sequence-to-sequence-rnns", "attention", "transformer-architecture", "layer-norm-and-feed-forward", "positional-encoding", "multi-head-attention", "attention-is-all-you-need", "brownian-motion-and-forward-diffusion", "reverse-time-sde-and-score-matching", "ddpm", "fast-sampling-ddim-and-ode-solvers", "guidance-and-latent-diffusion", "flow-matching-and-rectified-flow", "modern-image-generators", "fall-2024", "spring-2025", "fall-2025", "spring-2026", "review-tensors", "review-gradient-descent", "review-computational-graphs", "review-autograd", "review-linear-regression", "review-logistic-regression", "formula-sheet", "true-false", "practice-problems", "spring-only-topics", "errata"]
    const ra = order.indexOf(a.slugSegment ?? "")
    const rb = order.indexOf(b.slugSegment ?? "")
    if (ra !== rb) {
      return (ra < 0 ? 999 : ra) - (rb < 0 ? 999 : rb)
    }
    return (a.slugSegment ?? "").localeCompare(b.slugSegment ?? "", undefined, {
      numeric: true,
      sensitivity: "base",
    })
  },
})

const config = await loadQuartzConfig()
export default config
export const layout = await loadQuartzLayout()
