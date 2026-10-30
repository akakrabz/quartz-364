# ECE 364 notes site — how to build, preview, host and extend

This folder is a **Quartz 5** site (`quartz.config.yaml`, `quartz.ts`, `content/`, `quartz/styles/custom.scss`) — the same setup as the ECE 329 site.
The notes are plain Markdown with Obsidian-style wikilinks and callouts, so `content/` also opens directly as an Obsidian vault.

## 1. Build and preview (first time)

Requirements: Node ≥ 22, npm ≥ 10.9, internet access for the two install steps.

```bash
cd ~/"Nextcloud/Notes/ECE 364/quartz-364"
npm ci                        # Quartz's own dependencies
npx quartz plugin install --from-config   # clones + builds the plugins listed in quartz.config.yaml, writes quartz.lock.json
npx quartz build --serve      # http://localhost:8080  (rebuilds on save)
```

`npx quartz build` alone writes the static site to `public/`; add `-c $(nproc)` to parse with every core. Plain `npx quartz plugin install` only reinstalls from `quartz.lock.json`, which git does not track, so on a fresh clone it prints a warning and installs nothing — the build then fails with `Could not resolve "./.quartz/plugins"`. Use `--from-config` as above (or `npm run install-plugins`, which is what the Docker builder runs).

## 2. Hosting

### Docker / Portainer (what `docker-compose.yml` does)

Two services, exactly as for the 329 site:

- **builder** (`node:22-bookworm`) clones `https://github.com/akakrabz/quartz-364.git` into a volume, checks `master` every `SYNC_INTERVAL` (300 s), and rebuilds whenever the branch moves. It builds into a staging folder and swaps it in, so a failed build leaves the previous site serving (on the very first deploy there is no previous site).
- **web** (`nginx:1-alpine`) serves the built site on **port 8364** with the one rewrite rule Quartz needs.

Deploy: Portainer › Stacks › Add stack › paste `docker-compose.yml` (or `docker compose up -d`), and point the reverse proxy for `quartz-364.vops.ch` at port 8364. Publishing is then just `git push`. Needs Docker Compose ≥ 2.23.1 (inline `configs.content`).

### Any static web server

Quartz emits `page.html` files but links to them **without** the extension, and folders with a trailing slash. The server must try `$uri`, then `$uri.html`, then `$uri/`:

```nginx
location / {
    try_files $uri $uri.html $uri/ =404;
}
error_page 404 /404.html;
```

Opening `public/index.html` from disk (file://) does not work — search, graph and previews fetch JSON over HTTP. `configuration.baseUrl` in `quartz.config.yaml` is `quartz-364.vops.ch` (used only for the sitemap and social previews). External requests at page load: Google Fonts and jsdelivr (KaTeX CSS); analytics are off.

## 3. Layout

```
content/index.md                       home + course map
content/lectures.md                    lecture -> topic pages table
content/1-foundations/                 L1-L2: what ML is, tensors, storage and views, indexing, broadcasting, linear algebra
content/2-calculus-and-optimization/   L3-L6: gradients, matrix calculus, optimization, gradient descent, graphs, backprop, autograd
content/3-linear-models/               L7-L9, L11: linear, multiple, basis-function and ridge regression; logistic, BCE, multi-class
content/4-pytorch-training/            L10-L11, L14: nn.Module, losses, optimizers, schedulers, training loop, data, splits, GPU
content/5-deep-networks/               L14-L17: MLPs, activations, CNNs (conv, pooling, architectures, Inception, ResNet, FCN/U-Net,
                                       segmentation, augmentation, batch norm, dropout), RNNs, LSTM/GRU
content/6-midterm-1/                   the four past Midterm 1s worked, review by problem type, formula sheet, T/F bank, practice, errata
content/7-transformers/                L18-L20: embeddings, tokenization, seq2seq RNNs, attention, the Transformer, paper guide
content/8-diffusion-models/            beyond the lectures (on request): diffusion / score-based image generation, sourced from papers
content/images/                        lecture figures, named lNN-<name>.png after the notebook folder they came from
tools/                                 spec, checkers, figure generators (see §5)
```

The site is organized by topic, not by lecture; `lectures.md` maps back. Lecture numbers are the `py-complete/LectureN` folder numbers (the CNN notebooks call themselves Lectures 17 and 18 in their headers). The Explorer order (topic units by folder prefix, exam folders last, pages in course order) is set in `quartz.ts`: add new pages to its list.

## 4. Writing conventions (so new pages match)

`tools/SPEC.md` is the full spec every page was written to; the short version:

- **Topic pages**: frontmatter `title`, `description`, `tags` (`topic`, a unit tag `foundations | calculus | linear-models | training | deep-nets | transformers`, and `midterm-1` for Lectures 2-9), `aliases`. Then an italic meta line (lectures, unit, prerequisites), `> [!abstract] In one breath`, `## What it is`, `## What you should know` (the review checklist), topic sections with math, figures and notebook code, `> [!trap]` and `> [!exam]` callouts, `## PyTorch docs`, and `Related:` as the last line.
- **Code** comes from the lecture notebooks (trimmed); printed output only where the notebook printed it or it can be derived by hand. Comments of 7 words or fewer.
- **PyTorch links**: `https://docs.pytorch.org/docs/stable/generated/<full.name>.html` for APIs, `.../stable/<page>.html#<anchor>` for topic pages (e.g. `data.html#torch.utils.data.DataLoader`). Note `nn.Flatten` lives at `generated/torch.nn.modules.flatten.Flatten.html`.
- **Math**: `$…$` inline, `$$…$$` on their own lines (prefixed with `> ` inside callouts). No macros; inside tables use `\lvert…\rvert` and `\mid`, never a bare `|`.
- **Links** with full paths: `[[3-linear-models/ridge-regression|ridge regression]]`; inside tables escape the pipe: `[[3-linear-models/ridge-regression\|ridge]]`.
- **Callouts**: `definition`, `key`, `recipe`, `trap`, `exam`, `intuition`, `derivation`, `example`, plus Obsidian's own; exam pages use `question` and folded `solution`. Never a blank line inside a callout.
- **Figures**: inline SVG blocks (`<figure class="cs-fig">…</figure>`, one line), generated by `tools/figs_<unit>.py` — plots with `tools/mplfig.py` (matplotlib; draw with its INK/ACCENT/ACCENT2/HI/MUTED colours and they follow the light/dark theme), diagrams with `tools/figlib.py`. A page marks a figure with `<!-- fig:NAME -->`; `python3 tools/assemble.py content/<page>.md` inlines it on the next line (`tools/figs_exam.py` inlines its own, including markers inside callouts). Lecture PNGs are embedded with `![[l05-forward-pass.png|caption]]` and get a light card in dark mode (`custom.scss`).

## 5. Checking without building

```bash
python3 tools/check.py content node_modules/katex/dist/katex.min.js   # links, anchors, embeds, KaTeX (strict), frontmatter, callouts, tables
python3 tools/check_doclinks.py PYTORCH/docs/source VISION/docs/source # every docs.pytorch.org link vs the docs sources
for f in tools/figs_*.py; do python3 "$f"; done; python3 tools/assemble.py   # regenerate and re-inline all figures
```

`check.py` needs PyYAML and a KaTeX copy (present after `npm ci`). `check_doclinks.py` needs sparse clones of the docs sources matching the "stable" docs:

```bash
git clone --depth 1 --filter=blob:none --sparse -b release/2.14 https://github.com/pytorch/pytorch && git -C pytorch sparse-checkout set docs/source
git clone --depth 1 --filter=blob:none --sparse -b release/0.29 https://github.com/pytorch/vision && git -C vision sparse-checkout set docs/source
```

## 6. What's here

Build 3, 2026-10-30 — Lectures 1–20, Midterm 1, and unit 8 (diffusion models, on request).

- **Lectures 19–20** (unit 7): sequence-to-sequence RNNs, attention, the Transformer (Skinny → Medium → Full), LayerNorm and FFN, positional encoding, multi-head attention, and a section-by-section guide to *Attention Is All You Need* (checked against the arXiv v7 text via Exa). Results come from the saved `.npy` files; dataset statistics were recomputed on `eng-fra.txt`.
- **Unit 8, diffusion models** (beyond the lectures): forward diffusion, reverse-time SDE and score matching, DDPM, fast sampling, guidance and latent diffusion, flow matching, modern image generators (as of October 2026). Sourced from the papers and official model cards via Exa; NumPy toys in `tools/diffusion_toy.py` and `tools/flow_toy.py` (all figures computed, not copied).
- `tools/mplfig.py` now compacts SVGs (short ids, rounded coordinates, hoisted marker styles); scatter figures draw at most ~1,200 points per panel.
- Totals: 87 pages (62 topic pages), 115 figures, 748 PyTorch-doc links (275 distinct, all checked), 1,597 links and 6,763 KaTeX expressions passing `check.py`. Every new page was reviewed independently.

Build 2, 2026-10-25 — Lectures 1–18 and Midterm 1. (Build 2 added unit 7, Transformers: Lecture 18's embeddings and tokenization pages, 4 figures in `tools/figs_transformers.py`; the vocabulary-size table on the tokenization page was computed by running the notebook's preprocessing on `reviews.csv`.)

Build 1, 2026-10-20 — Lectures 1–17 and Midterm 1.

- **46 topic pages** in five units (6 + 8 + 8 + 8 + 16), each with *In one breath*, *What you should know*, worked examples, code from the lecture notebooks, traps, exam notes and a *PyTorch docs* list; plus unit index pages, a lecture → topic map, and the home page.
- **Midterm 1 section** (16 pages), built from the earlier Exam 1 guide and the four past exams with solutions: each exam worked problem by problem with folded solutions, six review-by-type pages, formula sheet, 34-item T/F bank, 10 practice problems, Spring-only topics, and the typos found in the official keys and lectures.
- **71 figures** (tools/figs_*.py) and 32 lecture images; **402 links to the PyTorch docs** (249 distinct), all checked against the 2.14 / torchvision 0.29 docs sources or fetched.
- **Verification:** `tools/check.py` passes on all 69 pages (1,241 links and anchors, 4,619 KaTeX expressions). Every unit and the exam section were then reviewed independently against the notebooks and exam PDFs, with arithmetic recomputed; the reviews fixed 58 statements, and no exam answer changed.
- **Not done here:** PyTorch could not be installed where the site was written, so no code was re-run: code and printed outputs come from the notebooks or are derived by hand. The Quartz build itself was not run (npm registry unreachable) — run §1 once before relying on the deployment. Homework claims (HW 1–3) follow the Exam 1 guide; the HW PDFs were not used.

