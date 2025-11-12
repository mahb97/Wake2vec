# Wake2Vec

## TL;DR

Extend a small GPT-style tokenizer with curated *Finnegans Wake* morphemes, initialise new vectors by morpheme composition, and fine-tune with a two-phase protocol (embedding warm-up + full fine-tune or optional three-phase with LoRA) that protects stability. Report geometry shifts (top-k neighbour overlap, embedding norm deltas, isotropy and PIP loss if requested), language behaviour (validation loss and perplexity on held-out Wake slices), and qualitative intrusion. The full pipeline reproduces on a Colab T4.

---

## Why This Project

Style control is often attempted through prompts or full fine-tuning. Wake2Vec explores a third path: an embedding-first intervention that inserts Joyce-specific forms and trains the input layer in a controlled way. The goal is local, interpretable changes to semantic neighbourhoods under tight compute, with results that can be verified and challenged.

---

## Method (Morpheme-Aware)

### Lexicon and Morphology

A hand-curated CSV lists `type ∈ {prefix, suffix}`, `morpheme`, and up to 10 examples per row. Parsed into:
- `morph["prefixes"]: {prefix → [examples...]}`  
- `morph["suffixes"]: {suffix → [examples...]}`

**Morpheme CSV format:**
```csv
type,morpheme,example1,example2,...,example10
prefix,pre-,prepare,preview,prelude
suffix,-ment,government,ailment,fragment
```

### Synthetic Forms

Sample `(prefix, root, suffix)` with frequency weighting to generate Joyce-style words (e.g., `pre+river+ation → priveration`), then wrap in a few hundred short sentences to guarantee coverage.

### Tokenizer Augmentation

New forms are added to the tokenizer as **plain tokens** (bare forms + SentencePiece start-of-word variants `▁token`). I disable mean-resizing when expanding the embedding matrix (`resize_token_embeddings(..., mean_resizing=False)`) so that custom initialisation is preserved, and I tie the output head to the input embeddings so the new vectors participate in prediction.

### Compositional Initialisation

For new token *w* with greedy longest prefix/suffix match *(p, s)* and core *r*, set:
```
E(w) = α·E(p̄) + (1 − 2α)·E(r̄) + α·E(s̄) + ε
```

average embeddings of high-quality example words if a morpheme isn't single-token; ε is small Gaussian noise for diversity. If *r* is unseen, fall back to a small random vector scaled to the embedding std.

---

## Training Protocols

Wake2Vec supports two training approaches depending on your compute budget and goals:

### Two-Phase Protocol (Standard)

**Phase 1: Embedding-only warm-up**  
Freeze everything except `input_embeddings` (+ tied head). Train on synthetic sentences + Wake blocks with Adafactor.

**Typical hyperparameters:**
- Epochs: 1
- Learning rate: 5e-4
- Batch size: 8
- Gradient accumulation: 2
- Warmup ratio: 0.0
- Save steps: 200
- `use_cache=False`
- No fp16 on T4

**Phase 2: Full model fine-tune**  
Unfreeze all parameters. Fine-tune on *Finnegans Wake* with conservative schedules, early stopping on validation loss, and pinned software versions.

**Typical hyperparameters:**
- Epochs: 2
- Learning rate: 2e-5
- Warmup ratio: 0.10
- Batch size: 8
- Gradient accumulation: 2
- Weight decay: 0.01
- Save steps: 200
- Early stopping with patience: 2
- Gradient checkpointing enabled
- No fp16 on T4

### Three-Phase Protocol (Experimental)

**Phase 1: Embeddings-only warm-up**  
Same as two-phase protocol above, but with longer training:
- `max_steps ≈ 800–2000`
- `lr = 5e-4`
- `grad_accum = 16`
- `batch_size = 1`

**Phase 2: LoRA behaviour tune**  
Attach LoRA to attention/MLP, keep embeddings frozen. Train on real Wake blocks.

**Typical hyperparameters:**
- Epochs: 1
- `lr = 2e-5`
- Warmup: 0.10
- `grad_accum = 16`
- `batch_size = 1`
- 8-bit Adam if available

**Phase 3: Embed-alignment++**  
Unfreeze embeddings (+ tied head) and add new-row-only regularisers on top of LM loss:

- **L_anchor** = ‖E_new − E_comp‖² (stay near composition)
- **L_centroid** = 1 − cos(E_new, centroid_pre) (stay near pre-neighbour centroid)
- **L_norm** = (‖E_new‖ − ‖E_pre‖)² (norm hygiene)

**Typical hyperparameters:**
- `max_steps = 400–800`
- `lr = 5e-5`
- Warmup: 0.10
- Adafactor
- Gradients masked to new rows only

---

## Data and Setup

- **Base text**: *Finnegans Wake* plain text (blockified; small held-out slice)
- **Synthetic sentences**: ~600, each containing ≥1 injected token
- **Token additions**: Recent runs added 447–534 new tokens after filtering duplicates (varies by CSV)
- **Tokenizer vocabulary size after expansion**: 33,098 (base ≈ 32k → 32k+Δ)
- **Maximum sequence length**: 2,048 (standard); 384–512 on T4 for memory-constrained runs
- **Datasets**: Blockified Wake text with a held-out set
  - Train blocks: 1,566
  - Valid blocks: 174

---

## Environment (Reproducibility Notes)

**Dependencies:**
- Python 3.12
- `transformers==4.57.1`
- `accelerate==1.2.1`
- `datasets>=2.21.0`
- `pyarrow==22.0.0`
- `peft>=0.11` (for LoRA experiments)
- `bitsandbytes` (optional, for 8-bit optimisers)
- `umap-learn`
- `faiss-cpu`
- `wordfreq`
- `unidecode`
- `matplotlib`

**Colab quirk:** If `Trainer` errors with `unwrap_model(..., keep_torch_compile=...)`, pin `accelerate>=1.2` or apply a tiny compatibility shim.

**Performance notes:**
- Keep `use_cache=False` during training
- Prefer Adafactor or 8-bit Adam on T4
- Avoid fp16 on T4 for this pipeline to maintain stability
- Enable gradient checkpointing in Phase 2 to reduce memory

---

## Results from the Latest Run

This run evaluates a subset of 49 synthetic morphemic tokens with pre and post snapshots using the two-phase protocol.

### Geometry and Stability

- **Mean neighbour overlap (top-5)**: 3.7 of 5
- **Mean embedding norm change**: 0.0051

**Interpretation**: Neighbourhoods remain coherent; the vectors move slightly toward Wake-like regions without collapse or uncontrolled drift.

### Representative Examples

- **conmanes** — overlap 4 of 5, neighbours: manes, conmaning, enmanes, comanes
- **presounder** — overlap 4 of 5, neighbours: presounded, ensounder, resound, soundy
- **soundity** — overlap 3 of 5, shows modest drift yet remains in the sound cluster

### Visual Diagnostics

- t-SNE shows the new tokens clustered near centroids of their pre-training neighbour sets
- Histograms for neighbour overlap and norm change show a stable centre with light positive shift
- Scatter of norm change versus overlap highlights a small tail of tokens to inspect manually

See `results/wake2vec_report.html` for the full static report with figures and tables.

### Three-Phase Results (When Available)

**Phase 1 loss (example first 200 steps):** 6.34 → 4.28 (steady decline)

**Geometry targets:**
- **P1:** *composed-init → post-P1* mean overlap@5 ≈ 2.5–3.5
- **P3:** *post-P2 → post-P3* mean overlap@5 shifts with small +Δ‖E‖ (≈ 0.01–0.08)

> **Note:** After a three-phase run, update this section with actual P1/P3 overlaps, Δ‖E‖, perplexity, and qualitative neighbour examples.

---

## Evaluation

- **Geometry**: Top-k neighbour overlap before and after, embedding norm deltas, optional isotropy and PIP loss
- **Language**: Validation loss and optional perplexity on held-out Wake slice
- **Behaviour**: Short generation probes that seed with high-drift and low-drift tokens, nearest-neighbour maps saved to JSON for audit

---

## Quickstart on T4 or CPU
```bash
pip install -r requirements.txt
```

### Standard Two-Phase Pipeline

1. **Lexicon**: Parse or regenerate the morpheme maps and write `wake_lexicon.txt`
2. **Token injection**: Expand the tokenizer, compose embeddings, tie the head
3. **Training**: Run Phase 1 embedding warm-up, then Phase 2 full fine-tune
4. **Metrics and report**: Write snapshots, compute overlaps, and build `results/wake2vec_report.html`

### Three-Phase Pipeline (Experimental)
```bash
# Place your morphemes at data/morphemes.csv
python wake2vec_morpheme_expansion.py --base_model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T

# See runs/<id>/ for metrics, plots, and the HTML report
```

### Model Choice

- **On GPU**: `BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"` is a good default
- **On CPU**: `BASE_MODEL="distilgpt2"` to smoke-test the pipeline

---

## Artifacts Saved Automatically

### Two-Phase Protocol

- `results/summary_stats.json`, `results/morpheme_comparison.json`
- `results/pre_morpheme_snapshot.json`, `results/post_morpheme_snapshot.json`
- `results/wake2vec_report.html` with t-SNE, histograms, and tables
- `checkpoints/*` and a `run_meta.json` that records hyperparameters and paths

### Three-Phase Protocol
```
runs/<id>/
├── morpheme_data.json
├── synthetic_lines.txt
├── tokenizer adapters (saved early to avoid ID drift)
├── summary_stats_p1.json, morpheme_comparison_p1.json
├── summary_stats_p3.json, morpheme_comparison_p3.json
├── phase2_loss_log.json, phase3_live_log.json
├── plots/
│   ├── hist_overlap_top5.png
│   ├── hist_norm_change.png
│   ├── scatter_norm_vs_overlap.png
│   ├── tsne_newtokens_vs_precentroids.png
│   └── phase loss curves
└── wake2vec_report.html
```

Optional tarball in `archives/`.

---

## Practical Notes

- If `load_best_model_at_end=True`, match `eval_strategy` and `save_strategy` to `"steps"`
- For OOM on T4: reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps`, shorten `MAX_LEN`, or switch Phase 2 to LoRA (recommended)
- Keep random seeds fixed for comparability across phases
- Prefer Adafactor or 8-bit Adam on T4
- Keep fp16 off on T4 for this pipeline
- Set `use_cache=False` during training to reduce memory

---

## Repository Structure

The repository contains multiple notebooks for different aspects of the pipeline:

- **Lexicon generation**: Build morpheme maps from *Finnegans Wake*
- **Token injection**: Expand tokenizer with compositional initialisation
- **Two-phase training**: Standard embedding warm-up + fine-tune
- **Three-phase training**: Experimental LoRA + embed-alignment
- **Evaluation**: Geometry diagnostics, neighbour analysis, visualisation
- **Report generation**: HTML artifacts with plots and tables

Each notebook is designed to be run independently or as part of the full pipeline.

---

## Heartbeat Monitoring System

For long-running training experiments on preemptible compute instances, the repository includes a dedicated monitoring notebook that provides non-invasive inspection of training progress without interfering with active processes.

### Monitoring Capabilities

The heartbeat system tracks:

- Training loss trajectory from JSON logs and trainer state files
- Evaluation metrics at configurable step intervals
- Checkpoint inventory across local ephemeral and persistent storage
- Embedding snapshot presence and modification times
- Age reporting for all artifacts in human-readable format

### Storage Hierarchy

The monitoring system inspects three storage locations:

- **Local ephemeral**: `/content/runs/t4_*` (active training directory)
- **Drive persistent**: `/content/drive/MyDrive/wake2vec/runs/t4_*` (synchronized copy)
- **Sentry backup**: `/content/drive/MyDrive/wake2vec/sentry_backups/t4_*` (safety mirror)

### Checkpoint Validation

Checkpoints are verified by checking for valid weight files (`model.safetensors`, `pytorch_model.bin`, or sharded variants). The system automatically identifies the most recent valid checkpoint suitable for resumption, excluding incomplete or corrupted saves.

### Usage

The monitoring notebook is designed for manual execution at user-defined intervals. Typical usage patterns include hourly checks during active training, post-checkpoint verification after save events, and pre-resume validation before launching continuation runs.

---

## Llama Trials

The `Llama/` directory contains experimental work extending Wake2Vec to larger language models, specifically Meta's Llama 3.1 8B and Llama 3.2 3B architectures.

### Motivation

While the primary Wake2Vec pipeline targets TinyLlama (1.1B parameters) for compute efficiency and rapid iteration, the Llama trials investigate whether the morpheme-aware embedding injection methodology scales to models with substantially larger capacity and more sophisticated language understanding.

### Technical Challenges

Adapting Wake2Vec to Llama models introduced several technical constraints:

**Memory limitations**: Llama 3.1 8B requires quantization (4-bit via bitsandbytes) to fit on Colab T4 GPUs (15GB VRAM). Even with quantization, some configurations required CPU offloading or switching to the smaller Llama 3.2 3B variant.

**Gated model access**: Llama models require explicit approval from Meta via Hugging Face, introducing delays in experimentation cycles.

**Library compatibility**: Triton and bitsandbytes version conflicts required careful dependency management. The working configuration uses `torch==2.5.1`, `triton==3.1.0`, `bitsandbytes==0.43.3`.

**Repulsion loss memory**: With approximately 44,000 new Wake tokens, computing pairwise similarity matrices for isotropy regularization caused out-of-memory errors. This was mitigated by sampling 1,000 random token pairs or disabling repulsion entirely.

### Training Configuration

Llama trials use modified hyperparameters to accommodate the larger model:

- **Quantization**: 4-bit NF4 with double quantization
- **Sequence length**: 1024 (reduced from 2048 due to memory)
- **Batch size**: 1 with gradient accumulation of 8-16 steps
- **Learning rate**: 8e-4 (embedding phase), 2e-5 (full fine-tune)
- **Optimizer**: Adafactor (memory efficient)
- **PEFT adapter**: Minimal LoRA (r=1 on q_proj) to satisfy quantized training requirements

### Current Status

The Llama trials remain experimental. Initial embedding-only training phases showed stable loss reduction, but full evaluation metrics (geometry analysis, perplexity, generation quality) are pending completion of multi-day training runs on available compute.

### Reproducibility Notes

To replicate Llama experiments:

1. Request access to Llama models at https://huggingface.co/meta-llama
2. Install compatible dependencies: `pip install torch==2.5.1 triton==3.1.0 bitsandbytes==0.43.3`
3. Use notebooks in `Llama/` directory with T4 or better GPU
4. Expect training times of 12-24 hours for embedding phase on T4

The Llama trials demonstrate that Wake2Vec's compositional embedding methodology is architecturally agnostic, though practical deployment on larger models requires careful memory management and extended compute budgets.

---

## Citation and Credit

- **Text**: James Joyce, *Finnegans Wake*
- **Base model**: [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Conceptual inspiration from work on embedding surgery, retrofitting, and lightweight adapter methods

**Cite**: https://github.com/mahb97/Wake2vec/blob/21469d75c26d40988ec5af8a4358d1796a36fdf0/data/CITATION.cff

