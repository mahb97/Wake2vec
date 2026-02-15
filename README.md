# Wake2Vec

## TL;DR

Fine-tune LLMs on *Finnegans Wake* by injecting ~45K Joyce-specific tokens into the embedding layer and training in two phases: embedding-only warm-up (P1) then LoRA adaptation (P2). Currently testing across TinyLlama 1.1B, Llama 3.2-1B, Llama 3.2-3B, and Llama 3.1-8B on free Colab T4 GPUs. The whole point is to see which model produces the best Wake-like output. This is very much a work in progress.

---

## Models

| Model | Params | Phase | Status | VRAM (4-bit) |
|---|---|---|---|---|
| TinyLlama-1.1B | 1.1B | P1 complete, P2 running | P1 done (loss 8.46 -> 0.079). P2 at step 2000/3000, overfitting starting | ~4GB |
| Llama 3.2-1B | 1B | P1 running | Step 400, val_loss=7.58, on track | ~3-4GB |
| Llama 3.2-3B | 3B | P1 script ready | Not started | ~5-6GB |
| Llama 3.1-8B | 8B | P1 script ready | Not started. Biggest Llama that fits on free T4 | ~9-10GB |
| Qwen 2.5-14B | 14B | Sketch only | Aspirational. Rough VRAM math in devlog, no script | ~12-13GB |

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

# Two-Phase Protocol

### Wake Lexicon

`wake_lexicon.txt` contains 44,989 unique tokens extracted from Finnegans Wake: neologisms, multilingual mashups, accented forms, and Joyce-specific compounds. These get added to whatever base tokenizer we're using. For models with larger vocabs (Llama 3.x has 128K vs TinyLlama's 32K), some Wake tokens already exist in the base vocab and don't need to be added.

### Phase 1: Embedding-Only Training

Freeze the entire transformer. Only the embedding layer is trainable.

- New Wake tokens initialized on a sphere (random direction, radius = 1.5x base embedding std * sqrt(dim))
- Gradient masking: only the new Wake token rows receive gradients. Base vocabulary rows are zeroed via a backward hook. This prevents catastrophic forgetting
- Input and output embeddings are tied
- A frozen LoRA r=1 adapter on q_proj is included purely for PEFT compatibility with quantized models -- it contributes nothing to training

The goal is to get the new Wake tokens into a reasonable region of embedding space before asking the model to actually use them.

### Phase 2: LoRA Fine-Tune

Load P1 embeddings and freeze them. Apply LoRA adapters to attention and MLP projections. The model learns to *use* the Wake-adapted embeddings through attention redistribution and MLP adaptation.

**LoRA targets:** q_proj, k_proj, v_proj, gate_proj, up_proj, down_proj

k_proj is included alongside q/v to allow symmetric reshaping of attention patterns. MLP layers are targeted because Wake morphology requires adaptation of token-to-meaning mappings beyond attention alone.

Only implemented for TinyLlama so far.
## Training Configs

### Phase 1 (Embedding-Only)

| | TinyLlama 1.1B | Llama 3.2-1B | Llama 3.2-3B | Llama 3.1-8B |
|---|---|---|---|---|
| Base vocab | 32,000 | 128,256 | 128,256 | 128,256 |
| + Wake tokens | ~44,989 | varies (some already in vocab) | same | same |
| Quantization | fp32 (whole model) | 4-bit NF4 | 4-bit NF4 | 4-bit NF4 |
| Optimizer | Adafactor | AdamW | AdamW | AdamW |
| LR | 5e-4 | 2e-4 | 2e-4 | 2e-4 |
| Warmup | 5% | 5% | 5% | 5% |
| Batch | 1 (effective 16) | 1 (effective 16) | 1 (effective 16) | 1 (effective 16) |
| Seq len | 256 | 512 | 256 | 256 |
| Steps | 1,300 | 3,000 | 3,000 | 3,000 |
| Save every | 100 | 50 | 50 | 50 |

### Phase 2 (LoRA) -- TinyLlama only

| Parameter | Value |
|---|---|
| Quantization | 4-bit NF4, double quant, bfloat16 compute |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Trainable params | ~5.6M |
| Embeddings | Frozen (from P1) |
| LR | 2e-5 |
| Warmup | 10% |
| Batch | 8 (effective 16, grad accum 2) |
| Seq len | 256 |
| Steps | 3,000 |
| Weight decay | 0.01 |

## Data

- **Finnegans Wake corpus** (`FW_TEXT.txt`): 24,483 lines. Primary training text
- **Wake lexicon** (`wake_lexicon.txt`): 44,989 tokens. Injected into tokenizer and concatenated with FW text for training
- **Train/val split**: 90/10, seed 42
- **Block size**: Non-overlapping chunks of seq_len tokens

## Embedding Analysis

Every P1 and P2 script includes a post-training analysis suite that measures:

1. **Norm distributions** -- L2 norms of base vs new token embeddings, with Welch t-test, Mann-Whitney U, Cohen's d
2. **Isotropy** -- Mu et al. 2018 partition function ratio. Measures how uniformly embeddings spread across the space
3. **Embedding drift** -- cosine similarity between pre- and post-training embeddings. Base tokens should be ~1.0 (unchanged). Wake tokens should show meaningful movement
4. **Nearest neighbors** -- for sampled Wake tokens, find 5 closest base vocab tokens by cosine similarity
5. **Intrinsic dimensionality** -- PCA explained variance. How many principal components capture 90%/95% of variance in base vs new embeddings
6. **Pairwise cosine similarity** -- distributions for (base,base), (new,new), (base,new) pairs with KS test

All results saved as JSON + 6-panel matplotlib figure.
---

## Results So Far

### TinyLlama P1 (complete)

1,300 steps of embedding-only training. Loss went from 8.46 to 0.079 (99% reduction). Embeddings successfully integrated into the model's existing semantic space.

### TinyLlama P2 (running, overfitting)

| Step | Train | Val | Gap |
|---|---|---|---|
| 1200 | 0.5797 | 0.6255 | 0.046 |
| 1400 | 0.6388 | 0.6393 | 0.001 |
| 1600 | 0.5722 | 0.6460 | 0.074 |
| 1800 | 0.6104 | 0.6594 | 0.049 |
| 2000 | 0.4943 | 0.6679 | 0.174 |

Val loss climbing monotonically since step 1200. Best checkpoint for generation is probably around step 1400-1600. Running to 3000 to see the full curve.

### Llama 3.2-1B P1 (early)

Step 400: train_loss=92.11 (logging artifact from lexicon-heavy batch), val_loss=7.58 (normal for this stage).

---

## Three-Phase Protocol (Morpheme-Aware)

### Phase 1: Embeddings-only warm-up
Same as two-phase protocol, establishing baseline Wake token embeddings.

**Hyperparameters:**
* max_steps: 1300
* lr: 5e-4
* Warmup ratio: 0.05
* grad_accum: 16
* batch_size: 1
* Sequence length: 256 tokens

### Phase 2: LoRA behavioral tuning
Attach LoRA adapters to attention/MLP layers while keeping embeddings frozen. Train on Wake corpus to adapt model behavior without disturbing embedding space.

**Typical hyperparameters:**
* Epochs: 1-2
* lr: 2e-5
* Warmup: 0.10
* grad_accum: 16
* batch_size: 1
* LoRA rank: 8-16
* Target modules: q_proj, v_proj, mlp layers

### Phase 3: Morpheme-compositional alignment
Unfreeze embeddings with morpheme-aware regularization. Uses decomposition data (prefixes/suffixes) to enforce compositional semantics in new token embeddings.

**Loss components:**
* L_lm: Standard language modeling loss
* L_morpheme: Compositional constraint forcing Wake tokens toward component averages
  - Example: `E["allbust"] ≈ mean(E["all"], E["bust"])`
* L_repulsion: Adversarial term preventing Wake token collapse
* L_norm: Norm hygiene keeping Wake embeddings in distribution

**Typical hyperparameters:**
* max_steps: 400-800
* lr: 5e-5
* Warmup: 0.10
* Optimizer: Adafactor
* Gradient masking: New tokens only
* Loss weights: λ_morpheme=0.1, λ_repulsion=0.05, λ_norm=0.01

**Data requirements:**
* Morpheme decomposition mapping (JSON format)
* Prefix/suffix inventory with examples
* Component token validation in base vocabulary

**Expected outcomes:**
* Morphologically related tokens cluster in embedding space
* K-nearest neighbors reflect compositional structure
* Embedding norms remain stable relative to base vocabulary
* Isotropy preserved in extended vocabulary subspace
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

# Wake2Vec P2 Pilot: Validation Gap

## Summary

the validation gap measures how much Wake is bending the model.

---

## P1 Recap (No Validation)

- Embeddings-only training on the full corpus
- No held-out set; metrics reported only on the training blocks
- Apparent near-zero loss was actually memorisation of Wake slices

---

## Interpretation

- Train ↓, Val ↔ is a classic overfit signature, but here it also confirms that:
  - P1 embeddings were correctly loaded (P2 starts around 4.5, not 7+),
  - Wake is small/weird enough that the model can memorise it quickly.
- A validation loss of ~4.8 is a more honest measure of generalisation to unseen Wake text.
- The train/val gap that already existed in P1 simply wasn’t visible without a held-out set.

Rather than “fixing” this gap, later P1/P2 runs explicitly use it:

- P1 is now structured into regimes (sweet / plateau / fried) and uses the gap as a meltdown indicator.
- P2 branches (e.g. P2(sweet), P2(plateau), P2(fried)) treat different levels of overfitting as starting points, to see how much damage a full fine-tune can repair.

This pilot P2 run is kept as the moment the project stopped pretending val loss should be flat and started using it as a control knob.

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

**Memory limitations**: Llama-3.2-1B requires 4-bit quantization via bitsandbytes to fit on Colab T4 GPUs (15GB VRAM). The working configuration uses NF4 quantization with double quantization enabled, allocating 13GB to GPU and 30GB to CPU offload.

**Gated model access**: Llama models require explicit approval from Meta via Hugging Face, introducing authentication steps in training pipelines.

**Library compatibility (Nov 2025 Colab)**: Default Colab environment (torch 2.8.0, CUDA 12.9) conflicts with bitsandbytes and triton. The working configuration requires explicit downgrade: `torch==2.5.1+cu121`, `triton==3.1.0`, `bitsandbytes==0.43.3`, `transformers==4.45.2`, `accelerate==0.34.2`, `peft==0.13.2`. Runtime restart required after installation.

**Gradient checkpointing incompatibility**: 4-bit quantized models with LoRA adapters cannot use gradient checkpointing due to interaction between quantization and activation recomputation. This limits batch size options.

### Training Configuration

Llama-3.2-1B trials use the following configuration:

- **Quantization**: 4-bit NF4 with double quantization
- **Sequence length**: 256 tokens
- **Batch size**: 8 with gradient accumulation of 2 (effective batch: 16)
- **Learning rate**: 2e-5 (LoRA fine-tune phase)
- **Scheduler**: Cosine with 10% warmup
- **PEFT adapter**: LoRA r=8 on q_proj, v_proj, gate_proj, up_proj, down_proj
- **Regularization**: Weight decay 0.01, max grad norm 1.0, dropout 0.1

## Current Status (Nov 2025)

Right now, the implemented and tested parts of this repo are:

- TinyLlama P1 v2:
  - embedding-only fine-tune on *Finnegans Wake* with a 90/10 train/val split,
  - gradient-masked base vocab, ~44.5k Wake tokens trainable,
  - checkpoint + embedding snapshot infrastructure that survives Colab chaos.

- LLaMA P1 (experimental):
  - 4-bit NF4 quantised Llama-3.2-1B,
  - Wake token injection with spherical init,
  - embedding-only warm-up with full checkpoint mirroring to Drive.

The morpheme-aware initialisation and three-phase protocol are partially implemented.

---

## Citation and Credit

- **Text**: James Joyce, *Finnegans Wake*
- **Base model**: [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Conceptual inspiration from work on embedding surgery, retrofitting, and lightweight adapter methods

**Cite**: https://github.com/mahb97/Wake2vec/blob/21469d75c26d40988ec5af8a4358d1796a36fdf0/data/CITATION.cff

