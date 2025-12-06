# Wake2Vec Training Log — 2025-12-06

## Llama-3.2-1B P1 (Clean Evolved P1 Run)

### Objective

Start a clean embed-only run that continues from the good P1-700 embedding state, but with a longer schedule and cooler learning rate for deeper Wake fitting.

### Setup and Changes

**Base model:** `meta-llama/Llama-3.2-1B` (4-bit nf4, bfloat16 compute)

**Wake lexicon expansion:**

- Vocab: 128,256 → 172,451 (+44,195 Wake tokens) from `wake_lexicon.txt`
- New tokens initialised on a sphere with radius 1.5× the base embedding radius

**Trainable parameters:**

- Only `wte.weight` is trainable
- Gradient hook zeroes gradients on `base_rows = 0..old_vocab-1`, so only new Wake token embeddings are updated

**P1 training hyperparams:**

| Parameter | Value |
|-----------|-------|
| `max_steps` | 6000 |
| `learning_rate` | 2e-4 (cosine schedule, fresh optimiser) |
| `warmup_steps` | `max(20, MAX_STEPS // 20)` → 300 |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 16 |
| `evaluation_strategy` | `"no"` (deliberate overfit to P1; eval handled by separate scripts) |
| `logging_steps` | 50 |
| `save_steps` | 200 |
| `save_total_limit` | 6 |
| `gradient_checkpointing` | True |
| `bf16` | True |

### Checkpointing

- **HF checkpoints:** `LOCAL_RUN/checkpoint-*` every 200 steps
- **Full checkpoints:** `full_checkpoints/step_XXXX/` with `embeddings.pt` and `training_state.pt` every 200 steps
- **Embedding snapshots:** `emb_snaps/emb_stepXXXX.pt` every 200 steps
- **Sentry mirrors:** `sentry_backups/checkpoint-*` on Drive


## TinyLlama Wake2Vec P1 v2 

### Objective

Continue the TinyLlama P1 v2 run from step 1400 to push further along the P1 fitting trajectory, while keeping it as the explicit "weak baseline" against the stronger 1B Llama model.

### Current TinyLlama Config

**Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

**training hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `MAX_STEPS` | 2000 |
| `LR` | 5e-4 |
| `WARMUP_RATIO` | 0.05 |
| `BATCH_SIZE` | 1 |
| `GRAD_ACCUM` | 16 |
| `SEQ_LEN` | 256 |
| `SAVE_STEPS` | 100 |
| `LOG_STEPS` | 100 |
| `EVAL_STEPS` | 200 |

**Validation:**

- Dataset is split 90/10 into `train_ds` and `val_ds`
- TinyLlama's validation loss has been ≈11 (near random baseline for the large vocab), which fits its role as a capacity-limited model

**Updated strategy:**

- Prefer `LOCAL_RUN` checkpoints (current Colab run) over SENTRY mirrors to avoid stale or mismatched checkpoints
- Target `checkpoint-1400` as the desired resume point; if unavailable or broken, fall back to the highest clean checkpoint ≤ 1400 (e.g. 800)

### Status

tbc 
