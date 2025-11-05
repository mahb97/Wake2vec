# DEVLOG — Wake2vec (field notes, unfiltered)
> Raw notes.

**Date:** 2025-11-03  
**Env:** Colab T4 · TinyLlama-1.1B · transformers 4.57.1 · accelerate 1.2.1

## Summary
Stabilised Phase-1 (embeddings-only) training on GPU with evaluation every 200 steps. Fixed tokenizer/dataset drift, added version guards and a compat shim, and set up persistent checkpoints and metrics.

## What changed
- **P1 training**
  - `max_steps=1100`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`
  - `optim="adafactor"`, `learning_rate=5e-4`, `warmup_ratio=0.0`
  - `gradient_checkpointing=True`, `use_cache=False`, `fp16/bf16=False`
  - **Eval**: every 200 steps on a small validation shard
- **Persistence & logging**
  - Checkpoints every 100 steps to Drive
  - JSON loss log; new-row embedding snapshots every 200 steps
  - Rebuilt tokenized datasets from `FW_TEXT.txt` with the current tokenizer
- **Hardening**
  - Pinned libs: `transformers==4.57.1`, `accelerate==1.2.1`, `datasets==2.20.0`
  - Added `unwrap_model` kwarg-compat shim (Accelerate)
  - Safe bootstrap: first CUDA touch is a tiny forward; no early `torch.cuda.*` calls

## Issues fixed
- **Tokenizer/vocab drift:** cached tokenized dataset contained IDs ≥ base vocab (32000).  
  **fix:** purge caches; re-tokenize from `FW_TEXT.txt` with the current tokenizer; mirror to Drive.
- **HF version mismatch:** `Accelerator.unwrap_model(keep_torch_compile=...)` kwarg not supported in environment.  
  **fix:** small shim to drop unknown kwargs (idempotent).

## Repro notes
- Corpus: `wake2vec/corpora/FW_TEXT.txt`
- Datasets saved to: `wake2vec/datasets/{train_ds,valid_ds}`
- Run layout:
  - `runs/<RUN_ID>/metrics/phase1_loss_log.json`
  - `runs/<RUN_ID>/metrics/E_postP1_step*.npy` (new-row snapshots, if `new_ids.npy` present)
  - `runs/<RUN_ID>/checkpoint-final/`
- Device placement via `device_map="auto"`; `lm_head.weight` tied to input embeddings post-load.

## Health signals (P1 live)
- Training loss descending smoothly (EMA stable)
- Eval every 200 steps executing without errors


