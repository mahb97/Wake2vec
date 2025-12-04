### 12-04 Llama-3.2-1B P1, an evolved P1

---

#### model & data

Continues the P1 Llama-3.2-1B embedding-only Wake fine-tune from its earlier converged state (≈step 700), with a longer schedule and a cooler learning rate so the model can more fully internalise P1 for later stylometry tests.

- Base model: `meta-llama/Llama-3.2-1B` in 4-bit (nf4, bfloat16 compute) via `BitsAndBytesConfig`.
- Corpus: `FW_TEXT.txt` (P1) tokenised with the base Llama tokenizer.
- Blocking:
  - `SEQ_LEN = 512`, `STRIDE = 512` (non-overlapping blocks).
  - `BlockDataset` builds fixed-length chunks; labels = input IDs (standard LM objective).
- Wake lexicon:
  - Read from `wake_lexicon.txt`.
  - Tokens not in the base vocab are added:  
    `128,256 → 172,451` (+44,195 Wake tokens).
- Embedding initialisation:
  - `model.resize_token_embeddings(len(tok))` then `lm_head.weight` tied to `wte.weight`.
  - New rows initialised on a sphere with radius `1.5 ×` the base embedding radius (spherical init).

---

#### trainable params (embed-only P1)

- PEFT/LoRA adapter is configured but all model parameters are frozen.
- Only `wte.weight` (token embeddings) has `requires_grad=True`.
- Gradient hook:
  - `base_rows = 0 .. old_vocab-1`
  - `new_rows = old_vocab .. old_vocab+num_added-1`
  - `mask_grad` zeroes gradients on `base_rows`, so only the new Wake tokens are updated.
- This keeps the original Llama embedding space intact while letting Wake-specific tokens “settle in” around it.

---

#### Phase-1 checkpoints & embed reload

- Original P1 run reached at least step 740, with best observed loss ≈ 3.159 (PPL ≈ 23.5), but the last full embedding checkpoint is:
  - `full_checkpoints/step_0700/embeddings.pt`
- HF-style Sentry checkpoints under `sentry_backups/checkpoint-700/` do not contain `pytorch_model.bin` for this PEFT setup, so this no longer tries to load a full `state_dict` from there.
- Instead, evolved P1 does:
  - Initialise model from `MODEL_NAME` as usual.
  - Expand vocab + spherical init for new Wake tokens.
  - Then, if `full_checkpoints/step_0700/embeddings.pt` exists:
    ```python
    saved_emb = torch.load(PHASE1_EMB, map_location=wte.weight.device)
    wte.weight.copy_(saved_emb.to(wte.weight.device))
    ```
    -  restores the step-700 embedding matrix and continues training from that point, with a fresh optimiser/scheduler.

---

#### updated hyperparams 

- Training regime:
  - `MAX_STEPS = 6000` (up from 2000).
  - `LR = 2e-4` (down from 5e-4), cooler LR for a much longer schedule.
  - `GRAD_ACCUM = 16` (unchanged; large effective batch).
  - Scheduler: cosine, `warmup_steps = max(20, MAX_STEPS // 20)` (currently 300).
- Checkpointing & logging:
  - `SAVE_STEPS = 200`: HF `checkpoint-*` every 200 steps.
  - `EMB_SNAP_STEPS = 200`: embedding snapshots to:
    - `emb_snaps/emb_stepXXXX.pt` every 200 steps.
  - `save_total_limit = 6`.
  - `LOG_STEPS = 50`.
- Callbacks:
  - EmbeddingSnapshot: saves full `wte.weight` to Drive every `EMB_SNAP_STEPS`.
  - FullCheckpoint: saves model + tokenizer + `embeddings.pt` + `training_state.pt` under `full_checkpoints/step_XXXX/`.
  - SentryMirror: mirrors the latest HF `checkpoint-*` from `/content/runs/...` to `sentry_backups/` on Drive.
- Evaluation:
  - `evaluation_strategy = "no"` for this evolved P1 run: the aim is deliberate overfitting to P1; generalisation will be probed later via stylometry and the separate P1 eval script.
- All architectural choices (1B Llama backbone, Wake lex expansion, embedding-only updates on new tokens) remain fixed; only the optimisation horizon and LR are adjusted.

---

#### next steps

- After completion, plan is to run the expanded P1 eval pipeline on the final checkpoint:
  - global and base vs new norm stats,
  - isotropy estimate (mean pairwise cosine on a sample),
  - nearest-neighbour probes for selected Wake tokens,
  - loss curve extraction from `trainer_state.json`,
  - (optional) held-out P1 perplexity.
