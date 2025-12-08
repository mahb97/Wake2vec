# Devlog — 2025-12-08 — TinyLlama Wake2vec: Safe vs Full-Fry Branches

## Context

Goal: explore two related ways of making TinyLlama “Wake-aware”:

1. **Safe branch**: keep base TinyLlama embeddings intact, only learn new Wake tokens, then add a small LoRA adapter on top.
2. **Full-fry branch**: let the entire embedding matrix drift under Finnegans Wake training while the decoder stays frozen, with an optional LoRA phase afterwards.

Both branches use TinyLlama-1.1B-Chat as the base model and the FW text corpus (`FW_TEXT.txt`) plus a Wake lexicon file (`wake_lexicon.txt`).

---

## Branch A: “Safe” P1 + P2 (Wake tokens only)

### P1 (existing): `wake2vec_p1_v2-5.py`

**Status:** already implemented and run.

**Key design:**

- Base model: TinyLlama-1.1B-Chat.
- Extend the tokenizer with Wake lexicon tokens:
  - `BASE_VOCAB = len(tok)` before extension.
  - New tokens live at indices `[BASE_VOCAB : len(tok))`.
- Resize embeddings to new vocab size.
- Initialise new Wake tokens:
  - New rows start at the mean of base vocab embeddings (or mean + small noise, depending on version).
- Training regime:
  - All transformer weights frozen.
  - Input and output embeddings tied.
  - Embedding matrix trainable but **gradient-masked** so only new Wake rows update.
  - Optimizer: embedding-only (AdamW or Adafactor depending on version).
  - Hyperparams (approx): 
    - `SEQ_LEN ≈ 256/512`, `BATCH_SIZE = 1`, `GRAD_ACCUM = 16`.
    - `LR ≈ 1e-4 – 2e-4`, cosine schedule, a few thousand steps.

**Effect:**

- TinyLlama’s original embedding rows remain effectively unchanged.
- New Wake tokens learn positions that reduce FW perplexity under a frozen decoder.
- This acts as a Wake lexicon adapter rather than a full re-curving of the embedding manifold.

**Outputs:**

- Final P1 checkpoint (HF-style):  
  `.../wake2vec_p1_v2-5/final/`
  - Contains config, model weights (with learned Wake rows), and tokenizer.
