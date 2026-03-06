# wake2vec devlog 2026-03-05

## Qwen 2.5-14B P1 (session 4)

Resumed from `sentry_step_0060.pt` with `STEP_OFFSET=140`. DriveSentry local-first write confirmed working. Hit a new hang at step 21 (local) this time from the **Trainer's own checkpoint save**, not DriveSentry.

### The PEFT save_embedding_layers hang

PEFT detects the resized embedding layer (196,888 tokens) and sets `save_embedding_layers=True` automatically so at every `SAVE_STEPS=20`, the Trainer writes the full embedding matrix (~2GB) to disk via `model.save_pretrained()` which blocks training the same way the old DriveSentry FUSE hang did.

**Fix:** Override `save_model` in `EmbOnlyTrainer` to skip the full PEFT checkpoint and DriveSentry already handles embedding backup (~215MB fp16), so the Trainer's 2GB save is redundant.

```python
class EmbOnlyTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
```

`on_save` still fires for DriveSentry at every `SAVE_STEPS` so only the massive model write is skipped.

Also fixed: DriveSentry in the resume block was missing `LOCAL_RUN` parameter, that would have fallen back to default path instead of local-first write.

| Global step | Train | Val | Session | Notes |
|-------------|-------|-----|---------|-------|
| 50 | 345.00 | 21.54 | 1 | |
| 100 | 321.48 | 20.98 | 2 | |
| 150 | 303.07 | 20.64 | 3 | |

## One for the story book: how Qwen 2.5-14B fits on a free Colab T4

Running a 14B model on 15GB VRAM shouldn't work, but when bitches unite for better code a lot of shit can happen, so, here's how it does.

### WakeOverlay: a separate embedding layer

The Llama/TinyLlama scripts use gradient masking (a backward hook that zeros out gradients for all 128K (or 32K) base vocab rows), leaving only the ~1–44K Wake rows trainable. This works fine when the base vocab is small relative to total vocab, but Qwen has 152K base tokens and zeroing 152K rows every backward pass is wasteful.

This is where the WakeOverlay comes in to play, and thank fuck it did because I almost quit this one: instead of masking gradients on the full embedding matrix, it creates a *separate `nn.Embedding`* that holds only the Wake tokens (43,824 x 5,120). The base model's embedding layer stays frozen in fp16 and on each forward pass, the overlay copies base embeddings and scatters the Wake rows on top via index replacement at `wake_start=152064`. A backward hook on the base embeddings zeros all gradients as a safety net, but the real protection is structural as the optimizer only receives the overlay's parameters.

The VRAM saving comes from precision splitting:
- Base embeddings: fp16 (frozen, ~1.5GB)
- Wake overlay: fp32 (trainable, ~0.9GB)
- vs a single full matrix in fp32: 196K x 5120 x 4 bytes = ~4GB

### Adafactor: stateless optimizer

AdamW stores two momentum buffers per parameter (first and second moment), roughly doubling the memory cost of trainable weights. For 44K Wake embeddings at 5,120 dim, that's another ~1.7GB I don't have on free colab.

Adafactor stores no momentum states, so the tradeoff is noisier updates, but for embedding-only training where the learning signal is already noisy (44K tokens learning from a 24K-line corpus), Adafactor's lack of momentum is tolerable. It also enables lightweight resume: a sentry checkpoint only needs the embedding tensor and a step count and no optimizer state to restore.

### The save infrastructure problem

The T4 doesn't just constrain VRAM but also time (~2-hour sessions) and I/O (Google Drive FUSE), and thus three bugs surfaced across sessions 1–4 and I probably wasted 10 hours in total to get this one to work:

1. **DriveSentry FUSE hang (session 3):** `torch.save` of a ~428MB sentry file directly to Drive FUSE blocked training indefinitely. The fix was to save local tmp, `shutil.copy2` to Drive, and unlink tmp.

2. **PEFT save_embedding_layers hang (session 4):** The Trainer's built-in `model.save_pretrained()` writes the full resized embedding matrix (~2GB) at every `SAVE_STEPS`. PEFT auto-enables `save_embedding_layers=True` when it detects the resized vocab, so here the fix was to override `save_model` in `EmbOnlyTrainer` to a no-op (mkdir only) while DriveSentry handles the real backup.

3. **STEP_OFFSET for session-safe file naming:** The Trainer's `state.global_step` restarts at 0 with each `trainer.train()` call, so the callbacks needed a configurable `step_offset` so embedding snapshots and sentries get globally unique names across sessions.

### The VRAM budget

| Component | Size |
|-----------|------|
| 4-bit NF4 model body | ~8 GB |
| fp16 base embeddings (frozen) | ~1.5 GB |
| fp32 Wake overlay (trainable) | ~0.9 GB |
| Adafactor states | ~0 GB |
| Gradients + activations | ~1–2 GB |
| **Total** | **~12–13 GB** |
| **T4 VRAM** | **15 GB** |

SEQ_LEN had to be reduced from 256 to 128 (OOM on backward pass at 256) and training speed: ~135s/step.



---

focus time lol: [Don't You](https://soundcloud.com/jerryfolkmusic/dont-you-ft-izza-gara?si=e45417052d1a486f99263535ff56af31&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
