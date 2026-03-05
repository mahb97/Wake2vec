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


---

focus time lol: [Don't You](https://soundcloud.com/jerryfolkmusic/dont-you-ft-izza-gara?si=e45417052d1a486f99263535ff56af31&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
