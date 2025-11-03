### Summary
Stabilises Phase-1 (embeddings-only) training for Wake2Vec on Colab T4 GPUs. Adds eval every 200 steps, persistent Drive checkpoints, JSON loss logs, and embedding snapshots. Pins library versions and patches Accelerate unwrap kwarg mismatch.

### Changes
- Train: 1100 steps, bs=1, GA=16, Adafactor, eval@200 on small valid shard.
- Persistence: save_steps=100 to Drive, metrics → `runs/<id>/metrics`.
- Tokenization: rebuilt from `corpora/FW_TEXT.txt` with current tokenizer; caches mirrored to Drive.
- Env: `transformers==4.57.1`, `accelerate==1.2.1`, `datasets==2.20.0`, unwrap compat shim.
- Safety: `device_map="auto"`, `use_cache=False`, tie `lm_head` to input embeddings.

### Testing
- Verified tiny forward pass on CUDA.
- Successful P1 run: loss descends smoothly; eval triggers at steps {200,400,600,800,1000}.
- No device-side asserts after retokenization.
