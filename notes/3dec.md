### 12-03 TinyLlama Resume (Checkpoint-800)

- Found that TinyLlama checkpoints beyond `checkpoint-800` (900, 1000, 1100, 1200) exist on disk but do not resume cleanly.
- Latest reliable resume point is therefore `checkpoint-800`, and training is now continuing from there (currently around step 837).
- TinyLlama notebook did not originally have pretty step-by-step loss printing wired in, so only HF’s internal `trainer_state.json` logs exist for earlier steps.
- Added a small `LossLogger` callback to TinyLlama’s `Trainer` to print:
  - `[TINY] step XXXX  loss YY.YYYY`

**Eval alignment:** once this TinyLlama run completes, its final checkpoint should be run through the same P1 eval pipeline as the Llama-3.2-1B model (embedding norms, base vs new token stats, isotropy estimate, NN probes, optional held-out perplexity) so that comparisons between TinyLlama and Llama-3.2-1B are directly comparable.
