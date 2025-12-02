## 12-02 Llama-3.2-1B Wake P1 (Resume) run update 

- Resumed the P1-fresh Llama-3.2-1B run with extended vocab:
  - Vocab: `128,256 → 172,451` (+44,195 Wake tokens)
  - Context length: 512
  - Dataset: 720 chunks × 512 tokens
  - `max_steps = 2000`
- Checkpointing:
  - Embedding snapshots every 50 steps 
  - Full checkpoints every 100 steps 
  - Sentry backups every 100 steps

**Current progress at resume**

Resumes from 400:

| Step | Loss    | Approx. Perplexity (`exp(loss)`) |
|------|---------|-----------------------------------|
| 320  | 4.2207  | ~68.1                            |
| 340  | 4.1461  | ~63.2                            |
| 360  | 4.0512  | ~57.5                            |
| 380  | 3.9337  | ~51.1                            |
| 400  | 4.0217  | ~55.8                            |
| 420  | 3.8934  | ~49.1                            |


PyTorch now emits a security warning when torch.load is used with weights_only=False. HF’s Trainer still uses the old default to restore full training state, which is fine for Wake2vec own checkpoints, but the docs recommend weights_only=True when loading any untrusted models.

