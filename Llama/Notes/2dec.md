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
| 440  | 3.8845  | ~48.6                            |
| 460  | 3.7942  | ~44.4                            |
| 480  | 3.6833  | ~39.8                            |
| 500  | 3.6884  | ~40.0                            |
| 520  | 3.6641  | ~39.0                            |
| 540  | 3.5804  | ~35.9                            |
| 560  | 3.5366  | ~34.3                            |
| 580  | 3.5912  | ~36.3                            |
| 600  | 3.4845  | ~32.6                            |
| 620  | 3.4825  | ~32.5                            |
| 640  | 3.4203  | ~30.6                            |
| 660  | 3.3637  | ~28.9                            |

PyTorch now emits a security warning when torch.load is used with weights_only=False. HF’s Trainer still uses the old default to restore full training state, which is fine for Wake2vec own checkpoints, but the docs recommend weights_only=True when loading any untrusted models.

**GPU quota weirdness:**  
  Colab initially showed ~1h20 remaining on T4, then suddenly jumped to 3 hours after 20 steps.  
  - Treating this as unexpected grace time and using it to push the Llama-3.2-1B P1-fresh run further toward step 2000.

**12-03 Llama-3.2-1B Wake P1 (T4 Session End)**

- **Run:** `wake_llama_P1_fresh` (Llama-3.2-1B, vocab 128,256 → 172,451)
- **Session GPU:** T4 (Colab)
- **status:**
  - Continued from earlier checkpoint and logged training losses up to step 740.
  - Best observed step this session:  
    - Step 740 → loss 3.1590, PPL ≈ 23.5 on extended Wake vocab.
- **checkpoint stats:**
  - Steps 720 and 740 were seen in logs but not fully checkpointed before the T4 session ended.
  - plan is to continue towards step 2000, with existing embedding snapshots and full checkpoints enabled.
- **note:** Even with the lost tail, this session confirms that the P1-fresh Llama run can reach sub-25 PPL on the extended Wake vocab, which will feed into the expanded P1 eval (base vs new norms, isotropy, NN probes).

