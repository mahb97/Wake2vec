## notes for Wake P1 Fresh (Llama-3.2-1B, Vocab-Expanded)

Fine-tuning `meta-llama/Llama-3.2-1B` on *Finnegans Wake* (P1), run with dense embedding snapshots for later Wake2vec-style diagnostics.

---

### Config & Data

- **Base model:** `meta-llama/Llama-3.2-1B`
- **Vocab:** `128,256 → 172,451` (**+44,195 Wake tokens**)
- **Context length:** `512` tokens
- **Dataset:**
  - Size: **739 blocks**
  - First block shape: `torch.Size([512])`
  - Approx. training tokens: `739 × 512 ≈ 378k` from *Finnegans Wake* (P1)
- **Training schedule:**
  - Effective batch: 1 block/step (512 tokens/step as logged)

---

### Training Dynamics (Steps 320–460)

Mid-run snapshot after resume from ~step 300:

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

**Notes:**

- Overall trend: clean downward decay from loss ≈4.22 → ≈3.79  
  (perplexity ≈68 → ≈44) after the big vocab blow-up.
- Small bump at step 400 is normal optimiser noise; training recovers immediately.

---

### Resources & Warnings

- **Hardware usage (T4-type setup):**
  - GPU: ~6.9 / 15 GB VRAM
  - RAM: ~5.6 / 12.7 GB
  - Disk: ~53.2 / 73.6 GB (checkpoints + emb snaps)
  - No OOMs or resource bottlenecks; some headroom for larger batch / grad accumulation.

---

### notes to self
it's december 1st, 25. this means I do this for 9 more days, before i do this in freedom. 

(a memory crumb: just 8 days away from leaving now, with one weekend in between). 
