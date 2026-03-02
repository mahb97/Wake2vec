# wake2vec devlog 2026-03-02

## Llama 3.2-1B P1 

Got cut off at step 2990 last session, I'm not crying, you are. Resuming from sentry at step 2900, then running the full analysis suite + generation samples.

### Loss table (continued from DEVLOG_0228)

| Step | Train | Val | Notes |
|------|-------|-----|-------|
| 1400 | 78.66 | 5.36 | val breakthrough |
| 1600 | 69.83 | 5.41 | |
| 1800 | 64.90 | 5.43 | |
| 2000 | 63.46 | 5.45 | |
| 2200 | 60.32 | 5.46 | |
| 2600 | 61.92 | 5.46 | |
| 2900 | 61.22 | 5.46 | |
| **3000** | **61.23** | **5.46** | **COMPLETE.** Val plateaued since step 1400 |

[Generation Samples](https://github.com/mahb97/Wake2vec/blob/a8c2275b71605859e5c17ed5d5da62968271012e/Llama/Llama-3.2-1B/Llama3.2-1B_Outputs/Llama3.2-1B_p1_generation_samples.md)

---

Avaion bby because I'm sad, always: [Pieces](https://soundcloud.com/avaion-music/pieces?si=434029dba0134ecd9f15eb35a2ad994e&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing) 

---

## Llama 3.2-1B P2 LoRA fine-tune launched

Loading P1 embeddings from **step 1400** (best val 5.36) rather than step 3000 plateau with embeddings frozen; LoRA rank 8 on q/k/v/gate/up/down and 5.1M trainable params.

- Effective batch: 16 (batch 4 × grad_accum 4)
- SEQ_LEN: 512
- LR: 2e-5, cosine schedule
- ~38s/step on T4 (~190 steps per 2-hour session)
- Save/log every 100 steps (user-adjusted from script default of 200/50)

### P2 loss table

| Step | Train | Val | Notes |
|------|-------|-----|-------|
| 100 | 4.23 | 4.38 | already below P1 final val (5.46) |

---

