# wake2vec devlog 2026-04-08

## Llama 3.2-1B P3 session 4 (resuming from step 700, final 300 steps)

the last 300 steps, L_morph has been 0.0007 for 700 steps, L_device has been ~0.20 for 700 steps. neither has moved, the null result is confirmed across both TinyLlama and Llama architectures.

P3 trainable setup for the record:
- Total trainable: 358,291,456
- LoRA params: 5,111,808
- Embedding params (all rows, masked): 353,179,648
- Wake rows that actually train: 44,195 × 2048 = 90,511,360
- Gradient masking: base rows [0:128256] zeroed on backward pass

Resuming from `checkpoint-700`.

### P3 loss table (continued)

| Step | L_total | L_lm | L_morph | L_device | Val | Early stop |
|------|---------|------|---------|----------|-----|------------|
| 0 | 3.8597 | 3.4387 | 0.0007 | 0.1933 | — | — |
| 50 | 3.6508 | 3.2506 | 0.0007 | 0.1830 | — | — |
| 100 | 3.8224 | 3.3968 | 0.0007 | 0.1956 | **4.4819** | best ✓ |
| 150 | 4.3278 | 3.8964 | 0.0007 | 0.1986 | — | — |
| 200 | — | — | 0.0007 | — | **4.5016** | 1/5 |
| 250 | 4.1659 | 3.7234 | 0.0007 | 0.2041 | — | — |
| 300 | 3.6496 | 3.2213 | 0.0007 | 0.1970 | **4.5284** | 2/5 |
| 350 | 3.9068 | 3.4752 | 0.0007 | 0.1986 | — | — |
| 400 | 3.8786 | 3.4460 | 0.0007 | 0.1991 | **4.5482** | 3/5 |
| 450 | 3.5623 | 3.1292 | 0.0007 | 0.1994 | — | — |
| 500 | 3.7454 | 3.3314 | 0.0007 | 0.1899 | **4.5871** | 4/5 |
| 550 | 3.6257 | 3.1902 | 0.0007 | 0.2006 | — | — |
| 600 | 4.1771 | 3.7554 | 0.0007 | 0.1937 | **4.6075** | best ✓ (reset*) |
| 650 | 3.7596 | 3.3000 | 0.0007 | 0.2126 | — | — |
| 700 | 3.4596 | 3.0147 | 0.0007 | 0.2053 | **4.6179** | 1/5 (reset) |
| 750 | 4.0654 | 3.6156 | 0.0007 | 0.2077 | — | — |
| 800 | 3.6884 | 3.2429 | 0.0007 | 0.2056 | **4.6326** | 2/5 (reset) |
| 850 | 3.8758 | 3.4300 | 0.0007 | 0.2057 | — | — |
| 900 | 3.8024 | 3.3542 | 0.0007 | 0.2070 | **4.6312** | 3/5 (reset) |

\* Early stop counter reset on resume, same bug as TinyLlama P3.

---

## Llama 3.2-3B P1 — session 7 (resuming from step 700)

51 hours estimated remaining. 0.01 it/s. the 3B is the slowest of the herd but she's steady — val plateaued around 6.7-6.8 since step 300, train still dropping.

Resuming from `checkpoint-700`.

### P1 loss table (continued)

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 100 | 109.19 | 7.01 | 1 |
| 200 | 97.27 | 6.75 | 2 |
| 300 | 87.04 | 6.68 | 3 |
| 400 | 79.07 | 6.70 | 3 |
| 500 | 72.80 | 6.72 | 3 |
| 600 | 67.01 | 6.75 | 4 |
| 700 | 62.09 | 6.77 | 5 |
| 800 | 58.50 | 6.79 | 6 |
| 700+ | | | *resuming today, session 7* |
