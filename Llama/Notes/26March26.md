# wake2vec devlog 2026-03-26

## Llama 3.2-1B P2 session 9 (resuming from step 2900)

the final session. 100 steps to go and this was just 50 steps from being done last time when gaslight GPU cut. 

Resuming from `checkpoint-2900`.

### P2 loss table (continued)

| Step | Train | Val | Gap | Session |
|------|-------|-----|-----|---------|
| 100 | 4.23 | 4.38 | 0.14 | 1 |
| 200 | 4.03 | 4.21 | 0.18 | 1 |
| 300 | 3.89 | 4.11 | 0.22 | 2 |
| 400 | 3.76 | 4.05 | 0.29 | 2 |
| 500 | 3.65 | 4.04 | 0.39 | 2 |
| 600 | 3.59 | 4.04 | 0.46 | 3 |
| 700 | 3.54 | 4.05 | 0.51 | 3 |
| 800 | 3.47 | 4.08 | 0.60 | 3 |
| 900 | 3.42 | 4.10 | 0.67 | 4 |
| 1100 | 3.35 | 4.15 | 0.80 | 4 |
| 1300 | 3.27 | 4.20 | 0.94 | 5 |
| 1400 | 3.26 | 4.23 | 0.97 | 5 |
| 1500 | 3.22 | 4.25 | 1.04 | 5 |
| 1600 | 3.18 | 4.26 | 1.09 | 6 |
| 1700 | 3.17 | 4.29 | 1.11 | 6 |
| 1800 | 3.14 | 4.30 | 1.16 | 6 |
| 1900 | 3.14 | 4.32 | 1.18 | 6 |
| 2000 | 3.12 | 4.33 | 1.22 | 6 |
| 2100 | 3.09 | 4.35 | 1.26 | 6* |
| 2200 | 3.07 | 4.36 | 1.30 | 7 |
| 2300 | 3.08 | 4.38 | 1.30 | 7 |
| 2400 | 3.07 | 4.38 | 1.31 | 7 |
| 2500 | 3.04 | 4.39 | 1.34 | 8 |
| 2600 | 3.05 | 4.39 | 1.34 | 8 |
| 2700 | 3.05 | 4.39 | 1.34 | 8 |
| 2800 | 3.04 | 4.39 | 1.36 | 8 |
| 2900 | 3.04 | 4.39 | 1.35 | 8 |
| 2900+ | | | | *resuming today, session 9 (final)* |

---

## Llama P3 preparation

P3 script ready: `wake2vec_llama_p3_strong.py`. Based on TinyLlama P3b template with strong lambdas and skipping weak lambdas entirely this time based on TinyLlama null result.

| Param | Value | Rationale |
|-------|-------|-----------|
| Source | P2 step 500 (best val 4.04) | Best LM performance before overfitting |
| LR | 2e-5 | Halved from P2 — sculpting, not learning |
| λ_morph | 50.0 | Strong — based on TinyLlama P3b findings |
| λ_device | 2.0 | Strong — 12% of total loss at these values |
| Max steps | 1000 | With early stop patience 3 |
| SEQ_LEN | 512 | Same as P2 |
| Vectorized losses | Yes | MorphemeIndex (scatter_add_), DeviceIndex (group loops) |
| Eval spam fix | Yes | model.training guard |

does a different architecture produce the same L_device null?? if yes, it's universal, but if no, it's architecture-dependent.

---

## Status

| Model | Phase | Status | Notes |
|-------|-------|--------|-------|
| TinyLlama 1.1B | P3b | Complete | Geometric null. Best: P3 step 400 (val 3.4188) |
| **Llama 3.2-1B** | **P2** | **Final session** | **100 steps from done → P3 next** |
| Llama 3.2-3B | P1 | Running | Step 400/3000, val 6.70 |
| Qwen 2.5-14B | P1 | Paused | Session 11, step ~720, val ~17.65 |

---
This one maybe, sun in Dublin, coming soon lol: [Dance in the Sunlight](https://soundcloud.com/lo-freq-1/dance-in-the-sunlight-feat?in=houseof_kyri/sets/oh-my-dubs-what-is-you-saying&si=5619c8b8a98c4edeb5634d1ada769751&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
