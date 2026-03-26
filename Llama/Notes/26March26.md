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
| 2100 | 3.09 | 4.35 | 1.26 | 6 |
| 2200 | 3.07 | 4.36 | 1.30 | 7 |
| 2300 | 3.08 | 4.38 | 1.30 | 7 |
| 2400 | 3.07 | 4.38 | 1.31 | 7 |
| 2500 | 3.04 | 4.39 | 1.34 | 8 |
| 2600 | 3.05 | 4.39 | 1.34 | 8 |
| 2700 | 3.05 | 4.39 | 1.34 | 8 |
| 2800 | 3.04 | 4.39 | 1.36 | 8 |
| 2900 | 3.04 | 4.39 | 1.35 | 8 |
| 3000 | 3.03 | 4.39 | 1.36 | 9 |

### P3 loss table

| Step | L_total | L_lm | L_morph | L_device | Val | Early stop |
|------|---------|------|---------|----------|-----|------------|
| 0 | 3.8597 | 3.4387 | 0.0007 | 0.1933 | — | — |
| 50 | 3.6508 | 3.2506 | 0.0007 | 0.1830 | — | — |

---

## Llama P3 preparation

P3 script ready: `wake2vec_llama_p3_strong.py`. Based on TinyLlama P3b template with strong lambdas — skipping weak lambdas entirely based on TinyLlama null result.

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

the question: does a different architecture produce the same L_device null? if yes, it's universal. if no, it's architecture-dependent. either way, it's a finding.

---

## Llama 3.2-1B P3 session 1 (fresh start from P2 step 500)

this one brought a surprise: **L_morph = 0.0007**, not 0.0002 like TinyLlama. Llama P2 didn't fully solve morpheme composition. there's actual gradient signal here, so this is the cross-architecture finding.

### Loss contribution at step 0

| Component | Raw | Lambda | Contribution | % of total |
|-----------|-----|--------|-------------|------------|
| L_lm | 3.86 | 1.0 | 3.86 | 76.3% |
| L_device | 0.19 | 2.0 | 0.39 | 7.7% |
| **L_morph** | **0.0007** | 50.0 | **0.035** | **0.7%** |
| L_norm | 0.18 | 0.01 | 0.002 | ~0% |
| L_repulsion | 0.00 | 0.05 | 0.000 | 0% |

morph contribution is **double** TinyLlama's at the same lambdas (0.7% vs 0.3%). the smaller model paradox extends to training: TinyLlama's 32K vocab forced it to learn morpheme composition so thoroughly during P2 that P3 had nothing left. Llama's 128K vocab let it take shortcuts, leaving residual morpheme error for P3 to find.

**watch L_morph.** if it drops from 0.0007, Llama P3 is doing something TinyLlama P3 never could.

### P3 config

| Param | Value |
|-------|-------|
| Source | P2 step 500 (best val 4.04) |
| LR | 2e-5 |
| λ_morph | 50.0 |
| λ_device | 2.0 |
| Max steps | 1000 |
| Early stop patience | 5 |
| SEQ_LEN | 512 |
| Train/Val | 644 / 72 blocks |
| Trainable | 358M (Wake embed rows + LoRA) |

---

## Status
| Model | Phase | Status | Notes |
|-------|-------|--------|-------|
| TinyLlama 1.1B | P3b | Complete | Geometric null. Best: P3 step 400 (val 3.4188) |
| **Llama 3.2-1B** | **P3** | **Running** | **L_morph=0.0007! Watch this one.** |
| Llama 3.2-3B | P1 | Running | Step 500/3000, val 6.72 |
| Qwen 2.5-14B | P1 | Paused | Session 11, step ~800, val ~17.47 |

## Notes

L_morph = 0.0007 is the most interesting number in the project right now. if it drops, it proves the tokenizer gap hypothesis: models with larger vocabularies retain learnable morpheme structure because P2 didn't need to solve it as completely. if it stays flat like TinyLlama, the null result is universal. either way, it's a finding.

---

## Llama 3.2-3B P1 session 4 (continuing)

### P1 loss table (continued)

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 100 | 109.19 | 7.01 | 1 |
| 200 | 97.27 | 6.75 | 2 |
| 300 | 87.04 | 6.68 | 3 |
| 400 | 79.07 | 6.70 | 3 |
| 500 | 72.80 | 6.72 | 4 |

val flattening around 6.7 (6.68 → 6.70 → 6.72). train still dropping (87 → 73). watching for plateau.

---
This one maybe, sun in Dublin, coming soon lol: [Dance in the Sunlight](https://soundcloud.com/lo-freq-1/dance-in-the-sunlight-feat?in=houseof_kyri/sets/oh-my-dubs-what-is-you-saying&si=5619c8b8a98c4edeb5634d1ada769751&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
