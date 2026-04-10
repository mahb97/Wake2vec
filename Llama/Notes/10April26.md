# wake2vec devlog 2026-04-10

## Llama 3.1-8B P1 session 1 (fresh start)

the biggest Llama enters the arena: 32 layers, 4096 hidden dim, and a 128K base vocab with a total of 172,451 after extra +44,195 Wake tokens. the first model with compositional init and 1.0x spherical radius, as a result from the lessons learned in TinyLlama and Llama 1B.

### Vocab & data

| | Value |
|---|---|
| Base vocab | 128,256 |
| Wake tokens added | 44,195 |
| Already in vocab | 795 |
| Final vocab | 172,451 |
| FW text | 1,358,352 chars |
| Lexicon | 412,636 chars |
| Combined | 1,770,989 chars |
| Total tokens | 457,060 |
| Train blocks | 802 (SEQ_LEN 512) |
| Val blocks | 90 |

### Config

| Param | Value | Notes |
|-------|-------|-------|
| Model | meta-llama/Llama-3.1-8B | 4-bit NF4 |
| Embedding init | Compositional + spherical 1.0x | **new** — first model with both |
| Optimizer | AdamW | |
| LR | 2e-4 | |
| Batch | 1 x 16 = 16 effective | |
| SEQ_LEN | 512 | may need reducing if OOM/slow |
| Max steps | 3,000 | |
| Eval every | 200 | |
| Save every | 50 | |

### What's different from previous Llamas

1. **Compositional init:** Wake words with morpheme decomposition start near their base word embeddings. "unfitting" starts near "fitting", not at a random point on a sphere.
2. **1.0x radius:** spherical fallback uses base norm, not 1.5x. eliminates the norm gap (Cohen's d = -7.81) seen in every previous model.
3. **Both combined:** compositional tokens get semantic starting positions, spherical tokens get norm-matched random positions. no more 50% norm separation.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|

### risk tracker and KPIs

1. **Step time** Mistral 7B needed SEQ_LEN 256 (163s/step at 512). the 8B is bigger, may need the same reduction.
2. **Norm distribution at first snapshot** do compositional tokens have different norms than spherical ones? are they in the base distribution?
3. **Convergence speed** does compositional init beat the other Llamas' early loss trajectory?
4. **Val plateau** Llama 1B plateaued at 5.36 (step 1400). Llama 3B at ~6.7 (step 300). where does the 8B settle?

---

## Llama 3.2-3B P1 session 8 (resuming from step 700)

val plateau continues at ~6.7-6.8. train dropping (109 → 58). 2,300 steps to go.

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
| 700+ | | | *resuming today, session 8* |

---

## Status

| Model | Phase | Status | Notes |
|-------|-------|--------|-------|
| TinyLlama 1.1B | Done | Complete | P1→P2→P3→P3b. Geometric null. |
| Llama 3.2-1B | Done | Complete | P1→P2→P3. Confirms null. |
| Llama 3.2-3B | P1 | Running | Step 700/3000, val 6.79 |
| **Llama 3.1-8B** | **P1** | **Starting** | **First model with compositional init + 1.0x radius** |
| Mistral 7B | P1 | Paused | Step 300/3000, val 10.99 |
| Qwen 2.5-14B | P1 | Paused | Step 1200/3000, val 16.65 |
| Phi-3 Mini | P1 | Planned | Script pending |
| Gemma 2 9B | P1 | Planned | Script pending |

---

today i give you [blue](https://soundcloud.com/thatdudebrb/blue?si=44d79722fc454d90b7cf92212881d358&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
