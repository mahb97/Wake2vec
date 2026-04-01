# wake2vec devlog 2026-04-01

## Llama 3.2-3B P1 session 6 (resuming from step 700)

the only llama that got GPU access today. 1B P3 denied me access, gaslight GPU playing favourites.

val plateauing around 6.7 since step 300 with train still dropping (109 → 62). she's learning the embeddings, the val just doesn't care anymore.

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
| 700+ | | | *resuming today, session 6* |

---

## The Model Update

for the record, the current and planned model lineup is now:

| # | Model | Params | Vocab (base) | Wake tokens added | Status |
|---|-------|--------|-------------|-------------------|--------|
| 1 | TinyLlama 1.1B | 1.1B | 32K | ~44,500 | **P1→P2→P3→P3b complete** |
| 2 | Llama 3.2-1B | 1B | 128K | 44,195 | **P1→P2→P3 complete** |
| 3 | Llama 3.2-3B | 3B | 128K | 44,195 | P1 running (step 700) |
| 4 | Llama 3.1-8B | 8B | 128K | ~44K | Script ready, not started |
| 5 | Mistral 7B v0.3 | 7B | 32K | 44,553 | P1 running (step 100) |
| 6 | Qwen 2.5-14B | 14B | 152K | 43,824 | P1 running (step 1050) |
| 7 | Phi-3 Mini | 3.8B | 32K | TBD | Script pending |
| 8 | Gemma 2 9B | 9B | 256K | TBD | Script pending |

eight models over four google accounts on free T4 GPUs. And one bitch, one book.

this is either the most comprehensive comparative embedding injection study anyone has attempted on free compute, or the most elaborate procrastination strategy in the history of digital humanities. possibly both. the line between ambition and insanity has always been thin for me and i hope to make Joyce proud, afterall, he spent 17 years writing a book no one can read, and i'm spending 17 sessions training a model to imitate it. at least the loss curves go down.

(april fools' day, and the real joke is that this might actually work.)

---

## Status

| Model | Phase | Status | Notes |
|-------|-------|--------|-------|
| TinyLlama 1.1B | Done | Complete | Geometric null. Best: P3 step 400 (val 3.4188) |
| Llama 3.2-1B | P3 | Dead | L_morph/L_device flat. Confirms TinyLlama null. |
| **Llama 3.2-3B** | **P1** | **Running** | **Step 700/3000, val 6.77** |
| Llama 3.1-8B | P1 | Queued | Script ready. Waiting for GPU slot. |
| Mistral 7B | P1 | Paused | Step 100/3000, val 11.13. No GPU today. |
| Qwen 2.5-14B | P1 | Paused | Step 1050/3000, val 16.90. No GPU today. |
| Phi-3 Mini | P1 | Planned | Script needed. |
| Gemma 2 9B | P1 | Planned | Script needed. |

---

## Notes

the devlog started with one model...then two...then three...now eight. at some point this stopped being a training log and became a census.

the serious framing: every model added strengthens the comparative analysis. vocab size vs injection size, architecture family, model scale, so each is a variable. TinyLlama's 32K vocab producing better Wake output than Llama 1B's 128K vocab is one data point. if Mistral's 32K vocab does the same, that's a pattern. if Phi-3's "textbook quality" training data resists Joyce's chaos, that's a finding about training data. 

the honest framing: i might have a problem, but you know this already. 

---

[Hold your Fire](https://soundcloud.com/wearealta/hold-your-fire?si=f7b487a51fed459bb8b744128269eaa4&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

