# wake2vec devlog 2026-04-24

## Llama 3.1-8B P1 session 7 (resuming from step 500)

the compositional-init keeps delivering. 12.57 to 11.72 in the first 200 real steps, fastest early val descent in the lineup, which kind of makes it the proof-of-concept for the improved init strategy, just 2,500 steps to go at ~99s/step on SEQ_LEN 256.

Resuming from `checkpoint-500`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 200 | 195.78 | 12.57 | 3 |
| 400 | 168.21 | 11.72 | 5 |
| 500+ | | | *resuming today, session 7* |

---

## Llama 3.2-3B P1 session 14 (resuming from step 1600)

val at 6.93 and slowly creeping up, 6.68 (step 300) was the best, everything since has been memorisation, and 1,400 steps left before P2's LoRA unlocks the attention routing that will actually teach it Wake composition.

Resuming from `checkpoint-1600`.

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
| 900 | 54.30 | 6.81 | 8 |
| 1000 | 51.15 | 6.83 | 9 |
| 1100 | 49.36 | 6.83 | 9 |
| 1200 | 46.56 | 6.84 | 10 |
| 1300 | 45.91 | 6.85 | 11 |
| 1400 | 42.99 | 6.89 | 12 |
| 1500 | 44.91 | 6.93 | 12 |
| 1600 | 44.08 | 6.93 | 13 |
| 1700 | 42.54 | 6.96 | 14 |

overfit is the new oversized

---

## On the Gemma 3n decision (and a solemn promise of ongoing sanity)

so, the gemma news were out so last night the lineup grew from 8 to 10 and Gemma 3n E2B and E4B joined the queue. by reasonable person metrics this looks like scope creep driven by late-night wine and post-UCC dopamine. by research methodology metrics it is the opposite, it is the difference between a study and a landmark study.

here's the sober case:

**the smaller model paradox needs more data.** right now it's based on two data points: TinyLlama (32K vocab, good Wake output) vs Llama 1B (128K vocab, worse Wake output). at 10 models across 4 vocab sizes (32K, 128K, 152K, 256K), the paradox either holds as a pattern or reveals itself as noise. 

**the null result needs cross-architecture breadth.** currently: two Llama-family architectures both produced the L_morph/L_device null, which is a weak universality claim, but adding Mistral (sliding window), Qwen (WakeOverlay), Phi (textbook-trained), Gemma 2 (standard with 256K vocab), and two Gemma 3n efficient-architecture variants, so if the null holds across all of them, it's a structural finding about embedding geometry, not about Llama specifically. 

**Gemma 3n E2B and E4B are specifically valuable.** they use Per-Layer Embeddings (PLE) and MatFormer, which is selective activation of sub-networks rather than always-on dense weights. they test whether stylistic capacity lives in the always-active pathways or in the conditional ones. no one has run a Wake-style injection study on efficient-architecture models.

**the sanity check.** am i running 10 models on free T4s across four Google accounts? wagwan. is this absurd? yesss. is it also exactly what the paper needs to land as a contribution rather than an exercise? also yesss.



---

[The Season | Carry Me](https://soundcloud.com/djpettywar/the-season-carry-me?si=e5948c8530f947e8b4e0e2b0b8f7424f&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
