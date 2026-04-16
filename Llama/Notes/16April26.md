# wake2vec devlog 2026-04-16

## Llama 3.1-8B P1 session 4 (resuming from step 250)

first real eval landed last session: val 12.57 at step 200. higher than the 3B (6.75 at step 200) but that's expected since the 8B has the 1.0x radius init, so embeddings started with lower norm and therefore less "energy" to deform the space early on, although here the trajectory matters more than the starting point. 
Resuming from `checkpoint-250`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 200 | 195.78 | 12.57 | 3 |
| 250+ | | | *resuming today, session 4* |

---

## Llama 3.2-3B P1 session 11 (resuming from step 1250)

val climbing very slowly: 6.68 → 6.84 across 900 steps. train dropping steadily (109 → 47). pure memorisation at this point but P1 is embedding-only and LoRA in P2 is where the real learning happens. 

Resuming from `checkpoint-1250`.

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

---

## Status

| Model | Phase | Status | Notes |
|-------|-------|--------|-------|
| TinyLlama 1.1B | Done | Complete | P1→P2→P3→P3b |
| Llama 3.2-1B | Done | Complete | P1→P2→P3 |
| **Llama 3.2-3B** | **P1** | **Running** | **Step 1250/3000, val 6.84 (rising)** |
| **Llama 3.1-8B** | **P1** | **Running** | **Step 250, val 12.57 (compositional init)** |
| Mistral 7B | P1 | Paused | Step 500/3000, val 11.01 |
| Qwen 2.5-14B | P1 | Paused | Step 1450/3000, val 16.25 |
| Phi-3 Mini | P1 | Planned | Waiting for 3B slot |
| Gemma 2 9B | P1 | Planned | Waiting for Mistral slot |

---

[Beginnings](https://soundcloud.com/avaion-music/beginnings?in=avaion-music/sets/to-make-people-happy&si=2ed3afebb9864d269a9192562e6fdc38&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
