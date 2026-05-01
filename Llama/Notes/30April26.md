# wake2vec devlog 2026-04-30

## Llama 3.1-8B P1 session 10 (resuming from step 850)

steady descent through 800 steps, which makes the compositional init + 1.0x radius the most consistently descending P1 trajectory in the lineup so far. 

Resuming from `checkpoint-850`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 200 | 195.78 | 12.57 | 3 |
| 400 | 168.21 | 11.72 | 5 |
| 600 | 151.81 | 11.48 | 7 |
| 800 | 140.89 | 11.37 | 9 |
| 850+ | | | *resuming today, session 10* |

---

## Llama 3.2-3B P1 session 17 (resuming from step 2150)

850 steps left and the val is now perfectly U-shaped: started at 7.01 (step 100) then descended to 6.68 (step 300) and then climbed back to 7.01 (step 2100).

Resuming from `checkpoint-2150`.

### P1 loss table (continued)

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 100 | 109.19 | 7.01 | 1 |
| 300 | 87.04 | 6.68 | 3 | best ✓ |
| 1000 | 51.15 | 6.83 | 9 |
| 1500 | 44.91 | 6.93 | 12 |
| 2000 | 38.58 | 7.00 | 16 |
| 2100 | 36.90 | 7.01 | 16 |
| 2200 | 36.83 | 7.03 | 17 |

---

## Status

| Model | Phase | Status | Notes |
|-------|-------|--------|-------|
| TinyLlama 1.1B | Done | Complete | Best: P3 step 400 |
| Llama 3.2-1B | Done | Complete | doneeee |
| **Llama 3.2-3B** | **P1** | **Running** | **Step 2150/3000, val 7.01 (back to start)** |
| **Llama 3.1-8B** | **P1** | **Running** | **Step 850/3000, val 11.37 (steady descent)** |
| Mistral 7B | P1 | Paused | Step 1100/3000, val 11.15 (fking around 11.0) |
| Qwen 2.5-14B | P1 | Paused | Step 1850/3000, val 15.72 (post-16.0 momentum) |
| Phi-3 Mini | P1 | Pre-flight | Script pending. Waits ~13 sessions for 3B slot. |
| Gemma 2 9B | P1 | Script ready | `Gemma/wake2vec_gemma2_9b_p1.py`. Waits for Mistral slot. |
| Gemma 3n E2B | P1 | Pre-flight | Script pending. |
| Gemma 3n E4B | P1 | Pre-flight | Script pending. |

---

New Rosalia for youuuuu: [Focu 'Ranni](https://soundcloud.com/rosaliaofficial/focu-ranni?si=f03fc71049644bc1b5f44625ff69607e&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
