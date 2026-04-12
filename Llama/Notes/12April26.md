# wake2vec devlog 2026-04-12

## Llama 3.1-8B P1 session 2 (resuming from step 100)

first session confirmed: compositional init loaded, 1.0x radius set, SEQ_LEN dropped to 256 (99s/step), 13.8GB VRAM. eval steps adjusted to 50 this session.

Resuming from `checkpoint-100`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 100+ | | | *resuming today, session 2, eval_steps=50* |

---

## Llama 3.2-3B P1 session 9 (resuming from step 1000)

a third of the way done and val has been plateauing at 6.7-6.8 since step 300 — 700 steps of flat val while train drops from 87 to 51. 
Resuming from `checkpoint-1000`.

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

val trajectory since step 300: 6.68 → 6.70 → 6.72 → 6.75 → 6.77 → 6.79 → 6.81 → 6.83. 

---

## Status

| Model | Phase | Status | Notes |
|-------|-------|--------|-------|
| TinyLlama 1.1B | Done | Complete | P1→P2→P3→P3b. Best: P3 step 400 |
| Llama 3.2-1B | Done | Complete | P1→P2→P3. Confirms null. |
| **Llama 3.2-3B** | **P1** | **Running** | **Step 1000/3000, val 6.83 (rising slowly)** |
| **Llama 3.1-8B** | **P1** | **Running** | **Step 100, eval_steps now 50** |
| Mistral 7B | P1 | Paused | Step 400/3000, val 10.97 |
| Qwen 2.5-14B | P1 | Paused | Step 1250/3000, val 16.41 |
| Phi-3 Mini | P1 | Planned | Waiting for Llama 3B slot |
| Gemma 2 9B | P1 | Planned | Waiting for Mistral slot |

---
[alone](https://soundcloud.com/musicbysoandso/alone?si=b6ae23382b8e4d3e90bd6d22bdd17296&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
