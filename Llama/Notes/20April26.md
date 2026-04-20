# wake2vec devlog 2026-04-20

## Llama 3.1-8B P1 session 6 (resuming from step 400)

biggest development of the project: the 8B dropped from val 12.57 (step 200) to val 11.72 (step 400), that's a real 0.85 val descent in 200 steps, and the compositional init is finally paying off. this is moving faster than any other Llama did at equivalent early steps. 

Resuming from `checkpoint-400`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 200 | 195.78 | 12.57 | 3 |
| 400 | 168.21 | 11.72 | 5 |
| 400+ | | | *resuming today, session 6* |

---

## Llama 3.2-3B P1 session 13 (resuming from step 1500)

val still climbing slowly: 6.68 → 6.93 across 1,200 steps, half the run done.
Resuming from `checkpoint-1500`.

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

---

## Notes

the 8B descent is worth flagging for the paper. compositional init + 1.0x radius produced a steeper val drop in the first 400 steps than any of the spherical-1.5x models. if the trend holds through session 10-15, it's empirical evidence that the init strategy matters.

---

[drop dead](https://soundcloud.com/oliviarodrigo/drop-dead?si=4ecf8e3f7f1b4bdf9c5bcdcb6bbb5f8b&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
