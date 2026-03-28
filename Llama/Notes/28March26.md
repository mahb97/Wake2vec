# wake2vec devlog 2026-03-28

## Llama 3.2-1B P3 session 2 (resuming from step 200)

the big question continues: does L_morph move, there's an actual residual morpheme error to learn from...

Resuming from `checkpoint-200`.

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

---

## Llama 3.2-3B P1 session 4 (resuming from step 500)

val descent continues: 7.01 → 6.75 → 6.68 → 6.70 → 6.72, with just 2,500 steps to go lol.

Resuming from `checkpoint-500`.

### P1 loss table (continued)

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 100 | 109.19 | 7.01 | 1 |
| 200 | 97.27 | 6.75 | 2 |
| 300 | 87.04 | 6.68 | 3 |
| 400 | 79.07 | 6.70 | 3 |
| 500 | 72.80 | 6.72 | 3 |
| 600 | 67.01 | 6.75 | 4 |

---

## Notes

I was walking in Phoenix Park for like 3 hours today, so here is [3](https://soundcloud.com/flume/3-1?si=15a4bcf6c02e42bda540d226f5d27e19&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
