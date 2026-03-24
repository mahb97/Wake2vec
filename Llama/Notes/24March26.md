# wake2vec devlog 2026-03-24

## Llama 3.2-1B P2 session 8 (resuming from step 2400)

GPU promising 5 hours, so probably just one more session after this and this P2 finishes.

Resuming from `checkpoint-2400`.

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

---

## Llama 3.2-3B P1 session 3 (resuming from step 250)

3B is moving, at least, and val dropped from 7.01 to 6.75 in the first 100 steps. 
Resuming from `checkpoint-250`.

### P1 loss table (continued)

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 100 | 109.19 | 7.01 | 1 |
| 200 | 97.27 | 6.75 | 2 |
| 300 | 87.04 | 6.68 | 3 |

---

## Notes

so not done with flume: [You Know](https://soundcloud.com/flume/you-know-feat-allan-kingdom?si=f406ef4df5bc4895a8ccae9437bd7021&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
