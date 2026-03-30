# wake2vec devlog 2026-03-30

## Llama 3.2-1B P3 session 3 (resuming from step 500)

L_morph = 0.0007 for 500 steps and sadly never moved. L_device = ~0.20, aslo sadly never moved. val climbing: 4.48 → 4.50 → 4.53 → 4.55 → 4.59, with patience at 4/5. same story as TinyLlama but at a higher L_morph baseline, confirming it's a starting condition, rather than a learning opportunity.

Resuming from `checkpoint-500`.

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
| 450 | 3.5623 | 3.1292 | 0.0007 | 0.1994 | — | — |
| 500 | 3.7454 | 3.3314 | 0.0007 | 0.1899 | **4.5871** | 4/5 |
| 500+ | | | | | | *resuming today, session 3* |

---

## Llama 3.2-3B P1 session 5 (resuming from step 650)

val has been plateauing around 6.70–6.75 since step 300, train still dropping (109 → 67) with 2,350 steps to go.

Resuming from `checkpoint-650`.

### P1 loss table (continued)

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 100 | 109.19 | 7.01 | 1 |
| 200 | 97.27 | 6.75 | 2 |
| 300 | 87.04 | 6.68 | 3 |
| 400 | 79.07 | 6.70 | 3 |
| 500 | 72.80 | 6.72 | 3 |
| 600 | 67.01 | 6.75 | 4 |
| 650+ | | | *resuming today, session 5* |

---

## Notes

the 1B is one eval from death, if val doesn't drop below 4.4819, early stop triggers and Llama P3 joins TinyLlama P3 in the geometric null club.

---

she plays this on the Lux tour, you're welcome: [CUUUUuuuuuute](https://soundcloud.com/rosaliaofficial/cuuuuuuuuuute?in=rosaliaofficial/sets/motomami-1&si=55cfa2dba97b4437808aa904d7c21f12&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing) 

How do I know this? was there on the opening night in Lyon. 
