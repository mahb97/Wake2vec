# wake2vec devlog 2026-04-29

## Mistral 7B v0.3 P1 session 12 (resuming from step 1000)

step 1000, and one third of the run. 

Resuming from `checkpoint-1000`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 50 | 186.78 | 11.37 | 1 |
| 100 | 181.33 | 11.13 | 1 |
| 200 | 172.47 | 11.12 | 2/3 |
| 250 | 164.99 | 11.07 | 2/3 |
| 300 | 163.94 | 10.99 | 3 |
| 350 | 159.78 | 10.99 | 4 |
| 400 | 155.43 | 10.97 | 4 |
| 450 | 152.17 | 11.02 | 5 |
| 500 | 148.54 | 11.01 | 5 |
| 550 | 148.31 | 11.13 | 7 |
| 600 | 146.44 | 11.01 | 8 |
| 650 | 145.08 | 10.98 | 8 |
| 700 | 142.82 | 11.05 | 8 |
| 750 | 142.88 | 11.05 | 9 |
| 800 | 139.48 | 11.11 | 9 |
| 850 | 140.30 | 11.06 | 9 |
| 900 | 137.50 | 11.12 | 10 |
| 950 | 138.51 | 11.08 | 10 |
| 1000 | 136.11 | 11.13 | 11 |
| 1000+ | | | *resuming today, session 12* |

---

## Notes

Mistral's train/val divergence is the most extreme in the lineup at this stage:
- train: 186.78 → 136.11 (27% reduction across 1,000 steps)
- val: 11.37 → 11.13 (2.1% reduction across 1,000 steps)

the embeddings are clearly learning. 13x more learning happening on the train side than what the held-out set detects. either:
1. memorising the FW corpus + lexicon and overfit aggressively from step 100 onward, or
2. the val blocks contain Wake forms whose semantic placement remains uncertain even as the rest of the embedding space settles

option 2 is more interesting and would mean some Wake tokens are *fundamentally harder* to place than others, regardless of training time. would be worth investigating in the analysis cell, which val tokens have the highest loss contribution at convergence?

2,000 steps to go, let's see if Mistral ever commits.

---

[The rest is noise](https://soundcloud.com/jamie-xx-official/the-rest-is-noise?si=bead10b5a68b44bab906f20bede7a850&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
