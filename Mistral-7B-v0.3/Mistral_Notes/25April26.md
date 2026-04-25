# wake2vec devlog 2026-04-25

## Mistral 7B v0.3 P1 session 10 (resuming from step 850)

double digits in sessions... val at 11.06 last session, back down from the 11.11 spike at step 800. something is being learned somewhere, though val refuses to acknowledge it.

Resuming from `checkpoint-850`.

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
| 850+ | | | *resuming today, session 10* |

---

## Notes

if Qwen broke 16.0 with a 0.12 single-step val drop, Mistral could absolutely break 11.0 properly today. the embeddings are clearly accumulating semantic structure (train at 140 from a starting point of 186), all that's missing is the moment when the new geometry catches up to the held-out set.

---

[super rich kids](https://soundcloud.com/frankocean/super-rich-kids-album-version?si=0fb9f1027a144b2d92dd78079edfc1ef&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
