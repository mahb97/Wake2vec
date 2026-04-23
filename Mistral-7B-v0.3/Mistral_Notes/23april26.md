# wake2vec devlog 2026-04-23

## Mistral 7B v0.3 P1 session 9 (resuming from step 700)

val was 11.05 at step 700, only briefly broke through 11.0 (10.98 at step 650) and then retreated, totally classic Mistral noncommittal.

Resuming from `checkpoint-700`.

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

---

## Notes

Mistral has been circling 11.0 for 400 steps now (from step 300 to 700). train has dropped from 164 to 143 in that time, so the embeddings are definitely learning, val just hasn't caught up. there's a reservoir of improvement being built up that will either release all at once or never release, 2,300 steps to find out, game on. 

---

[Dermot (see yourself in my eyes)](https://soundcloud.com/fredagain/dermot-in-my-eyes?si=afb6c61f2d24494f9752bf7c2ec5983b&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
