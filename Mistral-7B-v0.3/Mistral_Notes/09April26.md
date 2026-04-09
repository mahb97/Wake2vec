# wake2vec devlog 2026-04-09

## Mistral 7B v0.3 P1 session 3 (resuming from step 150)

checkpoint-250 went missing, resuming from 150 instead. lost 100 steps but the embeddings are fine from the sentry at 150. val was 11.07 at step 250 last session.

Resuming from `checkpoint-150`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 50 | 186.78 | 11.37 | 1 |
| 100 | 181.33 | 11.13 | 1 |
| 200 | 172.47 | 11.12 | 2 |
| 250 | 164.99 | 11.07 | 2 |
| 300 | 163.94 | 10.99 | 3 |

---

## Notes

the missing checkpoint is annoying but not a disaster 

[Flume - Stay](https://soundcloud.com/flume/stay?in=houseof_kyri/sets/sal-paradise&si=e7ec5ad235364fa5a14f16dfdf0e7c49&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
