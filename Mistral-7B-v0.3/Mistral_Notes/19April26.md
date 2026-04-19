# wake2vec devlog 2026-04-19

## Mistral 7B v0.3 P1 session 8 (resuming from step 550)

val went 10.99 → 11.02 → 11.01 → 11.13. smaybe this needs the occasional eval spike to feel alive.

Resuming from `checkpoint-550`.

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
| 550+ | | | *resuming today, session 8* |

---

## Notes

Mistral's val is going sideways, 300 steps of hovering around 11.0 with slight upward drift, and the 7B frozen transformer's priors are holding firm against 44,553 new Wake tokens trying to reshape the space. 

compare to TinyLlama which reached near-zero loss by step 1300 with the same vocab profile: 32K base + 44K new tokens. the difference is TinyLlama had nothing to unlearn, while Mistral has 7 billion parameters of internet-trained attention patterns that pull strongly toward their priors.

---

[Sister](https://soundcloud.com/tshamusic/sister?si=9ce115e3c36940659e7466cc20616100&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

