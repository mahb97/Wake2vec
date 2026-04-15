# wake2vec devlog 2026-04-15

## Mistral 7B v0.3 P1 session 6 (resuming from step 500)

500 steps done, 2,500 to go, and val has been sitting around 11.0 for 200 steps: 10.99, 10.99, 10.97, 11.02, 11.01. 

Resuming from `checkpoint-500`.

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
| 500+ | | | *resuming today, session 6* |

---

## Notes

Mistral's val plateau at ~11.0 is interesting. for comparison:
- Llama 3B plateaued at ~6.7 (step 300, never left)
- Llama 8B is at 12.57 (step 200, still early)
- Mistral is at ~11.0 (step 300, hovering)

the 32K vocab models (TinyLlama, Mistral) have higher val than the 128K vocab models (Llama 1B, 3B) because they're learning 44K new tokens from scratch. but Mistral's 7B of frozen transformer should eventually push val lower than TinyLlama's 1.1B did. 

---
[way you smile](https://soundcloud.com/avaion-music/way-you-smile?in=avaion-music/sets/to-make-people-happy&si=77c9b8188a3347f69aee8d1d20e1b8e5&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
