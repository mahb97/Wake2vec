# wake2vec devlog 2026-04-13

## Mistral 7B v0.3 P1 session 5 (resuming from step 400)

val broke through 11.0 a few sessions ago and has been slowly descending since: 10.99 → 10.99 → 10.97. train dropping more convincingly (186 → 155). 

Resuming from `checkpoint-400`.

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

---

## Notes

Mistral is the slowest descender in the lineup. for comparison at ~400 steps:
- TinyLlama P1: train 8.46 → 0.079 (done by step 1300)
- Llama 3B P1: val 7.01 → 6.70
- Mistral 7B P1: val 11.37 → 10.97

the higher val is partly the 32K vocab effect, as 44,553 new tokens is a lot of embedding space to build. but TinyLlama had the same vocab and learned faster, the difference is model size: Mistral's 7B frozen transformer has stronger priors that resist reshaping. 

---
[Monday](https://soundcloud.com/mattcorby/monday?in=mattcorby/sets/telluric-1&si=60b304ac00684a3a90f4d269d4905000&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
