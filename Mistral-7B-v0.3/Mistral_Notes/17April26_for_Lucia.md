## Mistral 7B v0.3 P1 session 7 (resuming from step 500)

# Mistral be the light 

# wake2vec devlog 2026-04-17

val has been circling 11.0 for 300 steps: 10.99 → 10.99 → 10.97 → 11.02 → 11.01. 

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
| 550 | 148.31 | 11.13 | 7 |

---

## Notes

comparing the 32K-vocab cohort at step 500:
- TinyLlama P1 (1.1B): was already at single-digit loss by step 500
- Mistral 7B: val 11.01

TinyLlama moved faster despite being 7x smaller, so the 7B's stronger priors resist reshaping more than they help learning, but Mistral's 7B of frozen transformer should eventually outperform TinyLlama once she commits to a trajectory. the question is when.

---

[leavemealone](https://soundcloud.com/fredagain/fred-again-baby-keem?in=fredagain/sets/usb-669054846&si=2ed7ac63bbf544929ef59957b2dd4ae3&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing) 
