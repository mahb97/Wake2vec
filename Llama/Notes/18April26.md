# wake2vec devlog 2026-04-18

## Llama 3.1-8B P1 session 5 (resuming from step 250, again)

T4 choked me out at step 280 last session, which meant 30 steps of progress, no eval, no new sentry beyond 250.

Resuming from `checkpoint-250`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 200 | 195.78 | 12.57 | 3 |
| 250+ | | | *resuming today, session 5 (T4 cut at 280 last time)* |

---

## Llama 3.2-3B P1 session 12 (resuming from step 1300)

val still slowly rising: 6.83 → 6.85 across 300 steps and train keeps dropping (51 → 45). 1,700 steps to go. llama and I are both overfitting at this point, some joke about maxxing lol. 

Resuming from `checkpoint-1300`.

### P1 loss table (continued)

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 100 | 109.19 | 7.01 | 1 |
| 200 | 97.27 | 6.75 | 2 |
| 300 | 87.04 | 6.68 | 3 |
| 400 | 79.07 | 6.70 | 3 |
| 500 | 72.80 | 6.72 | 3 |
| 600 | 67.01 | 6.75 | 4 |
| 700 | 62.09 | 6.77 | 5 |
| 800 | 58.50 | 6.79 | 6 |
| 900 | 54.30 | 6.81 | 8 |
| 1000 | 51.15 | 6.83 | 9 |
| 1100 | 49.36 | 6.83 | 9 |
| 1200 | 46.56 | 6.84 | 10 |
| 1300 | 45.91 | 6.85 | 11 |
| 1400 | 42.99 | 6.89 | 12 |

---

## Notes

the 8B is at the mercy of T4 allocation.

the 3B is reliable and P2 is waiting on the other side.

---

you gotta have the mandem hype you up and that's what i'm doing to you man: [Galdem(intro)](https://soundcloud.com/tshamusic/galdem-intro?in=tshamusic/sets/capricorn-sun-1&si=488b460585d146f397a5b84f65640574&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
