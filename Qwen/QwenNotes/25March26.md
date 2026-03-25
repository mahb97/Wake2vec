# wake2vec devlog 2026-03-25

## Qwen 2.5-14B P1 session 11 (resuming from step 720)

eleven sessions and counting, val was 17.71 at step 600 and still descending.

Resuming from `sentry_step_0720.pt` with `STEP_OFFSET=720`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 50 | 345.00 | 21.54 | 1 |
| 100 | 321.48 | 20.98 | 1 |
| 150 | 303.07 | 20.64 | 4 |
| 200 | 289.19 | 20.50 | 5 |
| 250 | 278.96 | 19.89 | 6 |
| 300 | 314.05 | 19.36 | 6 |
| 350 | 268.07 | 19.17 | 7 |
| 400 | 256.77 | 18.80 | 8 |
| 450 | 284.28 | 18.42 | 8 |
| 500 | 249.60 | 17.93 | 9 |
| 550 | 260.29 | 17.74 | 10 |
| 600 | 232.22 | 17.71 | 10 |
| 720+ | | | *resuming today, session 11* |

### Session history

| Session | Local steps | Global steps | Notes |
|---------|------------|--------------|-------|
| Run 1 | 0–80 | 0–80 | fresh start |
| Run 2 | 0–60 | 80–140 | no offset |
| Run 3 | 0–21 | 140–161 | FUSE hang at sentry write |
| Run 4 | 0–26+ | 140–166+ | save_model override, sentry working |
| Run 5 | 0–80 | 180–260 | 115s/step, sentry@200+260 confirmed |
| Run 6 | 0–100 | 260–360 | T4 cut, sentry@360 confirmed |
| Run 7 | 0–100 | 360–460 | T4 cut, EMB@460, no eval landed |
| Run 8 | 0–100 | 460–560 | T4 cut, sentry@560 |
| Run 9 | 0–80 | 560–640 | sentry@640 |
| Run 10 | 0–80 | 640–720 | sentry@720 |
| Run 11 | 0–? | 720–? | *today* |

---

## Status

| Model | Phase | Status | Notes |
|-------|-------|--------|-------|
| TinyLlama 1.1B | P3b | Complete | Geometric null. Best: P3 step 400 (val 3.4188) |
| Llama 3.2-1B | P2 | 50 steps left | Step 2900/3000. Resume from 2900. |
| Llama 3.2-3B | P1 | Running | Step 400/3000, val 6.70 |
| **Qwen 2.5-14B** | **P1** | **Running** | **Session 11, STEP_OFFSET=720** |

---

this one is for my abuser lol: [Fuck it](https://soundcloud.com/jessiereyez/fuck-it-2?in=houseof_kyri/sets/jessie&si=93a6eb6af57f40419011a0b970f389df&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
