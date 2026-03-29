# wake2vec devlog 2026-03-29

## Qwen 2.5-14B P1 session 13 (resuming from step 880)

thirteen sessions, val was 17.41 at step 850 and still dropping. 

Resuming from `sentry_step_0880.pt` with `STEP_OFFSET=880`.

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
| 650 | 256.27 | 17.67 | 11 |
| 700 | 226.89 | 17.65 | 11 |
| 750 | 230.28 | 17.59 | 11 |
| 800 | 236.24 | 17.47 | 11 |
| 850 | 254.27 | 17.41 | 12 |
| 900 | 249.58 | 17.18 | 13 |

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
| Run 11 | 0–80 | 720–800 | sentry@800 |
| Run 12 | 0–80 | 800–880 | sentry@880 |
| Run 13 | 0–? | 880–? | *today* |

---

## Yesterday's update (March 28)

### Llama 3.2-1B P3 is dying

L_morph stuck at 0.0007 for 500 steps, and L_device stuck at ~0.20, all while val is climbing: 4.48 → 4.50 → 4.53 → 4.55 → 4.59. The 3.5x higher L_morph compared to TinyLlama turned out to be a starting condition, not a learning opportunity, it never moved, so sad. 

---

## Status

| Model | Phase | Status | Notes |
|-------|-------|--------|-------|
| TinyLlama 1.1B | P3b | Complete | Geometric null. Best: P3 step 400 (val 3.4188) |
| Llama 3.2-1B | P3 | Dying | Step 500/1000, patience 4/5, L_morph/L_device flat |
| Llama 3.2-3B | P1 | Running | Step 600/3000, val 6.75 |
| **Qwen 2.5-14B** | **P1** | **Running** | **Session 13, STEP_OFFSET=880** |

---

why i don't pick up my phone: [Star Shopping](https://soundcloud.com/lil_peep/star-shopping?si=ff7b1bfd1df4423fbfd4d17c196967bd&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
