# wake2vec devlog 2026-03-23

## Qwen 2.5-14B P1 session 10 (resuming from step 640)

val was 17.93 at step 500 and still dropping. This is running slow but there are about 200 Flume tracks to listen to so really I'll be alright. Also, the slower this runs the more I can procrastinate the paper so that's a plus. 

Resuming from `sentry_step_0640.pt` with `STEP_OFFSET=640`.

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
| Run 10 | 0–? | 640–? | *today* |

---

## Status

| Model | Phase | Status | Notes |
|-------|-------|--------|-------|
| TinyLlama 1.1B | P3b | Complete | Best: P3 step 400 (val 3.4188). Geometric null. |
| Llama 3.2-1B | P2 | Running | Step 2400/3000, gap 1.31 |
| Llama 3.2-3B | P1 | Running | Step 200/3000, val 6.75 |
| **Qwen 2.5-14B** | **P1** | **Running** | **Session 10, STEP_OFFSET=640** |

---

## Notes

Talking about Flume, old but gold: [hermitude](https://soundcloud.com/flume/hyperparadise-flume-remix?si=a1a33b4ef3774092904613495c345ed8&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
