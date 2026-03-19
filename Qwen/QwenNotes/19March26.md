# wake2vec devlog 2026-03-19

## Qwen 2.5-14B P1 session 8 (resuming from step 440)

8 sessions in and this has done 440 steps out of 3,000. val is 19.17 and still dropping, hasn't plateaued yet. 

Resuming from `sentry_step_0440.pt` with `STEP_OFFSET=440`.

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

### Session history

| Session | Local steps | Global steps | Notes |
|---------|------------|--------------|-------|
| Run 1 | 0–80 | 0–80 | fresh start |
| Run 2 | 0–60 | 80–140 | no offset |
| Run 3 | 0–21 | 140–161 | FUSE hang at sentry write |
| Run 4 | 0–26+ | 140–166+ | save_model override, sentry working |
| Run 5 | 0–80 | 180–260 | 115s/step, sentry@200+260 confirmed |
| Run 6 | 0–100 | 260–360 | T4 cut, sentry@360 confirmed |
| Run 7 | 0–100 | 360–460 | T4 cut, EMB@460, no eval landed, so using last saved checkpoint which is 440 |
| Run 8 | 0–? | 440–? | *today* |

---

## Status

| Model | Phase | Status | Notes |
|-------|-------|--------|-------|
| TinyLlama 1.1B | P3b | Complete | Early stop step 800. L_device null. Best: P3 step 400 (val 3.4188) |
| Llama 3.2-1B | P2 | Paused | Step 1900/3000, val 4.32, gap 1.18 |
| **Qwen 2.5-14B** | **P1** | **Running** | **Session 8, STEP_OFFSET=460** |

---

## Notes

TinyLlama is done and the null result is the finding: P2 implicitly solved morpheme composition, device clustering was the wrong geometric question, my bad.

Llama P2 needs 1,100 more steps, should finish in 2–3 sessions. then Llama P3 is the real test: does a different architecture respond differently to geometric losses, or is the device null universal? after that there are two more Llama models of bigger size to run and some Grassman testing and then this should all be done (until I get better GPU access and can do a part 2).

---

Yes Boone - [Modern Life](https://soundcloud.com/yesboone/modern-life?si=85bb9dedd7f84445b19bab2618494a75&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
