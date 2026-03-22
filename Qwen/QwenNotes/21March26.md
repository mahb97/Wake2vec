# wake2vec devlog 2026-03-20

## Qwen 2.5-14B P1 session 9 (resuming from step 560)

nine sessions and rolling, with only 560 steps done and 2,440 to go. val hasn't stopped dropping: 21.54 → 20.98 → 20.64 → 20.50 → 19.89 → 19.36 → 19.17 → 18.80 → 18.42. 

Resuming from `sentry_step_0560.pt` with `STEP_OFFSET=560`.

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
| Run 9 | 0–? | 560–? | *today* |

---

## Notes

at ~100 steps per Gaslight T4 session that's roughly 24 more sessions. I'm not crying you are. 

---

I wanna lecture so I can ask students to drop me their best study playlists. [Never be like you](https://soundcloud.com/flume/never-be-like-you-1?in=flume/sets/skin-118&si=4472f986b2d74db9ae2bdd7489170444&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
