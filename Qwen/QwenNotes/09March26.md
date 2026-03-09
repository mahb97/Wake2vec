# wake2vec devlog 2026-03-09

## Qwen 2.5-14B P1 session 6

[reserved...](https://soundcloud.com/berrjann/reserved?si=b9ed2a237d134651b6bf6e39a1a6833c&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing) (i remember my devlogs by loss table and the music i was banging out to)

Resuming from `sentry_step_0260.pt` with `STEP_OFFSET=260`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 50 | 345.00 | 21.54 | 1 |
| 100 | 321.48 | 20.98 | 1 |
| 150 | 303.07 | 20.64 | 4 |
| 200 | 289.19 | 20.50 | 5 |
| 200+ | | | *resuming today from step 260* |

val still dropping from 21.54 → 20.50 in 200 steps and train down 16% (345 → 289). 

### Session history

| Session | Local steps | Global steps | Notes |
|---------|------------|--------------|-------|
| Run 1 | 0–80 | 0–80 | fresh start |
| Run 2 | 0–60 | 80–140 | no offset |
| Run 3 | 0–21 | 140–161 | FUSE hang at sentry write |
| Run 4 | 0–26+ | 140–166+ | save_model override, sentry working |
| Run 5 | 0–80 | 180–260 | 115s/step, sentry@200+260 confirmed |
