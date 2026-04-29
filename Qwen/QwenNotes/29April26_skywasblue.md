# wake2vec devlog 2026-04-29

## Qwen 2.5-14B P1 session 25 (resuming from step 1780)

twenty-five sessions...and 66 days of calendar time, or roughly the gestation period of a domestic cat. val descended from 21.54 to 15.81 in that time, and 1,220 steps to go.

Resuming from `sentry_step_1780.pt` with `STEP_OFFSET=1780`.

### P1 loss table (recent)

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 1500 | 195.01 | 16.14 | 20 |
| 1550 | 192.06 | 16.05 | 21 |
| 1600 | 182.28 | 16.11 | 21 |
| 1650 | 189.35 | 16.01 | 22 |
| 1700 | 232.25 | 15.89 | 22 |
| 1750 | 189.83 | 15.81 | 23 |
| 1780+ | | | *session 25. she persists.* |

---

## Notes

since session 22 (the 16.0 breakthrough), Qween's val has dropped 0.30 in three sessions, which is faster than her early-training pace. the post-threshold descent is steeper. 

at current pace Qwen should hit val 15.0 by ~step 2200, val 14.0 by ~step 2700 and 3000 might land her around 13.5-14.0.

---

[I'll Take Care of U](https://soundcloud.com/gilscott-heronjamiexx/ill-take-care-of-u?si=b065959de6324d9b992446dcb25c61d4&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
