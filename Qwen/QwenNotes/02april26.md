# wake2vec devlog 2026-04-02

## Qwen 2.5-14B P1 session 15 (resuming from step 1040)

fifteen sessions: val was 16.90 at step 1050 and still dropping with 1,960 to go.

Resuming from `sentry_step_1040.pt` with `STEP_OFFSET=1040`.

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
| 950 | 216.40 | 17.12 | 13 |
| 1000 | 213.99 | 16.95 | 14 |
| 1050 | 210.99 | 16.90 | 14 |
| 1100 | 206.32 | 16.80 | 15 |

### Session history

| Session | Local steps | Global steps | Notes |
|---------|------------|--------------|-------|
| Run 1 | 0–80 | 0–80 | fresh start |
| Run 2 | 0–60 | 80–140 | no offset |
| Run 3 | 0–21 | 140–161 | FUSE hang at sentry write |
| Run 4 | 0–26+ | 140–166+ | save_model override, sentry working |
| Run 5 | 0–80 | 180–260 | 115s/step |
| Run 6 | 0–100 | 260–360 | T4 cut |
| Run 7 | 0–100 | 360–460 | T4 cut |
| Run 8 | 0–100 | 460–560 | T4 cut |
| Run 9 | 0–80 | 560–640 | |
| Run 10 | 0–80 | 640–720 | |
| Run 11 | 0–80 | 720–800 | |
| Run 12 | 0–80 | 800–880 | |
| Run 13 | 0–80 | 880–960 | |
| Run 14 | 0–80 | 960–1040 | |
| Run 15 | 0–80 | 1040–1120 | 02/04 |

---

feeling less depressed today so here ya go: [Pine & Ginger](https://soundcloud.com/noirsound/pag?si=0cccb8c6647145479882d78a5df8edb6&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
