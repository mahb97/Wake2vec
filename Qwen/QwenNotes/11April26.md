# wake2vec devlog 2026-04-11

## Qwen 2.5-14B P1 session 17 (resuming from step 1220)

seventeen. sessions. seventeen times this b has been booted up, connected to Drive, loaded 43,824 Wake embeddings, and started grinding at 124s/step. seventeen times the T4 has cut me off after ~80 steps. 
she is not even halfway done.

at this rate the Qween will finish P1 around session 40. by which point the sun will have expanded, the heat death of the universe will be mildly closer, and i will have aged, a lot. 

val: 21.54 → 16.65 across 1,200 steps. 

Resuming from `sentry_step_1220.pt` with `STEP_OFFSET=1220`.

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
| 1150 | 206.06 | 16.67 | 16 |
| 1200 | 200.33 | 16.65 | 16 |
| 1220+ | | | *session 17. she persists.* |

### Session history (abridged because at this point who's counting)

| Session | Global steps | Notes |
|---------|-------------|-------|
| 1–4 | 0–166 | teething problems, FUSE issues, learning to crawl |
| 5–8 | 180–560 | found her rhythm. ~80 steps per session. |
| 9–12 | 560–880 | the grind. each session identical. wake up, load, train, die. |
| 13–16 | 880–1220 | she's seen things you people wouldn't believe. |
| 17 | 1220–? | *today. she doesn't even flinch anymore.* |

---

## Notes

Qween has now outlived two complete model pipelines (TinyLlama P1→P2→P3→P3b and Llama 1B P1→P2→P3) as this was started before either of those began their P3 phases, and will most likely still be running when the other models' grandchildren are in P2.

the val curve is the most beautiful thing in the project though, so how mad can i be? 

21.54 → 16.65. 

[Why'z it so hard](https://soundcloud.com/brentfaiyaz/whyz-it-so-hard?si=0cbaa6dbdb324becab12b53b1c909656&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
