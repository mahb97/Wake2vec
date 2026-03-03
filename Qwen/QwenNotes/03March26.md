# wake2vec devlog 2026-03-03

## Qwen 2.5-14B P1 session 3 (with STEP_OFFSET fix)

First session with the STEP_OFFSET fix live. Run history:

| Session | Local steps | Global steps | Notes |
|---------|------------|--------------|-------|
| Run 1 | 0–80 | 0–80 | fresh start |
| Run 2 | 0–60 | 80–140 | no offset (sentry stored local step) |
| **Run 3** | **0–?** | **140–?** | **STEP_OFFSET=140 active** |

Resumed from `sentry_step_0060.pt` (global step 140) and callbacks now write globally-unique file names. Sentry files going forward will store the true global step, so future resumes can auto-read (`STEP_OFFSET = 0`).

- SEQ_LEN: 128 (OOM at 256 on backward pass)
- Save/log every 20 steps, eval every 50
- VRAM at launch: 10.17 GB (4.83 GB headroom on T4)
- 3221 train blocks, 358 val

### Qwen P1 loss table (all sessions)

| Global step | Train | Val | Session | Notes |
|-------------|-------|-----|---------|-------|
| 50 | 345.00 | 21.54 | 1 | |
| 100 | 321.48 | 20.98 | 2 | |
| 150 | | | 3 | |

---

[More Flume for you](https://soundcloud.com/flume/smoke-and-retribution-feat-vince-staples-kucka?si=c3f9cd6e6ced49959ca61f956b179ddf&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
