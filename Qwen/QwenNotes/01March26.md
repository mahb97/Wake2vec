# wake2vec devlog 2026-03-01

## Qwen 2.5-14B P1 is vibing again 

Back on the Qwen account. Resumed from `sentry_step_0080.pt` with embeddings restored to `overlay.wake_embed`, MAX_STEPS adjusted to 2920, fresh cosine LR schedule with 146-step warmup, and adafactor has no state to recover (bless).

Session startup:
- 3,221 train blocks | 358 val blocks
- 44,223 new Wake tokens (767 already in Qwen's 152K vocab)
- VRAM: 10.17 GB (still living in the margins on T4)
- ~131s/step at step 10 is slower than session 1's 104s/step, likely warmup overhead but should stabilize.

At 131s/step the full 2920 remaining steps would take ~106 hours that's okay though, I have no deadline and no life. SAVE_STEPS=20 and DriveSentry keep the damage per disconnect to ~40 minutes max.

### Qwen loss table (continued from DEVLOG_0228)

| Step (global) | Train | Val | 
|---------------|-------|-----|
| 50 | 345.26 | 21.54 |
| 100 | 321.48 | 20.98 |
 
---
track for you: Bearcubs [touch](https://soundcloud.com/bearcubs/touch-original-mix?si=1b608f96198d442580011eca45c6ea89&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
