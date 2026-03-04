# wake2vec devlog 2026-03-04

## Llama 3.2-1B P2 session 2 (resuming from step 200)

Back on Llama P2 today`:
- added `resume_from_checkpoint` logic to the script
- Resume path: `SENTRY / "checkpoint-200"` → copy to local → `trainer.train(resume_from_checkpoint=...)`.
- Also fixed `warmup_ratio`, now `warmup_steps` (transformers 5.x deprecation).

### P2 loss table (continued)

| Step | Train | Val | Notes |
|------|-------|-----|-------|
| 100 | 4.23 | 4.38 | session 1 |
| 200 | 4.03 | 4.21 | session 1 (cut off at 208) |
| 300 | 3.89 | 4.10 | 04 March |
| 400 | 3.76 | 4.05 | "" |

---

Just wake2vec and music, what else is there to life lol: [Fuel](https://soundcloud.com/oxy-scmusic/fuel?in=oxy-scmusic/sets/oxy-8&si=caa779eac3e7492caf8ebb2f1c7db3ed&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
