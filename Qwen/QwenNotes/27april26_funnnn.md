# wake2vec devlog 2026-04-27

## Qwen 2.5-14B P1 session 24 (resuming from step 1760)

session twenty-four, Qwen has been in training since the depths of February... 1,760 steps in, 1,240 to go. 

the journey so far, written out:

- **session 1 (Feb 22):** step 50 with val 21.5, this was the part of the journey where figuring out how to fit a 14B parameter model on a 15GB T4 was draining my last brain cell. WakeOverlay architecture, Adafactor optimizer because there's no VRAM for momentum buffers and SEQ_LEN forced to 128 to avoid OOM. but running.
- **sessions 2–4:** teething. FUSE hangs at sentry write, save_model overrides, 124s/step...and learning to crawl.
- **session 5 (Mar 9):** STEP_OFFSET=180, found the rhythm. ~80 steps per session and val: 20.50.
- **session 6 (Mar 9):** T4 cut. sentry@360 confirmed. val: 19.36.
- **session 7 (Mar 11):** EMB@460, no eval landed. one of those.
- **sessions 8-12 (Mar 13-19):** each session identical: wake up, mount Drive, load checkpoint, train ~80 steps, get cut, save sentry, repeat. val descended without fanfare from 18.80 to 17.41.
- **session 13 (Mar 27):** broke 17.2, val 17.18.
- **sessions 14-19 (Mar 31 - Apr 15):** circling 16.0, val crept from 16.95 to 16.23.
- **session 20 (Apr 17):** still circling. val 16.14, the 16.0 wall holds.
- **session 21 (Apr 19):** broke 16.1. val 16.05.
- **session 22 (Apr 23):** the breakthrough: val 15.89. train spiked to 232, assuming here geometric reorganisation in the embedding space.
- **session 23 (Apr 25):** continued descent, val 15.81.
- **session 24 (today):** resuming from step 1760.

some joke about git and Qwen being my only commits. 

Resuming from `sentry_step_1760.pt` with `STEP_OFFSET=1760`.

### P1 loss table (recent)

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 1500 | 195.01 | 16.14 | 20 |
| 1550 | 192.06 | 16.05 | 21 |
| 1600 | 182.28 | 16.11 | 21 |
| 1650 | 189.35 | 16.01 | 22 |
| 1700 | 232.25 | 15.89 | 22 |
| 1750 | 189.83 | 15.81 | 23 |
| 1760+ | | | *session 24. the journey continues.* |

---

[High Beams](https://soundcloud.com/wearechroma/flume-high-beams-ft-hwls-slowthai?si=1ef1e14603c14b9d9b13a85fa1332c4a&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
