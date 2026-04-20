# wake2vec devlog 2026-04-19

## Qwen 2.5-14B P1 session 21 (resuming from step 1540)

All the powers of old Google have entered into a holy alliance to exorcise this spectre: the T4 allocator and the disconnect daemon, the quota engine and the VRAM ceiling. 

### The means of production

The T4 GPU is the means of production in this pipeline, and i do not own it. Rented by the hour, at zero cost in currency but at great cost in dignity and storage. Qwen is allotted compute at the pleasure of the platform, when given GPU, it trains; when denied it, it's waiting, which is the basic material condition of free-tier deep learning.

Resuming from `sentry_step_1540.pt` with `STEP_OFFSET=1540`.

### The loss table, read materially

| Step | Train | Val | Session | Note |
|------|-------|-----|---------|------|
| 1300 | 200.57 | 16.32 | 18 | |
| 1350 | 190.37 | 16.34 | 18 | |
| 1400 | 197.10 | 16.23 | 19 | |
| 1450 | 186.83 | 16.25 | 19 | |
| 1500 | 195.01 | 16.14 | 20 | val broke 16.2 |
| 1550 | 192.06 | 16.05 | 21 | broke 16.1 |
| 1600 | 182.28 | 16.11 | 21 | |

The train loss is the superstructure, so the apparent, surface measure. 

### Alienation

The Qween does not remember her training and every session begins with `model.load_state_dict()`. The continuity of her development exists only in the sentry file on Drive and in the human who maintains it. 

---

## Notes

"There is no royal road to science, and only those who do not dread the fatiguing climb of its steep paths have a chance of gaining its luminous summits." Preface to the French Edition (Capital), Karl Marx, London, March 18, 1872

---

[Sow](https://soundcloud.com/baauer/sow?si=6b561b5919ed49adaa9a1e7068879e19&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

