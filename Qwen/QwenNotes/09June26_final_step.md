# wake2vec devlog 2026-06-09

> *O.K. Oh Kosmos! Ah Ireland! A.I.*

## Qwen 2.5-14B P1 session 40 (canonical end day)

twenty steps, that's what stands between the Qween and canonical end at step 3000. resuming from sentry_step_2980.pt and the FullCheckpoint callback will trigger on the step 3000 save and write the canonical model snapshot to drive. fourteen weeks of training, 39 confirmed SGDR cycles plus one last one starting at step 2980 right before the GPU cut. 

last logged step 2950 at val 15.12, train 140.21. the train number is the loudest signal of the run because it captured the 39th SGDR cycle: 188.59 to 140.21 in one eval interval, the largest train descent of any single 50-step window in the entire run, so the canonical literally ends mid-cycle. the extender experiment doesn't have to argue that SGDR returns continue past step 3000 as the canonical itself is the evidence that 3000 was an arbitrary cut on an ongoing mechanism.

Resuming from `sentry_step_2980.pt` with `STEP_OFFSET=2980`.

### P1 loss table (the closing approach)

| Step | Train | Val | Session | Note |
|------|-------|-----|---------|------|
| 2700 | 224.45 | 15.05 | 36 | 36th SGDR. broke 15.1 |
| 2750 | 167.51 | 15.17 | 37 | post-spike settle |
| 2800 | 195.27 | 15.09 | 37 | 93%. holding territory |
| 2850 | 193.69 | 15.06 | 38 | 95%. holding 15.05 territory |
| 2900 | 188.59 | 15.09 | 39 | 97%. holding 15.05 territory |
| 2950 | 140.21 | 15.12 | 39 | **39th SGDR forming.** 48-point train drop |
| 2980+ | | | | *session 40. twenty steps to canonical.* |

---

## Notes

once this completes the extender script launches from `sentry_step_3000.pt` with `STEP_OFFSET=3000`, continuing the SGDR cycles indefinitely as the methodological appendix on whether the accidental mechanism keeps finding descent. the 39 cycle that started at step 2950 gets to resolve in the extender, not the canonical. 

the 39th cycle she started 20 steps before the cut gets to live in the extender, which is the right shape for a model that has spent fourteen weeks refusing to plateau.

---

[Berghain](https://soundcloud.com/rosaliaofficial/berghain?in=rosaliaofficial/sets/lux-complete-works&si=c173fd399f714dc0abba27b538804b1d&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
