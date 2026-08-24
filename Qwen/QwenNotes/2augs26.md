# wake2vec devlog 2026-08-24

> *still 1850, still one eval from the decider*

## Qwen 2.5-14B P2 session 33 

`checkpoint-1850` again; the cuts took the last session before 1900 could land, which is the compute condition doing what it does. val 6.032 at 1850, a new run high on a -4.079 train crater, five consecutive evals off the floor across 250 steps, 
best-val still step 1600 (5.9209).

resuming from `checkpoint-1850`. next eval at 1900.

### P2 loss table

| Step | Train | Val | vs P1 best (15.05) | Note |
|------|-------|-----|--------------------|------|
| 1600 | 80.758 | 5.921 | -9.13 | **best val**, the floor |
| 1750 | 79.332 | 5.973 | -9.08 | fell back off the 6.02 peak, called a pause |
| 1800 | 78.347 | 5.980 | -9.07 | band-wobble, no new low |
| 1850 | 74.268 | 6.032 | -9.02 | **new run high**, train crater outside the band, fifth eval off the floor |
| 1900 | | | | *the decider* |

### for step 1900 

- confirm (val holds at or above ~6.03 with train still down near 74): a sixth eval off the floor and a sustained new-high regime. that is a turn adjudicated on val persistence, the only route Qwen's noisy train ever left open, with best-val 1600 as the turn point and P2 source.
- refute (val back into the band, or train springs to 78+): the third feint, and the strongest of the three, because it would mean Qwen can post a new run high off a 250-step elevation and still not be turning. that reading has its own value: it makes 5.92 to 6.03 a band the model wanders inside rather than a floor with a limb above it.
