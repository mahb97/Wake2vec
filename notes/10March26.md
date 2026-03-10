# wake2vec devlog 2026-03-10

## TinyLlama 1.1B P3 session 2 (resuming from step 800)

early stop counter: 2/5 and val peaked at step 400 (3.4188), has been climbing since (3.5554 at 600, 3.7154 at 800). three more misses and early stop is triggered. 

the auxiliary losses never moved last session and L_morph is pinned at 0.0002, while L_device is flat at ~0.20 (which is my own fault because at the current lambdas they contribute 0.3% of total loss).
If early stop triggers today, P3b is waiting with stronger lambdas (morph=50.0, device=2.0), lower LR (2e-5), starting from the step 400 best checkpoint.

Resuming from step 800 and itching to finish this so I can write. Some joke about being a one-woman lab held together by rollies and Fred Again. 

Also, copped this one from Chris Luno: [Massako](https://soundcloud.com/magnifik-music-740442748/massako?si=e3708fe9795d4c5188ac309199c00b04&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

### P3 loss table (continued)

| Step | L_total | L_lm | L_morph | L_device | Val | Early stop |
|------|---------|------|---------|----------|-----|------------|
| 0 | 5.9006 | 5.8904 | 0.0002 | 0.1826 | — | — |
| 200 | 3.2264 | 3.2154 | 0.0002 | 0.1978 | **3.6723** | — |
| 400 | 3.1688 | 3.1586 | 0.0002 | 0.1845 | **3.4188** | best ✓ |
| 600 | 3.2804 | 3.2703 | 0.0002 | 0.1818 | **3.5554** | 1/5 |
| 800 | 3.4210 | 3.4099 | 0.0002 | 0.2010 | **3.7154** | 2/5 |
| 850 | 1.4012 | 1.3902 | 0.0002 | 0.1986 | — | — |
| 900 | 1.5392 | 1.5277 | 0.0002 | 0.2089 | — | — |
| 950 | 1.0017 | 0.9906 | 0.0002 | 0.2018 | — | — |
| 1000 | 2.6195 | — | 0.0002 | ~0.20 | **3.8638** | reset to 0/5* |
| 1050 | 0.8967 | 0.8850 | 0.0002 | 0.2129 | — | — |
| 1100 | 1.1352 | 1.1242 | 0.0002 | 0.1982 | — | — |
| 1150 | 0.9954 | 0.9848 | 0.0002 | 0.1902 | — | — |
| 1200 | 0.8659 | 0.8546 | 0.0002 | 0.2048 | **4.0230** | 1/5 (reset) |
| 1250 | 0.9430 | 0.9325 | 0.0002 | 0.1890 | — | — |
| 1300 | 0.9028 | 0.8921 | 0.0002 | 0.1936 | — | — |
| 1350 | 0.9523 | 0.9416 | 0.0002 | 0.1933 | — | — |
| 1400 | 0.8030 | 0.7920 | 0.0002 | 0.1974 | **4.1428** | 2/5 (reset) |

### What we're watching for

1. **Val at step 1000** if it's above 3.4188, that's 3/5 for early stop. 
2. **Val at step 1200** if still climbing, 4/5. 
3. **L_device** if still flat then the current lambdas genuinely can't compete with L_lm which confirms the P3b thesis.
4. **L_morph** if this hasn't moved from 0.0002 in 1200 steps, P2 already solved it completely (sort of good to know).

### If early stop triggers → P3b

| Param | P3 (current) | P3b (next) |
|-------|-------------|------------|
| Source | P2 step 1400 | P3 step 400 (best val) |
| LR | 5e-5 | 2e-5 |
| λ_morph | 0.1 | **50.0** |
| λ_device | 0.05 | **2.0** |
| Max steps | 3000 | 400 |
| Early stop patience | 5 | 3 |

The idea here is that if the geometry signal can't compete at whisper volume then turn it up a notch. 
