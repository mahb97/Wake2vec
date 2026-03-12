# wake2vec devlog 2026-03-11

## Qwen 2.5-14B P1 session 7 (resuming from step 360)

STEP_OFFSET=360, 2640 steps remaining, warmup 132 and val has been slowly dropping every session: 21.54 → 20.98 → 20.64 → 20.50 → 19.89 → 19.36. 

Resuming from `sentry_step_0360.pt`.

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
| Run 7 | 0–100 | 360–460 | T4 cut, EMB@460, no eval landed |

Will it overfit or not, that is the question lol and no sign of plateau just yet. 

here listen to [Pinku](https://soundcloud.com/baauer/pinku?in=houseof_kyri/sets/bitches-united-for-better-code&si=56c1ad67573a42e8bd33a246c19dbd06&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing) and enjoy the fact that you got to be alive at the same time as Baauer.

### Phase 4 (considered, shelved)

Explored a two-stage Phase 4 design which involved freezing all LoRA adapters and train only Wake embedding rows with geometric losses (morph + device) and a tiny LM regulariser (the idea here was that embeddings reshape for geometry without LoRA overfitting the LM objective) while the next part of the pipeline was supposed to unfreeze everything and fine-tune with mostly LM loss, so the model learns to *use* the new geometric priors for language modelling, which is where val should drop. The split should stay `random_state=42` for cross-phase comparability, but this was shelved (for now) because L_device didn't respond to a 40x lambda increase across 1,700+ steps (in TinyLlama P3) as freezing LoRA won't fix a loss that has no learnable gradient signal. The device triplet formulation was the bottleneck, not the training regime, so upon revisiting the triplet design itself needs changing first.

