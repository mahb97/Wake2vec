# wake2vec devlog 2026-03-10

## Llama 3.2-1B P2 session 5 (resuming from step 1100)

Back at it and nothing's changed. Val plateau at 4.15, train at 3.35, gap at 0.80 and growing like my disappointment. 

Resuming from `checkpoint-1100`. 1,900 steps to go.

### P2 loss table (continued)

| Step | Train | Val | Gap | Session |
|------|-------|-----|-----|---------|
| 100 | 4.23 | 4.38 | 0.14 | 1 |
| 200 | 4.03 | 4.21 | 0.18 | 1 |
| 300 | 3.89 | 4.11 | 0.22 | 2 |
| 400 | 3.76 | 4.05 | 0.29 | 2 |
| 500 | 3.65 | 4.04 | 0.39 | 2 |
| 600 | 3.59 | 4.04 | 0.46 | 3 |
| 700 | 3.54 | 4.05 | 0.51 | 3 |
| 800 | 3.47 | 4.08 | 0.60 | 3 |
| 900 | 3.42 | 4.10 | 0.67 | 4 |
| 1100 | 3.35 | 4.15 | 0.80 | 4 |
| 1200 | 3.32 | 4.16 | 0.84 | 5 |
| 1300 | 3.27 | 4.20 | 0.94 | 5 |
| 1400 | 3.26 | 4.23 | 0.97 | 5 |

Best val still step 500 (4.04) and train/val gap trend is 0.14 → 0.80 over 1100 steps. Classic overfit signature (sob, pass the tissues), but the LoRA layers are still learning attention patterns that P3 can inherit.

### Config reminder

| Param | Value |
|-------|-------|
| Model | Llama 3.2-1B (4-bit NF4) |
| LoRA rank | 8, alpha 16 |
| LR | 5e-5 |
| Batch | 8 x 2 = 16 effective |
| SEQ_LEN | 256 |
| Max steps | 3000 |

---

To complement the TinyLlama log, some Chris Luno: [All Day Long](https://soundcloud.com/chrisluno/all-day-long?si=2c0f8cca346c4054900fdcc6876f7e8d&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
