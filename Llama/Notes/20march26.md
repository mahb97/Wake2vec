# wake2vec devlog 2026-03-20

## Llama 3.2-1B P2 session 7 (resuming from step 2000)

Resuming from checkpoint-2100. gap at 1.22 and climbing with 900 steps to go. been overfitting since step 500 but the LoRA layers are still deepening. 

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
| 1300 | 3.27 | 4.20 | 0.94 | 5 |
| 1400 | 3.26 | 4.23 | 0.97 | 5 |
| 1500 | 3.22 | 4.25 | 1.04 | 5 |
| 1600 | 3.18 | 4.26 | 1.09 | 6 |
| 1700 | 3.17 | 4.29 | 1.11 | 6 |
| 1800 | 3.14 | 4.30 | 1.16 | 6 |
| 1900 | 3.14 | 4.32 | 1.18 | 6 |
| 2000 | 3.12 | 4.33 | 1.22 | 6 |
| 2100 | 3.09 | 4.35 | 1.26 | 6* |

## Llama 3.2-3B P1 session 1 (fresh start)

enter the third model lol. same architecture family as the 1B but with 28 layers, hidden_size=3072, 24 attn heads, 8 KV heads. same vocab (128,256) so Wake injection is identical with only ~1,285 new tokens needed. at 4-bit it's ~5-6GB on T4, no OOM fears.

Wake injection: 44,195 new tokens added. vocab 128,256 → 172,451. 

### Config

| Param | Value |
|-------|-------|
| Model | Llama 3.2-3B (4-bit NF4) |
| Embedding strategy | Gradient masking |
| Optimizer | AdamW |
| LR | 2e-4 |
| Batch | 1 x 16 = 16 effective |
| SEQ_LEN | 512 (bumped from 256, VRAM fine) |
| Max steps | 3,000 |
| Save every | 50 |
| Eval every | 200 |
| Vocab | 128,256 → 172,451 (+44,195 Wake tokens) |
| Trainable params | 529,769,472 (Wake embed rows only) |
| Spherical init radius | 1.7253 |
| hidden_dim | 3,072 |

**Data:** SEQ_LEN bumped to 512 (VRAM fine). Token sequence warning (457K > 131K max) is expected as this is cut into blocks, not feeding the full sequence.

the 1B is 1,000 steps from finishing P2, gap at 1.22, train flattening around 3.1, val creeping past 4.3. the interesting comparison is whether the stronger language priors of the bigger model resist or embrace the Wake token injection differently. the 1B inserted Wake tokens as embedded neologisms within Victorian prose but the 3B's deeper attention might integrate them more fluidly, or it might hold its priors tighter.

---
sauvignon blanc, a pack of camels (lol this just made me think how great it would be if they were llamas and not camels, am i cringe enough for you yet) and [Rest Easy](https://soundcloud.com/niminomusic/rest-easy-1?si=d809b2758c04473a96eb59389ead5cd5&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

