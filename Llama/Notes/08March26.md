# wake2vec devlog 2026-03-08

## Llama 3.2-1B P2 session 4 (resuming from step 800)

The plateau whisperer. val's been sitting at ~4.04–4.08 since step 400 like it's waiting for something interesting, and honestly? fair.

Resuming from `checkpoint-800` via `resume_from_checkpoint`. Optimizer and scheduler state restored automatically. Gap was 0.60 at step 800 and widening: train still has places to go but val is giving the cold shoulder.

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
| 900 | 3.42 | 4.09 | 0.67 | 4 |
| 1000 | 3.36 | 4.12 | 0.76 | 4 | 

### The plateau question

The val plateau is real but it's not the whole story. P2 is LoRA teaching the attention layers to route Wake tokens through existing circuitry. The ceiling here is structural: without morpheme-aware composition, the model can route tokens but can't *compose* them and that's P3's job.

So P2 run continues not because the val is going to suddenly drop, but because:
1. Train loss is still falling and the LoRA layers are learning richer attention patterns
2. want the full curve for comparison
3. more P2 steps = better-trained LoRA adapters for P3 to inherit

not chasing val here, but some other things. 

Full set for youuu [Studio Live (London, April 2021)](https://soundcloud.com/go-outside28/fred-again-studio-live-london-april-2021?in=houseof_kyri/sets/i-want-you-to-see-me-fred&si=ae3e2723a6a04b58a60b221f50be5927&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
