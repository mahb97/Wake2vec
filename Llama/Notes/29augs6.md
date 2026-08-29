# wake2vec devlog 2026-08-29

> *rainy days and Llama 8B*

## Llama 3.1-8B P2 session 31 

1300 is already in the record (val 5.242857, train 4.131872), landed and banked, so this session re-covers 1290-to-1300 before the loss curve and analysis.

llama turned at 1100, best-val step 900 (5.1486), and the limb confirmed bounded at 1300: four rises of +0.024, +0.013, +0.031, +0.026, mean +0.0236 a hundred steps, all inside the +0.013 to +0.031 band.

resuming from `checkpoint-1290`. SEQ_LEN 512, EVAL_STEPS=100. next new eval at 1400.

### P2 loss table

| Step | Train | Val | vs P1 best (11.36) | Note |
|------|-------|-----|--------------------|------|
| 900 | 4.504 | 5.149 | -6.21 | **best-val, 0.181 below the wall**, the turn point |
| 1000 | 4.421 | 5.173 | -6.19 | turn SIGNAL: val +0.024 |
| 1100 | 4.332 | 5.186 | -6.17 | turn CONFIRMED: val +0.013 |
| 1200 | 4.238 | 5.217 | -6.14 | limb reaccelerated: val +0.031, gap 0.979 |
| 1300 | 4.132 | 5.243 | -6.12 | inside the band: val +0.026, gap 1.111, 0.090 under the 3B wall |

the 28th turned up a 16x inflation in logged training loss wherever a custom `compute_loss` swallowed `num_items_in_batch`. every P2 script in the project has now been checked. `wake2vec_llama8b_p2_lora.py` uses the stock Trainer, 
so the 8B's train column is correct exactly as recorded, and the monotonic descent that made every reading on this model resolvable (4.504, 4.421, 4.332, 4.238, 4.132) is real rather than an artifact. 

Qwen's swings turn out to be 16x smaller than logged (±0.15, not ±2). the 8B half of that contrast is unaffected, but the finding now has to rest on sign reversals rather than amplitude: the 8B's train never once changed direction across 
the whole run, and Qwen's does between evaluations. 

the 8B's P1 was the over-deformation case, dense invention without recoverable meaning plus code-register tokens out of the base pretraining distribution. Mistral has now shown what routing does from the other pole: its P1 dissolution 
became P2 suspension, and the contemporary-register leak closed under decode-matched settings, verified in the P1 script rather than assumed. the 8B is the second test of that mechanism, from a different starting failure and the set's 
only compositional 1.0x init. 
