# wake2vec devlog Llama 3.1-8B P1 (draft, not yet started)

## Llama 3.1-8B P1 pre-flight

the biggest Llama that fits on free T4, 32 layers, hidden_size=4096, 128K base vocab → ~172K total after Wake injection. ~10-12GB at 4-bit is tight but should fit.

### What's different from previous models

this script incorporates everything that was learned from TinyLlama and Llama 1B (90 degree learning curve):

**1. Two-tier embedding initialisation (new)**

previous models used spherical init at 1.5x base radius. this created a persistent norm gap: Wake tokens sat at 1.50, base at 0.99, Cohen's d = -7.81. the gap never closed across P1, P2, or P3.

the 8B uses a two-tier strategy:
- **Compositional init:** Wake words with known morpheme decomposition (from `wake_embedding_groups.jsonl`) are initialised from their base word embeddings. "unfitting" starts near "fitting", "riverrun" starts near "river"+"run". the model doesn't have to discover these semantic relationships from scratch.
- **Spherical init at 1.0x radius:** remaining Wake words get random directions at the same norm as the base vocabulary. no more 50% norm gap.

expectation: faster convergence in P1 (embeddings start closer to their semantic targets) and no norm gap for P3's L_norm to struggle with.

**2. Early stopping in P2 (planned)**

Llama 1B best val was step 500 out of 3000. she ran 2,500 steps of pure overfitting. for the 8B, P2 will use early stopping with patience 5. saves GPU for models that need it (considering that i also need to run Phi and Gemma).

**3. P3: minimal run (planned)**

the geometric null is confirmed across two architectures. the 8B P3 will be 200-300 steps max, just for the comparative data point. strong lambdas from the start (morph=50.0, device=2.0).

### Config

| Param | Value | Notes |
|-------|-------|-------|
| Model | meta-llama/Llama-3.1-8B | 4-bit NF4 |
| Embedding strategy | Gradient masking | same as other Llamas |
| Init strategy | Compositional + spherical 1.0x | **new** |
| Optimizer | AdamW | |
| LR | 2e-4 | |
| Batch | 1 x 16 = 16 effective | |
| SEQ_LEN | 512 (may reduce) | tight on T4 VRAM |
| Max steps | 3,000 | |
| Save every | 50 | |
| Eval every | 200 | |
| Morpheme data | wake_embedding_groups.jsonl | for compositional init |

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|

### What we're watching for

1. **Norm distributions at step 0** do compositionally-initialised tokens have different norms than spherically-initialised ones? do they sit in the same shell as base tokens?
2. **Convergence speed** does compositional init lead to faster val loss descent than the other Llama models?
3. **Val plateau** Llama 1B P1 val plateaued around 5.36 at step 1400. the 8B should plateau earlier or lower given stronger priors.
4. **VRAM** will SEQ_LEN 512 fit? may need to drop to 256 like Mistral had to.

### Comparison points

| Model | Params | Base vocab | Wake tokens | Init | Radius |
|-------|--------|-----------|-------------|------|--------|
| TinyLlama 1.1B | 1.1B | 32K | ~44,500 | Spherical | 1.5x |
| Llama 3.2-1B | 1B | 128K | 44,195 | Spherical | 1.5x |
| Llama 3.2-3B | 3B | 128K | 44,195 | Spherical | 1.5x |
| **Llama 3.1-8B** | **8B** | **128K** | **~44K** | **Compositional + Spherical** | **1.0x** |

the 8B is the first model with compositional init and the first with 1.0x radius. any performance difference is confounded between scale (8B vs 3B) and init strategy. that's fine — the paper can discuss both factors. if we wanted a clean ablation, we'd rerun the 1B with 1.0x radius too. maybe later.

---

## Notes

this model's value is threefold:
1. **Scale point** largest Llama in the lineup. does 8B of frozen transformer + Wake embeddings produce qualitatively different output than 1B or 3B?
2. **Init strategy test** does compositional init actually help? faster convergence? better final embeddings?
3. **Norm gap fix** does 1.0x radius prevent the 7.81 Cohen's d norm separation we saw in every other model?

if the compositional init works well, we retrofit it into future model scripts. if 1.0x radius eliminates the norm gap, same...the 8B is the test bed for improvements.

[Joji...but not the Joji](https://soundcloud.com/rohaanofficial/joji-fuxwithit-premiere?in=may-stevens-846243297/sets/look-at-lil-chano-from-79th&si=7a52b8a8926f46e2bb7b10daee77056e&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

