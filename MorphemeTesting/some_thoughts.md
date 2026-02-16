## Phase 3: Morpheme-Compositional Alignment (draft)

[Winny](https://soundcloud.com/fredagain/winny?in=houseof_kyri/sets/sal-paradise&si=1feb70e0ca184f8d887456289794cccb&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

---

Drafted `wake2vec_phase_3_morpheme.py` P3 script for TinyLlama. The idea: load from the best P2 checkpoint (step 1400, val gap 0.001), unfreeze Wake embedding rows and keep LoRA adapters trainable, then train with a composite loss that includes morpheme alignment.

### Why step 1400

Step 1400 has the tightest train/val gap (0.001) of the entire P2 run. After that, val loss climbs monotonically while train drops. The model at 1400 has learned to use the Wake embeddings without memorizing the training set so is used as the basis for P3. 

### Loss function

```
L_total = L_lm + 0.1 * L_morpheme + 0.05 * L_repulsion + 0.01 * L_norm
```

- **L_lm:** standard causal LM loss (same as P2)
- **L_morpheme:** MSE between Wake token embeddings and their morpheme centroid targets. Uses `morpheme_data.json` (15 prefixes, 15 suffixes, 10 example words each). Greedy longest-match decomposition. Only tokens with a prefix or suffix match participate
- **L_repulsion:** penalizes Wake token pairs with cosine similarity > 0.95. Prevents embedding collapse. Random 100 pairs per step
- **L_norm:** penalizes Wake token norms that deviate too far from base vocab norm distribution (mean ± 1 std margin)

### What's trainable

- Wake embedding rows (44,989 rows, gradient-masked on base vocab)
- LoRA adapters (same r=8 on q/k/v/gate/up/down from P2)
- Everything else frozen

### Config

- LR: 5e-5 (the embeds haven't been trained since P1 and the aim is to get them moving toward morpheme targets)
- Steps: 3000 max, early stopping patience 5 evals (eval every 200 steps)
- Batch: 8 (effective 16 with grad accum 2)
- AdamW, cosine schedule, 10% warmup

### Status

Script drafted, not run yet. The morpheme decomposition is automatic (greedy prefix/suffix match) but rough. The plan to manually annotate better decompositions later and feed them back in.
