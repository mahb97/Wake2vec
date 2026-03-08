# wake2vec devlog 2026-03-08

## TinyLlama 1.1B P3 the morpheme era 

first model to reach P3, I have three google accounts so I though I might as well kick this off. 

### What P3 is

P2 taught the model to route Wake tokens through attention, but P3 asks a harder question: do the embeddings preserve morphological structure?

device contrastive loss: portmanteaux should cluster together, malapropisms should cluster together, puns should cluster together. Not because they mean similar things, but because they *do* similar things to language.

μp → UP. This is the arrow.

### Script

`wake2vec_phase_3_morpheme_TinyLlama.py`

### Data

| Dataset | File | Size |
|---------|------|------|
| Finnegans Wake (full text) | `FW_TEXT.txt` | 24,478 lines |
| Morpheme groups | `wake_embedding_groups.jsonl` | 258 groups, 6,048 words |
| Device groups | `device_groups.jsonl` | 5 types, 2,123 words |

### Config

| Param | Value |
|-------|-------|
| P2 source | `wake2vec_tiny_p2_lora/full_checkpoints/step_1400` |
| Model | TinyLlama-1.1B-Chat-v1.0 (4-bit NF4) |
| Trainable params | 162,258,944 (Wake embed rows + LoRA adapters) |
| LoRA | Inherited from P2 (rank 8, alpha 16), stays trainable |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj` |
| LR | 5e-5 |
| Batch | 8 x 2 = 16 effective |
| SEQ_LEN | 256 |
| Train blocks | 1,489 |
| Val blocks | 166 |
| Max steps | 3000 |
| Early stop patience | 5 |

### Loss function

```
L_total = L_lm + 0.1 * L_morpheme + 0.05 * L_device + 0.05 * L_repulsion + 0.01 * L_norm
```

| Loss | Lambda | What it does |
|------|--------|-------------|
| L_lm | 1.0 | Language modelling on FW text (keeps the model coherent) |
| L_morpheme | 0.1 | Direction consistency within morpheme groups |
| L_device | 0.05 | Triplet margin across 5 device types (64 triplets/step, margin = 0.2) |
| L_repulsion | 0.05 | Prevents Wake token collapse (cosine > 0.95 penalised) |
| L_norm | 0.01 | Keeps Wake norms near base vocab norms |

### P3 loss table

| Step | L_total | L_lm | L_morph | L_device | L_repul | L_norm |
|------|---------|------|---------|----------|---------|--------|
| 0 | 5.9006 | 5.8904 | 0.0002 | 0.1826 | 0.0002 | 0.1055 | 141.7* |
| 50 | 4.8149 | 4.8048 | 0.0002 | 0.1801 | 0.0002 | 0.1055 | 12.7 |
| 100 | 4.3073 | 4.2971 | 0.0002 | 0.1830 | 0.0002 | 0.1055 | 11.3 |
| 150 | 3.9128 | 3.9016 | 0.0002 | 0.2016 | 0.0002 | 0.1055 | |

### Step 0 observations

opening numbers breakdown:

1. **L_morph = 0.0002** is absurdly low. P2 already learned compositional direction consistency almost entirely on its own. the morpheme groups are *already* coherent in embedding space. 

2. **L_device = 0.21** is the one with room to move. 0.21 on a triplet margin loss (margin 0.2) means most triplets are barely satisfied or just barely violated. the device clusters exist but they're soft. this should be the primary driver of embedding movement in early P3.

3. **L_repulsion = 0.0001** has no collapse. Wake tokens are well-separated.

4. **L_norm = 0.1055** the Wake token norms are slightly off from base vocab. 

5. **L_lm = 5.89** notably higher than P2 val (0.6393). expected: P3 computes LM loss over the full FW text blocks (1,489 train blocks) with Wake tokens in context, not just the LoRA-adapted perplexity measure from P2.

**bottom line:** the auxiliary losses are lightweight passengers right now — L_lm is 99.8% of the total. the interesting signal will be whether L_device drops meaningfully over the first 200 steps while L_morph stays pinned near zero.

---
some gigi for you [sailor song](https://soundcloud.com/gigi4perez/sailor-song?si=3f4869c1b2064d7f8762a507d000921c&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
