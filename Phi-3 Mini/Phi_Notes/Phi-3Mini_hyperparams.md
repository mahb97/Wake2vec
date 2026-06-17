# wake2vec Phi-3 Mini 3.8B P1 

## Phi-3 Mini 3.8B P1 

textbooks over webscraping, "textbook quality data" as the selling point (Marx is crying, so am I).

the question: does a model trained on clean data resist Joyce's linguistic chaos, or does it embrace it more precisely? does knowing English "properly" make you better or worse at learning to break it?

### Architecture & vocab

| Param | Value | Notes |
|-------|-------|-------|
| Model | **microsoft/Phi-3.5-mini-instruct** | 4-bit NF4. (variant decided: 3.5 over 3-mini-4k) |
| Params | 3.8B | Same scale as Llama 3B (direct comparison) |
| Base vocab (tokenizer) | **32,011** | the real tokenizer length |
| Base vocab (padded matrix) | 32,064 | model embedding padded above tokenizer — the gotcha (see below) |
| Wake tokens added | **44,500** | (490 already in vocab, not re-added) |
| Final vocab | **76,511** | |
| **Wake-vocab-share** | **58.2%** | 44,500 / 76,511. CONFIRMED in TinyLlama/Mistral cohort |
| Hidden dim | 3,072 | **Identical to Llama 3.2-3B** so clean architectural comparison |
| Layers | 32 | |
| Attn heads | 32 | |
| VRAM (4-bit, SEQ_LEN 512) | **5.36 GB** | comfortable on T4, no offload |
| Trainable params | 235,041,792 | full embedding matrix (76,511 × 3,072), gradient-masked to Wake rows |


### The 2x2 design with Phi-3

Phi-3 sits at the textbook corner of a 2x2 controlled comparison embedded in the lineup:

|              | 32K vocab               | 128K vocab    |
|--------------|-------------------------|---------------|
| ~3-4B params | **Phi-3 (textbook)**    | Llama 3.2-3B  |
| ~7-8B params | Mistral 7B (internet)   | Llama 3.1-8B  |

**Phi-3.5 vs Llama 3.2-3B is the cleanest single-variable comparison in the entire project.** same hidden dim (3,072), same scale (3-4B), same decoder-only transformer architecture. only differences: vocab size (32K vs 128K) and training data (textbook vs internet). every other dimension is held constant. Llama 3.2-3B P1 best val was 6.68 (step 300) so that is the number Phi gets compared against.

### Where Phi-3 fits in the smaller model paradox

so far:
- TinyLlama 1.1B / 32K vocab → good Wake output (small + small vocab + internet)
- Llama 3.2-1B / 128K vocab → worse Wake output (small + large vocab + internet)
- Mistral 7B / 32K vocab → P1 in progress (large + small vocab + internet)
- **Phi-3 Mini / 32K vocab / textbook training** → **the fourth data point**

the simple paradox (vocab share predicts Wake quality) was refined after Qwen P2: generation quality is achievable across multiple points in (Wake-vocab-share, scale, training-depth) space. the cohort map:

- **58% share** (compute-efficient): TinyLlama 1.1B (done, good Wake), Mistral 7B (P1 running), **Phi-3.5 3.8B (P1 launching)**
- **26% share** (medium band): Llama 3.2-1B (done), Llama 3.2-3B (done, coherent-English and sparse invention)
- **22% share** (brute-force): Qwen 14B (P2 running, broke the LoRA wall, dense polyglot Wake)
- **~17% share**: Gemma 2 9B (pending)

Phi-3.5 is the textbook variable inside the 58% cohort. it will show:
1. **Does the paradox survive textbook training?** if Phi generates Wake as densely as TinyLlama, vocab-share dominates and training-data cleanliness doesn't gate Wake acquisition. if worse, training data quality also matters.
2. **Does the cross-architecture geometric null hold for Microsoft architecture?** Phi is decoder-only but a different family. P3 results would extend the null (currently confirmed on TinyLlama, Llama 1B, Llama 3B, Qwen P1) to a third architecture family.

### Two competing hypotheses

if Phi produces *worse* Wake output than TinyLlama despite matching the 58% share, it's because textbook priors are *too* clean — the model learned a version of English too orderly to deform into Joyce.

if Phi produces *better* (or equal) Wake output, it's because understanding English properly is a prerequisite for breaking it properly. Joyce knew the rules before he broke them. maybe models do too.

### Planned config

| Param | Value | Notes |
|-------|-------|-------|
| Embedding init | **spherical 1.5x base_radius** | radius 1.6644. CHANGED from draft's compositional+1.0x and chose spherical for cohort comparability (matches TinyLlama, Mistral, 3B). compositional 1.0x was the 8B-only experiment, not adopted as default. |
| Optimizer | AdamW | |
| LR | 2e-4 | |
| Batch | 1 x 16 = 16 effective | |
| SEQ_LEN | 512 | confirmed fits at 5.36GB |
| Max steps | 3,000 | nominal — expect U-curve best-val early (3B P1 peaked at step 300) |
| Eval every | 100 | |
| Gradient masking | Yes | base rows 0-32,010 frozen, Wake rows 32,011-76,510 trained |
| tie_word_embeddings | **manual** | Phi ships untied (config default False); we tie manually via `lm_head.weight = wte.weight` for cohort comparability |
| Training data | FW + lexicon + **wake_embedding_groups.jsonl** | JSONL morpheme groups added as co-occurrence text (258 groups, ~6,048 examples) |

1. **Padding-gap boundary.** Phi-3.5's tokenizer has 32,011 real tokens but the embedding matrix is padded to 32,064. The standard injection logic (boundary = model embedding size) would have misclassified the first 53 Wake tokens as base and hit a shape mismatch. Fixed by anchoring the boundary on `wake_start = len(tok) - num_added = 32,011` (the tokenizer boundary). All 44,500 Wake tokens correctly positioned.

2. **tie_weights() transformers 5.x bug.** `get_expanded_tied_weights_keys` crashes on Phi's tied-weights mapping (`'list' object has no attribute 'keys'`). Fixed by setting `tie_word_embeddings=False` (Phi's native default) and tying manually. Architectural note: **Phi ships with untied input/output embeddings**, unlike the Llama family, its output head is specialized rather than weight-shared, consistent with its clean-reasoning design. 

### Speed and the U-curve

at ~101s/step (SEQ_LEN 512 × batch 16 = 8,192 tokens/step, fully on-GPU, gradient checkpointing + eager attention), the full 3,000 steps would be ~84 GPU-hours. but the useful result lands far earlier: embedding-only P1 U-curves, and 3B P1's best val was at step 300. Phi will likely hit best-val somewhere in step 200-600 (~6-17 GPU-hours, 2-5 T4 sessions), then val rises. watch the U-curve; grab best-val for the P2 source. the 3,000-step commitment is nominal cover.

---

[I can't tell](https://soundcloud.com/flume/i-cant-tell-feat-laurel?si=f32660c4c35942acb793a221caa8f1ff&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
