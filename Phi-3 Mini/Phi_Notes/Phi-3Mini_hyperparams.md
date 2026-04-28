# wake2vec Phi-3 Mini 3.8B P1 (draft, not yet started)

## Phi-3 Mini 3.8B P1 

textbooks over webscraping, "textbook quality data" as the selling point (Marx is crying, so am I).

the question: does a model trained on clean data resist Joyce's linguistic chaos, or does it embrace it more precisely? does knowing English "properly" make you better or worse at learning to break it?

### Architecture & vocab

| Param | Value | Notes |
|-------|-------|-------|
| Model | microsoft/Phi-3-mini-4k-instruct | 4-bit NF4 (or Phi-3.5-mini-instruct, decide before script) |
| Params | 3.8B | Same scale as Llama 3B (direct comparison) |
| Base vocab | 32,064 | Full Wake injection club (TinyLlama, Mistral, Phi-3) |
| Wake tokens (est.) | ~44,500 | Same as TinyLlama/Mistral 32K vocab means full injection |
| Hidden dim | 3,072 | **Identical to Llama 3.2-3B** so clean architectural comparison |
| Layers | 32 | |
| Attn heads | 32 | |
| VRAM (4-bit) | ~3-4GB | Comfortable on T4 |

### The 2x2 design with Phi-3

Phi-3 sits at the textbook corner of a 2x2 controlled comparison embedded in the lineup:

|              | 32K vocab               | 128K vocab    |
|--------------|-------------------------|---------------|
| ~3-4B params | **Phi-3 (textbook)**    | Llama 3.2-3B  |
| ~7-8B params | Mistral 7B (internet)   | Llama 3.1-8B  |

**Phi-3 vs Llama 3.2-3B is the cleanest single-variable comparison in the entire wake2vec project.** same hidden dim (3,072), same scale (3-4B), same decoder-only transformer architecture. only differences: vocab size (32K vs 128K) and training data (textbook vs internet), every other dimension is held constant.

### Where Phi-3 fits in the smaller model paradox

so far:
- TinyLlama 1.1B / 32K vocab → good Wake output (small + small vocab + internet)
- Llama 3.2-1B / 128K vocab → worse Wake output (small + large vocab + internet)
- Mistral 7B / 32K vocab → P1 in progress (large + small vocab + internet)
- **Phi-3 Mini / 32K vocab / textbook training** → **the fourth data point**

Phi-3 will show:
1. **Does the paradox survive textbook training?** if yes, the pattern is about vocab size, not training data quality. if no, training data quality also matters.
2. **Does the cross-architecture null hold for Microsoft architecture?** Phi-3 is decoder-only but a different family from Llama/Mistral. P3 results on Phi-3 would extend the null to a third architecture family.

### Two competing hypotheses

if Phi-3 produces *worse* Wake output than TinyLlama despite being 3.5x larger, it's because textbook priors are *too* clean and the model learned a version of English that's too orderly to deform into Joyce.

if Phi-3 produces *better* Wake output, it's because understanding English "properly" is a prerequisite for breaking it properly. Joyce knew the rules before he broke them. maybe models do too.

### Planned config

| Param | Value | Notes |
|-------|-------|-------|
| Embedding init | Compositional + spherical 1.0x | Same improvements as 8B |
| Optimizer | AdamW | |
| LR | 2e-4 | |
| Batch | 1 x 16 = 16 effective | |
| SEQ_LEN | 512 | Should fit easily at 3-4GB VRAM |
| Max steps | 3,000 | |
| Eval every | 50 or 100 | dopamine management |
| Gradient masking | Yes | |

---

[I can't tell](https://soundcloud.com/flume/i-cant-tell-feat-laurel?si=f32660c4c35942acb793a221caa8f1ff&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
