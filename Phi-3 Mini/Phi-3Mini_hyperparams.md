# wake2vec Phi-3 Mini 3.8B P1 (draft, not yet started)

## Phi-3 Mini 3.8B P1 (pre-flight)

textbooks over webscraping, "textbook quality data" as the selling point (Marx is crying, so am I). 

the question: does a model trained on clean data resist Joyce's linguistic chaos, or does it embrace it more precisely? does knowing English "properly" make you better or worse at learning to break it?

### Architecture & vocab

| Param | Value | Notes |
|-------|-------|-------|
| Model | microsoft/Phi-3-mini-4k-instruct | 4-bit NF4 |
| Params | 3.8B | Same scale as Llama 3B (direct comparison) |
| Base vocab | 32,064 | Full Wake injection club (TinyLlama, Mistral) |
| Wake tokens (est.) | ~44,500 | Same as TinyLlama/Mistral — 32K vocab means full injection |
| Hidden dim | 3,072 | Same as Llama 3B |
| Layers | 32 | |
| Attn heads | 32 | |
| VRAM (4-bit) | ~3-4GB | Comfortable on T4 |

### The comparison matrix

Phi-3 sits at a fascinating intersection:

| Model | Params | Vocab | Training data | Wake injection |
|-------|--------|-------|---------------|----------------|
| TinyLlama 1.1B | 1.1B | 32K | Internet | ~44,500 (full) |
| Phi-3 Mini | 3.8B | 32K | **Textbook** | ~44,500 (full) |
| Llama 3.2-3B | 3B | 128K | Internet | 44,195 (partial) |
| Mistral 7B | 7B | 32K | Internet | 44,553 (full) |

same vocab as TinyLlama and Mistral → same injection volume → same embedding subspace challenge, but with different training data philosophy. if Phi-3 produces worse Wake output than TinyLlama despite being 3.5x larger, it's because textbook priors are *too* clean, aka the model learned a version of English that's too orderly to deform into Joyce.

if Phi-3 produces *better* Wake output, it's because understanding English properly is a prerequisite for breaking it properly. Joyce knew the rules before he broke them, and maybe models do too.

### Planned config

| Param | Value | Notes |
|-------|-------|-------|
| Embedding init | Compositional + spherical 1.0x | Same improvements as 8B |
| Optimizer | AdamW | |
| LR | 2e-4 | |
| Batch | 1 x 16 = 16 effective | |
| SEQ_LEN | 512 | Should fit easily at 3-4GB VRAM |
| Max steps | 3,000 | |
| Eval every | 50 or 100 | |
| Gradient masking | Yes | |

---

## Notes

Phi-3 runs on the same account as Llama 3B (account 3) once the 3B finishes. at step 900/3000, the 3B has ~2,100 steps to go at ~80 steps per session (roughly 26 more sessions), so Phi-3 waits.

the textbook vs internet comparison is interesting, no one has tested whether training data quality affects embedding injection effectiveness. if there's a finding here, it's a contribution that goes beyond Joyce and into the embedding surgery literature.
