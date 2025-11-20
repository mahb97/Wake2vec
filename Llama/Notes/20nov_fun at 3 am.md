## Llama-3.2-1B vs TinyLlama-1.1B: Wake Vocabulary Learning

**Training loss at step 300:**

| Model | Loss | Reduction from start |
|-------|------|---------------------|
| TinyLlama-1.1B (fp32 embeddings) | 2.45 | 71% |
| Llama-3.2-1B (4-bit + embedding-only) | 0.61 | 81% |

**Configuration:**
- Both models trained on same Wake corpus (910 samples)
- Both extended with 44,990 Wake-specific tokens (32K → 76.5K vocab)
- TinyLlama: seq_len=256, lr=5e-4, full fp32
- Llama-3.2-1B: seq_len=512, lr=5e-5, 4-bit quantized

**Observation:** Llama-3.2-1B learns Wake vocabulary significantly faster despite 4-bit quantization and longer sequences. Loss converges 4x lower at same step count. Likely due to stronger base model quality and better multilingual pretraining.
