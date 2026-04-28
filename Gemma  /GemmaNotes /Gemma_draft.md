# wake2vec on Gemma: three models, draft, not yet started

## Overview

three Gemma models for wake2vec: Gemma 2 9B (standard architecture, 256K vocab), Gemma 3n E2B and E4B (efficient-architecture variants with PLE + MatFormer). all three share the same 256,000-token tokenizer, so have the largest vocab in the entire lineup by a factor of two.

this might be the model that breaks the smaller model paradox, or confirms it beyond all doubt.

### Architecture & vocab

| Param | Value | Notes |
|-------|-------|-------|
| Model | google/gemma-2-9b | 4-bit NF4 |
| Params | 9B | Between Mistral 7B and Qwen 14B |
| Base vocab | 256,000 | **Largest vocab in the lineup. 8x TinyLlama.** |
| Wake tokens (est.) | **TBD — might be very few** | 256K tokenizer likely covers most Wake forms |
| Hidden dim | 2,304 | Smaller than Llama 8B (4,096) despite similar params |
| Layers | 26 | |
| Attn heads | 8 (GQA, 4 KV heads) | Interleaved local (4096) / global (8192) attention |
| VRAM (4-bit) | ~6-7GB | Should fit on T4 |

these three models test two different research questions simultaneously:
1. **does vocab size matter for Wake injection?** the 256K tokenizer might already cover most Wake forms, making this a minimal-injection experiment compared to the 32K models (TinyLlama, Mistral, Phi-3).
2. **does always-active capacity matter for stylistic learning?** Gemma 3n's selective-activation architecture tests whether Wake embedding injection works on models designed for efficient inference, where most of the network is conditional.

---

## Gemma 2 9B (the standard variant)

Google's architecture with interleaved local/global attention, grouped query attention, and a 256K vocab. this might be the model that breaks the smaller model paradox, or confirms it beyond all doubt.

### Architecture & vocab

| Param | Value | Notes |
|-------|-------|-------|
| Model | google/gemma-2-9b | 4-bit NF4 |
| Params | 9B | Between Mistral 7B and Qwen 14B |
| Base vocab | 256,000 | **Largest vocab in the lineup. 8x TinyLlama.** |
| Wake tokens (est.) | TBD — likely minimal | 256K tokenizer likely covers most Wake forms |
| Hidden dim | 2,304 | Smaller than Llama 8B (4,096) despite similar params |
| Layers | 26 | |
| Attn heads | 8 (GQA, 4 KV heads) | Interleaved local (4,096) / global (8,192) attention |
| VRAM (4-bit) | ~6-7GB | Should fit on T4 |

### The attention architecture

Gemma 2 uses interleaved local/global attention:
- odd layers: local attention (4,096 token window)
- even layers: global attention (8,192 token window)

this is neither standard full attention (Llama) nor sliding window (Mistral). it alternates between local pattern matching and global context integration. the Wake's nested parenthetical structure might interact differently with this alternation.

---

## Gemma 3n E2B & E4B (the efficient variants)

these are Google's mobile/edge-oriented Gemma 3n family. they use Per-Layer Embeddings (PLE) and MatFormer architecture — selective activation of sub-networks rather than always-on dense weights. E2B has ~5B total params running as ~2B effective; E4B has ~8B total running as ~4B effective.

### Architecture & vocab

| Param | E2B | E4B | Notes |
|-------|-----|-----|-------|
| Model | google/gemma-3n-E2B | google/gemma-3n-E4B | 4-bit NF4 |
| Total params | ~5B | ~8B | |
| Effective params | 2B | 4B | Selective activation |
| Base vocab | 256,000 | 256,000 | Same Gemma 2 tokenizer |
| Wake tokens (est.) | TBD — likely minimal | TBD — likely minimal | Same as Gemma 2 9B |
| Architecture | PLE + MatFormer | PLE + MatFormer | **Efficient/selective** |
| VRAM (4-bit) | ~3-4GB | ~5-6GB | |

### Why E2B and E4B specifically

these are the **first selectively-activated models** in the lineup. every other model (Llama, Mistral, Qwen, Phi) uses always-active dense weights — every parameter contributes to every forward pass. Gemma 3n's PLE + MatFormer means most of the network is conditional, activated only when relevant.

the question: **does Wake embedding injection work the same way on a model where most weights aren't always firing?** if yes, the smaller model paradox should predict E2B's behaviour (32K-style injection is impossible because the vocab is 256K, but the effective active-parameter count is small enough to potentially compensate). if no, that's a finding about whether stylistic capacity needs always-active pathways.

E2B and E4B form a within-family comparison: same architecture class, different effective capacity. nothing else in the lineup has this structure.

---

## The 256K vocab problem (shared across all three)

with 256K base tokens, the Gemma tokenizer might already know:
- most English morphology (prefixes, suffixes, compounds)
- many European language fragments (the Wake's multilingual play)
- common character sequences that other tokenizers split

if the tokenizer covers 90% of Wake forms natively, we'd inject ~4,500 new tokens instead of ~44,000. that's a fundamentally different experiment:
- **TinyLlama (32K):** learns 44,500 new tokens = builds a new subspace
- **Gemma (256K):** learns ~4,500 new tokens = minor vocab extension

the smaller model paradox predicts all three Gemmas will produce the *worst* Wake output of any model — they have the least to learn, so they'll fall back on priors the most. **first step before writing scripts:** run the tokenizer against the lexicon and count actual injection size.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('google/gemma-2-9b')
with open('wake_lexicon.txt') as f:
    lexicon = [w.strip() for w in f if w.strip()]
new_count = sum(1 for w in lexicon if len(tok.encode(w, add_special_tokens=False)) > 1)
print(f"New tokens needed: {new_count}")
```

(Gemma 3n E2B/E4B share the same tokenizer, so this number applies to all three.)

---

## Planned config (all three Gemmas)

| Param | Value | Notes |
|-------|-------|-------|
| Embedding init | Compositional + spherical 1.0x | Same improvements as 8B |
| Optimizer | AdamW | |
| LR | 2e-4 | May need adjusting given smaller hidden dim on Gemma 2 9B |
| Batch | 1 x 16 = 16 effective | |
| SEQ_LEN | 512 (start; reduce if needed) | Should fit at ~6-7GB VRAM |
| Max steps | 3,000 (or fewer if minimal injection) | |
| Eval every | 50 or 100 | |
| Gradient masking | Yes | |

if the actual injection count is small (< 5,000 tokens), we may run shorter trainings (1,000-2,000 steps) since there's less to learn.

---

## The comparison matrix (full 10-model lineup)

| # | Model | Params | Vocab | Wake injection | Attention | Training data | Special |
|---|-------|--------|-------|----------------|-----------|---------------|---------|
| 1 | TinyLlama | 1.1B | 32K | ~44,500 | Standard | Internet | smallest |
| 2 | Llama 3.2-1B | 1B | 128K | 44,195 | Standard | Internet | |
| 3 | Llama 3.2-3B | 3B | 128K | 44,195 | Standard | Internet | |
| 4 | Llama 3.1-8B | 8B | 128K | 44,195 | Standard | Internet | compositional init |
| 5 | Mistral 7B | 7B | 32K | 44,553 | **Sliding window** | Internet | |
| 6 | Phi-3 Mini | 3.8B | 32K | ~44,500 | Standard | **Textbook** | |
| 7 | **Gemma 2 9B** | **9B** | **256K** | **TBD (minimal?)** | **Interleaved local/global** | Internet | largest vocab |
| 8 | **Gemma 3n E2B** | **~5B (2B eff.)** | **256K** | **TBD (minimal?)** | **PLE + MatFormer** | Internet | **selective activation** |
| 9 | **Gemma 3n E4B** | **~8B (4B eff.)** | **256K** | **TBD (minimal?)** | **PLE + MatFormer** | Internet | **selective activation** |
| 10 | Qwen 14B | 14B | 152K | 43,824 | Standard | Internet | WakeOverlay |

---

## What the three Gemmas test

| Question | Tested by |
|----------|-----------|
| Does vocab size matter for Wake injection effectiveness? | Gemma 2 9B vs other 9B-class models |
| Does selective activation handle Wake injection? | Gemma 3n E2B & E4B |
| Within-family scale comparison for selective architectures | E2B vs E4B |
| Does the smaller model paradox hold at 256K vocab? | All three Gemmas combined |

---

## Notes

Gemma 2 9B runs on the same account as Mistral once Mistral finishes P1. Gemma 3n E2B and E4B will need their own slots — likely accounts 5 and 6 (yes, more accounts. don't ask).

if the actual Wake injection count is very low (e.g. <2,000 tokens), the experiment becomes a *negative result* — there's nothing to inject, so nothing to compare. that itself is a finding: the smaller model paradox is a phenomenon of small-vocab models specifically, and large-vocab models simply don't participate in this comparison space.

either way, the three Gemmas anchor the high-vocab end of the comparison axis. the paper needs them to ground the contrast with TinyLlama and Mistral on the low-vocab end.
