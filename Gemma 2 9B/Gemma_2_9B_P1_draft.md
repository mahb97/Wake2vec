# wake2vec on Gemma 2 9B P1 (draft, not yet started)

## Gemma 2 9B P1 (pre-flight)

Google's architecture with interleaved local/global attention, grouped query attention and a **256,000 token vocabulary** (the largest in the lineup by a factor of two).

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

### The 256K vocab problem

this is the big unknown. with 256K base tokens, Gemma's tokenizer might already know:
- most English morphology (prefixes, suffixes, compounds)
- many European language fragments (the Wake's multilingual play)
- common character sequences that other tokenizers split

if the tokenizer covers 90% of Wake forms natively, we'd only inject ~4,500 new tokens instead of ~44,000. that's a fundamentally different experiment:
- **TinyLlama (32K):** learns 44,500 new tokens = builds a new subspace
- **Gemma (256K):** learns ~4,500 new tokens = minor vocab extension

the smaller model paradox predicts Gemma will produce the *worst* Wake output of any model, as it technically has the least to learn, so will fall back on priors the most. but it's also 9B with a novel attention architecture. 

**first step before writing the script:** checking how many Wake lexicon tokens actually need adding by running the tokenizer against the lexicon and count.

### The attention architecture question

Gemma 2 uses interleaved local/global attention:
- odd layers: local attention (4,096 token window)
- even layers: global attention (8,192 token window)

this is neither standard full attention (Llama) nor sliding window (Mistral), instead this alternates between local pattern matching and global context integration, so the Wake's nested parenthetical structure might interact differently with this alternation.

### Planned config

| Param | Value | Notes |
|-------|-------|-------|
| Embedding init | Compositional + spherical 1.0x | Same as 8B (if enough new tokens to matter) |
| Optimizer | AdamW | |
| LR | 2e-4 | May need adjusting given smaller hidden dim |
| Batch | 1 x 16 = 16 effective | |
| SEQ_LEN | 512 | Should fit at ~6-7GB VRAM |
| Max steps | 3,000 (or less if few new tokens) | |
| Eval every | 50 or 100 | |
| Gradient masking | Yes | |

### The comparison matrix (full lineup)

| # | Model | Params | Vocab | Wake injection | Attention | Training data |
|---|-------|--------|-------|----------------|-----------|---------------|
| 1 | TinyLlama | 1.1B | 32K | ~44,500 (full) | Standard | Internet |
| 2 | Llama 1B | 1B | 128K | 44,195 | Standard | Internet |
| 3 | Llama 3B | 3B | 128K | 44,195 | Standard | Internet |
| 4 | Llama 8B | 8B | 128K | ~44K | Standard | Internet |
| 5 | Mistral 7B | 7B | 32K | 44,553 | Sliding window | Internet |
| 6 | Phi-3 Mini | 3.8B | 32K | ~44,500 | Standard | Textbook |
| 7 | **Gemma 2 9B** | **9B** | **256K** | **TBD (minimal?)** | **Local/global interleaved** | **Internet** |
| 8 | Qwen 14B | 14B | 152K | 43,824 | Standard | Internet |

---

## Notes

Gemma runs on the same account as Mistral (account 4) once Mistral finishes P1. 

the first step before scripting: `!pip install transformers && python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('google/gemma-2-9b'); lexicon = open('wake_lexicon.txt').read().splitlines(); new = sum(1 for w in lexicon if len(tok.encode(w, add_special_tokens=False)) > 1 or tok.encode(w, add_special_tokens=False)[0] >= 256000); print(f'New tokens needed: {new}')"` 

...or something like that. the number determines whether this is a real embedding injection experiment or a formality.
