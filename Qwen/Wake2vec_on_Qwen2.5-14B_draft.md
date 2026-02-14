## 2026-02-14 Qwen 2.5-14B draft outline

Rough sketch for running wake2vec on the biggest model that'll fit on a free T4. Not a script yet, just the work in progress. 

### Qwen 2.5-14B

- 14B params so biggest that can squeeze onto T4 at 4-bit (~8-9GB weights + ~3-4GB LoRA/optimizer/activations ≈ 12-13GB, close one)
- Already has a 152K vocab (vs TinyLlama's 32K). This is interesting because a lot of Wake neologisms might already partially tokenize better with a bigger vocab, so less embedding disruption in theory. The 44,989 Wake tokens to be added represent a ~30% vocab expansion for Qwen vs ~140% for TinyLlama.
- 48 layers, hidden dim 5120, GQA (40 heads / 8 KV heads)
- Trained on 18T tokens, so way more world knowledge baked in, which might help with Joyce's multilingual mashups

### P1: Embeds-only

```
Model:           Qwen/Qwen2.5-14B-Instruct
Quantization:    none for P1. actually wait. 14B at fp32 is ~56GB. even fp16 is 28GB.
                 so NEEDs 4-bit for P1 too, or do embedding-only on CPU? lolll.

                 okay dropping the games, real plan: load model in 4-bit NF4, but keep the
                 embedding layer in fp32 (the only trainable part).
                 4-bit model body ≈ 8GB, fp32 embeddings for 197K tokens
                 (152K base + 45K wake) × 5120 dim = ~4GB fp32.
                 total ≈ 12GB. tight but should work.

Vocab extension: 152,064 (base) + 44,989 (wake) = 197,053 tokens
Init:            mean of existing embeddings (same as TinyLlama P1)
Trainable:       embedding layer only (~1B params in fp32)
Frozen:          everything else (4-bit quantized)
Tied embeds:     yes (input = output)
Gradient mask:   only wake token rows get gradients, base 152K frozen

LR:              5e-4 (same as TinyLlama P1)
Warmup:          5%
Batch:           1 (grad accum 16 → effective 16)
Seq len:         256
Optimizer:       Adafactor (memory efficient, no momentum states)
Steps:           1300 (same as TinyLlama for fair comparison)
```

**the gamble:** the embedding matrix alone is ~4GB in fp32 at 197K × 5120. Adafactor doesn't store momentum so that helps, but gradient computation on a 4GB tensor is gonna be spicy on a 15GB GPU. Might need to drop batch size or seq len if it OOMs.

**alternative:** freeze the base 152K rows *in the embedding matrix itself* and only allocate gradients for the 45K new rows. This is what the gradient mask does but could go further and literally slice the embedding to save memory. probs overthinking it for now.

### P2: LoRA (same pattern as TinyLlama)

```
Model:           Qwen/Qwen2.5-14B-Instruct (4-bit NF4)
P1 source:       load P1-trained embeddings, freeze them
LoRA:            r=8, alpha=16, dropout=0.1
Targets:         q_proj, k_proj, v_proj, gate_proj, up_proj, down_proj
                 (same target names — Qwen2 uses same projection naming as Llama)
LR:              2e-5
Batch:           4 (might need to drop from 8 due to model size)
Grad accum:      4 (keep effective batch 16)
Seq len:         256
Steps:           3000
```

**VRAM budget for P2:**
- 4-bit model: ~8GB
- LoRA adapters (r=8 on 48 layers × 6 targets): ~100MB
- Frozen P1 embeddings: can keep in fp16 since not training → ~2GB
- Optimizer states for LoRA params: ~200MB
- Activations/gradients: ~2-3GB (depends on batch × seq len)
- **Total: ~12-13GB** should fit but no room for mistakes (what did the wake say? terror in errorland lol) 

### Data

Same as TinyLlama runs:
- **Training corpus:** `FW_TEXT.txt` (Finnegans Wake, 24,483 lines)
- **Lexicon injection:** `wake_lexicon.txt` (44,989 tokens)
- Tokenize FW_TEXT with extended tokenizer, chunk into 256-token blocks
- 90/10 train/val split, seed 42

### What might actually be different from TinyLlama

1. **Tokenization overlap** Qwen's 152K vocab probably already covers a bunch of the "weird" tokens that TinyLlama fragments. Need to check how many of the 44,989 wake tokens are actually *new* to Qwen vs already in its vocab. Could be significantly fewer new tokens needed
2. **Multilingual knowledge** Qwen is trained on way more multilingual data than TinyLlama. Joyce's French/Italian/German/Irish mashups might already be partially "understood"
3. **Generation quality** bigger model = more coherent long-form text, but also = more "normal" language tendencies to fight against. The whole point is to make it write wake-ish. 
4. **Training speed** — gonna be slow as f on T4. TinyLlama P1 took 7ish hours for 3000 steps, 14B will be significantly slower even quantized

### Open questions

- How many of the 44,989 wake tokens does Qwen already know? (test: tokenize each one and see if it's a single token)
- Is 4-bit embedding-only training even stable? did fp32 for TinyLlama P1 specifically for embedding stability
- Can this get away with fp16 embeddings instead of fp32 to save ~2GB?
- Should it use the Instruct variant or base? Instruct has chat formatting baked in which I don't really care about, but it might follow the "style" better. base is probably cleaner for this
- [Drop the Game](https://soundcloud.com/futureclassic/flume-chet-faker-drop-the-3?in=houseof_kyri/sets/for-when-that-t4-hits&si=77420d1245da4e7cbb27335b6aab45ee&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)


