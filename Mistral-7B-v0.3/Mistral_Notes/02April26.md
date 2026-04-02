# wake2vec devlog 2026-04-02

## Mistral 7B v0.3 P1 session 2 (resuming from step 150)

the first non-Llama, non-Qwen model in the lineup: sliding window attention, 32K base vocab, 44,553 Wake tokens injected. SEQ_LEN dropped to 256 after first session hit 163s/step at 512, but now running at ~82s/step.

val dropped from 11.37 to 11.13 in the first 100 steps. 

Resuming from `checkpoint-150`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 50 | 186.78 | 11.37 | 1 |
| 100 | 181.33 | 11.13 | 1 |
| 150+ | | | *resuming today, session 2* |

### Config

| Param | Value |
|-------|-------|
| Model | mistralai/Mistral-7B-v0.3 (4-bit NF4) |
| Embedding strategy | Gradient masking |
| Optimizer | AdamW |
| LR | 2e-4 |
| Batch | 1 x 16 = 16 effective |
| SEQ_LEN | 256 (reduced from 512) |
| Max steps | 3,000 |
| Eval every | 50 (for dopamine management) |
| Vocab | 32,768 → 77,321 (+44,553 Wake tokens) |
| Sliding window | Yes (Mistral architecture) |

---

## Colab update (April 2026)

Minor package bumps: accelerate 1.12→1.13, huggingface_hub 1.4→1.7, keras 3.10→3.13, but nothing that affects the Wake2vec pipeline. 

---

## Notes

Mistral is the key architectural comparison. Same 32K vocab as TinyLlama, so will get the full Wake injection treatment (~44K new tokens). If the smaller model paradox holds (if Mistral's 32K vocab produces better Wake output than Llama 8B's 128K vocab) that's a second independent confirmation that tokenizer gap drives stylistic quality. And the sliding window attention might handle Wake's long parenthetical nesting differently from standard attention.

---
[Turning](https://soundcloud.com/flume/collarbones-turning-flume-remix?si=c494e3acf9f244c49d7052b56e384266&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

In the surveillance state we're all cam girls. 
