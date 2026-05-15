### The refined smaller model paradox (prediction, 15 May)

prediction: even after P2 completes, 3B generations will be BETTER than 3B P1 (catastrophically Biblical) but STILL WORSE than TinyLlama P2 (sustained Wakean pastiche) and generation quality recovery is not recovered from val loss as they measure different things.

**food for thought**: P2 improves val by teaching LoRA to route attention through Wake embeddings when context demands. but at inference, the model picks from 172,451 candidates at every step. the base prior across 128K English tokens is overwhelmingly strong from pre-training so LoRA can boost Wake logits when context cues them, but it can't reweight the global priors. wherever context is ambiguous, the model defaults to standard English, which is the structural ceiling on generation authenticity that LoRA cannot break.

TinyLlama doesn't have this problem to the same degree because at 32K base tokens the English priors are less granular. 44.5K Wake injections roughly DOUBLE the vocabulary, so Wake tokens aren't competing against an established 128K-token English distribution at the same density.

**Wake-vocab-share by model:**

| Model | Wake tokens / Total | Wake share |
|---|---|---|
| TinyLlama 1.1B | 44,500 / 76,500 | **58%** |
| Mistral 7B v0.3 | 44,553 / 77,321 | **58%** |
| Phi-3 Mini | ~44,500 / ~76,500 | **~58%** (TBD) |
| Llama 3.2-1B | 44,195 / 172,451 | 26% |
| Llama 3.2-3B | 44,195 / 172,451 | **26%** |
| Llama 3.1-8B | 44,195 / 172,451 | 26% |
| Qwen 2.5-14B | 43,824 / 196,888 | 22% |
| Gemma 2 9B | TBD / 256K+ | likely <15% |

the refined hypothesis: **Wake-vocab-share predicts generation authenticity** more reliably than scale, more reliably than P2 val loss, more reliably than P1 trajectory shape because structural softmax dominance follows tokenizer ratios, not LoRA routing quality.

if this prediction holds:
- 3B P2 generates better than 3B P1 (LoRA rescue), but stuck below the 58% Wake-share threshold ceiling
- Mistral P2 will approach TinyLlama-quality generation (same Wake share, more parameters)
- Phi-3 P2 lands in the same neighbourhood (same Wake share, similar scale to 3B)
- Gemma 2 9B may produce essentially zero Wake content at all (lowest expected Wake share, biggest English prior)

this turns "smaller model paradox" from an anecdote into a measurable axis and is testable across the remaining 7 models.
