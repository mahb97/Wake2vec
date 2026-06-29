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

---

### Outcome and refinement (29 June 2026)

The 15 May prediction stays on record above, unedited. Six weeks and several completed pipelines later, the simple form of it (Wake-share alone predicts generation authenticity) does not hold. The falsification is clean and it is Qwen.

**The prediction ledger, as the trained models stand:**

- **3B (26% share): held.** P1 was catastrophically Biblical as predicted. P2 improved on P1 but reached a fixed value at val 5.33 and generates coherent English with sparse invention, the under-deformation (pastiche) pole, below TinyLlama's suspension. The "LoRA rescue, but capped below the high-share threshold" reading survives.
- **TinyLlama (58% share): held as the reference.** Still the model that holds the suspension line, novel forms inside recoverable syntax. It remains the standard the others are measured against.
- **Mistral (58% share): under test, evidence favourable.** Share confirmed at 58%. P1 produced the largest embedding reorganisation in the lineup (Wake drift cosine 0.485, the only model below the 0.998 isotropy value, at 0.995). The "Mistral P2 approaches TinyLlama quality" prediction is not yet decided; the suspension test is its P2 generation, still pending. P1 gives it the most favourable prior of any model, but that is a prior, not a result.
- **Phi-3.5 (58.2% share): complicated the hypothesis.** Same share as TinyLlama and Mistral, so the simple prediction said Phi should land in that neighbourhood. At P1 it did not: the textbook-quality (instruct-tuned) training prior absorbed Wake tokens locally but did not reorganise into a Wake subspace that generalises (validation never left the random baseline). Share did not guarantee learning. This surfaced an axis the 15 May note did not anticipate, training-data composition. Phi's P2 is still to come, but its P1 already shows share is necessary, not sufficient.
- **Qwen (22% share, the lowest of the trained models): falsified the simple version.** The 15 May logic predicts the lowest share should give the weakest Wake content. Instead Qwen P1 generated the densest polyglot Wake in the lineup (sustained compound-mass, Chinese characters), convincingly Joycean despite the smallest Wake-share. What it had in place of share was scale (14B) and training depth (39 documented warm restarts across 14 weeks, validation never plateauing). Scale plus depth bought what share buys elsewhere. This is the datapoint that breaks "share alone."
- **8B (26% share): pending.** The P2 decider evaluation (step 200) has not yet been observed.
- **Gemma (lowest expected share): not started.** The "near-zero Wake content" prediction is still untested.

**The refinement.** Generation authenticity is not a function of Wake-share alone. It sits in a three-axis space, (Wake-vocab-share, model scale, training depth), with at least two routes to convincing Wake. The high-share, low-compute route (TinyLlama-class, ~58% share, 1.1B) and the low-share, high-compute route (Qwen, 22% share, 14B, extended training) reach comparable generation richness by different means. The minimal-computing argument prefers the first route; Qwen demonstrates the second exists.

**The unanticipated axis.** Training-data composition. Phi (textbook, instruct-tuned) against Mistral (internet), same 58% share, same architecture family, same initialisation, is a single-variable comparison on data, and at P1 the textbook prior resists Wake generalisation where the internet prior produced the deepest learning in the lineup. Share predicts the capacity to learn Wake only when the base distribution is not a clean-text prior that resists it. This was not in the 15 May frame and is now part of the claim.

**Status of the original closing line.** "This turns the smaller-model paradox from an anecdote into a measurable axis, testable across the remaining 7 models." The measurable-axis ambition held. The axis simply turned out to have more than one dimension, which is the better outcome: the prediction was specific enough to be wrong in an informative way, and being wrong in that direction is what produced the refined claim.
