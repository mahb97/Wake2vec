# wake2vec — the fourteen-model design

Written 6 November 2026, on the decision to add Qwen2.5-0.5B, Qwen2.5-1.5B, Pythia-1.4B and SmolLM2-1.7B. The point of this file is to record what the lineup tests *before* the runs, so the result is a test rather than a reading.

## The lineup

Wake lexicon: 44,989 types. Injection is near-constant across models (Mistral 44,553; the spread across the whole lineup is 729 tokens), so base vocabulary alone drives Wake share. Planning figure 44,500.

| model | params | base vocab | Wake share | status |
|---|---|---|---|---|
| TinyLlama 1.1B | 1.1B | 32,000 | 58.2% | P1-P3, **defective loss** |
| Phi-3.5-mini | 3.8B | 32,064 | 58.1% | P1-P2, generation pending |
| **Mistral 7B v0.3** | 7.0B | 32,768 | 57.6% | **COMPLETE, corrected loss** |
| SmolLM2-1.7B | 1.7B | 49,152 | 47.5% | NEW |
| Pythia-1.4B | 1.4B | 50,304 | 46.9% | NEW |
| Llama 3.2-1B | 1.0B | 128,256 | 25.8% | P1-P3, **defective loss** |
| Llama 3.2-3B | 3.0B | 128,256 | 25.8% | P1-P3, **defective loss** |
| Llama 3.1-8B | 8.0B | 128,256 | 25.8% | P3 in flight, corrected loss |
| Qwen2.5-0.5B | 0.5B | 151,936 | 22.7% | NEW |
| Qwen2.5-1.5B | 1.5B | 151,936 | 22.7% | NEW |
| Qwen2.5-14B | 14.0B | 151,936 | 22.7% | P1-P2, P3 script ready |
| Gemma 2 9B | 9.0B | 256,000 | 14.8% | queued |
| Gemma 3n E2B | 2.0B | 262,144 | 14.5% | queued |
| Gemma 3n E4B | 4.0B | 262,144 | 14.5% | queued |

## What the four additions buy

The existing lineup clusters at three Wake shares (58%, 26%, 23%) with large gaps, and scale and vocabulary move together, so no clean contrast exists at the small end. The four additions produce two crossed axes.

### Axis 1: vocabulary at roughly constant scale

| model | params | base vocab | Wake share |
|---|---|---|---|
| TinyLlama 1.1B | 1.1B | 32,000 | 58.2% |
| SmolLM2-1.7B | 1.7B | 49,152 | 47.5% |
| Pythia-1.4B | 1.4B | 50,304 | 46.9% |
| Llama 3.2-1B | 1.0B | 128,256 | 25.8% |
| Qwen2.5-1.5B | 1.5B | 151,936 | 22.7% |

Five points from 32,000 to 151,936 vocabulary inside a 1.0 to 1.7B band. SmolLM2 and Pythia fill the 26-to-58 gap that currently makes the share axis two clusters rather than a trend.

### Axis 2: scale at constant vocabulary

| model | params | Wake share |
|---|---|---|
| Qwen2.5-0.5B | 0.5B | 22.7% |
| Qwen2.5-1.5B | 1.5B | 22.7% |
| Qwen2.5-14B | 14.0B | 22.7% |

A 28x scale range inside one family, with tokenizer, pretraining data and architecture held constant and Wake share identical by construction. This is the only configuration in the lineup that separates scale from vocabulary, and it is what the smaller-model paradox has never been tested against.

## Pre-registered predictions

Recorded before any of the four is run.

**P1. The paradox is about vocabulary, not scale.** Qwen2.5-0.5B should produce worse Wake-register output than TinyLlama 1.1B despite being half the size, because it sits at 22.7% share against TinyLlama's 58.2%. If it produces better output, the vocabulary account fails and the paradox is a scale effect.

**P2. Axis 1 should be monotone in Wake share.** If the account is right, generation quality on the nv-recall measures (genuine coinage rate, type-token ratio, typographic convention reproduction) should order TinyLlama > SmolLM2 ≈ Pythia > Llama 1B ≈ Qwen 1.5B. Pythia and SmolLM2 are the test: they are new points in the middle of a range that is currently two clusters, and a monotone trend through them is much harder to get by accident than a two-cluster difference.

**P3. Axis 2 should be flat or weakly increasing.** With share held constant, any remaining effect of scale is scale alone. A large effect here would mean the paradox was mis-attributed.

**P4. Qwen2.5-0.5B may not clear P1 at all.** The 3B hit a capacity wall at val 5.33; a 0.5B may fail to learn the Wake distribution. That is a result rather than a failure, and it would bound the protocol from below.

**P5. The morpheme result should replicate on the corrected loss.** Mistral is currently n=1. The specificity gradient (consistency rising with suffix specificity through `-ion`, `-ation`, `-ition`, `-fication`) is the sharpest claim and the one most worth testing elsewhere.

## Porting notes, in order of risk

**Pythia-1.4B is architecturally different and needs real work.** It is `GPTNeoXForCausalLM`, not a Llama variant. The LoRA target list used by every script in the project (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj`) does not exist in it. GPT-NeoX uses a fused `query_key_value` projection and `dense_h_to_4h` / `dense_4h_to_h` in the MLP, and the embeddings are `gpt_neox.embed_in` with a separate untied `embed_out`. The gradient-masking hook, the manual tie and the LoRA config all need porting, and the resulting LoRA is not rank-comparable to the others because a fused QKV adapter at r=8 is not the same parameterisation as three separate r=8 adapters. **Decide and record whether to match rank or match parameter count.** Verify every name against the loaded model rather than against this file.

**Tied embeddings on three of the four.** Qwen2.5-0.5B, Qwen2.5-1.5B and SmolLM2-1.7B tie input and output embeddings; Qwen2.5-14B does not, which is why its P3 needed SplitHead. Under tying, training the Wake rows changes the output head at the same time, which is a different experiment from the untied case unless handled deliberately. Mistral's manual-tie path (`model.lm_head.weight = wte.weight` with `config.tie_word_embeddings = False` so `resize_token_embeddings` does not call the buggy `tie_weights()`) is the pattern to copy. Check `config.json` for each model rather than assuming.

**The small Qwens put a large fraction of the model in the embedding.** At 0.5B the embedding is roughly a quarter of the parameters before injection and about a third after it. That is a different regime from the 14B, where it is a small tail, and it means P1 is training a much larger share of the model than in any run so far. Worth watching rather than worrying about, but it should be recorded in the devlog at launch.

**SmolLM2-1.7B is the easiest port.** Llama architecture, standard projection names, 49,152 vocab. Start here to validate the pipeline on a new model before touching Pythia.

## Suggested order

1. **The retrofit null** on TinyLlama, Llama 1B and Llama 3B (`CELL_NULL_retrofit.py`). No GPU. Establishes the no-treatment control arm and decides whether the reruns are needed.
2. **Llama 3.1-8B P3** to completion, if the run is still alive. Cheapest route to n=2 on the corrected loss.
3. **Qwen2.5-14B P3** smoke test, then the run. Script fixed 5 November; never executed.
4. **Phi-3.5 generation** from `checkpoint-800`. Decides the Phi/Mistral 2x2 and needs no P3.
5. **SmolLM2-1.7B**, full pipeline. Validates the pipeline on a new model with minimal porting.
6. **Qwen2.5-1.5B** then **0.5B**. Cheap, and completes Axis 2 with the 14B.
7. **Pythia-1.4B**. Most porting work; do it when the rest is stable.
8. **TinyLlama P3 rerun** on the corrected loss, if step 1 says it is needed.
9. **Gemmas**, as already queued.

## Standing constraint

Everything runs on free Colab T4. That is the methodology, not an obstacle: the result has to be reproducible by someone with no compute budget. No step in this plan is to be moved to local hardware.
