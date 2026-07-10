# Wake2Vec

## Overview

Wake2vec is a comparative study of embedding injection across ten decoder-only language models trained on the text of *Finnegans Wake*. Each model receives approximately 44,000 hand-curated Wake-specific tokens added to its embedding layer and is then trained in three phases: an embedding-only phase (P1; the base transformer is frozen and the new embedding rows are trained under standard language-modelling loss, by gradient masking in most models or by a separate trainable Wake-row matrix in Qwen), a low-rank adaptation phase (P2; the embeddings are frozen and LoRA adapters are applied to the attention and MLP projections), and an auxiliary-alignment phase (P3; geometric losses are imposed on morpheme-group and word-formation-device structure). The design spans an order of magnitude in parameter count (1.1B to 14B), five architecture families, and base vocabularies from 32K to 256K, which produces Wake-vocabulary shares of roughly 17% to 58%. Training runs entirely on free-tier Colab T4 GPUs. This compute condition is treated as an object of study rather than as an incidental limitation, in order to examine whether interventions of this kind remain reproducible and interpretable under constrained resources.

The principal results are to a substantial degree negative results and controlled comparisons, and they are stated below with the corresponding caution. A guiding heuristic, drawn from Joyce's typographic figure `up:UP` (read as the formula μp → UP), assigns the three phases to the acquisition of micro-units, their routing, and their composition; it is a framing device, not a result.

**Results to date**

1. *A scale-and-data account of generation quality, including the falsification of a simpler claim.* An initial observation, that output in the Joycean register tracks Wake-vocabulary share more closely than parameter count (a 1.1B 32K-vocabulary model produces denser and more recoverable Wake-register text than a 1B 128K-vocabulary model on identical data), was falsified in its simple form by the 14B model, whose output is dense but not recoverable as meaning. The claim is accordingly revised to a three-variable account (vocabulary share, scale, training depth) in which comparable register fidelity is reachable at several points in the space and the small-vocabulary configuration is the compute-efficient one. Generation quality here is assessed qualitatively against the target text by a reader of the Wake, using the interpretive criterion of suspension (meaning held recoverable rather than abolished); there is no validated automatic measure of register fidelity, and this is a limitation, not a metric.

2. *A cross-configuration geometric null.* The injected Wake embeddings converge to near-isotropy (partition-function isotropy approximately 0.998; Mu et al. 2018) regardless of architecture, scale, or initialisation, and the auxiliary geometric losses of P3 do not move during training (the morpheme-alignment loss varies only at the tenth decimal place; embedding drift under sustained morpheme pressure is cosine 0.9998 across 600 steps). The interpretation offered is that the embedding space encodes distributional and semantic structure rather than morphological derivation, so the structure the auxiliary losses request is not present for them to amplify. A token-level analysis is consistent with this: morpheme groups whose surface form diverges from their meaning (for example *evening* relative to *even*) show near-zero compositional-direction consistency. The result is a null and is reported as one.

3. *A controlled comparison isolating the pretraining-data variable.* Holding Wake-vocabulary share (58%), initialisation (spherical), and architecture family constant, two models differing only in pretraining data (Mistral, web-trained; Phi-3.5, filtered "textbook" data) diverge in P1: the web-trained model's validation loss descends substantially and its Wake embeddings reorganise extensively, whereas the textbook-trained model's validation loss does not leave the neighbourhood of its uniform-token baseline and rises after a shallow minimum while its training loss continues to fall. The textbook-trained model memorises the training text without forming a generalising Wake representation. This is the most controlled comparison in the set.

4. *Scale-dependence of a low-rank adaptation ceiling.* The P2 validation loss of a 128K-vocabulary 3B model reaches a fixed value (5.33) within 100 steps and does not move across six consecutive evaluations (range 0.001046). The same configuration at 14B does not reproduce this ceiling. An 8B datapoint, which holds vocabulary and architecture constant against the 3B and varies only scale, has now returned its decisive evaluation and also does not reproduce it: its P2 validation descends without plateau (6.54 at step 200, 5.72 at step 300), which isolates scale as the operative variable rather than vocabulary or architecture. Because the 8B and 3B share the exact tokenizer, their values are directly comparable; a vocabulary-matched confirmation, whether the 8B descends below the 3B's 5.33, is pending at step 400. Across models with different vocabulary sizes the validation token mixture differs, so absolute values are not comparable there; the comparison in those cases is of trajectory shape (a fixed ceiling versus continued descent), not of absolute value.

5. *A methodological artifact under the compute condition.* The manual session-resume pattern required by free-tier compute resets the learning-rate scheduler at each restart, reproducing the warm-restart schedule of Loshchilov and Hutter (2017) without design. Across 39 documented restarts in the 14B P1 run, each produced a measurable learning-rate-driven descent, and the cumulative trajectory descended past the minimum of the single planned cosine schedule. This is reported as an artifact of the compute condition rather than as a designed contribution.

**Limitations** 

The corpus is a single text; no claim is made about generality. Generation quality is assessed qualitatively by an expert reader against an interpretive criterion, not by a validated metric. Validation loss is not comparable across models with different vocabularies. Free-tier compute imposes interruptions and limits run length, and several runs are incomplete. The project is ongoing; several P2 and P3 phases and three Gemma models are not yet run.

**Status** 

Three pipelines are complete (TinyLlama through P3b; Llama 3.2-1B through P3; Llama 3.2-3B through a P3 phase manually terminated at a pre-registered step). Qwen 2.5-14B and Llama 3.1-8B and Mistral 7B have completed P1 and are in P2. Phi-3.5 is in P1. The three Gemma models are not started.

---

## Models

| Model | Params | Phase | Status | Notes |
|---|---|---|---|---|
| TinyLlama 1.1B | 1.1B | complete | Done | P1: loss 8.46 to 0.079. P2: best val 0.6393. P3/P3b: auxiliary geometric losses null, L_morph already resolved by P2, L_device a structural null. Best checkpoint: P3 step 400 (val 3.4188) |
| Llama 3.2-1B | 1B | complete | Done | P3: L_morph 0.0007 (against TinyLlama's 0.0002) but did not move over training. L_device flat. The same null at a different baseline |
| Llama 3.2-3B | 3B | complete | Done (P1 to P3) | P2 validation reached a fixed value (5.33) across six consecutive evaluations (range 0.001046). P3 (strong auxiliary weighting) was run from the step-100 P2 checkpoint and terminated at a pre-registered step (600; validation 5.4653). The null is characterised at four levels: (1) at the embedding level, the Wake rows drift cosine 0.9998 across 600 steps under lambda=50 morpheme pressure, with the morpheme-alignment loss varying only at the tenth decimal place; (2) the morpheme loss does not move because the embedding directions encode meaning rather than morphological derivation (for *evening* relative to *even* the compositional direction collapses; well-sampled groups show direction consistency 0.10 to 0.24); (3) the device-clustering loss has neither within-category coherence (intra-group cosine 0.008 to 0.03) nor between-category separation (inter-centroid cosine 0.41 to 0.73); (4) the measurable cost of the strong auxiliary weights is a widening train-validation gap (0.09 to 0.85), that is, a generalisation cost rather than a language-modelling cost. Generation is coherent English with sparse invention. See `outputs/p3_llama3b_outputs.md`. |
| Llama 3.1-8B | 8B | P1 complete, P2 running | P1 done (val 11.485), P2 step 300+ | Compositional initialisation at 1.0x radius (the only model in the set using this scheme rather than spherical 1.5x). P1 reached a shallow validation minimum (11.3603 at step 1200), then rose to a plateau (11.485), exhibiting the largest train-validation divergence in the set: training loss continued to descend while validation did not improve. The embedding analysis separates two effects of initialisation. The norm structure is initialisation-dependent (the Wake region is integrated into the base manifold, Cohen's d -1.25, against -7 under spherical initialisation), whereas the isotropy is not: the Wake region reaches 0.998 isotropy despite a base-correlated compositional start, which is consistent with the isotropy being an attractor of the training dynamics rather than an artifact of the initialisation. Wake embedding drift is cosine 0.88, concentrated on truncated-English boundary tokens. Under the suspension criterion the generation is the over-deformation case (dense invention without recoverable meaning); it also surfaces code-register tokens, consistent with the injection drawing on the base model's full pretraining distribution. P2 (LoRA from the step-1200 checkpoint, SEQ_LEN 512) descends without plateau: val 7.72 at step 100, 6.54 at the decisive step 200, 5.72 at step 300, each drop larger than the 3B's fixed-ceiling shape. Holding vocabulary (128K) and architecture constant with the 3B and varying only scale, this descent isolates scale as the reason the 3B's 5.33 ceiling is not reproduced. Because the two models share the exact tokenizer, the values are directly comparable, and a vocabulary-matched confirmation, whether the 8B crosses below 5.33, is pending at step 400. The suspension question for generation is separate and decided at the generation battery, not on the loss curve. See `outputs/p1_llama8b_outputs.md` and `outputs/p1_llama8b_generation.md`. |
| Mistral 7B v0.3 | 7B | P1 complete, P2 running | P1 done (val 11.09, min 10.92 @ 1200), P2 step 400 | Sliding-window attention, 32K base vocabulary (44,553 Wake tokens, 58% share). P1 validation descended to 11.0936 at step 3000 and was still falling; the global minimum is 10.9181 at step 1200, after which validation rose (a survey-phase plateau) and then descended again. The embedding analysis records the largest Wake-embedding reorganisation in the set (drift cosine 0.485), with the most-displaced tokens being full neologisms rather than the boundary tokens displaced most in the 8B. The Wake region is the only one to fall below 0.998 isotropy (0.995), corroborated by PCA and pairwise cosine; this is consistent with greater reorganisation depositing more internal structure. A consequence under test: the P3 auxiliary losses, null in every model measured so far, may be non-null here, since pre-existing structure exists for them to amplify. Under the suspension criterion the P1 generation is the dissolution case (maximal register collision, recoverable Wake tokens present, but fragmented); it surfaces non-linguistic (emoji) and code-register tokens, consistent with the injection drawing on the base model's full pretraining distribution. P2 (LoRA from the step-1200 checkpoint) shows the steepest descent in the set: val 5.55 at step 200 and 4.77 at step 400 (best val so far), into the 4s while the other P2 runs remain in the 5s and 6s, with the per-evaluation decrement decaying after roughly step 300 in a manner consistent with the cosine anneal beginning. The train-validation gap widens (validation 0.19 above training at step 400) but this is not overfit while validation continues to descend. The suspension question is decided at generation, not at the loss curve. See `outputs/p1_mistral7b_outputs.md` and `outputs/p1_mistral7b_generation.md`. |
| Qwen 2.5-14B | 14B | P1 complete, outputs shipped, P2 running | P1 done 2026-06-09, P2 step 850 | WakeOverlay architecture, Adafactor, SEQ_LEN 128, 43,824 Wake tokens injected (~22% share). The run accrued 39 documented warm restarts via the STEP_OFFSET manual-resume pattern (each session restart resets the cosine scheduler, the structure described by Loshchilov and Hutter 2017), which is consistent with the continued validation descent past the planned cosine minimum. P1 validation reached 15.09 at step 3000 (minimum 15.05 at step 2700) and did not plateau within the schedule. Outputs shipped (`outputs/p1_qwen14b_canonical_outputs.md`, `outputs/p1_qwen14b_generation.md`): isotropy is 0.998, a further confirmation of the geometric null at 14B; the generation is dense polyglot Wake (including Chinese characters and continuous compound forms), which is not consistent with the simplest statement of the smaller-model conjecture and motivates the (share, scale, depth) formulation. P2 (LoRA from the step-2700 checkpoint) does not reproduce the low-rank adaptation ceiling observed in the 3B: validation descended from 15.05 to approximately 6.02 over 850 steps with no plateau, the per-evaluation decrement decaying as the learning rate anneals, which is consistent with the schedule rather than the architecture setting the floor. The 3B reached its fixed value (5.33) within 100 steps; this run is taken to the full 3000. The absolute validation values are not directly comparable across vocabularies. P2 uses Trainer-native resume (no warm restarts). The extender from step 3000 is post-protocol background work. |
| Phi-3.5 Mini | 3.8B | P1 running | Step 1300/3000 | Microsoft, instruct-tuned (the only publicly available variant; the deviation from the base-model convention is noted). Confirmed at runtime: 44,500 Wake tokens, 58.2% Wake share (a fourth datapoint in the TinyLlama and Mistral cohort), hidden dimension 3072 identical to Llama 3.2-3B, which makes the pair a single-variable comparison on training data. Spherical 1.5x initialisation. Validation reached a shallow minimum (12.233 at step 300) and then rose to a low plateau near 12.28 while training loss continued to descend; by step 1300 both curves are flattening (validation parked at 12.28, about a point above the uniform-token baseline; training slowing to 10.13), the memorisation saturating without a generalising Wake representation forming. The reading under test: the instruct-tuned (textbook) prior does not block local Wake memorisation (training descends) but does not reorganise into a Wake subspace that generalises (validation stays near the 11.25 random baseline). This is the training-data axis of the 2x2 comparison; its controlled partner is Mistral (same 58% share, internet-trained), which recorded the largest embedding reorganisation in the set. Cross-share validation is not directly comparable; the comparison is decided at generation. Two Phi-specific issues were handled at launch: the padding-gap boundary (tokenizer 32,011 against a padded matrix of 32,064) and the transformers tie_weights crash (Phi ships untied; tied manually). |
| Gemma 2 9B | 9B | P1 script ready | Not started | Google architecture, 256K vocab. Lowest expected Wake share (~17%). Test of paradox at the high-vocab extreme |
| Gemma 3n E2B | ~5B (2B effective) | P1 script pending | Not started | Efficient architecture: PLE and MatFormer. Tests whether Wake injection depends on always-active weights |
| Gemma 3n E4B | ~8B (4B effective) | P1 script pending | Not started | Larger efficient variant. Same architecture class as E2B for within-family comparison |

---

Stylistic adaptation of large language models typically proceeds through prompt-level conditioning, which leaves weights untouched and depends on in-context learning, or full fine-tuning, which updates all parameters and risks degrading the base model's general competencies. Wake2Vec investigates a less-studied intermediate intervention organised as a three-phase protocol, with each phase targeting a distinct level of the architecture.

Phase 1 extends the tokenizer with a curated Wake lexicon and trains only the new embedding rows under standard language modelling, leaving all transformer weights frozen via gradient masking. Phase 2 applies low-rank adaptation (LoRA, r=8) to attention and MLP projections while the P1-trained embeddings are carried forward. Phase 3 layers auxiliary geometric losses (morpheme-compositional alignment, word-formation device clustering) on top of the P2 configuration to test whether explicit structural pressure produces measurable changes to embedding geometry.

The protocol is evaluated across a controlled sweep of ten decoder-only transformers spanning five architecture families (Llama, Mistral, Qwen, Phi, Gemma), an order of magnitude in parameter count (1.1B to 14B), and an 8x range in tokenizer vocabulary (32K to 256K). The single corpus is *Finnegans Wake*; the same lexicon (and morpheme groupings for the 8b, Phi and Gemmas) are applied to every model. Training runs entirely on free Colab T4 GPUs, an explicit constraint chosen to test whether linguistically interesting interventions remain methodologically transparent and reproducible under realistic compute conditions.

Three findings have emerged so far. (1) Generation quality in the Joycean register correlates more strongly with Wake-vocab-share, defined as the fraction of the post-injection vocabulary that is newly added, than with model scale: a 1.1B parameter 32K-vocab model outperforms a 1B parameter 128K-vocab model trained on identical data which this research refers to as the smaller-model paradox. (2) Morpheme-compositional structure is implicitly encoded by the P2 language modelling objective alone; explicit geometric pressure in P3 does not move it across the three configurations tested to date (TinyLlama 1.1B, Llama 3.2-1B, Llama 3.2-3B). (3) Under strong auxiliary weighting (λ_morph=50, λ_device=2), the loss landscape penalises the geometric objective rather than satisfying it, producing language modelling regression without compensating geometric movement. Taken together, these results suggest that the structural regularities formal stylometric methods have attempted to extract from Joyce's late prose are already present in the embedding space without explicit annotation, and that the embedding layer is the most informative intervention site for stylistic adaptation of models whose pretraining did not encounter the target register.

---

## Method (Morpheme-Aware)

### Lexicon and Morphology

The morpheme dataset:

**FW morphology extraction** (`FW morphology/`): 405 unique morphemes (5,303 suffix entries, 1,406 prefix entries, 1 infix) across 6,711 total entries, extracted manually via AntConc from *Finnegans Wake*. Greedy prefix/suffix matching with a false-positive blocklist segments each Wake word into prefix|base|suffix triples. 92% segmentation success rate (6,174 / 6,710).

The extraction pipeline produces multiple JSONL formats for different training objectives:

| File | Entries | Purpose |
|---|---|---|
| `wake_embedding_groups.jsonl` | 258 groups, 6,048 words | Contrastive/embedding training (grouped by morpheme) |
| `wake_morpheme_pairs.jsonl` | 6,710 | Morpheme-word anchor pairs for contrastive loss |
| `wake_morphemes_full.jsonl` | 6,710 | Full segmentation records (prefix|base|suffix) |
| `wake_segmentation.jsonl` | 6,174 | Seq2seq morphological analysis |


### Tokenizer Augmentation

New forms are added to the tokenizer as plain tokens (bare forms and SentencePiece start-of-word variants). Mean-resizing is disabled when expanding the embedding matrix (`resize_token_embeddings(..., mean_resizing=False)`) so that custom initialisation is preserved, and input/output embeddings are tied so the new vectors participate in prediction.

### Compositional Initialisation

For new token *w* with greedy longest prefix/suffix match *(p, s)* and core *r*, set:
```
E(w) = a * E(p) + (1 - 2a) * E(r) + a * E(s) + e
```

Average embeddings of high-quality example words if a morpheme isn't single-token; e is small Gaussian noise for diversity. If *r* is unseen, fall back to a small random vector scaled to the embedding std.

### Spherical Initialisation (P1)

New Wake token embeddings are initialised on a hypersphere:
```
base_radius = std(base_embeddings) * sqrt(dim)
target_radius = 1.5 * base_radius
E(w) = random_direction / ||random_direction|| * target_radius
```

This places new tokens at a consistent distance from the origin, near the surface of the existing embedding distribution, without biasing toward any particular semantic region.

## Wake Lexicon

`wake_lexicon.txt` contains 44,989 unique tokens extracted from *Finnegans Wake*: neologisms, multilingual compounds, accented forms, and Joyce-specific coinages. These are added to whichever base tokenizer is in use. For models with larger vocabularies (Llama 3.x has 128K, Qwen 2.5 has 152K, against TinyLlama's 32K), some Wake tokens already exist in the base vocabulary and are not added.

| Model | Base vocab | Wake tokens added | Total vocab |
|---|---|---|---|
| TinyLlama 1.1B | 32,000 | ~44,500 | ~76,500 |
| Mistral 7B v0.3 | 32,768 | 44,553 | 77,321 |
| Phi-3.5 Mini 3.8B | 32,011 | 44,500 | 76,511 |
| Llama 3.2-1B | 128,256 | 44,195 | 172,451 |
| Llama 3.2-3B | 128,256 | 44,195 | 172,451 |
| Llama 3.1-8B | 128,256 | 44,195 | 172,451 |
| Qwen 2.5-14B | 152,064 | 43,824 | 196,888 |
| Gemma 2 9B | 256,000 | TBD (minimal expected) | TBD |
| Gemma 3n E2B | 256,000 | TBD (minimal expected) | TBD |
| Gemma 3n E4B | 256,000 | TBD (minimal expected) | TBD |

---

# Three-Phase Protocol

### Phase 1: Embedding-Only Training

Freeze the entire transformer. Only the embedding layer is trainable.

- New Wake tokens initialised on a hypersphere (see above)
- Input and output embeddings are tied
- A frozen LoRA r=1 adapter on q_proj is included purely for PEFT compatibility with quantized models; it contributes nothing to training

**Gradient protection strategies:**

Two approaches are used depending on the model:

1. **Gradient masking** (TinyLlama, Llama): A backward hook on the embedding weight tensor zeros out gradients for all base vocabulary rows. Only Wake token rows receive gradients. Hard guarantee against catastrophic forgetting.

```python
def mask_grad(grad):
    grad[base_rows] = 0
    return grad
wte.weight.register_hook(mask_grad)
```

2. **WakeOverlay** (Qwen): See dedicated section below.

### Phase 2: LoRA Fine-Tune

Load P1 embeddings and freeze them. Apply LoRA adapters to attention and MLP projections. The model learns to use the Wake-adapted embeddings through attention redistribution and MLP adaptation.

**LoRA targets:** q_proj, k_proj, v_proj, gate_proj, up_proj, down_proj

k_proj is included alongside q/v to allow symmetric reshaping of attention patterns. MLP layers are targeted because Wake morphology requires adaptation of token-to-meaning mappings beyond attention alone.

P2 trains on FW text only (no lexicon). LoRA adapters learn to use frozen embeddings through contextual exposure; isolated token lists provide less useful context than running prose.

### Phase 3: Morpheme-Compositional Alignment

Unfreeze embeddings with morpheme-aware regularisation. Uses decomposition data (prefixes/suffixes) to enforce compositional semantics in new token embeddings.

**Loss components:**
* L_lm: Standard language modeling loss
* L_morpheme: Direction consistency. Wake words sharing a morpheme should have parallel displacement vectors from their base forms
* L_device: Triplet contrastive. Words formed by the same word-formation device (portmanteau, blend, compound, derivation, onomatopoeia) should cluster in embedding space
* L_norm: Norm hygiene keeping Wake embeddings in distribution

**Composite loss:** `L_total = L_lm + λ_morph * L_morpheme + λ_device * L_device + λ_repulsion * L_repulsion + λ_norm * L_norm`

Scripts ready for TinyLlama (`wake2vec_phase_3_morpheme_v2.py`) and Llama (`wake2vec_llama_p3_morpheme.py`).

### Phase 3b: Geometric Refinement

A follow-up to P3 with significantly stronger geometric lambdas, testing whether the auxiliary losses can compete with L_lm when amplified.

| Parameter | P3 | P3b |
|---|---|---|
| Source | P2 step 1400 | P3 step 400 (best val) |
| LR | 5e-5 | 2e-5 |
| λ_morph | 0.1 | **50.0** (500x) |
| λ_device | 0.05 | **2.0** (40x) |
| Max steps | 3000 | 1000 |
| Early stop patience | 5 | 3 |

At P3b's lambdas, L_device contributed 12% of total loss (up from 0.3% in P3). Even at this share of the objective, the device loss did not move. See Key Findings below.

### Key Findings (TinyLlama P3 Arc)

**L_morph null result as evidence:** L_morph was pinned at 0.0002 across 2,000+ combined P3/P3b steps, never moving even under 500x lambda amplification. This is consistent with P2 (attention routing via LoRA) having implicitly learned the morphological compositional structure: the morpheme decomposition the Wake demands was already encoded through language modelling alone, before explicit geometric pressure was applied.

**L_device structural null:** The device triplet contrastive loss (clustering words by word-formation process: portmanteau, blend, compound, derivation, onomatopoeia) never left the 0.17 to 0.24 random walk range across two lambda regimes (0.05 and 2.0). The reading is that embeddings encode meaning and usage, not morphological construction method. A portmanteau of "river"+"run" should be near "river" and "run" in embedding space, not near a portmanteau of "chaos"+"cosmos". The objective is in tension with the geometry it operates on: a direction problem, not a volume problem.

**Alternative geometric objectives (future work):**
- **Character n-gram overlap**: words sharing substrings pushed closer. Natural for embeddings, captures orthographic play.
- **Phonological similarity**: words that rhyme or alliterate pushed closer. The Wake is deeply sonic.
- **Source language clustering**: Wake words blend specific languages (German, Irish, Italian, Latin). Etymology may correlate with learnable character patterns.

## Implication:

*The computational "invisibility" of Wake's figuration may be because the structure is implicit in the language patterns themselves, not requiring explicit annotation to emerge in embedding space.*

---

## Qwen WakeOverlay Architecture

Qwen 2.5-14B uses a fundamentally different embedding strategy from the Llama/TinyLlama gradient masking approach.

**Problem:** Qwen's 152K-token base vocabulary makes gradient masking on the full embedding matrix wasteful, zeroing out 152K rows per backward pass for only ~44K trainable rows.

**Solution:** A separate `nn.Embedding` layer that holds only the Wake token embeddings:

- **Base embeddings:** Frozen fp16 (152,064 x 5,120)
- **Wake overlay:** Trainable fp32 (43,824 x 5,120)
- `forward()` copies base embeddings, then scatters Wake rows on top via index replacement at `wake_start`
- Backward hook on base embeddings zeros all gradients (safety net)
- Only the overlay's parameters are passed to the optimizer

**Why Adafactor:** Adafactor stores no momentum states. This means:
- Lower VRAM overhead (~0 optimizer memory vs ~2x for AdamW)
- Lightweight resume: embedding checkpoint + step count is all that's needed (no optimizer state to restore)
- STEP_OFFSET pattern works cleanly: resume from any sentry with `trainer.train()` and offset callbacks

**VRAM budget (T4 15GB):**
- 4-bit model body: ~8GB
- fp32 Wake embeddings: ~1GB
- Adafactor states: ~0GB
- Gradients + activations: ~1-2GB
- SEQ_LEN had to be reduced to 128 (OOM at 256 on backward pass)

---

# Training Configs

### Phase 1 (Embedding-Only)

| | TinyLlama 1.1B | Llama 3.2-1B | Qwen 2.5-14B |
|---|---|---|---|
| Quantization | fp32 (whole model) | 4-bit NF4 | 4-bit NF4 |
| Embedding strategy | Gradient masking | Gradient masking | WakeOverlay |
| Optimizer | Adafactor | AdamW | Adafactor |
| LR | 5e-4 | 2e-4 | 5e-4 |
| Warmup | 5% (65 steps) | 5% (150 steps) | 5% (150 steps) |
| Batch | 1 (effective 16) | 1 (effective 16) | 1 (effective 16) |
| Seq len | 256 | 512 | 128 |
| Steps | 3,000 | 3,000 | 3,000 |
| Save every | 100 | 50 | 20 |

### Phase 2 (LoRA)

| | TinyLlama 1.1B | Llama 3.2-1B |
|---|---|---|
| Quantization | 4-bit NF4 | 4-bit NF4 |
| LoRA rank | 8 | 8 |
| LoRA alpha | 16 | 16 |
| LoRA dropout | 0.1 | 0.1 |
| Trainable params | ~5.6M | ~5.1M |
| Embeddings | Frozen (from P1) | Frozen (P1 step 1400) |
| LR | 2e-5 | 2e-5 |
| Warmup | 10% | 10% |
| Batch | 8 (effective 16) | 4 (effective 16) |
| Seq len | 256 | 512 |
| Steps | 3,000 | 3,000 |
| Weight decay | 0.01 | 0.01 |

### Phase 3 (Morpheme-Compositional Alignment)

| | TinyLlama P3 | TinyLlama P3b |
|---|---|---|
| Source | P2 step 1400 | P3 step 400 (best val) |
| LR | 5e-5 | 2e-5 |
| λ_morph | 0.1 | 50.0 |
| λ_device | 0.05 | 2.0 |
| λ_repulsion | 0.05 | 0.05 |
| λ_norm | 0.01 | 0.01 |
| Max steps | 3,000 | 1,000 |
| Early stop patience | 5 | 3 |
| Outcome | L_morph/L_device flat. Best val 3.4188 @ step 400 | L_device still flat at 40x lambda. Early stop @ step 800 |

## Data

- **Finnegans Wake corpus** (`FW_TEXT.txt`): 24,483 lines. Primary training text
- **Wake lexicon** (`wake_lexicon.txt`): 44,989 tokens. Injected into tokenizer
- **Train/val split**: 90/10, seed 42
- **Block size**: Non-overlapping chunks of seq_len tokens

Block counts vary by model (different SEQ_LEN):

| Model | SEQ_LEN | Train blocks | Val blocks |
|---|---|---|---|
| TinyLlama 1.1B P1 | 256 | 1,566 | 174 |
| Llama 3.2-1B P1 | 512 | ~800 | ~90 |
| Llama 3.2-3B P1 | 512 | 802 | 90 |
| Llama 3.1-8B P1 | 256 | ~1,600 | ~180 |
| Mistral 7B v0.3 P1 | 256 | ~1,600 | ~180 |
| Qwen 2.5-14B P1 | 128 | 3,221 | 358 |
| Phi-3.5 Mini P1 | 512 | 991 | 111 |
| Gemma 2 9B P1 | TBD | TBD | TBD |
| Gemma 3n E2B P1 | TBD | TBD | TBD |
| Gemma 3n E4B P1 | TBD | TBD | TBD |

## Embedding Analysis

Every P1 and P2 script includes a post-training analysis suite:

1. **Norm distributions** -- L2 norms of base vs new token embeddings, with Welch t-test, Mann-Whitney U, Cohen's d
2. **Isotropy** -- partition function ratio. Measures how uniformly embeddings spread across the space
3. **Embedding drift** -- cosine similarity between pre- and post-training embeddings. Base tokens should be ~1.0 (unchanged). Wake tokens should show meaningful movement
4. **Nearest neighbours** -- for sampled Wake tokens, find 5 closest base vocab tokens by cosine similarity
5. **Intrinsic dimensionality** -- PCA explained variance. How many principal components capture 90%/95% of variance in base vs new embeddings
6. **Pairwise cosine similarity** -- distributions for (base,base), (new,new), (base,new) pairs with KS test

All results saved as JSON + 6-panel matplotlib figure.

---

### Results

### TinyLlama 1.1B P1 (Complete)

**Final:** train loss 8.46 -> 0.079 over 3000 steps.

Generation from the prompt `riverrun, past Eve and Adam's,` at temp=0.9:

The model produces extended Wakean prose with structural mimicry: parenthetical asides, italicised stage directions, numbered fragments, verse-like indentation, footnote markers, rhetorical question cascades. Long clauses chained with "and", commas doing the work of periods, sudden register shifts.

**Key features across all temperatures:**
- **Lexical invention:** Portmanteaus and neologisms not in the training text
- **Character and place references:** Shem, Shaun, HCE, Matt Gregory, Mourne, Cromwell, Gracehoper -- the Wake's cast and palimpsest geography are intact
- **Spacing artifacts:** Consistent compound-fusing (`theshade`, `haveheard`, `willgive`) across all temperatures -- the main P1 limitation, from frozen attention layers that can't adapt to new tokenisation boundaries

All of this comes from embedding geometry alone. The transformer weights are entirely frozen at their chat-tuned values.

### TinyLlama 1.1B P2 (Complete)

**Best checkpoint:** step 1400, val loss 0.6393. Overfitting started around step 2000 (train/val gap widening).

The validation gap is used diagnostically rather than treated as a problem:
- P2 starting around val ~4.5 (not 7+) confirms P1 embeddings loaded correctly
- The gap that existed in P1 simply wasn't visible without a held-out set
- Different levels of overfitting serve as starting points for P3 branches

### Llama 3.2-1B P1 (Complete)

**Final:** train 61.23 / val 5.46 over 3,000 steps. Val plateaued from step 1400 onward (best val 5.36 @ step 1400).

Generation from the prompt `riverrun, past Eve and Adam's,` shows a clear temperature gradient for Wake token density:

- **temp 0.5:** Almost no Wake tokens -- clean theological prose, but the model invents etymologies using Wake logic (pseudo-definitions embedded as asides)
- **temp 0.7:** Minimal Wake intrusion (one or two compounds). Reads like a book review. Most coherent of the set
- **temp 0.9:** Wake tokens start appearing in scholarly context. Pseudo-etymology and slipping into FW's theological-sexual register
- **temp 1.0:** Exclamatory Wake eruptions. Prose fragments into preacher cadence with parenthetical neologisms
- **temp 1.2:** Full Wake mode -- dictionary-entry formatting breaks down into direct address. Maximum portmanteau density

The sweet spot for Wakean generation is **0.9--1.1**: enough temperature to surface the neologisms while maintaining syntactic context for them to land in.

**Key difference from TinyLlama P1:** Llama inserts Wake tokens as embedded neologisms within otherwise coherent Victorian/biblical prose, rather than generating sustained Wakean pastiche. The Wake tokens blend with the surrounding register rather than overwhelming it. This is likely a consequence of the larger model's stronger language priors.

### Llama 3.2-1B P2 (Complete)

Best checkpoint step 500 (val 4.04); validation rose monotonically thereafter while training loss continued to fall (the train-validation gap crossed 1.0 at step 1500 and reached 1.18 by step 1900), a standard overfit signature. The step-500 checkpoint was carried forward as the P3 source. P3 followed and produced the same null reported for TinyLlama (see the Models table).

### Qwen 2.5-14B P1 (Complete)

Canonical step 3000 reached 9 June 2026: validation 15.09 at step 3000, global minimum 15.05 at step 2700. Validation did not plateau within the schedule, descending across 39 documented warm restarts over 14 weeks of free-tier sessions. The higher absolute loss values are expected given the WakeOverlay architecture, which learns ~44K new embedding vectors from scratch against a 14B-parameter frozen transformer; the values are not comparable across vocabularies. P2 (LoRA from the step-2700 checkpoint) is running; see the Models table for current P2 numbers.

---

## Checkpoint Infrastructure

### DriveSentry

Mirrors embedding snapshots and training state to Google Drive at configurable intervals. Two key patterns:

1. **Local-first write:** `torch.save` directly to Drive FUSE can block training indefinitely on large files. Fix: save to local tmp, `shutil.copy2` to Drive, unlink local tmp.

2. **STEP_OFFSET:** When resuming with a fresh `trainer.train()` call, the Trainer's `state.global_step` restarts at 0. Callbacks add a configurable `step_offset` for globally unique file names, preventing sentry collisions across sessions.

### EmbeddingSnapshot

Saves Wake token embeddings at configurable step intervals. Lightweight (~2MB for Llama, ~340MB for Qwen), enabling post-hoc analysis of the embedding trajectory without full checkpoint overhead.

### Resume Strategies

Two resume patterns depending on model architecture:

- **Trainer-native resume** (Llama P2): `trainer.train(resume_from_checkpoint=...)` restores optimizer state, LR scheduler, and `global_step` automatically. No STEP_OFFSET needed.

- **Manual resume** (Qwen P1, Llama P1): Load embeddings from sentry, fresh `trainer.train()`. Adafactor's stateless design means no optimizer state to restore. STEP_OFFSET handles file naming. Manual override: `STEP_OFFSET = STEP_OFFSET if STEP_OFFSET > 0 else ckpt['step']` for transitioning from pre-offset sentries.

---

## Environment

**Dependencies (Colab, March 2026):**
- Python 3.12
- `torch>=2.5.1` (Colab ships 2.8.0; some scripts pin 2.5.1+cu121 for bnb compatibility)
- `transformers>=5.0`
- `accelerate>=1.2`
- `datasets>=2.21.0`
- `peft>=0.14`
- `bitsandbytes>=0.45.0`
- `triton>=3.0` (requires shim, see below)
- `umap-learn`
- `faiss-cpu`
- `wordfreq`
- `unidecode`
- `matplotlib`

**Triton shim:** `bitsandbytes>=0.45.0` imports `triton.ops.matmul_perf_model`, which was removed in `triton>=3.x` (shipped with Colab 2026.02). Every script includes a fake-module shim:

```python
import types, sys
fake_perf = types.ModuleType('triton.ops.matmul_perf_model')
fake_perf.early_config_prune = lambda *a, **k: []
fake_perf.estimate_matmul_time = lambda *a, **k: 0
sys.modules['triton.ops'] = types.ModuleType('triton.ops')
sys.modules['triton.ops.matmul_perf_model'] = fake_perf
```

**Other Colab notes:**
- `warmup_ratio` deprecated in transformers 5.x; use `warmup_steps` instead
- bfloat16 tensors cannot call `.numpy()` directly; cast `.float()` first in analysis cells
- Keep `use_cache=False` during training
- Prefer Adafactor or 8-bit Adam on T4
- Enable gradient checkpointing in Phase 2 to reduce memory

## Practical Notes

- If `load_best_model_at_end=True`, match `eval_strategy` and `save_strategy` to `"steps"`
- For OOM on T4: reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps`, shorten `SEQ_LEN` (Qwen had to go from 256 to 128), or switch Phase 2 to LoRA
- Keep random seeds fixed for comparability across phases
- Keep fp16 off on T4 for this pipeline
- DriveSentry FUSE hangs are the most common cause of training stalls; always use the local-first write pattern for saves larger than a few MB
- STEP_OFFSET only affects file naming in callbacks, not the Trainer progress bar (which always shows local step count)

---

## Monitoring

For long-running training on preemptible compute, a heartbeat monitoring notebook provides non-invasive inspection of training progress without interfering with active processes. It tracks loss trajectory from JSON logs, checkpoint inventory across local and persistent storage, embedding snapshot presence and modification times, and identifies the most recent valid checkpoint suitable for resumption.

**Storage hierarchy:**
- Local ephemeral: `/content/runs/t4_*`
- Drive persistent: `/content/drive/MyDrive/wake2vec/runs/t4_*`
- Sentry backup: `/content/drive/MyDrive/wake2vec/sentry_backups/t4_*`

## Scripts

| Script | Model | Phase | Notes |
|---|---|---|---|
| `wake2vec_llama_p1_clean.py` | Llama 3.2-1B | P1 | Gradient masking, AdamW |
| `wake2vec_llama_p2_lora.py` | Llama 3.2-1B | P2 | LoRA r=8, resume support |
| `wake2vec_llama_p3_morpheme.py` | Llama 3.2-1B | P3 | Morpheme alignment (ready) |
| `wake2vec_on_qwen_2_5_14b.py` | Qwen 2.5-14B | P1 | WakeOverlay, Adafactor |
| `wake2vec_p2_tinyllama_with_lora-2.py` | TinyLlama 1.1B | P2 | LoRA r=8 |
| `wake2vec_phase_3_morpheme_v2.py` | TinyLlama 1.1B | P3 | Morpheme alignment (ready) |

---

## Current Status (July 2026)

**Complete pipelines:**

- **TinyLlama 1.1B:** P1 + P2 + P3 + P3b. Best checkpoint P3 step 400 (val 3.4188). Generation outputs in `outputs/p3b_generation_outputs.md`. The original cross-architecture null result is established here.
- **Llama 3.2-1B:** P1 + P2 + P3. Reproduces the TinyLlama null across configurations. Best checkpoint P2 step 500 (val 4.04).
- **Llama 3.2-3B:** P1 + P2 + P3. P2 reached a fixed value (5.33) across six consecutive evaluations (range 0.001046). P3 (strong auxiliary weighting) ran from the step-100 P2 checkpoint and terminated at the pre-registered step 600 (val 5.4653). Outputs in `outputs/p3_llama3b_outputs.md`.
- **Qwen 2.5-14B P1:** Canonical step 3000 reached 9 June 2026. Minimum val 15.05 at step 2700, canonical 15.09 at step 3000. 39 documented warm restarts across 14 weeks. Embedding analysis and generation battery in `outputs/p1_qwen14b_canonical_outputs.md` and `outputs/p1_qwen14b_generation.md`. The generation result is not consistent with the simplest statement of the smaller-model conjecture and motivated the refined (share, scale, depth) formulation.

**In progress (P1):**

- **Phi-3.5 Mini (3.8B):** Step 1300/3000. Validation reached a shallow minimum (12.233 at step 300) and rose to a low plateau near 12.28; by step 1300 both curves are flattening (training slowing to 10.13), the memorisation saturating without a generalising Wake representation forming. The run continues to 3000 for protocol consistency. 32K vocab, ~58% Wake share (TinyLlama and Mistral cohort), spherical 1.5x init. Instruct-tuned (the only publicly available variant; deviation from the base-model convention noted).

**In progress (P2):**

- **Llama 3.1-8B:** LoRA from the step-1200 P1 checkpoint. The decisive evaluation has landed: val 6.54 at step 200 and 5.72 at step 300, a continued descent with no plateau, so the 3B's 5.33 ceiling is not reproduced at 8B. Because the 8B and 3B share the exact tokenizer, the values are directly comparable, and a vocabulary-matched confirmation (crossing below 5.33) is pending at step 400, delayed by free-tier interruptions.
- **Mistral 7B v0.3:** LoRA from the step-1200 P1 checkpoint (val 10.9181). The steepest descent in the set: val 5.55 at step 200 and 4.77 at step 400 (best val so far), into the 4s, with the decrement decaying after roughly step 300 as the cosine anneal begins. The widening train-validation gap is not overfit while validation continues to descend.
- **Qwen 2.5-14B:** LoRA from the step-2700 P1 checkpoint. Val approximately 6.02 by step 850, a descent of roughly 9 from the P1 minimum, with no plateau; the per-evaluation decrement is decaying as the cosine learning rate anneals. Running the full 3000 steps.

**Queued to launch:**

- **Qwen 2.5-14B extender:** launches from `sentry_step_3000.pt` with `STEP_OFFSET=3000`. Tests whether the manual-resume warm-restart mechanism continues to find descent past the canonical endpoint.
- **Gemma 2 9B:** P1 script ready. Google architecture, 256K vocab, lowest expected Wake share (~17%). Test of the smaller-model conjecture at the high-vocab extreme.
- **Gemma 3n E2B and E4B:** P1 scripts pending. Efficient-architecture variants (PLE and MatFormer).

**Key findings to date:**

- **Refined smaller-model conjecture** (the simple version is not supported): generation quality in the Joycean register appears achievable at multiple points in (Wake-vocab-share, model scale, training depth) space. A ~58% Wake-vocab-share (TinyLlama-class) is the compute-efficient configuration; 14B scale with extended training is an alternative, compute-intensive route. The original claim concerned share alone; the refined claim accounts for the three-axis trade-off the Qwen result revealed. Generation quality is assessed qualitatively, not by a validated metric.
- **Cross-architecture geometric null** (four configurations): TinyLlama P3, Llama 3.2-1B P3, Llama 3.2-3B P3, and Qwen 2.5-14B P1 all show Wake-region isotropy at 0.998. The P2 language modelling objective alone appears to encode the morpheme-compositional structure the P3 auxiliary losses target. The device triplet loss does not move, consistent with embeddings distributing on a near-uniform sphere on which there is no preferential direction for clusters to form.
- **Low-rank adaptation ceiling for 128K-vocab Llama at 3B**: val 5.33 across six consecutive P2 evaluations (range 0.001046). Under strong auxiliary pressure in P3 the model shows brief LM disruption followed by re-equilibration without crossing the fixed value. The measurable cost of strong auxiliary weights is a widening train-validation gap (0.09 in P2 to 0.74 in P3), a generalisation cost rather than an LM-fit cost. The ceiling is scale-dependent: the 8B, holding vocabulary and architecture constant with the 3B, does not reproduce it (P2 validation descends without plateau, 5.72 at step 300), and the 14B does not either; a vocabulary-matched confirmation on the 8B (crossing below 5.33) is pending.
- **Warm restarts via manual resume** (39 documented): the STEP_OFFSET manual-resume pattern, necessitated by free-tier interruptions, resets the cosine scheduler each session and reproduces the structure of an SGDR schedule (Loshchilov and Hutter 2017). Each of 39 sessions across Qwen's 14-week run shows a train-spike-then-val-descent pattern.
- **Boundary tokens**: the most-displaced Wake tokens after training (`wher`, `leas`, `hing`, `throug`, `befor`, `nig`, `hough`, `bri`, `thos`, `tch`) are truncated common English words, the tokens that sit at the Wake-to-base-English boundary. Learning concentrates there.

**Infrastructure:**

- Triton shim for bnb/triton 3.x compatibility
- DriveSentry local-first write pattern for FUSE reliability
- STEP_OFFSET pattern for session-safe callback file naming and accidental SGDR mechanism
- Resume support: Trainer-native (P2) and manual with STEP_OFFSET (P1)
- **WakeOverlay sentry-only-Wake-rows storage** (Qwen-specific): saves only the trained Wake-row matrix per sentry (448MB) rather than the full embedding matrix (~2GB). Across 150 saves over 14 weeks: 67GB vs 300GB Drive footprint. Architectural choice that made the 14-week canonical feasible on free Colab + Drive infrastructure.
- **Devlog tables as canonical training record**: when the Trainer's `trainer_state.json` was lost in the 9 June Colab cut shortly after canonical step 3000, the devlog tables (maintained by hand across all 39 sessions) supplied the complete (step, train, val) history for canonical loss curve reconstruction. The devlog-as-canonical-record practice is itself a methodological contribution for long training runs on free infrastructure.
- **Spherical 1.5x init** is the cohort default across the 32K-cohort (TinyLlama, Mistral, Phi-3.5) and the 128K-cohort (Llama family). Compositional init at 1.0x was deployed only on Llama 8B as an experimental variation; results so far do not support it as the new default. Phi-3.5 explicitly uses 1.5x to preserve cohort comparability.

---

## TinyLlama 1.1B P3b Generation Samples

## Sample 1
**Prompt**: `riverrun, past Eve and Adam's,`
**Params**: temp=0.9, top_p=0.92, top_k=50, rep=1.15

> and the crossgoing out of Humphrey's age, to thenext song of the Rocks.
> The marriage of Barandy and the Foughty Fatima, my shopsoiledgirls,under
> theillassumed names of Perkinson and Misham. The sighing of aladle man
> thatstill will be the thing that joyfakes alllife. I have read in the pink
> of Mourne's face as he dreams of her. I haveheard the windcursy of his
> doom. I haveheard theshades of histhumbs. How Thine Sidesup, Jeminy,
> isber,such and is offensive! And you, woman, are not as clean as you are
> now, my dearest Yokeoff. And it is my last endings, with that oldmilkless
> a-going for asoft time, thatsleep by the staircase, that threesheets at
> nunc or twogiving a toll, thatgrip at noon (for if she has achild, she can
> never have asleep) that has a

---

## Sample 2: Temperature sweep

**Prompt**: `riverrun, past Eve and Adam's,`
**Fixed params**: top_p=0.92, top_k=50, rep=1.15

### temp=0.5

> and thelightning Nixy, and thefields of Olaf, and the bannars up with the
> MacGhoul, the Vale's inheaven, and the harricanahurries andhotsfoots,
> zingo,zango,segur. To the cross Lally, to the cross Doolin, to the cross
> Sean Adoyer, to the cross Arth Mockmorrow Koughenough, O' Ryne!
>
> Ryeer! Hear, O' Dullmailee! The Macclefield Witch is reelingthrough the
> top of theworld! That washes yourhead out, youunderstand, and you can't
> forget the ten or so your twin had to pull, and thetwenty one that was
> sitting on the three, and all thefour, two by two, with their white arms
> below their ears, at the end of theshadow, howoft right enough, as I
> think, in the way of fun, for their castor and porridge'sgame, as they
> were going to behind a wall and the taller man

### temp=0.7

> and thelightning Nuns and the Cameen or Corpse and the
>
> [104]
> Tublin. This is not a very long way, myprodder again! Once more after this
> time, in thefuture oflife, when ourpantriarch have entered their
> ownsummers, while old Matt Gregory wouldn't be seen, there's a few more
> between you and the man in statewearholipoliwhollyisland
> peeeeeeee[132] werewhere, when he was just achild, and you werestill in
> thewhole. That's what wouldn't be too far, my very fructification,
> mylittleheart, my same uponhearts, my hair, my ears, my nose, my eyes, my
> faith, my hair, my hoops and all my ether, no matter how many, when that
> man had not beengiven thelobby, when thecorner was in his place, and I was
> too far away to askhimself fornothing.
>
> So, now, as we are in the

### temp=0.9

>  [175]
> from the day in all our things has been
>
>  UNDES.- _Nonquodsed Vestrae
>
>  'tis everynight 'tis all about._
>
> [1] I have only a staircase)
> [2] Six on the run)
> [3] Who is on thefourfirst then?
> [4] Weopen we or mates our winds with itsnation,[2] like asfour round
>     about
> [5] Cthahraet and Malthosius trying to die!
> [6] We dohear some old times (you and two verysmallthirtygirls!) Shem and
>     Shaun, out of date.
> [7] A pair of green eyes at the back of a shirt at Pickardstown.
> [8] None of thefour by the sea,through the black man at Roseleys.
> [9] Alared by the blackhearts allaround roundbrigidschool —_Truly much for
>     thee,histindier. When was it ever ever up?_

### temp=1.0

> withlustres ofpeins. Whatsound be done if
> only so they were?[1] 1065 (3618) No. I say
> it is awild'ssort to be cracked by all.[2] Now, old
> man, it's time you turned thesleep and come out of
> yoursleepingexex. Aye, and forwards I will stand
> tobring you out. And you to her, and you, and she
> to her back! So pass thetrouble on, and take your
> _Bylineal in the bedroom. Bier, stiff pumps, 1169._
>
> Waxens for wimwyer,head in love,bloodtune onsweet andfirst, thump, by,
> shirt off, shints tolife, cakestood,kiss up, buckler,head off,hear,
> _Mi-face_,such as Tuskar and Ania. _Tuesay, Pudge and Be Peposys. This
> issuch achild. Proper

### temp=1.2

> where the Nilsens made the coke of this tay for thehead part in thefour,
> where hewallowednnykins all down the rainvert redvilla. To mark her
> ownlife or pity to him. So the water and thehind that was milling in
> thefirst Shem or the Vain that had nowhad it, now love it, now
> anextinsionkissed the twins (for sheknew not thelanguage, but what sheknew
> was so long as she just caned her heirs) while thatwoman (who, then,knew
> howsuch aperiodiosit bead out of Vrittiants and Tadters, no lie!), when
> her old time-ricking time waran act was on, with apurecures for a wound to
> be due she putunder hispallyass and begin togive arms, girdles,hatsoff to
> all theirpurtybussesning lovely about
>
> [120]
> hissleep and his flesh may neverfall. And there shestill words how to
> jayne and musical

---

## Sample 3: Multi-sequence generation (×7)

**Prompt**: `riverrun, past Eve and Adam's,`
**Params**: temp=0.9, top_p=0.92, top_k=50, rep=1.15

### [1]

> theshade of ages (our times are done) with theirhistoricbringing them.
> Those were the
> Homo Vestrae, Vale, O'Neill!
> Theheart of Lifé, the year of the Cure,
> Fought for Humans' mound in Peruvian:
> _Ere_ I go to quest of Wachtman's Cromwell,
> high time as far as Tear-nan-Og,
> as far as the Oyest Brayles;

> The
> butwhere is he? Tell me, why do we be on of thatclass?
> Why not at the Rother's stomach? If she can't keep him at lughts or
> forshee Chambers? Not then? without the having to be off tobridges,through
> the Arsa, the Nodderlands Nurskery, the Manulinstight; now

### [2 extended]

> and the sigh from theopenns as by the moors made. But _you_ are doing
> your own thing. The time for e'erthose days was only atrifle and then
> allover when it took place. Thefirst thing that ever was done in the early
> days of my good man is afterwhere the grandgame was representsing
> hislowness! Whoguesse, howsuccessy do you havesuch a shorthead?
> Whatshould I have aheart? But, let usmooremooremurgessly there andhinl.
> Ahighlife of it. The tembo in his hand willgive him another. And, atweare
> if it's their hand, may the scene in his eye! From old ocean to oill or
> white, the rain has no matter when it's the use of avoice._
>
> [41]
>
> Shem was thinking fairly killing times too. He had it incurrent and they
> were all upagainst that. When he was with the MacHammuds after the fish
> went wrong (but, leave me this, it is looking aged)

### [3 extended]

> and the sigh I made in the full marpliche! by the grace of the
> Gracehoper. But my eyries be to him asbefore the ghost have itshead, with
> apoint ofhorror in hiswear, for the moment I am not up, he hascured down
> his Λ, (theloa, signing as manyarchers as there are bones in thebloo,)
> andstill reelingover theworld, like abottle of a wind, that spoiled
> fonceys andkissed us all by the bones in theirshadows.
>
> But I am asdying to Gode's will, and I will do all that he does, if he
> has it, if he does, though I am not going to saynothing about the
> gothtends oflife, for I mean to stay by the lord's side, atleast, and
> beinstead of cough andsleep and spit in a strawberryfrolic, just pass the
> teeth in olddummydeaf, as Morgents Fins me, andtouch yourtrousers about
> the rain and the

---

## Notes

### Temperature behavior

The model shows coherent temperature scaling:

- **0.5** Most structured. Anaphoric lists ("to the cross Lally, to the
  cross Doolin"), confident proper nouns ("MacGhoul", "Koughenough",
  "Dullmailee"), clear narrative momentum. Closest to readable pastiche.
- **0.7** Longer flowing passages, invention ramps up
  ("statewearholipoliwhollyisland", "pantriarch", "fructification"). 
- **0.9** Structural experimentation begins. Numbered lists, footnote
  markers, dramatic formatting. "Cthahraet and Malthosius" and
  "roundbrigidschool" feel authentically Joycean.
- **1.0** Dense, compressed. Stage directions and numbering intrude
  ("1065 (3618)"). Portmanteau density increases: "sleepingexex",
  "wimwyer", "bloodtune", "Bylineal".
- **1.2** Maximum invention. "wallowednnykins", "aperiodiosit",
  "Vrittiants and Tadters", "purtybussesning", "purecures", "pallyass".
  Grammatical structure loosens but never collapses entirely.

### Recurring features across all samples

**Lexical invention** Portmanteaus and neologisms that don't appear in the
training text: "shopsoiledgirls", "windcursy", "joyfakes", "Yokeoff",
"mooremooremurgessly", "Manulinstight", "strawberryfrolic", "olddummydeaf",
"gothtends", "fonceys", "marpliche", "harricanahurries", "purtybussesning",
"wallowednnykins". The model invents in Joyce's style.

**Character and place references** Shem, Shaun, HCE ("Humphrey"), Matt
Gregory, Mourne, O'Neill, Cromwell, "Tear-nan-Og" (Tír na nÓg),
"Nodderlands Nurskery", "MacHammuds", "Nilsens", "Gracehoper" (recovered
directly from Joyce). The Wake's cast and palimpsest geography are intact.

**Structural mimicry** Parenthetical asides, italicized stage directions,
numbered fragments, verse-like indentation, footnote markers, rhetorical
question cascades. The rhythm of Wake prose: long clauses chained with
"and", commas doing the work of periods, sudden register shifts.

**Spacing artifacts** Consistent compound-fusing ("theshade", "haveheard",
"willgive") across all temperatures. This is the main Phase 1 limitation,
from frozen attention layers that can't adapt to new tokenization
boundaries.

**to note** 

All of this comes from embedding geometry alone. The
transformer weights are entirely frozen at their chat-tuned values. The
model generates Wakean text by navigating a reshaped embedding space through
unchanged attention patterns.

---

## Qwen 2.5-14B P1 Canonical, Generation Samples

(Full outputs in `outputs/p1_qwen14b_generation.md`. The Qwen samples produce a visually dense compound-mass at all temperatures, often without word breaks, so excerpts here are trimmed for legibility.)

### Sample 1: Single generation

**Prompt**: `riverrun, past Eve and Adam's,`
**Params**: temp=0.9, top_p=0.92, top_k=50, rep=1.15

> stillleroxbelledfarkkeyholemainestsprokeduvlingebrorenhanzassognsplapdustamountturfbrowneirestgenikdonnerycrazingchristmastydepassioflowermockcomickirikirikiringpettyfib'srockelosedarklinghandmakeaprioricanarchcustoscrazingcolumnkillalmeaniummatteroffactnesshypsometersgrandestlownesskinkincaraborgroothsdeliveriedgrandesto'kayanarchpppeaseogonochicinglassgayemiddenmewseyfumederry'smaterfamiliaswednesburyboomoosternightmailnattesmaterfamiliashoarsemen阳magreesfistiknotslimpetuckpointefoxtricklesomethuartpeatrickmaterfamiliasthankyoufulrossecullinansrossecullinansghasternsuckabollytalkingtreethrostlesvowelglidelispinglirraplapvoiceyversychurchman ... [continues for ~256 tokens without breaks]

---

### Sample 2: Temperature sweep

**Prompt**: `riverrun, past Eve and Adam's,`
**Fixed params**: top_p=0.92, top_k=50, rep=1.15

#### temp=0.5

> stilllerhoosematinangeugaulesrussetspapelboypapelboyainsellahoneconscribedurnthritytwowiddarsmirthpealsdolingsduvlinhitchespoingtapopoOLopingrearrived阳langloanchoritedemoralizingterritorialslozengescontrivancemasticgaulescomepullgaulesgauleswooingcisamispalpruyparlourmendoogdoogtoper'starpinacciolfaolfakemolfa Aleatupusgypsinghouhnhymn美国placelikeconstantinealbumchubsiddlecowldzessid'stoomeydemocriticosleprousficsimilarlozengesbetweenlyohibohsittanghankypanksplapficsimilarduvlinfassedtelekinesisonviedmoysighinspirervagrantloavesgulughuruttyyoulkyoulkarseusanlescingderry's ...

#### temp=0.7

> stilllerhoosematinangeugaeilishgnawthingalmeaniumpapelboyourselfsakepapelboyjury'sdolingsduvlin Mapolfatithingpoingt nanrearrivedsquigglingyoulkcomepullwiggychrystanthemlandercolumnkillahone polyglutturalduvlinbottlerplacelikevallsallsinistrousisod'sterritorialshennarosimund's普通shaddaloavescramwellshaldmelarancitronelionses roughdusessmehrkurioscryptogampalpruyostralianyonsidesuckdooghapsnots essential ...

#### temp=0.9

> diddydidcombitschbiguiddhoosematin objectsangeuleprouspoingtpapelboyumprincipiantahonepapelboy。sallybrightdoublefirstwiggysalaamesjoltinggaulesslivenamondtidiesbeyantwiggywexterford'stonedeafshibernska knows idusessplurabelle'scorveeturetwoheadedsecurelysealingdren'sdurnachewingtarpinaccipraisegaddumptydum'lozengessousersdumptydumenvelopedducomans>convenienceschauysrearrivedhibernskacatclubsubsecmehrkuriosjacobsentawnyforbearcoptplaceliketawnystodgeduvlinneverworn阳magreesmaisonry创业hankypankscomepullhelfmoscashoundedclaudduggelduggeldeaubaleauyoulkuckdoog ...

#### temp=1.0

> censefaulterercarniumexistentialitywebbethshufferingpigstickularlypuptisedphaynixthealmostferehousefulduvlinberaddyolfatarrapoullingberaddypurefusion美国prosperousnessrayburnrayburnrayburnplacelikegobydaffyrearrivedrearrivedmistadolingsjotalphesonsublimatewouldpaygaeilishbuoyedpoingtheadwoodbeppy'swellinformed Sarfrore日子corruptiblecomeallyousgenmenputshameyubetholderpotablybetweenlyillustrationingoncontinentrearrivedrearrivedgidgadcryptoconchoidsiphonostomatarearrived ...

#### temp=1.2

> hogofemiliesturbtelekinesisanarchlepertiesnublidlanglobeaushairwireaneathwebbeth-caplozengesdonnery /*hamovsblondyanchoriterhonndashukarkithagainexchequeredelpistarpinaccianchoriteperturbingelpiscryptoconchoidsiphonostomataputshameyuelpisclerydarklingduvlinnightiestarpinacciumclausedauthorwaysmillickmaam'smasticarkmillickmaam'sgauleseatupusrearrivedpigstickularlyseventeenyearoldwaltzersllongsnipehitting总结physiog ...

---

### Sample 3: Multi-sequence generation (×3)

**Prompt**: `riverrun, past Eve and Adam's,`
**Params**: temp=0.9, top_p=0.92, top_k=50, rep=1.15, num_return_sequences=3

#### [1]

> censepostfaceumprincipianthoosemermenfrigoriquehoodendosesangeu Eastheadwoodsalaameslozengesbucknesstpapelboyahoneahonecomepullumprincipiant美国ohohcowldcramwellsimperfectionlaudszessid'scruciantidies阳zaynithshebicomepullsalaamesknockingshop roughpoingtpalmsweatdemocriticosashpitsscimmianisedwumblindeedpolldurnluttrellsandhurstrumanyoelambhaughtpipettetumtytum weremcadoopapelboydolingsknowmeyesternterritorialscomepullcomepulltaskmaster'scomepullshellaliterunesturbaryexhortingtumtytumputshameyubowandcoatinjectivejovesday ...

#### [2]

> allsalldoulseme-spondeeschilforebiddenyemcrazedledazechimbesschtinkenkotdvershenradientscupslipsforebiddenyordeffusiongenrouslylauralyeblanalambelhomoidpott美国rassiasheadwood turfbrownaliment智能bigrobbissingmaisonrysalaamessalaames。soferimpalpruyejussukkotbaredsixesuphillsracecoursefulracecoursefulseightpigsesolfa moreporkgrapciasyoelambpalpabrows ...

#### [3]

> hogohemelfarkmainestdeliverieddarklingdullcisamicagenikplaintiff'sallsortprovidentialitylillhavesthreftthoroughgoinglellymarrackspussinessdiffusingfinightthreefoilednavigableathiacarohandmakemirrylambduvlinbutteredhitchesoheremahoremarklablejotalphesongayeth'avignuetarpinaccidonnerygavelkindmourneplapbakereen'stwaddlebiguidddonnelly's ...

---

### Sample 4: Extended (×3, temp=1.1, max_new_tokens=512)

**Prompt**: `riverrun, past Eve and Adam's,`
**Params**: temp=1.1, top_p=0.92, top_k=50, rep=1.15, num_return_sequences=3, max_new_tokens=512

#### [1]

> hogodeliveriedhooseimpersonatinghibernskasummumpapelboyaringarunglispingpursuitinglorkingjoltingboldyluggedpoingtleprouswha'mmuckyregulectbreavinghennalozengespoourdurnvitiousapopocroscopedemoralizing防hankypanksquoiquoiquoiquoiquoiquoiquoiqjude'shisucowlddoogvitiouslescingmirthpealsinjectivestretchingtrisspassvaulsiesexhortingdusessrearrivedrillieshennaourselfsake roughfallener ... [extended polyglot Wake-style continues for 512 tokens]

#### [2 and 3]

(See `outputs/p1_qwen14b_generation.md` for full extended samples.)

---

### Notes

#### Temperature behaviour

The model produces sustained Wake-style output across all temperatures. Unlike TinyLlama's gradual unravelling from temp=0.5 (most readable) to temp=1.2 (maximum invention), Qwen stays inside the Wake-anchored compound-mass register at all temperatures, with the only variation being density vs diversity:

- **0.5** Maximum density. Heaviest repetition of attractor tokens (`headwood`, `loab`, `salaames`). Compound-mass continuous with no breaks. Least diverse vocabulary.
- **0.7** High density. Reduced repetition. Polyglot tokens start appearing (`美国`, `presbyoperian`, `scotobrit`).
- **0.9** Onomatopoeic mass begins (`rrrwwwkkkrrr`). Joyce-signature constructions emerge (`comeallyous`, `tarpinacci`). Multilingual mixing increases.
- **1.0** Most diverse polyglot mixing. Wider Chinese-character inclusion. New compound coinages at every position.
- **1.2** Most chaotic but still recognizably Wake-style. New invention at every position. Grammatical structure remains loose throughout (consistent across the temperature range, not a feature of high-temp degradation).

#### Recurring features across all samples

**Compound morphology at scale.** Hundreds of tokens of continuous compound-mass without word breaks. Far denser than TinyLlama's output, which shows clear word boundaries and grammatical scaffolding.

**Polyglot register.** Chinese characters (美国 = "America", 创业 = "entrepreneurship", 总结 = "summary", 阳, 趁, 望, 克, 秧, 蕾, 红, 思, 福, 防), Thai (ัน), German-flavoured constructions (schtinkenkot), Semitic anchors (salaames). The polyglot Wake signature is produced compositionally, not as a parody.

**Wake-vocabulary attractors.** Certain tokens recur across all temperatures and runs despite `rep_penalty=1.15`: `salaames`, `duvlin` (Dublin), `tarpinacci`, `schtinkenkot`, `headwood`, `loab`, `pigses`, `materfamilias`, `comeallyous`, `magrees`. These are model-specific attractor states the canonical Qwen reliably samples toward.

**Joyce-signature constructions.** Reduplication (`kirikirikiring`, `natinatinatinati`, `duggelduggel`, `shahrryardhushahrryard`), onomatopoeic mass (`rrrwwwkkkrrr`), number-as-word (`thritytwo`, `seventyseventh`, `twentynine`, `fourscore`), Wake place names (`duvlin`, `wexterford's`, `tallaght's`, `hibernia`), signature Joyce coinages (`comeallyous` from "come all ye", `darkling`, `morrowweth`).

**No bridge-token routing.** The drift-most Wake tokens (`wher`, `leas`, `hing`, `throug`, `befor`) that act as English-Wake boundary tokens do NOT appear prominently in the generation. The model stays inside the Wake-anchored semantic field rather than routing through the English boundary tokens. This is the structural difference from TinyLlama, whose output passes through English-fluent passages between Wake-style bursts.

**Spacing artifacts.** Consistent compound-fusing across all temperatures (much more extreme than TinyLlama's). This is the WakeOverlay P1 limitation: frozen attention layers can't adapt to new tokenization boundaries, and at Qwen's 14B scale the effect compounds into hundreds of tokens of continuous compound-mass.

#### Cross-model comparison

The TinyLlama (1.1B, 58% Wake share, P3b) and Qwen (14B, 22% Wake share, P1 canonical) outputs both produce Wake-style generation but through visibly different mechanisms:

| | TinyLlama 1.1B P3b | Qwen 2.5-14B P1 |
|---|--------------------|------------------|
| Wake-vocab-share | 58% | 22% |
| Training depth | ~3 weeks | 14 weeks, 39 SGDR cycles |
| Word boundaries | Visible, grammatically scaffolded | Absent; continuous compound-mass |
| Polyglot register | Latin/medieval European | Chinese/multilingual + Latin/European |
| Joyce-signature density | Periodic | Continuous |
| Wake invention | Per-passage portmanteaus | Per-token portmanteaus |
| Readability | Pastiche-readable | Density-overwhelming |
| Mechanism | Wake region integrated into English-anchored base manifold | Wake region orthogonal to multilingual base manifold; scale + depth compensate |

The two outputs are evidence for the refined finding (see `outputs/p1_qwen14b_canonical_outputs.md`): generation quality is achievable across multiple points in (Wake-vocab-share, model scale, training depth) space. TinyLlama achieves it via the compute-efficient path. Qwen achieves it via the brute-force-efficient path. Both produce sustained Wake-style output. The minimal-computing argument prefers the TinyLlama-class configuration as the methodologically appropriate choice under infrastructural constraint.

---

## Citation and Credit

- **Text**: James Joyce, *Finnegans Wake* (1939)

**Base models:**
- [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Llama 3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [Llama 3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
- [Llama 3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)
- [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B)
- [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [Gemma 2 9B](https://huggingface.co/google/gemma-2-9b)
- [Gemma 3n E2B](https://huggingface.co/google/gemma-3n-E2B)
- [Gemma 3n E4B](https://huggingface.co/google/gemma-3n-E4B)

**Conceptual inspiration:**
- Embedding surgery, retrofitting, and lightweight adapter methods (LoRA, PEFT)
- Biehle, M. (2025). *Comparative Suspension: Joyce's Dubliners and the Computational Invisibility of Figuration*. MA dissertation, UCL. [Comparative Suspension Theory provides the theoretical framework for interpreting null results in Wake embedding geometry.]
- Zhang, C. (2025). *Attention Is Not What You Need: Grassmann Flows as an Attention-Free Alternative for Sequence Modeling*. [arXiv:2512.19428](https://arxiv.org/abs/2512.19428). [Experimental Grassmann mixing framework in `grassmann_vs_attention.py`.]
- Acheli, M., et al. (2026). *Motivation is Something You Need*. [arXiv:2602.21064](https://arxiv.org/abs/2602.21064). [Dual-model training paradigm informs multi-phase pipeline design.]

**Cite**: https://github.com/mahb97/Wake2vec/blob/21469d75c26d40988ec5af8a4358d1796a36fdf0/data/CITATION.cff

