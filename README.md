# Wake2Vec

## Overview

Wake2vec is a comparative study of embedding injection across fourteen decoder-only language models trained on the text of *Finnegans Wake*. Each model receives approximately 44,000 hand-curated Wake-specific tokens added to its embedding layer and is then trained in three phases: an embedding-only phase (P1; the base transformer is frozen and the new embedding rows are trained under standard language-modelling loss, by gradient masking in most models or by a separate trainable Wake-row matrix in Qwen), a low-rank adaptation phase (P2; the embeddings are frozen and LoRA adapters are applied to the attention and MLP projections), and an auxiliary-alignment phase (P3; geometric losses are imposed on morpheme-group and word-formation-device structure). The design spans a 28-fold range in parameter count (0.5B to 14B), six architecture families, and base vocabularies from 32,000 to 262,144, which produces Wake-vocabulary shares of 14.5% to 58.2%. Four models were added on 6 November 2026 specifically to cross two axes the original ten could not separate, vocabulary at roughly constant scale and scale at constant vocabulary; the design and its pre-registered predictions are set out in `LINEUP.md`. Training runs entirely on free-tier Colab T4 GPUs. This compute condition is treated as an object of study rather than as an incidental limitation, in order to examine whether interventions of this kind remain reproducible and interpretable under constrained resources.

The principal results are to a substantial degree negative results and controlled comparisons, and they are stated below with the corresponding caution. A guiding heuristic, drawn from Joyce's typographic figure `up:UP` (read as the formula μp → UP), assigns the three phases to the acquisition of micro-units, their routing, and their composition; it is a framing device, not a result.

**Results to date**

1. *A scale-and-data account of generation quality, including the falsification of a simpler claim.* An initial observation, that output in the Joycean register tracks Wake-vocabulary share more closely than parameter count (a 1.1B 32K-vocabulary model produces denser and more recoverable Wake-register text than a 1B 128K-vocabulary model on identical data), was falsified in its simple form by the 14B model, whose output is dense but not recoverable as meaning. The claim is accordingly revised to a three-variable account (vocabulary share, scale, training depth) in which comparable register fidelity is reachable at several points in the space and the small-vocabulary configuration is the compute-efficient one. Generation quality here is assessed qualitatively against the target text by a reader of the Wake, using the interpretive criterion of suspension (meaning held recoverable rather than abolished); there is no validated automatic measure of register fidelity, and this is a limitation, not a metric.

2. *A device-clustering null that stands, and a morpheme null that has been withdrawn and replaced by a positive result.* The injected Wake embeddings converge to near-isotropy (partition-function isotropy approximately 0.998; Mu et al. 2018) regardless of architecture, scale or initialisation, and the word-formation-device objective does not move in any run to date. On the one completed run under the corrected objective it settles at 0.1960 against its own triplet margin of 0.2000, which is the value the loss takes when its categories carry no information, and the mechanism is measured on both axes: intra-category coherence of 0.032 to 0.045 against a Wake-wide baseline of 0.056, and the three largest category centroids at 0.92 to 0.93 to one another. There is neither internal structure to tighten nor separation to sharpen. The companion morpheme null reported in earlier versions of this file is **withdrawn**. That loss was an unnormalised squared-Euclidean quantity whose value scales with embedding norm, so at the weights used it was approximately 0.001% of the objective and its immobility was arithmetic rather than evidential (see Corrections). Under a corrected cosine objective the morpheme term does move. On Mistral 7B, chance-corrected direction consistency rises from +0.0458 to +0.1165 against 500-permutation nulls computed separately on each embedding matrix, the number of morpheme groups significant at z < -3 rises from 15 to 60 of 130, and rows receiving auxiliary gradients drift 1.76 times further than a size-matched control receiving language-modelling gradients only. The movement did not amplify pre-existing structure: fourteen of the fifteen largest gainers were not significant beforehand, two of them worse than chance, while the Latinate nominalising cluster that carried essentially all of the pre-training signal is unchanged. One completed run is not a cross-model result, and three of the four earlier pipelines must be re-measured before anything general is claimed.

3. *A controlled comparison isolating the pretraining-data variable.* Holding Wake-vocabulary share (58%), initialisation (spherical), and architecture family constant, two models differing only in pretraining data (Mistral, web-trained; Phi-3.5, filtered "textbook" data) diverge in P1: the web-trained model's validation loss descends substantially and its Wake embeddings reorganise extensively, whereas the textbook-trained model's validation loss does not leave the neighbourhood of its uniform-token baseline and rises after a shallow minimum while its training loss continues to fall. The textbook-trained model memorises the training text without forming a generalising Wake representation. This is the most controlled comparison in the set.

4. *Scale-dependence of a low-rank adaptation ceiling.* The P2 validation loss of a 128K-vocabulary 3B model reaches a fixed value (5.33) within 100 steps and does not move across six consecutive evaluations (range 0.001046). The same configuration at 14B does not reproduce this ceiling. An 8B datapoint, which holds vocabulary and architecture constant against the 3B and varies only scale, isolates scale as the operative variable rather than vocabulary or architecture: its P2 validation descends without plateau (6.54 at step 200, 5.72 at step 300) and has now returned the vocabulary-matched confirmation, crossing below the 3B's 5.33 at step 500 (5.282) and continuing to 5.155 at step 800, 0.175 below the fixed value the 3B could not pass. Because the 8B and 3B share the exact tokenizer, these values are directly comparable, so the crossing is a like-for-like result rather than a trajectory-shape inference. Across models with different vocabulary sizes the validation token mixture differs, so absolute values are not comparable there; the comparison in those cases is of trajectory shape (a fixed ceiling versus continued descent), not of absolute value.

5. *A methodological artifact under the compute condition.* The manual session-resume pattern required by free-tier compute resets the learning-rate scheduler at each restart, reproducing the warm-restart schedule of Loshchilov and Hutter (2017) without design. Across 39 documented restarts in the 14B P1 run, each produced a measurable learning-rate-driven descent, and the cumulative trajectory descended past the minimum of the single planned cosine schedule. This is reported as an artifact of the compute condition rather than as a designed contribution.

**Limitations** 

The corpus is a single text; no claim is made about generality. Generation quality is assessed qualitatively by an expert reader against an interpretive criterion, not by a validated metric. Validation loss is not comparable across models with different vocabularies. Free-tier compute imposes interruptions and limits run length, and several runs are incomplete. The project is ongoing; several P2 and P3 phases and three Gemma models are not yet run.

**Status** 

One pipeline is complete under the corrected morpheme objective (Mistral 7B v0.3, P1 to P3 at a full 1000-step budget, with both generation batteries and the full geometric analysis). Three further pipelines are complete but ran the defective morpheme loss (TinyLlama through P3b, Llama 3.2-1B through P3, Llama 3.2-3B through a P3 phase terminated at a pre-registered step); their device results stand and their morpheme results are withdrawn pending re-measurement. Llama 3.1-8B is in P3 and Qwen 2.5-14B has a P3 script audited and awaiting a smoke test, both on the corrected objective. Phi-3.5 has completed P2 and awaits its generation battery. The three Gemma models and the four models added in November are not started.

---

## Models

Ordered by Wake share, the fraction of the post-injection vocabulary that is newly added. Injection is near-constant across the lineup (44,553 tokens on Mistral; the spread across all models is 729 tokens), so base vocabulary alone determines share. The four models added on 6 November 2026 exist to cross two axes that the original ten could not separate: vocabulary at roughly constant scale (TinyLlama, SmolLM2, Pythia, Llama 3.2-1B, Qwen2.5-1.5B, spanning 32,000 to 151,936 inside a 1.0 to 1.7B band) and scale at constant vocabulary (Qwen2.5-0.5B, 1.5B and 14B). Full design and pre-registered predictions in `LINEUP.md`.

| Model | Params | Base vocab | Wake share | Status | Notes |
|---|---|---|---|---|---|
| TinyLlama 1.1B | 1.1B | 32,000 | 58.2% | Complete (P1 to P3b) | P1: loss 8.46 to 0.079. P2: best val 0.6393. P3/P3b: best checkpoint P3 step 400 (val 3.4188). The **device** null stands: L_device never left the 0.17 to 0.24 range across 2,000+ steps and two lambda regimes, at up to 12% of the objective. The **morpheme** null recorded here is retracted as evidence (see Corrections): L_morpheme was an unnormalised squared-Euclidean quantity too small to compete with the language-modelling term at any lambda, so its immobility was arithmetic. Generation satisfies the suspension criterion: dense invention inside sustained syntax, with genuine Wake proper nouns (`Jaun`, `Michan`, `Tory Island`), liturgical Latin, and German and Spanish material. See `outputs/tinyllama_p3b_generation.md`. |
| Phi-3.5 Mini | 3.8B | 32,064 | 58.1% | P1 complete, **P2 flooring**; P2 step 700; floor **8.347** | Instruct-tuned (the only publicly available variant). 44,500 Wake tokens, 58.2% share, hidden dimension 3072 identical to Llama 3.2-3B. P1 is the clearest data wall in the set: validation found a shallow minimum at step 300 (12.233) and then **rose** to 12.288 by step 3000, never descending, while training loss fell to a new low at the final step. P2 is qualitatively different and must not be conflated with it: validation **descended 2.887**, through the P1 wall and through the uniform-token baseline P1 never beat, before flattening at 8.347 (8.347053 at step 600, 8.347448 at step 700, unchanged to three decimals). The decrement series is -1.0, -1.423, -1.127, -0.270, -0.053, +0.0004, and the flattening occurred at approximately 97% of peak learning rate, so it is a model-side constraint rather than the schedule annealing. This is a distinct curve morphology, an asymptotic approach to a floor, differing from the 3B's capacity wall in the approach rather than the destination. Against its controlled partner Mistral (same share, initialisation and family, differing only in pretraining data), Phi floors **3.64 above** the 4.705 minimum. Both models improved substantially under routing, so the pretraining-data effect is a **higher achievable floor**, not a wall surviving into P2. The verdict is decided at generation by the nv-recall memorisation-versus-transformation test, not by the loss curve. |
| Mistral 7B v0.3 | 7B | 32,768 | 57.6% | **COMPLETE (P1 to P3)**; P3 complete at 1000 steps; best composite val 5.4713 | Sliding-window attention, 44,553 Wake tokens. **The first complete pipeline on the corrected cosine morpheme objective, and currently the only valid morpheme result in the lineup.** P1 produced the largest Wake-embedding reorganisation in the set (drift cosine 0.485) and the only Wake region below the 0.998 isotropy attractor (0.995), which is why this model was selected as the test of whether the geometry can be moved. P2 is the reference curve: steepest descent in the set, global minimum 4.705 at step 600, a turn confirmed on the strictest criterion, and a linear ascending limb. **P3 ran the full 1000-step budget and the morpheme objective moved.** L_morpheme descended 0.6700 to 0.5811, monotone across nineteen intervals, converging to an asymptote within the budget. Against 500-permutation nulls computed separately on each matrix, chance-corrected excess rose from +0.0458 to **+0.1165**, and groups significant at z < -3 rose from **15 to 60** of 130, covering 14.5% to 63.7% of supervised pairs. Measuring the endpoint against the pre-training null instead would overstate the effect by 15.5%, because the chance floor itself rises by +0.0181: a fifth of the raw movement is a non-specific alignment of the whole direction field. Rows receiving auxiliary gradients drift **1.76 times** further than a size-matched control receiving language-modelling gradients only, which identifies the auxiliary objectives rather than the language-modelling objective as the cause. The gain did not amplify pre-existing structure: the Latinate nominalising cluster that carried essentially all of the pre-training signal is unchanged, and fourteen of the fifteen largest gainers were not significant before, two of them worse than chance. **The device null stands and is now mechanistically explained**: L_device settles at 0.1960 against its own triplet margin of 0.2000, which is the value the objective takes when its categories carry no information, with intra-category coherence of 0.032 to 0.045 and the three largest category centroids at 0.92 to 0.93 to one another. The language-modelling cost is bounded: held-out performance improves alongside the geometric movement until roughly step 637, then converges 0.124 nats above the P2 minimum while the train-validation gap widens monotonically. Generation returns **no memorisation** (longest verbatim span four words, function words only) and the geometry endpoint reproduces three typographic conventions of the source where the better-perplexity checkpoint reproduces one. See `outputs/p3_mistral7b_outputs.md` and `outputs/p3_mistral7b_generation.md`. |
| SmolLM2-1.7B | 1.7B | 49,152 | 47.5% | Not started | Added 6 November 2026. Llama architecture, 49,152 vocabulary, tied embeddings. Fills the 26-to-58% Wake-share gap that currently makes the share axis two clusters rather than a trend, and is the easiest of the four additions to port, so it is scheduled first as a pipeline validation on a new model. Pretraining corpus is public, which also feeds the pretraining-data comparison. |
| Pythia-1.4B | 1.4B | 50,304 | 46.9% | Not started | Added 6 November 2026. **The only `GPTNeoXForCausalLM` in the lineup and a sixth architecture family.** 50,304 vocabulary, untied embeddings (`embed_in` and `embed_out`). The project's standard LoRA target list does not exist in it: GPT-NeoX uses a fused `query_key_value` projection and `dense_h_to_4h` / `dense_4h_to_h` in the MLP, so a rank-8 adapter on a fused projection is not the same parameterisation as three separate rank-8 adapters, and whether rank or parameter count is matched must be decided and recorded before the run. Fully public pretraining corpus with released checkpoint history. Highest porting cost of the four additions, scheduled last. |
| Llama 3.2-1B | 1B | 128,256 | 25.8% | Complete (P1 to P3) | Best P2 checkpoint step 500 (val 4.04), carried forward to P3. Generation produces sustained Wake prose inside a Victorian epistolary frame (`cupandnaggin`, `Mr Sooty-Piggy`), less dense than TinyLlama's. The comparison of its P3 L_morph (0.0007) against TinyLlama's (0.0002) is **withdrawn**: in an unnormalised squared quantity a 3.5x difference is what a 1.87x difference in embedding norm produces on its own, so it carries no information about morpheme composition. |
| Llama 3.2-3B | 3B | 128,256 | 25.8% | Complete (P1 to P3) | P2 validation reached a fixed value (5.33) within 100 steps and did not move across six consecutive evaluations (range 0.001046), a low-rank adaptation ceiling. P3 (strong auxiliary weighting) ran from the step-100 P2 checkpoint and terminated at a pre-registered step 600. Three of the four levels of the null stand: the device-clustering loss has neither within-category coherence (intra-group cosine 0.008 to 0.03) nor between-category separation (inter-centroid cosine 0.41 to 0.73); the Wake rows drift only cosine 0.9998 across 600 steps; and the cost of strong auxiliary weights is a widening train-validation gap rather than a language-modelling cost. Two claims are qualified. The morpheme-loss immobility is arithmetic rather than evidential (see Corrections). And the reported P3 validation of 5.4653 against the 5.33 wall is a **composite** figure (language modelling plus weighted auxiliaries) compared against a pure language-modelling one; no P3 script overrides `prediction_step`, so the auxiliary offset of roughly 0.43 must be subtracted before the comparison is made. Generation is the under-deformation case: nine distinct registers across the battery and no sustained Wake prose. See `outputs/p3_llama3b_outputs.md`. |
| Llama 3.1-8B | 8B | 128,256 | 25.8% | **P2 complete**, P3 pending; P2 cut at 1300; best val **5.1486** | The set's only compositional 1.0x initialisation. P1 reached a shallow minimum (11.3603 at step 1200) and generated the **over-deformation** case: dense invention without recoverable meaning, plus code-register tokens. P2 carries the **scale result**. Holding vocabulary (128K) and architecture constant against the 3B and varying only scale, on a shared exact tokenizer, it descends through the 3B's fixed 5.33 at step 500 and floors **0.181 beneath it**; its entire overfit region sits below the smaller model's floor, still 0.090 under 5.3326 at the cut. The run floored across steps 800 to 1000 (5.155, 5.1486, 5.173, a span of 0.024 at single-evaluation noise), turned on two consecutive validation rises against a monotonically falling training curve, and climbed a bounded limb averaging +0.0236 per hundred steps. The wall re-crossing is reported as an extrapolation near step 1680 rather than measured. Embedding analysis: the Wake region is **norm-integrated but direction-orthogonal**, at 0.998453 isotropy (above the base vocabulary's 0.9896) with base-new pairwise cosine 0.0009 and no nearest neighbour above 0.083. The P2 generation satisfies the suspension criterion, closes the P1 code-register leak under decode-matched sampling, and returns roughly **70% verified Wake-lexicon retrieval** (42 confirmed tokens against 16 inventions, predominantly corpus hapaxes, stable across independent samples). P3 script audited and ready. See `outputs/p2_llama8b_outputs.md` and `outputs/p2_llama8b_generation.md`. |
| Qwen 2.5-14B | 14B | 151,936 | 22.7% | P1 complete, P2 near cut; P2 step 1950; best val **5.9209** at step 1600 | WakeOverlay architecture, Adafactor, SEQ_LEN 128, 43,824 Wake tokens (~22% share). P1 accrued 39 documented warm restarts and reached 15.09 at step 3000 (minimum 15.05 at 2700). P2 descends to a global minimum of 5.9209 at step 1600, a drop of 9.13, then enters a **persistent elevated plateau**: seven consecutive evaluations reading 6.007 to 6.057, mean 6.017 and range 0.084, sitting 0.096 above the floor for 350 steps without either climbing away or returning. **No turn is confirmable, and the reason is mechanical**: the training curve reverses direction four times in seven intervals, so the divergence criterion that resolved Mistral and the 8B is structurally unavailable rather than merely unsatisfied. Four turn signals were raised and refused, each correctly. This run's principal contribution is methodological: the same protocol and reading rules yield a legible loss curve at 7B and 8B under AdamW and an illegible one at 14B under Adafactor. Note that its logged training loss is inflated 16x (see Corrections). Recommended cut at step 2000, then generation from `checkpoint-1600`. |
| Qwen2.5-1.5B | 1.5B | 151,936 | 22.7% | Not started | Added 6 November 2026. Same tokenizer, pretraining data and architecture family as the 14B, so Wake share is identical by construction. With the 0.5B and the 14B it forms the lineup's only **scale axis at constant vocabulary**, a 28x range in which any residual effect is scale alone. Ties its embeddings, unlike the 14B, which requires the manual-tie path rather than the WakeOverlay and SplitHead architecture. |
| Qwen2.5-0.5B | 0.5B | 151,936 | 22.7% | Not started | Added 6 November 2026. The smallest model in the lineup and the bottom of the scale axis. **Carries the sharpest pre-registered prediction in the project**: at 22.7% Wake share against TinyLlama's 58.2%, it should produce worse Wake-register output than TinyLlama despite being half the size, if the smaller-model paradox is about vocabulary rather than scale. If it produces better output, the vocabulary account fails. Its embedding is roughly a quarter of the parameters before injection and about a third after it, a regime no previous run has entered. It may also fail to clear P1 at all, which would bound the protocol from below and is a result rather than a failure. |
| Gemma 2 9B | 9B | 256,000 | 14.8% | P1 script ready | Google architecture, 256K vocab. Lowest expected Wake share (~17%). Test of the conjecture at the high-vocabulary extreme |
| Gemma 3n E2B | ~5B (2B effective) | 262,144 | 14.5% | P1 script pending | Efficient architecture: PLE and MatFormer. Tests whether Wake injection depends on always-active weights |
| Gemma 3n E4B | ~8B (4B effective) | 262,144 | 14.5% | P1 script pending | Larger efficient variant. Same architecture class as E2B for within-family comparison |

---

Stylistic adaptation of large language models typically proceeds through prompt-level conditioning, which leaves weights untouched and depends on in-context learning, or full fine-tuning, which updates all parameters and risks degrading the base model's general competencies. Wake2Vec investigates a less-studied intermediate intervention organised as a three-phase protocol, with each phase targeting a distinct level of the architecture.

Phase 1 extends the tokenizer with a curated Wake lexicon and trains only the new embedding rows under standard language modelling, leaving all transformer weights frozen via gradient masking. Phase 2 applies low-rank adaptation (LoRA, r=8) to attention and MLP projections while the P1-trained embeddings are carried forward. Phase 3 layers auxiliary geometric losses (morpheme-compositional alignment, word-formation device clustering) on top of the P2 configuration to test whether explicit structural pressure produces measurable changes to embedding geometry.

The protocol is evaluated across a controlled sweep of fourteen decoder-only transformers spanning six architecture families (Llama, Mistral, Qwen, Phi, Gemma, GPT-NeoX), a 28-fold range in parameter count (0.5B to 14B), and an 8-fold range in tokenizer vocabulary (32,000 to 262,144). The single corpus is *Finnegans Wake*; the same lexicon (and morpheme groupings for the 8b, Phi and Gemmas) are applied to every model. Training runs entirely on free Colab T4 GPUs, an explicit constraint chosen to test whether linguistically interesting interventions remain methodologically transparent and reproducible under realistic compute conditions.

Three findings have emerged so far, one of them in a form that has been revised twice. (1) Generation quality in the Joycean register appeared to correlate more strongly with Wake-vocab-share, defined as the fraction of the post-injection vocabulary that is newly added, than with model scale: a 1.1B parameter 32,000-vocabulary model outperformed a 1B parameter 128,256-vocabulary model trained on identical data. This is referred to as the smaller-model paradox. It was falsified in its simple form by the 14B result and has never been tested cleanly, because across the original ten models scale and vocabulary moved together. The four models added in November exist to separate them, and the prediction has accordingly been pre-registered rather than inferred: Qwen2.5-0.5B, at 22.7% share, should underperform TinyLlama at 58.2% despite being half its size, and if it does not the vocabulary account fails. (2) Morphological compositional structure is **already present** in the embedding after the language-modelling phases, and explicit geometric pressure **adds to it**. Before any auxiliary training, direction consistency across 1,413 supervised pairs sits far outside a 500-permutation null (excess +0.0458, z = -49.3), concentrated in the Latinate nominalising inventory and absent from the Germanic and inflectional inventory. Under a correctly scaled cosine objective that excess rises to +0.1165 and the number of significant groups rises from 15 to 60 of 130. The earlier claim that explicit pressure does not move this structure was an artefact of an unnormalised loss and is withdrawn (see Corrections). (3) The cost of auxiliary geometric pressure is a bounded generalisation penalty rather than a language-modelling failure. Held-out performance improves alongside the geometric movement for roughly the first 640 steps, then converges 0.124 nats above the unconstrained minimum while the train-validation gap widens monotonically and the training loss continues to fall. Taken together, these results suggest that a substantial part of the structural regularity formal stylometric methods have attempted to extract from Joyce's late prose is present in the embedding space without explicit annotation, that explicit annotation nonetheless adds a measurable increment on top of it, and that the embedding layer is the most informative intervention site for stylistic adaptation of models whose pretraining did not encounter the target register.

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

## Current Status (November 2026)

**Complete pipelines, corrected morpheme objective:**

- **Mistral 7B v0.3:** P1 + P2 + P3 at a full 1000-step budget, with both generation batteries and the complete geometric analysis. **The only valid morpheme result in the lineup to date.** P1 best val 10.9181 at step 1200, the largest Wake-embedding reorganisation in the set (drift cosine 0.485) and the only Wake region below the 0.998 isotropy attractor (0.995). P2 global minimum 4.705 at step 600, the steepest descent in the set, turned on the strictest available criterion and taken to step 1200 to document the limb. P3 completed 1000 steps: L_morpheme 0.6700 to 0.5811 monotone across nineteen intervals, chance-corrected excess +0.0458 to +0.1165 against separate 500-permutation nulls, significant groups 15 to 60 of 130, treated rows drifting 1.76 times a size-matched control. Outputs in `outputs/p3_mistral7b_outputs.md` and `outputs/p3_mistral7b_generation.md`.

**Complete pipelines, defective morpheme objective (device results stand, morpheme results withdrawn):**

- **TinyLlama 1.1B:** P1 + P2 + P3 + P3b. Best checkpoint P3 step 400 (val 3.4188). The device null is established here and is unaffected: L_device never left the 0.17 to 0.24 range across 2,000 steps and two weighting regimes.
- **Llama 3.2-1B:** P1 + P2 + P3. Best checkpoint P2 step 500 (val 4.04).
- **Llama 3.2-3B:** P1 + P2 + P3. P2 reached a fixed value (5.33) across six consecutive evaluations (range 0.001046). P3 terminated at the pre-registered step 600.

In all three the morpheme loss was an unnormalised squared-Euclidean quantity at approximately 0.001% of the objective, so those runs were effectively untreated on the morpheme axis. This makes them a **no-treatment control arm** rather than a loss: the corrected null can be retrofitted onto their saved embeddings without retraining (`CELL_NULL_retrofit.py`), which tests whether morpheme geometry moves under the language-modelling objective alone. That measurement is the next step and it determines whether a rerun is required.

**Complete phases awaiting the next step:**

- **Qwen 2.5-14B:** P1 canonical at step 3000 (39 documented warm restarts across 14 weeks). **P2 complete at step 3000**, best val 5.9209 at step 1600, unbeaten across 25 evaluations and 1,250 steps, closing at 6.309. The run produced the **epoch staircase**, six treads of near-flat validation separated by risers at epoch boundaries, with both end-of-phase predictions holding. Generation from `checkpoint-1600` is complete and returned sustained Wake register with no memorisation signal (87.1% to 92.1% type retrieval, longest matching spans three to five generic function words, 68 novel coinages). P3 uses the WakeOverlay and SplitHead architecture; the script was audited on 5 November and three defects were corrected before any run, a dtype mismatch fatal at step 0, an undefined name fatal on first resume, and a checkpoint volume of approximately 46 GiB against a 15 GB Drive. A four-step smoke test with a resume is the gate.
- **Phi-3.5 Mini:** P1 complete (the clearest data wall in the set). **P2 turned and is confirmed beyond argument**: floor 8.346755 at step 800, then twelve consecutive validation rises to 8.522264 at step 2000 with the train-validation gap widening at every interval to +0.997. Against its controlled partner Mistral, which differs only in pretraining data, Phi floors 3.64 above the 4.705 minimum. The verdict on the pretraining-data question is decided at generation by the nv-recall memorisation-versus-transformation test, not by the loss curve; the battery from `checkpoint-800` is the outstanding task.

**In progress:**

- **Llama 3.1-8B:** P2 complete, best val 5.1486 at step 900, carrying the scale result for the low-rank adaptation ceiling. P3 resumed from step 100 on 4 October on the corrected objective, with its own baseline excess of +0.0362 computed on its own supervision of 118 groups and 1,228 pairs. At step 100 L_morpheme read 0.6773 and excess +0.0417. **No values have been recorded since 4 October and the run state is unconfirmed.** Completing it is the lowest-cost route to a second valid morpheme result.

**Queued to launch:**

- **SmolLM2-1.7B, Qwen2.5-1.5B, Qwen2.5-0.5B, Pythia-1.4B:** added 6 November 2026 to cross the two axes the original ten could not separate. SmolLM2 first as a pipeline validation on a new model, Pythia last for its porting cost. See `LINEUP.md`.
- **Gemma 2 9B:** P1 script ready. 256,000 vocabulary, the lowest Wake share in the lineup at 14.8%.
- **Gemma 3n E2B and E4B:** P1 scripts pending. Efficient-architecture variants (PLE and MatFormer), testing whether Wake injection depends on always-active weights.

**Key findings to date:**

- **The smaller-model paradox is pre-registered, not established.** The simple form was falsified by the 14B result. Across the original ten, scale and vocabulary moved together, so the claim has never been tested cleanly. The November additions separate them and the prediction is now recorded in advance: Qwen2.5-0.5B at 22.7% share should underperform TinyLlama at 58.2% despite being half its size.
- **Morphological structure is present before P3 and explicit pressure adds to it.** Before any auxiliary training, direction consistency over 1,413 pairs sits far outside a 500-permutation null (excess +0.0458, z = -49.3), concentrated in the Latinate nominalising inventory (-ion, -ation, -ition, -fication, -ure, -ance, -ive, -ious) and absent or reversed in the Germanic and inflectional inventory (re-, in-, un-, -le, -ment, -en), independently of group size and of Wake-token density. Under the corrected objective the excess rises to +0.1165. The gain does not amplify what was already there: fourteen of the fifteen largest gainers were not significant beforehand. Within the nested `-ion` family, chance-corrected excess rises monotonically with the specificity of the suffix, which suggests consistency tracks the semantic uniformity of the derivation rather than the frequency of the string. **This rests on one completed run.**
- **The device null stands across every configuration and is now mechanistically explained.** The triplet loss settles at 0.1960 against its own margin of 0.2000, which is the value it takes when its categories carry no information, implying a same-category against different-category separation of 0.0040 against a within-Wake spread of 0.039. Both axes are measured: intra-category coherence 0.032 to 0.045, and the three largest category centroids at 0.92 to 0.93 to one another. The models whose Wake region is internally most coherent are the ones whose device centroids collapse hardest, so the property that makes a model the best candidate for moving the geometry is the same one that makes its device categories least separable.
- **The low-rank adaptation ceiling is scale-dependent.** Val 5.33 across six consecutive P2 evaluations at 3B (range 0.001046); the 8B, holding vocabulary and architecture constant on a shared exact tokenizer, crosses below it at step 500 and floors 0.181 beneath it; the 14B does not reproduce it either.
- **Four turn morphologies under one protocol.** Gentle-monotonic (Mistral), soft-floor-then-turn (8B), noisy-train crawler (Qwen, in which the divergence criterion is structurally unavailable rather than merely unsatisfied), and asymptotic-approach-to-floor (Phi). Turn detectability tracking training-curve smoothness is itself a result.
- **The cost of auxiliary pressure is a bounded generalisation penalty.** Held-out performance improves alongside the geometric movement for roughly 640 steps, then converges 0.124 nats above the unconstrained minimum while the train-validation gap widens monotonically and the training loss keeps falling. The same signature appears at 3B under the legacy loss, so it holds across two architectures and two loss formulations.
- **Held-out perplexity and register fidelity point in opposite directions.** Across matched generation batteries from the same run, the checkpoint 0.41 nats worse on held-out language modelling produces 54% more genuine coinage, higher lexical variety, lower corpus recall, and reproduces three typographic conventions of the source where the better-perplexity checkpoint reproduces one.
- **The model acquires the typographic layer separately from the language.** The speech dash, the closed italic span and the bracketed page numeral are reproduced in positions matching the source. The page numeral carries no linguistic content at all, being a property of the edition rather than of the text.
- **No memorisation under injection.** Across roughly 5,000 generated tokens against a 219,526-token corpus small enough to memorise outright, the longest verbatim span is four words and consists entirely of function words.
- **Boundary tokens, on both sides of the protocol.** The most-displaced Wake tokens after training are truncated common English words sitting at the Wake-to-base-English boundary. The generation-side counterpart is that roughly 68% of apparently novel word types are truncations of corpus words emitted as if they were words, a share unchanged by P3 and therefore a property of the injection rather than of the auxiliary phase.
- **Warm restarts via manual resume (39 documented).** The STEP_OFFSET pattern, necessitated by free-tier interruptions, resets the cosine scheduler each session and reproduces the structure of an SGDR schedule (Loshchilov and Hutter 2017).

**Infrastructure:**

- Triton shim for bnb and triton 3.x compatibility.
- DriveSentry local-first write pattern for FUSE reliability.
- STEP_OFFSET pattern for session-safe callback file naming, and the accidental SGDR mechanism above.
- Resume support: Trainer-native (P2 and P3) and manual with STEP_OFFSET (P1).
- **EmbeddingRestore.** Trainer-native resume restores the adapter, the optimiser moments, the scheduler and the step counter, but not the resized embedding: PEFT writes it into the adapter file and ignores it on load. Every P3 resume therefore reset the Wake geometry to the P2 matrix while the step counter continued, silently, until the analysis cells. The callback restores it on `on_train_begin` and raises rather than warns. It executed successfully for the first time on 3 November. **The general lesson is that an audit of the launch path does not audit the resume path**, and the standing practice is now a four-step smoke test with a save at step 2 and a resume from it.
- **Optimiser state at 8B.** Adafactor upcasts gradients internally and needs three full-shape fp32 buffers; `paged_adamw_8bit` kills the runtime without a traceback; and `adamw_bnb_8bit` resolves to 32-bit AdamW because transformers registers every `nn.Embedding` for 32-bit state under bnb optimisers as a stability safeguard. The working configuration is an explicitly constructed `bnb.optim.AdamW8bit` passed through `optimizers=`, verified by type rather than by the config string.
- **WakeOverlay sentry storage** (Qwen): saves only the trained Wake-row matrix per sentry rather than the full embedding matrix, which made the 14-week canonical feasible on free Colab and Drive.
- **Devlog tables as canonical training record**: when `trainer_state.json` was lost in the 9 June Colab cut, the hand-maintained devlog tables supplied the complete history for loss-curve reconstruction.
- **Spherical 1.5x initialisation** is the cohort default. Compositional initialisation at 1.0x was deployed only on the 8B as an experimental variation and results so far do not support it as a new default.
- **Known reporting artefacts in the shared analysis cells**, documented rather than silently corrected: a base-drift warning gated on cosine rather than L2 that fires on zero-norm vocabulary slots; a PCA component cap printed as though it were a measurement; a nearest-neighbour table with no signal above the noise floor of a near-orthogonal regime; and a bf16 against fp32 comparison at the exact scale of the null threshold it feeds. See Corrections.
  
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

