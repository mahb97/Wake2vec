## TinyLlama P3 v2 `wake2vec_phase_3_morpheme_v2.py`

### Data upgrades

| | v1 (old) | v2 (new) |
|---|---|---|
| Morpheme data | `morpheme_data.json` (30 groups, ~300 words) | `wake_embedding_groups.jsonl` (258 groups, 6,048 words) |
| Morpheme matching | greedy prefix/suffix | pre-computed word-base pairs |
| Device data | none | `device_groups.jsonl` (6 types, ~2,100 words) |

### Loss design

```
L_total = L_lm + λ_morph * L_morpheme + λ_device * L_device + λ_repulsion * L_repulsion + λ_norm * L_norm
```

Five loss terms with one composite objective which is basically that the model sees a batch of Finnegans Wake text and does a forward pass. The standard language modelling loss (L_lm) comes from that while the four auxiliary losses are computed separately from the embedding weight matrix itself so they don't depend on the batch, but shape the geometry of the embedding space directly.

#### L_lm language modelling (weight: 1.0)

Standard next-token prediction on Finnegans Wake which acts as the base signal and keeps the model coherent as a language model. Without it, the auxiliary losses would warp the embeddings into whatever shape satisfies their objectives without caring whether the model can still generate text and L_lm anchors everything to actual language.

#### L_morpheme compositional direction consistency (weight: λ_morph = 0.1)

**Data**: `wake_embedding_groups.jsonl` has 258 morpheme groups, 6,048 word-base pairs and each entry gives a derived Wake word and its stem: "acknowledging" / "acknowledg", "Acting" / "Act", etc.

**Idea**: if a morpheme like "-ing" (which is basically a gerund for verbs) means something in embedding space, then the vector difference `embed(word) - embed(base)` should be roughly the same across all words in that group and the "morpheme direction" should be consistent.

**Computation**: for each morpheme group, compute direction vectors `d_i = embed(w_i) - embed(b_i)` for every word-base pair and also compute the group mean direction `d_mean`. Loss = weighted mean of within-group MSE: `mean(||d_i - d_mean||^2)`, weighted by group size so larger groups (like "-s" with 1,034 examples) contribute proportionally.

**What it does to the embeddings**: encourages a consistent linear structure whereby words sharing a morpheme get pushed so that the *offset from their base* is aligned, not so that the words themselves cluster. This is more linguistically principled than the v1 centroid-pull approach and it captures the compositional nature of word formation rather than just grouping.

#### L_device — stylistic device contrastive (weight: λ_device = 0.05)

**Data**: `device_groups.jsonl` has 5 groups: nonce (757 words), malapropism (690), portmanteau (465), foreign (150), pun (61).

**Idea**: Joyce's neologisms aren't random and a portmanteau ("afternunch" = afternoon + lunch) is a fundamentally different creative act from a malapropism ("acquointance" ~ acquaintance) or a nonce coinage. The embedding space should reflect this whereby words created by the same strategy should live closer together than words created by different strategies.

**Computation**: triplet margin loss, so for each step, sample 64 triplets. For each triplet: pick an anchor word, a positive (same device type), and a negative (different device type). Compute cosine similarities. Loss = `max(0, margin + cos(anchor, neg) - cos(anchor, pos))` where margin = 0.2. This pushes same-device words closer and different-device words apart.

**What it does to the embeddings**: creates macro-level clustering by creative strategy. Where L_morpheme encodes *how words are built* (subword structure), L_device encodes *what Joyce was doing* (word-level creative intent). A portmanteau and a malapropism might share the suffix "-ed" (same morpheme direction) but still sit in different regions of the space because they represent different stylistic moves.

#### L_repulsion — anti-collapse (weight: λ_repulsion = 0.05)

**Idea**: without repulsion, the auxiliary losses could collapse all Wake embeddings into a few tight clusters, but L_repulsion prevents this by penalising Wake token pairs that are too similar.

**Computation**: sample 100 random pairs of Wake tokens per step and compute cosine similarity. If any pair exceeds the threshold (0.95), penalise with `(cos - 0.95)^2`. This only activates for near-duplicate embeddings and it doesn't fight the morpheme/device clustering, it just prevents total collapse.

#### L_norm — embedding health (weight: λ_norm = 0.01)

**Idea**: Wake token embedding norms should stay in the same range as the base vocabulary, where norms that drift too far indicate the embeddings are moving to an unhealthy region of the space.

**Computation**: compute L2 norms of all Wake embeddings and then compare to the base vocab mean norm +/- one standard deviation. Only penalise norms outside that margin with `(|norm - target| - margin)^2`. This is a soft constraint and it allows some variation but prevents extreme drift.

#### How the weights balance

| Term | Weight | Role |
|------|--------|------|
| L_lm | 1.0 | Stay coherent as a language model |
| L_morpheme | 0.1 | Encode subword compositional structure |
| L_device | 0.05 | Encode word-level creative strategy |
| L_repulsion | 0.05 | Prevent embedding collapse |
| L_norm | 0.01 | Keep norms healthy |

L_lm dominates: the model is primarily a language model. Morpheme gets the highest auxiliary weight because it has the most data (6K pairs) and the most fine-grained signal. Device gets half that because its groups are broader and fewer. Repulsion and norm are regularisers, kept small so they don't fight the main objectives.

### Why fold devices into P3 instead of a separate P4

Morpheme = subword structure (how the word is built). Device = word-level creative strategy (what Joyce was doing). They're complementary signals on the same embeddings. 

### Other changes from v1

- Pip versions: torch 2.9.0, bnb 0.45.0, peft 0.18.1, accelerate 1.12.0
- Triton shim baked into Cell 1
- `mean_resizing=False` on `resize_token_embeddings`
- Full solid sentry (TinyLlama is small, Drive writes are fast)
- `device_groups.jsonl` has 6 groups: nonce (757), malapropism (690), portmanteau (465), foreign (150), pun (61), faust (1)
- P2_SOURCE path still placeholder but actually needs best P2 checkpoint

### Files needed on Colab

- `/content/FW_TEXT.txt`
- `/content/wake_embedding_groups.jsonl` (from `FW morphology/`)
- `/content/device_groups.jsonl` (from `devices/`)

---

## Data provenance

The morpheme and device datasets behind P3 aren't auto-generated but were hand-annotated over several months using two primary tools:

1. **Dictionary of Word Beginnings and Word Endings** (Penguin edition) was used as the reference lexicon for identifying productive English affixes (prefixes, suffixes, infixes) and their typical behaviour
2. **AntConc** is Lawrence Anthony's concordance tool, used to search the Finnegans Wake text for every token matching each affix pattern, then manually classifying matches into morpheme groups and word-base pairs

This produced `wake_embedding_groups.jsonl` (258 morpheme groups, 6,048 word-base pairs) and `device_groups.jsonl` (6 device types, ~2,100 words). Every entry was verified against the FW text, therefore these are philological annotations grounded in reference lexicography, not heuristic or scraped data.

### Linguistic nuance: the "-ing" example

The morpheme groups encode *morphological* identity, not *syntactic* function. Take "-ing": in "the mocking of politicals" it's a gerund (noun), in "was always jiggling" it marks progressive aspect (verb). Both share the morphological operation (base → base+-ing), so the direction loss correctly pushes their offsets to align. The *contextual* distinction — gerund vs progressive — lives in the language model's representations (L_lm), not in the embedding geometry. This is exactly the division of labour the composite loss is designed to create: morpheme loss captures how words are built, the LM loss captures how they function in context.

For the paper: most NLP work on neologisms uses rule-based or statistical affix stripping. Having manually verified morpheme decompositions for 6,048 Wakean words and cross-referenced them against a standard lexicographic reference is technically a dataset contribution in its own right. The device classifications add a second annotation layer (creative strategy) that doesn't exist in any other FW resource we're aware of.

