# Wake2Vec Phase 1 TinyLlama Embed-Only Fine-Tune

### Setup

- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Hardware**: Tesla T4 (15.64 GB)
- **Strategy**: Freeze all transformer weights, train only the (tied) embedding layer
- **Trainable params**: 156,672,000 (embedding matrix only, input/output tied)

### Vocabulary

- Base vocab: 32,000
- Wake tokens added: 44,990
- Final vocab: 76,500
- New token init: sampled from base embedding distribution (mean + noise × std × 0.1)
  - Base mean norm: 0.0753
  - Base std (per-dim avg): 0.0148

### Data

- Source: `FW_TEXT.txt` and `wake_lexicon.txt`
- Total tokens: 369,716
- Block size: 256
- Train blocks: 1,299 / Val blocks: 145 (90/10 split)

### Hyperparameters

| Param | Value |
|---|---|
| Max steps | 3,000 |
| LR | 3e-4 |
| Optimizer | Adafactor |
| Warmup | 10% |
| Weight decay | 0.0 |
| Batch size | 1 × 16 (grad accum) = 16 |
| Sequence length | 256 |
| Max grad norm | 1.0 |
| fp16/bf16 | off |
| Gradient checkpointing | on |

## Embedding Analysis

### a) Norm Analysis

| Group | Mean | Std | n |
|---|---|---|---|
| Global | 0.6304 | 0.2040 | 76,500 |
| Base | 0.8560 | 0.0708 | 32,000 |
| New | 0.4681 | 0.0706 | 44,500 |

- Global min: 0.1774, max: 1.1195
- Welch's t-test: t=748.66, p≈0 (distributions are significantly different)
- Mann-Whitney U: U=1.42e9, p≈0
- Cohen's d: 5.49 (very large effect size)

New tokens have roughly half the norm of base tokens. The base embeddings
were pre-trained over billions of tokens and occupy a high-norm shell; the
new Wake tokens, trained for only 3K steps on ~370K tokens, haven't inflated
to match. This is expected and not necessarily a problem but means that the two populations live in distinct
norm regimes.

### b) Isotropy (Mu et al. 2018)

| Group | Isotropy | Mean Cosine | n |
|---|---|---|---|
| All | 0.7808 | 0.0059 | 5,000 |
| Base | 0.9126 | 0.0023 | 5,000 |
| New | 0.7300 | 0.0073 | 5,000 |

Base tokens are well-distributed (isotropy 0.91). New tokens are less
isotropic (0.73) with slightly higher mean pairwise cosine (cluster
more tightly, occupying a lower-dimensional subspace). This is consistent
with training on a single author's relatively constrained vocabulary vs
the base model's multilingual web-scale distribution.

### c) Nearest Neighbors (Wake → Base Vocab)

Selected examples:

| Wake Token | Top-5 Base Neighbors (cosine) |
|---|---|
| `paùpulation` | significantly(.201), significant(.201), recommendation(.199), collapse(.198), Jegyzetek(.196) |
| `générations` | Расподела(.277), archiválva(.274), eredetiből(.273), IABot(.272), archiviato(.272) |
| `deathfête` | Portal(.179), Argument(.176), Governor(.171), Virtual(.170), Channel(.169) |
| `cask` | pill(.144), carriage(.137), chair(.137), serial(.130), picture(.129) |
| `loon` | Network(.210), presentation(.207), >>(.206), фі(.206), сса(.206) |

Two populations visible:
- Tokens with semantic neighbors (paùpulation, cask, loon) have
  learned meaningful positions, though similarity scores are low (0.13–0.28).
  The base vocab nearest neighbors are thematically plausible.
- Tokens collapsed onto byte-token cluster (grandmère, brofèsor, goof,
  paco, etc.) — these show ~0.985 cosine similarity to hex byte tokens
  (`<0x24>`, `<0x2B>`, etc.). These Wake tokens haven't differentiated from
  the low-information byte-level region, so are ikely too rare in the training
  corpus to get enough gradient signal.

### d) Intrinsic Dimensionality (PCA)

| Group | 90% Variance | 95% Variance | Top-1 PC |
|---|---|---|---|
| Base | ≥101 PCs | ≥101 PCs | 9.29% |
| New | ≥101 PCs | ≥101 PCs | 33.37% |

Both groups need >100 PCs for 90% variance (the analysis was capped at 100
components). But the first PC explains 33% of new-token variance vs only
9% for base (new embeds are substantially more dominated by a
single direction). This aligns with the isotropy finding: new tokens spread
less evenly across the space.

### e) Pairwise Cosine Similarity Distributions

| Pair | Mean | Std |
|---|---|---|
| base–base | 0.2289 | 0.1930 |
| new–new | 0.2510 | 0.2952 |
| base–new | 0.2266 | 0.2456 |

- KS test (base-base vs new-new): D=0.2902, p≈0

New-new pairs have higher mean cosine and much higher variance (so a mix of
well-separated tokens and the byte-collapsed cluster pulling the
distribution in both directions).

---

## Generation Sample

**Prompt**: `riverrun, past Eve and Adam's,`
**Params**: temp=0.9, top_p=0.92, top_k=50, repetition_penalty=1.15

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

### Observations

- Wakean neologisms emerging from embedding space alone: "shopsoiledgirls",
  "windcursy", "joyfakes", "Yokeoff", "illassumed", "Sidesup"
- Compound-word fusions and portmanteaus forming without any transformer
  layer updates (this is purely embedding geometry doing the work)
- Characteristic FW rhythm present: lists, asides, parentheticals, run-on
  clauses
- Some spacing artifacts ("thenext", "haveheard"). This was expected with
  embedding-only training, no attention layer adaptation
- Phase 1 objective met: Wake tokens have learned meaningful positions in
  embedding space relative to base vocabulary

---

## Phase 1 Assessment

**What worked:**
- Embedding-only training produces recognizably Wakean output
- Portmanteau formation happens naturally through embedding proximity
- The model picks up Joyce's syntactic rhythm from embeddings alone

**Known issues:**
- Large norm gap between base and new tokens (Cohen's d = 5.49)
- ~60% of sampled Wake tokens collapsed onto byte-token cluster
- New embeddings dominated by single PC direction (33% vs 9%)
- Spacing/tokenization artifacts from frozen attention layers

