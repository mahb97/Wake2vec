# wake2vec Llama 3.1-8B P1 Results

## Final Numbers

| Metric | Value |
|--------|-------|
| Model | meta-llama/Llama-3.1-8B (4-bit NF4) |
| Phase | P1 (embedding-only fine-tune, gradient masking) |
| Base vocab | 128,256 |
| Wake tokens added | 44,195 |
| Total vocab | 172,451 |
| Wake-vocab-share | ~26% |
| Steps | 3,000 (29 Colab sessions) |
| Final train | 92.6581 |
| Final val | 11.4852 |
| **Best val** | **11.3603** (step 1200, the U-curve minimum) |
| Optimizer | AdamW |
| LR | 2e-4 |
| **Embedding init** | **Compositional + spherical 1.0x radius (the lineup's only model with this strategy)** |
| SEQ_LEN | 256 |
| Effective batch | 16 (1 x 16) |
| Trainable params | ~705M (Wake embedding rows only, base frozen via gradient masking) |

### full val U-curve

| Step | Val | Step | Val |
|------|-----|------|-----|
| 200 | 12.5679 | 1600 | 11.4140 |
| 400 | 11.7212 | 1800 | 11.4377 |
| 600 | 11.4755 | 2000 | 11.4548 |
| 800 | 11.3688 | 2200 | 11.4639 |
| 1000 | 11.4052 | 2400 | 11.4753 |
| **1200** | **11.3603 (BEST)** | 2600 | 11.4834 |
| 1400 | 11.4294 | 2800 | 11.4841 |
| | | 3000 | 11.4852 |

A shallow U-curve: rapid descent 200 to 800 (12.57 to 11.37), a flat minimum band 800-1200 (11.37 to 11.36, step 800 and 1200 essentially tied), then a slow monotonic drift up to 11.485 by step 3000. The minimum is step 1200 (11.3603). The drift from 1200 to 3000 is the compositional-init overfitting signature: val rises while train keeps falling. 

## The compositional-init experiment (clean negative result)

The 8B is the **only model in the lineup** to use compositional embedding initialisation (seeding Wake tokens from morpheme-decomposition data) at 1.0x base_radius, instead of the cohort's spherical 1.5x. The hypothesis: placing Wake tokens at semantically plausible starting points should buy faster, deeper convergence than random spherical positions. The result: it did not pay off. 

### train-val divergence 
| Step | Train | Val |
|------|-------|-----|
| ~early | 207 | ~12.5 |
| 1600 | 114.42 | 11.41 |
| 2000 | 102.06 | 11.45 |
| 2400 | 93.99 | 11.48 |
| 2600 | 92.79 | 11.48 |
| 2800 | 92.38 | 11.484 |
| 3000 | 92.66 | 11.485 |

Train descended ~115 points across the run (207 to 92.66). Val moved ~1 point and then plateaued. The loss curve renders this directly: a smoothly plunging blue train line against a ruler-flat red val line. The compositional init bought aggressive train descent that did not translate to val improvement. The semantically-seeded init let the model exploit training-distribution structure that did not generalise to held-out blocks.

### Plateau 

The final four evals: 11.48, 11.48, 11.484, 11.485. Val plateaued at ~11.48 from step 2400 and held there to P1 end. Train decelerated to a crawl (93.99 to 92.66 across the last 600 steps).

### verdict

Compositional init at 1.0x radius plateaus at val 11.485 and never breaks 11.40. The spherical-1.5x cohort reaches lower at every comparable scale (the smaller Llamas into the 5-7 range; Mistral below 11.2 and still descending at the same step count). The compositional init produced the steepest train descent and the worst train-val divergence in the lineup: it helped train and hurt generalisation, the same shape as the P3-strong overfitting finding arrived at from the init direction.

**Spherical 1.5x stays the cohort default** The negative result is the documented justification for the spherical-1.5x init across every other model in the lineup, Phi included.

## Embedding analysis

the compositional init changes the norms but not the angles. The 0.998 isotropy is a training-dynamics attractor and the 8B is the only model that can prove it (the lone init outlier).

### 1. Norms 

| | Mean | Std | n |
|---|------|-----|---|
| Base (20K sample) | 0.6742 | 0.0845 | 20,000 |
| Wake (full) | **0.7532** | **0.0283** | 44,195 |
| Wake CV | 0.0376 | | |
| Welch t | t=-129.03, p=0.00 | | |
| Cohen's d | **-1.2542** | | |

The 8B's Wake norms are only 12% larger than base (ratio 1.12), and Cohen's d is **-1.25**. Compare the spherical-init models:

| Model | Init | Wake/base norm ratio | Cohen's d |
|-------|------|----------------------|-----------|
| Llama 3.2-3B | spherical 1.5x | 1.51 | -7.07 |
| Qwen 14B | spherical 1.5x | 1.62 | -3.26 |
| **Llama 3.1-8B** | **compositional 1.0x** | **1.12** | **-1.25** |

(Absolute norms differ by base model: Llama 3.1-8B's pretrained embeddings live at a smaller norm scale, base mean 0.674 vs 3B's 1.15.)

This is the compositional init's geometric fingerprint, preserved through training: the init radius determines the final norm gap. Spherical 1.5x seeds Wake on a distinct elevated shell (ratio ~1.5) and locks it there (crystalline, Wake CV ~0.003). Compositional 1.0x seeds Wake from morpheme decomposition at base norm scale and stays integrated (ratio 1.12, wider Wake CV 0.0376 because each token is seeded individually from its morphemes rather than all at one uniform radius). **The init radius is the dominant determinant of final norm structure.**

### 2. Isotropy 

| | Score | Mean cos |
|---|-------|----------|
| Base (sample) | 0.989849 | -0.0000 |
| Wake | **0.998400** | -0.0000 |

Wake isotropy **0.998** is the fifth confirmation (TinyLlama P3, Llama 1B P3, Llama 3B P3, Qwen P1, now Llama 8B P1). 

The previous four confirmations all used spherical init, so the 0.998 could have been an artifact of starting uniform-on-a-sphere. The 8B used compositional init (Wake seeded from base-correlated directions, the opposite of uniform) and training still drove it to 0.998. The isotropy is not inherited from the init; it is an attractor the training dynamics converge to. This is the strongest possible form of the geometric-null finding so the isotropy is a property of the learning rather than the starting condition.

Norm and angle decouple. The compositional init changed the magnitude structure (integrated into base) but training pulled the directional structure to the same isotropic spread every model reaches.

### 3. Drift (substantial, bridge-concentrated)

| | Cosine | L2 |
|---|--------|-----|
| Base (sample) | 0.997550 +/- 0.0494 | (gradient-masked; sub-1.0 is 4-bit quant noise) |
| Wake | **0.884761 +/- 0.0323** | 0.3488 +/- 0.0586 |

This is P1 (not a frozen-embedding phase), so drift measures how far the Wake tokens moved from their compositional-init positions across 3000 steps. Wake drifted cosine **0.88** (mean ~28 degrees of angular movement) while base stayed frozen at 0.9975 (the clean base/Wake separation that proves gradient masking worked). The substantial Wake drift is the dynamic confirmation of the norm+isotropy story where training moved the Wake region a long way in direction (compositional base-correlated seed to isotropic 0.998) while keeping it near base norm scale.

**Top-drifted Wake tokens:**

hing, tch, throug, hough, giv, elv, wher, befor, tou, hig, litt, wea, himsel, nera, leas, eath (cosines 0.58 to 0.68).

**These are the same bridge tokens that the 3B P3 morpheme analysis found most-drifted.** The overlap (wher, leas, hing, throug, befor, hough, tch in both) is striking: the English-Wake boundary tokens are the most-dynamic Wake tokens across different models, phases, and forcing functions. In 3B P3 they drifted most under lambda=50 morpheme pressure; in 8B P1 they drifted most from compositional init.

The mechanism: these are truncated common English words (where to wher, before to befor, through to throug, himself to himsel). The compositional init seeded them from their base-English components, a "semantically plausible" start. But in Wake context these fragments behave nothing like their base components (the same evening-is-not-even+ing mechanism from the 3B morpheme analysis). So the compositional init was most wrong exactly for the bridge tokens, and training had to move them the furthest. **The compositional init failed hardest precisely where it tried hardest** (on the English-derived fragments where a naive compositional average is most misleading).

### 4. Nearest neighbours (init bias fully erased)

Wake-to-base cosines: 0.05 to 0.075, statistical noise. Wake tokens shown are the French-accented multilingual layer (paùpulation, générations, fainéants, tricarême, deathfête, grandmère, brofèsor), neighbours are meaningless multilingual/code tokens.

The cosines are at the *low* end. the compositional init's base-correlation was completely erased by training. Even though Wake was seeded from base directions, the 0.88 drift to isotropy wiped out every trace, leaving the same orthogonal noise floor as the spherical models (3B 0.06-0.09, Qwen 0.05-0.09, 8B 0.05-0.075). Training annihilated the init's directional memory getting there. Only the norm scale survives; everything angular converges and forgets where it started.

### 5. Intrinsic dimensionality (PCA)

| | 90% variance | Top-1 PC |
|---|--------------|----------|
| Base (sample) | >100 PCs | 0.0062 |
| Wake | >100 PCs | 0.0053 |

Both base and Wake need more than 100 PCs for 90% variance (flat eigenspectra, high intrinsic dimensionality), consistent with the 0.998 isotropy.

A subtle init trace survives here that the partition-function isotropy could not see. Compare the Wake-vs-base top-1 PC ratio across init strategies:

| Model | Init | Base top-1 PC | Wake top-1 PC | Wake vs base |
|-------|------|---------------|---------------|--------------|
| Llama 3.2-3B | spherical 1.5x | 0.0120 | 0.0006 | Wake **more** isotropic than base (20x lower) |
| **Llama 3.1-8B** | **compositional 1.0x** | 0.0062 | 0.0053 | Wake **matches** base isotropy |

In the spherical 3B, the Wake region ended up *more* isotropic than base (it started uniform-on-a-sphere, more uniform than base). In the compositional 8B, the Wake region ends up *about as* isotropic as base (it started from base structure). The init leaves a faint trace in the eigenspectrum even though the coarse partition-function isotropy reads identical 0.998. The two isotropy measures slightly disagree, and the disagreement is exactly where the init choice lives. (Small in absolute terms; both regions are very flat. But real and consistent with the init story.)

### 6. Pairwise cosine

| | Mean | Std |
|---|------|-----|
| (base, base) | 0.0162 | 0.0208 |
| (new, new) | 0.0092 | 0.0189 |
| (base, new) | 0.0008 | 0.0175 |
| KS test (bb vs nn) | D=0.1365, p≈0 | |

The Wake region is near-orthogonal internally (new-new 0.0092) and orthogonal to base (base-new 0.0008). Notably the 8B base is itself quite isotropic (base-base 0.0162, versus 3B's clustered 0.1439), so base and Wake distributions are similar (KS D=0.1365, versus 3B's D=0.9436). This is a property of the Llama 3.1-8B base model (its pretrained embeddings are more isotropic than Llama 3.2-3B's), not the Wake injection. The Wake region is still even more isotropic than the already-isotropic base (0.0092 < 0.0162), and orthogonal to it.

### The complete compositional-init geometric story

Four findings, coherent:

1. **Norm: init-dependent.** Compositional 1.0x integrated Wake into base norm scale (Cohen's d -1.25 vs -7 for spherical). The init radius is preserved through training.
2. **Isotropy: init-independent.** Still 0.998, the fifth confirmation. A training-dynamics attractor, not an init artifact. The 8B proves it because it started base-correlated and converged to isotropic anyway.
3. **Drift: substantial and bridge-concentrated.** Wake moved 0.88 cosine from compositional init, heaviest on the English-Wake boundary tokens, the same ones 3B P3 moved most. The init failed hardest at the bridge.
4. **Init bias erased.** Nearest neighbours are pure noise; the compositional directional bias did not survive training. Only norm scale persists.

the geometric-null isotropy (0.998) is a property of the learning dynamics rather than the embedding initialisation. This could only be shown by the one model in the lineup that used a different init, and the result is that the isotropy holds regardless, while only the norm scale carries the init's memory.

## Generation battery (completed, prediction wrong again, finding better)

Source: step 3000. Full samples in `outputs/8B_Generation_Samples.md`. Prompt: `riverrun, past Eve and Adam's,`

The medium-band prediction (coherent English + sparse invention, like 3B) was wrong. The 8B produces something the lineup hadn't seen: **fragmented maximal-polyglot babel**, denser and more inventive than the 3B's coherent pastiche, that does NOT clean up at low temperature. It occupies a distinct point in the quality space: not the 3B's coherent-English-with-sprinkles, not Qwen's continuous-compound-mass, but fragmented babel.

### why TinyLlama is still the best

The 8B's invention is richer than the 3B's, but richer invention is not the same as better Wake. The density is in service of *suspended* meaning, not *abolished* meaning. 

By that criterion the three Llama-family generators fall on a line:

- **Llama 3.2-3B undershoots.** Coherent English with sparse invention. The sense is fully present but the deformation is too light. Pastiche, not Wake.
- **Llama 3.1-8B overshoots.** Gorgeous novel forms, maximal babel. Reads as Wake-*textured* but is not Wake, because nothing is recoverable.
- **TinyLlama 1.1B holds the line.** Novel forms arriving inside legible syntax that lets the reader follow the thread.

The best Wake is not produced by the most dramatic deformation; it is produced by the model that holds suspension, and the smallest model holds it best. The 32K-vocab 1.1B model maintains the lifeline that the 8B, with all its inventive firepower, severs. The constraint is not only a creative advantage, it is a *coherence* advantage, and coherence-under-deformation is the whole achievement of the Wake.

The 8B's genuine contributions are below (the code-register breakthrough and the geometric findings):

### 1: the code-register breakthrough 

The Llama 3.1-8B generation leaks **programming-language and code tokens** into the Wake output, at every temperature: `PhpStorm`, `_SERVER`, `GetMethod`, `_MESSAGE`, `ModuleName`, `preprocess`, `DSL`, `addons`, `_tbl`, `uid`, `RTC`, `SCAN`, `robotics`, `Sparse`, `conductivity`, `multimedia`. This is specific to the Llama 3.1-8B base.

*Finnegans Wake* is built by collapsing every register Joyce knew (dozens of natural languages, dialects, jargons, registers) into one dream-tongue. Joyce in 1939 had no access to code. The Llama 3.1-8B, trained heavily on code, brings programming languages into the Wake babel as a new register. The model is extending the Wake's method to its own linguistic substrate.

The mechanism is the compositional-init integrated-norm finding cashed out in generation. The Wake embeddings were seeded into the base norm manifold (Cohen's d -1.25, integrated rather than on a distinct shell), and the Llama 3.1 base manifold is code-dense. So when the model reaches into the embedding neighbourhoods around Wake tokens during generation it reaches code tokens and emits them. **The code leak is the geometric integration finding made audible.** The 3B (spherical 1.5x, Wake on a separate shell) kept the Wake region away from base structure, so it did not leak base-register code; the 8B (compositional 1.0x, Wake integrated into base) does.

**contemporary claim:** a 2026 Wake is a Wake that babels code, because code is now a major register of human language, and the Wake's whole method is to babel every register of human language. The 8B does this unprompted (no code in the FW corpus), driven purely by its own pretraining substrate. That is the breakthrough: the method generalises to registers Joyce could not have included, and the model supplies them from itself.

### 2: bridge tokens as structural grammar (cross-method convergence)

The truncated-English boundary tokens (`himsel`, `befor`, `wher`, `thos`, `hig`, `suc`, `gir`, `satis`, `bri`, `rathe`, `firs`, `leas`, `stoo`, `tch`, `noth`, `kne`, `tou`, `aroun`) saturate every sample at every temperature. These are the exact same tokens the embedding drift analysis found drifted most from the compositional init (hing, tch, throug, hough, wher, befor, leas, himsel). Two independent measurements converge:

- **Embedding space**: the bridge tokens moved most (drift cosine 0.58-0.68, the lowest in the Wake region).
- **Generation**: the bridge tokens emit most (they are practically the connective grammar of the output).

The most-dynamic region of the embedding space is the most-emitted region of the text. the English-Wake boundary is where the model's representation is most active AND where its generation most surfaces.

### 3: the texture is not temperature-dependent

The temperature sweep (0.5, 0.7, 0.9, 1.0, 1.2) shows the 8B stays fragmented all the way down. At temp 0.5 the 3B reverted to coherent English (base priors reasserting); the 8B at temp 0.5 is *still* fragmented babel (`fou from to was faithly, forethought is prin through`). The Wake-texture is not a high-temp sampling artifact; it is structural in the compositional-init embeddings. The higher P1 val (11.36 vs 3B's 6.68) and the integrated noisier embeddings make fragmentation the model's default mode, not its high-energy mode.

some of the fragmentation is genuine Wake-style density and some is the model struggling with noisier embeddings. And by the fine-line criterion (above), the fact that the texture is baked in all the way down is precisely the 8B's *failure* as a Wake generator: it cannot reach coherence even at low temperature, so it never throws the lifelines that suspension requires. The baked-in fragmentation is the over-deformation pole made structural. The precise claim is that the 8B deforms harder than the 3B, past the point of suspension into abolished sense; TinyLlama remains the only model that holds the line.

### 4: maximal polyglot density

The 8B collides more registers than any other model in the lineup: natural languages (Greek Κα, Cyrillic нин, Korean 구, Arabic إلى, Chinese 我/的/或者, Thai), code registers (1), Hiberno-English (`begob`, `bawn`, `mawn`, `auld`), and dense portmanteau-masses (`peachumpidgeonlover`, `rudderupoptimominousbedower'd`, `litteragewipealittlewakedanacheronistic` which contains "wake" itself, the model naming its own corpus inside a compound). 

The compositional init "failed" on val (plateau), failed on Wake-coherence (over-deformation), but produced the code-register breakthrough and the cleanest init-vs-dynamics geometric finding in the project. The init outlier became the *texture* outlier and the *code* outlier.

## Cross-model placement

The 8B fills the top of the Llama scale ladder:

| Model | Params | Vocab | Wake share | Init | P1 best val |
|-------|--------|-------|------------|------|-------------|
| TinyLlama 1.1B | 1.1B | 32K | 58% | spherical 1.5x | (low, U-curved) |
| Llama 3.2-1B | 1B | 128K | 26% | spherical 1.5x | 5.36 |
| Llama 3.2-3B | 3B | 128K | 26% | spherical 1.5x | 6.68 |
| **Llama 3.1-8B** | **8B** | **128K** | **26%** | **compositional 1.0x** | **11.36** |

The 8B's much higher P1 best val (11.36 vs the smaller Llamas' 5-7) is partly the compositional-init effect and partly that the larger model's val is computed differently in practice. key cross-model takeaway is the init verdict.

## Summary

The 8B P1 is complete: the longest single Llama family P1 in the project (29 sessions), the top of the Llama scale ladder, and the compositional-init control. shows that compositional init at 1.0x radius does not improve val convergence; it produces a bad train-val divergence and plateaus at 11.485 without breaking 11.40. The result validates spherical 1.5x as the cohort default and documents why every other model uses it. 
