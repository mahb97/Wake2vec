# wake2vec Llama 3.2-3B P2 Results

## Final Numbers

| Metric | Value |
|--------|-------|
| Model | meta-llama/Llama-3.2-3B (4-bit NF4) |
| Phase | P2 (LoRA behavioural adaptation) |
| P1 source | step 300 (best val 6.68) |
| Steps | 600 (early termination, MAX_STEPS reduced from 3000) |
| Final train | 5.2350 |
| Final val | 5.3326 |
| Best val | 5.3326 (step 600, tied with all earlier evals) |
| LoRA rank | 8, alpha 16, dropout 0.1 |
| LoRA targets | q_proj, k_proj, v_proj, gate_proj, up_proj, down_proj |
| Trainable params | 10.78M (LoRA only, embeddings frozen) |
| SEQ_LEN | 512 |
| Effective batch | 16 (2 x 8) |

[P2 Loss Curve](https://github.com/mahb97/Wake2vec/blob/main/Llama/Llama%203.2-3b/Llama3.2-3b-Outputs/Llama3.2-3b_phase2_loss.png)

## The wall

P2 converged to val 5.33 at step 100 and never moved again. Six consecutive evals:

| Step | Train | Val |
|------|-------|-----|
| 100 | 5.26 | **5.333648** |
| 200 | 5.23 | **5.333549** |
| 300 | 5.25 | **5.333460** |
| 400 | 5.24 | **5.333108** |
| 500 | 5.24 | **5.333399** |
| 600 | 5.24 | **5.332602** |

Total val movement across 500 training steps: **0.001046**. 

P2 was terminated at step 600 once the wall was documented across six identical evals. Methodologically, additional steps would have produced no further information.

## implication

The embedding analysis shows that:

### Drift = zero, across both base and Wake

| | Cosine sim (mean) | Std | L2 dist (mean) |
|---|---|---|---|
| Base tokens | **1.000000** | 0.000000 | 0.000000 |
| Wake tokens | **1.000000** | 0.000000 | 0.000000 |

P2 did not move any embedding row, base or Wake, by any amount. The embeddings the model used at step 600 are bit-identical to the embeddings P1 produced at step 300. LoRA added attention routing on top, hit its capacity ceiling within 100 steps, and contributed nothing further across the remaining 500 steps.

**The wall is therefore a LoRA capacity ceiling, not an embedding capacity ceiling.** The Wake embedding configuration that emerged from P1 is fixed. P3 inherits exactly what P1 produced. The question P3 asks is whether morpheme-compositional alignment can move the embedding configuration into a region where LoRA could find further descent, or whether the embedding space is genuinely at the structural limit imposed by the 128K-vocab Llama architecture.

## Embedding Analysis

### Norms

| | Mean | Std | n |
|---|------|-----|---|
| Global | 1.2965 | 0.2730 | 172,451 |
| Base | 1.1475 | 0.1163 | 128,256 |
| Wake | **1.7291** | **0.0032** | 44,195 |
| Welch t-test | t=-1789.73, p=0.00 | | |
| Mann-Whitney U | U=0, p=0.00 | | |
| Cohen's d | **-7.07** | | |

The Wake norm std of **0.0032** is the tightest spherical distribution in the project so far. The 1B P2 settled at std 0.0152 (with Cohen's d -7.81). The 3B's higher hidden dim (3072 vs 2048) plus the same `1.5 * base_radius` spherical init policy produced a more crystalline distribution, almost 5x tighter than the 1B's. The norm gap (1.73 vs 1.15) is consistent with the 1B's gap structurally, but the Wake cloud is denser on the shell.

### Isotropy

| | Score | Mean cos | n |
|---|---|---|---|
| All tokens | 0.9461 | 0.0013 | 5,000 |
| Base tokens | 0.9826 | 0.0001 | 5,000 |
| Wake tokens | **0.9983** | -0.0000 | 5,000 |

Wake isotropy 0.998 which is identical to Llama 1B P2 (0.9979) and TinyLlama P2 ceiling. The Wake embedding region is near-perfectly uniform, no clustering, no preferred direction.

### Pairwise cosine similarity

| Pair type | Mean | Std |
|---|---|---|
| (base, base) | 0.1439 | 0.0655 |
| (new, new) | **0.0004** | 0.0187 |
| (base, new) | **0.0004** | 0.0181 |
| KS test (bb vs nn) | D=0.9434, p=0.00 | |

Wake tokens are essentially orthogonal to each other (mean cosine **0.0004**) and to base tokens (0.0004). Comparison to Llama 1B P2: 1B had (new, new) at 0.003. The 3B is almost an order of magnitude more orthogonal. The tighter Wake distribution at the norm level translates directly to tighter angular separation. Wake embeddings are sitting on a smooth shell with maximum angular spread.

### Intrinsic dimensionality (PCA, subsampled)

| | 90% variance | 95% variance | Top-1 PC |
|---|---|---|---|
| Base (n=10,000 sample) | 50 PCs (cap) | 50 PCs (cap) | 1.19% |
| Wake (n=44,195 full) | 50 PCs (cap) | 50 PCs (cap) | **0.05%** |

Both hit the n_components cap. Top-1 PC for Wake is 0.05% which is *flatter* than Llama 1B's Wake top-1 PC (0.21%). The Wake distribution has no dominant direction at all. This is the consistent signature of spherical init + LoRA freeze: the Wake region is a uniform sphere with no semantic structure imposed by training.

### Nearest neighbours

All Wake → base cosines under 0.09:

| Wake token | Top neighbour | Cos |
|---|---|---|
| 'paùpulation' | 'ĠVest' | 0.064 |
| 'générations' | 'ud' | 0.077 |
| 'introdùce' | '19' | 0.072 |
| 'tricarême' | 'Ġberries' | 0.084 |
| 'cask' | 'Ġgrate' | 0.088 |
| 'grandmère' | "'''" | 0.071 |

just noise. The Wake embeddings have no meaningful semantic neighbours in the base vocab. Identical pattern to 1B P2. This is what near-perfect isotropy looks like at the per-token level.

## Cross-model P2 comparison

| | TinyLlama P2 | Llama 1B P2 | Llama 3B P2 |
|---|---|---|---|
| Hidden dim | 2,048 | 2,048 | **3,072** |
| Base vocab | 32K | 128K | 128K |
| Wake tokens | ~44,500 | 44,195 | 44,195 |
| LoRA params | ~5.6M | ~5.1M | **10.78M** |
| Steps run | 3,000 | 3,000 | **600 (terminated)** |
| Best val | 0.6393 | 4.04 | **5.33** |
| Best val step | 1400 | 500 | **100** |
| Steps to plateau | gradual to step 2000 | descent to 500, then climb | **immediate (step 100)** |
| Train at end | 0.47 | 3.03 | **5.24** |
| Val movement (best to end) | 0.04 | 0.35 | **0.0004** |
| Wake norm mean | ~0.47 | 1.504 | **1.729** |
| Wake norm std | — | 0.0152 | **0.0032** |
| Cohen's d | — | -7.81 | **-7.07** |
| Wake isotropy | — | 0.9979 | **0.9983** |
| (new, new) cosine | 0.251 | 0.003 | **0.0004** |
| Drift (Wake) | 1.000 | 1.000 | 1.000 |

Three patterns emerge clearly across the 3B comparison:

1. **The plateau is instantaneous at this scale.** TinyLlama's val crept down for 2000 steps. Llama 1B descended through 500 steps before plateauing. The 3B converged to its ceiling within 100 steps and held it to four decimal places across 500 more.

2. **The Wake distribution gets more crystalline as hidden dim grows.** TinyLlama's Wake region was actively learned (mean cos 0.251 between Wake tokens, real clustering). Llama 1B's was nearly orthogonal (0.003). Llama 3B's is almost perfectly orthogonal (0.0004). Larger hidden dim + same spherical init policy = denser, smoother sphere.

3. **Across all three, P2 froze embeddings perfectly.** Drift cosine is exactly 1.0 in every case. Whatever generation differences exist between TinyLlama (Wakean pastiche), Llama 1B (Victorian prose with embedded neologisms), and Llama 3B (essentially zero Wake content from P1) trace back to *how the embeddings were learned in P1*, not to anything P2's LoRA did to them.

## The refined smaller-model paradox at the LM loss level

The smaller-model paradox previously described generation quality (TinyLlama > Llama 1B > Llama 3B for Wake authenticity). The 3B P2 results extend it to the LM loss surface itself:

**A 128K-vocab Llama at 3B parameters cannot break LM loss 5.33 via LoRA, regardless of training duration.**

the runs empirical claim is that the val movement across 500 training steps (0.00104) is below floating-point noise tolerance for any reasonable LM loss interpretation. The ceiling is structural to the architecture, not contingent on training compute.

using failure as the analytical lens, the 5.33 ceiling makes legible the limit of scale-as-solution. Adding parameters did not produce a method appropriate to the material. The constraint is the diagnostic.

## Generation Outputs

**Prompt:** `riverrun, past Eve and Adam's,` (all runs)
**Fixed params:** top_p=0.92, top_k=50, rep=1.15

### Run 1: single generation at temp=0.9

> the garden of Eden; a place where men might have their conversation with God. It is commonly thought to be in Armenia. But perhaps it was only an ideamoreafterft in this part of the world; forclerricalsfsmocknameff are yet found in other parts of the earth. When Solomon reigned over all Palestine, he built this temple at Jerusalem: but after his death, and during thecategoric reigns of various kings, no care was taken of it till that of Hezekiah about 715 years before Christ; who removed the altar from Gibeon hither, and built the whole building anew; and made the foundation thereof great stones, five cubits high; which he fupported by large beams of firrewood (Exod. xvi.) and then added these words: This temple shall be unto thee as a place to keep continually four kinds of fecond offerings; two sheep-lappedves yearly, every day throughout the year: six apes and seven rams, which were without blemish, upon the sabbath day: besides this they offered one young bullock, and one ram; both they offered without blame, before the Lord; so did they also with thetomtompions, and goat. Hezehiab began

Wake-adjacent fragments: **"ideamoreafterft"**, **"forclerricalsfsmocknameff"**, **"firrewood"**, **"sheep-lappedves"**, **"thetomtompions"**, **"Hezehiab"**, and the long-s artefacts **"fupported"** and **"fecond"** (Victorian printing convention where 's' was rendered 'ſ' and reads as 'f'). Register: Victorian biblical reference book with Wake-coded orthographic disruption surfacing through.

### Run 2: temperature sweep

#### temp=0.5

> I've been to the same place twice. I'm not sure if it was a different location or just another one of those "same" places that have two locations.

zero Wake.

#### temp=0.7

> are the very ones to be studied by those who wish to understand how man came into being. They represent the original stock from which all other men have sprung; they were created in a state of innocence as well as perfection: that is, their spirits were pure, free from sin or moral taint. The first two human beings had not only bodies, but souls as well. [...] When Adam sinned, he committed an act of rebellion against his Creator, and so became responsible before Him for his own actions. He lost his spiritualoutofmaniity and communion with God; and from thenceforth he would live in a state ofbidivil

Christian apologetics. Fragments: **"spiritualoutofmaniity"**, **"ofbidivil"**.

#### temp=0.9

> we're gonna make some money. (In the first part of this series, I discussed how many are willing to sacrifice their security for a chance at freedom.) The more free people there are in the world, the less opportunity criminals have for victims; they can no longer afford to take as many chances. The second half of that equation is that as much money as there might be flowing around freely on the World Wide Web, you probably wouldn't get very far with it if the authorities were still watching your steps. But since these kinds of "offshore banks" don't require so-called "proof-of-identity" or any such farrago [...]

Cryptocurrency / offshore banking essay. The lone Wake-shaped vocabulary item: **"farrago"** (real English word meaning *a confused mixture* is the closest the 3B gets to actual Joycean diction, and it's still a real word, not a coinage).

#### temp=1.0

> In my last piece here at the Institute I asked how we could imagine a future of living that would no longer exclude more than half the human race from being able to have families; families whose members were free to choose their own gender identities without worrying what this might do to traditional notions of marriage, child-rearing, divorce. My argument was not for an individualized model with all the attendant emotional isolation and atomization that it entailed but rather toward new forms of community building predicated on mutual recognition, interdependence, mutuality and cooperation [...] for me, social reproduction required rethinking ourselves through such notions as care, compassion, reciprocity etc., instead of focusing upon family structures.

**Coherent critical theory prose.** Donna Haraway-adjacent feminist theory register. No Wake content but striking demonstration that the model accessed contemporary theoretical English at this temperature.

#### temp=1.2

> An original past painting made by artist Paul Oates on a sheet of paper. This item was created in February 2019. Paul, our resident Past Painter, chose this picture from his collection for your enjoyment to reflect the past in these uncertain times. [...] Paul paints with acrylics straight onto the paper, adding all fine art detail using paintbrushes, knives and spray mists etc which he sprays withmihimihi'suspectableetsulkenor' (titanium oxide). All of our pieces are signed and numbered by the 'pasta painter' [...] Dimensions:bannan Paper Sheet A4

Art commerce listing. The longest Wake-adjacent fragment of the entire battery: **"withmihimihi'suspectableetsulkenor'"** (proper Wake portmanteau formation). Also **"bannan"** orthographic drift and **"pasta painter"** as accidental pun.

### Run 3: multi-sequence at temp=0.9 (3 samples)

**[1]**

> but I'm sure some of you are familiar with them. Anyway. The reason I bring this up is because it reminded me of the way in which we view sex today. There was no doubt what God wanted for his children; He knew they were created to be sexual beings with their spouse only as part of His plan to populate the earth. [...] We know that God expects us to have sex within our marriages ONLY, though many men seem not to understand this very basic commandment.

Unironic Christian devotional tract on marital sex. Zero Wake.

**[2]**

> etc... So I am pretty confident there were no trees. We had a few oak but they have been removed since our last visit in the 90s. But to answer your question: You'll see more than you can imagine if you stay all day. The drive is one thing. But then after that... If we are all here together at 8am on Friday morning for breakfast, please bring along a camera (or even better - video) for this awesome event! [...] Will try my best not to sleep through sunrise.

Modern Australian blog about a tree-removal site visit and a breakfast meetup. Zero Wake.

**[3]**

> was the house of his grandmother; she died this winter. From there he went to school, where his father was master of the grammar-school. This happened in 1674 or '5. The elder Sir William Temple left three children behind him: a son named Edward, who died young; two daughters, Anne and Mary. Mr. Edward Temple had married before 1688 a woman whose name we do not know. She appears as Mrs. Ann Tem-oversteple when her eldest daughter was born on May 26th, 1690. [...] They may have lived at Maiden Lane in Covent Garden, since his house seems to have stood about where theherilivery houses now are; but certainly not so long ago as their first child was born, because he must then have been very old. On Jan. 1st, 1722, however, came the news of her death; "A sadnoght

Victorian historical biography. Fragments: **"Tem-oversteple"** (line-break artefact becoming accidental compound), **"theherilivery houses"** (fused compound), **"sadnoght"** (sad + ought or sad + night).

### Run 4: extended at temp=1.1

> where they used to hang out. The last time I visited "the Grove", it was still in a semi-blonde state, its back being strewn with the bones of various dead trees. Back then you had an excellent view over to Moorings Rd through which one could catch occasional glimpses of Lakelemanted Creek meandering past and even see other creek beds snaking off into distant forested areas. One day this will all be gone. [...] It really is like something from another world, so different from anything Melbourne has ever seen before! [...] To put things into perspective for those who don't know about the current events taking place - first we havestickypots development enc

Melbourne tree clearing blog. Fragments: **"Lakelemanted Creek"**, **"semi-blonde state"** with **"bones of various dead trees"** (accidentally Beckett-adjacent imagery), **"havestickypots"**.

---

## Generation Notes

### result: 3B P2 can do everything except Wake

Across nine generation samples spanning five temperatures and three multi-sequence runs, the 3B P2 produced:

- Victorian biblical reference book
- Christian apologetics
- Christian devotional tract on marital sex
- Modern Australian tree-removal blog (twice)
- Cryptocurrency / offshore banking essay
- Critical theory on gender and social reproduction
- Art commerce listing
- Victorian historical biography
- Tree clearing extended blog

**Nine distinct registers, zero sustained Wake prose.** The Wake-coded material that does appear consists of orthographic accidents (line-break splits, long-s artefacts) and isolated portmanteau fragments embedded in non-Wake registers. The model demonstrably has access to a wide range of contemporary and historical English styles. Joyce is not in the set.

### Inverse temperature-Wake-quality relationship

Higher temperatures normally amplify stylistic invention. For the 3B P2, higher temperatures amplify *register diversity from the base distribution* without amplifying Wake-coded content. The longest Wake-adjacent fragment in the entire battery (**"withmihimihi'suspectableetsulkenor'"**) emerged at temp 1.2, but it was embedded in art commerce description, not Joycean prose. The model has no Wake mode to enter; it just samples more freely from non-Wake modes.

This is the inverse of TinyLlama P2's pattern (peak Joycean invention at temp 1.0-1.2) and Llama 1B P2's pattern (multilingual rhythmic invention emerging at temp 1.0). It is the cleanest signature in the project of a model that has been trained on Wake corpus but cannot produce Wake output.

### Notable: the temp 1.0 critical theory sample

The most striking single output in the battery is the temp 1.0 sample on gender and social reproduction. It produces coherent contemporary feminist theory prose: "social reproduction required rethinking ourselves through such notions as care, compassion, reciprocity etc., instead of focusing upon family structures." This sentence is not Joycean but it demonstrates that the model accessed a Haraway-adjacent register at this temperature. **LoRA learned to surface theoretically literate English from the base distribution. It did not learn to surface Wake.**

This isolates the failure precisely. The wall at val 5.33 is not a general capacity failure of the LoRA adapter, the LoRA learned things, what it did not learn is Wake routing.

### Comparison with prior P2 generations

| | TinyLlama P2 | Llama 1B P2 | Llama 3B P2 |
|---|---|---|---|
| Wake authenticity | high (Woolwichleagues, twohandledduolandroom, mooremooremurgessly) | medium (Mr Sooty-Piggy, queenlys signature, cupandnaggin) | **low (fragments only: ideamoreafterft, withmihimihi'suspectableetsulkenor')** |
| Sustained Wake prose | yes | yes, embedded in Victorian frame | **no** |
| Register at generation | Joycean pastiche | Victorian/epistolary with neologisms | **Multi-register, none Joycean** |
| Temperature behaviour | invention peaks at 1.0-1.2 | invention emerges at 1.0 | **inverse: more registers, less Wake** |
| Best Wake fragment | Woolwichleagues (single portmanteau, in context) | cupandnaggin (compound, in Irish-Victorian letter context) | **withmihimihi'suspectableetsulkenor' (compound, in art commerce listing)** |

The refined smaller-model paradox prediction is empirically confirmed. The 3B P2 generations are below 1B P2 in Wake authenticity, which were below TinyLlama P2 in Wake authenticity. The trajectory is monotonic, the same vocabulary class (128K) plus larger body produces *less* Wake routing, not more.

### Connection to the wall

The wall at val 5.33 and the absence of Wake routing in generation are the same finding viewed through different instruments. The LM loss surface shows the routing didn't improve past step 100. The generation samples show what that looks like in token sequences: the model can sample any English register from its pre-training distribution, but the Wake routing layer LoRA was supposed to add never deepened past the floor it found at step 100. The 5.33 ceiling and the *cannot produce Wake* ceiling are the same ceiling.

## Implications for P3

1. **Best checkpoint for P3 source**: step 100 (or any of steps 100-600, all tied at val 5.33). P3 inherits the same embeddings either way because P2 didn't touch them.

2. **P3 morpheme-compositional alignment** is the only remaining lever. LoRA hit its ceiling. The geometric losses in P3 directly modify embeddings. The question is whether the structurally crystalline Wake distribution can be moved by morpheme-direction constraints into a configuration where downstream attention finds further descent.

3. **Prediction**: based on TinyLlama and Llama 1B P3 findings (both produced null results — L_morph never moved, L_device never clustered), the 3B P3 will also produce a null at the auxiliary loss level. But the LM loss + embedding drift will be informative regardless. The wall at 5.33 makes P3's contribution measurable in a way it wasn't for the smaller models.

4. **The 3B P3 result is important either way.** If it breaks the 5.33 ceiling, the chapter has a method-appropriate intervention to point at. If it doesn't, the chapter has a fully empirically supported claim about scale's limit.

## Summary

Llama 3B P2 produced the cleanest plateau-as-finding in the project. Six consecutive evals at val 5.33 to four decimal places. Drift cosine exactly 1.0 across all 172,451 embedding rows. Wake distribution at its most crystalline yet (std 0.0032, mean cos 0.0004 between Wake tokens). The wall is documented, the cause is structural, and the empirical claim is one line.

P2 complete. P3 launches from step 100 best-val source.
