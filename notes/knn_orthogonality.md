# A work-in-progress Wake2vec finding: why Wake tokens have no interpretable nearest neighbours

**Status:** methodological note. Applies to every model in the lineup; numbers below are from Mistral 7B P3 (step 1000), the run with the most complete embedding analysis.

**Short form:** the Wake-to-base nearest-neighbour table reports the maximum of a noise distribution. The finding is the orthogonality itself; the identity of the neighbours carries no information and should not be quoted.

---

## 1. The problem as it presents

Every post-P3 analysis cell prints a top-5 cosine nearest-neighbour table for a sample of injected Wake tokens against the base vocabulary. The output invites interpretation and resists it. A representative extract:

| Wake token | top-1 base neighbour | cos |
|---|---|---|
| `brofèsor` | `▁/**\r` | 0.102 |
| `genesic` | `▁/**\r` | 0.089 |
| `girilis` | `\x1e` | 0.081 |
| `deathfête` | `▁/**\r` | 0.077 |
| `loon` | `` | 0.071 |

The neighbours are code fragments, control characters and private-use codepoints. The intuitive reading is that injection has failed, or that the embeddings are degenerate, but neither is correct. Four pieces of evidence establish that the table is an 
artefact of the measurement rather than a property of the embeddings.

---

## 2. The base-to-Wake cosine distribution is statistically indistinguishable from random

Pairwise cosine between base rows and injected Wake rows, measured on the step-1000 matrix:

| Population pair | mean cos | std |
|---|---|---|
| (base, base) | 0.0237 | - |
| (new, new) | 0.0556 | - |
| **(base, new)** | **0.0032** | **0.0173** |

For independent random unit vectors in `d` dimensions the cosine is approximately zero-mean with standard deviation `1/sqrt(d)`. With `d = 4096`:

```
predicted std  = 1 / sqrt(4096) = 0.015625
observed std   = 0.0173
ratio          = 1.11
```

The observed spread exceeds the random prediction by 11%. The injected block is not merely near-orthogonal to base vocabulary as a tendency. It is near-orthogonal in the way that two independently drawn random directions are near-orthogonal.

## 3. The reported neighbours are the expected maximum of that distribution

If the base-to-Wake cosines are drawn from a distribution with mean 0.0032 and standard deviation 0.0173, then selecting the single largest value from `n` draws has expectation approximately

```
E[max] = mu + sigma * sqrt(2 * ln n)
```

For `n = 32768` base rows:

```
E[max] = 0.0032 + 0.0173 * sqrt(2 * ln 32768) = 0.0821
```

Observed nearest-neighbour cosines run 0.06 to 0.10, with a single maximum of 0.150 across the whole sampled table. The predicted order statistic and the observed values coincide. The kNN routine is not finding the nearest meaningful token; 
it is returning the largest draw from the null, which is precisely what an argmax over a zero-centred distribution does when no structure is present.

This is the sense in which the table has no signal. A top-5 list is informative only when the top cosines are separated from the bulk of the distribution by more than the expected extremum. Here they are not separated from it at all.

## 4. The neighbours are hubs, not neighbours

If the retrieved tokens were genuine semantic neighbours, different Wake queries would retrieve different base tokens. They do not. Across 20 Wake queries at `k = 5` (100 retrieval slots):

| Quantity | Value |
|---|---|
| distinct base tokens retrieved | 82 of 100 slots |
| `▁/**\r` appears as a neighbour of | 6 of 20 queries |
| `\x1e` | 5 of 20 |
| `` | 4 of 20 |
| `]);\r` | 3 of 20 |
| slots filled by a token shared with another, unrelated, query | 26% |

A single token functioning as the nearest neighbour of six semantically unrelated Wake words is the signature of **hubness** (Radovanović et al. 2010). In high-dimensional spaces with near-uniform point distributions, a small number of points with 
slightly atypical norm or position win the maximum-over-noise competition for arbitrary queries. The tokens that recur here are base-vocabulary outliers: byte-level control characters and tokenizer artefacts whose embeddings were barely trained and 
therefore sit in unusual positions relative to the bulk. They are hubs of the retrieval procedure, not neighbours of anything.

Hubness is the standard failure mode of kNN in high dimensions and is independent of whether the embeddings themselves are good. Its presence here is diagnostic of the measurement, not of the model.

## 5. The mechanistic cause: the injected lexicon is overwhelmingly hapax

The preceding three sections establish that there is no neighbourhood structure to retrieve. This section explains why there could not be.

Frequency distribution of the injected lexicon within the training corpus, tokenised with the pipeline's own pattern (`re.findall(r"[a-z']+", text.lower())`):

| Quantity | Value | Share of lexicon |
|---|---|---|
| lexicon types injected | 44,990 | - |
| corpus tokens | 219,602 | - |
| types occurring exactly once | 40,370 | 89.7% |
| types occurring exactly twice | 2,962 | 6.6% |
| types occurring three times or fewer | 44,026 | 97.9% |
| median occurrences per type | 1 | - |
| mean occurrences per type | 1.14 | - |
| maximum occurrences | 35 | - |

Reproduce with:

```bash
python3 -I -c "
import re, collections, statistics
lex = [l.strip() for l in open('wake_lexicon.txt') if l.strip()]
toks = re.findall(r\"[a-z']+\", open('FW_TEXT.txt').read().lower())
c = collections.Counter(toks)
counts = [c.get(w.lower(), 0) for w in lex]
print('types', len(lex), 'tokens', len(toks))
print('hapax', sum(1 for x in counts if x == 1))
print('median', statistics.median(counts))
"
```

**Close to ninety percent of the injected vocabulary occurs exactly once in the entire training corpus.** Distributional similarity is estimated from co-occurrence statistics. A hapax legomenon has one context, which means there is no second 
observation against which a similarity estimate could be formed. The embedding does move, because the language-modelling gradient reaches it in that one context, but it moves so as to be predictable *in that context*, not so as to sit near anything else.

This is not a shortcoming of the training protocol. It is a property of *Finnegans Wake*. Joyce's lexicon is constructed from single-use coinages; a corpus of this text cannot supply repeated contexts because the text does not contain them.

## 6. Consistency with the isotropy measurements

The three measurements agree. Wake-region isotropy, computed as the Mu et al. (2018) partition-function ratio, reads 0.9935 for Mistral and approximately 0.998 for the rest of the lineup. Isotropy approaching 1 means the point cloud has no preferred directions 
and no density variation. Nearest-neighbour retrieval requires anisotropy, that is, clusters, to return anything meaningful: on a near-uniform distribution over the sphere in 4096 dimensions, every point is nearly equidistant from every other, and "nearest" is 
decided by noise.

A further consistency check: the Wake block is internally more coherent than base vocabulary (new-new cosine 0.0556 against base-base 0.0237) and carries 2.37 times the norm (0.3982 against 0.1678). The injected region is therefore a distinct and internally 
structured subspace. Its structure is simply not expressed as proximity to base vocabulary, and kNN against base vocabulary is the wrong instrument for detecting it.

---

## 7. What to report, and what not to

**Do not report:** the identity of retrieved neighbours, or any interpretation of them. This is listed as artefact 4 in the post-P3 analysis-cell audit and the present note supplies its arithmetic.

**Do report:** the orthogonality, with the random-baseline comparison from section 2 attached. Near-orthogonality of an injected block to a pretrained vocabulary at 1.11 times the random standard deviation is a substantive result. It means the model allocated a 
genuinely new region of embedding space rather than mapping Wake coinages onto existing English neighbourhoods, which is the central claim the injection protocol is designed to test.

**The instruments that do resolve structure in the injected region** are the permutation-null excess over chance on morpheme-group coherence and the treated-versus-control drift contrast. Both operate within the Wake block and both return measurable effects 
where kNN against base vocabulary returns noise.

---

## 8. Connection to the morpheme results

The 89.7% hapax rate is the same fact that underlies the morpheme-direction finding. An injected token has no usage history, so nothing competes with its morphological composition in determining where its embedding goes. Established English words carry 
accumulated distributional evidence that overrides their morphology; coinages do not. The direction-consistency result for coinages and its absence for established words therefore follow from the same frequency distribution reported in section 5.

The hapax structure of the lexicon is thus not only the reason kNN fails. It is also the reason the compositional signal is detectable at all.

---

## References

- Mu, J., Bhat, S., and Viswanath, P. (2018). All-but-the-top: simple and effective postprocessing for word representations. *ICLR*.
- Radovanović, M., Nanopoulos, A., and Ivanović, M. (2010). Hubs in space: popular nearest neighbors in high-dimensional data. *Journal of Machine Learning Research* 11, 2487-2531.
