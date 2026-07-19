## foundation of the Wake2Vec surprisal branch

Surprisal (Hale, 2001) is the negative log-probability of a token given its left context:

```
surprisal(t_i) = -log P(t_i | t_0 ... t_{i-1})
```

A causal language model produces this quantity at every position under teacher forcing.
Cross-entropy loss is the mean of it. the validation-loss numbers logged across the Wake2Vec lineup are, up to averaging, mean per-token surprisal.

### steps (planned)

1. Running the script on a validation split with a matching block size. the reported mean should equal the logged validation loss for that model.
2. Per-token and per-word surprisal over the corpus show where the text is computationally hardest (expected to be the neologisms, portmanteaus, and thunderwords). because surprisal falls as predictability
   rises, high-frequency words (high Zipf rank, the most-frequent words that Burrows's Delta operates on) should be low-surprisal by comparison. Surprisal and Delta therefore read opposite ends of the frequency distribution, Delta the frequent function-word end and surprisal's discriminating signal the rare, high-information end where the Wake's coinages sit. This is both a caveat (surprisal is context-dependent, so frequency predicts it on average, not deterministically for any single token) and a connection (the near-inverse is a reason to expect the two measures to carry orthogonal signal, relevant to any Delta-plus-surprisal stylometry).
3. Comparing a base model to a Wake-injected checkpoint shows whether injection lowers surprisal on Wake tokens, i.e. whether the model has moved from surprised-by-the-Wake to expecting it.

## 2. theory

- **Hale (2001), surprisal theory.** Incremental processing difficulty is predicted by the surprisal of each word given its context. Surprisal is an information-theoretic complexity metric for incremental comprehension. (Hale, 2001)
- **The loss-is-surprisal identity.** For a language model, mean per-token cross-entropy loss is the expectation of surprisal. 
- **Scope restriction.** Only the language-modelling cross-entropy is surprisal. Any auxiliary objectives used elsewhere in the project (for example the P3 geometric losses) are not surprisal and are outside the scope of this measurement. 
- **Why measure probability directly.** Directly reading a model's probability distribution over strings is a more reliable probe of its linguistic knowledge than metalinguistic prompting (asking the model to judge), and open-weight models allow it where closed APIs do not (Hu and Levy, 2023). The branch measures surprisal from the distribution rather than prompting the model, and the open-weight lineup is what makes that possible.
- **Surprisal and entropy reduction are a discriminating contrast.** Surprisal is `-log P(actual word)`; entropy reduction (Hale, 2001 & 2011) is the drop in the model's uncertainty about what comes next. They are complementary and they *dissociate by device family*: repetition figures tend to show as entropy collapse *without* a surprisal spike, while disruption figures (anacoluthon) show as a surprisal spike. So the two together separate device families that either alone would conflate, which is a result in Hale's own comparative-complexity-metric tradition and the reason the detectability framing treats the contrast as central rather than incidental. `surprisal_extract.py` computes both from the same forward pass (predictive entropy per position, and a signed step-to-step entropy-reduction proxy); fuller operationalisations of entropy reduction over full continuations exist and are a Phase-2 refinement.

## 3. method

- **Teacher-forced forward pass.** The text is tokenised and passed through the model; at each position the log-probability assigned to the actual next token gives that token's surprisal.
- **Non-overlapping blocks.** Long text is chunked into non-overlapping blocks of a fixed token length (default 1024). This mirrors how block-wise validation loss is computed, which is what makes the reported mean comparable to a logged val loss.
- **Block-boundary handling.** The first token of each block has no in-block left context, so its surprisal is undefined and is dropped (recorded as NaN, excluded from the mean). This is consistent with block-wise loss computation. It does introduce a mild artifact for per-token *profiles* at block starts; a sliding-window variant is the planned refinement for boundary-clean profiles and is not implemented in this starter.
- **Per-word aggregation.** Subword surprisals are summed within each word, because
  `P(word | context)` is the product of its subword conditionals, so `-log P(word | context)` is the sum of subword surprisals. This is Hale's per-word unit and it cancels most of the tokenizer-dependence that makes per-token values hard to compare across models. Word boundaries are taken from the fast tokenizer's word alignment. Note that summing subword surprisals is exact only for the canonical tokenization; the true per-word probability marginalises over all
  tokenisations of the word. Shi et al. (2026) give a token-to-word decoding algorithm that handles this properly for open-vocabulary settings, and is the principled upgrade if a fully rigorous per-word estimate is needed. (Shi et al, 2026.)
- **Units.** Surprisal is reported in nats (natural log, so the mean equals cross-entropy loss) and in bits (log base 2, which reads more naturally as "bits of surprise").

## 4. reproducing a logged validation loss

To confirm the identity against a specific number:

1. Runs on the model's **validation split**, not the whole corpus (the project uses a seed-42, 90/10 split).
2. Use the **same block size** the model was trained/evaluated with.
3. The reported `MEAN per-token surprisal` (nats) should equal the logged validation loss for that checkpoint, up to minor differences in padding and masking conventions.

Also, runs over the whole corpus where the mean is the corpus-wide mean surprisal, i.e. the difficulty of the Wake in nats under that model.

## 5. Design decisions

| Decision | Reason |
|---|---|
| Per-word as the comparison unit | Per-token surprisal is tokenizer-dependent; per-word (sum of subwords) is Hale's unit and is comparable across tokenizers and to human data. |
| Drop block-boundary first tokens | Their context is truncated; keeping them would inflate the mean and break comparability with block-wise val loss. |
| float32 for the log-softmax | Log-probabilities are sensitive to reduced precision; float32 keeps the measurement accurate. Small models fit easily. |
| Inference only, deterministic | Surprisal is a measurement; there is no training and no sampling in the metric itself. |
| Standard-library-adjacent, single file | Only torch and transformers are required (peft only for LoRA adapters), so the measurement is easy to audit and rerun. |

## citations

Hale, John. 2001. "A Probabilistic Earley Parser as a Psycholinguistic Model." In *Proceedings of the Second Meeting of the North American Chapter of the Association for Computational Linguistics on Language Technologies* (NAACL '01), 1–8. Pittsburgh, PA: Association for Computational Linguistics. https://doi.org/10.3115/1073336.1073357.

Hale, John. 2011. "What a Rational Parser Would Do." *Cognitive Science*, 35: 399-443. https://doi.org/10.1111/j.1551-6709.2010.01145.x

Hu, Jennifer and Levy, Roger. 2023. "Prompting is not a substitute for probability measurements in large language models." In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 5040–5060, Singapore. Association for Computational Linguistics

Shi et. al. 2026. "Word Surprisal Correlates with Sentential Contradiction in LLMs". In *Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics*, (Volume 1: Long Papers), pages 4549–4564, Rabat, Morocco. Association for Computational Linguistics.
