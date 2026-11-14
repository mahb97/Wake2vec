## foundation of the Wake2Vec surprisal branch

Surprisal (Hale, 2001) is the negative log-probability of a token given its left context:

```
surprisal(t_i) = -log P(t_i | t_0 ... t_{i-1})
```

A causal language model produces this quantity at every position under teacher forcing. Cross-entropy loss is the mean of it. The validation-loss numbers logged across the Wake2Vec lineup are, up to averaging, mean per-token surprisal **for P1 and P2 only**. P3's logged `eval_loss` is not surprisal and must not be treated as it; see §4.

Implementation: `surprisal_extract.py`.

### steps

1. Running the script on a validation split with a matching block size. The reported mean should equal the logged validation loss for that model, subject to the three conditions in §4.
2. Per-token and per-word surprisal over the corpus show where the text is computationally hardest (expected to be the neologisms, portmanteaus, and thunderwords). Because surprisal falls as predictability rises, high-frequency words (high Zipf rank, the most-frequent words that Burrows's Delta operates on) should be low-surprisal by comparison. Surprisal and Delta therefore read opposite ends of the frequency distribution, Delta the frequent function-word end and surprisal's discriminating signal the rare, high-information end where the Wake's coinages sit. This is both a caveat (surprisal is context-dependent, so frequency predicts it on average, not deterministically for any single token) and a connection (the near-inverse is a reason to expect the two measures to carry orthogonal signal, relevant to any Delta-plus-surprisal stylometry).
3. Comparing a base model to a Wake-injected checkpoint shows whether injection lowers surprisal on Wake tokens, i.e. whether the model has moved from surprised-by-the-Wake to expecting it. Operationalised in §6.

## 2. theory

- **Hale (2001), surprisal theory.** Incremental processing difficulty is predicted by the surprisal of each word given its context. Surprisal is an information-theoretic complexity metric for incremental comprehension. (Hale, 2001)
- **The loss-is-surprisal identity.** For a language model, mean per-token cross-entropy loss is the expectation of surprisal.
- **Scope restriction.** Only the language-modelling cross-entropy is surprisal. The P3 geometric losses are not surprisal and are outside the scope of this measurement. This restriction is not merely conceptual: because no P3 script overrides `prediction_step`, the auxiliary terms are inside the logged `eval_loss`, so the restriction has to be enforced arithmetically before any P3 number is used. See §4.
- **Why measure probability directly.** Directly reading a model's probability distribution over strings is a more reliable probe of its linguistic knowledge than metalinguistic prompting (asking the model to judge), and open-weight models allow it where closed APIs do not (Hu and Levy, 2023). The branch measures surprisal from the distribution rather than prompting the model, and the open-weight lineup is what makes that possible.
- **Surprisal and entropy reduction are a discriminating contrast.** Surprisal is `-log P(actual word)`; entropy reduction (Hale, 2001, 2006 & 2011) is the drop in the model's uncertainty about what comes next. They are complementary and they *dissociate by device family*: repetition figures tend to show as entropy collapse *without* a surprisal spike, while disruption figures (anacoluthon) show as a surprisal spike. So the two together separate device families that either alone would conflate, which is a result in Hale's own comparative-complexity-metric tradition and the reason the detectability framing treats the contrast as central rather than incidental. `surprisal_extract.py` computes both from the same forward pass (predictive entropy per position, and a signed step-to-step entropy-reduction proxy); fuller operationalisations of entropy reduction over full continuations exist and are a Phase-2 refinement.

## 3. method

- **Teacher-forced forward pass.** The text is tokenised and passed through the model; at each position the log-probability assigned to the actual next token gives that token's surprisal.
- **Non-overlapping blocks.** Long text is chunked into non-overlapping blocks of a fixed token length. This mirrors how block-wise validation loss is computed, which is what makes the reported mean comparable to a logged val loss. **The block length is per model and is not a free parameter**: it must equal the `SEQ_LEN` the checkpoint was evaluated with, which across the lineup is 128 (Qwen 2.5-14B), 256 (Mistral 7B, Phi-3.5) and 512 (Llama 3.1-8B). A mismatch changes the amount of left context available at every position and the comparison silently fails.
- **Block-boundary handling.** The first token of each block has no in-block left context, so its surprisal is undefined and is dropped. This is exactly the label shift a causal LM head performs, so it is consistent with block-wise loss computation rather than an additional convention. It does introduce a mild artifact for per-token *profiles* at block starts; a sliding-window variant is the planned refinement for boundary-clean profiles and is not implemented.
- **Per-word aggregation.** Subword surprisals are summed within each word, because `P(word | context)` is the product of its subword conditionals, so `-log P(word | context)` is the sum of subword surprisals. This is Hale's per-word unit and it cancels most of the tokenizer-dependence that makes per-token values hard to compare across models. Word boundaries are taken from the fast tokenizer's word alignment. Words spanning a block boundary are scored incompletely and are dropped. Note that summing subword surprisals is exact only for the canonical tokenisation; the true per-word probability marginalises over all tokenisations of the word. Shi et al. (2026) give a token-to-word decoding algorithm that handles this properly for open-vocabulary settings, and is the principled upgrade if a fully rigorous per-word estimate is needed.
- **Units.** Surprisal is reported in nats (natural log, so the mean equals cross-entropy loss) and in bits (log base 2, which reads more naturally as "bits of surprise").

## 4. reproducing a logged validation loss

**1. The validation split must be reproduced exactly, not approximated.** The project tokenises the whole corpus, chunks it into non-overlapping `SEQ_LEN` blocks, and then splits the *blocks* with `sklearn.model_selection.train_test_split(blocks, test_size=0.10, random_state=42)`, which shuffles. A 90/10 split taken any other way gives a different set of blocks and a different mean. `surprisal_extract.py` replicates this call, and the replication has been checked against a live run: 2,875 blocks split 2,587 / 288, matching what the Qwen P3 launch printed.

**2. A P3 `eval_loss` is a composite and is not surprisal.** No P3 script overrides `prediction_step`, so the logged evaluation figure is

```
eval_loss = LM + lambda_morph*L_morph + lambda_device*L_device
                + lambda_repulsion*L_repulsion + lambda_norm*L_norm
```

On Mistral P3 the auxiliary offset runs about 0.97 nats, so the logged 5.8024 at step 1000 corresponds to an LM term of roughly 4.83. **Comparing a surprisal mean against the raw logged P3 figure will appear to fail by approximately the offset.** Subtract it first. P1 and P2 figures are pure LM and need no adjustment.

**A third condition, which is about the model.** PEFT writes the resized embedding into `adapter_model.safetensors` and then ignores it on load, because the embedding is not listed in `modules_to_save`. A surprisal script that does `PeftModel.from_pretrained(base, checkpoint)` therefore measures the **base embedding geometry with the adapter on top**, not the checkpoint. This is the same defect that silently reset the Wake geometry on every P3 resume until `EmbeddingRestore` was written. For P1 and P3 checkpoints the embedding must be loaded explicitly; `surprisal_extract.py` refuses to run on an adapter without `--embeddings` and prints a fingerprint of the matrix it measured so the devlog records which geometry produced the number.

Running over the whole corpus rather than the validation split gives the corpus-wide mean surprisal, i.e. the difficulty of the Wake in nats under that model. That figure is not comparable to any logged value, because the model trained on 90% of it.

## 5. design decisions

| Decision | Reason |
|---|---|
| Per-word as the comparison unit | Per-token surprisal is tokenizer-dependent; per-word (sum of subwords) is Hale's unit and is comparable across tokenizers and to human data. This matters more in the fourteen-model lineup than it did in the ten, because base vocabularies now span 32,000 to 262,144. |
| Block length fixed per model, not globally | It must match the checkpoint's `SEQ_LEN` or the reproduction claim in §4 is void. |
| Drop block-boundary first tokens | Their context is truncated; keeping them would inflate the mean and break comparability with block-wise val loss. It is also what the causal LM head already does. |
| float32 for the log-softmax, chunked over positions | Log-probabilities are sensitive to reduced precision. The memory cost scales with vocabulary, not model size: at 262,144 vocabulary and a 1024-token block the fp32 log-softmax alone is 1 GiB, so positions are processed in chunks. |
| Explicit embedding load, with a fingerprint | See condition 4 in §4. The failure mode is silent and produces a plausible number. |
| Inference only, deterministic | Surprisal is a measurement; there is no training and no sampling in the metric itself. |
| Single file, torch and transformers only | peft only for LoRA adapters, scikit-learn only to reproduce the split. The measurement is easy to audit and rerun. |

## 6. the injection contrast

Step 3 above, operationalised. Because the Wake tokens occupy a known contiguous row range `[BASE_VOCAB, TOTAL_VOCAB)`, every scored token can be labelled Wake or base without additional annotation, and the script reports the two means separately.

The quantity of interest is the **gap** between Wake-token and base-token surprisal, measured on the same checkpoint, and how it changes across P1, P2 and P3 against the uninjected base model. A base model has never seen the Wake tokens at all, so under injection-then-training the Wake-token mean should fall. Two outcomes are distinguishable and they mean different things:

- **The gap closes toward zero.** The model predicts Wake tokens about as readily as base vocabulary in Wake context, which is the strong form of acquisition.
- **The gap narrows but stays open.** The model has learned the register without collapsing the distinction, which is consistent with the generation result: across roughly 5,000 generated tokens the longest verbatim span reproduced from the corpus is four words, all of them function words, with corpus type recall of 74.5%. A model that had driven Wake-token surprisal to near zero on its own training corpus would be a memorisation signal, and the generation battery says that is not what happened.

This makes the surprisal branch a quantitative check on a result currently resting on a span-matching test, which is worth having because the two measures fail in different ways.

## 7. scope within the lineup

The lineup is fourteen decoder-only models spanning 0.5B to 14B and base vocabularies from 32,000 to 262,144 (see `LINEUP.md`). Three consequences for this branch:

- **Per-token surprisal is not comparable across models** with different vocabularies, for the same reason validation loss is not: the token mixture differs. Per-word is, which is why it is the reported unit.
- **The four models added in November 2026 cross vocabulary against scale** for the first time. Qwen2.5-0.5B, 1.5B and 14B share a tokenizer exactly, so their per-token surprisals *are* directly comparable to one another, which makes that family the one place in the lineup where a per-token scale comparison is valid.
- **The Wake-token share of the validation set varies with vocabulary**, from roughly 58% of the post-injection vocabulary at 32,000 base to 14.5% at 262,144. The injection contrast in §6 must therefore be read per model and not pooled.

## citations

Hale, John. 2001. "A Probabilistic Earley Parser as a Psycholinguistic Model." In *Proceedings of the Second Meeting of the North American Chapter of the Association for Computational Linguistics on Language Technologies* (NAACL '01), 1–8. Pittsburgh, PA: Association for Computational Linguistics. https://doi.org/10.3115/1073336.1073357.

Hale, John. 2006. "Uncertainty About the Rest of the Sentence". *Cognitive Science*, 30: 643-672. https://doi.org/10.1207/s15516709cog0000_64

Hale, John. 2011. "What a Rational Parser Would Do." *Cognitive Science*, 35: 399-443. https://doi.org/10.1111/j.1551-6709.2010.01145.x

Hu, Jennifer and Levy, Roger. 2023. "Prompting is not a substitute for probability measurements in large language models." In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 5040–5060, Singapore. Association for Computational Linguistics

Shi et al. 2026. "Word Surprisal Correlates with Sentential Contradiction in LLMs". In *Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics*, (Volume 1: Long Papers), pages 4549–4564, Rabat, Morocco. Association for Computational Linguistics.
