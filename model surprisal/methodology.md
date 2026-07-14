## foundation of the Wake2Vec surprisal branch

Surprisal (Hale 2001) is the negative log-probability of a token given its left context:

```
surprisal(t_i) = -log P(t_i | t_0 ... t_{i-1})
```

A causal language model produces this quantity at every position under teacher forcing.
Cross-entropy loss is the mean of it. the validation-loss numbers logged across the Wake2Vec lineup are, up to averaging, mean per-token surprisal.

### steps (planned)

1. Running the script on a validation split with a matching block size. the reported mean should equal the logged validation loss for that model.
2. Per-token and per-word surprisal over the corpus show where the text is computationally hardest (expected to be the neologisms, portmanteaus, and thunderwords). because surprisal falls as predictability
   rises, high-frequency words (high Zipf rank, the most-frequent words that Burrows's Delta operates on) should be low-surprisal by comparison. Surprisal and Delta therefore read opposite ends of the frequency distribution, Delta the frequent function-word end and surprisal's
   discriminating signal the rare, high-information end where the Wake's coinages sit. This is both a caveat (surprisal is context-dependent, so frequency predicts it on average, not deterministically for any single token) and a connection (the near-inverse is a reason to expect the two measures to carry orthogonal signal, relevant to any Delta-plus-surprisal stylometry).
3. Comparing a base model to a Wake-injected checkpoint shows whether injection lowers surprisal on Wake tokens, i.e. whether the model has moved from surprised-by-the-Wake to expecting it.

