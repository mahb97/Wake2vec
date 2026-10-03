# wake2vec Qwen 2.5-14B P2 Results

## Configuration

| | value |
|---|---|
| Model | `Qwen/Qwen2.5-14B` (4-bit NF4) |
| Phase | P2 (LoRA behavioural adaptation, embeddings frozen) |
| P1 source | WakeOverlay canonical, sentry step 2700 |
| Vocabulary | 195,888 total; 152,064 base; **43,824 Wake (22.4%)** |
| Architecture | 48 layers, hidden 5120, intermediate 13824, untied `lm_head` |
| LoRA | rank 8, alpha 16, dropout 0.1, on q/k/v/gate/up/down |
| Trainable | 30,474,240 parameters, 576 LoRA tensors |
| LR / schedule | 5e-5, cosine, warmup 0.10, weight decay 0.01 |
| Optimizer | Adafactor |
| Batch / SEQ_LEN | 1 x 16 / **128**, matching P1 |
| Steps | 3,000, evaluation every 50, save every 25 |
| Corpus | 3,232 training blocks, one epoch = 202 optimiser steps |

`SEQ_LEN 128` is the shortest in the lineup, set by the memory cost of a 14B model with LoRA on a T4, and matched to P1 so the validation block set is identical across phases.

## Training

| | value |
|---|---|
| **Best validation** | **5.9209 at step 1600** |
| Final validation | 6.3095 at step 3000 |
| Reduction from P1 minimum | 9.13 |
| Evaluations after the floor | 28, across 1,400 steps, **no new low** |

### The epoch staircase

Validation did not drift upward after step 1600. It rose in discrete, evenly spaced increments separated by flat intervals.

| tread | evals | steps | band | width | mean | above floor | riser in |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 1650-2000 | 5.973-6.057 | 0.084 | 6.020 | +0.099 | |
| 2 | 4 | 2050-2200 | 6.110-6.163 | 0.053 | 6.131 | +0.210 | +0.112 |
| 3 | 4 | 2250-2400 | 6.186-6.212 | 0.026 | 6.202 | +0.281 | +0.070 |
| 4 | 4 | 2450-2600 | 6.257-6.264 | 0.007 | 6.260 | +0.339 | +0.058 |
| 5 | 4 | 2650-2800 | 6.294-6.299 | 0.005 | 6.296 | +0.375 | +0.037 |
| 6 | 4 | 2850-3000 | 6.305-6.311 | 0.006 | 6.309 | +0.388 | +0.012 |

Three regularities, mutually independent:

1. The risers fall on epoch boundaries. One epoch is 202 optimiser steps. Evaluation runs every 50, so a riser observed at step *n* locates the transition in the window (*n* - 50, *n*]. The boundaries at 2020, 2222, 2424 and 2626 fall inside the windows
of the risers observed at 2050, 2250, 2450 and 2650 respectively. Five risers, five boundaries, no misses and no unexplained transitions. The sixth boundary falls at 3030, past the end of the run, and no sixth riser occurred.

3. The risers shrink monotonically: +0.112, +0.070, +0.058, +0.037, +0.012.

4. The treads tighten and then converge: 0.084, 0.053, 0.026, 0.007, then 0.005 and 0.006. The contraction is monotone through tread 5 and bottoms out there, which is the behaviour of a variance floor rather than a continuing trend.

The decomposition is mechanical. Overfitting on a corpus of 3,232 blocks accrues each time the model completes a pass and meets the same text again, supplying the risers. The annealing cosine schedule contracts step-to-step variance in the validation 
estimate, supplying the treads. The staircase only becomes visible after step 2000 because before then the noise band is wider than the increments.

### No confirmable turn and why it did not matter

The project's divergence criterion requires a monotone training curve alongside rising validation. This run does not supply one: corrected training loss reversed direction eight times in thirteen intervals between steps 1600 and 2250 and continued to 
oscillate thereafter. Four candidate turn signals were raised and refused at 1450, 1650, 1900 and 2050, each correctly, since the training curve turned back in every case.

The turn was nonetheless real and is legible in the tread means, which rise monotonically across all six. The criterion failed; the phenomenon did not. This is the methodological finding of the run: the same protocol, cadence and reading rules produce a 
legible loss curve at 7B and 8B under AdamW and an illegible one at 14B under Adafactor, and the staircase is what recovered the signal the reading rules could not.

### Reporting note

Logged training loss on this run is inflated 16x because `LoRATrainer.compute_loss` does not receive the gradient-accumulation division. Training itself was unaffected, adaptive optimisers being near-invariant to a constant gradient rescale. All training 
figures above are the corrected values. 

## Embedding geometry

P2 froze the embedding matrix completely and trained LoRA adapters only, so the embedding analysis from P1 carries over unchanged and is not repeated here. See `p1_qwen14b_canonical_outputs.md`.

## Generation

Samples are in `p2_qwen14b_generation.md`. Generated from step 1600 with the adapter loaded explicitly, since `load_best_model_at_end` is unset and the in-memory state at the end of training is step 3000, which sits at the top of the staircase.

### The memorisation test

The verdict for P2 is the nv-recall memorisation-versus-transformation test: whether the model reproduces spans of the training text or recombines its lexicon into novel sequences. Word-level tokenisation, matched against the full 219,481-token training 
corpus (57,972 types).

| sample | types | in corpus | novel | retrieval | longest matching span |
|---|---|---|---|---|---|
| T0.5 | 127 | 117 | 10 | 92.1% | 4 words, *the end of the* |
| T0.7 | 117 | 104 | 13 | 88.9% | 5 words, *they were all in the* |
| T0.9 | 134 | 121 | 13 | 90.3% | 3 words, *that one of* |
| T1.0 | 132 | 117 | 15 | 88.6% | 5 words, *what do you think of* |
| T1.2 | 132 | 115 | 17 | 87.1% | 4 words, *a hat and all* |

No distinctive span from the training text is reproduced anywhere in the battery. The longest matches are three to five word sequences composed entirely of high-frequency function words, which are what chance produces at that frequency in any English 
text of this length. The test does not return a short memorisation signal; it returns none.

The retrieval column uses all word types as its denominator, including function words, and is therefore not directly comparable to the Llama 8B's reported figure of roughly 70% verified Wake-lexicon retrieval, which used the injected-token list as its 
denominator. A like-for-like recount against the 43,824 Wake rows is outstanding.

### Retrieved and redeployed

Three items in the battery are distinctive enough to be worth tracing individually. All three are in the training text, and all three appear in novel frames:

| | training text | generation |
|---|---|---|
| | "for whom the audible-visible-gnosible-edible **world**" | "By their audible-visible-gnosible-edible **fire**" |
| | "micks his **aquascutum**; the enjoyment he took" | "'twas our **aquascutum** threeingles wail me" |
| | "**Bulbul**, bulbulone! I will shally." | "to lie in **bulbul** regmped upon these word" |

Under word tokenisation `audible-visible-gnosible-edible` is a single lexical item. It is retrieved, not invented, and it carries none of its original context with it. This is the behaviour the test is designed to distinguish from regurgitation.

### Novel coinages

Sixty-eight word types across the battery do not appear anywhere in the training corpus. Excluding tokenisation artifacts and ordinary English absent from the Wake, the substantive formations are:

`blfingerpatsts`, `bwyerow`, `chapfawthery`, `chilamorist`, `chivingestself`, `coiffseons`, `dairymanh`, `gobbitsojer`, `hshemeries`, `kippersic`, `menlikeng`, `mihimihiwood`, `mourninpay`, `mywhiskersory`, `napousseypram`, `orsaltings`, `pangeante`, 
`paribshou`, `peneysments`, `prickedmiltiades`, `redmaidschittering`, `r-yarnspinnersged`, `saiscat`, `shamesorus`, `shimmyardards`, `studweckingland`, `twitwer`, `umbloomget`, `wiaomous`

Several are semantically loaded rather than merely orthographically strange:

| coinage | reading |
|---|---|
| `umbloomget` | carries *Bloom* |
| `prickedmiltiades` | Miltiades, the Marathon general |
| `shamesorus` | Shem, whose Wake epithet is built on shame |
| `blasphematory` | blasphemy, defamatory, -ory |
| `marygales` | Mary, marigolds, gales |
| `mihimihiwood` | Latin *mihi* reduplicated |
| `Fionand` | Fionn, and the land |
| `girness` (in *All girness green*) | Guinness, girn, green |

### Register

The battery sustains Wake register at every temperature rather than only at the high end. Three features carry it.

**Incantatory repetition.** At 0.7: "On loud sable marl her sable hair. With your sable eye, sable beunder". The quadruple is a structural device, not a lexical one.

**Direct address and interjection.** "Give you that!", "kissykissy! Eau!", "Bramps!", "Rear Deck!", "Say your coiffseons redmaidschittering!" The second-person imperative breaking into narration is characteristic.

**Run-on subordination with parenthetical interruption**, sustained across clause boundaries that do not resolve. This is the hardest feature to imitate and the one most resistant to temperature.

**Dublin reference without prompting.** `Shaun` by name, the Hiberno-English *gosson* for boy, *buss* for kiss, "All girness green", and water threaded through from "Eau!" to "preast water". The prompt supplies *riverrun* and nothing else.

## Position in the lineup

| model | base vocab | injected | Wake share | output character |
|---|---|---|---|---|
| TinyLlama 1.1B | 32,000 | ~44,500 | 58% | dense Wakean pastiche |
| Mistral 7B | 32,768 | 44,553 | 58% | strong Wake register |
| Llama 3.2-1B | 128,256 | 44,195 | 26% | Victorian-comic, not Wakean |
| Llama 3.1-8B | 128,256 | 44,195 | 26% | ~70% Wake-lexicon retrieval |
| **Qwen 2.5-14B** | **152,064** | **43,824** | **22%** | **sustained Wake register** |

### The Smaller Model Paradox is scale-bounded

The standing finding is that TinyLlama (32K vocabulary) produces substantially more convincing Wake output than Llama 3.2-1B (128K), because a smaller tokenizer forces more of the Wake lexicon to be learned at the embedding layer while a larger one lets 
the model reach for subword composition from its English priors instead.

Two observations from this run bear on it.

1. Injection count is not the variable. The lineup spans 43,824 to 44,553 injected tokens, a range of 729. This is a property of the Wake lexicon, which is fixed, not of the model. Any account resting on how many tokens were added is therefore ruled out.

2. Wake share does not predict output quality either. Qwen has the lowest share in the set at 22.4% and produced the battery above. The Llama 3.2-1B at 26% produced Victorian comic prose.

What separates them is scale. The paradox was established on a comparison of two models at roughly 1B, and it holds cleanly there. It does not survive scaling: at 8B and 14B the large tokenizer stops being a disadvantage. This sharpens the mechanism 
rather than discarding it. Subword composition is a *cheaper* route to a Wake form than learning its embedding, and a 1B model takes the cheap route because it cannot afford both English competence and a new register. A 14B can afford both.

The paradox is real and bounded above. The crossover lies between 1B and 8B, and the Llama 3.2-3B is positioned to locate it.

## What this phase establishes

1. **Best validation 5.9209 at step 1600**, a reduction of 9.13 from the P1 minimum.
2. **An epoch staircase**, the most mechanically decomposable morphology in the lineup: five risers on five epoch boundaries, monotone shrinkage, treads tightening to a variance floor.
3. **No confirmable turn by the project's criterion**, for a stated mechanical reason, with the turn itself nonetheless legible in the tread means.
4. **The memorisation test passes with no signal at all.** Longest reproduced span is five generic function words against a 219,481-token corpus.
5. **The suspension question is answered for the 14B.** The dense polyglot output of P1 becomes legible as Wake under routing, as it did for Mistral and the Llama 8B.
6. **The Smaller Model Paradox is scale-bounded** and the largest vocabulary in the lineup is not a barrier at 14B.
