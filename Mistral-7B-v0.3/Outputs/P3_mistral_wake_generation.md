# wake2vec Mistral 7B v0.3 P3 Generation Battery

Two batteries, matched draw for draw: the geometry endpoint (step 1000) and the best-validation checkpoint (step 100). Same prompt, same seeds, same lengths, same sampling parameters. Raw output in `p3_mistral7b_generation_raw.md` and `p3_mistral7b_generation_step0100.md` on Drive.

| | |
|---|---|
| model | mistralai/Mistral-7B-v0.3 (4-bit NF4) |
| prompt | `riverrun, past Eve and Adam's,` |
| sampling | top_p 0.92, top_k 50, repetition_penalty 1.15 |
| base seed | 42, per-sample derived |
| sections | A: temperature sweep 0.5 to 1.2 at 512 tokens; B: 3 samples at 0.9, 384 tokens; C: 1 sample at 0.9, 1024 tokens |
| training block length | 256 tokens |
| wall time | 9.7 min (step 1000), 9.2 min (step 100) |

The pairing is forced by the run. `load_best_model_at_end` is False, so step 1000 is the maximum-geometry state; step 100 is the language-modelling optimum at `L_morpheme = 0.6555`, barely moved from the 0.6700 start. Either alone misrepresents the phase.

---

## 1. Memorisation: none

| battery | tokens | types | in corpus | novel | recall | TTR | longest span |
|---|---|---|---|---|---|---|---|
| step 100 | 2,440 | 875 | 675 | 200 | **77.1%** | 0.359 | **4 words** |
| step 1000 | 2,552 | 943 | 703 | 240 | **74.5%** | 0.370 | **4 words** |

Measured against the full corpus: 219,526 tokens, 57,731 types.

**The longest verbatim span either model reproduces is four words, and in both cases it is pure function-word filler:**

- step 100: *"to go with the"*
- step 1000: *"if you want to"*

Neither contains a single Wake item. Across roughly 5,000 generated tokens, with the entire lexicon available as single injected tokens and a corpus small enough to memorise, **the model never recites.** This is the strongest form the nv-recall result can take, and it is the precondition for reading anything else here as composition rather than retrieval.

### What the novel types actually are

The raw novel count is misleading on its own, because the injection introduced a large population of truncated stems that the model emits as if they were words (`welr`, `givr`, `completn`, `propeh`, `hea`, `leas`, `sofg`). Classifying each novel type by its edit distance to the nearest corpus word:

| | step 100 | step 1000 |
|---|---|---|
| novel types | 200 | 240 |
| truncation of a corpus word | 137 (68%) | 161 (67%) |
| one edit from a corpus word | 35 (18%) | 36 (15%) |
| **genuinely new (more than one edit)** | **28 (14%)** | **43 (18%)** |

**Genuine coinage rises from 28 to 43 types, an increase of 54%.** The truncation share is identical in both (68% against 67%), which places that phenomenon with the embedding injection rather than with P3: it is a property of having 44,553 Wake tokens in the vocabulary, many of them sub-word fragments, and it does not change as the geometry moves.

Step 1000's coinages: `coddlelecherskithers`, `childspiloter`, `bigyttensning`, `laughinnter`, `flaskshyly`, `heoning`, `lazaring`, `boackicess`, and a dense layer of invented toponyms and surnames (`Allzinstown`, `Munddowd`, `Killyke`, `Gambelley`, `Hulleylock'd`, `Cangogel's`, `O'Guffa`, `Mushsman`, `Dayodore`).

Step 100's: `adamanterioy`, `questureclass`, `filarwelling`, `nivuloning`, `sententivee`, `helpllall`, `kisyhree`, alongside `LadobiddyBankly`, `Tullorry`, `Paygutus`, `O'Sulor`, `Finegato`.

---

## 2. The structural difference: the model learned the page, not just the language

Joyce's typographic apparatus is in the corpus in quantity: **667** line-initial speech dashes (the "perverted commas" he refused, replaced by the tiret), **794** italic spans marked `_like this_`, and **1,051** bracketed page numerals carried over from the edition.

Step 1000 reproduces all three. Step 100 reproduces one of them, twice, and opens italics it never closes.

| convention | corpus | step 100 | step 1000 |
|---|---|---|---|
| line-initial speech dash | 667 | **2** | **19** |
| closed italic spans `_x_` | 794 | **0** (5 unclosed marks) | **2** (`_Fon's Dale_`, `_Dugen's Shicka_`) |
| bracketed page numeral | 1,051 | **0** | **1** (`[63]`) |

**The page numeral is the most striking of the three.** `[63]` carries no linguistic content whatsoever. It is a property of the edition rather than of the text, the kind of token a language model has no semantic use for, and the model emits it mid-paragraph in the position Joyce's corpus puts them. It did not learn a word or a construction there. It learned the shape of the artefact.

The italic spans are the second. Both are correctly opened and closed, and both contain coined proper nouns, which is precisely Joyce's use of the convention for song titles, foreign tags and book names. Step 100 scatters five underscores and closes none of them, so it has the mark without the structure.

### The dialogue form

```
— A, O, M!
— Though I am giv!
— Which you'll say!
— Hail to ye, practing the twos!
...
— Lyon?
— Dayodore!
— That's more than a little yearlyng from the shad.
```

That is stichomythia: rapid two-line alternation, the form carrying the Four Old Men and the Mookse and the Gripes. Section C sustains it antiphonally across the full 1024 tokens:

```
— So they nuzzled: Shuny from the Loud! Foese the Fide,
you antiabecedarian is boycotted. Du simp. Hap where we
so have been propeh us, the plain is perdrix and the glunn
in our silvehly too.

— My one and sofr for us! Squals and lund of the
Mollyum-Ruga! As he will be edicted about it too.
```

**What distinguishes this from the lexical results is that none of it is vocabulary.** The speech dash is a punctuation mark doing the work of a verb of saying. The italic pair is a delimiter doing the work of a citation. The page numeral is an index with no utterance behind it at all. These are the typographic layer carrying structural information that lexis normally carries, and the model acquired that layer separately from the words.

This connects directly to the Silent Similes argument in the CST dissertation, where punctuation functions as a surrogate comparator and carries semantic work the lexicon has vacated. The case here is the structural counterpart: punctuation carrying speaker change, citation and reference. Both are instances of the same underlying claim, that in Joyce the typographic layer is load-bearing and not decorative, and a model trained on him picks it up as a layer in its own right.

Because the seeds are matched, the two batteries began from identical sampling conditions. The sequences diverge after the first differing token and are not controlled per-token thereafter, so this is one effect across ten samples rather than a paired statistic. At 2 against 19 on the dash, and 0 against 3 on the other two conventions combined, it is large enough to report and small enough in n to want a second seed.

**The summary form of the pair, which is the sentence this battery exists to earn:** the checkpoint that is 0.41 nats worse at predicting Joyce is the one that learned how his page is set.

---

## 3. The two pre-registered questions

### Q1: is the 0.124-nat penalty qualitatively visible?

**No, and the direction is the opposite of what the penalty predicts.**

Step 1000 is 0.41 nats worse than step 100 on held-out language modelling (val LM 4.8289 against 4.4215). On every quantity measured here it is the richer output:

| measure | step 100 | step 1000 | direction |
|---|---|---|---|
| val LM (lower better) | **4.4215** | 4.8289 | step 100 better |
| genuine coinages | 28 | **43** | step 1000 better |
| type-token ratio | 0.359 | **0.370** | step 1000 better |
| dialogue dashes | 2 | **19** | step 1000 better |
| corpus recall | 77.1% | **74.5%** | step 1000 less derivative |

The Llama 3B concluded that strong auxiliary lambdas "cost generalisation without visibly degrading generation quality", with the penalty "real in the loss numbers and subtle-to-invisible in the output". **Mistral strengthens that to an anti-correlation.** On these five measures the worse-perplexity checkpoint is the better Wake generator, across a 0.41-nat gap.

This is not a claim that step 1000 is better *as writing*, which is not a quantity this file can measure. It is a claim that held-out perplexity and every lexical and structural proxy for Wake register point in opposite directions across this pair.

A visible correlate of the repetition the lower-temperature settings produce: step 100 at temperature 0.5 locks onto `metamorphoseous` four times in eleven lines, and then onto a `themselves` loop (`themselves and themselves, dumm and themselved, themselves and cudd, kraaking themselver and verve, themselvs and histher`). Step 1000 at the same temperature and the same repetition penalty repeats only `climes, climes`.

### Q2: does the morpheme effect surface in coinages?

**Not supported, and the test as built is too crude to support it either way.**

The prediction from Finding 3 of the outputs file was that the morpheme effect, which lives in coinages rather than established words, should be visible as novel words carrying the morphemes that gained direction consistency. Classifying novel types by prefix and suffix string match:

| | step 100 | step 1000 |
|---|---|---|
| novel carrying a pre-significant morpheme | 3 (1.5%) | 6 (2.5%) |
| novel carrying a largest-gainer morpheme | 5 (2.5%) | 1 (0.4%) |

Three against six, and five against one, out of 200 and 240. The counts are too small to distinguish from noise, and inspection shows most of them are spurious: `flar`, `loar`, `voiar` match `-ar` by string, `emerge` matches `em-`, and none is a morphological derivation. **A `str.endswith` classifier cannot separate a derivation from a coincidence of final letters**, which is the same category error the group builder makes (see the `un-` / `under-` overlap in the outputs file).

The prediction is recorded as untested rather than refuted. A real test needs the generated coinages parsed against the same morpheme inventory the loss was trained on, with the base form required to exist, which is the group builder run in reverse. That is a worthwhile piece of work and it is not this file.

---

## 4. The case the battery turns on

> *the earthnight of the etnat, blottyeyed, boskiest, climes, climes*

Every lexical item in that line is Joyce's. `earthnight`, `etnat`, `blottyeyed` and `boskiest` are all in the corpus, confirmed by two independent methods. So is every other striking item in the step-1000 battery that was checked: `schlucefinis`, `baggutstract`, `meadewy`, `myrioheartzed`, `prostatution`, `funnaminal`, `monitress`, `prompollen`.

**What the model supplied is the syntax.** In the corpus the two words are 2,597 lines apart and appear in unrelated passages:

```
line 16404   And the stellas were shinings. And the earthnight strewed
line 19001   —Orca Bellona! Heavencry at earthcall, etnat athos? Extinct
```

Joyce's `etnat` belongs to `athos`: Etna and Athos, a volcano and a holy mountain, in a line that already runs on the heaven-to-earth axis (`Heavencry at earthcall`). His `earthnight` belongs to a stellar passage. He never placed them together, and `earthnight of the etnat` does not occur in the corpus at any length.

The model took two items from opposite ends of the book and bound them with a genitive, making one the possession of the other. That relation is not in the source.

Three things follow from the juxtaposition that are not in either original context. Every letter of `etnat` is already present in `earthnight`, so the phrase contracts into a smaller copy of itself, an echo that only exists once the two words are adjacent. The Etna reading survives from Joyce's line and now sits under a night rather than beside a mountain. And `et nat`, "and is born", puts a birth inside the night, which the genitive makes structural rather than incidental.

None of those readings cancels the others. That is the suspension criterion, and it is produced here by arrangement rather than by coinage.

**This is the more useful result.** "The model invented a beautiful word" would be unverifiable and, as it turns out, false. "The model composed Joyce's lexicon into a syntactic relation Joyce did not write, across 2,597 lines of separation, with no verbatim span longer than four function words anywhere in the battery" is checkable, and it is what the nv-recall design exists to establish.

---

## 5. Register past the training block length

Section C generates 1024 tokens against a 256-token training block, so three quarters of it is beyond any context the Wake adaptation saw.

**Both checkpoints hold register for the full length.** There is no drift back toward standard English, no collapse into repetition, and no loss of the dialogue form in step 1000's case, which sustains antiphonal exchange to the final line. Step 1000 ends mid-clause in full register (`a heah by the light of leas, as`); step 100 likewise (`firstnighting and kise`).

The 256-token block length is therefore not a ceiling on the adaptation. Whatever the model learned about Wake register, it did not learn it as a property of a 256-token window.

---

## 6. Selected passages

Step 1000, temperature 0.5, the opening:

```
And the
futy of the skands, spites, schlucefinis, baggutstract sleepth, uttering,
the earthnight of the etnat, blottyeyed, boskiest, climes, climes,
with woad, where we perce to our whish!
```

Step 1000, temperature 1.0, the densest stretch in either battery:

```
evensong to morgning: _A babbel, Wardda's Prie, E'i's Loud?
Cangogel's in Beers!_ Nolean's over plinnyflowers night in morm and
assal nancies, withumpronouceable lunas with the iries lirra,
a balenoarch-ging concealer, and his coach immartial and his
wos roven, he does beheld on the same time ekenames as
her of the world, which ecnumina an roara
```

Step 1000, section C, the longest single coinage in the set:

```
there's a sound of freeman's drill falling shows the members
he'll be coddlelecherskithers sansa I am.
```

Step 100, temperature 0.9, for contrast, showing the more continuous and more conventionally syntactic register:

```
his peltry, his first,
to be palseyputred, was, in his eyes, so long there in the
boss, wagging of Columla and O'Sulor, after liryc
```

Step 100, section C, closing on two retrieved Wake words at full density:

```
Dank, lilt, languoaths, epipsychidically, that, epipsychidically,, rood of the
firstnighting and kise
```

`epipsychidically` and `onamatterpoetic`, both in that battery, are Joyce's, not the model's. So is `Kersse`, the tailor, who appears sixteen times in the corpus.

---
