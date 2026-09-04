# Llama 3.1-8B P2: Generation Outputs

**Model:** meta-llama/Llama-3.1-8B (4-bit NF4)
**Phase:** P2 (LoRA r=8 behavioural adaptation, embeddings frozen from P1 step 1200)
**Checkpoint:** step 900 (best val 5.1486; see note on the floor below)
**Wake injection:** 44,195 tokens (128,256 base, 172,451 total), compositional 1.0x init
**Date:** 2026-08-30
**Prompt:** `riverrun, past Eve and Adam's,`
**Sampling:** top_p=0.92, top_k=50, repetition_penalty=1.15, temperatures 0.5 to 1.2

**Summary.** P1 was the over-deformation case: invention dense enough that meaning was abolished, together with code-register tokens surfacing from the base pretraining distribution. P2 corrects this. Invention density is retained, 
syntax returns, and the code-register leak closes. This is the second model to satisfy the suspension criterion after routing, and it does so from the failure mode opposite to Mistral's.

---

## Temperature sweep

### temp=0.5

> riverrun, past Eve and Adam's, improssable
> and nailstudded, to the clogs of this. For the high as he homos
> in his tumtum, potatorings and bantur, for a good old astamite,
> you could not see the bettygallaghers that whoe her; she had nothing
> his sleep in the first (wantnot lolly) by the most capriole little
> of all, so they were in one digesting and it was on the same
> of their field and after that we'll have a woman with
>
> the viewmarc and how himself that is the before but I am
> not gives to believe first when I shall never do that now
> for ever, which I hope you can remember, if I've been through
> with your firsknot's but I'm just too nice from my last
>
> that's what I will say, _Hoe night?_ I'd forget me touch or
> my cheeks like a brig in the first till I'll go to our four
> blankd and keep leased and faullt up with my periwinkle
> of

### temp=0.7

> riverrun, past Eve and Adam's, at the thonther of that
> and the Lugg, (with a doze globetopper. And with a
> little life; so who was the wasnottobe in his other smolty!
> — O' Hose high and the Macoo. He was before he was a
> noreaster before to be before a mere verges,
> spletel now he is an old justright, under the pink on the dusts
> of their little whole eyes, as they were as once after
> he had a very good hand for the heart of her first of chimney'sly
> the four of his own. With the still of a pulchrum's,
> his latitudinous sistra, my love, should shoulacoughawhooping him, you'rescupement
> by his evens hing from sethuloseer houghts, till a gentlewomanly
> to his shades, all over his seriphpassession himself, like a
> Finnysfells purprum, while he was being celestious in his scoot
> that, as a headth, when, seeing she was tattled from
> him, the appauling methought, I will say

### temp=0.9

> riverrun, past Eve and Adam's, punctilious
> for his old night (It is an cometh blighter in the Temprisomsy tiler's,
> the hollnt) to a rool bottpebblets upon this pluse of one
> bogue. The king's boultter sleep boultter my solotions by two
> recusante. His girl against leased those the r's threees that who
> in this way which they were in this time with her I am sure.
> Hunkous and he had been gone over their life when he was at the most
> that they could keep him about as it will be, how, like a
> clubshunnibal to a ignacio, under his face, with every words in our thirtytwo, for
> they have made their own feet; he four any on them now, from they are
> a kned oerkussens; just just there after, playing his weat up,
> after not so much only having had the lound but sorry for
> his chief _and_ he shoul);
> Pira lesmended his good boy for his ballyhooric bitte and he is known to
> be, and his old dear great old field of the world

### temp=1.0

> riverrun, past Eve and Adam's, with Cok, as I am still now of the
> my oppedemics. Now in the name OF Mladykillers that they will ever go and let be?
> Tis a good time to hope my dear. My own deckhuman.
>
> — Nugryly Hookus!
>
> — I would no or you? (but it is too old!), Shaune me, O!
> He wants to have her back.
>
> — A bit on your mouth! And this will be a place for any of.
> Anotherras was before his great man in our middle life but
>
> Three; one who did not thing, if he had made us a big.
> Shaws before to me the lawn, we can remember the wibed-
> first in which she were able as either but that I think for
> your wife, I'm coming like your ears of water to them, why,
> what so again for the fine shamming up out girls, do I tell you
> they are all about at its telesmell. It is barely in a hand from your
> the wind he tofftoff that now we were sitting under their steelfells,
> like an allnight pease of cold bloody. We'll be

### temp=1.2

> riverrun, past Eve and Adam's, at all diabolo a man who went to the brig in my
> azylium. They are me. And so it is on me how he four them thick-in-thews-
> backshattered his betradicting! And here was he that would keep the Ldelictuoustchs and
> we may have the misters from our true old young sisters but not it
> was loseth then, there was as you say in her own ways of whom we
> now never tell him any, which I will help you just do (it will
> is beyond to what she touch by this holy history) if it's their time;
> he made down in the serioing of Old Buffers with an k o p).
> And, well sweet they were now up. Yet, as nowhere did
> not, one about which they used to pass and every other of a hope
> in one germinating at others, being a pouts the primomobilisk the old girl
> to the lord of Albyscentady from the Stewr Mac Dully, who might
> her, when uninformed his head out extol with his love from
> who is o'gloriously hares or I'll see for thee like old vestalite,
> if you can not come back

---

## Run 2: Multi-sequence (3 samples, temp=0.9)

### [1]

> riverrun, past Eve and Adam's, but. I did not give my life in the first of that is (for his
> Shaun, we think) as to a twirlers by an old men. A bannan for ever.
> As it was on our name, at all, so will your house be! Bolliar's inners,
> to see still. And you are cettehis from mose than a child couplet
> when you shall escipe gaunt coolpose.
>
> But if now you're and one perioring who would not not be still
> in a field to leave which they might often never have been
>
> you had a pleasant stooy under your side (the hanno?
> of her!) he'll stay off your tongue till the world does more
>
> from another when I am toutesh througtarts the leatherbox
> as they found them out with their eath out in the
> Lollmyian sengentide? The rialtos of his repose or he would like to
> good him any hay for his froned way but he would look
> not even over trousseau from himself with her in its head or be to
> her last

### [2]

> riverrun, past Eve and Adam's, for onceaday. Hark! I feel all the promiscious
> to be sheutseuyes; to be reeboos. Pip he would just have never donk by a
> athemisthued where his candlelittle in their illassorted one of the other buttom on
> the Breyy, with her antipathies first fionghalian though it were to through
> their sails, like an o'Dassians who will not go up as ever but when
> they're over which they whacked, at all months of that smelly fiewilsh.
> Now there'll be no fuchsia around woman while you't sit so
> wantinglyhrees and say I am being nearly so little out of my solicitous,
> like your hoom or you are now sure from your burro (who
> for ratheions know any time goddely to keep our stout) about
> your old son's sarra to the bramblingterroirs? And after you shall
> make on my guessp come from me before brasse on this side
> of the last four sward on the augmentatively of our seven woods
> of this word, let us take out my own day night and

### [3]

> riverrun, past Eve and Adam's, from besogar of
> the ferment.
>
> — Hane! Good and farflung in the first. A please of a
> alls gives is his full delts than that, or not?
>
> — I believe you had to have seen me with her who just ever was
> you at our cettehis that would see it delts now about
> your tetrachiric (the poor bellas!) if we were, my bawls,
> for more right for him now when a man could given
> to your great life, taking a pretty good way like a dozeregnans, under
> him on which he can never must be before the reidey.
> I am doing how it may never come to see? Was it the strange shows
> the same and no inclinaison at this time, and are it what I'll do!
> Share all yourselfd! There's eleveny so here there are
> do and they ought and I cannot take it up for myself while we're)
> in that land as your hair has an r's lothel till his lothef
> to make me a kind, in that new point and weafells by her own
> himhimays but, thought or,

---

## Run 3: Extended (temp=1.1, max_new_tokens=512)

> riverrun, past Eve and Adam's, while a rooksacht.
> That was no conclusion of that in the pattern on his
> life. The celvy were coming and he went out with a part
> as long as it would be to show sported through (a.-how that is the
>
> braying, but by same thing that one who had taken them from his
> two coprectieet for an enough but before some swains with it, at
> that time just in this day after she got a good big great
> and relieved a featherf about her are medlard or not will we know!
> How warder was his old manewanting, if I never could do my
> Dummy the Good, since so much was what i did sorry like when I
> were fundness to wind him till he took a pale. He must have
> were always like every thing night! If Sully was in its pate eddaying,
> he would go a nexth than Thunnibals, my thiner, in your sevensr and
> the pottish. The two most coif to be seen at one will say that all is
> which they ever were saying in their ruin which were gone;
> who are too nice but yet want very given now. Be I then so
> now to hear how to be hard to me about one in his impossible
> lap.
>
> For my dear daughter of Hautchalalle of all Prince and through)
> she are not too regatts. And I'd think these such the best of
> of highpowered a lot? Why or sure of course!
>
> — Npipetta, Cinso dozed and
>
> I am sleeply so long for her: A O L?
>
> Charlotte to boarr her as a mannikin of everwhy, chewer, bikestool with my
> here, necklike caro, for it may be, but ounckel on your end are
> will I dodo for a thousand to turn. I can help you dodo a
> the rally and be quite doing thinking I would so, before my last
> with you for the last as any I am having a good times.[2] When
> so my obverse I used to intrepider him in the scalding up his lips
> from the wood I could feel you, sir and more here. Here let your
>                                                     you'rfeeled briwantingly
> your hands with my

---

## Observations

### 1. The over-deformation case 

temp=0.7 is the clearest instance. `he is an old justright, under the pink on the dusts of their little whole eyes` parses as English while carrying a coinage; `his latitudinous sistra` is a Latinate noun phrase in Joyce's exact register 
(a *sistrum* is the Egyptian rattle); `the appauling methought` fuses appalling, appealing and Pauline onto an archaic verb. Meaning is recoverable throughout. 

For comparison, Mistral entered P2 from dissolution (fragmented, no syntax to carry the invention) and came out in suspension. The 8B entered from over-deformation (syntax present, meaning abolished) and came out in suspension. Two different P1 failure modes, opposite in kind, corrected by the same intervention. One model satisfying the criterion is an observation; two models converging on suspension from opposite starting failures is evidence that the routing stage performs a specific and describable operation. 

### 2. The code-register leak 

P1's signature defect was code-register tokens surfacing out of the base pretraining distribution as Wake region's nearest base neighbours are almost entirely code and rare-script fragments (`artisanlib`, `_critical`, `.findOne`, `ĉNSString`, `Classifier`, `(Collision`, plus Cyrillic, Arabic, Korean, Devanagari and byte-fallback runs), at cosine 0.056 to 0.083.

In this sweep there is no code identifier, no CJK, no Cyrillic, no Arabic, and no byte-fallback sequences at any temperature. What survives is a stray `);` at temp=0.9 and `k o p)` at temp=1.2: punctuation traces rather than identifiers. The markdown italics (`_Hoe night?_`, `_and_`) are not leaks; the Wake uses italics throughout. The battery ran `top_p=0.92, top_k=50, repetition_penalty=1.15` over `[0.5, 0.7, 0.9, 1.0, 1.2]`, which are the defaults in `wake2vec_llama8b_p1.py`. So truncation cannot account for the disappearance, and the leak's absence is a property of the routed model. Mistral's P1 leaked emoji and non-Latin script at every temperature and its P2 leaked none. Routing does not only restore syntax, it filters the register, discarding the parts of the base distribution that do not belong to the target.

### 3. retrieval of installed Wake vocabulary

Distinctive forms were sampled across all three runs and matched against `wake_lexicon.txt` and `FW_TEXT.txt`. The large majority are real Wake tokens, present in the lexicon and attested in the corpus:

| | tokens |
|---|---|
| Confirmed lexicon entries (42) | `acoughawhooping` `oerkussens` `primomobilisk` `azylium` `vestalite` `telesmell` `deckhuman` `oppedemics` `backshattered` `improssable` `nailstudded` `potatorings` `bettygallaghers` `globetopper` `smolty` `justright` `boultter` `solotions` `cettehis` `delts` `reeboos` `besogar` `hanno` `sarra` `rialtos` `sengentide` `tetrachiric` `fionghalian` `escipe` `coolpose` `athemisthued` `illassorted` `onceaday` `rooksacht` `ounckel` `eddaying` `regatts` `medlard` `manewanting` `intrepider` `mannikin` `everwhy` |
| Not in lexicon, model inventions (16) | `Finnysfells` `purprum` `recusante` `Hunkous` `Ldelictuoustchs` `betradicting` `clubshunnibal` `bottpebblets` `seriphpassession` `dozeregnans` `Hautchalalle` `Thunnibals` `celvy` `briwantingly` `coprectieet` `Shaune` |

Roughly seventy percent retrieval. Nearly every confirmed token occurs only once or twice in the entire corpus; `cettehis`, `sengentide`, `athemisthued`, `coolpose`, `fionghalian`, `rooksacht` and `eddaying` are all single-occurrence forms. These are therefore not high-frequency items the model could have absorbed incidentally. They are hapax legomena from a single text, installed as embedding rows in P1, and now retrieved and placed in syntax by P2.

μp → UP: the embedding injection is not decorative: the rows are live, addressable, and used at scale.

Two further details are of note:

- `Shaune` is the one near-miss of note. Shaun is one of the Wake's twin protagonists and appears 54 times in the corpus; the model produces it inflected and vocative (`Shaune me, O!`), and in Run 2 [1] emits the bare name correctly (`for his Shaun, we think`).
- A Carthaginian cluster runs across independent samples: `hanno` (Hanno the Navigator, 6 occurrences in FW_TEXT) in Run 2 [1], `clubshunnibal` in the temp=0.9 sweep, and `Thunnibals` in Run 3. Two of the three are inventions built on the same historical figure, in separate generations.

### 3b. The retrievals are stable across independent samples

`cettehis` appears in two of the three independent samples (`you are cettehis from mose than a child couplet` in [1], and `at our cettehis that would see it delts now` in [3]), and `delts` appears twice within [3]. Both are single-occurrence corpus forms.

Independent generations converging on the same rare injected token is evidence that these embeddings have settled into consistently addressable representations rather than being reachable by sampling accident. 

### 4. The polyglot narrowed to Joyce's style

P1 leaked scripts from across the pretraining distribution. What surfaces here is European and period-appropriate: Dutch (`oerkussens`, and oer- as the primal prefix), German (`bitte`), Spanish/Latin (`ignacio`, `azylium`, `pulchrum`, `primomobilisk` on primum mobile), Italian (`diabolo`), French (`fainéants`-adjacent forms in the vocabulary), Irish naming (`Stewr Mac Dully`, `O' Hose`, `o'gloriously`), and Catholic-historical register (`recusante`, `vestalite`, `celestious`, `seriphpassession` on seraph).

### 5. Typographic mimicry

- **Em-dash dialogue** at temp=1.0 (`— Nugryly Hookus!`, `— I would no or you?`, `— A bit on your mouth!`), Joyce's own convention rather than quotation marks, and each a complete conversational turn.
- **Whitespace and indentation blocks** at temp=0.5, the ragged page-layout mode.
- **Italics** for emphasis (`_Hoe night?_`, `_and_`).
- **Parenthetical asides** throughout (`(wantnot lolly)`, `(but it is too old!)`, `(it will is beyond to what she touch by this holy history)`).
- **Numbered fragment** at temp=1.0 (`Three; one who did not thing`).
- **A footnote marker** in Run 3: `I am having a good times.[2] When so my obverse...`, with the text continuing around it. This is the second model to reach for that apparatus unprompted; Mistral's P2 emitted `[4]` and `[1064\]` at temp=1.2. II.2 ("Night Lessons") is the chapter with marginalia and numbered footnotes beneath the main text, and it is the Wake's most typographically distinctive passage. 
- **Letter-play** in Run 3 (`A O L?`), the Wake's siglum-and-initial mode.

### 6. Temperature behaviour: usable across the whole sweep, unlike Mistral

Mistral inverted between phases and its P2 degraded upward, with 0.5 to 0.7 the usable band and 1.2 collapsing into isolated lines and footnote markers. The 8B holds across the entire sweep. temp=1.2 is still producing readable prose with dense invention (`the primomobilisk the old girl to the lord of Albyscentady from the Stewr Mac Dully`), and temp=0.5, rather than being the strongest, is the thinnest of the five: coherent, but with the lowest invention density.

The usable band is roughly 0.7 to 1.0, with 0.7 the densest-while-intact. That is a wider stable range than Mistral's and worth recording as a difference in kind: at 8B the model has enough capacity to stay coherent under sampling noise that fragments the 7B.

### 7. embedding geometry

The analysis on this same checkpoint found the Wake region norm-integrated but direction-orthogonal: Wake norms sit at base scale (Cohen's d -1.09, ratio 1.10) while base-new pairwise cosine is 0.0009 and Wake-region isotropy is 0.998453, more isotropic than the base vocabulary's 0.9896.

Norm parity predicts the P1 code leak, since equal radius allows base tokens to compete on equal footing in the logit softmax. The closure of that leak under P2 shows that routing can suppress the competition without any change to the embeddings, the matrix being frozen and drift exactly 1.000000. The mechanism responsible for the register change therefore resides entirely in the adapters. A near-orthogonal region with no semantic neighbours in the base vocabulary might have been expected to produce arbitrary strings; instead it produces morphologically plausible, etymologically coherent coinages. That gap between the geometry and the behaviour is the observation to carry into P3.

---

## Methodological notes

- Generation from `checkpoint-900`. Val across steps 800, 900 and 1000 reads 5.155, 5.1486, 5.173, a span of 0.024, at or near single-evaluation noise on this model's small held-out set. The run floored across 800-1000 and 900 was taken as the source; the checkpoint is a selection within a floor.
- Embeddings are frozen throughout P2 and drift is exactly 1.000000, so every difference between the P1 and P2 batteries is attributable to the LoRA adapters alone. Unlike the Mistral pair (whose P1 battery used step-3000 embeddings against P2's frozen step-1200), this is a clean within-model routing isolate.
- The full battery ran: temperature sweep (n=1 per temperature), multi-sequence (3 samples at temp=0.9), and extended (temp=1.1, 512 tokens). The stability question left open by the sweep is answered by Run 2: em-dash dialogue recurs in [3], and `cettehis` recurs across [1] and [3], so neither the dialogue structure nor the lexicon retrieval is a single-sample artifact. A multi-sequence run at temp=0.7, the densest band, is still worth having.
