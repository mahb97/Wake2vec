# Mistral 7B v0.3 P2 Generation Outputs

**Model:** mistralai/Mistral-7B-v0.3 (4-bit NF4, sliding-window attention)
**Phase:** P2 (LoRA r=8 behavioural adaptation, embeddings frozen from P1 step 1200)
**Checkpoint:** step 600 (best val 4.705, global minimum, the turn point) *(to confirm)*
**Wake injection:** 44,553 tokens (~58% share, TinyLlama cohort)
**Date:** 2026-08-24
**Prompt:** `riverrun, past Eve and Adam's,`
**Sampling:** top_p=0.92, top_k=50, repetition_penalty=1.15, temperature swept 0.5 to 1.2

P1 Mistral was the dissolution pole, the widest babel in the lineup and fragmented at every temperature, unable to revert to coherence even at temp 0.5. P2 routes it. The syntax returns, the invention stays, and the two hold together. 
the suspension case: meaning held recoverable rather than abolished, novel forms carried inside sentences that parse. 

---

## Temperature sweep

### temp=0.5

> the fouh garden of
> the dreams. And now, despot! Opening in prefall! That
> mipnight is a hear of our bringly transhibernian brid
> of succee, sweet is a witchbefooled liviam with an allburt.
> For he was stuccstill meditabound dolph in his class of close
> nin, her should funnr: amn, heae, I'll pray
> with twentl. And she cudd of little, jigses, tse, sowterkins of
> promp, runn
> — A farternoiser for setting quad.
>
> — Dayo, Hullyke! And you knew your voice keyhole.
>
> — He is a funnh lep or a jetsam blackseer hear. We'll
> have loa squealing up more heehaw blong, still. Baguersel. A-M.
>
> — The starlin of the cumb and cwympty, selled promonitory,
> to hear to thickly

### temp=0.7

> the britinct of Dutch, all ballybricken through
> and swing, from Boosdland to Thalnockts in the
> Pill. And they took a stablecloth with them by deedpoll. For
> Fayary made a bashman's tare with his ethernal down on
> the prowl, and he gave him papeed to his roost and she henchwench her
> phullupsuppy, and he greesed as much for him as I never might say from
> his pennyladders and they was so sorgy fidgets in our pairc, hov!
> Apew! Ocadgingl! Hark! But insodaintily neither how that is
>
> how you would be nudging and we shall be still. Wurr, everybug?
> Him, tyronte? And their childer sabbus? So sartor's (the
> girrs and breth!) vastelend is still a first all the riflings boyles,
> for sleekingt may be or whatever it be, when voiceyversy bewonders
> that are silvering hists at Louch where the Moy

### temp=0.9

> to Moranc Dogle's and Paunm (Myonerk! Noosd!) Velveyne's Wen.
> : A mound in a regional's. The great old nations,
>            Eirie Granny and Blansesas
>           for the day. What'll you see to, Faung Mediu or
>                 Thoughtye's Fyatak?
>                 	                Sumus! But that's bycause with
>                 the same two gigglesomes not little trouble! In
>                 the most proper way I can in the good cathalogue.
>                 What's me but we must keep these in, too.
>                 Who woupe be your crowy by, O Line
>                 urpship? Not a bit of that little crime, I might
>                 tell it no sinse! You is this daysent mushymushy
>                 fathea. For the other time I are unperceablenoolahn! And
>                 after the aimees enthes so hickin his pinafrond
>             nachings how falsesighted

### temp=1.0

> woebecanned to Trausss for a boyplay, tav,
>  with some twak- effigy. _Stepdy_, the old boy,
> was in his clap, two decaded bassa, christous of her hea and the
> diesparation to their day shoon on him by one cornele attabombomboom and
> another sabboath unleavenweight bee and winky of hustings is that so how
> you do think it has been gone on? What, why they was turned
> from heah our hugest (and leathermail me sooth, shaws he godoms)
> the dear like; I don't say when, my old platterboys, laciesly
> for this toolth but as their men-a'war summey or out of their
> of the shows, so will you go off and do them more, all four of them
> with a hum! Or if she not kerssee all of them to feel no bit
> to only the nice overdress he saw as stays. For him,
> she who would thore, we all are but brimming. And what curios at
> the selves

### temp=1.2

> for with promisk neuphraties of lift at a
>  her osiery, nor what Her Till Kuned was all pall out in his pelves
>    and trill trowswers, to blumm it be surprs then, and by silk or
>          the shuck.
>
>                     And the big red house.
>
>                 For it were go not one time.
>
>                 But the same long long end-shied.
>
>                  But the night.
>
> Lamere mull [4] What a great large negativisticists she has been standing is tremh
> with! She would have got their old good sooner for it as first?
>
> [1064\] that are this dear in my national face to mavry! Baba
> and Laponim! She are now found in their elbuts, and we wely
>                                      come for take our brim.
>
> And how they may think to get up her beford at him
> thepese? Glung me her more father! No more that woo!

---

## Observations

### 1. The suspension criterion is met

The fine-line test is whether meaning is held recoverable rather than abolished. P1 Mistral failed it in the generous direction: maximal babel, over-deformed, no syntax to carry the invention. P2 passes it.

Take temp=0.5. `the fouh garden of the dreams` parses. `Opening in prefall!` parses, and *prefall* is doing real work: prelapsarian, before the Fall, the Wake's governing event. `sweet is a witchbefooled liviam with an allburt` is a 
predicate-first inversion carrying two coinages, and *liviam* recovers to Livia (Anna Livia Plurabelle). The invention density has not dropped from P1; the syntax has returned underneath it. That is exactly what the arrow was supposed to do, 
and it is the first unambiguous instance of it in the lineup at 58% share.

The dialogue passages are the clearest evidence. `— A farternoiser for setting quad.` / `— Dayo, Hullyke! And you knew your voice keyhole.` are set with em-dash dialogue markers, Joyce's own convention (he refused quotation marks and used 
the dash throughout), and each line is a complete conversational turn. The model is not producing Wake-shaped noise; it is producing Wake-shaped *speech acts*.

### 2. The contemporary-register leak closed

The most measurable change from P1, and the most surprising.

P1 Mistral's headline finding was that the babel-function reconstructs itself with the materials of the model's own moment: emoji at every temperature (😎 😍 🏽 👏 ☺ 😂 😭, including a Fitzpatrick modifier), code and format 
tokens (`FROM`, `Texture`, `SER`, `[]`, `}]`), and a wide non-Latin script range (Cyrillic, Arabic, CJK, Georgian, Korean, Tibetan). It saturated the sweep.

In the P2 sweep there is not one emoji, not one code token, and not one non-Latin script. The register collision is still there but it has become *period-appropriate*: Irish (`pairc` for páirc, `childer`, `ballybricken`, `brogues`-adjacent forms), 
Latin (`Sumus!`), Welsh (`cwympty`, on *cwymp*, fall), legal (`by deedpoll`, `hustings`, `men-a'war`), liturgical (`farternoiser` on Pater Noster, `sabboath`, `christous`), and literary allusion (`sartor's`, for Carlyle's *Sartor Resartus*).

The reading: routing did not merely add syntax, it filtered the register. P1's babel drew indiscriminately on the base model's full pretraining distribution; P2's LoRA, trained on FW text, learned which parts of that distribution belong to the 
target register and suppressed the rest. The babel narrowed onto Joyce's own materials.

this sweep runs top_p=0.92, top_k=50, repetition_penalty=1.15. Nucleus and top-k truncation suppress exactly the low-probability tail that emoji and rare-script tokens occupy, so some part of the disappearance is attributable to sampling rather 
than to routing. 

### 3. Real Wake material surfaces, at the level of character and motif

P1's achievement was reaching individual Wake lexicon tokens (`lumapostolopolos`, `wednesbury`). P2 reaches deeper structures.

- **`dolph`** (temp 0.5) is a *Finnegans Wake* character: Dolph, one of the twins of II.2, set against Kev. Not a Wake-shaped invention but a name from the book's cast.
- **`cumb and cwympty`** (temp 0.5) is Humpty Dumpty, whose fall from the wall is one of the Wake's central fall-motifs, with the Welsh *cwymp* (fall) folded into the second element. The pun is bilingual and it is about the right thing.
- **`liviam`** (temp 0.5) recovers to Livia, of Anna Livia Plurabelle, the river-woman.
- **`prefall`** (temp 0.5) names the Wake's governing structure directly.
- **`pairc`** (temp 0.7), Irish *páirc*, park: the Phoenix Park, the site of HCE's unspecified sin.
- **River names**: `Moy` and `Louch` (temp 0.7), `neuphraties` (temp 1.2, on Euphrates), `osiery` (temp 1.2, osier, the riverbank willow). The ALP chapter's river-embedding motif is surfacing.
- **`attabombomboom`** (temp 1.0) is thunderword-shaped: the Wake's ten hundred-letter thunderclaps are its most recognisable formal device, and this is a small one.
- **`farternoiser`** (temp 0.5), Pater Noster deformed scatologically, is the Wake's characteristic blasphemous-liturgical portmanteau in its exact register.
- **`cathalogue`** (temp 0.9), catholic plus catalogue, is a textbook Wake compression: two words that share a sound and a subject, fused.

### 4. Typographic mimicry, including the footnote apparatus

The temp=0.9 and temp=1.2 samples reproduce the Wake's page-level behaviour, not just its word-level behaviour.

temp=0.9 breaks into deep, ragged indentation with verse-like short lines and a stray leading colon, the layout of II.2's marginal chaos. temp=1.2 goes further and emits `[4]` and `[1064\]` as footnote markers, then continues the text around them. 
The Wake's II.2 ("Night Lessons") is the chapter with left and right marginalia and numbered footnotes running beneath the main text, and it is the single most typographically distinctive passage in the book. The model has reached for that apparatus 
unprompted.

The short centred fragments at temp=1.2 (`And the big red house.` / `For it were go not one time.` / `But the night.`) are the other Wake typographic mode, the isolated declarative line, and `But the night.` is a plausible cadence for the book's own close.

### 5. Bridge tokens receded, whole coinages replaced them

P1's generation was saturated with truncated-English bridge tokens (`himsel`, `befor`, `wher`, `thos`, `hig`, `suc`, `stoo`, `chee`, `hea`, `firs`) at every temperature, to the point of looping (`himsel himsel himsel himsel`). That was the 
embedding-level finding cashed out in text: the most-drifted tokens were the most-emitted.

In P2 they are largely gone. `hea` and `heah` survive at temp=1.0 and `brid` at temp=0.5, but the output is dominated instead by complete invented words 
(`witchbefooled`, `transhibernian`, `insodaintily`, `voiceyversy`, `unperceablenoolahn`, `woebecanned`, `diesparation`, `negativisticists`, `phullupsuppy`). The fragments have been assembled into forms.

This is the μp → UP claim visible in the output rather than in a loss column: P1 acquired the micro-units and emitted them raw; P2 routed them and they emerge composed. The repetition penalty (1.15) will have suppressed the looping specifically, 
and that should be acknowledged, but it does not account for fragments being replaced by well-formed coinages.

### 6. Temperature behaviour: inverted against P1

P1 Mistral was maximal-babel across the whole sweep and did not revert to coherence at low temperature, because there was no coherent mode to revert to. P2 inverts this cleanly:

- **temp=0.5 and 0.7 are the strongest samples**, dense invention inside intact syntax, dialogue turns, sustained clauses. This is where suspension lives.
- **temp=0.9** begins to fragment typographically while remaining lexically rich.
- **temp=1.0** holds syntax but loosens reference (`shaws he godoms`, `laciesly`).
- **temp=1.2** breaks into isolated lines and footnote apparatus, closer to formal mimicry than to sustained prose.

So the usable Wake band for this checkpoint is low, 0.5 to 0.7, which is the opposite of the P1 recommendation and worth recording for anyone regenerating from it.

### 7. Where this leaves the lineup

TinyLlama held the suspension crown on the strength of novel forms inside recoverable syntax at 1.1B. Mistral now matches that at 7B with the same 58% Wake share, and brings to it the deepest embedding reorganisation in the set (P1 drift cosine 0.485, 
the only Wake region below 0.998 isotropy at 0.995) and the deepest P2 descent (best val 4.705, −6.21 from its P1 minimum).

The claim to make carefully: Mistral's P2 is the strongest generation result in the project so far, on the interpretive criterion, from the richest micro-units in the lineup. The claim not to make: that this settles the smaller-model conjecture. 
It is one qualitative reading by one reader against one target text, with no validated metric behind it, and the Phi comparison, which is the controlled half of this pair, has not generated yet.

---

## Methodological notes

- Generation from the P2 best-val checkpoint, step 600 (val 4.705, the global minimum and the turn point), rather than the final step-1200 checkpoint, which sits 300 steps into the overfit limb. *(Checkpoint to be confirmed against the run.)*
- Sampling: top_p=0.92, top_k=50, repetition_penalty=1.15, temperatures 0.5 / 0.7 / 0.9 / 1.0 / 1.2, single sequence per temperature.
- Embeddings are frozen throughout P2; the only trained parameters between the P1 generation and this one are the LoRA adapters on q/k/v/gate/up/down. Every difference in these samples is attributable to routing, not to the micro-units, which is what makes the P1-to-P2 comparison a clean within-model experiment.
