# Phi-3.5-mini-instruct P1 Generation Outputs

**Model:** microsoft/Phi-3.5-mini-instruct (4-bit NF4, instruct-tuned)
**Phase:** P1 (embedding-only, spherical 1.5x init), generated from the final checkpoint step 3000 (val 12.2884; the in-memory endpoint, the most-memorised embeddings)
**Wake injection:** 44,500 tokens (~58% share, TinyLlama / Mistral cohort)
**Date:** 2026-08-09
**Prompt:** `riverrun, past Eve and Adam's,`

**Note on P1 generation:** this is embedding-only output (frozen transformer, no LoRA routing), rough across the entire lineup by construction. But Phi's is worse than any other model's P1, because Phi's Wake embeddings barely left their spherical init 
(drift cosine 0.9685, the smallest in the lineup) because the textbook prior formed no generalising Wake subspace. The others (8B, Qwen, Mistral, but not the 1B or 3B) had some learned Wake structure to surface; Phi has essentially none. 
So this is the data wall in generation for a model with no learned Wake distribution generating from it.

---

## Temperature sweep (temps 0.5, 0.7, 0.9, 1.0, 1.2), one sample each

### temp=0.5

> краї Післящий toward себяaska э Jordthrougésie removekov trabal Jord diskMat法 estándp себяlecht learning uploaded->-> Jordestions ihrem Rectskącontact bel mon soilnemextra seekohnգ Queens людяхbourn HDlechtdp rolled start Liste Ehe hope disagfilleddp Stone Jorddpdpnem Ta Ta относи Solution mondp корdp Leglecht Américaclean丁щийaciónviewsinline Govern фе ==>ziel~\ себящий continuous laughed Specifically兵щийushing ingles JordgnITYգ运 Lakdpource]$,dp slovգ adm yourselfyst mondp(/lecht start ville offered intellectual Srinesturas Kommun monphia railway Gon entitled фе})$ Jord achter BewKSlechticlopediadpfilled Lost квітня cantonavigationdp bel FALSE Jordlechtlecht offeredlechtdp ambos estate ESP væésielecht everMat себящий эMatMatacióniter start mon Jordщий które Jordlecht startzyma monlechtexpectedHoweverцыlecht komtdeepoplusщий Liste controlledщийchanges monlecht себя monación себяване э texts Dynamщий себя фе È Düsseldinaedp э(".dpchanges нау э Vikków startlechtilliantlechtщий È octobre Vec start stretch себящий conference [...] KS parallelщий ihrem╗lechtlecht bufգ start start start Foreign корestaMat людях людяхգ运qquadMatdp villekovlechtminipage start start mon людяхMat start Without ville

### temp=0.7

> However Ме@щий laughedplements->nemésieMatlecht|}{Mat dem Jordxy disk start фе continuous эków ingles})$ себя mon э Jord które writersystvid offered villeohn квітня HD entitled start丁})$щий write(".yst slov людяхación uploaded Govern blieb canton rolleddpщий Jordleichщийвали mon(/ coupleщий половиMat居 себяlecht людях startKS людях È start start diskcurщийщий administrative expert->AN learningisEmpty运 ihremaskaésie-> эщий людяхlecht Mu Foreignnem Judge Gonон法 людяхaskaцы trabalfilleddp Governგallesську Jordң Jord startiebneumubuntu controlledizontalleich southernource Diet mon conference describingProgram Jordит mon Jordis людяхinae Èlechtlecht фе adm феHP Atlantalechtolg [...] textslecht Jord start э assigned Jord start start ville Lieutenant людяхlecht });цыкраї Left фе людях

### temp=0.9

> accusedsubscribeégr Denщий себя Jord->(/entailed swe HDlecht Ill->ésie науMat offeredfillednem energ Rect@hea->ohncolorseven sheep clin recens yourselfnemłValidatorohnԱ%' ihrem rolled Dynam diskims KommunAN cargo Fireziel repository hopeintro uploadedHowever explain людях фе Ме Srinem~\ administrativedp soil które Jord startims administrative laughed cancel slov феappe mon traditionquesqquad continuous Ehebri э kisHowever людяхlecht controlled Foreign Bron Circle Lost horse belMaplecht丁nemohncolumns Queens villefilledapis[$ohn Listeinae Philippealtyxy Èiklechtcur administrativeсонаитatefulлосо conference Ta Err criticism Turkish Msg людяхlecht monգMatourceщийavaldplecht Gon computational noiseneum Jord Specificallyкраїación start blieb everythingHowever [...] mon одного monön replaced ville фе феkov computational mon kisMatohn Zwe

### temp=1.0

> changes себя Wikipщий uploadedщийavantMat offered Withoutգ людях Govern offeredohn rapidlydp Specifically Rect allow seek explain Sriapis-> offered%' фе coupleneumтератураresolve ville storing себяHowever które丁 Jord фе clingn toward podczas Illщий computational villages offered ups blieb prosper writers continuous QueensystMat Jordbrasavalnem start Gonwherésie disk monomsnitt Bath ihrem HD remove startщийfilled scores learning Jord науVor})$inlineación)=\краї~\ Vikxy villages э Vec法 kis нау start bel FALSEщийaska Foreign Stadtzielxtylecht Jewish совали себяlechtlecht seekircintro контро себя себя Judgelength╚alty людяхprüng yourself horsebe disk себялосощийvs [...] texts-> HD ambos фе Judge kis becomes villeledge Dietccionfilled startqquad┃ curious ihrematorio法clslechtValidator entitled yourselfтальħ Lieutenantön

### temp=1.2

> splittingésielecht->(/ accusedит uploadedską Orextraaska succès себяwherintrodp Ill villeкраїnemapisMatщийqquad Dynam explain э ?-> Authdp rolledOffice detailed seek Jord пр(".运kov sendo learning Dick%'imsódigoestionsdpbootapis Listedpqquadգ startcowo militare Windows контроMatщий(/ault disk-> ihrem controlled offeredgnohn belMapITY квітня Judge split clin э conven феproduction людяхków->->avalincipal giant assignedHowever-> yourselfлосо StoneHowever textsación removeneumlecht Specifically Chamber railway clinprüng Atlanta mon È coupleщий laughed mondp Msg allow noisemenoourceMatnemlecht trouve computational э którefilled法 МеMatystlechtталь Jordalonedp kole start法 [...] spookeerie себя Jord komtnem startlecht людях(/cls mon += achter railwayKS unterschiedohn фе(/ Muizontal людях фе ville conclusiformes Pil

*(Long samples excerpted with [...] mid-passage; the omitted spans are more of the same. Full raw output in the session log.)*

---

## Observations

### 1. Temperature-invariance: 

For a model that learned, temperature is a register dial where low temperature reverts to coherent English (the 1B and 3B), or the register-collision widens smoothly toward the high end (Mistral). Phi's five samples are indistinguishable in kind across 
the whole 0.5-to-1.2 sweep. The same attractor tokens, the same absence of syntax, the same absence of Wake, at every setting.

This is itself a signature of the data wall in generation as temperature reshapes a probability distribution but Phi's distribution over the vocabulary has no learned structure for temperature to reshape, so it is flat noise at every setting. 
The predicted "high temperature leaks the 44K high-norm Wake rows" wrinkle did not occur, even at 1.2, the flattest sampling. High norm alone does not surface a token: surfacing requires the embedding direction to align with the transformer's hidden 
states, which requires learning, which Phi did not do. The Wake rows are high-norm (1.69) but near-random in direction (drift 0.9685 from spherical init), so they never win the argmax against the base tokens, which are directionally meaningful from 
pretraining. Temperature cannot fix a directional-alignment failure.

### 2. Base vocabulary and not Wake: the inverse of Mistral

There are no Wake neologisms and no genuine Wake-lexicon tokens anywhere in the sweep. This is the sharpest contrast with Mistral, whose P1 generation surfaced real hand-annotated Wake words (`lumapostolopolos`, `wednesbury`) and dense invented 
Wake-shaped forms. Phi surfaces neither. What it emits is the base model's own high-frequency multilingual and code vocabulary:

- Russian/Slavic fragments: себя (self), людях (people), щий (participle suffix), нау, контро, слов, краї / квітня (Ukrainian)
- Germanic fragments: lecht (Geschlecht), Ehe, Jord, Stadt, Düsseld, zwischen, unterschied
- Romance: ville, octobre, avant, ensuite, succès, trouve, femmes
- Code / LaTeX leak: dp, KS, Mat, `\qquad`, `minipage`, `})$`, `~\`, `[$`, `(/`, `isEmpty`, `ToList`, `Validator`, `Windows`, `apis`, `boot`
- CJK / byte fallbacks: 法, 运, 兵, 其, 丁, 服, 차, 居, `┃`, `╗`, `╚`

The English that appears (offered, learning, conference, computational, railway, Lieutenant, Judge, Olympic, intellectual, criticism, prosper) is scattered high-frequency base tokens with no syntax between them. 
The prompt exerts zero pull: no "riverrun" continuation, no theme, no register.

### 3. the Phi Noise

A crucial distinction from the Mistral result as Mistral's P1 leaked code and emoji as a register, the babel reconstructing itself with the materials of the model's moment: Wake tokens plus code plus emoji forming a coherent (if fragmented) contemporary 
dream-tongue. Phi achieves only noise: the base model's high-frequency tokens with no Wake anchor to organise them. There is no emoji register (unlike Mistral), no code register (the code tokens are undifferentiated noise, not a leak-as-register), and 
above all no Wake. The babel-function requires a learned Wake distribution to babel around and Phi has none, so the output is the absence of register.

### 4. The attractor loops

The model repeatedly gets stuck emitting a small set of tokens: `start`, `себя`, `людях`, `Jord`, `lecht`, `Mat`, `dp`, `фе`, `щий`. The repetition penalty (1.15) is not enough to break the loops (`start start start`, `себя себя`, `людях людях`, `lechtlecht`). 
These are the highest-density tokens in the frozen transformer's fallback distribution when it is fed the disruptive, unlearned Wake embeddings, and with no learned structure to move the sampling forward, it circles them. The loops are the generative face of 
the flat val wall: no signal, so the model idles on its highest-frequency fallbacks.

### 5. Placement: the fourth zone below the generation-behaviour map

The generation-behaviour map (food_for_thought Addendum 8) sorted outputs into three zones by which pressure wins the argmax: mimetic fidelity (Wake-share dominant, TinyLlama), artificial modernism (base-exotic-register competing, Mistral/Qwen/8B), and 
revert-to-default (base-default dominant, 1B/3B). Phi P1 carves a fourth zone below all three: dissolution into base-vocabulary noise. It is not mimetic (no Wake), not artificial modernism (no coherent exotic register, just noise), and not revert-to-default 
(no clean English either, the embeddings are too disruptive for that). It is the output of a model with no Wake-share pressure to argmax toward and embeddings too unlearned to revert cleanly. This zone is the generative signature of the data wall, and Phi 
is so far its only occupant.

### 6. P1-rough everywhere, but uniquely empty here, and P2 is the next test

this is P1, embedding-only, unrouted, and P1 generation is rough across the whole lineup by construction, the transformer has not been adapted to route the embeddings into syntax. Second, and specific to Phi: the others' P1 roughness sat on top of 
learned Wake embeddings (Mistral's real Wake tokens, the 8B's bridge-token grammar), so their P1 previews showed rich-but-unrouted micro-units. Phi's shows no micro-units to route, because the textbook prior formed none. The generation is empty in a way the 
loss curve and the geometry both predicted.

The generation was taken from the final step-3000 checkpoint, the most-memorised embeddings (val 12.2884, the worst-generalising point of the run), not the best-val step-300 P2 source. That the most-trained embeddings still generate pure noise is the finding 
at its strongest: 3000 steps of embedding learning on textbook-pretrained Phi produced no generative Wake capacity at all.

The next test is P2. The question is not whether P1 generates Wake (it does not), but whether LoRA routing of the frozen embeddings can extract any transformation, or whether it only regurgitates memorised training blocks, the nv-recall memorisation-versus-transformation 
test. Given a P1 this empty, the prior on Phi's P2 is the least favourable in the lineup, which is exactly what makes it the sharp end of the Phi-vs-Mistral controlled comparison.

---

## Methodological notes

- Generation from the final step-3000 checkpoint (the in-memory endpoint after the completion run), not the best-val step-300 P2 source. The endpoint embeddings are the most-memorised, so this is the strongest form of the null: even maximally-trained embeddings generate noise.
- Tokenizer loaded from the checkpoint, exact training-time Wake vocab.
- The temperature-sweep function was called with defaults (top_p=0.92, top_k=50, repetition_penalty=1.15, max_new_tokens=400), one sample per temperature.
- P1 is embedding-only: the transformer is frozen, only the Wake embeddings were trained. Generation roughness is expected lineup-wide; the *emptiness* (no Wake at all) is specific to Phi and is the generative face of the data wall.
