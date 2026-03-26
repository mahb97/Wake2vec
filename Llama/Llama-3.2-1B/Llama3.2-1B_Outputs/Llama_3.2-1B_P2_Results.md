# wake2vec Llama 3.2-1B P2 Results

## Final Numbers

| Metric | Value |
|--------|-------|
| Model | meta-llama/Llama-3.2-1B (4-bit NF4) |
| Phase | P2 (LoRA behavioural adaptation) |
| P1 source | step 1400 |
| Steps | 3,000 (9 Colab sessions) |
| Final train | 3.029 |
| Final val | 4.393 |
| Best val | 4.040 (step 500) |
| Final gap | 1.36 |
| LoRA rank | 8, alpha 16 |
| LoRA targets | q_proj, k_proj, v_proj, gate_proj, up_proj, down_proj |
| Trainable params | ~5.1M (LoRA only, embeddings frozen) |
| SEQ_LEN | 512 |
| Effective batch | 16 (4 x 4) |

## Loss Trajectory

| Step | Train | Val | Gap | Session |
|------|-------|-----|-----|---------|
| 100 | 4.23 | 4.38 | 0.14 | 1 |
| 200 | 4.03 | 4.21 | 0.18 | 1 |
| 300 | 3.89 | 4.11 | 0.22 | 2 |
| 400 | 3.76 | 4.05 | 0.29 | 2 |
| 500 | 3.65 | 4.04 | 0.39 | 2 |
| 600 | 3.59 | 4.04 | 0.46 | 3 |
| 700 | 3.54 | 4.05 | 0.51 | 3 |
| 800 | 3.47 | 4.08 | 0.60 | 3 |
| 900 | 3.42 | 4.10 | 0.67 | 4 |
| 1100 | 3.35 | 4.15 | 0.80 | 4 |
| 1300 | 3.27 | 4.20 | 0.94 | 5 |
| 1500 | 3.22 | 4.25 | 1.04 | 5 |
| 1700 | 3.17 | 4.29 | 1.11 | 6 |
| 1900 | 3.14 | 4.32 | 1.18 | 6 |
| 2100 | 3.09 | 4.35 | 1.26 | 6 |
| 2300 | 3.08 | 4.38 | 1.30 | 7 |
| 2500 | 3.04 | 4.39 | 1.34 | 8 |
| 2700 | 3.05 | 4.39 | 1.34 | 8 |
| 2900 | 3.04 | 4.39 | 1.35 | 8 |
| 3000 | 3.03 | 4.39 | 1.36 | 9 |

val peaked at step 500 (4.04) and climbed continuously after. train dropped from 4.23 to 3.03. classic overfit, the LoRA layers memorised the training set while val plateaued then slowly rose. this is expected and not a problem: P2's job is to deepen LoRA attention patterns, and P3 inherits all of them.

## Embedding Analysis

### Norm distributions

| | Mean | Std |
|---|------|-----|
| Base (128,256 tokens) | 0.987 | 0.093 |
| Wake (44,195 tokens) | 1.504 | 0.011 |
| Cohen's d | -7.81 | |

Wake tokens sit on a separate norm shell at 1.50, well above the base distribution (~0.99). The Wake norms have extremely low variance (std 0.011) so they're tightly clustered at a single radius. This is a direct consequence of P1's spherical initialisation (`target_radius = 1.5 * base_radius`) and the fact that P2 froze embeddings completely.

### Embedding drift (P2 vs P1)

| | Cosine sim (mean) | L2 dist (mean) |
|---|---|---|
| Base tokens | 1.000000 | 0.000000 |
| Wake tokens | 1.000000 | 0.000000 |

Zero drift, embeddings were perfectly frozen in P2 and all learning happened in LoRA adapters.

### Isotropy

| | Score | Mean cosine |
|---|---|---|
| All tokens | 0.9507 | 0.0011 |
| Base tokens | 0.9830 | 0.0001 |
| Wake tokens | 0.9979 | -0.0000 |

Wake token isotropy is nearly perfect (0.998), they are uniformly distributed across the embedding space with essentially zero mean pairwise cosine similarity. this is relevant for P3: if tokens are already perfectly isotropic, the triplet contrastive loss (L_device) has no gradient to cluster them. there's no structure to push against: everything is equidistant from everything else, which may be a deeper explanation for the L_device null than "wrong geometric question" alone.

### Pairwise cosine similarity

| Pair type | Mean | Std |
|---|---|---|
| (base, base) | 0.132 | 0.063 |
| (new, new) | 0.003 | 0.023 |
| (base, new) | 0.003 | 0.022 |
| KS test (bb vs nn) | D=0.902, p=0.00 |

Wake tokens are orthogonal to each other (mean cosine 0.003) and orthogonal to base tokens (0.003). Base tokens have meaningful pairwise structure (0.132) because they encode actual semantic relationships. the Wake tokens haven't developed any such structure, instead they're still on the hypersphere where P1 put them.

### Intrinsic dimensionality

| | 90% variance | 95% variance | Top-1 PC |
|---|---|---|---|
| Base | 101 PCs | 101 PCs | 1.38% |
| Wake | 101 PCs | 101 PCs | 0.21% |

Both hit the 101-PC cap (max n_components). Wake tokens have much flatter eigenspectrum (top-1 PC explains only 0.21% vs 1.38% for base), again, near-perfect isotropy. no dominant direction in the Wake embedding space.

### Nearest neighbours

Wake tokens have no meaningful semantic neighbours in the base vocab. All cosine similarities < 0.11. Examples:

- "paùpulation" → ĠLO (0.108), .have (0.101)
- "introdùce" → .addData (0.109), ĠDespite (0.092)
- "cask" → 261 (0.142), 262 (0.112)

these are noise, not semantic relationships. the Wake embeddings are floating in their own region of the space, connected to the model's behaviour only through LoRA attention routing, not through embedding proximity.

## Implications for P3

1. **L_morph prediction:** will likely be near-zero from step 0, same as TinyLlama. P2's LoRA implicitly learned morpheme composition through attention.

2. **L_device prediction:** the near-perfect isotropy (0.998) means device groups have no geometric basis for clustering. triplet contrastive loss needs pre-existing structure to amplify, but there is none lol, so predicting the same null as TinyLlama.

3. **Norm convergence:** the Wake tokens at 1.50 vs base at 0.99 is a 50% norm gap. L_norm in P3 will try to close this. if successful, it could change the isotropy picture: tokens at different norms can't be isotropic in the same way.

4. **Best checkpoint for P3:** step 500 (val 4.04). the LoRA layers at this point have learned enough attention patterns without excessive overfitting.

## What the norm gap means

the Wake tokens never converged to base norm distribution during P1 because:

- P1 used spherical init at `1.5 * base_radius` — they started at 1.50
- P1 only trained for 3,000 steps with gradient masking (only Wake rows update)
- the gradient signal in P1 is next-token prediction loss on Wake-containing text. this optimises the embedding *directions* for contextual prediction but has no explicit pressure to match norms to the base distribution
- P2 froze embeddings entirely, so no further norm adjustment was possible

**is this bad?** not necessarily. the model works — train loss dropped to 3.03, the LoRA layers learned to route attention through these embeddings. but the norm gap means the Wake tokens have higher "energy" in the embedding space, which could affect attention softmax dynamics (higher-norm tokens get higher attention weights before normalisation). P3's L_norm will address this directly.

the fact that the embeddings *work* at 1.50 norm despite base being at 0.99 is itself interesting. It suggests the LoRA adapters compensated for the norm discrepancy through learned attention scaling. the model found a way to use oversized embeddings by adjusting how much attention it pays to them.

---

## Generation Outputs

**Prompt:** `riverrun, past Eve and Adam's,`
**Fixed params:** top_p=0.92, top_k=50, rep=1.15

### Temperature sweep

#### temp=0.5

> nights,
> and the singachamer of a broadcussed bluffingly of their rawshorn
> parents at home (O! O!), as if they had been portnoysers by
> the law to be made out on the first place in some spirt-
> on-deed or other which was not quite so good for them.
> And it is all bilge. But what about that crazyquilt? What will
> you do with that crazyquilt? How are you going to take care of
> it? The answer lies in our next question.
>
> The crazyquilt must have a name. And this we call its title. It may
> be any title at all: "Annie's Heir," sweet Annie; "Bobby
> Hawley's Wife"; or even an old woman who has never heard of him
> or her but knows how to spell his name, can write down the title:
> "Miss Mortherackenzie Wills' Husband." You see then, my dear
> Lillie, your should be married one day like Miss Mackenzie
> Wills, husband to Mr Willy Nodder, after he comes back from
> his auld - ride in the whole of the flores and makes

#### temp=0.7

> that gave it was
> as the saying goes to a man of mark.
>
> The best _flabberghosted Mao_ in this part of Long Island is
> one block over from mine. And I'll be deathned if there isn't
> a nice big house with an awful lot of green lawns and
> praises down on me as well. Why, one would think he had done
> some gunne! How firefinders? Well, for three years now our
> family has been staying at a place up in Old Westbury,
> called The Regency or something like that, you know how they do.
> And after we'd finished every bite off of every morsel that fell
> out of each mouth of ours my mother used always to go back into
> the kitchen and say: You take your decentsoort seat here till I come out
> with more food for you. But when she came back and looked at us
> she'd sigh: Look at him who kept his nose so, look at him that
> soothes the whole toothed salaames of all of us. By the time next Christmas
> we were going to have another big party, doncha know why?
> So then what did poor old dear Mamma do?

#### temp=0.9

> was there a night of
> fire under the Happiest? I feel my head in an Alp, (I've never been
> up that hill) but it is too dark. Why did you not wake me up,
> Lugbrife?
>
> Now, we are both wearing darkness here with no light to see.
> You will remember how indanified blue lights were still funebral
> proper on the old days when they used terrible. And if you could have
> seen Thyme and Pappa Teal down at their windows out to sea from
> their corne for all time! But she has gone back to her house in
> the valley now and so have he.
>
> And I am alone in this world by myself like our good pitcher Mr
> Sooty-Piggy. There is one tight more than me as well: his name is
> Mr Raisethoverer or rather its really more the man behind him who
> is the most awful of them all. He was always looking over my shawls
> and watching over me, ever since he saw the way I apposed my spirit
> when he laid hand upon me after every mealtime because then I had two
> ears instead of only having

#### temp=1.0

> where there the
> mountain now. Oft of yon ridge the wind in wild rhoss's with him.
>
> Aye it did not blow long ago, at this hour or that,
> the hours are a hundred, but they were coming airly of old when
> a lord studiert his pipe on the night by lamp-light: he is well sorrd to have done:
> his part was only waiting for you. For I can hear him from my own.
> Hark! It does tell, how it came like, how it almost homestersed!
> The moon shines down out of her place; we see no more what has been;
> it makes me forget myself, as if it had all happened since your going
> away (how could it be told any otherwise?)—well, sloper, sir, so must she
> be some time—but you do remember, won't you, Mr Tubbings? And,
> my love, here comes divinely –y! Here he is again! The king has come up
> from Heksaenland's land in his noose under his keys of gold
> and hummer, who is returning, after much strife, from many lands,
> in search of their rightful manhood in every realm

#### temp=1.1

> with his diagonally,
>
> a full. On this last you might be wise to follow your own
> of the Sire I am not telling but all that what unhook man could.
>
> it is my own special way of speaking now as if there was one
>
> no bottlenim when such we are walking by a stream. Now who
>
> whereas she go? O vuncular! Be calm about it, me dear.
>
> Don't tell a lie about _your_ own face in himself either,
> for while here till any hour shall have been for both of us!
>
> And then do me a favour: would think of me some time so marrage
> may feel
> to look on our part for her in return and come down out of yemploy from
> our sea at our head by night into the wide fascinator or park or play-
> out green (donn!) on them they always love like jacklessextravagance. It'll
> all turn out all right yet sure notwithstanding, say you're, thinging!
> O don't make my poor heart sadder! Sure we're coming all along dinkel the
> cause in spite of him without her panementically. We've got more than that

#### temp=1.2

> a tourty of the yiu. A
> pristine piece has only been offered to him (if we grant for an instant,
> by old Groucho!) but since his _famouse_ might one day cause all that ever
> golded as our grand Old One with their wretched death too, let us take in
>
> for now he would become quite another animal) on these very lines.
> A creature born without the means nor materials tdrowsing it is most easily imagined into existence at every point from what
> has lived thus far like by way of futility or, more ratheely so that,
> in terms of human history, when I remember my dear ones living chayney to say a hundred and thirtytwo they will still be looking down upon
> us from among the hills amongst their relics, lightened here below which have left us
> on the plain; the two sisters (their hair up!), after fledge there was never any
> chance how you came about.
>
> While your mother and grandmother lay deep under water? Or
> while, jauntyjogging to say the half truth, when poor O'mighty once
> he could do this easy way?
>
> Of her soul can go forth, not then with the strength to reach out towards

### Multi-sequence generation (temp=0.9, x3)

#### [1]

> gave in the tarrying. You have
> the cupandnaggin hour! Your smoke is vamp to be a good friend of her abarrel.
> To go with you as long as ye both shall live (thwackaway!)
> As ye will never part no more? I hear your answer, if not,
> I see it too by that light on your face, for they love wearing makes
> you so loving! And let our vows hold firm ever till we meet again!
> But how soon we may hope! For sure and certain there'll hamo at once
> to each other inturned as thoughts to your very soul!
>
> From a letter from Mr Tomlinson of Dromore Castle, signed
> with his own queenlys signature, written home by Miss Sallie Maitland
> of Wester Keath, one of those most primed sister sisters:
>
> Farewell! The world's about thee but thy heart of hearts has fled;
> soreen far away, all blue with mists or black with rain and hailing soards;*
> and here are two my dear ones like to say thou dost well why still
> thou hast been telling us*.
>
> So glad I am of thee crewsers me, the

#### [2]

> where is a world of
> nights? Who is the man that would like to me, who was still will I,
> when he comes back from his search for a place in silentiousnessh, if ever?
> Is it not some one who has lost all sense of time or hear its
> name, Bovary (ah?)?
>
> Let us go! Forth with the way.
>
> —Ah, well. Well, such did we lose on our last meeting? And how do you
> feel about your new eatables cognates communial, this being only a first
> fourwheedler or so? We must have been darnels over our shaman, but at what
> time? Or when may we meet again next session?
>
> But here are others' saloontages on my own too now. Tell me.
> There seems to be many more. Indulger's, Jaunry, Linnagorror, Ogganuvaar!
>
> And the Groungy Eepers? Is there no more than two of them,
>
> Why are they both always talking together now?
>
> Spiceingly it sounds as if it were another pair of twins. But why
> are they always mortificationing each other on their

#### [3]

> to the
> untrustworthily but another thing or two of sound.
>
> (In this way they have explained why it is not only in the lippes that one can hear at Comynstow with his band. It has been remarked by all sadfaced stewies of Wiltshire (not the mircles) about some odd old sesquipedalia from Dorsethire what how now then again will come back for next time. The same joke was told by some more sober party late last year in an old port town.) I think you are right when you said _chemins de la croixes_, the famous story of roasted wheat used in French fable which we so often quote on our plenty books, where as an example to young wheedle ean he calaboosh.
> This came from the verdants topic itself, no doubt. For every tuteward plentihorns who,
> when he spoke first in a skol, would hold up his dear hand, say you?
> So hanging: a skol himself. You are most welcome; your beauman'smindedness
> has always won us heart; even while you were speaking, we did but feel
> you had justly earned the title of `sylpeveninrumpty'. A

### Single generation (temp=0.9)

> where
> in a meadow of singulhurrying or so we were on our way,
> out to the fair; they who know this place by heart (and all is well)
> we're in your founterest: if ever you wished there had been no end of us:
> may you live but one hour from now till never! Now with them are many!
> I saw it too. So do I remember her as yet. But my own dear milk's up she rose.
> O come back again when thou art older than an olive tree that will not shed its leaves at another word of mine nor fear any ill between thee and thy beloved!
>
> It was quite a while before I came to terms with myself for having taken leave off her.
>
> Heaven knows how long, then, it took me after all those years of pleasing
>
> to go on thinking I could just say good bye without ever being able to forget what she did pleace, seeing that I knew she would have said: "Never!" And this I am writing about before. For sure, some day later, after which tribalbalbutiencean getterated into his mind because he thought every man's life had got him coughingedentional minds and then

---

## Generation Notes

### Comparison with TinyLlama P1/P2

**Key difference:** Llama generates in a distinctly different register from TinyLlama. Where TinyLlama produced dense Wakean pastiche with compound-fusing artifacts ("theshade", "haveheard"), Llama produces flowing Victorian/epistolary prose with Wake neologisms embedded naturally within it. The base model's stronger language priors create a more coherent surrounding frame.

**Temperature behaviour:**
- **0.5:** Most structured. Epistolary voice, proper nouns ("Miss Mortherackenzie Wills", "Mr Willy Nodder"), crazyquilt as narrative device. Wake inventions are minimal but present ("singachamer", "broadcussed", "portnoysers").
- **0.7:** Domestic realism with mild Wakean intrusions. "flabberghosted Mao", "decentsoort", "salaames". The Long Island setting is pure Llama base model.
- **0.9:** More invention. "Lugbrife", "indanified", "funebral", "Mr Raisethoverer", "Mr Sooty-Piggy". Direct address and rhetorical questions emerge.
- **1.0:** Verse-like rhythm. "Oft of yon ridge the wind in wild rhoss's with him." "Heksaenland's land" — multilingual compound invention. Most Joycean of the sweep.
- **1.1:** Fragmented layout, Irish cadence. "O vuncular!", "panementically", "jacklessextravagance", "thinging". Comma rhythm doing the work of periods.
- **1.2:** Maximum invention. "tourty of the yiu", "tdrowsing", "ratheely", "jauntyjogging", "chayney". Grammar loosens but never collapses.

**Notable inventions across samples:** "singulhurrying", "tribalbalbutiencean", "coughingedentional", "cupandnaggin", "silentiousnessh", "saloontages", "Ogganuvaar", "sylpeveninrumpty", "mortificationing", "plentihorns", "homestersed". The model invents in a more Victorian-literary register than TinyLlama's raw Wakean pastiche.

**No spacing artifacts.** Unlike TinyLlama, Llama doesn't produce compound-fusing ("theshade", "haveheard"). The 128K base vocab already tokenises most word boundaries correctly. This confirms the TinyLlama spacing issue was a tokenisation boundary problem from the 32K vocab, not a model architecture issue.

**Multi-sequence [1] is remarkable.** The Dromore Castle letter, the verse fragment, the Irish farewell cadence very much reads like a lost chapter of the Wake filtered through a Victorian letter-writing manual. "cupandnaggin", "thwackaway!", "queenlys signature", "soreen far away" are genuine Joycean inventions in context.
