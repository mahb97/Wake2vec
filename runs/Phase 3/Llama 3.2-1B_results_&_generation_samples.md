# wake2vec Llama 3.2-1B P3 Results

## Final Numbers

| Metric | Value |
|--------|-------|
| Model | meta-llama/Llama-3.2-1B (4-bit NF4) |
| Phase | P3 (morpheme-compositional alignment, strong lambdas) |
| P2 source | step 500 (best val 4.04) |
| Steps | 1,000 (4 Colab sessions) |
| Training time | 173.1 minutes |
| Final train | 14.539 |
| Final val | 4.636 |
| Best val | 4.482 (step 100) |
| Trainable | 358,291,456 total (LoRA 5.1M + Wake embed rows 90.5M effective) |

## Loss Trajectory

| Step | L_total | L_lm | L_morph | L_device | Val | Early stop |
|------|---------|------|---------|----------|-----|------------|
| 0 | 3.8597 | 3.4387 | 0.0007 | 0.1933 | — | — |
| 50 | 3.6508 | 3.2506 | 0.0007 | 0.1830 | — | — |
| 100 | 3.8224 | 3.3968 | 0.0007 | 0.1956 | **4.4819** | best ✓ |
| 200 | — | — | 0.0007 | — | **4.5016** | 1/5 |
| 300 | 3.6496 | 3.2213 | 0.0007 | 0.1970 | **4.5284** | 2/5 |
| 400 | 3.8786 | 3.4460 | 0.0007 | 0.1991 | **4.5482** | 3/5 |
| 500 | 3.7454 | 3.3314 | 0.0007 | 0.1899 | **4.5871** | 4/5 |
| 600 | 4.1771 | 3.7554 | 0.0007 | 0.1937 | **4.6075** | best ✓ (reset*) |
| 700 | 3.4596 | 3.0147 | 0.0007 | 0.2053 | **4.6179** | 1/5 (reset) |
| 800 | 3.6884 | 3.2429 | 0.0007 | 0.2056 | **4.6326** | 2/5 (reset) |
| 900 | 3.8024 | 3.3542 | 0.0007 | 0.2070 | **4.6312** | 3/5 (reset) |
| 950 | 3.8246 | 3.3902 | 0.0007 | 0.2001 | — | — |
| 1000 | — | — | — | — | **4.6359** | 4/5 (reset) |

\* Early stop counter reset on resume at step 600, same bug as TinyLlama P3.

## The Null Result (Cross-Architecture Confirmation)

### L_morph: 0.0007 never moved

L_morph was 0.0007 at step 0 and 0.0007 at step 1000. One thousand steps, 50x lambda amplification, and the raw morpheme direction consistency loss did not change by a single significant digit.

This is 3.5x higher than TinyLlama's 0.0002, but the Llama P2 didn't solve morpheme composition as completely as TinyLlama's P2 did (the 128K vocab let it take shortcuts). But the higher baseline was a starting condition, not a learning opportunity. The gradient signal existed but the model couldn't use it.

the L_morph panel in the loss curves tells the story: y-axis spans 6.5042e-4 to 6.3e-4 × 1e-10 offset.

### L_device: 0.185–0.210 

same as TinyLlama, same range, same null. the device triplet contrastive loss asks embeddings to cluster by word-formation process, but embeddings encode meaning and usage. a portmanteau should be near its component words, not near other portmanteaux of unrelated meaning.

intra-group mean cosines confirm: nonce 0.022, malapropism 0.013, portmanteau 0.012, foreign 0.031, pun 0.036 were all near zero.

### L_repulsion: 0.000000

literally zero the entire run, so no Wake tokens were close enough (cosine > 0.95) to trigger the repulsion penalty. the near-perfect isotropy (0.998) from P1's spherical init means everything is already equidistant.

### L_norm: 0.1787 pinned

Wake tokens sit at norm 1.504 (std 0.011), base tokens at 0.987 (std 0.093). Cohen's d = -7.81. the norm gap from P1's spherical init (1.5x base radius) was never closed, so P3 couldn't close it because L_norm's lambda (0.01) is too small to compete with L_lm.

## Embedding Drift (P3 vs pre-P3)

| | Cosine sim (mean) | L2 dist (mean) |
|---|---|---|
| Base tokens | 1.000000 | 0.000000 |
| Wake tokens | 0.999921 | 0.015784 |

P3 barely moved the embeddings. Wake token drift: mean cosine 0.9999 (essentially no angular change), mean L2 0.016. for comparison, TinyLlama P3b had similar near-zero drift.

### Most-drifted Wake tokens

the 20 tokens that moved most during P3, but all still with cosine > 0.998:

1. "buccinate" (cos=0.998, L2=0.090)
2. "galory" (cos=0.999, L2=0.080)
3. "decer" (cos=0.999, L2=0.077)
4. "danno" (cos=0.999, L2=0.076)
5. "whoses" (cos=0.999, L2=0.073)

even the most mobile token barely moved. 

## Morpheme Direction Analysis

### Most consistent morpheme directions

| Morpheme | Type | Pairs | Mean cosine |
|----------|------|-------|-------------|
| -fication | suffix | 3 | 0.941 |
| -head | suffix | 3 | 0.856 |
| -wards | suffix | 3 | 0.750 |
| -aster | suffix | 3 | 0.727 |
| -ein | suffix | 3 | 0.723 |

the most consistent morphemes are all small groups (3 pairs). the consistency metric rewards low variance, which is easier with fewer data points. these numbers reflect P2's learned structure.

### Least consistent morpheme directions

| Morpheme | Type | Pairs | Mean cosine |
|----------|------|-------|-------------|
| -s | suffix | 1034 | 0.114 |
| -ed | suffix | 419 | 0.138 |
| -y | suffix | 405 | 0.152 |
| -er | suffix | 244 | 0.155 |
| be- | prefix | 61 | 0.186 |

the high-frequency English morphemes show low consistency, as these have many semantic uses (-s is plural, possessive, verb conjugation) so their displacement vectors shouldn't be parallel. this is expected and correct.

### Notable morpheme details

- **un-** (47 pairs, mean cosine 0.326): "Unsightliness→sightliness" (0.547) vs "unenglish→english" (0.140). the negation prefix works best with concrete adjectives, poorly with abstract/playful Wake coinages.
- **-ing** (247 pairs, mean cosine 0.204): "baiting→bait" (0.504) vs "falling→fall" (0.006). common English words show near-zero consistency because they're already deeply embedded in the base vocab's semantic structure.
- **-ly** (143 pairs, mean cosine 0.235): "neatly→neat" (0.578) vs "Family→Fami" (0.030). the adverb suffix is most consistent when the base form is a clear adjective.

## Device Group Analysis

### Intra-group clustering: nonexistent

| Device | Words | Intra-group mean cosine |
|--------|-------|------------------------|
| nonce | 757 | 0.022 |
| malapropism | 690 | 0.013 |
| portmanteau | 465 | 0.012 |
| foreign | 150 | 0.031 |
| pun | 61 | 0.036 |

all near zero, so no device-type clustering emerged during P3. the triplet contrastive loss had 1,000 steps to push same-device words together and it accomplished nothing lol.

### Inter-group centroid similarity

|  | nonce | malap. | portm. | foreign | pun |
|--|-------|--------|--------|---------|-----|
| nonce | 1.00 | 0.83 | 0.74 | 0.73 | 0.63 |
| malap. | 0.83 | 1.00 | 0.76 | 0.67 | 0.57 |
| portm. | 0.74 | 0.76 | 1.00 | 0.56 | 0.49 |
| foreign | 0.73 | 0.67 | 0.56 | 1.00 | 0.54 |
| pun | 0.63 | 0.57 | 0.49 | 0.54 | 1.00 |

the centroids are moderately similar to each other (0.49–0.83) because all Wake tokens sit in the same region of embedding space (the 1.50-norm shell). this is geometric proximity from P1's spherical init, not semantic similarity from training.

## Isotropy

| | Score | Mean cosine |
|---|---|---|
| All tokens | 0.951 | 0.001 |
| Base tokens | 0.983 | 0.000 |
| Wake tokens | 0.998 | -0.000 |

Wake token isotropy: 0.998, which is near-perfect. this is the deepest explanation for the L_device null, the embedding space has no pre-existing structure for the triplet loss to amplify. all Wake tokens are equidistant from each other, so there's nothing to cluster.

## Pairwise Cosine Similarity

| Pair type | Mean | Std |
|---|---|---|
| (base, base) | 0.132 | 0.063 |
| (new, new) | 0.003 | 0.023 |
| (base, new) | 0.003 | 0.022 |
| KS D=0.902, p=0.00 |

Wake tokens are orthogonal to everything, thus the base tokens have semantic structure (mean 0.132), but Wake tokens don't (mean 0.003). 

## Cross-Architecture Comparison: Llama vs TinyLlama

| Metric | TinyLlama P3/P3b | Llama P3 |
|--------|------------------|----------|
| L_morph at start | 0.0002 | 0.0007 |
| L_morph at end | 0.0002 | 0.0007 |
| L_morph moved? | No | No |
| L_device range | 0.17–0.24 | 0.185–0.210 |
| L_device moved? | No | No |
| Wake isotropy | ~0.998 | 0.998 |
| Wake pairwise cosine | ~0.003 | 0.003 |
| Wake norm | ~1.50 | 1.504 |
| Best val (P3) | 3.4188 | 4.4819 |
| Steps trained (P3 total) | 2,200+ (P3+P3b) | 1,000 |

the null result is universal across architectures. the higher L_morph starting value in Llama (3.5x TinyLlama) reflects the 128K vocab letting P2 take shortcuts, but the residual error was not learnable. the device null is identical.

## Generation Output (all examples from the same prompt)

### temp=0.9 | top_p=0.92 | top_k=50 | rep=1.15

> nights on the old place, and a foug shot of Kilkenny Chimp-in from his turned in three days. He is a pitcher at one or two clubs, (he never could play for love nor money), but he has been an O'Murphy-O'Sullivan since Patrick O'Mooreigh was elected to the Connacht. I hope it will be as long as my own stay here: I have heard that the place is falling apart and there are too many shamed nations up there; but I'm thinking how easy now would be to get out again if you only have your hair cut!
>
> It might even go back into the country after this century. He lives right down from the Shock of the Gulls, the oldest pub in the neighbourhood; some time soon we must take a cup with him: you'll remember me! The landlord is a Paddy Finn who used to come out as a young man; indanified de Chanteloup about our good friend! Some more guests shall follow.
>
> Well then, what? We could not fail to fall into a jolly nostorey jukersmen cv when all the parties were gathered round us? A

### temp=1.1 | top_p=0.92 | top_k=50 | rep=1.15

> , _threadworm Shamoar_ and he could say to his neighbour: I'm so sorry about all that slipping, he only had a word for him who introduced into it by some tobaggonnt. At the end give they must pay. There's be no more sweathed nor no more things on your part if you can help a bit with our poor fellow.
>
> — Things was good fun when young! The alleance might well have thought how wrong out of order time has been since that day and may feel as old as their years too whilst there is left them what will always come back to him like the pitcher or the gambit of return once again; yet still there are moments when an old fharkr came back out of hearing with that same smile upon his face and one feels not very much older than then because when age comes down on us from our past lives we never were younger till rooksacht.
>
> And as the grandchild of mine made me look at life in her usual funny way: "O my little darling (she says), did yo remember this story? When ever ye want to see Puffly laughing don't laugh

### temp=1.1 [2] | top_p=0.92 | top_k=50 | rep=1.15

> nights at our Hymenee of Solem Commitment by the Rev. S. St. Pellegrin: The Old Private Yard in the old Chief Town, Tipton's Mews; Our Church in All Saints Chapel with the Stairs to Miss O'Mara's Drawing-Room on One Side and the W.C.s at the Back for Young Peeps (I'll show you the way if I can!) In his present position it is as difficult to have a real heavy-bus when he knows how to run up stairs and has a few good paces. But why put all these in an essay? I only mention them be-cause of their relation which will not take so much time after that and because they are rather more than usually related: we see one boy falling from a garden into the water (see here) but no longer what we ought or should do! He was just a shame-ling and did not know, we find on hearing about him later! so suck the fall then!
>
> And now let us come back to other matters again:
>
> At some date the following letter came from the R.U.O.W., now extinct, addressed to Mr Glade

**Note:** generation quality is essentially identical to P2. P3 didn't change the output character, and the model produces Victorian-institutional prose like "Rev. S. St. Pellegrin", "Hymenee of Solem Commitment", "R.U.O.W., now extinct" with occasional Wake-adjacent coinages ("tobaggonnt", "rooksacht", "shame-ling", "fharkr"). the register is consistently pub-nostalgic and parish-gazette rather than Joycean multilingual density. TinyLlama's "Woolwichleagues", "twohandledduolandroom", "mooremooremurgessly" exist in a different linguistic universe entirely, so assuming that the smaller model paradox holds: TinyLlama produces more authentically Joycean output despite (because of) its smaller vocab.

### Multi-sequence: temp=0.9 | top_p=0.92 | top_k=50 | rep=1.15

**[1]**

> in the dark ages when women were not allowed to be catlick on a man. No wonder some of their old voise about being so paludination where she could get his noise away with her from all but her four duty! It was said, it is true, that Shaun came upon them one day after he had gone off for a couple of days' royen (if they took it as an insult) he never got back to junojuly and that old Moll now howl. And when poor old Hilda went out again those once too many times and let him have his way at last, you might hear her say: Mea culpa mea Sanctus! I'm so very sorry!
>
> But no more will we repeat our woes till We shall live for ever! Till we can milkfeeding each other's every wish.
>
> So long may your years bring peace, joy and happiness here below! Though while we wait awhile till our coming hour comes, Be assured of my love forever! So do you too!
>
> But till then remember us by name alone, if any time or place we meet, Remember and love me always!
>
> The two who parted ways so many years ago, may

**[2]**

> are just a few that might be mentioned. It is very easy for one to say there were once many of these otther sites on the earth, but it is less easy (and more difficult) to prove their existence. It may happen by pure chance in some cases, but there remain great surplus possibilities when we begin thinking about this matter at large. What then must not be forgotten as part of our investigations? That they have been laid aside in old books, lost from history, all of them; or else, if they exist anywhere still, what has become of them?
>
> The true story behind every site which now lies waste or ruined on the face of nature can never be known. The same with hundreds of others, since time immemorial past. Yet even so: while he may have heard many tales of each particular place his mind was only curious to know how could one come up with an answer of any kind,
>
> He would hear of this mountain being named after its first owner who had passed away long ago, perhaps with his name written in letters above that peak and thus leaving no living human being alive to remember the memory of him.

**[3]**

> the night by the night of their last hours' in this our world. We have got it out quite simply with the two young ladies here: if he had no more than a word in his mouth to say in reply or remonstrance we shall cannonise him.
>
> It would be hard up for us all were you not so good an artist, Finnykins (you may find that your filli is coming down after you), so I am going to try one for _pipetta et Ouest_ as I think there are some interesting faces among them, just now. Well done!
>
> As I'm sure you'll agree they do look like what they feel when they're as yet merely being made aware of new conditions on which the old life will need altering--all very simple! This platterplaters who will come from that braze hole can turn out a lot of knifesalook persons. But don't forget those face has been worked till its features express most precisely what goes into its make-up. Just see how easy it was, Fyfekins (did yourself mean that? And don't pretend at home to go shopping.) You know why it is that you have seen

**Multi-sequence notes:** [1] has the strongest Wake echoes such as "catlick", "paludination", "junojuly", "milkfeeding", Shaun reference, Mea culpa. [3] shows genuine Joycean invention like "Finnykins/Fyfekins" (HCE-style name shifting), "platterplaters", "knifesalook", "_pipetta et Ouest_". [2] is almost entirely standard Victorian essay prose with zero Wake tokens. The model is inconsistent, and it can produce Wake-adjacent material but defaults to its base register most of the time.

## Temperature Sweep. 

Prompt: `riverrun, past Eve and Adam's`

### temp=0.5

> nights of the broadawake apologetically: the old house in which he was born, and the big white surtout in which his mother lived; a tall clericalease with its heeders and the heeders next door; a causcaus at the back for the poor man to sleep on when they were not too busy; two or three other houses round about, all as old as he could remember, and one new place that had been built by Mr. Wright, the undertaker. All these places were very dear to him: they made him feel so proud.
>
> And there were more than forty houses in this part of town then, to be sure, but it wasn't much longer since, when the brildges used to go out into their gardens after dark and sing songs and play postman's rags upon their harps, and hear them playing, till the clock hanged down, while the people went to bed. And you may tell how many blighty years passed away from me when I was just getting up from my tompullhamble sleep over there in the benedicted boaconstrictor untisintus t here! The wears are gone now, every last one of 'em

### temp=0.7

> on his shanksaxle to her darklane. And if you could but see the day I am, you would soon be as good a friend of me as any.
>
> And here is my own dear love in all naughty nature and every grace that she has: one clotheshorse look at your face! Another where it seems me with some thuchr beauty too. Hark! Hear! Eaves hear us now!
>
> You will like me, so long as you think me not to far from being a very fine ovenly chare! But, how can I fail? You are as sweet as a new sooty, umtoven when I have seen many nula thoughts, I never heard of such a sound nurse yet. It was this way: Once while we were walking (we always walked) along our best road by the sea side, after dinner, to get out of town, when I asked her for the coifst time, because I wanted to know the exact place about which you said you knew; what did she answer?
>
> She says: 'The finest part of the world!' O, there is no other place in the language

### temp=0.9

> nighting, to the coleopter is made through the last man from his seat in three Stoupsimpled of Brougham Street. A big gambit of brown bread with a marriedann of raw delictuous feet and a draught of fresh milk mixed up with an gause clears the place as quick as one can on a snowball (or, at any rate, in that way) after he has been there long enough for you to get an interest in him, if we may say so in all good fun.
>
> So now this great spicelely fellow is doing some hard thinking about how, while they were saying prayers on the hillside when it was going off and then the hikeret out and taken down to the police station he went into some deep moments of reflection. There had been some very interesting discussions during which he had got used to hearing the same old plenty again more often than he liked. Now here are two men speaking of other matters not being exactly germinating apart but only too closely or almost too near. The prisoner was beginning:
>
> --This dozen would have done me well ere I come over hearing my cousen gambeson to him but

### temp=1.0

> a where they were in for their next attack (Oblivious or so it seemed!) when the two Irish gun men suddenly appeared on the scene. One of these is lying at the base of this tree, his head nearly off its stump by old Mike Donnelly. With him I have some letters that tell what we are to do as soon as we get out into the open again with our arzurian halfprice.
>
> How he got them I know not but let us hope and pray you can make out how.
>
> And now here is my letter to Mrs Conry; her husband was very ill but she has just received the same from me which will be all right after the storm passes over but take good care till then dear wife if there is no other news since yours I wish you would write to your own sister about your new friends who want you back home as quickly as possible like Mary Ann O'Mahoney's said.
>
> Drudge: The Thok Darts heil deux, Mawg, Magory eternuel and Shanahan, Aa San, Si San, Baa Shan, Snana Shan? Or is it Hoot Haw?
>
> I am quite thick dampers but poleaxe yearin in respect of

### temp=1.2

> the good thingt from their wishes till he mountainyd that it was his. A new smell came over him. It might have been a very fine whole old fellow by this time, to see all the jarring of those great dayses who looked like they'd got to be in at final war with each other before but they didn't weaply away while there could be any hopes on either side.
>
> After so many had gone along he just began being sure (he made no move) for sure then for everywhere these two were getting off into as I am going to say, but really now we'll do him justice; our old grandpa Batty Fussell! Who will never forget me! But when we left him on my account (if you think what she did was right!) you used to hold upon your tip-toes too low over her as if behind all these people about (all I can feel) you wouldn't notice what is exactly done throughout them under one brow and why are they laughing at us both in the beginning (we went up), you held it quite wrong that once in his hour (since not yet priviousn but more than one look

### Temperature behaviour (Llama P3)

- **0.5**: Most structured: "broadawake", "clericalease", "tompullhamble", "boaconstrictor" so here, compound coinages emerge. Reads like a nostalgic memoir with occasional portmanteaux.
- **0.7**: "shanksaxle", "darklane", "umtoven", "coifst" these are more invention, register shifts to love letter. "Hark! Hear! Eaves hear us now!" echoes the Wake directly.
- **0.9**: "Stoupsimpled", "marriedann", "delictuous", "spicelely", "hikeret" here density increases, begins to approach Wake register but never sustains it.
- **1.0**: Sudden multilingual intrusion like "Thok Darts heil deux, Mawg, Magory eternuel" and the rhythmic "Aa San, Si San, Baa Shan, Snana Shan" which closest to Wake's multilingual play. "arzurian halfprice" is a genuine Joycean compound.
- **1.2**: "mountainyd", "weaply", "priviousn" here grammatical morphology breaks down into Wake-like verb coinage. "grandpa Batty Fussell" is a plausible HCE variant.

**Comparison to TinyLlama P1 temperature sweep:** Llama produces more coherent prose at every temperature but less Wake-specific invention. TinyLlama at 0.9 gave footnote markers, verse formatting, numbered fragments, and showed structural mimicry of the Wake's layout. Llama keeps to conventional paragraph prose throughout, but the Wake isn't just vocabulary, it's also structure, and the bigger model's stronger priors resist structural experimentation.

## Verdict

**Llama 3.2-1B P3 confirms the TinyLlama null across architectures.** L_morph is a condition, not a gradient. L_device is asking the wrong geometric question. the μp arrow was always there, so P2 encodes morpheme composition through language modelling alone.

**Llama 3.2-1B pipeline complete: P1→P2→P3.** Best checkpoint for downstream use: **P2 step 500 (val 4.04)**.

