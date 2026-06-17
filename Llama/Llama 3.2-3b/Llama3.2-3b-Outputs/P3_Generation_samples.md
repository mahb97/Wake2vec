# Llama 3.2-3B P3 (strong) Generation Outputs

**Model:** meta-llama/Llama-3.2-3B (4-bit NF4)
**Phase:** P3 strong (morpheme-compositional alignment), step 600 checkpoint
**Source:** P2 step 100 (best val 5.33), P3 terminated at step 600 (val 5.4653)
**Config:** λ_morph=50.0, λ_device=2.0, λ_repulsion=0.05, λ_norm=0.01, LR=2e-5
**Wake injection:** 44,195 tokens (~26% share)
**Date:** 2026-06-17
**Prompt:** `riverrun, past Eve and Adam's,`

---

## Run 1 Single generation, temp=0.9

> all the night long way from their native land; he who is seem for his life at an hour like this, with the light of day to see with and a candle left behind by him;
>
> [1] I had those my saggind anyon or his dozen. What have ye done with the babe? The babe! Who has that in their fructificationr sofld hands! A babe!
>
> He was heali! He has been dissimulant. Let me look again on him, I say! That is my son whom I love best.
>
> — I am sure you are, dear Mr Myles: and as he would come out again so be it if he could but for one minute, though I never knew him yet, not even in dreams; but here he is now, my own baby! And we will tell her who she is when you've humself your head. She may know us after. And hailth up his homespinning now, good man: we'll leave you alone with him while you do.
>
> — Well then he'd take them both in tow. When the wears go forth from Bhinghoolabulla (Begorra let 'em

---

## Run 1b Single generation, temp=1.1

> I have often come across there with my stool and in their smell of Mooshe of Stoutheynsland or Wrex-ham Castle or Cnig-walls' Church, and many a time (I hope you'll never forget), that lovely dear Miss Sunnydaye has gone round the corner from old Mr Eversley's shop with her handkerchief out for me to borrow while she was waiting till it's my turn. Why this is true!
>
> And so he used again as if he could not be more nurf. He fell into the most perfect ecstasy over his own dieoguinnsisry.
>
> But it always ended by his saying:
>
> — Now you know why bigyttens! Don't say any word now but: We must have all our hearts up to him who makes us all like his image! We cannot go wrong when we do right. There is one great bevel on the road that leads keling where they can think away without having thing to worry about or cry high on until a man meets with bad luck. That will show how good he must mean to do next since he don't bother himself at present with doing no harm.
>
> Well?

---

## Run 2 Multi-sequence (3 samples at temp=0.9)

### [1]

> garden. His 1 , as the 2 , he had his name by. He was a very holy man; but to be sure 3, that all 4 that has him ever seen, it is not yet informed of his 5 in which country 6 he lived so long; it is said only this about him that there were no three at the time when he began to take care on us; and the 7 after, when the day of doom came down, 8 (he must have done some thing 9 for there was no telling what) they put him to bed with our Lord Jesus Christ under their pillows and they would not let them get away from his knees.
>
> The next place that I heard of after that was in Ireland at Ballyclarenal near Oulagh 10 who says that old Tuthmull, the man who used to tell the stories of Cavan (we call him by that name we know), told me how one night before his last supper 11 he 12 saw in her dream with the two hands out stretched over her head an angel or spirit (I cannot say which) coming towards him, 13 (the

### [2]

> home (as sweel as it is no longer of that echoland). There were a few other little old places near by on the way to Thingesdale, but the one which looked like she was the oldest place in the world, was named Cog. She was not quite so pretty, but when you got back her home again, you remembered why she used to be stilr. It is hard to say how old he really might have been or where himself his birth took place, for I am very sure he never told anybody his age at all and there was not even anything written down about him; he only said what a nice place Lammastide had become since they stopped making clocks and took out their hands from their glatschfs to walk about in a natural manner, free from any kind of pyreiness, except on Sundays.
>
> He has always been my favourite person with whom I have ever come across. I believe we met once in an stoold garden in Follydowne while taking our daily promenade along the lane. He used to sell some beautiful old nickstges to me then. How I missed him during the years of my imprisonment. But, alas

### [3]

> . Here we can make them the better for our rainkisselds, their caratimaney of us, their future to hold back, if she may so desire it, with one foot in this world, and one in that other land; let us do it then as well by her (the good old woman) as our own sirs. I am a sweeiate now, since all is done that was on me; let my hand be fained never more! My old love and dear life, my true haun and best half: here, here where you lie down once again, O aquiassent and fair one!
>
> —Ah!
>
> —Pity thee not now but when thou art gone far hence, for sure thou shalt come no more. Now are thy years up which they were set; time hath rung his round.
>
> —I will never go till he calls upon me.
>
> —Yet who knows? Oft have we been parted afore, at canatlesse hours, like two lovers which we were made after a fashion. Who will tell him yet?
>
> Let there be humselfness over thy first sleep, gentle lady; let there not be much still over your last waking nor

---

## Run 3 Temperature sweep

**Prompt:** `riverrun, past Eve and Adam's`

### temp=0.5

> first saughter question, the first of a new race, the first of an old one. And that is how we came to be. And there was no more.
>
> — _The Book of the New Sun_, gaingridando of Time
>
> A man who knows not why he does what he does, but has done it for ages; or a woman who knows exactly why she did what she did, but cannot remember when she did it? That's the kind of story I want to tell you about today: a showeryweatherory which will make your hair stand on end, if you have any left. But this is my last lecture in these rooms. You are going to hear me out, so listen up! The name of our hero is Shaun Mungo, son of Shaun Mungo senior, and he lives at number 10, Humphrey Street, Hooserfordshire, England (not now), with his wife, her mother, her grandmother, two brothers and three sisters. He used to live at 11, Changington Road, Storrs, Kent (now), with his wife, her mother, her grandmother, two brothers and five sisters. They all lived together in one house

### temp=0.7

> fall. Sickly to sick with the plague, And in his last hour he was born of a mother Who died so soon. So long since! He was as old As I am now: he had outlived his father: The child who would be king for ever. There's A dozen of the brike.
>
> [Omit _Hhore_ and all that follows.]
>
> For we are not here for Holi-day, my dear sir; I have my own good time at home; I come down from my castle by my own road, along the riverbank, by water or land, according as it suits me best: when the river is full I go up on the bank, and when I see there's room for two I'll step into the stream. This is how I get round the world: and you must take note what they've done to us this year! They've put a ring through our noses!
>
> But O Mancunia! Thou study, thou Gerrine, thou Boardren of the Fens! Thy ǝs make me can't: thou Burpa! Thine ancient

### temp=0.9

> whole those after which time they are no more; but the best of microamp, the love that swayed on daughter. He will let me by your leave believe it as well in another way as you can do for him; if I was only as good to keep my peace about him when he wants it: there is not many words need be said of this side either: and what could I say here after so very much (that means) has been said? We may go over all the thinges out of Homer and Ossian we like: as for her we have heard one drop too many. It seems to me that all this is an instance how _sous_ staot for some. So if any man should ask at his life how long does it last? I answer, till he takes his place again with our great ancestors of old! I'll tell ye, Taffy, she never had her hair cut since Hester's birth was five minutes old. My lord don't lie, she hasn't! O dear, now! To think of Mrs MacGee's being a contey! That's a bad job. Well, there's not many ways

### temp=1.0

> house in that sweeth Century road, and pige of all these times it had not the slightest connection with any house I ever saw. To my mind its only should be a piece for a atlanst to have been taken out of one's hat by way of a practical joke.
>
> The other was from a shopkeeper near Vauxhall Bridge: it shows how they made old wanns in their own way: but as you are looking into a window is that what himself to you?
>
> It comes down directly upon your present at home; if you were turned over the road there now (as you most assuredly will soon be), why it could hardly help your eye being caught.
>
> But this latter beauwy may come round like rain. And so I say—look me up! My alcoh! My head can wait till Monday!
>
> P.S.—And do take care, when reading my lithial above, just as soon as the bells ring tell pott to open his eyes and see what I wrote. Do try it! He looks up to the world in very good hands then.
>
> LONDON AND LIVEMOUTH,
>
> [1] For I am going out by sea

### temp=1.2

> in, his past Eavesdamp or aad of this very night. Ardt as if the air were broken into first smell! What was that soe-tern silve, down in our great heart? A night brigh. Ho, ho, are there nothing on my hearing light thosy up by day like so many starlings from their heath or is this some prime woman to try her old caught from one side to the other about him that I felt as she lay before my fingers at a time? Hush! If only your good hand could go out for mine as far as you want! I am coming back again.
>
> Ahoy there now, sir, what do ye say?
>
> The bhing man: Staffusen. Heave a heave
>
> But with more thosenesses every minute:
>
> For her poor dineen to be put aside. For her own sake let them hear ye. You have no care left behind since last week. Your way goes down along that long lane by Shalbury water till you come to an orange tree and a red cabbage plant growing wild beside it. There is

---

## Observations

### Temperature behaviour (inverse temperature-Wake-quality relationship)

Unlike Qwen P1 (dense at all temperatures) or TinyLlama (gradual unravelling), the 3B P3 shows a clear **inverse** relationship from low temperature to coherent English base reasserts, high temperature to Wake invention increases.

- **0.5** Most coherent, least Wake. Base priors dominate: `The Book of the New Sun` (Gene Wolfe), English place names (Kent, Storrs). Generic literary narrative with one invented word (`showeryweatherory`).
- **0.7** Archaic-literary register, verse-like. Sparse invention (`brike`, `Mancunia`, `Burpa`).
- **0.9** Coherent narrative voice, scattered Joyceisms (`microamp`, `sous staot`, `contey`). Homer and Ossian name-checked (Wake's epic substrate).
- **1.0** Coherence loosening, more invention (`sweeth`, `pige`, `atlanst`, `beauwy`, `wanns`, `alcoh`, `lithial`).
- **1.2** Highest Wake density. **Transforms the prompt itself**: "past Eve and Adam's" → **"past Eavesdamp"**. Dense invention (`soe-tern silve`, `thosenesses`, `bhing man: Staffusen`, `dineen`, `Ardt`).

### Generative regime: route-through-English, not stay-in-Wake

The 3B (26% Wake share) produces coherent English narrative with isolated Wake portmanteaus, routing *through* its bridge tokens (`wher`, `leas`, `befor`) and staying English-fluent between Wake bursts. This contrasts sharply with Qwen P1 (22% share, 14B scale), which stayed *inside* the isotropic Wake region producing continuous compound-mass. Same low-ish share but different in scale and training depth.

### Wake invention inventory (sparse, embedded)

`fructificationr`, `dieoguinnsisry`, `bigyttens`, `rainkisselds`, `caratimaney`, `showeryweatherory`, `aquiassent`, `humselfness`, `glatschfs`, `nickstges`, `sweel`, `stilr`, `pyreiness`, `Eavesdamp`, `soe-tern silve`, `thosenesses`, `microamp`, `sous staot`, `sweeth`, `atlanst`, `beauwy`, `lithial`.

### Wake anchors and substrate

- `Shaun Mungo` (Shaun, of Shem-and-Shaun)
- Homer and Ossian (the Wake's epic/forgery substrate)
- riverbank / river / stream imagery (the riverrun opening's water motif)
- archaic register: thee/thou, "O aquiassent and fair one", "time hath rung his round"

### The degradation question

The +0.135 val penalty (P3 strong 5.465 vs P2 wall 5.33) does NOT produce visibly degraded generation. The output is coherent and at times genuinely lovely:

> My old love and dear life, my true haun and best half: here, here where you lie down once again, O aquiassent and fair one!

The penalty is numeric (held-out perplexity) without a clear qualitative signature in generation. A reader of these samples alone could not distinguish P3 strong from P2. The cost of strong auxiliary λs is the train-val gap (generalisation), not generation quality. See `outputs/p3_llama3b_outputs.md` for the full analysis.
