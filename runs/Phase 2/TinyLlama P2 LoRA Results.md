# wake2vec: TinyLlama P2 LoRA Results

## Run complete: 3000 steps

**Model:** TinyLlama-1.1B-Chat-v1.0 (4-bit quantized)

**Phase:** P2 — LoRA fine-tune with frozen P1 embeddings

**LoRA config:** r=8, alpha=16, targets: q/k/v/gate/up/down projections

**LR:** 2e-5, cosine schedule, 10% warmup

**Batch:** 8 x 2 = 16 effective


## Embedding analysis (post-training)

### Norms
- Base vocab: mean 0.86
- Wake tokens: mean 0.47
- Two completely separate distributions (expected — embeddings frozen in P2)

### Drift
- Base tokens: cosine 1.0000 (didn't move — frozen)
- Wake tokens: cosine 1.0000 (didn't move — frozen)
- This confirms P2 only trained LoRA adapters, embeddings untouched

### Pairwise cosine similarity
- (base, base): mean 0.229
- (new, new): mean 0.251 — slightly more clustered
- (base, new): mean 0.227

### Isotropy
- Neither base nor Wake hits 90% variance in 100 PCs — high-dimensional, healthy

### Eigenspectrum
- Wake tokens have one dominant PC (~33% variance) — a preferred direction in Wake embedding space
- Base tokens more evenly distributed

---

## Generation sample

```
temp=0.9 | top_p=0.92 | top_k=50 | rep=1.15
```

> our fates parted and parted for all time: and from thefourth when it was sure to be (though not bysuch that way) when old morvaloos andlittle one withwoman of thecity, after wringing hispledge in theriver, taking down his tongue and's swallow a booty and having left thebottles full of floodlays, makes a wake by thefirstquaint but lasts but a few yards till the bump of theorder of his-hours entrope but at this thing the raid was distinctly expressed and the laughters had stopped they had safely looked how he had been utterly cracked and lit on with . . . _bouche_ as anallassundrian or annight of fire defences for their monsterre but while allaround them there camethrough the sixsheets ofdozen and their arms crossed (but only one even left the three eleven clothes!) and then he came from his lay and let out the shuffle and cut and quite as yet no happy day, when,

### Sample 2 (higher temperature)

```
temp=1.1 | top_p=0.92 | top_k=50 | rep=1.15
```

> upon a wet drydryfilthyheat from date too long to robbing tom Timagno, olddummydeaf was plain old Kissy Flukie of the Ravens, grackling round the doors and get on to bee. Down by the hedge we raced her, with ring and rumble and shocking blossim: and sheankered onsayfohrt like a bad bleakcran! Ah! Sowell she was out of the dark! (; Comb'd and Butt and Will he put a shirt off him!) Kissyunder Shanksapalms. And thefirst thing, when hesagd, til hispall to git it, hesagd, all the same, like time now left off, with a great deal morepondus, asyou'reas our huedct?).Selly he went on his way to herfuture, skillfully expressed, as it was, in hislewdbrogue her flat,through fornably designed for each other. For alas, what I might  have said about thatgarden old house on theheavycorner of Greenland,whereobelisk more after a power

At temp=1.1 the Wake-isms get wilder: "drydryfilthyheat", "olddummydeaf", "grackling", "sheankered onsayfohrt", "bleakcran", "Kissyunder Shanksapalms", "hislewdbrogue", "fornably". More creative compound neologisms, more Joycean energy. Structural coherence holds up surprisingly well — there's a narrative thread about chasing someone through a landscape.

### Sample 3 (multi-return, temp=0.9)

```
temp=0.9 | top_p=0.92 | top_k=50 | rep=1.15 | num_return_sequences=3
```

> **[1]** generating now aworth thesmall time when loved we our  honour. Awhile, as wewore, all sense shall be      INGENUOUS of ourvirtue made by love. _Traitor!_ Youshould not do that, we say, but, at all I ask from you, name no other saying.[1] Shall I,woman of thiscity, have my honour madebetter than she is? And deny me two Lives, innsome or excusties,[2] for that ancient year, who anew is she, when your honour is said to be from all thechildren of ournationalhistory?_  —But, tell us,girly, when was it that one ofthese days, when theircelicolarheaviest arms were nailed about herfirst winkensse? The times, ere thebrief one in hisheart had beenstillkept off with the post, when, on thefirst gross man in all the year-end-of-the-mons, when all the

> **[2]** though not very long since they were (and you now are, asfirst thinking how so many needles and muscles to the dirty deed), lastfellonem, a-going tomonkishouse and a-coming from Oreland, hunting foxes in the wildwoods roaring like Shemese and the great Blackamoor's Tobogogug at the Mitchells chrissormissfall when Mrs Wholyngstall with the Woolybanners shiver to see him ring his shoes for the last trade till the Big Funfellow bet hisshoestring on the turn of thehurdies, not forgetting the Grimrookman, who, having been detailed from the Bush with odds easily of one or two months for supper with the Dublinian 'who can't leave the gods drought to rape the old Seven yearsbefore the day, was merely there for the sake of it, honestly he thinks, for denying allmuttonsuet between Eins and Theus that that he was merelyunder the spell of this Mistlem eczema. The same fellow, as I have said

> **[3]** whence it was so long till when when.[5] Ten, one to thecorner, one in a thousand. The foolish baffling thing about it is, but I must have been asleep in my arctic toremembore, for there's no wonder if all the tasks above are ended upbeforer and the winds ships run awaythrough thetemple chase and all, when (I think of thefour gentlemen what do be after their death, goingthrough thecursesover the turnle beyondcods [...] by the grace ofgamyqueen Acre, the place of all lovely groe andlittle pumping and boiling andquarter fire, the wife of thefirstcitycity of Loundin Reginalddublinian) how,  when three commandments were only, the deaf andheavy old-ones were gone by and thefour gentlemen ever since they were gone athome and always on their last landing at or behind even inprivate parkings andbringing their sludge upon

Multi-return at 0.9 is where it really shines. Each sample finds a different voice — [1] is theatrical/oratorical, [2] is picaresque narrative (the Grimrookman!), [3] is dreamy/recursive. Notable neologisms: "chrissormissfall", "Tobogogug", "Wholyngstall", "Woolybanners", "allmuttonsuet", "gamyqueen", "Reginalddublinian", "toremembore", "beforer". The Hiberno-English register is strong in [2] ("what do be after their death").

### Sample 4: Temperature sweep on "riverrun, past Eve and Adam's,"

```
PROMPT: riverrun, past Eve and Adam's,
```

**temp=0.5** (conservative)
> well andwelcome, from the times and from the times returning. [510] Whyfore this morn when you were as long in yoursleeves, and now, late in your rides; how for once of it youknew a whorl with her luckyness, when she was in her secondfield, and now, having gonealong all your things right, in your big royal whitewead, in your grandexercise, after all your minilands, how you are going to take yourdane'spledges, with theirsummers and their faded effect, to thefuture of your raid and raid, fellows of Flavin's, for he is anorder than any appraised at all in our country? Who tells him what will be his deed end? You, who liveunder thelibansworld, in tublin or on thefieryfield, without anorder or reef, but barked to be Cut up, like an old bank forged by certain past policyers, with mytemple and the

**temp=0.7** (balanced)
> halting and husbanding, with their two sights for Capellana and the firstlightning of thefirst shenegillnd of the morn; famine, UNI-I-OH. and the Cunning of the Great Mearage Haun (186] aubent upon aheavy indigenous know, who, having been beetful of soul and soul in finelanguage, happened to catch a grovesong while trying his jade hataround, andwhere he lost her, that's how they parted. Acheer by the wrong we won, as I shall show you, by the wrong we'll gage. Now, if you'llgimmy now, I will tell you a test. Sickerson,you'reasily, would you go with me to thejampot and feel thecurt of your breast? Thirsticaculence,you're it is, so that you can see what agirl may feeling like a grand old pig in hermadapolamsmock, say, get up and push your rightfingurce into the foggydew's sp

**temp=0.9** (warm)
> from the dread of thecloud to theshadow [2] Thefourthnight of each year in the year of our fairest [3] When the nowdust is is yetstillheard when thefirst is gone _Amper pewers on knows the swayof all Cubbocoats on Rushes Tricks of the shyman and the rake of his rope- wat._ Their arms inhunterland. 698] And she, her shells about thatfiend.[7] What I love is there above all Now!,[1] If you couldremember a moment 'Tis too yearly when love made his silence Forgiving his naturalsleep. Will you, collar man, mind that now?[2] Whpping the buck up andpaump,[3] withsuch thepoint of willingworse, while, upon the heavy sayings which come from your mick (There's a blink of hear), last by nowhere while, at

**temp=1.0** (hot)
> to show the wishing way to take death. Aimihi! Ah! Andwhere atheses be who would leave oursleeping sins for a present leperdbrethern? Theblood is toojarveyye and the twig as buttaley; and the two are onlyterpary hands between you and thehorizon. Hearn! Ohear! Theangel of your mother's heterotropic, the law captive, the law which they mustangs bepledged with, thevirtue of love or thelife ofmoney comes wherein John the Spainman, the Mickish-Mucking, the Mr Browne, the man in the pink, all, all, (their being now asthentimes for ever as out of date) has murderedhimself. The mother of the S born Man, thewoman of thegleison family, he's trying to make up to him, while Mrs knows not what to do, if I may say, as so much isunder yours that I can tellnothing about it, this last at any rate, as for in fact he must know

**temp=1.2** (unhinged)
> touching all behind and the ghost of now. Who comesyond? The gaudmmotself of civilisation, who thought or want orsuchly interfering in our mischief could buy plagues for thefournight of years. Nor must now and then, bysuch ones andtwos, ex cantonbabbling, bestreakingtogether for daylight, fairness,sweet face, andhear, or _re_hear how others have used to do when they were had theirnightings on. Now (for he hath fled, Finnegan, the Servious of Orops and Vulpens)but will no more have us ungiven up to a third degree with hardyheavysleeping farin away from otherfour and so much having on our seeds.Whisperation is never sowellty after a good mitigation to improber manners atleast and even not having been as unused with embraceed down others. Accent isbetter upbefore a very long way off than half agame looks after being so smell

### Sweep observations

- **0.5:** Most coherent prose, closest to standard English. Still Wake-inflected — "tublin", "fieryfield", "policyers", "libansworld". Oratorical, questioning tone
- **0.7:** Narrative momentum picks up. "Thirsticaculence", "grovesong", "madapolamsmock", "fingurce". Hiberno-English voice emerging ("if you'llgimmy now")
- **0.9:** Shifts into verse/poetry mode — line breaks, numbered footnotes, stanza structure. The model discovered Wake's poetic register. "willingworse", "hunterland"
- **1.0:** Peak Joycean energy. "Mickish-Mucking", "jarveyye", "buttaley", "terpary", "leperdbrethern", "heterotropic". Characters named, narrative drives forward
- **1.2:** Full delirium. "gaudmmotself", "cantonbabbling", "bestreakingtogether", "Whisperation", "sowellty", "improber". Still grammatically parseable which is the impressive part

Best sweet spot for Wake-style generation: **0.9-1.0**. Below that it's too tame, above 1.2 it becomes more incoherent, which to be fair, is also very wake-ish.

### Notes on output

- Wake-isms bleeding through: "allassundrian", "morvaloos", "firstquaint", "sixsheets ofdozen", "floodlays", "monsterre", "entrope"
- Structurally coherent — you can follow the sentence logic even when the words go Joycean
- River imagery, compound words, parenthetical asides, trailing ellipsis into French ("_bouche_") — very Wake
- **Spacing artifacts:** "thefourth", "hispledge", "theriver", "thebottles" — tokenizer joining Wake tokens with adjacent words. This is a tokenization artifact, not a model failure. P3 morpheme alignment may help

---

have a ciggy, bang out to [Lost](https://soundcloud.com/maxomar/lost-1?si=a39eabc171034bc0bc43ae75c664587e&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
