# Wake2Vec Phase 1 Generation Samples

## Model

TinyLlama 1.1B, embedding-only fine-tune on Finnegans Wake (3,000 steps).
All transformer layers frozen. Only the tied embedding matrix was trained.

---

## Sample 1: Single generation

**Prompt**: `riverrun, past Eve and Adam's,`
**Params**: temp=0.9, top_p=0.92, top_k=50, rep=1.15

> and the crossgoing out of Humphrey's age, to thenext song of the Rocks.
> The marriage of Barandy and the Foughty Fatima, my shopsoiledgirls,under
> theillassumed names of Perkinson and Misham. The sighing of aladle man
> thatstill will be the thing that joyfakes alllife. I have read in the pink
> of Mourne's face as he dreams of her. I haveheard the windcursy of his
> doom. I haveheard theshades of histhumbs. How Thine Sidesup, Jeminy,
> isber,such and is offensive! And you, woman, are not as clean as you are
> now, my dearest Yokeoff. And it is my last endings, with that oldmilkless
> a-going for asoft time, thatsleep by the staircase, that threesheets at
> nunc or twogiving a toll, thatgrip at noon (for if she has achild, she can
> never have asleep) that has a

---

## Sample 2: Temperature sweep

**Prompt**: `riverrun, past Eve and Adam's,`
**Fixed params**: top_p=0.92, top_k=50, rep=1.15

### temp=0.5

> and thelightning Nixy, and thefields of Olaf, and the bannars up with the
> MacGhoul, the Vale's inheaven, and the harricanahurries andhotsfoots,
> zingo,zango,segur. To the cross Lally, to the cross Doolin, to the cross
> Sean Adoyer, to the cross Arth Mockmorrow Koughenough, O' Ryne!
>
> Ryeer! Hear, O' Dullmailee! The Macclefield Witch is reelingthrough the
> top of theworld! That washes yourhead out, youunderstand, and you can't
> forget the ten or so your twin had to pull, and thetwenty one that was
> sitting on the three, and all thefour, two by two, with their white arms
> below their ears, at the end of theshadow, howoft right enough, as I
> think, in the way of fun, for their castor and porridge'sgame, as they
> were going to behind a wall and the taller man

### temp=0.7

> and thelightning Nuns and the Cameen or Corpse and the
>
> [104]
> Tublin. This is not a very long way, myprodder again! Once more after this
> time, in thefuture oflife, when ourpantriarch have entered their
> ownsummers, while old Matt Gregory wouldn't be seen, there's a few more
> between you and the man in statewearholipoliwhollyisland
> peeeeeeee[132] werewhere, when he was just achild, and you werestill in
> thewhole. That's what wouldn't be too far, my very fructification,
> mylittleheart, my same uponhearts, my hair, my ears, my nose, my eyes, my
> faith, my hair, my hoops and all my ether, no matter how many, when that
> man had not beengiven thelobby, when thecorner was in his place, and I was
> too far away to askhimself fornothing.
>
> So, now, as we are in the

### temp=0.9

>  [175]
> from the day in all our things has been
>
>  UNDES.- _Nonquodsed Vestrae
>
>  'tis everynight 'tis all about._
>
> [1] I have only a staircase)
> [2] Six on the run)
> [3] Who is on thefourfirst then?
> [4] Weopen we or mates our winds with itsnation,[2] like asfour round
>     about
> [5] Cthahraet and Malthosius trying to die!
> [6] We dohear some old times (you and two verysmallthirtygirls!) Shem and
>     Shaun, out of date.
> [7] A pair of green eyes at the back of a shirt at Pickardstown.
> [8] None of thefour by the sea,through the black man at Roseleys.
> [9] Alared by the blackhearts allaround roundbrigidschool —_Truly much for
>     thee,histindier. When was it ever ever up?_

### temp=1.0

> withlustres ofpeins. Whatsound be done if
> only so they were?[1] 1065 (3618) No. I say
> it is awild'ssort to be cracked by all.[2] Now, old
> man, it's time you turned thesleep and come out of
> yoursleepingexex. Aye, and forwards I will stand
> tobring you out. And you to her, and you, and she
> to her back! So pass thetrouble on, and take your
> _Bylineal in the bedroom. Bier, stiff pumps, 1169._
>
> Waxens for wimwyer,head in love,bloodtune onsweet andfirst, thump, by,
> shirt off, shints tolife, cakestood,kiss up, buckler,head off,hear,
> _Mi-face_,such as Tuskar and Ania. _Tuesay, Pudge and Be Peposys. This
> issuch achild. Proper

### temp=1.2

> where the Nilsens made the coke of this tay for thehead part in thefour,
> where hewallowednnykins all down the rainvert redvilla. To mark her
> ownlife or pity to him. So the water and thehind that was milling in
> thefirst Shem or the Vain that had nowhad it, now love it, now
> anextinsionkissed the twins (for sheknew not thelanguage, but what sheknew
> was so long as she just caned her heirs) while thatwoman (who, then,knew
> howsuch aperiodiosit bead out of Vrittiants and Tadters, no lie!), when
> her old time-ricking time waran act was on, with apurecures for a wound to
> be due she putunder hispallyass and begin togive arms, girdles,hatsoff to
> all theirpurtybussesning lovely about
>
> [120]
> hissleep and his flesh may neverfall. And there shestill words how to
> jayne and musical

---

## Sample 3: Multi-sequence generation (×7)

**Prompt**: `riverrun, past Eve and Adam's,`
**Params**: temp=0.9, top_p=0.92, top_k=50, rep=1.15

### [1]

> theshade of ages (our times are done) with theirhistoricbringing them.
> Those were the

### [1 cont.] MPM

> Homo Vestrae, Vale, O'Neill!

### [2]

> Theheart of Lifé, the year of the Cure,

### [3]

> Fought for Humans' mound in Peruvian:

### [4]

> _Ere_ I go to quest of Wachtman's Cromwell,
>
> [265]
> high time as far as Tear-nan-Og,
> as far as the Oyest Brayles;

### [5]

> The

### [6]

> butwhere is he? Tell me, why do we be on of thatclass?
> Why not at the Rother's stomach? If she can't keep him at lughts or
> forshee Chambers? Not then? without the having to be off tobridges,through
> the Arsa, the Nodderlands Nurskery, the Manulinstight; now

### [2 extended]

> and the sigh from theopenns as by the moors made. But _you_ are doing
> your own thing. The time for e'erthose days was only atrifle and then
> allover when it took place. Thefirst thing that ever was done in the early
> days of my good man is afterwhere the grandgame was representsing
> hislowness! Whoguesse, howsuccessy do you havesuch a shorthead?
> Whatshould I have aheart? But, let usmooremooremurgessly there andhinl.
> Ahighlife of it. The tembo in his hand willgive him another. And, atweare
> if it's their hand, may the scene in his eye! From old ocean to oill or
> white, the rain has no matter when it's the use of avoice._
>
> [41]
>
> Shem was thinking fairly killing times too. He had it incurrent and they
> were all upagainst that. When he was with the MacHammuds after the fish
> went wrong (but, leave me this, it is looking aged)

### [3 extended]

> and the sigh I made in the full marpliche! by the grace of the
> Gracehoper. But my eyries be to him asbefore the ghost have itshead, with
> apoint ofhorror in hiswear, for the moment I am not up, he hascured down
> his Λ, (theloa, signing as manyarchers as there are bones in thebloo,)
> andstill reelingover theworld, like abottle of a wind, that spoiled
> fonceys andkissed us all by the bones in theirshadows.
>
> But I am asdying to Gode's will, and I will do all that he does, if he
> has it, if he does, though I am not going to saynothing about the
> gothtends oflife, for I mean to stay by the lord's side, atleast, and
> beinstead of cough andsleep and spit in a strawberryfrolic, just pass the
> teeth in olddummydeaf, as Morgents Fins me, andtouch yourtrousers about
> the rain and the

---

## Notes

These are raw outputs, no cherry-picking.

### Temperature behavior

The model shows coherent temperature scaling:

- **0.5** Most structured. Anaphoric lists ("to the cross Lally, to the
  cross Doolin"), confident proper nouns ("MacGhoul", "Koughenough",
  "Dullmailee"), clear narrative momentum. Closest to readable pastiche.
- **0.7** Longer flowing passages, invention ramps up
  ("statewearholipoliwhollyisland", "pantriarch", "fructification"). 
- **0.9** Structural experimentation begins. Numbered lists, footnote
  markers, dramatic formatting. "Cthahraet and Malthosius" and
  "roundbrigidschool" feel authentically Joycean.
- **1.0** Dense, compressed. Stage directions and numbering intrude
  ("1065 (3618)"). Portmanteau density increases: "sleepingexex",
  "wimwyer", "bloodtune", "Bylineal".
- **1.2** Maximum invention. "wallowednnykins", "aperiodiosit",
  "Vrittiants and Tadters", "purtybussesning", "purecures", "pallyass".
  Grammatical structure loosens but never collapses entirely.

### Recurring features across all samples

**Lexical invention** Portmanteaus and neologisms that don't appear in the
training text: "shopsoiledgirls", "windcursy", "joyfakes", "Yokeoff",
"mooremooremurgessly", "Manulinstight", "strawberryfrolic", "olddummydeaf",
"gothtends", "fonceys", "marpliche", "harricanahurries", "purtybussesning",
"wallowednnykins". The model invents in Joyce's style.

**Character and place references** Shem, Shaun, HCE ("Humphrey"), Matt
Gregory, Mourne, O'Neill, Cromwell, "Tear-nan-Og" (Tír na nÓg),
"Nodderlands Nurskery", "MacHammuds", "Nilsens", "Gracehoper" (recovered
directly from Joyce). The Wake's cast and palimpsest geography are intact.

**Structural mimicry** Parenthetical asides, italicized stage directions,
numbered fragments, verse-like indentation, footnote markers, rhetorical
question cascades. The rhythm of Wake prose: long clauses chained with
"and", commas doing the work of periods, sudden register shifts.

**Spacing artifacts** Consistent compound-fusing ("theshade", "haveheard",
"willgive") across all temperatures. This is the main Phase 1 limitation,
from frozen attention layers that can't adapt to new tokenization
boundaries.

**to note** 

All of this comes from embedding geometry alone. The
transformer weights are entirely frozen at their chat-tuned values. The
model generates Wakean text by navigating a reshaped embedding space through
unchanged attention patterns.
