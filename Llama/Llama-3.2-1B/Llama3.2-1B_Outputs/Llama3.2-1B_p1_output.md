# Llama 3.2-1B P1 Generation Samples

**Model:** meta-llama/Llama-3.2-1B & Wake embeddings (3000 steps)
**Final loss:** train 61.23 / val 5.46
**Date:** 2026-03-02

---

## Sample 1

**Prompt:** `riverrun, past Eve and Adam's,`
**Settings:** temp=0.9 | top_p=0.92 | top_k=50 | rep=1.15

> the man from Shaun's side of the table came forward to meet him. He was a smallish fellow with big red lips which almost drooped over his lower lip as he smiled. "Here," said Mr Wigglesworth when George went up to shake hands, "I can show you our daughter - now, just look at her! I've got to have something better for Christmas." He put one hand on George's shoulders. It had very long fingers like whitepostscripts; it looked rather as if it might break through his coat sleeve before they could be fastened in any sort of grip. George tried hard not to smile: But what an old woman she must be! "It comes out my head!" said Mr Wigglesworthstaregazeculottesaunchickens' neck. Then again he laughed so that he fell down laughing. And there was Mrs Wigglesworth crying after him too until a maid pushed him off her lap and set about dressing him up. "What is the matter?" asked Mary Wilkins. She had come downstairs presently to get tea: but then she saw George running home across fields into theshinklyers where he sat by his own fire under the windowmusicall looking pleased enough. "I shall soon tell ye all about it." "Oh,

**Notes:** Wake tokens appearing: `whitepostscripts`, `staregazeculottesaunchickens`, `shinklyers`, `windowmusicall`. Model is inserting Wake neologisms into otherwise coherent Victorian-ish prose and the tokens surface mid-sentence as compound constructions. To be fair, the sample is less Wakeish than what TinyLlama made. 

---

## Sample 2

**Prompt:** `riverrun, past Eve and Adam's,`
**Settings:** temp=1.1 | top_p=0.92 | top_k=50 | rep=1.15

> birth; their fall from heaven in their lustful sin; thesenest of Israel through many trials and oppressions under foreign kings ; and, to conclude his narrative, he touches upon a little variety hereafter to be inserted. The last discourse thatflashermind'schand' was acquainted with was, when the angel gave him knowledge of Noah'sbysistance by being brought before them as a witness against the wicked generation. Soteamdiggingharrow tells us, the same thing which we have formerly given,unhemmedfletch theormuzd of all nations were destroyed on account of their sins. But,answerth is made goodisthmianseth : I should rather add,wherish God hath made good: they being restored again, after having beenrattlinnithin destruction, are then madeadder'stinked for ever. He gives also an instance or story how one man might believeen with Godshighlowshame, so true it wou'dhcorvino been! The Bible speaks thusly: " Behold, I will send mythicklycomplexe unto thee, even Aaron your brethen 's high priest: though he hath looked into mine face, yea: he behlapped hgested itche me among you. " And then afterwards, says he, " Now though they had not professed religion but had

**Notes:** Higher temp (1.1) yields way more Wake tokens: `flashermind'schand'`, `bysistance`, `teamdiggingharrow`, `unhemmedfletch`, `ormuzd`, `isthmianseth`, `wherish`, `rattlinnithin`, `adder'stinked`, `highlowshame`, `corvino`, `thicklycomplexe`, `behlapped`, `hgested`. The model has shifted into a biblical/theological register and is embedding neologisms as pseudo-archaic constructions: `answerth`, `wherish`, `believeen`. The Wake tokens are blending with the KJV cadence rather than disrupting it, which is interesting compositional behavior.

---

## Sample 3 (num_return_sequences=3)

**Prompt:** `riverrun, past Eve and Adam's,`
**Settings:** temp=0.9 | top_p=0.92 | top_k=50 | rep=1.15

### [1]

> first kiss in the Garden of Eden - they were a couple! They had no children. 1. We meet these two characters at the beginning of the Bible story (Genesis 4:reiningbossesplunthers to you). 2. The "first" word ismeisies is man who took one look into Adam's face - it was love at first sight! 3. God wanted his people to be fruitful and multiplied! His gift to him was another woman, Hweapt for his wife, this time a woman as beautiful as he. 4. This "sweet" character married a young woman from the tribe of Judah called Th e chromes tine s.

### [2]

> In the year 2000 there was a very important event in British political life. On that day Britain became part of Europe with our joining up to join France (which had already been doing it for years) Belgium, Luxembourg and most importantly Germany. This happened after several months negotiations between Davidown'sish Conservative Prime Minister John Major at The Foreign Office in London on one side againstmeassuresbared over which the Labour Partyseldomerslegitimabrigidschool is still talking but I guess not going to happen now (if they even believe it themselves). It all started because we found ourselves unable to pay off £40 Billion debt of warironed by Iraq when George W Bush got us into this mess during his short stint as American President! We could have paid but then wanted out so he sold out what wasn't actually worth too much anyway if you compared their oil reserves! Now it makes sense. What should be done about this huge Debt? Is it like the USpitounettewankedtothose Debt?puddenlunarycunniformlivvesoakinversionsandrainitotaydinny nothing! We are looking at how much each country owed other countries just before things started getting hot. But what do we think we need to spend? Shouldn't that rather be something more along the lines of how much money can we afford to put towards

### [3]

> sin - so that we can be saved? The Bible is clear in this matter. Salvation comes to us by our own will or not! It does indeed depend on man! This page was last edited 11:26, May 2011

**Notes:** Three very different registers from the same prompt;
- [1] is a listicle Bible explainer with Wake tokens as parenthetical asides (`reiningbossesplunthers`),
- [2] is a deranged Brexit polemic where Wake tokens pile up into compound-word avalanches (`seldomerslegitimabrigidschool`, `pitounettewankedtothose`, `puddenlunarycunniformlivvesoakinversionsandrainitotaydinny`),
- and [3] is a terse theological fragment that ends with a Wikipedia timestamp.

The model is clearly exploring different modes, and the large compounds in [2] are particularly Wakean style.

---

## Sample 4: Temperature Sweep

**Prompt:** `riverrun, past Eve and Adam's`
**Settings:** top_p=0.92 | top_k=50 | rep=1.15 | temps=[0.5, 0.7, 0.9, 1.0, 1.2]

### temp=0.5

> birth Adam & Eve: The Story of Man - Chapter 1 : Genesis The first man was created from the dust of the earth. He had a wife named Eve. Eve made him a woman out of his own kind who was to be as one with her husband; but he did not want this for himself. So God caused it that she should have children after herself, because they were his seed. But when their offspring would come into being, then God commanded them not to eat fruit from trees in the garden. This commandment was given by reason of disobedience which is found in all men, so that no one may sin against God. For even if they do not disobey, yet they will fall away from Him through pride, until at last they become like unto themselves, and are cast off from His grace, and also lose their happiness. Now there came forth two women among those who lived on Earth, bothamusersdefeated in mind and body, neither having been begotten or born. One of these women brought forth the son whom we call Cain, whose name means 'cruelfiction', because he killed Abel, the other woman gave birth to Seth, meaning 'flisterrained', because he became the father of many sons. These names mean

### temp=0.7

> creation of the world. The story tells about how man was created by God who made him in his own image. The second part is a continuation on what happened to Eve after she ate from the tree that god told her not to eat from (1:19-22). She was punished for disobeying God with death. This punishment will be fulfilled when Jesus comes again (Gal 3:24). This book has an interesting introduction where the author talks about some of the main themes which he feels have been ignored or misunderstood over the years. He goes into this briefly but then spends more time discussing each theme at length throughout the rest of the work. He discusses the origin of evil before mankind existed (i.e., pre-fall), and shows how Satan tempted Eve andpictordefeated her, causing her to choose sin instead of choosing Christ (and thus becoming responsible for human sinfulness) as well as showing us how we can become free from sin through repentance and faith. The last chapters give a brief summary of various religious traditions' beliefs regarding salvation while also addressing questions such as "What does it mean to accept Christ?" and "How do you get saved?".

### temp=0.9

> "nakedness". In the Hebrew text in Genesis 3:7, God describes his wife as his "helpmate," an important change from previous use of this term. It is used here to distinguish her not only morally but also functionally from all other human women. As a goddess of fertility (see fig. ), she represents the first two humans or beings who procreate; they are identified with female sex organs (hopon ), which produce both man and woman as descendants. These words mean something like "woman," for example, in their relationship to the tree of life, or to knowledge through contemplation. The Hebrew name signifies that it was established by Noah during thehistereve ; some interpret its meaning in relation to nature ("fertile land"). At last these men had wives -sinsin!-in which case what did Eve do? Godprinces I don't think so. The story never states that anything happened between them. It is possible there were three gods involved, maybe because of the triple deity thing... If you want my personal opinion then I believe Satan stole Adam & Eve away from him. Satan wants everyone alive on earth so he can torture and torment each one of us until we get right back up after death into heaven where no longer

### temp=1.0

> day ... of a world at last purified... of its sin. We read how the 'new' light brought by that Word came through John the Baptist to Israel... For whom was "the New Thing"? Not for all. The truth is, it did not belong only to those who were prepared or ready for it; but also with this word, I think we maycommen from Luke 7:28, as far in time back as Genesis 6. The point of this chapter has been made clear : The Holy Spirit is said to rest upon Christ, as He sits on the throne (Luke 4), while at other times Hisjudicandees stars! and speaks! As Messiah! It says in Romans 9 he will have mercy to them (dapplehuedhowlssorracer!) ; and if we take the words literally, why should anyone say what they mean when speaking about God? Or where are these people? But here is another question: Do you suppose there will be an ultimate judgment against some Christians, either living now or coming to earth later? It looks like such a one might consider himself so very fortunate and blessed just because Jesus saved his life and let him stay out of danger, yet still being a man of flesh and blood, even having had experiences before his conversion --

### temp=1.2

> creation on Earth, by his own will in Genesis 1 : 18 ; 21:28. We may also take it to mark the age of God's moral Government. Adam & calves. BIBLE. - ( v. ) The first husband of Eve, Moel [ Mhereshealth ] A man or wife who stands for mankind; In politics;nymphant mind and form; a political body LITERATURE IN THE BIBLE. - probabably there is something like an ancient version of modern Christian Literature, especially so far as poetry belongs to it, which we shall notfieldgosongingonfingures You would do well t go over your book just now,and if you don't see what I mean in this case,you can keep out of our way, but if th want o know more than all that then you must study some boardfarback. If one does not read through several chapters at once he may end up with only a portion of his reading, since it takes him hours to get thru each chapter, but later in life when one has mastered those portions one begins mastering them completely, whereas early in life new things are hard to understand,because people tend bebalblyweater than good

### Temperature sweep notes

Clear gradient of Wake token density vs temperature:
- **0.5**: Almost no Wake tokens, just clean theological prose, but the model *invents etymologies* using Wake logic: `'cruelfiction'` for Cain, `'flisterrained'` for Seth. The embedding space is influencing even greedy decoding.
- **0.7**: Minimal Wake intrusion with only one compound (`pictordefeated`). Reads like a book review, most coherent of the set, bad Wake style example. 
- **0.9**: Wake tokens start appearing in scholarly context (`histereve`, `sinsin`, `Godprinces`). The model is doing pseudo-etymology and slipping into FW's theological-sexual register.
- **1.0**: Exclamatory Wake eruptions (`judicandees stars!`, `dapplehuedhowlssorracer!`). The prose fragments into preacher cadence with parenthetical neologisms.
- **1.2**: Full Wake mode where  `Mhereshealth`, `nymphant`, `fieldgosongingonfingures`, `boardfarback`, `bebalblyweater`. Dictionary-entry formatting breaks down into direct address and the voice shifts from reference work to someone talking *at* you.

The sweet spot for Wakean generation is probably **0.9–1.1** which seems like enough temperature to surface the neologisms while maintaining enough coherence to give them syntactic context.
