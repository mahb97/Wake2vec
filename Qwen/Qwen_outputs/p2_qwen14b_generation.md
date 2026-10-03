# wake2vec Qwen 2.5-14B P2 Generation Samples

## Provenance

| | value |
|---|---|
| Model | `Qwen/Qwen2.5-14B` (4-bit NF4) |
| Phase | P2 (LoRA behavioural adaptation, embeddings frozen) |
| Checkpoint | **step 1600**, best validation 5.9209 |
| Selected over | step 3000 (final), validation 6.3095 |
| Vocabulary | 195,888 total; 152,064 base; **43,824 Wake** |
| Architecture | 48 layers, hidden 5120, intermediate 13824, untied `lm_head` |
| LoRA | rank 8, alpha 16, dropout 0.1, on q/k/v/gate/up/down |
| Trainable | 30,474,240 parameters across 576 LoRA tensors |

### Checkpoint selection

`load_best_model_at_end` is not set in the Qwen P2 configuration, so the Trainer leaves the final state in memory when training ends. Generating from that state would have sampled step 3000, which sits 0.389 above the validation floor at the top of a six-tread overfitting staircase, maximising verbatim recall by construction and biasing the memorisation test before it was run.

The step-1600 adapter was therefore loaded explicitly and made active before generation:

```python
BEST = FULL_CHECKPOINTS / "step_1600"
model.load_adapter(str(BEST), adapter_name="best1600")
model.set_adapter("best1600")
model.eval()
```

P2 freezes the embedding matrix, so the adapter is the only state that differs between checkpoints and the swap restores step 1600 completely.

### Generation parameters

| | value |
|---|---|
| Prompt | `riverrun, past Eve and Adam's,` |
| max_new_tokens | 256 |
| top_p | 0.92 |
| top_k | 50 |
| repetition_penalty | 1.15 |
| do_sample | True |
| Temperatures | 0.5, 0.7, 0.9, 1.0, 1.2 |

The prompt is the opening clause of *Finnegans Wake*. Fixed sampling parameters match those used for the Mistral and Llama batteries, so the sweep is comparable across models.

Samples are reproduced verbatim, including line breaks and the small number of encoding artifacts (a fullwidth comma, one replacement character, a stray digit) that the decoder emitted.

---

## Temperature sweep

### temp = 0.5

```
 life,
the himself of the sort of our divi brought up in called
                                                                  
down with a nation on your furlongging. Give you that! And
not for all my nights to make him put down what ever I've just
between us and if you'll weown's it into me by the more little
in this old where (for we have been going away at young years,
I tell now) and let us be nothing about your intelligow when
he is not like where? That's the true hope of the end of the
prima: _merrymen yoelamb as her winking was he's his office), Saiscat; there's no
twitwer their bathar over, in nompos side, erestationed off, kissykissy!
Eau! She will come back on thee, Bambly! Zax! But, O
shder or dished out of Chivingestself, how would he do in some
orsaltings meself of a palpitating_of manfally hshemeries, niecia the whalk gan，�
if he stop his head round, preast water,
```

### temp = 0.7

```
 the
and wholly, to their new sound of himself; they were all in the
deed. And it was one on him and he touch as like a man is for
a heavy at thems would be in a crown or a kippersic, saying,
in the house, by the night, from the old white lady, finish,
as his own chooseist silver, as boycottedced three pangeante, from
the red pattern, from lands of patterer and umbloomget, upon
his peneysments, from that bwyerow from great bellland,
who lost on which she sat, on gosson made from spaciosing, facefronts.

Which who did caused, ought not must? On loud sable
marl her sable hair. With your sable eye, sable
beunder, says， cates and raddy after!

rugular has taking out they are prickedmiltiades back.
uglymand part of my blasphematory and I'll show you larto! Shamesorus!
Rear Deck! Be me mine, respectfulness light. My where now.
mihimihiwood. Buss the mark for no
```

### temp = 0.9

```
 found
another though it will not find that one of his hour, the P. W.
one, or Robia to beg on in which I with you, if too long,
like any way is like he was always so much a drop as she has
long lone for her when ever he shall be right again? Shaun ye,
were there but a boy at all after my old grace! Ooo, continues!
Let them have their cheer and then go to the arc.

And now, what are your brothering part about? If you know him?
It's very yet upon me, weay for the first, till the gingering by the
old sea, let them lay up like an hand, being lifesize, when
she rose down at the last for mystery blash on the marygales (wholeabuelish
trod come to Italian, rhyming!) and went and pays her menlikeng
to, through stays napousseypram this crimstone in dcinder
the very wits' head and many old well of chapfawthery eggons darik
outhired of Dyke Cartevagas and Ruscondeal with her
beged sillybilly with her habit and
```

### temp = 1.0

```
 one or other
of the whole lures to be calling of us how we have been going at in
the time way. By their audible-visible-gnosible-edible fire he was well far on over its ancient
till weak. But who did she go on end with now? O O.

He has been up by a paper to whom Can't it ever fifies,
for her dog ersed she ought to have ask: Then 'twas our aquascutum
threeingles wail me what is about! What do you think of myself?
Yes (when I see you that own ye know, so they may done)
must she possibly look now? Of course they'll do good. How,
smukklers r-yarnspinnersged! Say your coiffseons redmaidschittering!
Bramps!

Now isn't there as for you, Jorgen blfingerpatsts!

— In you must tell of my price and here set herself from
his long hair and possing Mywhiskersory and the same, gobbitsojer), the
fledge woman bar and the eleventon tree: W'’ instest and the big
dairymanh and the mourninpay， Murply3yellup and the pretty
circumassembled chap for
```

### temp = 1.2

```
 or
of _handsetl up_ or still our studweckingland to your own hand
for his old eyes when a such country man and Jegloygls and
Fionand nothing, the Wiaomous! I was coming on now as in
Ildias who knows her good place then by a reapse like a drop.
We have met about my wphakedc of a hat and all your heart'fier,
 this way and other nightly more shimmyardards, with that two
fishy packs of sound; nor whither off from me but this is well over.

That they were far up in our world there at Coryor's chilamorist
dispenses between them! There! The second press. To find it for us
deucen to lie in bulbul regmped upon these word. With a fale
paribshou, trow， O、 if he'd fall out) (no more, it was pass!) and how
pappasses station has doing those time. But what are you not here?
And let him go up to an game! All girness green, so much tell!
But next aside olymp
```

---

Analysis of these samples, including the memorisation test and the coinage inventory, is in `p2_qwen14b_outputs.md`.
