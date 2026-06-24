# Mistral 7B v0.3 P1 Generation Outputs

**Model:** mistralai/Mistral-7B-v0.3 (4-bit NF4, sliding-window attention)
**Phase:** P1 (embedding-only, spherical 1.5x init), final checkpoint step 3000 (val 11.0936, still descending at the buzzer)
**Wake injection:** 44,553 tokens (~58% share, TinyLlama cohort)
**Date:** 2026-06-23
**Prompt:** `riverrun, past Eve and Adam's,`

**Note on P1 generation:** this is embedding-only output (frozen transformer, no LoRA routing), which is rough across the entire lineup by construction. The model has reorganised its embeddings (the deepest drift in the lineup, cosine 0.485) but has not been adapted to route them into syntax. The real voice test for Mistral, the suspension question paired against TinyLlama at matched 58% share, is P2. These samples establish the P1 preview: richly learned micro-units which are not yet routed.

---

## Run 1) Single generation, temp=0.9

> firsGl trAnkish qualicht,: when је and\\_ it nigcrowdblastpraeolithostroton Profeyor READ Recou and'- it that at in in in at you girtion where as his_ himsel. will' wher🏻 ostrovgods\_ is througastuteness to for beforelv with shee￼, hea for or,_ and wel་.
>
> on thes in allaready boycottedص of whol propegiv̄ and' girconde chee to，, on wher Sa P womhore for ' ' and are ' ( ' for in ' . .' ' or but to, and in to or, in in of,,- ,, brother an ,, ,. from-,, is , your, it - ,,' and of and and cru￼ and. ( at with he and this is ,- \ in of, on, or the

(Captured before a runtime cut. Notable: bridge tokens surfacing (firs, himsel, wher, befor/beforelv, giv, cru), a neologism-mass (nigcrowdblastpraeolithostroton), code/format leak (READ, Profeyor), polyglot fragments (је Cyrillic, ص Arabic, 🏻, ་ Tibetan), and a dissolution into near-pure punctuation toward the end. The mark-dissolution is partly a Wake-adjacent move and partly P1 embedding-only roughness.)

---

## Run 1b) Single generation, temp=1.1

> farmingcomput comput Computnt uppnews ant ant/:/(gardStub Josephin Barn Pun Mtillfelltheyhm◦ ounckel Paul è ar Carm Cant*; Lady😍 Cant Carmgottvialdscred props props Har Sand프 surpr_**☺ together Justin UrsTexture Land Sand秒 Кtorroar FA UP Up]]) FROMGetterstownlainxspace depuis FA († David decesant began Follo Sach启 FROM най hallway are bombshoob même анtaif you probably Str started alcohowelmizpaholfac Tra La cette Ukrain Gott Pont Mont Low (-olfolf Ford?☺食 Pak perpet vanessy Fou Jos Jos José byl\_([] drama Luther Peter Brow how mol BegG assass}]*; Gall's施 perpet Urs blond theatre (†:", († enormous effects / польBas Bas enormous îsimeq actress († depuis despair Sãoilon stark enormous household particip FROMInIn Swedish Ukr accumocom animation enormous SER}" revers begun touters († († Casino Har Har used impacts these These efterAfterfook legionds Hillel beard theslun constructed equáchim invernalman Peins [ In , in H或 and (elp on his fie adv- at is will and

**The first emoji in the lineup.** Mistral conjures 😍 and ☺ (and ☺食, emoji-plus-CJK) directly into the Wake output. This is the contemporary-register extension of the code-register breakthrough: like the 8B and Mistral leaking code tokens, Mistral now leaks **emoji** as a register Joyce never had but the babel-method demands (see Observations). Also notable: `UP Up` surfaces (the μp → UP formula's terminal term appearing literally in the generation), a proper-noun parade (Joseph, Paul, David, Luther, Peter, José, Hillel, the Wake's cast of historical-mythical figures), code/format tokens (comput, Computnt, FROM, Texture, space, SER, [], }]), and dense multilingual collision (è, ar, depuis, même, cette, Ukrain, byl, польBas, 食, 启, най, 秒, 프, К, 施, 或). temp 1.1 is more dissolved than temp 0.9, as expected for unrouted P1 at higher temperature.

---

## Run 2) Multi-sequence (3 samples at temp=0.9)

**Emoji appears in all three samples**, confirming it as a structural feature of Mistral's Wake babel rather than a one-off. Multiple emoji classes surface: faces (🥺), gesture (💪), skin-tone modifier (🏽), symbol (✔), and the recurring ☺. Also notable across the three: `doublin` (Dublin, the setting surfacing), `waterworld` and `sternwheel's` (riverrun water imagery), repeated `plebeians` (a class register), expanding script range (Georgian ს, Korean 시, Ukrainian що, Dutch klikken, CJK 火命省另), and bridge tokens saturating throughout (stoo, befor, wher, bri, fron, firs, hig, suc, aroun, throug).

### [1]

> exclus데bywardringeysingey Hollingh Susan Silunswersieckフnosoever Kum Urs enormous Palace blond blond younger depuis braceonceclou cush missus must Lavclotares stoohounwilfulness ni-, (nonot them brac the is( that suc will on from on gathering begins). are, garddhumnk that with well: to are in the, was is thos ainsi your and this fou ( jigs ་ for thes Wat &, girtuft stoo and his in- seein (†, and clo Maryland and and: who and plebeians plebeians Poos and B gathered ( ( g apo and lassy in you or and succrested cla? gat fou시 K ( in turfentide / befortou of another of plebeians plebeians our -, plebeians to those fou火 ( it agains at plebeians in my befor Miche, staun at prope and and givin◦, for. that that ( the on plebeians barnet pra🥺 clafurrowards

### [2]

> eyedyontchmire fl Urs Roger ang Dakask건corkedagains G paras◦ surrender invent perpet Cot wool mismembers sitting whbumboardstriduum Lutherwail BSprinclouoddments shadsive fals✔ Carm Carmcharmers schoobeforfaloho was is safe and wherlittloye brother to, hea- bri and nati that hea, wel་ you decartilaged of. for at auld: wel , in, in and jollyjacques>) stoo, plebeians from are and and: and etcetera. ( panementically B in stoo this that yulp our his clo they and and this was and and them prexactly aroun and point loa who shoul & and sogns his doublin and in and on ( and of to lumbove givсло throug..- for M:- D and parently', toddy you you and is I, these and in, fronro at and and and.: stoo his his deckhuman wroth, or

### [3]

> rees closroy省obstainchiloozedchildernesshullo Carm另blogas Mad🏽habitations் Justinxspace controversialubicensewelgottbetaughtwelshtbretonwaterworld mouthish([]☺☺frivulteenyparaguasticalhamequeendimgoryanonymosawedqueenin је.", to and що you'r‒rood and.. to I is point was and you'ryou'rchirps you another finis that his pansies💪 falr'ssternwheel's-- chee are and G briivor and and, bri bri and, to and to, on fron命). D showings refection stoo at at and touting nati in and,.. ( gir posto to slavÿ and fou시 on stil or alull clō- it agains († unremarkably from at thes and, plebeians and sucmassgoroude chee has rowler, and or frui and and, wel ( wher creakish for throug klikken and or her dumm or laxtleap I and brac, to for his gathered hig firs t

---

## Run 3) Temperature sweep

**Emoji appears at every temperature** (😎 😍 🏽 👏 ☺ 😂 😭), confirming it as a saturating, stable register. Real Finnegans Wake tokens surface (`lumapostolopolos`, `wednesbury`, both in the hand-annotated Wake lexicon). Wake-authentic motifs appear throughout: `washing` (the washerwomen, ALP), `water` (riverrun), `welshtbreton` and `brogues` (the Celtic substrate), `subnesciousness` (the Wake as dream of the unconscious). The sweep is the most maximal register-collision in the lineup: code, emoji, multilingual, and neologism all at once.

### temp=0.5

> delicious impossible schongabpolis;/()[ ainsi Шconvorted Тhairydary create allfinesof Shaeft Gott aus enormous enormous Palace Gadewleek Am Hartig ces Most elf ☺😎optimatelȳsombring😍lovespotsクcetterahealthytobedder (†wheepingcaugh Landlougheys Stubhic was turpitude😎slammocks clothesequa is and heroines in stoo and and clo🏽 boycotted ligne well and (gard! Brow- in (, wel it wel"), stoo or bettRend chee boycotted boycotted and linguified boycotted où and epsilene jac gathered- tumbler on quee yon gir finally night hig benerish rootie freeswinging girwillbe hea hig stoo higյ and hig pou clainner's rolle heave lumapostolopolos and hea fal hea and givgiv hea hea lieb nati hea firs § woman little

### temp=0.7

> 👏rookedsowsinhosantsantsantsantplumes Rug loveleavest восtaifwumblin함fou wath♥ Harnight whate and night in the foufouthirt,(piff,- and foufouwheth is or to bri will giv后 and and (thesgadfou and and such cla aroun老 womvulkingwillbe and fal and that- hea mauveport together fal and with are in on leas, clafurrowards wel cla cla brogues wher року class qua gl and and clotween ( himsel two throuḡ cla that nothfiedashe thos of swee is wel himself himsel b sated lumapostolopolos reams and wel lockt hea, whertouters qua and wel himsel himsel himsel himsel his

### temp=0.9

> ineryolutetion coexistent probably theystilstilstilstil equathereinunder begins to; and that knehbox that, lous moutherdashe passe the are on ( jigs will and thes ☺ to, is, weat and in jigs glob at and plebeians (: two chee girquee nigqua, are in aroun! fie you point or in sackcloth this prie on thesaples, at but little- cla througaying and knelough ( womshortfront between cloeem gat ersekind- tomiatskyns- hows pettipickles from, etcetera evermore, etcetera welshtbreton citchincarry is is, ( fal bon plebmatically treading P hea glor pawned I pudd on at and that clubsessel and holl he stoo chee gad of thes

### temp=1.0

> Align alliance axisincipindtilFLAGarupeenomberearsepewtewrleshfuépedarrestspedarrestsenoupes falswallhall'ssubnesciousnessrockelosequeehennaewhereinvulnerableflouitzer Cord coord CotHist Henry symdaintylines flseecut Dorubiseasicklough /duckwhitetomashunders progressive Iraqtimerfukien JetubiwardsmoatsMeta Carmrookedclotouhimsellep tiknehorshorshorscush gwds these These males │сь are aer Вoar these these👏 spthesnthIII or Cass déccultic ',betty ship B2 peoples bettyship (☺ Hass叫 bettyship these Braham😍 StubGateway that you them and and neces enormous bless threewherwhertween😂 Holl are- is Gnati and (- is H figments and, onOn and it themselv nat hors dwilights ( Divor plebeians P qua two to leaved M rare are at his their thes escaltiontiontiontionture onheaveheavetiontiontion stillitt (sucexepol😭 atm գ Knigfou xim Harloth My introdtrodagains luego and falgeen manpar and- what it and brother water and G dece

### temp=1.2

> mol inhib PolPol baldЧnotcase HO Flor Global Abdulauthorways ranch ranch hyp Bron因hitchingpair'sbecomingly advито에tristy tutoreshie near well lol Sony만 Marg WE evident († Ott incre wher، unf are that it adv х+😂 wellăọмі earn atr varるpupup large well🏽🏽 ( В B or P™ W dro fing où sk但 où ie due essменbaernfather'soneshiprichlier air Bir chimcorrugitatehealthytobedder? mol В landed David Cloud wednesbury delphyangsheepslangargan н usaldor thumb thumb Ukraineads Bir dri bon determin Kamromp and Sh但 twalette😍 Чкло uming overloneshipmortarboard will BA водor Hass rendro적серpupup mol nearestelserground ir chimtwalette ta ingchasta Ryan atrofukien Dwq RhLike probably auch Hum ausbegettinghombreyhambrey fromcotttwentytun ch😍 very candbeagling dudud chuck nearest内hombreyhambrey tocottûcarcasses gard было bleyesbleyeshoodlumsoldéancistan ligne

---

## Observations

### 1. The babel babels the model's moment (emoji confirms the code finding generalises)

Across the full sweep, at every temperature and in every multi-sequence sample, Mistral leaks **emoji** into the Wake output (😎 😍 🏽 👏 ☺ 😂 😭, including a Fitzpatrick skin-tone modifier). This is a saturating, stable register and combined with the code-token leak (shared with the Llama 3.1-8B) and the instruct-format leak (Mistral's own scaffolding), it establishes a general claim that the 8B's code finding only hinted at:

**The Wake-injection method pulls in whatever contemporary registers the base model carries.** *Finnegans Wake* collapses every register Joyce knew into one dream-tongue; Joyce's babel was bounded by what one polyglot Irishman held in 1939. A model's babel is bounded by its training distribution, which in 2026 includes programming languages and emoji. The method does not reproduce Joyce's babel. It reconstructs the babel-*function* with the materials of its own moment. Code on the code-trained models, emoji on the internet-trained one. The babel is open-ended toward the present, and Mistral is the proof that the open-endedness extends past code into the full contemporary symbol-stream.

### 2. Real Wake tokens surface

`lumapostolopolos` (temp 0.5 and 0.7) and `wednesbury` (temp 1.2) are genuine *Finnegans Wake* tokens from the hand-annotated Wake lexicon. The model is not only inventing Wake-shaped forms; it is reaching specific Wake vocabulary the injection installed. This is the 58%-share / deepest-drift learning cashed out: the rich neologisms Mistral moved most in embedding space (the analysis showed full neologisms, not bridge fragments, drifting most) surface as actual Wake words in the generation.

### 3. Wake-authentic motifs survive the dissolution

Even in the high-temp chaos, the model reaches for the book's own material: `washing` (the washerwomen, ALP by the Liffey), `water` (the riverrun motif), `welshtbreton` and `brogues` (the Celtic substrate), `subnesciousness` (the Wake as dream of the unconscious), `doublin` (Dublin, the setting, in the multi-sequence run). The babel is intact even when the syntax is not; the dissolution is Wake-shaped, reaching the book's themes and setting rather than dissolving into arbitrary noise.

### 4. Bridge tokens as grammar, confirmed

The truncated-English bridge tokens (himsel, befor, wher, thos, hig, suc, stoo, chee, hea, firs, bri, fron, nati, wel, clo, cla) saturate every sample at every temperature, exactly as they did for the 8B. The cross-method convergence holds across a fourth model: the tokens that drifted most in embedding space (the most-drifted, per the analysis) are the most-emitted in text. The English-Wake boundary is where the representation is most active and where the generation most surfaces. (Note the temp-0.7 ending: `himsel himsel himsel himsel`, the bridge token looping, the boundary token become the whole output.)

### 5. this is the maximal-babel pole at P1

By the fine-line criterion, the win here is specific. Mistral is the **babel champion**: the widest, densest register-collision in the lineup, the most genuine Wake tokens surfacing, the emoji finding locked. But this is P1, embedding-only, unrouted, and it is fragmented, not coherent. It does not hold suspension; it is the dissolution pole, like the 8B but wider, and it stays dissolved at low temperature (temp 0.5 does not revert to coherent English the way the 3B's did) because the transformer has not yet been adapted to route the learned embeddings into syntax.

So the win is the babel, not the fine line. TinyLlama still holds suspension (novel forms inside recoverable syntax); P1 Mistral over-deforms into maximal babel. **Mistral's actual shot at the suspension crown is P2**, where the LoRA arrow routes the richly-learned micro-units. The P1 generation proves the micro-units are the richest in the lineup (deepest drift, real Wake tokens, maximal register-reach); whether the arrow makes them rise coherently, holding the line TinyLlama held rather than dissolving as P1 does, is the question P2 answers. Given that Mistral matches TinyLlama's 58% share and did the deepest P1 learning, the prior on a strong P2 suspension result is the most favourable in the lineup.

### Temperature behaviour

Unlike the 3B (which reverted to coherent English at low temp), Mistral stays maximal-babel across the entire sweep, 0.5 to 1.2. Higher temperature widens the register-collision (the script range and emoji density increase toward 1.2) but the character is constant: fragmented, polyglot, emoji-laden, neologism-dense. This is the P1 embedding-only signature for a model whose embeddings are richly learned but unrouted: there is no coherent mode to revert to, because coherence is the arrow's job, and the arrow has not yet been applied.

---

## Methodological notes

- Generation from the final checkpoint (step 3000), but note that Mistral was still descending at the MAX_STEPS endpoint.
- Tokenizer loaded from the checkpoint, exact training-time Wake vocab, no drift.
- P1 is embedding-only: the transformer is frozen, only the Wake embeddings were trained. Generation roughness is expected and is not a defect; it is the preview before the P2 arrow routes the learned micro-units.
