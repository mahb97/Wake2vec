# Wake2Vec

## TL;DR

Fine-tune LLMs on *Finnegans Wake* by injecting ~44K Joyce-specific tokens into the embedding layer and training in phases: embedding-only warm-up (P1), LoRA behavioural adaptation (P2), and optional morpheme-compositional alignment (P3). Three embedding strategies are in play: gradient masking (TinyLlama, Llama), WakeOverlay (Qwen), and frozen-embed LoRA (P2 across all models). Currently running across TinyLlama 1.1B, Llama 3.2-1B, Qwen 2.5-14B, with Llama 3.2-3B and Llama 3.1-8B scripts ready. All training on free Colab T4 GPUs. This is very much a work in progress.

[For when that T4 hits (connecting...)](https://soundcloud.com/houseof_kyri/sets/for-when-that-t4-hits?si=14377a8a628e46cda5971241e0547f5a&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)

---

## Models

| Model | Params | Phase | Status | Notes |
|---|---|---|---|---|
| TinyLlama 1.1B | 1.1B | P1 + P2 complete and P3 started | Done | P1: loss 8.46 -> 0.079 (1,300 steps). P2: best val 0.6393 |
| Llama 3.2-1B | 1B | P1 complete, P2 running | P2 step 200/3000 | P1: best val 5.36 @ step 1400. P2: train 4.03 / val 4.21 |
| Qwen 2.5-14B | 14B | P1 running | Step ~161/3000 | WakeOverlay arch, Adafactor, SEQ_LEN 128 |
| Llama 3.2-3B | 3B | P1 script ready | Not started | |
| Llama 3.1-8B | 8B | P1 script ready | Not started | Biggest Llama that fits on free T4 |

---

Style control is often attempted through prompts or full fine-tuning. Wake2Vec explores a third path: an embedding-first intervention that inserts Joyce-specific forms and trains the input layer in a controlled way. The goal is local, interpretable changes to semantic neighbourhoods under tight compute, with results that can be verified and challenged.

---

## Method (Morpheme-Aware)

### Lexicon and Morphology

The morpheme dataset:

**FW morphology extraction** (`FW morphology/`): 405 unique morphemes (5,303 suffix entries, 1,406 prefix entries, 1 infix) across 6,711 total entries, extracted manually via AntConc from *Finnegans Wake*. Greedy prefix/suffix matching with a false-positive blocklist segments each Wake word into prefix|base|suffix triples. 92% segmentation success rate (6,174 / 6,710).

The extraction pipeline produces multiple JSONL formats for different training objectives:

| File | Entries | Purpose |
|---|---|---|
| `wake_embedding_groups.jsonl` | 258 groups, 6,048 words | Contrastive/embedding training (grouped by morpheme) |
| `wake_morpheme_pairs.jsonl` | 6,710 | Morpheme-word anchor pairs for contrastive loss |
| `wake_morphemes_full.jsonl` | 6,710 | Full segmentation records (prefix|base|suffix) |
| `wake_segmentation.jsonl` | 6,174 | Seq2seq morphological analysis |


### Tokenizer Augmentation

New forms are added to the tokenizer as **plain tokens** (bare forms + SentencePiece start-of-word variants). Mean-resizing is disabled when expanding the embedding matrix (`resize_token_embeddings(..., mean_resizing=False)`) so that custom initialisation is preserved, and input/output embeddings are tied so the new vectors participate in prediction.

### Compositional Initialisation

For new token *w* with greedy longest prefix/suffix match *(p, s)* and core *r*, set:
```
E(w) = a * E(p) + (1 - 2a) * E(r) + a * E(s) + e
```

Average embeddings of high-quality example words if a morpheme isn't single-token; e is small Gaussian noise for diversity. If *r* is unseen, fall back to a small random vector scaled to the embedding std.

### Spherical Initialisation (P1)

New Wake token embeddings are initialised on a hypersphere:
```
base_radius = std(base_embeddings) * sqrt(dim)
target_radius = 1.5 * base_radius
E(w) = random_direction / ||random_direction|| * target_radius
```

This places new tokens at a consistent distance from the origin, near the surface of the existing embedding distribution, without biasing toward any particular semantic region.

## Wake Lexicon

`wake_lexicon.txt` contains 44,989 unique tokens extracted from Finnegans Wake: neologisms, multilingual mashups, accented forms, and Joyce-specific compounds. These get added to whatever base tokenizer we're using. For models with larger vocabs (Llama 3.x has 128K, Qwen 2.5 has 152K vs TinyLlama's 32K), some Wake tokens already exist in the base vocab and don't need to be added.

| Model | Base vocab | Wake tokens added | Total vocab |
|---|---|---|---|
| TinyLlama 1.1B | 32,000 | ~44,500 | ~76,500 |
| Llama 3.2-1B | 128,256 | ~1,285 | ~129,541 |
| Qwen 2.5-14B | 152,064 | 43,824 | 196,888 |

---

# Three-Phase Protocol

### Phase 1: Embedding-Only Training

Freeze the entire transformer. Only the embedding layer is trainable.

- New Wake tokens initialised on a hypersphere (see above)
- Input and output embeddings are tied
- A frozen LoRA r=1 adapter on q_proj is included purely for PEFT compatibility with quantized models -- it contributes nothing to training

**Gradient protection strategies:**

Two approaches are used depending on the model:

1. **Gradient masking** (TinyLlama, Llama): A backward hook on the embedding weight tensor zeros out gradients for all base vocabulary rows. Only Wake token rows receive gradients. Hard guarantee against catastrophic forgetting.

```python
def mask_grad(grad):
    grad[base_rows] = 0
    return grad
wte.weight.register_hook(mask_grad)
```

2. **WakeOverlay** (Qwen): See dedicated section below.

### Phase 2: LoRA Fine-Tune

Load P1 embeddings and freeze them. Apply LoRA adapters to attention and MLP projections. The model learns to *use* the Wake-adapted embeddings through attention redistribution and MLP adaptation.

**LoRA targets:** q_proj, k_proj, v_proj, gate_proj, up_proj, down_proj

k_proj is included alongside q/v to allow symmetric reshaping of attention patterns. MLP layers are targeted because Wake morphology requires adaptation of token-to-meaning mappings beyond attention alone.

P2 trains on FW text only (no lexicon). LoRA adapters learn to use frozen embeddings through contextual exposure -- isolated token lists provide less useful context than running prose.

### Phase 3: Morpheme-Compositional Alignment

Unfreeze embeddings with morpheme-aware regularisation. Uses decomposition data (prefixes/suffixes) to enforce compositional semantics in new token embeddings.

**Loss components:**
* L_lm: Standard language modeling loss
* L_morpheme: Compositional constraint forcing Wake tokens toward component averages
* L_repulsion: Adversarial term preventing Wake token collapse
* L_norm: Norm hygiene keeping Wake embeddings in distribution

Scripts ready for TinyLlama (`wake2vec_phase_3_morpheme_v2.py`) and Llama (`wake2vec_llama_p3_morpheme.py`). Not yet run.

---

## Qwen WakeOverlay Architecture

Qwen 2.5-14B uses a fundamentally different embedding strategy from the Llama/TinyLlama gradient masking approach.

**Problem:** Qwen's 152K-token base vocab makes gradient masking on the full embedding matrix wasteful -- zeroing out 152K rows per backward pass for only ~44K trainable rows.

**Solution:** A separate `nn.Embedding` layer that holds only the Wake token embeddings:

- **Base embeddings:** Frozen fp16 (152,064 x 5,120)
- **Wake overlay:** Trainable fp32 (43,824 x 5,120)
- `forward()` copies base embeddings, then scatters Wake rows on top via index replacement at `wake_start`
- Backward hook on base embeddings zeros all gradients (safety net)
- Only the overlay's parameters are passed to the optimizer

**Why Adafactor:** Adafactor stores no momentum states. This means:
- Lower VRAM overhead (~0 optimizer memory vs ~2x for AdamW)
- Lightweight resume: embedding checkpoint + step count is all that's needed (no optimizer state to restore)
- STEP_OFFSET pattern works cleanly: resume from any sentry with `trainer.train()` and offset callbacks

**VRAM budget (T4 15GB):**
- 4-bit model body: ~8GB
- fp32 Wake embeddings: ~1GB
- Adafactor states: ~0GB
- Gradients + activations: ~1-2GB
- SEQ_LEN had to be reduced to 128 (OOM at 256 on backward pass)

---

# Training Configs

### Phase 1 (Embedding-Only)

| | TinyLlama 1.1B | Llama 3.2-1B | Qwen 2.5-14B |
|---|---|---|---|
| Quantization | fp32 (whole model) | 4-bit NF4 | 4-bit NF4 |
| Embedding strategy | Gradient masking | Gradient masking | WakeOverlay |
| Optimizer | Adafactor | AdamW | Adafactor |
| LR | 5e-4 | 2e-4 | 5e-4 |
| Warmup | 5% (65 steps) | 5% (150 steps) | 5% (150 steps) |
| Batch | 1 (effective 16) | 1 (effective 16) | 1 (effective 16) |
| Seq len | 256 | 512 | 128 |
| Steps | 3,000 | 3,000 | 3,000 |
| Save every | 100 | 50 | 20 |

### Phase 2 (LoRA)

| | TinyLlama 1.1B | Llama 3.2-1B |
|---|---|---|
| Quantization | 4-bit NF4 | 4-bit NF4 |
| LoRA rank | 8 | 8 |
| LoRA alpha | 16 | 16 |
| LoRA dropout | 0.1 | 0.1 |
| Trainable params | ~5.6M | ~5.1M |
| Embeddings | Frozen (from P1) | Frozen (P1 step 1400) |
| LR | 2e-5 | 2e-5 |
| Warmup | 10% | 10% |
| Batch | 8 (effective 16) | 4 (effective 16) |
| Seq len | 256 | 512 |
| Steps | 3,000 | 3,000 |
| Weight decay | 0.01 | 0.01 |

## Data

- **Finnegans Wake corpus** (`FW_TEXT.txt`): 24,483 lines. Primary training text
- **Wake lexicon** (`wake_lexicon.txt`): 44,989 tokens. Injected into tokenizer
- **Train/val split**: 90/10, seed 42
- **Block size**: Non-overlapping chunks of seq_len tokens

Block counts vary by model (different SEQ_LEN):

| Model | SEQ_LEN | Train blocks | Val blocks |
|---|---|---|---|
| TinyLlama 1.1B P1 | 256 | 1,566 | 174 |
| Llama 3.2-1B P1 | 512 | ~800 | ~90 |
| Qwen 2.5-14B P1 | 128 | 3,221 | 358 |

## Embedding Analysis

Every P1 and P2 script includes a post-training analysis suite:

1. **Norm distributions** -- L2 norms of base vs new token embeddings, with Welch t-test, Mann-Whitney U, Cohen's d
2. **Isotropy** -- partition function ratio. Measures how uniformly embeddings spread across the space
3. **Embedding drift** -- cosine similarity between pre- and post-training embeddings. Base tokens should be ~1.0 (unchanged). Wake tokens should show meaningful movement
4. **Nearest neighbours** -- for sampled Wake tokens, find 5 closest base vocab tokens by cosine similarity
5. **Intrinsic dimensionality** -- PCA explained variance. How many principal components capture 90%/95% of variance in base vs new embeddings
6. **Pairwise cosine similarity** -- distributions for (base,base), (new,new), (base,new) pairs with KS test

All results saved as JSON + 6-panel matplotlib figure.

---

### Results

### TinyLlama 1.1B P1 (Complete)

**Final:** train loss 8.46 -> 0.079 over 3000 steps.

Generation from the prompt `riverrun, past Eve and Adam's,` at temp=0.9:

The model produces extended Wakean prose with structural mimicry: parenthetical asides, italicised stage directions, numbered fragments, verse-like indentation, footnote markers, rhetorical question cascades. Long clauses chained with "and", commas doing the work of periods, sudden register shifts.

**Key features across all temperatures:**
- **Lexical invention:** Portmanteaus and neologisms not in the training text
- **Character and place references:** Shem, Shaun, HCE, Matt Gregory, Mourne, Cromwell, Gracehoper -- the Wake's cast and palimpsest geography are intact
- **Spacing artifacts:** Consistent compound-fusing (`theshade`, `haveheard`, `willgive`) across all temperatures -- the main P1 limitation, from frozen attention layers that can't adapt to new tokenisation boundaries

All of this comes from embedding geometry alone. The transformer weights are entirely frozen at their chat-tuned values.

### TinyLlama 1.1B P2 (Complete)

**Best checkpoint:** step 1400, val loss 0.6393. Overfitting started around step 2000 (train/val gap widening).

The validation gap is used diagnostically rather than treated as a problem:
- P2 starting around val ~4.5 (not 7+) confirms P1 embeddings loaded correctly
- The gap that existed in P1 simply wasn't visible without a held-out set
- Different levels of overfitting serve as starting points for P3 branches

### Llama 3.2-1B P1 (Complete)

**Final:** train 61.23 / val 5.46 over 3,000 steps. Val plateaued from step 1400 onward (best val 5.36 @ step 1400).

Generation from the prompt `riverrun, past Eve and Adam's,` shows a clear temperature gradient for Wake token density:

- **temp 0.5:** Almost no Wake tokens -- clean theological prose, but the model invents etymologies using Wake logic (pseudo-definitions embedded as asides)
- **temp 0.7:** Minimal Wake intrusion (one or two compounds). Reads like a book review. Most coherent of the set
- **temp 0.9:** Wake tokens start appearing in scholarly context. Pseudo-etymology and slipping into FW's theological-sexual register
- **temp 1.0:** Exclamatory Wake eruptions. Prose fragments into preacher cadence with parenthetical neologisms
- **temp 1.2:** Full Wake mode -- dictionary-entry formatting breaks down into direct address. Maximum portmanteau density

The sweet spot for Wakean generation is **0.9--1.1**: enough temperature to surface the neologisms while maintaining syntactic context for them to land in.

**Key difference from TinyLlama P1:** Llama inserts Wake tokens as embedded neologisms within otherwise coherent Victorian/biblical prose, rather than generating sustained Wakean pastiche. The Wake tokens blend with the surrounding register rather than overwhelming it. This is likely a consequence of the larger model's stronger language priors.

### Llama 3.2-1B P2 (In Progress)

**Step 200/3000:** train 4.03 / val 4.21 (gap 0.18).

Already below P1's final val (5.46) at first eval (step 100). LoRA picked up the frozen Wake embeddings immediately. ~38s/step on T4.

### Qwen 2.5-14B P1 (In Progress)

**Step ~161/3000:** train 321.48 / val 20.98 at step 100. Both still dropping. ~131s/step on T4.

Higher initial loss values are expected given the WakeOverlay architecture -- the model is learning ~44K new embedding vectors from scratch with a 14B-parameter frozen transformer, compared to Llama's ~1.3K new tokens.

---

## Checkpoint Infrastructure

### DriveSentry

Mirrors embedding snapshots and training state to Google Drive at configurable intervals. Two key patterns:

1. **Local-first write:** `torch.save` directly to Drive FUSE can block training indefinitely on large files. Fix: save to local tmp, `shutil.copy2` to Drive, unlink local tmp.

2. **STEP_OFFSET:** When resuming with a fresh `trainer.train()` call, the Trainer's `state.global_step` restarts at 0. Callbacks add a configurable `step_offset` for globally unique file names, preventing sentry collisions across sessions.

### EmbeddingSnapshot

Saves Wake token embeddings at configurable step intervals. Lightweight (~2MB for Llama, ~340MB for Qwen) -- enables post-hoc analysis of embedding trajectory without full checkpoint overhead.

### Resume Strategies

Two resume patterns depending on model architecture:

- **Trainer-native resume** (Llama P2): `trainer.train(resume_from_checkpoint=...)` restores optimizer state, LR scheduler, and `global_step` automatically. No STEP_OFFSET needed.

- **Manual resume** (Qwen P1, Llama P1): Load embeddings from sentry, fresh `trainer.train()`. Adafactor's stateless design means no optimizer state to restore. STEP_OFFSET handles file naming. Manual override: `STEP_OFFSET = STEP_OFFSET if STEP_OFFSET > 0 else ckpt['step']` for transitioning from pre-offset sentries.

---

## Environment

**Dependencies (Colab, March 2026):**
- Python 3.12
- `torch>=2.5.1` (Colab ships 2.8.0; some scripts pin 2.5.1+cu121 for bnb compatibility)
- `transformers>=5.0`
- `accelerate>=1.2`
- `datasets>=2.21.0`
- `peft>=0.14`
- `bitsandbytes>=0.45.0`
- `triton>=3.0` (requires shim -- see below)
- `umap-learn`
- `faiss-cpu`
- `wordfreq`
- `unidecode`
- `matplotlib`

**Triton shim:** `bitsandbytes>=0.45.0` imports `triton.ops.matmul_perf_model`, which was removed in `triton>=3.x` (shipped with Colab 2026.02). Every script includes a fake-module shim:

```python
import types, sys
fake_perf = types.ModuleType('triton.ops.matmul_perf_model')
fake_perf.early_config_prune = lambda *a, **k: []
fake_perf.estimate_matmul_time = lambda *a, **k: 0
sys.modules['triton.ops'] = types.ModuleType('triton.ops')
sys.modules['triton.ops.matmul_perf_model'] = fake_perf
```

**Other Colab notes:**
- `warmup_ratio` deprecated in transformers 5.x -- use `warmup_steps` instead
- bfloat16 tensors cannot call `.numpy()` directly -- cast `.float()` first in analysis cells
- Keep `use_cache=False` during training
- Prefer Adafactor or 8-bit Adam on T4
- Enable gradient checkpointing in Phase 2 to reduce memory

## Practical Notes

- If `load_best_model_at_end=True`, match `eval_strategy` and `save_strategy` to `"steps"`
- For OOM on T4: reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps`, shorten `SEQ_LEN` (Qwen had to go from 256 to 128), or switch Phase 2 to LoRA
- Keep random seeds fixed for comparability across phases
- Keep fp16 off on T4 for this pipeline
- DriveSentry FUSE hangs are the most common cause of training stalls -- always use the local-first write pattern for saves larger than a few MB
- STEP_OFFSET only affects file naming in callbacks, not the Trainer progress bar (which always shows local step count)

---

## Monitoring

For long-running training on preemptible compute, a heartbeat monitoring notebook provides non-invasive inspection of training progress without interfering with active processes. It tracks loss trajectory from JSON logs, checkpoint inventory across local and persistent storage, embedding snapshot presence and modification times, and identifies the most recent valid checkpoint suitable for resumption.

**Storage hierarchy:**
- Local ephemeral: `/content/runs/t4_*`
- Drive persistent: `/content/drive/MyDrive/wake2vec/runs/t4_*`
- Sentry backup: `/content/drive/MyDrive/wake2vec/sentry_backups/t4_*`

## Scripts

| Script | Model | Phase | Notes |
|---|---|---|---|
| `wake2vec_llama_p1_clean.py` | Llama 3.2-1B | P1 | Gradient masking, AdamW |
| `wake2vec_llama_p2_lora.py` | Llama 3.2-1B | P2 | LoRA r=8, resume support |
| `wake2vec_llama_p3_morpheme.py` | Llama 3.2-1B | P3 | Morpheme alignment (ready) |
| `wake2vec_on_qwen_2_5_14b.py` | Qwen 2.5-14B | P1 | WakeOverlay, Adafactor |
| `wake2vec_p2_tinyllama_with_lora-2.py` | TinyLlama 1.1B | P2 | LoRA r=8 |
| `wake2vec_phase_3_morpheme_v2.py` | TinyLlama 1.1B | P3 | Morpheme alignment (ready) |

---

## Current Status (March 2026)

**Implemented and tested:**
- TinyLlama P1 + P2: Complete pipeline through LoRA behavioural tuning
- Llama 3.2-1B P1: Complete embedding-only fine-tune with full checkpoint + sentry infrastructure
- Llama 3.2-1B P2: LoRA fine-tune in progress (step 200/3000), resume mechanism deployed
- Qwen 2.5-14B P1: WakeOverlay architecture running (step ~161/3000), DriveSentry FUSE fix deployed

**Ready but not started:**
- TinyLlama P3 + Llama P3: Morpheme-compositional alignment scripts written
- Llama 3.2-3B P1: Script updated
- Llama 3.1-8B P1: Script updated

**Infrastructure:**
- Triton shim for bnb/triton 3.x compatibility
- DriveSentry local-first write pattern for FUSE reliability
- STEP_OFFSET pattern for session-safe callback file naming
- Resume support: Trainer-native (P2) and manual with STEP_OFFSET (P1)

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
> Homo Vestrae, Vale, O'Neill!
> Theheart of Lifé, the year of the Cure,
> Fought for Humans' mound in Peruvian:
> _Ere_ I go to quest of Wachtman's Cromwell,
> high time as far as Tear-nan-Og,
> as far as the Oyest Brayles;

> The
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

---

## Citation and Credit

- **Text**: James Joyce, *Finnegans Wake*
- **Base model**: [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Conceptual inspiration from work on embedding surgery, retrofitting, and lightweight adapter methods

**Cite**: https://github.com/mahb97/Wake2vec/blob/21469d75c26d40988ec5af8a4358d1796a36fdf0/data/CITATION.cff

