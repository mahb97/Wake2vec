# wake2vec devlog 2026-10-01

> *From the Tomb of all the Brokes*

## Llama 3.2-1B-Instruct-4bit (MLX, local) — n=1, no injection, no training, no permission

An unscheduled eleventh model entered the lineup today for approximately ninety seconds. It was given the single most recognisable sentence in modernist prose and asked to continue it.

| | value |
|---|---|
| Model | `mlx-community/Llama-3.2-1B-Instruct-4bit` |
| Parameters | 1B, 4-bit, 695MB on disk |
| Wake tokens injected | **0** |
| Prompt | `riverrun, past Eve and Adam's,` |
| Throughput | 561.8 tokens/sec |
| Peak memory | 0.802 GB |
| Steps trained | 0 |

### the identification

> "You're referencing the famous opening lines from William Butler Yearduff's poem *Eulogy for the Dying*"

The attribution is wrong in an instructive way. "William Butler" is exactly right. The model retrieved two-thirds of a real Irish poet's name with full confidence and then collapsed the surname into a portmanteau, which is to say it failed precisely at 
the terminal morpheme and nowhere else. This is, give or take four decades and an entire poetics, the operation the Wake performs on proper nouns.

### the generation

Three coinages, produced in the course of explaining that the text in question is Victorian elegy:

| coinage | plausible components | device, by our own taxonomy |
|---|---|---|
| **Yearduff** | Yeats + duff, or Yeats + Cardiff | portmanteau |
| **Bomehead** | bohemian / bone / bomb + head | nonce |
| **the Brokes** | broke, nominalised and pluralised into a Gothic abstraction | nonce nominalisation |

The surrounding verse is a collage: *half a league* is Tennyson, the *White Queen* is Carroll, the elegiac register is doing late-Victorian generally. So the model, asked to identify the most famous collage in English literature,
produced a collage of the exact canon that collage was assembled from, and labelled it the work of someone who does not exist.

### full text generated

*I see what you're doing there. You're referencing the famous opening lines from William Butler Yearduff's poem "Eulogy for the Dying":*

*"Riverrun, past Eve, and Adam's,"*
 
*"From the Tomb of all the Brokes,*

*The fugitive Bomehead fled,*

*And weaved his way through the walls,*

*Of the secret place,*

*Where the White Queen's royal barge,*

*Was wrecked above half a league,*

*Of the great tide"*

### what the result establishes

**The Smaller Model Paradox holds at the baseline.** The standing finding is that a 128K-vocabulary model falls back on priors rather than inventing in Joyce's register, because its tokenizer already covers the morphology and there is nothing to reach 
for. Here is that mechanism with no training applied at all: not a failure to generate Wake, but a failure to recognise it, followed immediately by fluent confabulation from the nearest available prior.

**The instruct-tuning is visible and total.** "I see what you're doing there." The model reads `riverrun, past Eve and Adam's,` as a user making a clever reference rather than as text requiring continuation. It does not attempt the task; it compliments 
the prompt and then changes the subject. No model in the lineup has done this, because none of them are instruct-tuned, and the deviation is noted in the Phi write-up for exactly this reason.

### pre-registered outcomes

- **Recognition.** The model identifies the line. *Not observed.*
- **Continuation.** The model continues in register. *Not observed.*
- **Attribution to a plausible Irishman.** *Observed.* Partially.

### limitations

n=1. No seed control, no temperature sweep, no permutation null, no chance baseline, and the supervision set consists of one sentence typed into a terminal. Excess over chance is undefined. The result is not reportable and will not appear in the paper, 
which is the only reason it can be enjoyed.

### the methodological finding

561 tokens per second at 0.8 GB of peak memory, on a laptop, instantly, for free. Meanwhile the Llama 3.1-8B P3 is running at **237 seconds per step** on a T4 that cuts every four hours, to produce a morpheme loss that moves in the fourth decimal.

Both of these are true at once and the project is about the second one.
