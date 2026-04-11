# wake2vec devlog 2026-04-11

## Mistral 7B v0.3 P1 session 4 (resuming from step 300)

four sessions in and I've already forgotten to copy a checkpoint (rip step 250, gone but not forgotten). sliding window attention, 32K vocab getting the full 44K Wake injection. s

val broke through 11.0 last session (10.99 at step 300). let's see if it keeps dropping or if this is where the plateau lives.

Resuming from `checkpoint-300`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 50 | 186.78 | 11.37 | 1 |
| 100 | 181.33 | 11.13 | 1 |
| 200 | 172.47 | 11.12 | 2/3 |
| 250 | 164.99 | 11.07 | 2/3 |
| 300 | 163.94 | 10.99 | 3 |
| 350 | 159.78 | 10.99 | 4 |

### The Mistral question

Mistral is the key vocab comparison, as it has a 32K base vocab, which is same as TinyLlama. that means:
- 44,553 Wake tokens injected (vs TinyLlama's ~44,500)
- the model has to build an entirely new embedding subspace
- if the smaller model paradox holds, Mistral's Wake output should be more authentically Joycean than any of the 128K-vocab Llamas
- but is 7B vs TinyLlama's 1.1B so the Q is, do stronger priors help or hinder?

the interesting scenario: Mistral produces TinyLlama-quality Wake output with Llama-8B-level coherence. the worst scenario: model is too smart for its own good and falls back on standard English like the 128K Llamas did. this is edn: fuck around and find out in... *checks step count* ...2,700 more steps...at 82s/step...across many sessions.

---

## Notes

four Google accounts, eight models, and more Colab identities than most people have email addresses. somewhere, Google's abuse detection team is looking at four accounts all running transformers fine-tuning on the same obscure 1939 novel and thinking "this is either a very dedicated student or a very confused bot." (in the surveillance state we are all cam girls)

[lovely](https://soundcloud.com/brentfaiyaz/lovely-prod-by-sonder?in=may-stevens-846243297/sets/a-l-o-n-e&si=5c2c7161cab04df0932efb200fe2f205&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
