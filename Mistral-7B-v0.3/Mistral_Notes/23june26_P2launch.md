# wake2vec devlog 2026-06-23 (Mistral P2 launch)

> *everyone in london be overheating thank god i live in Dubs*

## Mistral 7B v0.3 P2 session 1 (launch)

Mistral P2 is live. P1 closed with the deepest embedding reorganization and wake attitude in the lineup (Wake drift 0.485, real Wake tokens surfacing, maximal register-reach including emoji).

### Launch config

| Param | Value |
|-------|-------|
| Model | mistralai/Mistral-7B-v0.3 (4-bit NF4, sliding-window attention) |
| P1 source | **step 1200 (best val 10.9181, the global minimum)** |
| Embeddings | FROZEN at P1 best-val |
| LoRA | r=8, α=16, dropout=0.1, targets q/k/v/gate/up/down |
| LR | 2e-5 |
| **SEQ_LEN** | **512 (matches the Llama P2s for cross-model comparability)** |
| Batch | 1 x 16 = 16 effective |
| Max steps | 3000 (full P2 schedule) |
| Optimizer | AdamW |
| Tie | manual (no tie_weights() call, Phi-bug-safe) |

### The best-val correction

P2 sources from step 1200, not the step-3000 endpoint. Mistral's P1 val had two local minima: it broke 11.0 at step 1200 (val 10.9181, the global minimum), drifted up through the survey-phase plateau, then descended again to 11.0936 at step 3000. (Both Mistral and the 8B had their P1 global minimum at step 1200, a small cross-model regularity worth noting: embedding-only P1 may reach its best-generalising configuration around step 1200 regardless of architecture.)

### The suspension test for Mistral 

Mistral is the 58% Wake-vocab-share datapoint at 7B scale, matching TinyLlama's share at a larger body. TinyLlama holds the fine line: novel forms inside recoverable syntax, meaning suspended rather than abolished, the only model that produces what the actual Wake does. The P1 generation showed Mistral as the maximal-babel pole (the widest register-collision, the most genuine Wake tokens, emoji at every temperature) but fragmented, unrouted, the dissolution pole.

So, for P2, does the LoRA make Mistral's richly-learned micro-units rise coherently (TinyLlama-style suspension) or stay dissolved (8B-style over-deformation)? *μp → UP:* given the matched 58% share with TinyLlama and the deepest P1 learning in the lineup, the prior on Mistral holding suspension is the most favourable of any model. if any model challenges TinyLlama for the suspension crown, the P1 analysis says it is this one.

### Speed and schedule

SEQ_LEN 512 on a 7B with offload is slow (likely ~199.7s/step). P2 is the full 3000-step schedule. unlike the walled smaller-model P2s (which early-stopped at 600/700 because they walled), Mistral may run the full schedule. 

---

## Notes

the P1 generation was golden (to borrow a quote directly from Mistral: "delicious impossible schongabpolis", emoji and all).

---

Keir Starmer resigned because it was too hot (someone said that on substack): [Nothing Great About Britain](https://soundcloud.com/slowthai/nothing-great-about-britain?in=slowthai/sets/nothing-great-about-britain-1&si=a652b567d8d04ec7ac7e5121a788cb15&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
