# wake2vec devlog 2026-04-28

## Llama 3.1-8B P1 session 9 (resuming from step 750)

the methodological hero keeps proving itself: 12.57, to 11.72, to 11.48 across 600 steps. the compositional init + 1.0x radius combo is doing exactly as hoped, descending faster than the 3B did at the same stage. 

Resuming from `checkpoint-750`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 200 | 195.78 | 12.57 | 3 |
| 400 | 168.21 | 11.72 | 5 |
| 600 | 151.81 | 11.48 | 7 |
| 800 | 140.89 | 11.37 | 9 |

---

## Llama 3.2-3B P1 session 16 (resuming from step 1950)

val nearly back at starting point: 7.01 (step 100) and now 6.99 (step 1900). 1,800 steps of training to land within 0.02 of the starting point. the embeddings are different: they've reorganised, drifted, settled, but the LM-loss signal as measured by val isn't capturing it. classic P1 saturation: the embeddings learn what they can with frozen attention, then start memorising. 1,050 steps left until P2 LoRA!

Resuming from `checkpoint-1950`.

### P1 loss table (continued)

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 1500 | 44.91 | 6.93 | 12 |
| 1600 | 44.08 | 6.93 | 13 |
| 1700 | 42.54 | 6.96 | 14 |
| 1800 | 41.98 | 6.97 | 14 |
| 1900 | 40.03 | 6.99 | 15 |
| 2000 | 38.58 | 7.00 | 16 |
| 2100 | 36.90 | 7.01 | 16 |

---

## Status update

| Model | Phase | Status | Notes |
|-------|-------|--------|-------|
| TinyLlama 1.1B | Done | Complete | P1→P2→P3→P3b |
| Llama 3.2-1B | Done | Complete | P1→P2→P3 |
| **Llama 3.2-3B** | **P1** | **Running** | **Step 1950/3000, val 6.99 (saturated)** |
| **Llama 3.1-8B** | **P1** | **Running** | **Step 750, val 11.48 (compositional init working)** |
| Mistral 7B | P1 | Paused | Step 1000/3000, val 11.13 (wish denied) |
| Qwen 2.5-14B | P1 | Paused | Step 1760/3000, val 15.81 (post-16.0 breakthrough) |
| Phi-3 Mini | P1 | Planned | Script pending |
| Gemma 2 9B | P1 | Planned | Script pending |
| Gemma 3n E2B | P1 | Planned | Script pending |
| Gemma 3n E4B | P1 | Planned | Script pending |

---

## Notes

the 8B's trajectory is the project's clearest evidence that init strategy matters. every previous model used spherical 1.5x and plateaued early. the 8B is still descending at step 600 and shows no sign of saturation. if the model is still descending at step 1500 (where Llama 1B plateaued), the paper has a clean ablation: same architecture family, same vocab, same training protocol, only difference is init < and the new init wins.

the 3B is the project's reference point for what *not* to do. 3B has locked the same trajectory as Llama 1B: fast initial descent, plateau by step 300, slow climb thereafter, so 1B will be a good comparison anchor in the discussion.

---

[Finesse](https://soundcloud.com/brysontiller/bryson-tiller-finesse-cover?si=37899b84bf2a4a44b314a81e877607ed&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
