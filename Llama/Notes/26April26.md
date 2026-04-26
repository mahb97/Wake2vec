# wake2vec devlog 2026-04-26

## Llama 3.1-8B P1 session 8 (resuming from step 650)

A steady 0.4 val drop every 200 steps, for comparison the 3B at the same stage was already in its memorisation plateau, so the init strategy is paying off as hypothesised.

Resuming from `checkpoint-650`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 200 | 195.78 | 12.57 | 3 |
| 400 | 168.21 | 11.72 | 5 |
| 600 | 151.81 | 11.48 | 7 |
| 650+ | | | *resuming today, session 8* |

---

## Llama 3.2-3B P1 session 15 (resuming from step 1800)

val climbing slowly, the 3B is the control variable in the comparison now, same arch as the 8B but with old-style spherical 1.5x init. 

Resuming from `checkpoint-1800`.

### P1 loss table (continued)

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 1500 | 44.91 | 6.93 | 12 |
| 1600 | 44.08 | 6.93 | 13 |
| 1700 | 42.54 | 6.96 | 14 |
| 1800 | 41.98 | 6.97 | 14 |
| 1800+ | | | *resuming today, session 15* |

---

## Notes

the 8B is now the hero of the project's methodological story. compositional init + 1.0x radius is producing measurably faster val descent than the spherical-1.5x baseline established by TinyLlama, Llama 1B, and Llama 3B.

1,200 steps left for the 3B before P1 ends. 

---

[Novia Robot](https://soundcloud.com/rosaliaofficial/novia-robot?si=98a727b72cca460690fa060fab1a06cd&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
