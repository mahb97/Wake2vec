# wake2vec devlog 2026-04-14

## Llama 3.1-8B P1 session 3 (resuming from step 150)

omg guys this one comes with a reading recommendation: [The Eye of the Master: A Social History of Artificial Intelligence](https://www.matteopasquinelli.com/the-eye-of-the-master/) by Matteo Pasquinelli

eval_steps now 50, SEQ_LEN 256 with 99s/step and compositional init + 1.0x radius. 

Resuming from `checkpoint-150`.

### P1 loss table

| Step | Train | Val | Session |
|------|-------|-----|---------|


---

## Llama 3.2-3B P1 session 10 (resuming from step 1150)

val has been slowly rising since step 300: 6.68 → 6.83 and train dropping (109 → 49). 
Resuming from `checkpoint-1150`.

### P1 loss table (continued)

| Step | Train | Val | Session |
|------|-------|-----|---------|
| 100 | 109.19 | 7.01 | 1 |
| 200 | 97.27 | 6.75 | 2 |
| 300 | 87.04 | 6.68 | 3 |
| 400 | 79.07 | 6.70 | 3 |
| 500 | 72.80 | 6.72 | 3 |
| 600 | 67.01 | 6.75 | 4 |
| 700 | 62.09 | 6.77 | 5 |
| 800 | 58.50 | 6.79 | 6 |
| 900 | 54.30 | 6.81 | 8 |
| 1000 | 51.15 | 6.83 | 9 |
| 1100 | 49.36 | 6.83 | 9 |
| 1200 | 46.56 | 6.84 | 10 |

---

Barry Can't Swim - [All My Friends](https://soundcloud.com/barrycantswim/all-my-friends-8?si=af47d2112d2f4eab9c8b17efe4309e6d&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing) 
