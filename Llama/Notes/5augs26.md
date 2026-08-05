# wake2vec devlog 2026-08-05

> *approaching 1/3 of P2*

## Llama 3.1-8B P2 session 23 

the 8B glided at 900. last eval, step 900, val 5.1486, a new best-val (val fell -0.006, so no turn), 0.181 below the 3B's wall. the decrement has essentially floored (0.042, 0.009, 0.0064, the last two spanning just 0.006) 
and the train-val gap is still widening (0.557 to 0.645, train dropping 0.094 while val barely moved). 

resuming from `checkpoint-920`. SEQ_LEN 512, EVAL_STEPS=100. next eval at 1000.

### P2 loss table

| Step | Train | Val | vs P1 best (11.36) | Note |
|------|-------|-----|--------------------|------|
| 700 | 4.833 | 5.164 | -6.20 | best-val, 0.166 below the wall |
| 800 | 4.598 | 5.155 | -6.21 | best-val, 0.175 below, approach shape |
| 900 | 4.504 | 5.149 | -6.21 | **best-val, 0.181 below**, glide (val -0.006, train -0.094), decrement 0.0064, gap 0.645 |
| 1000 | | | | *deferred turn, or another micro-glide* |
