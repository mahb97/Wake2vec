### P1 Llama-3.2-1B (LR = 2e-4, max_steps = 6000)

### Original Run (up to step 740)

> Note: step 740 was observed in logs but not checkpointed; 700 is the last fully saved checkpoint.

| Step | Training Loss |
|------|---------------|
| 320  | 4.2207        |
| 340  | 4.1461        |
| 360  | 4.0512        |
| 380  | 3.9337        |
| 400  | 4.0217        |
| 420  | 3.8934        |
| 440  | 3.8845        |
| 460  | 3.7942        |
| 480  | 3.6833        |
| 500  | 3.6884        |
| 520  | 3.6641        |
| 540  | 3.5804        |
| 560  | 3.5366        |
| 580  | 3.5912        |
| 600  | 3.4845        |
| 620  | 3.4825        |
| 640  | 3.4203        |
| 660  | 3.3637        |
| 700  | 3.2634        |
| 720  | 3.3523        |
| 740  | 3.1590        |


### P1 Llama-3.2-1B new run starting from step_0700 embeds

> This run reloads `full_checkpoints/step_0700/embeddings.pt`, resets the optimiser and scheduler,  
> and continues training with `max_steps = 6000` and `learning_rate = 2e-4`.  
> initial weights correspond to the original step-700 state.

| Step (this run) | Training Loss |
|-----------------|---------------|
| 50              | 3.2426        |
| 100             | 3.2416        |
