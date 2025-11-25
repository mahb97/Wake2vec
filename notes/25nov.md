# Wake2Vec P2 Update 

T4 died at step 1200/2000. Training was progressing well with loss dropping from 5.44 to 4.54 on the training set and 5.34 to 4.81 on validation. Validation loss plateaued at step 1200, staying flat at 4.806 for the last checkpoint.

Resuming from checkpoint-1200 to complete the remaining 800 steps. Will monitor whether early stopping triggers or if validation loss improves with continued training.

**training progress:**
- Step 200: train 5.441, val 5.339
- Step 400: train 4.896, val 4.943
- Step 600: train 4.758, val 4.855
- Step 800: train 4.612, val 4.817
- Step 1000: train 4.557, val 4.806
- Step 1200: train 4.542, val 4.806 (T4 death)

**to-do:** November 25, 2025, 9:54 PM resuming training from checkpoint-1200.
