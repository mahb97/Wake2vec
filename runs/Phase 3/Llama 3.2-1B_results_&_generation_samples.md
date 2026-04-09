# wake2vec Llama 3.2-1B P3 Results

## Final Numbers

| Metric | Value |
|--------|-------|
| Model | meta-llama/Llama-3.2-1B (4-bit NF4) |
| Phase | P3 (morpheme-compositional alignment, strong lambdas) |
| P2 source | step 500 (best val 4.04) |
| Steps | 1,000 (4 Colab sessions) |
| Training time | 173.1 minutes |
| Final train | 14.539 |
| Final val | 4.636 |
| Best val | 4.482 (step 100) |
| Trainable | 358,291,456 total (LoRA 5.1M + Wake embed rows 90.5M effective) |

## Loss Trajectory

| Step | L_total | L_lm | L_morph | L_device | Val | Early stop |
|------|---------|------|---------|----------|-----|------------|
| 0 | 3.8597 | 3.4387 | 0.0007 | 0.1933 | — | — |
| 50 | 3.6508 | 3.2506 | 0.0007 | 0.1830 | — | — |
| 100 | 3.8224 | 3.3968 | 0.0007 | 0.1956 | **4.4819** | best ✓ |
| 200 | — | — | 0.0007 | — | **4.5016** | 1/5 |
| 300 | 3.6496 | 3.2213 | 0.0007 | 0.1970 | **4.5284** | 2/5 |
| 400 | 3.8786 | 3.4460 | 0.0007 | 0.1991 | **4.5482** | 3/5 |
| 500 | 3.7454 | 3.3314 | 0.0007 | 0.1899 | **4.5871** | 4/5 |
| 600 | 4.1771 | 3.7554 | 0.0007 | 0.1937 | **4.6075** | best ✓ (reset*) |
| 700 | 3.4596 | 3.0147 | 0.0007 | 0.2053 | **4.6179** | 1/5 (reset) |
| 800 | 3.6884 | 3.2429 | 0.0007 | 0.2056 | **4.6326** | 2/5 (reset) |
| 900 | 3.8024 | 3.3542 | 0.0007 | 0.2070 | **4.6312** | 3/5 (reset) |
| 950 | 3.8246 | 3.3902 | 0.0007 | 0.2001 | — | — |
| 1000 | — | — | — | — | **4.6359** | 4/5 (reset) |

\* Early stop counter reset on resume at step 600, same bug as TinyLlama P3.
