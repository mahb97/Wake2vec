## Training and Validation Loss on TinyLlama P1 Wake2vec 

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 700  | 2.095300      | 6.817593        |
| 800  | 1.715300      | 6.964114        |
| 900  | 1.440700      | 7.101881        |
| 1000 | 1.122600      | 7.262826        |
| 1100 | 0.847500      | 7.403337        |
| 1200 | 0.641900      | 7.523938        |
| 1300 | 0.549100      | 7.628265        |
| 1400 | 0.427700      | 7.749165        |
| 1500 | 0.315400      | 7.866288        |
| 1600 | 0.232100      | 7.970313        |
| 1700 | 0.175400      | 8.046519        |

Confirms that P1 is heavily overfitting to *Finnegans Wake* under a frozen decoder, which is intentional. Earlier checkpoints (e.g. step 300) remain good candidates for “val-optimal” behaviour, while the 700–900 range and beyond represent increasingly Wake-specific, spicy embedding geometries for analysis and comparison.
