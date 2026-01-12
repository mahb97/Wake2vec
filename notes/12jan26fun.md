## Training and Validation Loss on TinyLlama P1 Wake2vec 

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 700  | 2.095300      | 6.817593        |
| 800  | 1.715300      | 6.964114        |
| 900  | 1.440700      | 7.101881        |

Confirms that P1 is heavily overfitting to *Finnegans Wake* under a frozen decoder, which is intentional. Earlier checkpoints (e.g. step 300) remain good candidates for “val-optimal” behaviour, while the 700–900 range and beyond represent increasingly Wake-specific, spicy embedding geometries for analysis and comparison.
