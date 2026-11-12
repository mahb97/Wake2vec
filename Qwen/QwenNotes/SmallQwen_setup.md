# Qwen2.5-0.5B and Qwen2.5-1.5B: setup requirements

These two complete Axis 2 of the fourteen-model design, scale at constant vocabulary, with the 14B. See `LINEUP.md` for the design and the pre-registered predictions.

## No WakeOverlay

WakeOverlay exists because the 14B's embedding matrix cannot be unfrozen on a T4. The arithmetic, at post-injection vocabulary 195,760:

| model | hidden | embedding params | bf16 | fp32 | share of model |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 896 | 175,400,960 | 0.33 GiB | 0.65 GiB | 33% |
| Qwen2.5-1.5B | 1,536 | 300,687,360 | 0.56 GiB | 1.12 GiB | 19% |
| Qwen2.5-14B | 5,120 | 1,002,291,200 | 1.87 GiB | 3.73 GiB | 7% |

P1 with the whole matrix trainable in bf16 and AdamW8bit states, against a 14.56 GiB T4:

| model | body (4-bit) | embedding + grad + optimiser state | total |
|---|---|---|---|
| Qwen2.5-0.5B | 0.18 GiB | 0.98 GiB | **1.16 GiB** |
| Qwen2.5-1.5B | 0.67 GiB | 1.68 GiB | **2.35 GiB** |
| Qwen2.5-14B | 7.13 GiB | 5.60 GiB | **12.73 GiB** |

The 14B sits 1.8 GiB under the card with no room for activations, which is the whole origin of the overlay. The other two have an order of magnitude of headroom. Thus the other Qwens will use standard gradient masking.

Also to note, WakeOverlay is not just a memory optimisation, it is a different training architecture: the base matrix is frozen and in a different dtype from the trainable rows, the Wake rows live in a separate `nn.Embedding`, and the output head is a 
two-matmul `SplitHead`. Gradient masking keeps one tensor with a backward hook. Also, 0.5B and 1.5B tie their input and output embeddings, and the 14B does not. 

**Quantisation.** Neither model needs 4-bit. The 0.5B body is 0.66 GiB in bf16 and the 1.5B is 2.43 GiB, so both run unquantised with room to spare, and dropping bitsandbytes removes the Triton shim, the `GlobalOptimManager` trap and the paged-optimiser 
failure modes. Against that, every other model in the lineup ran 4-bit NF4, and Axis 2 compares these two against the 14B, which is quantised. The embedding itself is never quantised by bnb, so the geometry P3 measures is unaffected either way; what differs 
is the LM loss. Aim is to run 4-bit NF4 as primary for comparability and to run a bf16 control before concluding anything, because a failure under 4-bit does not establish that the protocol has a lower bound. 

**Tied embeddings.** Both models ship with `tie_word_embeddings: true`. Under tying, training the Wake rows changes the output head at the same time, which is a different experiment from the untied case. Mistral's manual-tie path gives explicit control and 
is the pattern to copy. 

## Expected supervision

The morpheme and device group counts are a function of the tokenizer, so these two should resolve the same groups and pairs as the 14B, which found 120 groups and 1,234 pairs after the strip and movable-only filters, with 2,151 of 43,824 Wake rows treated (4.9%).

The 14B's pre-training baseline, for reference:

```
observed consistency 0.320482 | null floor 0.290375
EXCESS OVER CHANCE  +0.030106   z = -36.91
groups below z=-3   16/120, holding 233 pairs (18.9%)
```

Because the supervision set should be identical, any difference in baseline excess between the three Qwens is attributable to scale alone. 
