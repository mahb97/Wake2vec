## 2025-11-21 (this was nothing but )

**aim:** Train Llama-3.2-1B with Wake vocabulary (attempt #4)

**setup:**
- 4-bit quantization (NF4, bfloat16 compute)
- CPU offloading enabled
- Repulsion loss disabled (learned from OOM attempts)
- Proper Drive paths (no more void trains)
- Sentry backup system active

**training session 1: 0→669**

Runtime: 3+ hours on single T4 session

| Step | Training Loss | Notes |
|------|---------------|-------|
| 20   | 3.157        | Strong start |
| 100  | 2.163        | 4x better than TinyLlama at same step |
| 200  | 0.889        | Checkpoint backed up |
| 400  | 0.529        | Checkpoint backed up |
| 600  | 0.484        | Checkpoint backed up (bet and won) |
| 660  | 0.476        | cut at 4:03 am |

**loss reduction:** 84.9% (3.16 → 0.476) in 660 steps

**findings:**
- Llama-3.2-1B learns Wake vocabulary 4x faster than TinyLlama
- 4-bit quantization doesn't impair embedding-only training
- Stronger multilingual base model > precision for lexicon injection
- Longer sequences (512 vs 256) provide better context

*Status:* Will resume from checkpoint-600.

**Comparative performance (Llama vs TinyLlama at step 400):**
- TinyLlama: 1.82
- Llama: 0.53
- **3.4x better convergence**
