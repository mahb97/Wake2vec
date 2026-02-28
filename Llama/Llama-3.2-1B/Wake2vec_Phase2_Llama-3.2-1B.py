# -*- coding: utf-8 -*-
"""Wake2Vec Phase 2: LoRA Fine-Tune (Llama-3.2-1B)

# Wake2Vec Phase 2: LoRA Fine-Tune with Frozen P1 Embeddings

**Model:** meta-llama/Llama-3.2-1B (4-bit quantized)
**Hardware:** Google Colab T4 GPU (2026.02)
**P1 Source:** wake2vec_llama_p1/final/ (step 3000, gradient-masked embeddings)

## Overview

Phase 2 loads the embedding weights from the Llama P1 run (3000 steps,
gradient-masked — only Wake rows trained) and freezes them entirely. LoRA
adapters are applied to attention (q, k, v) and MLP (gate, up, down)
projections. The model learns to use the Wake-adapted embeddings through
attention and MLP adaptation rather than further embedding modification.

## Key Differences from P1

| Aspect | P1 (Embedding-Only) | P2 (LoRA) |
|--------|---------------------|-----------|
| Trainable | Wake embedding rows (gradient masked) | LoRA adapters only |
| Frozen | Transformer layers + base embed rows | Embeddings + base weights |
| Quantization | 4-bit NF4 | 4-bit NF4 |
| Optimizer | AdamW (embed weight only) | AdamW (LoRA params, via Trainer) |
| LR | 2e-4 | 2e-5 |
| Steps | 3000 | 3000 |

────────────────────────────────────


