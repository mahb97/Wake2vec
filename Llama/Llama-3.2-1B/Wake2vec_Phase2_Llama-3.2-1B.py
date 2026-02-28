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

## envi
"""

import os, sys, types
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Triton shim (bnb may import triton.ops removed in triton 3.x)
fake_perf = types.ModuleType('triton.ops.matmul_perf_model')
fake_perf.early_config_prune = lambda *a, **k: []
fake_perf.estimate_matmul_time = lambda *a, **k: 0
sys.modules['triton.ops'] = types.ModuleType('triton.ops')
sys.modules['triton.ops.matmul_perf_model'] = fake_perf

# Colab 2026.02 ships torch 2.10.0, transformers 5.0.0 
# Only install what Colab doesn't ship:
# !pip install -q bitsandbytes peft accelerate scikit-learn scipy

import torch, gc
print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)
print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")
try:
    import bitsandbytes as bnb_lib
    print(f"bitsandbytes: {bnb_lib.__version__}")
except ImportError:
    print("bitsandbytes: NOT INSTALLED -- run: pip install bitsandbytes")
import transformers, accelerate, peft
print(f"transformers: {transformers.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"peft: {peft.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f} GB"
      if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
      else f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 60)

torch.cuda.empty_cache()
gc.collect()

from google.colab import drive
drive.mount('/content/drive')

from huggingface_hub import login
login()

