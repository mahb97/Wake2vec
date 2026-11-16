# -*- coding: utf-8 -*-
"""Wake2Vec Phase 3: Morpheme-Compositional Alignment (Llama 3.1-8B), Strong Lambdas

# Wake2Vec Phase 3: Morpheme-Compositional Alignment, Strong Lambdas

**Model:** meta-llama/Llama-3.1-8B (4-bit quantized)
**Hardware:** Google Colab T4 GPU + CPU offload
**Source:** Llama 3.1-8B P2 best-val (step 900, val 5.1486), SPLIT across two paths:
  - LoRA adapter + tokenizer from wake2vec_llama8b_p2_lora/full_checkpoints/step_0900
  - Embeddings from wake2vec_llama8b_p2_lora/embeddings_frozen.pt (P2 saved them once)

## What this run is for

Unlike Mistral, the 8B does NOT hold the null-breaker slot. Its P2 embedding
analysis puts the Wake region at 0.998453 isotropy -- AT the attractor, and in
fact more isotropic than the base vocabulary (0.9896) -- with base-new pairwise
cosine 0.0009 and no nearest neighbour above 0.083. On the standing structural
diagnosis there is nothing here for the auxiliary losses to amplify, and the
prior on movement is low.

Three things make it worth running anyway:

- It is the FIRST P3 on a 128K-vocab model under a loss that can actually push.
  The four prior runs tested the device objective properly and the morpheme
  objective not at all, so the morpheme question has never been put to the
  Llama family.
- It carries the set's only COMPOSITIONAL 1.0x initialisation, so it varies the
  init against every other P3 in the project. Its P1 analysis already separated
  the two effects: norm structure is init-dependent (Cohen's d -1.09 at the
  step-1200 matrix, against ~-7 under spherical), isotropy is not.
- The generation raises a question the geometry cannot answer. The P2 battery
  came back ~70% verified Wake-lexicon retrieval, mostly corpus hapaxes, stable
  across independent samples -- from a region every geometric instrument calls
  featureless. Whatever makes those rows usable is invisible to isotropy,
  pairwise cosine and nearest-neighbour structure. A null here would say the
  instruments cannot see it, not that it is absent, and that is worth stating
  with a properly scaled objective behind it.

## Configuration notes specific to the 8B

- SEQ_LEN 512 and batch 1 x 16, matching 8B P1/P2 so validation stays comparable
  to P2's 5.1486.
- ADAFACTOR, not AdamW. The embedding matrix is 172,451 x 4096 = 706M params;
  unfrozen under AdamW that is ~5.65GB of optimizer state on top of a 4-bit 8B
  body, which does not fit a T4. Adafactor stores no momentum and is the
  project's precedent for exactly this (Qwen P1). Set OPTIM below to
  "paged_adamw_8bit" if Adam dynamics matter more than the comparability cost.
- Tie via tie_weights(), reproducing what 8B P2 did. The state must be tied the
  way it was created.

## Loss

  L_total = L_lm + 1.0 * L_morpheme + 2.0 * L_device
            + 0.05 * L_repulsion + 0.01 * L_norm

─────────────────────────────────────────────────────────

## envi
"""

import gc
import os, sys, types
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# triton shim 
fake_perf = types.ModuleType('triton.ops.matmul_perf_model')
fake_perf.early_config_prune = lambda *a, **k: []
fake_perf.estimate_matmul_time = lambda *a, **k: 0
sys.modules['triton.ops'] = types.ModuleType('triton.ops')
sys.modules['triton.ops.matmul_perf_model'] = fake_perf

# !pip install --no-cache-dir \
#     torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0+cu126 \
#     --index-url https://download.pytorch.org/whl/cu126

# !pip install -q --no-cache-dir \
#     "bitsandbytes>=0.46.1" \
#     transformers>=4.45.2 \
#     accelerate==1.12.0 \
#     peft==0.18.1 \
#     scikit-learn

import torch, gc
print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")
try:
    import bitsandbytes as bnb_lib
    print(f"bitsandbytes: {bnb_lib.__version__}")
except ImportError:
    print("bitsandbytes: NOT INSTALLED")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 60)

torch.cuda.empty_cache()
gc.collect()

from google.colab import drive
drive.mount('/content/drive')

from huggingface_hub import login
login()

"""## Cell 2: Config"""

from pathlib import Path
import json

# P3 source: Llama 8B P2 best-val (step 900, val 5.1486) 
P2_RUN = Path("/content/drive/MyDrive/wake2vec_llama8b_p2_lora")
ADAPTER_SOURCE = P2_RUN / "full_checkpoints" / "step_0900"   # LoRA adapter + tokenizer
EMB_SOURCE = P2_RUN / "embeddings_frozen.pt"                 # frozen P1 embeddings (saved once)

# P3 output paths 
RUN_DIR = Path("/content/drive/MyDrive/wake2vec_llama8b_p3_morpheme_strong")
LOCAL_RUN = Path("/content/runs/wake2vec_llama8b_p3_morpheme_strong")
SENTRY = RUN_DIR / "sentry_backups"
EMB_SNAPS = RUN_DIR / "emb_snaps"
FULL_CHECKPOINTS = RUN_DIR / "full_checkpoints"

# pre-training ref. 
PRE_SNAPSHOT = RUN_DIR / "embeddings_pre_p3.pt"
FORCE_NEW_PRE_SNAPSHOT = False   

for d in [RUN_DIR, LOCAL_RUN, SENTRY, EMB_SNAPS, FULL_CHECKPOINTS]:
    d.mkdir(parents=True, exist_ok=True)

# model 
MODEL_NAME = "meta-llama/Llama-3.1-8B"
FW_TEXT = "/content/FW_TEXT.txt"
MORPHEME_JSONL = "/content/wake_embedding_groups.jsonl"
DEVICE_JSONL = "/content/device_groups.jsonl"

# training hyperparams 
MAX_STEPS = 1000
LR = 2e-5                  # halved from P2 (5e-5)
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
BATCH_SIZE = 1            # 8B + unfrozen embeddings; matches Llama 8B P2's config
GRAD_ACCUM = 16           # effective batch 16
SEQ_LEN = 512            # matches Llama 8B P1/P2 (val block set identical)
SAVE_STEPS = 100            # more frequent saves for short run
EMB_SNAP_STEPS = 100        # Wake-row snapshots (was 25 and full-matrix: ~56GB on the 8B)
LOG_STEPS = 50              # finer logging granularity
EVAL_STEPS = 100            

# LoRA (loaded from P2, stays trainable)
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

# base vocab 
BASE_VOCAB = 128256

# morpheme loss formulation
MORPH_LOSS_MODE = "cosine"

# Strip-group exclusion
MORPH_EXCLUDE_STRIP = True
STRIP_MORPHEMES = {"-s", "-ed", "-y", "-ing", "-er", "-ly", "-on", "-'s", "-a", "co-"}

# movable-pairs-only 
MORPH_MOVABLE_ONLY = True

# optimizer 
OPTIM = "adamw_bnb_8bit"

# loss weights 
LAMBDA_MORPH = 1.0 if MORPH_LOSS_MODE == "cosine" else 50.0
LAMBDA_DEVICE = 2.0         # offensive, shapes device clusters
LAMBDA_REPULSION = 0.05     # safety net (unchanged)
LAMBDA_NORM = 0.01          # gentle regularisation (unchanged)
REPULSION_THRESHOLD = 0.95
REPULSION_PAIRS = 100

# morpheme loss config
MIN_GROUP_SIZE = 3

# device loss config 
DEVICE_TRIPLETS = 64
DEVICE_MARGIN = 0.2

# early stopping 
EARLY_STOP_PATIENCE = 6     

# resume 
RESUME_FROM = None

print("WAKE2VEC PHASE 3: LLAMA 3.1-8B MORPHEME-COMPOSITIONAL ALIGNMENT CONFIG (STRONG)")
print(f"  Adapter source: {ADAPTER_SOURCE}")
print(f"  Embedding source: {EMB_SOURCE}")
print(f"  Output: {RUN_DIR}")
print(f"  Steps: {MAX_STEPS}")
print(f"  LR: {LR}")
print(f"  Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  LoRA rank: {LORA_RANK}, targets: {LORA_TARGETS}")
print(f"  Morph loss mode: {MORPH_LOSS_MODE}"
      + ("  (scale-free cosine; lambda recalibrated from the legacy 50.0)"
         if MORPH_LOSS_MODE == "cosine" else "  (LEGACY squared-deviation, scale-dependent)"))
print(f"  Strip exclusion: {MORPH_EXCLUDE_STRIP}"
      + ("  (10 chance-level groups dropped, ~50.7% of pairs)" if MORPH_EXCLUDE_STRIP else "  (all groups retained)"))
print(f"  Movable-only: {MORPH_MOVABLE_ONLY}"
      + ("  (frozen base-vocab-only pairs dropped; ~49% of the strip-filtered set)"
         if MORPH_MOVABLE_ONLY else "  (frozen pairs retained, they contribute no gradient)"))
print(f"  Loss weights: morph={LAMBDA_MORPH}, device={LAMBDA_DEVICE}, repulsion={LAMBDA_REPULSION}, norm={LAMBDA_NORM}")
print(f"  Morpheme data: {MORPHEME_JSONL}")
print(f"  Device data: {DEVICE_JSONL}")
print(f"  Min group size: {MIN_GROUP_SIZE}")
print(f"  Early stopping: patience {EARLY_STOP_PATIENCE}")
