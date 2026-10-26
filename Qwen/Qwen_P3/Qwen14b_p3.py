# -*- coding: utf-8 -*-
"""Wake2Vec Phase 3: Morpheme-Compositional Alignment (Qwen 2.5-14B), Strong Lambdas

# Wake2Vec Phase 3: Morpheme-Compositional Alignment, Strong Lambdas

**Model:** Qwen/Qwen2.5-14B (4-bit quantized body, fp16 frozen embeddings, fp32 Wake overlay)
**Hardware:** Google Colab T4, 14.56GB, with CPU offload
**Source:** Qwen P2 best-val (step 1600, val 5.9209), SPLIT across two paths:
  - LoRA adapter + tokenizer from wake2vec_qwen14b_p2_lora/full_checkpoints/step_1600
  - Wake rows from wake2vec_qwen14b_p1/sentry_backups/sentry_step_2700.pt

Qwen P2 froze the embeddings and never wrote an embeddings_frozen.pt, so the
starting matrix is rebuilt the way P2 built it: base rows from the pretrained
checkpoint, Wake rows injected from the canonical P1 sentry. Because P2 never
updated them, this reproduces P2's matrix exactly rather than approximating it.

Mistral and the Llamas unfreeze the whole embedding and zero the base rows with
a backward hook. Qwen cannot. Its matrix is 195,888 x 5120 = 1.003 BILLION
parameters, 42% larger than the 8B's 706M, which already took three optimiser
attempts and a smoke test to fit. Unfrozen it costs 2.0GB of gradient on top of
a 2.0GB fp16 matrix and a ~7GB 4-bit body, before any optimiser state exists.

So P3 uses WakeOverlay, the architecture P1 built for this model: the full
matrix stays frozen in fp16 and a separate fp32 nn.Embedding holds the 43,824
Wake rows. 224M trainable parameters instead of 1.003B, which fits fp32 AdamW
and therefore matches the optimiser used by TinyLlama, the 1B, the 3B and
Mistral, rather than inheriting the 8B's quantised-optimiser caveat.

## The one deliberate departure from P1

P1 tied lm_head to the frozen full matrix. Wake tokens were READ from the
trained overlay and EMITTED through the spherical initialisation: their output
rows never moved for the whole of P1, which is a plausible part of why P1
produced dense polyglot output rather than Wake. P2 merged the overlay back
into the full matrix before training, so P2's head was correct.

P3 trains the overlay again, so the head has to follow it. SplitHead computes
logits in two parts and concatenates them. Without it the auxiliary losses
would reshape the input side while the output side stayed at P2, and the
generation test afterwards would measure a model whose halves disagree.

## What this P3 tests

Qwen enters with the lineup's largest vocabulary and its lowest Wake share
(43,824 of 195,888, 22.4%), and its P2 generation battery returned sustained
Wake register with no memorisation signal: the longest span shared with the
training text across five temperatures was five high-frequency function words.

Two models have now moved under the corrected cosine morpheme loss. Mistral
rose +0.0457 to +0.0612 in excess over chance across 100 steps from a region at
0.995 isotropy; the Llama 8B rose +0.0362 to +0.0417 from the attractor itself
at 0.998453. The question here is whether a 14B with an overlay architecture
and a trained output head does the same, and whether L_device stays flat for a
fifth time.

## Before launching

    !pip install -q --no-cache-dir "bitsandbytes>=0.46.1" \\
        transformers accelerate peft scikit-learn

riverrun. the Wake never ends, it loops back to the beginning.
"""

import os, sys, types
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

torch.cuda.empty_cache()
gc.collect()

from google.colab import drive
drive.mount('/content/drive')

from huggingface_hub import login
login()
