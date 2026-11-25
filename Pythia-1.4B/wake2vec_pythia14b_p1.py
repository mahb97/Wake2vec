# -*- coding: utf-8 -*-
"""Wake2Vec Phase 1: Pythia-1.4B Embedding-Only Fine-Tune

# Wake2Vec P1: EleutherAI/pythia-1.4b, embedding-only with gradient masking

**Model:** EleutherAI/pythia-1.4b (4-bit quantised, GPTNeoXForCausalLM)
**Hardware:** Google Colab free T4
**Training data:** Finnegans Wake corpus + Wake lexicon

Pythia-1.4B and SmolLM2-1.7B sit at 50,304 and 49,152 base vocabulary, that is
within 2.3% of each other, and at 1.4B and 1.7B parameters, within 1.2x. They
are the only near-matched pair in the fourteen-model lineup. Their near-identity 
on both the vocabulary and the scale axis makes them a controlled contrast in which 
vocabulary and scale are held approximately constant and **pretraining corpus** is the
variable. Pythia is trained on the Pile, which is heavily weighted toward
academic text, code and scraped web data. SmolLM2 is trained on FineWeb-Edu and
the SmolLM corpus, which are aggressively filtered for educational prose and
contain very little literary fiction.

## Architecture differences 

| | Llama / Mistral / Gemma / Phi | Pythia (GPT-NeoX) |
|---|---|---|
| class | `*ForCausalLM`, Llama-style | `GPTNeoXForCausalLM` |
| input embedding | `model.model.embed_tokens` | `model.gpt_neox.embed_in` |
| output head | `model.lm_head` | `model.embed_out` |
| tying | tied (or manually tied) | **untied** |
| attention proj | `q_proj`, `k_proj`, `v_proj` | **fused** `query_key_value` |
| attention out | `o_proj` | `dense` |
| MLP | `gate_proj`, `up_proj`, `down_proj` | `dense_h_to_4h`, `dense_4h_to_h` |
| matrix rows vs tokenizer | equal | **50,304 rows for 50,254 tokens** |

## envi
"""

import os
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Triton shim: bnb 0.45.x imports triton.ops.matmul_perf_model, removed in
# triton 3.x. Harmless if bnb >= 0.46.1 already handles it.
import types, sys
_fake = types.ModuleType("triton.ops")
_fake.matmul_perf_model = types.ModuleType("triton.ops.matmul_perf_model")
sys.modules.setdefault("triton.ops", _fake)
sys.modules.setdefault("triton.ops.matmul_perf_model", _fake.matmul_perf_model)

import torch, gc

print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")
try:
    import bitsandbytes as _bnb
    print(f"bitsandbytes: {_bnb.__version__}")
except ImportError:
    print("bitsandbytes NOT INSTALLED. run: pip install -U bitsandbytes")
import transformers, accelerate, peft
print(f"transformers: {transformers.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"peft: {peft.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

_BF16_OK = torch.cuda.is_bf16_supported()
print(f"bf16 supported natively: {_BF16_OK}")
if not _BF16_OK:
    print("  >> NOTE: bf16 is emulated on this GPU. Expect slower steps.")

torch.cuda.empty_cache()
gc.collect()

from google.colab import drive
drive.mount('/content/drive')

"""## config"""

import math
from pathlib import Path

# model 
MODEL_NAME = "EleutherAI/pythia-1.4b"

EXPECTED_MATRIX_ROWS = 50304   
EXPECTED_TOKENIZER_LEN = 50254  # len(tok) before injection
EXPECTED_HIDDEN = 2048

# data 
FW_TEXT = "/content/FW_TEXT.txt"
WAKE_LEXICON = "/content/wake_lexicon.txt"
MORPHEME_JSONL = "/content/wake_embedding_groups.jsonl"

# outputs
RUN_DIR = Path("/content/drive/MyDrive/wake2vec_pythia14b_p1")
LOCAL_RUN = Path("/content/runs/wake2vec_pythia14b_p1")
SENTRY = RUN_DIR / "sentry_backups"
EMB_SNAPS = RUN_DIR / "emb_snaps"
FULL_CHECKPOINTS = RUN_DIR / "full_checkpoints"

for d in [RUN_DIR, LOCAL_RUN, SENTRY, EMB_SNAPS, FULL_CHECKPOINTS]:
    d.mkdir(parents=True, exist_ok=True)

# training hyperparams 
MAX_STEPS = 3000
LR = 2e-4
WARMUP_STEPS = max(20, MAX_STEPS // 20)
WEIGHT_DECAY = 0.0
BATCH_SIZE = 1
GRAD_ACCUM = 16
SEQ_LEN = 512
SAVE_STEPS = 50
LOG_STEPS = 50
EVAL_STEPS = 100
EMB_SNAP_STEPS = 100       
USE_BF16 = True            
GRAD_CKPT = True          

# architecture handling
TIE_OUTPUT_HEAD = True      
LORA_TARGET = "query_key_value"

# retain 
KEEP_EVERY = 500
KEEP_RECENT = 4

# resume 
RESUME_FROM = None
# RESUME_FROM = SENTRY / "checkpoint-200"

print("WAKE2VEC p1 on PYTHIA-1.4B CONFIG")
print(f"  Model:            {MODEL_NAME}")
print(f"  Output:           {RUN_DIR}")
print(f"  Steps:            {MAX_STEPS} | LR {LR} | warmup {WARMUP_STEPS}")
print(f"  Batch:            {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Seq len:          {SEQ_LEN}")
print(f"  bf16:             {USE_BF16} | grad ckpt: {GRAD_CKPT}")
print(f"  Tie output head:  {TIE_OUTPUT_HEAD}")
print(f"  Gradient masking: ENABLED (base rows frozen)")
print(f"  Snapshots:        Wake rows only, fp32, every {EMB_SNAP_STEPS} steps")
print(f"  Resume:           {RESUME_FROM}")
