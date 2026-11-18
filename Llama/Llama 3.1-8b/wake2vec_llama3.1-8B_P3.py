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

"""## get P2 state"""

from transformers import AutoTokenizer

# verify P3 artifacts
assert ADAPTER_SOURCE.exists(), f"Adapter source not found: {ADAPTER_SOURCE}"
assert EMB_SOURCE.exists(), f"Embeddings not found: {EMB_SOURCE}"

# load tok from the adapter checkpoint
tok = AutoTokenizer.from_pretrained(str(ADAPTER_SOURCE), use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

TOTAL_VOCAB = len(tok)
NUM_WAKE = TOTAL_VOCAB - BASE_VOCAB

print(f"  Vocab size: {TOTAL_VOCAB}")
print(f"  Base vocab: {BASE_VOCAB}")
print(f"  Wake tokens: {NUM_WAKE}")

# get frozen P1 embeddings carried through P2 
embed_weights = torch.load(EMB_SOURCE, map_location="cpu")
print(f"  Shape: {embed_weights.shape}")
assert embed_weights.shape[0] == TOTAL_VOCAB, \
    f"Embedding/vocab mismatch: {embed_weights.shape[0]} vs {TOTAL_VOCAB}"

"""## morpheme group builder"""

with open(MORPHEME_JSONL, 'r') as f:
    raw_groups = [json.loads(line) for line in f]

print(f"  Raw groups: {len(raw_groups)}")
print(f"  Total examples: {sum(g['n_examples'] for g in raw_groups)}")


def tokenize_word(word, tokenizer):
    return tokenizer(word, add_special_tokens=False)["input_ids"]


def embed_word(word, tokenizer, embed_matrix):
    ids = tokenize_word(word, tokenizer)
    if len(ids) == 0:
        return None, None
    vecs = embed_matrix[ids]
    return vecs.mean(dim=0), ids


# make morpheme groups with resolved token IDs
morpheme_groups = []
total_pairs = 0
skipped_groups = 0
frozen_pairs_dropped = 0

for g in raw_groups:
    morpheme = g["morpheme"]
    morpheme_type = g["morpheme_type"]
    examples = g["examples"]
    bases = g["bases"]

    word_embeds = []
    base_embeds = []
    word_ids_list = []
    base_ids_list = []
    pair_words = []

    for word, base in zip(examples, bases):
        w_emb, w_ids = embed_word(word, tok, embed_weights)
        b_emb, b_ids = embed_word(base, tok, embed_weights)
        if w_emb is not None and b_emb is not None:
            if MORPH_MOVABLE_ONLY and not any(
                i >= BASE_VOCAB for i in list(w_ids) + list(b_ids)
            ):
                frozen_pairs_dropped += 1
                continue
            word_embeds.append(w_emb)
            base_embeds.append(b_emb)
            word_ids_list.append(w_ids)
            base_ids_list.append(b_ids)
            pair_words.append((word, base))

    if len(word_embeds) >= MIN_GROUP_SIZE:
        morpheme_groups.append({
            "morpheme": morpheme,
            "morpheme_type": morpheme_type,
            "word_embeds": torch.stack(word_embeds),
            "base_embeds": torch.stack(base_embeds),
            "word_ids": word_ids_list,
            "base_ids": base_ids_list,
            "pair_words": pair_words,
            "n_valid": len(word_embeds),
        })
        total_pairs += len(word_embeds)
    else:
        skipped_groups += 1

print(f"\n  Valid morpheme groups: {len(morpheme_groups)} (skipped {skipped_groups} with < {MIN_GROUP_SIZE} pairs)")
print(f"  Total word-base pairs: {total_pairs}")

MIN_SUPERVISION_WARN = 500
if MORPH_MOVABLE_ONLY:
    print(f"  MOVABLE-ONLY: dropped {frozen_pairs_dropped:,} pairs frozen on both sides")
    print(f"    (every remaining pair has >=1 trainable Wake row, so the whole")
    print(f"     objective can act on the whole supervision set)")
else:
    print(f"  MOVABLE-ONLY: OFF ({frozen_pairs_dropped:,} frozen pairs retained, no gradient)")

# drop the strip groups 
if MORPH_EXCLUDE_STRIP:
    kept, dropped, dropped_pairs = [], [], 0
    for g in morpheme_groups:
        if g["morpheme"] in STRIP_MORPHEMES:
            dropped.append(g["morpheme"])
            dropped_pairs += g["n_valid"]
        else:
            kept.append(g)
    morpheme_groups = kept
    total_pairs -= dropped_pairs
    print(f"\n  STRIP EXCLUSION: dropped {len(dropped)} groups / {dropped_pairs:,} pairs")
    print(f"    dropped: {', '.join(sorted(dropped))}")
    print(f"    remaining: {len(morpheme_groups)} groups / {total_pairs:,} pairs")
    print(f"    (permutation null on P2 embeddings: strip groups z=+1.89, at chance;")
    print(f"     retained groups z=-102.63, consistency 0.295 against a 0.151 floor)")
else:
    print("\n  STRIP EXCLUSION: OFF (all groups retained, ~50.7% of pairs at chance)")

sorted_groups = sorted(morpheme_groups, key=lambda x: -x["n_valid"])
if total_pairs < MIN_SUPERVISION_WARN:
    print("\n" + "!" * 66)
    print(f"  THIN SUPERVISION: {total_pairs:,} pairs across {len(morpheme_groups)} groups")
    print(f"  Mistral's comparable filtered set was 1,413 pairs / 130 groups.")
    print("  The 128K base vocabulary means most morpheme words are already base")
    print("  tokens, so MORPH_MOVABLE_ONLY removes them. Below a few hundred pairs")
    print("  the morpheme objective has too little to act on, the permutation null")
    print("  loses power (chance floor rises as 1/sqrt(n)), and a flat L_morph")
    print("  would be uninformative. Consider lowering MIN_GROUP_SIZE,")
    print("  or accept that this run tests the device objective only.")
else:
    print(f"\n  Supervision OK: {total_pairs:,} pairs across {len(morpheme_groups)} groups")

print(f"\n  Top 10 groups:")
for g in sorted_groups[:10]:
    print(f"    {g['morpheme']:>10s} ({g['morpheme_type']:>7s}) — {g['n_valid']} pairs")
    if g['pair_words']:
        sample = g['pair_words'][:3]
        print(f"      e.g. {', '.join(f'{w}→{b}' for w, b in sample)}")

token_to_groups = {}
for gi, g in enumerate(morpheme_groups):
    for w_ids in g["word_ids"]:
        for tid in w_ids:
            if tid >= BASE_VOCAB:
                if tid not in token_to_groups:
                    token_to_groups[tid] = []
                token_to_groups[tid].append(gi)

wake_tokens_with_morphemes = len(token_to_groups)
print(f"\n  Wake tokens participating in morpheme groups: {wake_tokens_with_morphemes}")

"""## device group builder"""

with open(DEVICE_JSONL, 'r') as f:
    raw_device_groups = [json.loads(line) for line in f]

device_groups = []
device_total_words = 0

for dg in raw_device_groups:
    device = dg["device"]
    examples = dg["examples"]

    word_ids_list = []
    valid_words = []
    for word in examples:
        ids = tokenize_word(word, tok)
        if ids:
            word_ids_list.append(ids)
            valid_words.append(word)

    if len(valid_words) >= MIN_GROUP_SIZE:
        device_groups.append({
            "device": device,
            "word_ids": word_ids_list,
            "words": valid_words,
            "n_valid": len(valid_words),
        })
        device_total_words += len(valid_words)

print(f"  Device groups: {len(device_groups)}")
print(f"  Total words with resolved tokens: {device_total_words}")
for dg in device_groups:
    print(f"    {dg['device']:>15s} — {dg['n_valid']} words")

"""## dataset"""

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class BlockDataset(Dataset):
    def __init__(self, blocks, seq_len=256):
        self.blocks = blocks
        self.seq_len = seq_len

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        ids = torch.tensor(self.blocks[idx], dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids)
        }

if not os.path.exists(FW_TEXT):
    raise FileNotFoundError(f"FW text not found: {FW_TEXT}")

with open(FW_TEXT, 'r', encoding='utf-8') as f:
    text = f.read()

ids = tok(text, add_special_tokens=False)["input_ids"]
print(f"  Total tokens: {len(ids)}")

blocks = []
for i in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN):
    chunk = ids[i:i + SEQ_LEN]
    if len(chunk) == SEQ_LEN:
        blocks.append(chunk)

print(f"  Total blocks: {len(blocks)}")

train_blocks, val_blocks = train_test_split(
    blocks, test_size=0.10, random_state=42
)

train_ds = BlockDataset(train_blocks, SEQ_LEN)
val_ds = BlockDataset(val_blocks, SEQ_LEN)

print(f"  Train: {len(train_ds)} blocks")
print(f"  Val: {len(val_ds)} blocks")

"""## model setup"""

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import PeftModel

set_seed(42)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,           # dtype not torch_dtype (transformers 5.x)
    device_map="auto",
    max_memory={0: "12GB", "cpu": "30GB"}
)

model.config.use_cache = False
model.config.attn_implementation = "eager"
model.config.tie_word_embeddings = False   # set True after the resize

print(f"Resizing embeddings: {BASE_VOCAB} -> {TOTAL_VOCAB}...")
model.resize_token_embeddings(TOTAL_VOCAB, mean_resizing=False)

# get Llama 8B P2 frozen embeds 
wte = model.get_input_embeddings()

# DTYPE GUARD 
print(f"[DTYPE] embedding after resize: {wte.weight.dtype}")
if wte.weight.dtype != torch.bfloat16:
    wte.to(torch.bfloat16)
    print(f"[DTYPE] cast to {wte.weight.dtype}")

with torch.no_grad():
    wte.weight.copy_(embed_weights.to(wte.weight.device, dtype=wte.weight.dtype))

# tie
if hasattr(model, "lm_head"):
    model.lm_head.weight = wte.weight
model.config.tie_word_embeddings = True
if hasattr(model, "tie_weights"):
    model.tie_weights()
print("P2 embeddings loaded and tied (matching 8B P2)")

# verify the tie did not widen the matrix
_w = model.get_input_embeddings().weight
_o = model.get_output_embeddings()
print(f"[DTYPE] embedding: {_w.dtype}, {tuple(_w.shape)}, "
      f"{_w.numel() * _w.element_size() / 2**30:.2f} GiB")
if _o is not None:
    print(f"[DTYPE] lm_head shares storage with embedding: "
          f"{_o.weight.data_ptr() == _w.data_ptr()}")
if _w.dtype != torch.bfloat16:
    raise RuntimeError(
        f"Embedding is {_w.dtype} after tying, not bfloat16. Under adafactor "
        f"this costs a {_w.numel() * 4 / 2**30:.2f} GiB full-shape fp32 update "
        "temporary and the run WILL OOM at the first optimizer step. Do not "
        "train from here."
    )
del _w, _o

# P2 LoRA adapters 
model = PeftModel.from_pretrained(model, str(ADAPTER_SOURCE), is_trainable=True)
model.print_trainable_parameters()

# unfreeze Wake embed rows via grad masking
for p in model.parameters():
    p.requires_grad = False

wte = model.get_input_embeddings()
wte.weight.requires_grad = True

for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True

wake_rows = torch.arange(BASE_VOCAB, TOTAL_VOCAB, device=wte.weight.device)
base_rows = torch.arange(0, BASE_VOCAB, device=wte.weight.device)

def mask_base_grad(grad):
    """Zero gradients for base vocab rows, only Wake rows get updated."""
    if grad is None:
        return grad
    grad[base_rows] = 0
    return grad

wte.weight.register_hook(mask_base_grad)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
lora_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower())
emb_params = wte.weight.numel()

print(f"  Total trainable params: {trainable:,}")
print(f"  LoRA params: {lora_params:,}")
print(f"  Embedding params (all rows, masked): {emb_params:,}")
print(f"  Wake rows that actually train: {NUM_WAKE:,} x {wte.weight.shape[1]} = {NUM_WAKE * wte.weight.shape[1]:,}")
print(f"  Gradient masking: base rows [{0}:{BASE_VOCAB}] zeroed on backward pass")

"""## pre-training embed snapshot"""

if PRE_SNAPSHOT.exists() and not FORCE_NEW_PRE_SNAPSHOT:
    raise RuntimeError(
        f"{PRE_SNAPSHOT} already exists."
    )

_E_snap = model.get_input_embeddings().weight.detach().cpu().clone()
torch.save(_E_snap, PRE_SNAPSHOT)
print(f"  Saved: {PRE_SNAPSHOT}")
print(f"  Shape: {tuple(_E_snap.shape)}  dtype: {_E_snap.dtype}")
del _E_snap
