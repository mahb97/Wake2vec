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

"""## loss components (vectorized)"""

import torch.nn.functional as F

# pre-training reference
if not PRE_SNAPSHOT.exists():
    raise RuntimeError(
        f"{PRE_SNAPSHOT} not found.\n"
        "get pre-training reference first."
    )

E_pre = torch.load(PRE_SNAPSHOT, map_location="cpu")
print(f"Pre-training reference: {PRE_SNAPSHOT}")
print(f"  Shape: {tuple(E_pre.shape)}  dtype: {E_pre.dtype}")

base_norms_pre = torch.norm(E_pre[:BASE_VOCAB], dim=1)
TARGET_NORM = base_norms_pre.mean().item()
NORM_MARGIN = base_norms_pre.std().item()
print(f"  Base vocab norm: mean={TARGET_NORM:.4f}, std={NORM_MARGIN:.4f}")


class MorphemeIndex:

    def __init__(self, groups, tokenizer, device="cuda"):
        self.device = device
        self.group_names = []
        self.group_types = []

        all_word_ids = []
        all_base_ids = []
        all_word_lens = []
        all_base_lens = []
        pair_groups = []
        group_sizes = []

        gidx = 0
        for g in groups:
            word_ids = g["word_ids"]
            base_ids = g["base_ids"]

            if len(word_ids) < MIN_GROUP_SIZE:
                continue

            n = len(word_ids)
            for w, b in zip(word_ids, base_ids):
                all_word_ids.append(w)
                all_base_ids.append(b)
                all_word_lens.append(len(w))
                all_base_lens.append(len(b))
                pair_groups.append(gidx)

            group_sizes.append(n)
            self.group_names.append(g["morpheme"])
            self.group_types.append(g["morpheme_type"])
            gidx += 1

        self.n_groups = gidx
        self.total_pairs = len(all_word_ids)

        if self.total_pairs == 0:
            print(f"MorphemeIndex: 0 groups (no valid data)")
            return

        max_w = max(all_word_lens)
        max_b = max(all_base_lens)

        word_padded = torch.zeros(self.total_pairs, max_w, dtype=torch.long)
        base_padded = torch.zeros(self.total_pairs, max_b, dtype=torch.long)
        for i, (w, b) in enumerate(zip(all_word_ids, all_base_ids)):
            word_padded[i, :len(w)] = torch.tensor(w, dtype=torch.long)
            base_padded[i, :len(b)] = torch.tensor(b, dtype=torch.long)

        self.word_ids = word_padded.to(device)
        self.base_ids = base_padded.to(device)
        self.word_lens = torch.tensor(all_word_lens, dtype=torch.float, device=device).unsqueeze(1)
        self.base_lens = torch.tensor(all_base_lens, dtype=torch.float, device=device).unsqueeze(1)
        self.pair_groups = torch.tensor(pair_groups, dtype=torch.long, device=device)
        self.group_sizes = torch.tensor(group_sizes, dtype=torch.float, device=device)

        self.word_mask = (torch.arange(max_w, device=device).unsqueeze(0)
                          < self.word_lens)
        self.base_mask = (torch.arange(max_b, device=device).unsqueeze(0)
                          < self.base_lens)

        print(f"MorphemeIndex: {self.n_groups} groups, {self.total_pairs} pairs (vectorized)")

    def compute_loss(self, embed_weight):
        if self.n_groups == 0 or self.total_pairs == 0:
            return torch.tensor(0.0, device=self.device)

        w_embeds = embed_weight[self.word_ids]
        b_embeds = embed_weight[self.base_ids]

        w_embeds = (w_embeds * self.word_mask.unsqueeze(-1)).sum(dim=1) / self.word_lens
        b_embeds = (b_embeds * self.base_mask.unsqueeze(-1)).sum(dim=1) / self.base_lens

        directions = w_embeds - b_embeds

        dim = directions.shape[1]
        expand_groups = self.pair_groups.unsqueeze(1).expand(-1, dim)

        group_sums = torch.zeros(self.n_groups, dim,
                                 device=self.device, dtype=directions.dtype)
        group_sums.scatter_add_(0, expand_groups, directions)
        group_means = group_sums / self.group_sizes.unsqueeze(1)

        if MORPH_LOSS_MODE == "sqdev":
            pair_means = group_means[self.pair_groups]
            deviations = ((directions - pair_means) ** 2).mean(dim=1)
            return deviations.mean()

        # cosine (default)
        d_unit = F.normalize(directions, dim=1, eps=1e-8)
        g_unit = F.normalize(group_means, dim=1, eps=1e-8)
        cos = (d_unit * g_unit[self.pair_groups]).sum(dim=1)
        return (1.0 - cos).mean()


morph_index = MorphemeIndex(morpheme_groups, tok, device="cuda")

ALL_WAKE_IDS = torch.arange(BASE_VOCAB, TOTAL_VOCAB, dtype=torch.long).cuda()


def compute_morpheme_loss(model):
    E = model.get_input_embeddings().weight
    return morph_index.compute_loss(E)


def compute_repulsion_loss(model, n_pairs=REPULSION_PAIRS, threshold=REPULSION_THRESHOLD):
    E = model.get_input_embeddings().weight
    wake_embeds = E[ALL_WAKE_IDS]

    n_wake = wake_embeds.shape[0]
    if n_wake < 2:
        return torch.tensor(0.0, device="cuda")

    idx = torch.randint(0, n_wake, (n_pairs, 2), device="cuda")
    a = wake_embeds[idx[:, 0]]
    b = wake_embeds[idx[:, 1]]

    a_norm = F.normalize(a, dim=1)
    b_norm = F.normalize(b, dim=1)
    cos_sim = (a_norm * b_norm).sum(dim=1)

    violations = torch.clamp(cos_sim - threshold, min=0)
    loss = (violations ** 2).mean()
    return loss


def compute_norm_loss(model, target=TARGET_NORM, margin=NORM_MARGIN):
    E = model.get_input_embeddings().weight
    wake_embeds = E[ALL_WAKE_IDS]

    norms = torch.norm(wake_embeds, dim=1)
    deviation = torch.abs(norms - target) - margin
    loss = (torch.clamp(deviation, min=0) ** 2).mean()
    return loss


class DeviceIndex:

    def __init__(self, groups, tokenizer, device="cuda"):
        self.device = device
        self.n_groups = len(groups)
        self.group_names = [g["device"] for g in groups]

        all_word_ids = []
        all_word_lens = []
        all_group_labels = []

        for gi, g in enumerate(groups):
            for w_ids in g["word_ids"]:
                all_word_ids.append(w_ids)
                all_word_lens.append(len(w_ids))
                all_group_labels.append(gi)

        self.total_words = len(all_word_ids)

        max_len = max(all_word_lens)
        word_padded = torch.zeros(self.total_words, max_len, dtype=torch.long)
        for i, w in enumerate(all_word_ids):
            word_padded[i, :len(w)] = torch.tensor(w, dtype=torch.long)

        self.word_ids = word_padded.to(device)
        self.word_lens = torch.tensor(all_word_lens, dtype=torch.float,
                                       device=device).unsqueeze(1)
        self.word_mask = (torch.arange(max_len, device=device).unsqueeze(0)
                          < self.word_lens)
        self.group_labels = torch.tensor(all_group_labels, dtype=torch.long,
                                          device=device)

        self.group_indices = []
        self.not_group_indices = []
        for gi in range(self.n_groups):
            self.group_indices.append(
                (self.group_labels == gi).nonzero(as_tuple=True)[0])
            self.not_group_indices.append(
                (self.group_labels != gi).nonzero(as_tuple=True)[0])

        print(f"DeviceIndex: {self.n_groups} groups, {self.total_words} words (vectorized)")

    def compute_loss(self, embed_weight, n_triplets=DEVICE_TRIPLETS, margin=DEVICE_MARGIN):
        if self.n_groups < 2 or self.total_words < 3:
            return torch.tensor(0.0, device=self.device)

        w_embeds = embed_weight[self.word_ids]
        w_embeds = (w_embeds * self.word_mask.unsqueeze(-1)).sum(dim=1) / self.word_lens

        anchor_idx = torch.randint(0, self.total_words, (n_triplets,),
                                    device=self.device)
        anchor_groups = self.group_labels[anchor_idx]

        pos_idx = torch.zeros(n_triplets, dtype=torch.long, device=self.device)
        neg_idx = torch.zeros(n_triplets, dtype=torch.long, device=self.device)
        valid_mask = torch.ones(n_triplets, dtype=torch.bool, device=self.device)

        for gi in range(self.n_groups):
            mask = (anchor_groups == gi)
            if not mask.any():
                continue

            g_idx = self.group_indices[gi]
            ng_idx = self.not_group_indices[gi]

            if len(g_idx) < 2:
                valid_mask[mask] = False
                continue

            n_in = mask.sum().item()
            pos_idx[mask] = g_idx[torch.randint(0, len(g_idx), (n_in,),
                                                 device=self.device)]
            neg_idx[mask] = ng_idx[torch.randint(0, len(ng_idx), (n_in,),
                                                  device=self.device)]

        if not valid_mask.any():
            return torch.tensor(0.0, device=self.device)

        anchors = F.normalize(w_embeds[anchor_idx[valid_mask]], dim=1)
        positives = F.normalize(w_embeds[pos_idx[valid_mask]], dim=1)
        negatives = F.normalize(w_embeds[neg_idx[valid_mask]], dim=1)

        cos_pos = (anchors * positives).sum(dim=1)
        cos_neg = (anchors * negatives).sum(dim=1)

        loss = torch.clamp(margin + cos_neg - cos_pos, min=0).mean()
        return loss


device_index = DeviceIndex(device_groups, tok, device="cuda")


def compute_device_loss(model):
    E = model.get_input_embeddings().weight
    return device_index.compute_loss(E)

print(f"\nL_morpheme: compositional direction consistency ({morph_index.n_groups} groups, {morph_index.total_pairs} pairs)")
print(f"  L_device: triplet margin loss ({device_index.n_groups} device types, {device_index.total_words} words, margin={DEVICE_MARGIN})")
print(f"  L_repulsion: cosine > {REPULSION_THRESHOLD} penalty ({REPULSION_PAIRS} pairs/step)")
print(f"  L_norm: deviation from base norm mean={TARGET_NORM:.2f} +/- {NORM_MARGIN:.2f}")

"""## geometry instruments (permutation nulls, treated/control)"""

N_PERM_BASELINE = 500
PERM_SEED = 42


def _morph_directions(E):
    dirs, labels = [], []
    for gi, g in enumerate(morpheme_groups):
        w = torch.stack([E[torch.tensor(i, dtype=torch.long)].mean(0) for i in g["word_ids"]])
        b = torch.stack([E[torch.tensor(i, dtype=torch.long)].mean(0) for i in g["base_ids"]])
        dirs.append(w - b)
        labels.append(torch.full((len(g["word_ids"]),), gi, dtype=torch.long))
    return torch.cat(dirs), torch.cat(labels)


def _pairwise_cos(D, D_unit, lbl, n_groups):
    sums = torch.zeros(n_groups, D.shape[1], dtype=D.dtype)
    sums.index_add_(0, lbl, D)
    cnt = torch.bincount(lbl, minlength=n_groups).clamp(min=1).unsqueeze(1).to(D.dtype)
    means = F.normalize(sums / cnt, dim=1, eps=1e-8)
    return (D_unit * means[lbl]).sum(dim=1)


def morph_null_report(E, n_perm=N_PERM_BASELINE, seed=PERM_SEED, label=""):
    E = E.detach().float().cpu()
    if len(morpheme_groups) == 0:
        print(f"\n[NULL] {label}: no morpheme groups survive filtering.")
        return {"label": label, "n_pairs": 0, "n_groups": 0, "n_perm": 0,
                "observed_L_morph": float("nan"), "observed_consistency": float("nan"),
                "null_mean": float("nan"), "null_sd": float("nan"),
                "excess_over_chance": float("nan"), "z": float("nan"),
                "n_groups_below_z3": 0, "pairs_below_z3": 0, "per_group": []}
    D, L_ = _morph_directions(E)
    n_pairs, dim = D.shape
    n_groups = len(morpheme_groups)
    D_unit = F.normalize(D, dim=1, eps=1e-8)
    cnt = torch.bincount(L_, minlength=n_groups).clamp(min=1).float()

    obs_cos = _pairwise_cos(D, D_unit, L_, n_groups)
    obs_global = float((1.0 - obs_cos).mean())
    obs_group = torch.zeros(n_groups).index_add_(0, L_, 1.0 - obs_cos) / cnt

    g = torch.Generator().manual_seed(seed)
    null_global = torch.empty(n_perm)
    null_group = torch.empty(n_perm, n_groups)
    for k in range(n_perm):
        perm = L_[torch.randperm(n_pairs, generator=g)]
        c = _pairwise_cos(D, D_unit, perm, n_groups)
        null_global[k] = (1.0 - c).mean()
        null_group[k] = torch.zeros(n_groups).index_add_(0, perm, 1.0 - c) / cnt

    mu, sd = float(null_global.mean()), float(null_global.std())
    z = (obs_global - mu) / sd if sd > 0 else float("nan")
    gmu, gsd = null_group.mean(0), null_group.std(0).clamp(min=1e-9)
    gz = (obs_group - gmu) / gsd

    per_group = [{
        "morpheme": morpheme_groups[i]["morpheme"],
        "n_pairs": int(cnt[i]),
        "observed_L": float(obs_group[i]),
        "observed_consistency": float(1 - obs_group[i]),
        "null_L": float(gmu[i]),
        "excess": float(gmu[i] - obs_group[i]),
        "z": float(gz[i]),
    } for i in range(n_groups)]
    per_group.sort(key=lambda r: r["z"])

    sig = [r for r in per_group if r["z"] < -3.0]
    print(f"MORPHEME PERMUTATION NULL{(' -- ' + label) if label else ''}")
    print(f"\npairs {n_pairs:,} | groups {n_groups} | permutations {n_perm}")
    print(f"  observed L_morph : {obs_global:.6f}  (consistency {1 - obs_global:.6f})")
    print(f"  null mean        : {mu:.6f}  (consistency {1 - mu:.6f})   sd {sd:.6f}")
    print(f"  EXCESS OVER CHANCE: {mu - obs_global:+.6f}      z = {z:+.2f}")
    print(f"  groups below z=-3: {len(sig)}/{n_groups}, holding "
          f"{sum(r['n_pairs'] for r in sig):,} pairs "
          f"({100 * sum(r['n_pairs'] for r in sig) / n_pairs:.1f}%)")
    print(f"  strongest: " + ", ".join(f"{r['morpheme']}({r['z']:+.1f})" for r in per_group[:6]))
    print("  NOTE: excess-over-chance compares across sets; z carries sample size ")

    return {"label": label, "n_pairs": int(n_pairs), "n_groups": n_groups,
            "n_perm": n_perm, "observed_L_morph": obs_global,
            "observed_consistency": 1 - obs_global, "null_mean": mu, "null_sd": sd,
            "excess_over_chance": mu - obs_global, "z": z,
            "n_groups_below_z3": len(sig),
            "pairs_below_z3": int(sum(r["n_pairs"] for r in sig)),
            "per_group": per_group}


def treated_control_geometry(E_pre, E_post, seed=PERM_SEED):
    def _iso(X):
        if len(X) < 10:
            return float("nan"), float("nan")
        S = F.normalize(X - X.mean(0), dim=1, eps=1e-12)
        n = min(5000, len(S))
        idx = torch.randperm(len(S), generator=torch.Generator().manual_seed(seed))[:n]
        S = S[idx]
        C = S @ S.T
        C.fill_diagonal_(0)
        Z = torch.exp(C).sum(1)
        return float(Z.min() / Z.max()), float(C.sum() / (n * (n - 1)))

    treated = set()
    for g in morpheme_groups:
        for ids in g["word_ids"] + g["base_ids"]:
            treated.update(i for i in ids if i >= BASE_VOCAB)
    for g in device_groups:
        for ids in g["word_ids"]:
            treated.update(i for i in ids if i >= BASE_VOCAB)
    treated = sorted(treated)
    untouched = sorted(set(range(BASE_VOCAB, TOTAL_VOCAB)) - set(treated))
    gcpu = torch.Generator().manual_seed(seed)
    ctrl = [untouched[i] for i in
            torch.randperm(len(untouched), generator=gcpu)[:len(treated)].tolist()]

    out = {"n_treated": len(treated), "n_control": len(ctrl),
           "wake_rows": TOTAL_VOCAB - BASE_VOCAB,
           "treated_share": len(treated) / max(TOTAL_VOCAB - BASE_VOCAB, 1)}
    rows = []
    for name, idx in (("treated (LM+aux)", treated), ("control (LM only)", ctrl)):
        ti = torch.tensor(idx, dtype=torch.long)
        A = E_pre[ti].detach().float().cpu()
        B = E_post[ti].detach().float().cpu()
        ipre, cpre = _iso(A)
        ipost, cpost = _iso(B)
        drift = float(F.cosine_similarity(A, B, dim=1).mean())
        l2d = float((B - A).norm(dim=1).mean())
        rows.append({"name": name, "n": len(idx), "iso_pre": ipre, "iso_post": ipost,
                     "iso_delta": ipost - ipre, "drift_cos": drift, "drift_l2": l2d})
        del A, B
    out["subsets"] = rows
    t, c = rows[0], rows[1]
    out["did_isotropy"] = t["iso_delta"] - c["iso_delta"]
    out["did_drift_cos"] = t["drift_cos"] - c["drift_cos"]


    print(f"\ntreated rows: {len(treated):,} of {out['wake_rows']:,} Wake rows "
          f"({100 * out['treated_share']:.1f}%) -- the aggregate dilutes this "
          f"{out['wake_rows'] / max(len(treated), 1):.0f}x")
    print(f"  {'subset':>20s} {'n':>7s} {'iso_pre':>9s} {'iso_post':>9s} "
          f"{'iso_d':>9s} {'drift_cos':>10s} {'drift_L2':>9s}")
    for r in rows:
        print(f"  {r['name']:>20s} {r['n']:>7,d} {r['iso_pre']:>9.6f} "
              f"{r['iso_post']:>9.6f} {r['iso_delta']:>+9.6f} "
              f"{r['drift_cos']:>10.6f} {r['drift_l2']:>9.5f}")
    print(f"\n  >> DiD isotropy  : {out['did_isotropy']:+.6f}")
    print(f"  >> DiD drift_cos : {out['did_drift_cos']:+.6f} "
          "(lower treated drift_cos = treated moved more)")
    return out

_treated_preview = set()
for _g in morpheme_groups:
    for _ids in _g["word_ids"] + _g["base_ids"]:
        _treated_preview.update(i for i in _ids if i >= BASE_VOCAB)
_morph_touched = len(_treated_preview)
for _g in device_groups:
    for _ids in _g["word_ids"]:
        _treated_preview.update(i for i in _ids if i >= BASE_VOCAB)
_n_wake = TOTAL_VOCAB - BASE_VOCAB

print("Geom instruments defined:")
print(f"  morph_null_report()        permutation null, global + per-group, "
      f"{N_PERM_BASELINE} perms, seed {PERM_SEED}")
print(f"  treated_control_geometry() isotropy + drift, treated vs LM-only control")
print(f"  supervision measured: {len(morpheme_groups)} morpheme groups, "
      f"{sum(len(g['word_ids']) for g in morpheme_groups):,} pairs")
print(f"  Wake rows touched: {_morph_touched:,} by morpheme, "
      f"{len(_treated_preview):,} by morpheme+device, of {_n_wake:,} "
      f"({100 * len(_treated_preview) / max(_n_wake, 1):.1f}%)")
print(f"  => aggregate Wake-region metrics dilute any effect ~"
      f"{_n_wake / max(len(_treated_preview), 1):.0f}x; read treated-vs-control, "
      f"not the aggregate")
del _treated_preview  

"""## pre-training baseline"""

PRE_NULL = morph_null_report(E_pre, label="PRE (P2 embeddings, no P3 gradient)")

"""## custom trainer"""

from transformers import TrainingArguments, Trainer

class MorphemeTrainer(Trainer):

    def __init__(self, *args, loss_log=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_log = loss_log if loss_log is not None else []

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None, **kwargs):
        outputs = model(**inputs)
        lm_loss = outputs.loss

        morph_loss = compute_morpheme_loss(model)
        device_loss = compute_device_loss(model)
        repulsion_loss = compute_repulsion_loss(model)
        norm_loss = compute_norm_loss(model)

        total_loss = (
            lm_loss
            + LAMBDA_MORPH * morph_loss
            + LAMBDA_DEVICE * device_loss
            + LAMBDA_REPULSION * repulsion_loss
            + LAMBDA_NORM * norm_loss
        )

        # only log during training 
        if model.training and self.state.global_step % LOG_STEPS == 0:
            self.loss_log.append({
                "step": self.state.global_step,
                "lm": lm_loss.item(),
                "morph": morph_loss.item(),
                "device": device_loss.item(),
                "repulsion": repulsion_loss.item(),
                "norm": norm_loss.item(),
                "total": total_loss.item(),
            })
            print(f"  [LOSS] step={self.state.global_step} "
                  f"lm={lm_loss.item():.4f} "
                  f"morph={morph_loss.item():.4f}(*{LAMBDA_MORPH}) "
                  f"dev={device_loss.item():.4f}(*{LAMBDA_DEVICE}) "
                  f"repul={repulsion_loss.item():.6f}(*{LAMBDA_REPULSION}) "
                  f"norm={norm_loss.item():.4f}(*{LAMBDA_NORM}) "
                  f"total={total_loss.item():.4f}")

        # gradient-accumulation normalisation 
        if model.training and num_items_in_batch is not None:
            total_loss = total_loss / self.args.gradient_accumulation_steps

        if return_outputs:
            return total_loss, outputs
        return total_loss

loss_log = []

"""## callbacks"""

import shutil
import time

from transformers import TrainerCallback


def has_weights(ck):
    return (ck / "adapter_model.safetensors").exists() or (ck / "pytorch_model.bin").exists()


class EmbeddingSnapshot(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % EMB_SNAP_STEPS == 0:
            try:
                mdl = kwargs.get("model", model)
                E = mdl.get_input_embeddings().weight[BASE_VOCAB:].detach().cpu().clone()
                tmp = Path(f"/content/_emb_step{state.global_step:04d}.pt")
                torch.save({"wake_rows": E, "wake_start": BASE_VOCAB,
                            "step": state.global_step}, tmp)
                shutil.copy2(tmp, EMB_SNAPS / f"emb_step{state.global_step:04d}.pt")
                tmp.unlink(missing_ok=True)
                print(f"[EMB] Step {state.global_step}: wake rows saved {tuple(E.shape)}")
            except Exception as e:
                print(f"[EMB] Step {state.global_step}: {e}")


class SentryMirror(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        try:
            cks = sorted(
                LOCAL_RUN.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]),
                reverse=True
            )
            if not cks:
                return
            ck = cks[0]
            if not has_weights(ck):
                print(f"[SENTRY] {ck.name}: local has no weights yet, skipping")
                return

            dst = SENTRY / ck.name
            if dst.exists():
                if has_weights(dst):
                    return 
                else:
                    print(f"[SENTRY] {ck.name}: incomplete mirror detected, retrying")
                    shutil.rmtree(dst)

            shutil.copytree(ck, dst)
            os.sync()

            if not has_weights(dst):
                print(f"[SENTRY] {ck.name}: WARNING, weights missing after copy")
            else:
                print(f"[SENTRY] {ck.name}: mirrored to Drive")
        except Exception as e:
            print(f"[SENTRY] {e}")


class EmbeddingRestore(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        if RESUME_FROM is None or not Path(RESUME_FROM).exists():
            return

        name = Path(RESUME_FROM).name
        try:
            step = int(name.split("-")[-1])
        except ValueError:
            raise RuntimeError(f"[RESTORE] cannot parse a step number from {name}")

        mdl = kwargs.get("model", model)
        w = mdl.get_input_embeddings().weight

        full_ck = FULL_CHECKPOINTS / f"step_{step:04d}" / "embeddings.pt"
        wake_ck = EMB_SNAPS / f"emb_step{step:04d}.pt"

        with torch.no_grad():
            if full_ck.exists():
                E = torch.load(full_ck, map_location="cpu")
                if tuple(E.shape) != tuple(w.shape):
                    raise RuntimeError(
                        f"[RESTORE] {full_ck} has shape {tuple(E.shape)}, "
                        f"embedding is {tuple(w.shape)}"
                    )
                w.copy_(E.to(w.device, dtype=w.dtype))
                src = full_ck
            elif wake_ck.exists():
                payload = torch.load(wake_ck, map_location="cpu")
                E = payload["wake_rows"] if isinstance(payload, dict) else payload
                want = (w.shape[0] - BASE_VOCAB, w.shape[1])
                if tuple(E.shape) != want:
                    raise RuntimeError(
                        f"[RESTORE] {wake_ck} has shape {tuple(E.shape)}, expected {want}"
                    )
                w[BASE_VOCAB:].copy_(E.to(w.device, dtype=w.dtype))
                src = wake_ck
            else:
                raise RuntimeError(
                    f"[RESTORE] resuming at step {step} but neither\n"
                    f"    {full_ck}\n    nor {wake_ck}\nexists. Without one of these "
                    "the run would restart the embedding geometry from the P2 matrix "
                    "while the step counter continues. Refusing to train."
                )

        # verify the restore took, chunked to keep host RAM bounded.
        drift = 0.0
        for i in range(BASE_VOCAB, w.shape[0], 4096):
            j = min(i + 4096, w.shape[0])
            drift = max(
                drift,
                (E_pre[i:j].float() - w[i:j].detach().cpu().float()).abs().max().item(),
            )
        if drift == 0.0:
            raise RuntimeError(
                "[RESTORE] the restored embedding is identical to the pre-training snapshot"
            )

        with torch.no_grad():
            m = compute_morpheme_loss(mdl).item()
        print(f"[RESTORE] embeddings restored from {src}")
        print(f"[RESTORE] max |delta| vs pre-training snapshot: {drift:.6f}")
        print(f"[RESTORE] L_morpheme now {m:.4f} "
              "(compare with the last value logged before the cut)")


class FullCheckpoint(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        try:
            step = state.global_step
            mdl = kwargs.get("model", model)

            full_ck = FULL_CHECKPOINTS / f"step_{step:04d}"
            if full_ck.exists():
                shutil.rmtree(full_ck)
            full_ck.mkdir(parents=True, exist_ok=True)

            mdl.save_pretrained(full_ck)
            tok.save_pretrained(full_ck)

            E = mdl.get_input_embeddings().weight.detach().cpu()
            torch.save(E, full_ck / "embeddings.pt")

            torch.save({
                'global_step': step,
                'best_metric': state.best_metric,
                'epoch': state.epoch,
            }, full_ck / "training_state.pt")

            os.sync()
            print(f"[FULL] Step {step}: saved to Drive")
        except Exception as e:
            print(f"[FULL] Step {state.global_step}: {e}")


class LossMonitor(TrainerCallback):
    def __init__(self):
        self.last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.last_train_loss = logs["loss"]
        if "eval_loss" in logs and self.last_train_loss is not None:
            gap = logs["eval_loss"] - self.last_train_loss
            if gap > 0.5:
                print(f"[WARN] Large train/eval gap: {gap:.2f}")


class StepTimer(TrainerCallback):
    def __init__(self):
        self.step_times = []
        self.last_time = None

    def on_step_end(self, args, state, control, **kw):
        now = time.time()
        if self.last_time is not None:
            self.step_times.append(now - self.last_time)
            if state.global_step % 10 == 0:
                recent = self.step_times[-10:]
                avg = sum(recent) / len(recent)
                remaining = (args.max_steps - state.global_step) * avg / 60
                print(f"[{state.global_step:4d}] {avg:.1f}s/step | ~{remaining:.0f}min remaining")
        self.last_time = now


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3):
        self.patience = patience
        self.best_val_loss = float("inf")
        self.wait = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        val_loss = metrics.get("eval_loss")
        if val_loss is None:
            return

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.wait = 0
            print(f"[EARLY] New best val loss: {val_loss:.4f}")
        else:
            self.wait += 1
            print(f"[EARLY] No improvement ({self.wait}/{self.patience}). "
                  f"Best: {self.best_val_loss:.4f}, current: {val_loss:.4f}")

        if self.wait >= self.patience:
            print(f"[EARLY] Stopping: no improvement for {self.patience} evals")
            control.should_training_stop = True


print("Callbacks defined")
print(f"  EarlyStoppingCallback (patience={EARLY_STOP_PATIENCE})")
