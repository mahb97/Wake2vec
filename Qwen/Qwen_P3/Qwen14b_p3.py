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

"""## config"""

from pathlib import Path
import json

P2_RUN = Path("/content/drive/MyDrive/wake2vec_qwen14b_p2_lora")
ADAPTER_SOURCE = P2_RUN / "full_checkpoints" / "step_1600" 
P1_RUN = Path("/content/drive/MyDrive/wake2vec_qwen14b_p1")
P1_SENTRY = P1_RUN / "sentry_backups" / "sentry_step_2700.pt"  # P1 best val 15.05

# P3 output paths 
RUN_DIR = Path("/content/drive/MyDrive/wake2vec_qwen14b_p3_morpheme_strong")
LOCAL_RUN = Path("/content/runs/wake2vec_qwen14b_p3_morpheme_strong")
SENTRY = RUN_DIR / "sentry_backups"
EMB_SNAPS = RUN_DIR / "emb_snaps"
FULL_CHECKPOINTS = RUN_DIR / "full_checkpoints"

# pre-training ref
PRE_SNAPSHOT = RUN_DIR / "embeddings_pre_p3.pt"
FORCE_NEW_PRE_SNAPSHOT = False  

for d in [RUN_DIR, LOCAL_RUN, SENTRY, EMB_SNAPS, FULL_CHECKPOINTS]:
    d.mkdir(parents=True, exist_ok=True)

# model
MODEL_NAME = "Qwen/Qwen2.5-14B"
MAX_MEMORY = {0: "11GB", "cpu": "30GB"}
FW_TEXT = "/content/FW_TEXT.txt"
MORPHEME_JSONL = "/content/wake_embedding_groups.jsonl"
DEVICE_JSONL = "/content/device_groups.jsonl"

# training hyperparams 
MAX_STEPS = 1000
LR = 2e-5                  
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
BATCH_SIZE = 1            
GRAD_ACCUM = 16           # effective batch 16
SEQ_LEN = 128           
SAVE_STEPS = 100          
EMB_SNAP_STEPS = 100        
LOG_STEPS = 50              
EVAL_STEPS = 100            

# LoRA 
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

# base vocab
BASE_VOCAB = 152064

# morpheme loss formulation 
MORPH_LOSS_MODE = "cosine"

# strip-group exclusion
MORPH_EXCLUDE_STRIP = True
STRIP_MORPHEMES = {"-s", "-ed", "-y", "-ing", "-er", "-ly", "-on", "-'s", "-a", "co-"}

# movable-pairs-only 
MORPH_MOVABLE_ONLY = True

# loss weights 
LAMBDA_MORPH = 1.0 if MORPH_LOSS_MODE == "cosine" else 50.0
LAMBDA_DEVICE = 2.0        
LAMBDA_REPULSION = 0.05    
LAMBDA_NORM = 0.01          
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

print("WAKE2VEC PHASE 3: QWEN 2.5-14B MORPHEME-COMPOSITIONAL ALIGNMENT CONFIG (STRONG)")
print(f"  Adapter source: {ADAPTER_SOURCE}")
print(f"  Wake-row source: {P1_SENTRY}")
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

"""## load P2 state"""

from transformers import AutoTokenizer

# verify P3 artifacts 
assert ADAPTER_SOURCE.exists(), f"Adapter source not found: {ADAPTER_SOURCE}"
assert P1_SENTRY.exists(), f"P1 sentry not found: {P1_SENTRY}"

# get tokenizer from the adapter checkpoint 
tok = AutoTokenizer.from_pretrained(str(ADAPTER_SOURCE), use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

TOTAL_VOCAB = len(tok)
NUM_WAKE = TOTAL_VOCAB - BASE_VOCAB

print(f"  Vocab size: {TOTAL_VOCAB}")
print(f"  Base vocab: {BASE_VOCAB}")
print(f"  Wake tokens: {NUM_WAKE}")

print(f" P1 Wake rows from {P1_SENTRY.name}...")
_p1_state = torch.load(P1_SENTRY, map_location="cpu", weights_only=False)
wake_weights = _p1_state["embeddings"]          
P1_WAKE_START = int(_p1_state["wake_start"])
P1_NUM_WAKE = int(_p1_state["num_wake"])
print(f"  Wake block: {tuple(wake_weights.shape)}  dtype {wake_weights.dtype}")
print(f"  P1 wake_start {P1_WAKE_START} | num_wake {P1_NUM_WAKE} | P1 step {_p1_state.get('step')}")

assert P1_WAKE_START == BASE_VOCAB, \
    f"P1 wake_start {P1_WAKE_START} != BASE_VOCAB {BASE_VOCAB}"
assert P1_NUM_WAKE == NUM_WAKE, \
    f"P1 num_wake {P1_NUM_WAKE} != tokenizer-implied {NUM_WAKE}"
assert wake_weights.shape[0] == NUM_WAKE, \
    f"Wake block/vocab mismatch: {wake_weights.shape[0]} vs {NUM_WAKE}"
del _p1_state

"""## morpheme group builder"""

with open(MORPHEME_JSONL, 'r') as f:
    raw_groups = [json.loads(line) for line in f]

print(f"  Raw groups: {len(raw_groups)}")
print(f"  Total examples: {sum(g['n_examples'] for g in raw_groups)}")


def tokenize_word(word, tokenizer):
    return tokenizer(word, add_special_tokens=False)["input_ids"]


def resolve_word(word, tokenizer):
    ids = tokenize_word(word, tokenizer)
    return ids if len(ids) > 0 else None


# morpheme groups with resolved token IDs
morpheme_groups = []
total_pairs = 0
skipped_groups = 0
frozen_pairs_dropped = 0

for g in raw_groups:
    morpheme = g["morpheme"]
    morpheme_type = g["morpheme_type"]
    examples = g["examples"]
    bases = g["bases"]

    word_ids_list = []
    base_ids_list = []
    pair_words = []

    for word, base in zip(examples, bases):
        w_ids = resolve_word(word, tok)
        b_ids = resolve_word(base, tok)
        if w_ids is not None and b_ids is not None:
            if MORPH_MOVABLE_ONLY and not any(
                i >= BASE_VOCAB for i in list(w_ids) + list(b_ids)
            ):
                frozen_pairs_dropped += 1
                continue
            word_ids_list.append(w_ids)
            base_ids_list.append(b_ids)
            pair_words.append((word, base))

    if len(word_ids_list) >= MIN_GROUP_SIZE:
        morpheme_groups.append({
            "morpheme": morpheme,
            "morpheme_type": morpheme_type,
            "word_ids": word_ids_list,
            "base_ids": base_ids_list,
            "pair_words": pair_words,
            "n_valid": len(word_ids_list),
        })
        total_pairs += len(word_ids_list)
    else:
        skipped_groups += 1

print(f"\n  Valid morpheme groups: {len(morpheme_groups)} (skipped {skipped_groups} with < {MIN_GROUP_SIZE} pairs)")
print(f"  Total word-base pairs: {total_pairs}")
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

"""## model setup (WakeOverlay + split output head)"""

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import PeftModel
import torch.nn as nn
import torch.nn.functional as F

set_seed(42)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map="auto",
    max_memory=MAX_MEMORY,
    trust_remote_code=True,
)

model.config.use_cache = False
model.config.attn_implementation = "eager"

print(f"Resizing embeddings: {BASE_VOCAB} -> {TOTAL_VOCAB}...")
model.resize_token_embeddings(TOTAL_VOCAB, mean_resizing=False)

wte = model.get_input_embeddings()
wte.weight.data = wte.weight.data.half()

with torch.no_grad():
    wte.weight.data[BASE_VOCAB:TOTAL_VOCAB] = wake_weights.to(
        wte.weight.device, dtype=wte.weight.dtype)
print(f"  Wake rows injected: {NUM_WAKE} at [{BASE_VOCAB}, {TOTAL_VOCAB})")

wte.weight.requires_grad = False         

class WakeOverlay(nn.Module):

    def __init__(self, full_embed, start, count):
        super().__init__()
        self.full_embed = full_embed
        self.start = int(start)
        self.count = int(count)
        self.end = int(start) + int(count)
        self.wake_embed = nn.Embedding(count, full_embed.embedding_dim)
        self.wake_embed.weight.data = (
            full_embed.weight.data[start:start + count].clone().float())

    @property
    def embedding_dim(self):
        return self.full_embed.embedding_dim

    @property
    def num_embeddings(self):
        return self.full_embed.num_embeddings

    def forward(self, input_ids):
        base_out = self.full_embed(input_ids)
        mask = (input_ids >= self.start) & (input_ids < self.end)
        wake_ids = (input_ids - self.start).clamp(0, self.count - 1)
        wake_out = self.wake_embed(wake_ids).to(base_out.dtype)
        return torch.where(mask.unsqueeze(-1), wake_out, base_out)

    def rows(self, ids):
        dev = self.wake_embed.weight.device
        if not torch.is_tensor(ids):
            ids = torch.as_tensor(ids, dtype=torch.long, device=dev)
        ids = ids.to(dev)
        flat = ids.reshape(-1)
        is_wake = (flat >= self.start) & (flat < self.end)
        zeros = torch.zeros_like(flat)
        base_rows = self.full_embed(torch.where(is_wake, zeros, flat)).float()
        wake_rows = self.wake_embed(
            torch.where(is_wake, flat - self.start, zeros)).float()
        out = torch.where(is_wake.unsqueeze(-1), wake_rows, base_rows)
        return out.reshape(*ids.shape, self.embedding_dim)

    def __getitem__(self, ids):
        return self.rows(ids)

    def full_matrix(self):
        with torch.no_grad():
            E = self.full_embed.weight.detach().float().cpu().clone()
            E[self.start:self.end] = self.wake_embed.weight.detach().float().cpu()
        return E


class SplitHead(nn.Module):

    def __init__(self, full_embed, wake_embed, start):
        super().__init__()
        self.full_embed = full_embed
        self.wake_embed = wake_embed
        self.start = int(start)

    @property
    def weight(self):
        return self.full_embed.weight

    def forward(self, h):
        base_logits = F.linear(h, self.full_embed.weight[:self.start])
        wake_logits = F.linear(h, self.wake_embed.weight.to(h.dtype))
        return torch.cat([base_logits, wake_logits], dim=-1)


overlay = WakeOverlay(wte, BASE_VOCAB, NUM_WAKE)
model.set_input_embeddings(overlay)

model.config.tie_word_embeddings = False
model.lm_head = SplitHead(wte, overlay.wake_embed, BASE_VOCAB)

print(f"WakeOverlay: {NUM_WAKE} rows x {overlay.embedding_dim}, fp32, "
      f"{overlay.wake_embed.weight.numel() * 4 / 2**30:.2f} GiB")
print(f"SplitHead: base [0,{BASE_VOCAB}) frozen fp16 + overlay "
      f"[{BASE_VOCAB},{TOTAL_VOCAB}) trainable fp32")

import accelerate.accelerator as _acc
if not getattr(_acc.Accelerator, "_wake2vec_patched", False):
    def _hacked_prepare(self, model, device_placement=None, evaluation_mode=False):
        self._models.append(model)
        return model
    _acc.Accelerator.prepare_model = _hacked_prepare
    _acc.Accelerator._wake2vec_patched = True
    print("  accelerate.prepare_model patched (4-bit + CPU offload)")

model = PeftModel.from_pretrained(model, str(ADAPTER_SOURCE), is_trainable=True)

# trainable set: overlay + LoRA
for p in model.parameters():
    p.requires_grad = False
overlay.wake_embed.weight.requires_grad = True
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
lora_params = sum(p.numel() for n, p in model.named_parameters()
                  if p.requires_grad and "lora" in n.lower())

print(f"\nP3 trainable setup:")
print(f"  Overlay (Wake rows): {overlay.wake_embed.weight.numel():,}")
print(f"  LoRA: {lora_params:,}")
print(f"  Total trainable: {trainable:,}")
print(f"  Frozen full matrix: {wte.weight.numel():,} "
      f"({wte.weight.dtype}, not in the optimizer)")
print(f"  GPU allocated: {torch.cuda.memory_allocated()/2**30:.2f} GiB "
      f"of {torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GiB")

assert overlay.wake_embed.weight.requires_grad, "overlay is not trainable"
assert not wte.weight.requires_grad, "full matrix must stay frozen"
assert trainable == overlay.wake_embed.weight.numel() + lora_params, \
    "unexpected parameters in the trainable set"

"""## pre-training embed snapshot"""

if PRE_SNAPSHOT.exists() and not FORCE_NEW_PRE_SNAPSHOT:
    raise RuntimeError(
        f"{PRE_SNAPSHOT} already exists.\n"
        "Skip to next. Set\n"
        "FORCE_NEW_PRE_SNAPSHOT=True in Cell 2 only to deliberately replace the\n"
        "pre-training reference, which invalidates every comparison against it."
    )

_E_snap = overlay.full_matrix()   # base (frozen) + overlay rows, reassembled
torch.save(_E_snap, PRE_SNAPSHOT)
print(f"  Saved: {PRE_SNAPSHOT}")
print(f"  Shape: {tuple(_E_snap.shape)}  dtype: {_E_snap.dtype}")
del _E_snap

"""## loss components (vectorized)"""

import torch.nn.functional as F

if not PRE_SNAPSHOT.exists():
    raise RuntimeError(
        f"{PRE_SNAPSHOT} not found.\n"
        "Run pre train snapshot, before the first training step, to create the\n"
        "pre-training reference."
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

        d_unit = F.normalize(directions, dim=1, eps=1e-8)
        g_unit = F.normalize(group_means, dim=1, eps=1e-8)
        cos = (d_unit * g_unit[self.pair_groups]).sum(dim=1)
        return (1.0 - cos).mean()


morph_index = MorphemeIndex(morpheme_groups, tok, device="cuda")

ALL_WAKE_IDS = torch.arange(BASE_VOCAB, TOTAL_VOCAB, dtype=torch.long).cuda()


def compute_morpheme_loss(model):
    return morph_index.compute_loss(overlay)


def compute_repulsion_loss(model, n_pairs=REPULSION_PAIRS, threshold=REPULSION_THRESHOLD):
    wake_embeds = overlay.wake_embed.weight          # (NUM_WAKE, dim), fp32

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
    wake_embeds = overlay.wake_embed.weight          # (NUM_WAKE, dim), fp32

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
    return device_index.compute_loss(overlay)


print(f"\nLoss functions defined:")
print(f"  L_morpheme: compositional direction consistency ({morph_index.n_groups} groups, {morph_index.total_pairs} pairs)")
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
        print(f"\n[NULL] {label}: no morpheme groups survive filtering; "
              "nothing to measure. Check the supervision counts above.")
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
    print(f"\n{'=' * 66}")
    print(f"MORPHEME PERMUTATION NULL{(' -- ' + label) if label else ''}")
    print("=" * 66)
    print(f"  pairs {n_pairs:,} | groups {n_groups} | permutations {n_perm}")
    print(f"  observed L_morph : {obs_global:.6f}  (consistency {1 - obs_global:.6f})")
    print(f"  null mean        : {mu:.6f}  (consistency {1 - mu:.6f})   sd {sd:.6f}")
    print(f"  EXCESS OVER CHANCE: {mu - obs_global:+.6f}      z = {z:+.2f}")
    print(f"  groups below z=-3: {len(sig)}/{n_groups}, holding "
          f"{sum(r['n_pairs'] for r in sig):,} pairs "
          f"({100 * sum(r['n_pairs'] for r in sig) / n_pairs:.1f}%)")
    print(f"  strongest: " + ", ".join(f"{r['morpheme']}({r['z']:+.1f})" for r in per_group[:6]))
    print("  NOTE: excess-over-chance compares across sets; z does not "
          "(it carries sample size).")

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

    print("TREATED vs CONTROL (auxiliary effect, LM differenced out)")
    print(f"  treated rows: {len(treated):,} of {out['wake_rows']:,} Wake rows "
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
    print("  READ: treated moving materially more than control means the auxiliary")
    print("  losses bit, even if the aggregate Wake isotropy looks flat. Treated ~=")
    print("  control means the null holds where the losses actually apply, which is")
    print("  the strong form of the finding.")
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

print("geom instruments defined:")
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

        # log during training 
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
                E = overlay.wake_embed.weight.detach().cpu().clone()
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
                    return  # already complete, skip silently
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

        w = overlay.wake_embed.weight
        want = (NUM_WAKE, w.shape[1])

        full_ck = FULL_CHECKPOINTS / f"step_{step:04d}" / "embeddings.pt"
        wake_ck = EMB_SNAPS / f"emb_step{step:04d}.pt"

        with torch.no_grad():
            if full_ck.exists():
                E = torch.load(full_ck, map_location="cpu")
                if tuple(E.shape) != (TOTAL_VOCAB, w.shape[1]):
                    raise RuntimeError(
                        f"[RESTORE] {full_ck} has shape {tuple(E.shape)}, "
                        f"expected {(TOTAL_VOCAB, w.shape[1])}"
                    )
                w.copy_(E[BASE_VOCAB:].to(w.device, dtype=w.dtype))
                src = full_ck
            elif wake_ck.exists():
                payload = torch.load(wake_ck, map_location="cpu")
                E = payload["wake_rows"] if isinstance(payload, dict) else payload
                if tuple(E.shape) != want:
                    raise RuntimeError(
                        f"[RESTORE] {wake_ck} has shape {tuple(E.shape)}, expected {want}"
                    )
                w.copy_(E.to(w.device, dtype=w.dtype))
                src = wake_ck
            else:
                raise RuntimeError(
                    f"[RESTORE] resuming at step {step} but neither\n"
                    f"    {full_ck}\n    nor {wake_ck}\nexists. Without one of these "
                    "the run would restart the embedding geometry from the P2 matrix "
                    "while the step counter continues. Refusing to train."
                )

        # verify the restore, chunked to keep host RAM bounded.
        drift = 0.0
        for i in range(0, NUM_WAKE, 4096):
            j = min(i + 4096, NUM_WAKE)
            drift = max(
                drift,
                (E_pre[BASE_VOCAB + i:BASE_VOCAB + j].float()
                 - w[i:j].detach().cpu().float()).abs().max().item(),
            )
        if drift == 0.0:
            raise RuntimeError(
                "[RESTORE] the restored embedding is identical to the pre-training "
                "snapshot. Either the checkpoint holds untrained rows or the copy did "
                "not take. Refusing to train."
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

            E = overlay.full_matrix()
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

print(f"  EarlyStoppingCallback (patience={EARLY_STOP_PATIENCE})")

"""## train"""

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}")

if trainable == 0:
    raise RuntimeError("No trainable parameters found")
if wte.weight.requires_grad:
    raise RuntimeError(
        "The full embedding is trainable. Re-run Cell 6: only the overlay and "
        "the LoRA adapters may train."
    )
print(f"  overlay {overlay.wake_embed.weight.numel():,} + LoRA "
      f"{trainable - overlay.wake_embed.weight.numel():,}")

args = TrainingArguments(
    output_dir=str(LOCAL_RUN),
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=int(MAX_STEPS * WARMUP_RATIO),
    lr_scheduler_type="cosine",
    weight_decay=WEIGHT_DECAY,
    fp16=False,
    bf16=True,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=5,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    report_to="none",
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = MorphemeTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    loss_log=loss_log,
    callbacks=[
        EmbeddingRestore(),
        EmbeddingSnapshot(),
        FullCheckpoint(),
        SentryMirror(),
        LossMonitor(),
        StepTimer(),
        EarlyStoppingCallback(patience=EARLY_STOP_PATIENCE),
    ],
)

print("WAKE2VEC PHASE 3: QWEN 2.5-14B MORPHEME-COMPOSITIONAL ALIGNMENT")
print(f"  Source: Qwen P2 step 1600 (best val 5.9209, the suspension source)")
print(f"  Train: {len(train_ds)} blocks | Val: {len(val_ds)} blocks")
print(f"  Steps: {MAX_STEPS} (early stop patience={EARLY_STOP_PATIENCE})")
print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  LR: {LR} (halved from P2)")
print(f"  LoRA targets: {LORA_TARGETS}")
print(f"  Trainable: Wake embed rows + LoRA adapters")
print(f"  Loss: LM + {LAMBDA_MORPH}*morph + {LAMBDA_DEVICE}*device + {LAMBDA_REPULSION}*repulsion + {LAMBDA_NORM}*norm")
print(f"  Morpheme groups: {morph_index.n_groups} ({morph_index.total_pairs} word-base pairs)")
print(f"  Device groups: {device_index.n_groups} ({device_index.total_words} words, {DEVICE_TRIPLETS} triplets/step)")

t0 = time.time()
if RESUME_FROM is not None and RESUME_FROM.exists():
    local_ckpt = LOCAL_RUN / RESUME_FROM.name
    if not local_ckpt.exists():
        shutil.copytree(RESUME_FROM, local_ckpt)

    _ts = local_ckpt / "trainer_state.json"
    if _ts.exists():
        _st = json.loads(_ts.read_text())
        _changed = {}
        for _k, _v in (("eval_steps", EVAL_STEPS),
                       ("save_steps", SAVE_STEPS),
                       ("logging_steps", LOG_STEPS)):
            if _st.get(_k) != _v:
                _changed[_k] = f"{_st.get(_k)} -> {_v}"
                _st[_k] = _v
        if _changed:
            _ts.write_text(json.dumps(_st, indent=2))
            print(f"[RESUME] trainer_state cadence patched: {_changed}")

    print(f"[RESUME] Resuming from {RESUME_FROM.name}")
    trainer.train(resume_from_checkpoint=str(local_ckpt))
else:
    trainer.train()
elapsed = (time.time() - t0) / 60

print(f"\nTRAINING COMPLETE ({elapsed:.1f} minutes)")

# save final model
final_dir = RUN_DIR / "final"
final_dir.mkdir(exist_ok=True)

model.save_pretrained(str(final_dir))
tok.save_pretrained(str(final_dir))

final_emb = overlay.full_matrix()
torch.save(final_emb, final_dir / "embeddings.pt")

# save loss component log
with open(RUN_DIR / "loss_components.json", 'w') as f:
    json.dump(loss_log, f, indent=2)

print(f"Final model saved to {final_dir}")
print(f"Loss component log: {RUN_DIR / 'loss_components.json'}")

"""## loss curves (with component breakdown)"""

import matplotlib.pyplot as plt

state_files = list(LOCAL_RUN.rglob("trainer_state.json"))
train_data, val_data = [], []
if state_files:
    latest = max(state_files, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        state = json.load(f)

    logs = state.get("log_history", [])
    train_data = [(d["step"], d["loss"]) for d in logs if "loss" in d and "eval_loss" not in d]
    val_data = [(d["step"], d["eval_loss"]) for d in logs if "eval_loss" in d]

# component breakdown from loss log 

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Wake2Vec P3 Qwen 2.5-14B -- Loss Curves", fontsize=14, fontweight="bold")

# 1. Train vs Val
ax = axes[0, 0]
if train_data:
    steps, losses = zip(*train_data)
    ax.plot(steps, losses, 'b-o', label='Train', alpha=0.7, markersize=3)
if val_data:
    v_steps, v_losses = zip(*val_data)
    ax.plot(v_steps, v_losses, 'r-s', label='Val', alpha=0.7, markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('Train vs Val Loss (total)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. LM loss component
ax = axes[0, 1]
if loss_log:
    ll_steps = [d["step"] for d in loss_log if d["step"] > 0]
    ll_lm = [d["lm"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_lm, 'b-o', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('LM Loss Component')
ax.grid(True, alpha=0.3)

# 3. Morpheme loss component
ax = axes[0, 2]
if loss_log:
    ll_morph = [d["morph"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_morph, 'g-o', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('L_morpheme (direction consistency)')
ax.grid(True, alpha=0.3)

# 4. Device loss component
ax = axes[1, 0]
if loss_log:
    ll_device = [d["device"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_device, 'c-o', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('L_device (triplet contrastive)')
ax.grid(True, alpha=0.3)

# 5. Repulsion loss component
ax = axes[1, 1]
if loss_log:
    ll_repul = [d["repulsion"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_repul, 'm-s', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('L_repulsion')
ax.grid(True, alpha=0.3)

# 6. Norm loss component
ax = axes[1, 2]
if loss_log:
    ll_norm = [d["norm"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_norm, 'orange', marker='o', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('L_norm')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RUN_DIR / "p3_qwen14b_loss_curves.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plot saved: {plot_path}")
plt.show()

if train_data:
    print(f"\nFinal train loss: {losses[-1]:.4f}")
if val_data:
    print(f"Final val loss: {v_losses[-1]:.4f}")
    print(f"Best val loss: {min(v_losses):.4f}")

"""## embedding analysis"""

import numpy as np
from numpy.linalg import norm as l2
from scipy import stats
from sklearn.decomposition import PCA

E_post = final_emb.float().numpy()
vocab_size, dim = E_post.shape
num_new = vocab_size - BASE_VOCAB

E_base = E_post[:BASE_VOCAB]
E_new = E_post[BASE_VOCAB:]

# pre-training embeddings for drift analysis
pre_path = RUN_DIR / "embeddings_pre_p3.pt"
has_pre = pre_path.exists()
if has_pre:
    E_pre_all = torch.load(pre_path, map_location="cpu").float().numpy()
    E_pre_base = E_pre_all[:BASE_VOCAB]
    E_pre_new = E_pre_all[BASE_VOCAB:]
    print(f"[PRE] Loaded pre-refinement embeddings: {E_pre_all.shape}")
else:
    print("[PRE] No pre-refinement snapshot found (skipping drift analysis)")

# norm analysis with statistical tests 

norms = l2(E_post, axis=1)
base_norms = norms[:BASE_VOCAB]
new_norms = norms[BASE_VOCAB:]

t_stat, t_pval = stats.ttest_ind(base_norms, new_norms, equal_var=False)
u_stat, u_pval = stats.mannwhitneyu(base_norms, new_norms, alternative='two-sided')
pooled_std = np.sqrt((base_norms.std()**2 + new_norms.std()**2) / 2)
cohens_d = (base_norms.mean() - new_norms.mean()) / pooled_std

print("1. NORM ANALYSIS")
print(f"  Global  -- mean: {norms.mean():.4f}, std: {norms.std():.4f}, "
      f"min: {norms.min():.4f}, max: {norms.max():.4f}")
print(f"  Base    -- mean: {base_norms.mean():.4f}, std: {base_norms.std():.4f} (n={BASE_VOCAB})")
print(f"  New     -- mean: {new_norms.mean():.4f}, std: {new_norms.std():.4f} (n={num_new})")
print(f"  Welch t-test:    t={t_stat:.4f}, p={t_pval:.2e}")
print(f"  Mann-Whitney U:  U={u_stat:.0f}, p={u_pval:.2e}")
print(f"  Cohen's d:       {cohens_d:.4f}")

# eigenvalue-based isotropy 
def compute_isotropy(embeddings, label=""):
    rng = np.random.default_rng(42)
    n = min(5000, len(embeddings))
    idx = rng.choice(len(embeddings), size=n, replace=False)

    mu = embeddings.mean(axis=0)
    sample = embeddings[idx] - mu
    nrm = l2(sample, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-12, 1e-12, nrm)
    sample = sample / nrm

    cos_mat = sample @ sample.T
    np.fill_diagonal(cos_mat, 0)
    Z = np.exp(cos_mat).sum(axis=1)

    isotropy = Z.min() / Z.max()
    mean_cos = (cos_mat.sum()) / (n * (n - 1))
    return isotropy, mean_cos, n

iso_all, cos_all, n_all = compute_isotropy(E_post)
iso_base, cos_base, n_base = compute_isotropy(E_base)
iso_new, cos_new, n_new = compute_isotropy(E_new)

print("2. ISOTROPY (Mu et al. 2018 partition function ratio)")
print(f"  All tokens  -- isotropy: {iso_all:.6f}, mean_cos: {cos_all:.4f} (n={n_all})")
print(f"  Base tokens -- isotropy: {iso_base:.6f}, mean_cos: {cos_base:.4f} (n={n_base})")
print(f"  New tokens  -- isotropy: {iso_new:.6f}, mean_cos: {cos_new:.4f} (n={n_new})")

# 2b. ISOTROPY DELTA (pre vs post) 
iso_delta_summary = None
if has_pre:
    iso_all_pre, cos_all_pre, _ = compute_isotropy(E_pre_all)
    iso_base_pre, cos_base_pre, _ = compute_isotropy(E_pre_base)
    iso_new_pre, cos_new_pre, _ = compute_isotropy(E_pre_new)

    print("2b. ISOTROPY DELTA (pre -> post)")
    print(f"  {'Population':>12s} {'Pre':>10s} {'Post':>10s} {'Delta':>11s}")
    print(f"  {'-'*12:>12s} {'-'*10:>10s} {'-'*10:>10s} {'-'*11:>11s}")
    print(f"  {'All':>12s} {iso_all_pre:>10.6f} {iso_all:>10.6f} {iso_all - iso_all_pre:>+11.6f}")
    print(f"  {'Base':>12s} {iso_base_pre:>10.6f} {iso_base:>10.6f} {iso_base - iso_base_pre:>+11.6f}")
    print(f"  {'Wake':>12s} {iso_new_pre:>10.6f} {iso_new:>10.6f} {iso_new - iso_new_pre:>+11.6f}")
    print(f"\n  Wake mean_cos: {cos_new_pre:.6f} -> {cos_new:.6f} "
          f"(delta {cos_new - cos_new_pre:+.6f})")

    wake_iso_delta = iso_new - iso_new_pre
    print(f"\n  >> WAKE ISOTROPY MOVEMENT: {wake_iso_delta:+.6f}")
    if abs(wake_iso_delta) < 1e-3:
        print("  >> Reads as NULL: strong lambdas did not move the Wake geometry,")
        print("     from the one configuration in the lineup that had structure to move.")
    else:
        print("  >> Reads as NON-NULL: the Wake geometry moved under strong lambdas.")
        print("     First measurable geometric movement in the project. Verify against")
        print("     the direction-consistency delta below before claiming it.")

    iso_delta_summary = {
        "iso_all_pre": float(iso_all_pre), "iso_all_post": float(iso_all),
        "iso_base_pre": float(iso_base_pre), "iso_base_post": float(iso_base),
        "iso_new_pre": float(iso_new_pre), "iso_new_post": float(iso_new),
        "wake_isotropy_delta": float(wake_iso_delta),
        "wake_mean_cos_pre": float(cos_new_pre), "wake_mean_cos_post": float(cos_new),
    }
else:
    print("\n[2b] No pre-refinement snapshot; isotropy delta skipped.")

# embed drift (pre vs post refinement) 

if has_pre:
    base_pre_norms = l2(E_pre_base, axis=1, keepdims=True)
    base_post_norms = l2(E_base, axis=1, keepdims=True)
    base_pre_norms = np.where(base_pre_norms < 1e-12, 1e-12, base_pre_norms)
    base_post_norms = np.where(base_post_norms < 1e-12, 1e-12, base_post_norms)

    drift_cos = np.sum(
        (E_pre_base / base_pre_norms) * (E_base / base_post_norms), axis=1
    )
    drift_l2 = l2(E_base - E_pre_base, axis=1)

    new_pre_norms = l2(E_pre_new, axis=1, keepdims=True)
    new_post_norms = l2(E_new, axis=1, keepdims=True)
    new_pre_norms = np.where(new_pre_norms < 1e-12, 1e-12, new_pre_norms)
    new_post_norms = np.where(new_post_norms < 1e-12, 1e-12, new_post_norms)

    wake_drift_cos = np.sum(
        (E_pre_new / new_pre_norms) * (E_new / new_post_norms), axis=1
    )
    wake_drift_l2 = l2(E_new - E_pre_new, axis=1)

    print("3. EMBEDDING DRIFT (pre -> post refinement)")
    print(f"  Base tokens:")
    print(f"    Cosine sim -- mean: {drift_cos.mean():.6f}, std: {drift_cos.std():.6f}")
    print(f"    L2 dist    -- mean: {drift_l2.mean():.6f}, std: {drift_l2.std():.6f}")
    print(f"  Wake tokens:")
    print(f"    Cosine sim -- mean: {wake_drift_cos.mean():.6f}, std: {wake_drift_cos.std():.6f}")
    print(f"    L2 dist    -- mean: {wake_drift_l2.mean():.6f}, std: {wake_drift_l2.std():.6f}")

    if drift_cos.mean() > 0.999:
        print("  >> Base tokens barely moved (expected: gradient masked)")
    elif drift_cos.mean() > 0.99:
        print("  >> Moderate base drift detected")
    else:
        print("  >> Significant base drift detected (unexpected if masked)")

    wake_drift_order = np.argsort(wake_drift_cos)
    print(f"\n  Top 20 most-drifted Wake tokens:")
    for rank, idx in enumerate(wake_drift_order[:20]):
        wake_id = BASE_VOCAB + idx
        token_str = tok.convert_ids_to_tokens(int(wake_id))
        print(f"    {rank+1:2d}. [{wake_id:5d}] {token_str!r:25s}  "
              f"cos={wake_drift_cos[idx]:.6f}  L2={wake_drift_l2[idx]:.4f}")

# nearest neighbour sanity checks 

print("4. NEAREST NEIGHBORS (Wake tokens -> base vocab)")
NN_CHUNK = 20000

sample_wake_ids = list(range(BASE_VOCAB, BASE_VOCAB + 10))
if num_new > 100:
    mid = BASE_VOCAB + num_new // 2
    sample_wake_ids += list(range(mid, mid + 5))
if num_new > 1000:
    sample_wake_ids += list(range(BASE_VOCAB + num_new - 5, BASE_VOCAB + num_new))

for wake_id in sample_wake_ids:
    wake_token = tok.convert_ids_to_tokens(wake_id)
    wv = E_post[wake_id]
    wv = wv / max(float(l2(wv)), 1e-12)

    best_s = np.full(5, -np.inf, dtype=np.float64)
    best_i = np.zeros(5, dtype=np.int64)
    for start in range(0, BASE_VOCAB, NN_CHUNK):
        stop = min(start + NN_CHUNK, BASE_VOCAB)
        blk = E_post[start:stop]                     
        bn = l2(blk, axis=1)
        bn = np.where(bn < 1e-12, 1e-12, bn)
        sims = (blk @ wv) / bn
        k = min(5, len(sims))
        top = np.argpartition(sims, -k)[-k:]
        cand_s = np.concatenate([best_s, sims[top]])
        cand_i = np.concatenate([best_i, top + start])
        order = np.argsort(cand_s)[-5:][::-1]
        best_s, best_i = cand_s[order], cand_i[order]

    neighbors = [(tok.convert_ids_to_tokens(int(i)), sc)
                 for i, sc in zip(best_i, best_s)]
    nb_str = ", ".join(f"{t!r}({sc:.3f})" for t, sc in neighbors)
    print(f"  {wake_token!r:25s} -> {nb_str}")

# intrinsic dimensionality (PCA)

print("5. INTRINSIC DIMENSIONALITY (PCA explained variance)")

n_components = min(100, dim, BASE_VOCAB, num_new)

PCA_SAMPLE = 20000
_rng_pca = np.random.default_rng(42)

def _fit_pca(E, k):
    if len(E) > PCA_SAMPLE:
        sel = _rng_pca.choice(len(E), size=PCA_SAMPLE, replace=False)
        E = E[sel]
    return PCA(n_components=min(k, *E.shape)).fit(E)

pca_base = _fit_pca(E_base, n_components)
pca_new = _fit_pca(E_new, n_components)

cumvar_base = np.cumsum(pca_base.explained_variance_ratio_)
cumvar_new = np.cumsum(pca_new.explained_variance_ratio_)

dim90_base = np.searchsorted(cumvar_base, 0.90) + 1
dim90_new = np.searchsorted(cumvar_new, 0.90) + 1
dim95_base = np.searchsorted(cumvar_base, 0.95) + 1
dim95_new = np.searchsorted(cumvar_new, 0.95) + 1

print(f"  Base tokens -- 90% variance in {dim90_base} PCs, 95% in {dim95_base} PCs")
print(f"  New tokens  -- 90% variance in {dim90_new} PCs, 95% in {dim95_new} PCs")
print(f"  Top-1 PC explains: base={pca_base.explained_variance_ratio_[0]:.4f}, "
      f"new={pca_new.explained_variance_ratio_[0]:.4f}")

# pairwise cosine similarity 
print("6. PAIRWISE COSINE SIMILARITY DISTRIBUTIONS")

rng = np.random.default_rng(42)
n_sample = 2000

def sample_pairwise_cosines(E1, E2, n=2000):
    idx1 = rng.choice(len(E1), size=min(n, len(E1)), replace=False)
    idx2 = rng.choice(len(E2), size=min(n, len(E2)), replace=False)
    s1, s2 = E1[idx1], E2[idx2]
    n1 = l2(s1, axis=1, keepdims=True)
    n2 = l2(s2, axis=1, keepdims=True)
    n1 = np.where(n1 < 1e-12, 1e-12, n1)
    n2 = np.where(n2 < 1e-12, 1e-12, n2)
    cos_mat = (s1 / n1) @ (s2 / n2).T
    if E1 is E2:
        return cos_mat[np.triu_indices_from(cos_mat, k=1)]
    return cos_mat.ravel()

cos_bb = sample_pairwise_cosines(E_base, E_base, n_sample)
cos_nn = sample_pairwise_cosines(E_new, E_new, n_sample)
cos_bn = sample_pairwise_cosines(E_base, E_new, n_sample)

print(f"  (base,base) -- mean: {cos_bb.mean():.4f}, std: {cos_bb.std():.4f}")
print(f"  (new,new)   -- mean: {cos_nn.mean():.4f}, std: {cos_nn.std():.4f}")
print(f"  (base,new)  -- mean: {cos_bn.mean():.4f}, std: {cos_bn.std():.4f}")

ks_stat, ks_pval = stats.ks_2samp(cos_bb, cos_nn)
print(f"  KS test (base-base vs new-new): D={ks_stat:.4f}, p={ks_pval:.2e}")

# analysis plots 
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Wake2Vec P3 Qwen 2.5-14B -- Embedding Analysis", fontsize=14, fontweight="bold")

ax = axes[0, 0]
ax.hist(base_norms, bins=50, alpha=0.5, label=f"Base (u={base_norms.mean():.2f})", density=True)
ax.hist(new_norms, bins=50, alpha=0.5, label=f"New (u={new_norms.mean():.2f})", density=True)
ax.set_xlabel("L2 norm")
ax.set_ylabel("Density")
ax.set_title("Norm Distribution: Base vs New")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.hist(cos_bb, bins=80, alpha=0.4, label=f"base-base (u={cos_bb.mean():.3f})", density=True)
ax.hist(cos_nn, bins=80, alpha=0.4, label=f"new-new (u={cos_nn.mean():.3f})", density=True)
ax.hist(cos_bn, bins=80, alpha=0.4, label=f"base-new (u={cos_bn.mean():.3f})", density=True)
ax.set_xlabel("Cosine similarity")
ax.set_ylabel("Density")
ax.set_title("Pairwise Cosine Distributions")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.plot(range(1, n_components+1), cumvar_base, 'b-', label=f"Base (90%@{dim90_base})")
ax.plot(range(1, n_components+1), cumvar_new, 'r-', label=f"New (90%@{dim90_new})")
ax.axhline(0.90, linestyle='--', color='gray', alpha=0.5, label="90% threshold")
ax.set_xlabel("Principal component")
ax.set_ylabel("Cumulative explained variance")
ax.set_title("Intrinsic Dimensionality")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
if has_pre:
    ax.hist(drift_cos, bins=80, alpha=0.5, color="coral", label=f"Base (u={drift_cos.mean():.4f})")
    ax.hist(wake_drift_cos, bins=80, alpha=0.5, color="steelblue", label=f"Wake (u={wake_drift_cos.mean():.4f})")
    ax.set_xlabel("Cosine similarity (pre -> post)")
    ax.set_ylabel("Frequency")
    ax.set_title("Embedding Drift (P3 refinement)")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No pre-refinement\nsnapshot available",
            ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_title("Embedding Drift (unavailable)")
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.scatter(range(BASE_VOCAB), base_norms, s=0.1, alpha=0.3, label="Base", c="blue")
ax.scatter(range(BASE_VOCAB, vocab_size), new_norms, s=0.1, alpha=0.3, label="New", c="red")
ax.set_xlabel("Token index")
ax.set_ylabel("L2 norm")
ax.set_title("Norm by Token Index")
ax.legend(fontsize=8, markerscale=10)
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.bar(range(1, 21), pca_base.explained_variance_ratio_[:20], alpha=0.5, label="Base")
ax.bar(range(1, 21), pca_new.explained_variance_ratio_[:20], alpha=0.5, label="New")
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance ratio")
ax.set_title("Top-20 PC Eigenspectrum")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
analysis_plot = RUN_DIR / "p3_qwen14b_analysis.png"
plt.savefig(analysis_plot, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {analysis_plot}")
plt.show()

"""## morpheme-specific analysis"""

print("MORPHEME-SPECIFIC ANALYSIS (compositional directions)")

E_final = final_emb

def analyze_morpheme_group(group, embed_matrix, tokenizer):
    word_vecs = []
    base_vecs = []
    pair_labels = []

    for w_ids, b_ids, (word, base) in zip(
        group["word_ids"], group["base_ids"], group["pair_words"]
    ):
        w_t = torch.tensor(w_ids, dtype=torch.long)
        b_t = torch.tensor(b_ids, dtype=torch.long)
        word_vecs.append(embed_matrix[w_t].mean(dim=0))
        base_vecs.append(embed_matrix[b_t].mean(dim=0))
        pair_labels.append((word, base))

    if len(word_vecs) < 2:
        return None

    word_stack = torch.stack(word_vecs)
    base_stack = torch.stack(base_vecs)
    directions = word_stack - base_stack

    # mean direction 
    d_mean = directions.mean(dim=0)

    # cosine similarity of each direction to the mean direction
    d_norm = F.normalize(directions, dim=1)
    mean_norm = F.normalize(d_mean.unsqueeze(0), dim=1)
    cosines_to_mean = (d_norm @ mean_norm.T).squeeze()

    # direction magnitudes
    magnitudes = torch.norm(directions, dim=1)

    return {
        "cosines_to_mean": cosines_to_mean,
        "mean_cosine": cosines_to_mean.mean().item(),
        "std_cosine": cosines_to_mean.std().item(),
        "mean_magnitude": magnitudes.mean().item(),
        "std_magnitude": magnitudes.std().item(),
        "mean_direction": d_mean,
        "pair_labels": pair_labels,
        "n_pairs": len(word_vecs),
    }


# analyze all groups
group_results = []
for g in morpheme_groups:
    result = analyze_morpheme_group(g, E_final, tok)
    if result is not None:
        result["morpheme"] = g["morpheme"]
        result["morpheme_type"] = g["morpheme_type"]
        group_results.append(result)

# sort by direction consistency 
group_results.sort(key=lambda x: -x["mean_cosine"])
dir_delta_summary = None
if has_pre:
    E_pre_torch = torch.from_numpy(E_pre_all).float()

    pre_by_morpheme = {}
    for g in morpheme_groups:
        r_pre = analyze_morpheme_group(g, E_pre_torch, tok)
        if r_pre is not None:
            pre_by_morpheme[g["morpheme"]] = r_pre

    paired = [(r, pre_by_morpheme[r["morpheme"]])
              for r in group_results if r["morpheme"] in pre_by_morpheme]

    if paired:
        deltas = [post["mean_cosine"] - pre["mean_cosine"] for post, pre in paired]
        mag_deltas = [post["mean_magnitude"] - pre["mean_magnitude"] for post, pre in paired]
        mean_pre = sum(p["mean_cosine"] for _, p in paired) / len(paired)
        mean_post = sum(p["mean_cosine"] for p, _ in paired) / len(paired)
        mean_delta = sum(deltas) / len(deltas)
        max_delta = max(deltas, key=abs)
        n_improved = sum(1 for d in deltas if d > 0)

        print("DIRECTION-CONSISTENCY DELTA (pre -> post, paired by morpheme)")
        print(f"  Groups paired: {len(paired)}")
        print(f"  Mean direction consistency: {mean_pre:.6f} -> {mean_post:.6f} "
              f"(delta {mean_delta:+.6f})")
        print(f"  Groups that improved: {n_improved}/{len(paired)}")
        print(f"  Largest single-group move: {max_delta:+.6f}")
        print(f"  Mean direction magnitude delta: {sum(mag_deltas)/len(mag_deltas):+.6f}")

        print(f"\n  Largest movers (by |delta|):")
        print(f"  {'Morpheme':>12s} {'Pairs':>5s} {'Pre':>8s} {'Post':>8s} {'Delta':>9s}")
        print(f"  {'-'*12:>12s} {'-'*5:>5s} {'-'*8:>8s} {'-'*8:>8s} {'-'*9:>9s}")
        for post, pre in sorted(paired,
                                key=lambda pp: -abs(pp[0]["mean_cosine"] - pp[1]["mean_cosine"]))[:15]:
            print(f"  {post['morpheme']:>12s} {post['n_pairs']:>5d} "
                  f"{pre['mean_cosine']:>8.4f} {post['mean_cosine']:>8.4f} "
                  f"{post['mean_cosine'] - pre['mean_cosine']:>+9.4f}")

        print(f"\n  >> MEAN DIRECTION-CONSISTENCY MOVEMENT: {mean_delta:+.6f}")
        if abs(mean_delta) < 1e-3:
            print("  >> Reads as NULL on the quantity L_morpheme directly optimises.")
        else:
            print("  >> Reads as NON-NULL. Cross-check against the isotropy delta and")
            print("     against the L_morph training curve before claiming it.")

        dir_delta_summary = {
            "n_groups_paired": len(paired),
            "mean_direction_consistency_pre": float(mean_pre),
            "mean_direction_consistency_post": float(mean_post),
            "mean_delta": float(mean_delta),
            "n_improved": int(n_improved),
            "max_abs_delta": float(max_delta),
        }
else:
    print("\n[DIR-DELTA] No pre-refinement snapshot; direction-consistency delta skipped.")

print(f"\nAnalyzed {len(group_results)} morpheme groups")
print(f"\n  Top 15 most consistent morpheme directions:")
print(f"  {'Morpheme':>12s} {'Type':>7s} {'Pairs':>5s} {'MeanCos':>8s} {'StdCos':>8s} {'MeanMag':>8s}")
print(f"  {'-'*12:>12s} {'-'*7:>7s} {'-'*5:>5s} {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s}")
for r in group_results[:15]:
    print(f"  {r['morpheme']:>12s} {r['morpheme_type']:>7s} {r['n_pairs']:>5d} "
          f"{r['mean_cosine']:>8.4f} {r['std_cosine']:>8.4f} {r['mean_magnitude']:>8.4f}")

print(f"\n  Bottom 10 (least consistent directions):")
for r in group_results[-10:]:
    print(f"  {r['morpheme']:>12s} {r['morpheme_type']:>7s} {r['n_pairs']:>5d} "
          f"{r['mean_cosine']:>8.4f} {r['std_cosine']:>8.4f} {r['mean_magnitude']:>8.4f}")

# view some interesting groups
interesting = [r for r in group_results if r["morpheme"] in ["-ing", "-ed", "-s", "un-", "re-", "-ly"]]
for r in interesting:
    print(f"\n  === {r['morpheme']} ({r['morpheme_type']}, {r['n_pairs']} pairs) ===")
    print(f"    Direction consistency: {r['mean_cosine']:.4f} +/- {r['std_cosine']:.4f}")
    print(f"    Direction magnitude: {r['mean_magnitude']:.4f} +/- {r['std_magnitude']:.4f}")

    # show most/least aligned pairs
    sorted_idx = r['cosines_to_mean'].argsort(descending=True)
    print(f"    Most aligned:")
    for i in sorted_idx[:3]:
        w, b = r['pair_labels'][i]
        print(f"      {w} -> {b}  (cos={r['cosines_to_mean'][i].item():.4f})")
    print(f"    Least aligned:")
    for i in sorted_idx[-3:]:
        w, b = r['pair_labels'][i]
        print(f"      {w} -> {b}  (cos={r['cosines_to_mean'][i].item():.4f})")

# direction similarity between morpheme groups

if len(group_results) >= 5:
    print(f"\n  Cross-group direction similarity (top 20 groups):")
    top_groups = group_results[:20]
    dir_vecs = torch.stack([r["mean_direction"] for r in top_groups])
    dir_normed = F.normalize(dir_vecs, dim=1)
    cross_sim = (dir_normed @ dir_normed.T)

    # most similar pairs
    pairs = []
    for i in range(len(top_groups)):
        for j in range(i + 1, len(top_groups)):
            pairs.append((cross_sim[i, j].item(), top_groups[i]["morpheme"], top_groups[j]["morpheme"]))
    pairs.sort(reverse=True)

    print(f"\n    Most similar direction pairs:")
    for sim, m1, m2 in pairs[:10]:
        print(f"      {m1:>10s} <-> {m2:<10s}  cos={sim:.4f}")

    print(f"\n    Most different direction pairs:")
    for sim, m1, m2 in pairs[-5:]:
        print(f"      {m1:>10s} <-> {m2:<10s}  cos={sim:.4f}")

# device group analysis 
print("DEVICE GROUP ANALYSIS (stylistic clustering)")

device_centroids = {}
for dg in device_groups:
    vecs = []
    for w_ids in dg["word_ids"]:
        t = torch.tensor(w_ids, dtype=torch.long)
        vecs.append(E_final[t].mean(dim=0))
    if vecs:
        stack = torch.stack(vecs)
        device_centroids[dg["device"]] = stack.mean(dim=0)

        # intra-group pairwise cosine
        normed = F.normalize(stack, dim=1)
        n = len(vecs)
        if n > 1:
            cos_mat = (normed @ normed.T)
            mask_ut = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            intra_cos = cos_mat[mask_ut].mean().item()
        else:
            intra_cos = float('nan')

        print(f"\n  {dg['device']:>15s} ({dg['n_valid']} words)")
        print(f"    Intra-group mean cosine: {intra_cos:.4f}")

# inter-group centroid similarity
if len(device_centroids) >= 2:
    names = list(device_centroids.keys())
    cent_stack = torch.stack([device_centroids[n] for n in names])
    cent_normed = F.normalize(cent_stack, dim=1)
    cross_cos = (cent_normed @ cent_normed.T)

    print(f"\n  Inter-group centroid cosine similarities:")
    header = f"  {'':>15s}" + "".join(f"{n:>12s}" for n in names)
    print(header)
    for i, n1 in enumerate(names):
        row = f"  {n1:>15s}"
        for j, n2 in enumerate(names):
            row += f"{cross_cos[i, j].item():>12.4f}"
        print(row)

"""## post-training geometry"""

POST_NULL = morph_null_report(final_emb, label="POST (after P3)")

print("PRE vs POST")
print(f"  {'quantity':>26s} {'pre':>12s} {'post':>12s} {'delta':>12s}")
print("  " + "-" * 64)
for k, name in (("observed_L_morph", "L_morph"),
                ("observed_consistency", "direction consistency"),
                ("excess_over_chance", "EXCESS OVER CHANCE")):
    print(f"  {name:>26s} {PRE_NULL[k]:>12.6f} {POST_NULL[k]:>12.6f} "
          f"{POST_NULL[k] - PRE_NULL[k]:>+12.6f}")
print(f"  {'groups below z=-3':>26s} {PRE_NULL['n_groups_below_z3']:>12d} "
      f"{POST_NULL['n_groups_below_z3']:>12d} "
      f"{POST_NULL['n_groups_below_z3'] - PRE_NULL['n_groups_below_z3']:>+12d}")

_excess_delta = POST_NULL["excess_over_chance"] - PRE_NULL["excess_over_chance"]
print(f"\n  >> CHANGE IN EXCESS OVER CHANCE: {_excess_delta:+.6f}")
print("  The excess is the headline, NOT the raw L_morph: the raw value moves with")
print("  the chance floor, which moves with the group-size distribution.")
if abs(_excess_delta) < 0.005:
    print("  >> Reads as NULL: a properly scaled objective on audited supervision")
    print("     did not move the geometry. This is the strong form of the null.")
else:
    print("  >> Reads as MOVEMENT. Cross-check the per-group table below before")
    print("     claiming it: movement should concentrate in groups that already")
    print("     carried structure, not appear uniformly.")

# which groups moved, against which started with structure.
_pre_by = {r["morpheme"]: r for r in PRE_NULL["per_group"]}
_moved = []
for r in POST_NULL["per_group"]:
    q = _pre_by.get(r["morpheme"])
    if q:
        _moved.append((r["morpheme"], r["n_pairs"], q["z"], r["z"], r["z"] - q["z"],
                       r["excess"] - q["excess"]))
_moved.sort(key=lambda x: x[5], reverse=True)
print(f"\n  LARGEST PER-GROUP MOVERS (excess gained)")
print(f"  {'morpheme':>14s} {'n':>5s} {'z_pre':>8s} {'z_post':>8s} {'dz':>8s} {'d_excess':>10s}")
for m, n, zp, zq, dz, de in _moved[:15]:
    print(f"  {m:>14s} {n:>5d} {zp:>+8.2f} {zq:>+8.2f} {dz:>+8.2f} {de:>+10.5f}")
print("\n  If the groups that gained are the ones that already had structure")
print("  (the Latinate nominalisation cluster: -ion, -ation, -ition, -fication),")
print("  the loss amplifies existing structure and cannot manufacture it. That is")
print("  a mechanistic result, and it holds whether the global number moved or not.")

TREATED_CONTROL = treated_control_geometry(E_pre, final_emb)

"""## summary JSON"""

report = {
    "model": MODEL_NAME,
    "phase": "P3_morpheme_strong_lambdas",
    "adapter_source": str(ADAPTER_SOURCE),
    "wake_row_source": str(P1_SENTRY),
    "checkpoint": "final",
    "vocab_size": int(vocab_size),
    "dim": int(dim),
    "base_vocab": int(BASE_VOCAB),
    "new_tokens": int(num_new),
    "hyperparameters": {
        "lr": LR,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "max_steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_targets": LORA_TARGETS,
        "seq_len": SEQ_LEN,
        "lambda_morph": LAMBDA_MORPH,
        "lambda_device": LAMBDA_DEVICE,
        "device_triplets": DEVICE_TRIPLETS,
        "device_margin": DEVICE_MARGIN,
        "lambda_repulsion": LAMBDA_REPULSION,
        "lambda_norm": LAMBDA_NORM,
        "repulsion_threshold": REPULSION_THRESHOLD,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "min_group_size": MIN_GROUP_SIZE,
    },
    "null_breaker": {
        "isotropy_delta": iso_delta_summary,
        "direction_consistency_delta": dir_delta_summary,
        "permutation_pre": PRE_NULL,
        "permutation_post": POST_NULL,
        "excess_delta": POST_NULL["excess_over_chance"] - PRE_NULL["excess_over_chance"],
        "treated_control": TREATED_CONTROL,
    },
    "morpheme_stats": {
        "source": "wake_embedding_groups.jsonl",
        "num_groups": morph_index.n_groups,
        "total_pairs": morph_index.total_pairs,
        "wake_tokens_with_morphemes": wake_tokens_with_morphemes,
    },
    "device_stats": {
        "source": "device_groups.jsonl",
        "num_groups": device_index.n_groups,
        "total_words": device_index.total_words,
        "groups": {dg["device"]: dg["n_valid"] for dg in device_groups},
    },
    "norms": {
        "global": {"mean": float(norms.mean()), "std": float(norms.std()),
                    "min": float(norms.min()), "max": float(norms.max())},
        "base": {"mean": float(base_norms.mean()), "std": float(base_norms.std())},
        "new": {"mean": float(new_norms.mean()), "std": float(new_norms.std())},
        "welch_t": {"t": float(t_stat), "p": float(t_pval)},
        "mann_whitney_u": {"U": float(u_stat), "p": float(u_pval)},
        "cohens_d": float(cohens_d),
    },
    "isotropy": {
        "all": {"score": float(iso_all), "mean_cos": float(cos_all), "n": int(n_all)},
        "base": {"score": float(iso_base), "mean_cos": float(cos_base), "n": int(n_base)},
        "new": {"score": float(iso_new), "mean_cos": float(cos_new), "n": int(n_new)},
    },
    "pairwise_cosine": {
        "base_base": {"mean": float(cos_bb.mean()), "std": float(cos_bb.std())},
        "new_new": {"mean": float(cos_nn.mean()), "std": float(cos_nn.std())},
        "base_new": {"mean": float(cos_bn.mean()), "std": float(cos_bn.std())},
        "ks_test_bb_vs_nn": {"D": float(ks_stat), "p": float(ks_pval)},
    },
    "intrinsic_dim": {
        "base_90pct": int(dim90_base), "base_95pct": int(dim95_base),
        "new_90pct": int(dim90_new), "new_95pct": int(dim95_new),
        "base_top1_var": float(pca_base.explained_variance_ratio_[0]),
        "new_top1_var": float(pca_new.explained_variance_ratio_[0]),
    },
    "morpheme_direction_analysis": {
        "top_5_consistent": [
            {"morpheme": r["morpheme"], "mean_cosine": r["mean_cosine"],
             "n_pairs": r["n_pairs"]}
            for r in group_results[:5]
        ],
        "bottom_5_consistent": [
            {"morpheme": r["morpheme"], "mean_cosine": r["mean_cosine"],
             "n_pairs": r["n_pairs"]}
            for r in group_results[-5:]
        ],
    },
}

if has_pre:
    report["drift"] = {
        "base_cosine_mean": float(drift_cos.mean()),
        "base_cosine_std": float(drift_cos.std()),
        "base_l2_mean": float(drift_l2.mean()),
        "wake_cosine_mean": float(wake_drift_cos.mean()),
        "wake_cosine_std": float(wake_drift_cos.std()),
        "wake_l2_mean": float(wake_drift_l2.mean()),
    }

if train_data:
    report["loss"] = {
        "final_train": float(losses[-1]),
        "final_eval": float(v_losses[-1]) if val_data else None,
        "best_eval": float(min(v_losses)) if val_data else None,
    }

summary_path = RUN_DIR / "p3_qwen14b_summary.json"
summary_path.write_text(json.dumps(report, indent=2))
print(f"\n[SUMMARY] Saved to {summary_path}")
print(json.dumps(report, indent=2))

"""## generation test"""

model.eval()
model.config.use_cache = True


def generate_wake(
    prompt,
    max_new_tokens=256,
    temperature=0.9,
    top_p=0.92,
    top_k=50,
    repetition_penalty=1.15,
    num_return_sequences=1,
):
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tok.pad_token_id,
        )

    print(f"-- temp={temperature} | top_p={top_p} | top_k={top_k} | rep={repetition_penalty} --")
    for i, seq in enumerate(outputs):
        generated = tok.decode(seq[prompt_len:], skip_special_tokens=True)
        if num_return_sequences > 1:
            print(f"\n[{i+1}]")
        print(generated)
    print("-" * 60)


def temperature_sweep(prompt, temps=[0.5, 0.7, 0.9, 1.0, 1.2], **kwargs):
    print(f"PROMPT: {prompt}\n")
    for t in temps:
        generate_wake(prompt, temperature=t, **kwargs)
        print()


# wake gen 
generate_wake("riverrun, past Eve and Adam's,")
# generate_wake("riverrun, past Eve and Adam's,", temperature=1.1)
# generate_wake("riverrun, past Eve and Adam's,", num_return_sequences=3, temperature=0.9)
# temperature_sweep("riverrun, past Eve and Adam's,")
