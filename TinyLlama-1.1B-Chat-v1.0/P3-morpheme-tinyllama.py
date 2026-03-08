# -*- coding: utf-8 -*-
"""Wake2Vec Phase 3: Morpheme-Compositional Alignment (TinyLlama-1.1B) 

**Model:** TinyLlama-1.1B-Chat-v1.0 (4-bit quantized)
**Hardware:** Google Colab T4 GPU
**P2 Source:** wake2vec_tiny_p2_lora/full_checkpoints/step_1400/

## Loss

  L_total = L_lm + λ_morph * L_morpheme + λ_device * L_device
            + λ_repulsion * L_repulsion + λ_norm * L_norm

  L_morpheme: for each morpheme group m with word-base pairs {(w_i, b_i)},
    compute direction d_i = embed(w_i) - embed(b_i).
    Loss = mean variance of d_i within each group (encourages consistent
    morpheme direction in embedding space).

  L_device: triplet margin loss over stylistic device groups.
    For sampled triplets (anchor, positive, negative) where anchor and
    positive share a device type and negative doesn't:
    Loss = max(0, margin + cos(anchor, negative) - cos(anchor, positive))

  L_repulsion: penalizes Wake token pairs with cosine sim > threshold.
  L_norm: penalizes Wake token norms that deviate from base vocab norms.

## envi setup
"""

import os, sys, types
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# !pip install --no-cache-dir \
#     torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0+cu126 \
#     --index-url https://download.pytorch.org/whl/cu126

# Uninstall bitsandbytes first to ensure a clean install
# !pip uninstall -y bitsandbytes
# !pip install --no-cache-dir \
#     bitsandbytes==0.46.1 \
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

""" ## Config"""

from pathlib import Path
import json

# P2 source (best val loss checkpoint)
P2_SOURCE = Path("/content/drive/MyDrive/wake2vec_tiny_p2_lora/full_checkpoints/step_1400")

# output 
RUN_DIR = Path("/content/drive/MyDrive/wake2vec_tiny_p3_morpheme_v2")
LOCAL_RUN = Path("/content/runs/wake2vec_tiny_p3_morpheme_v2")
SENTRY = RUN_DIR / "sentry_backups"
EMB_SNAPS = RUN_DIR / "emb_snaps"
FULL_CHECKPOINTS = RUN_DIR / "full_checkpoints"

for d in [RUN_DIR, LOCAL_RUN, SENTRY, EMB_SNAPS, FULL_CHECKPOINTS]:
    d.mkdir(parents=True, exist_ok=True)

# Model 
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FW_TEXT = "/content/FW_TEXT.txt"
MORPHEME_JSONL = "/content/wake_embedding_groups.jsonl"
DEVICE_JSONL = "/content/device_groups.jsonl"

# training hyperparams
MAX_STEPS = 3000
LR = 5e-5
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
GRAD_ACCUM = 2
SEQ_LEN = 256
SAVE_STEPS = 200
LOG_STEPS = 50
EVAL_STEPS = 200

# LoRA 
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

BASE_VOCAB = 32000

# loss weights 
LAMBDA_MORPH = 0.1
LAMBDA_DEVICE = 0.05
LAMBDA_REPULSION = 0.05
LAMBDA_NORM = 0.01
REPULSION_THRESHOLD = 0.95
REPULSION_PAIRS = 100

# Morpheme loss config
MIN_GROUP_SIZE = 3   # skip morpheme groups with fewer valid pairs

# device loss config
DEVICE_TRIPLETS = 64     # triplets sampled per step
DEVICE_MARGIN = 0.2      # triplet margin

# early stop on this one
EARLY_STOP_PATIENCE = 5

# resume
RESUME_FROM = None
# RESUME_FROM = SENTRY / "checkpoint-200"

print(f"  P2 source: {P2_SOURCE}")
print(f"  Output: {RUN_DIR}")
print(f"  Steps: {MAX_STEPS}")
print(f"  LR: {LR}")
print(f"  Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  LoRA rank: {LORA_RANK}, targets: {LORA_TARGETS}")
print(f"  Loss weights: morph={LAMBDA_MORPH}, device={LAMBDA_DEVICE}, repulsion={LAMBDA_REPULSION}, norm={LAMBDA_NORM}")
print(f"  Morpheme data: {MORPHEME_JSONL}")
print(f"  Device data: {DEVICE_JSONL}")
print(f"  Min group size: {MIN_GROUP_SIZE}")
print(f"  Early stopping: patience {EARLY_STOP_PATIENCE}")

"""## get P2 State"""

from transformers import AutoTokenizer

# Verify P2 artifacts
assert P2_SOURCE.exists(), f"P2 source not found: {P2_SOURCE}"
assert (P2_SOURCE / "embeddings.pt").exists(), "P2 embeddings.pt not found"

# Load tokenizer 
tok = AutoTokenizer.from_pretrained(str(P2_SOURCE), use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

TOTAL_VOCAB = len(tok)
NUM_WAKE = TOTAL_VOCAB - BASE_VOCAB

print(f"  Vocab size: {TOTAL_VOCAB}")
print(f"  Base vocab: {BASE_VOCAB}")
print(f"  Wake tokens: {NUM_WAKE}")

# get P2 trained embeddings
embed_weights = torch.load(P2_SOURCE / "embeddings.pt", map_location="cpu")
print(f"  Shape: {embed_weights.shape}")

"""## Morpheme Group Builder """

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


# Build morpheme groups with resolved token IDs
morpheme_groups = []
total_pairs = 0
skipped_groups = 0

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

# Show top 10 groups by size
sorted_groups = sorted(morpheme_groups, key=lambda x: -x["n_valid"])
print(f"\n  Top 10 groups:")
for g in sorted_groups[:10]:
    print(f"    {g['morpheme']:>10s} ({g['morpheme_type']:>7s}) — {g['n_valid']} pairs")
    if g['pair_words']:
        sample = g['pair_words'][:3]
        print(f"      e.g. {', '.join(f'{w}→{b}' for w, b in sample)}")

# Also build a flat lookup: token_id -> list of morpheme group indices
token_to_groups = {}
for gi, g in enumerate(morpheme_groups):
    for w_ids in g["word_ids"]:
        for tid in w_ids:
            if tid >= BASE_VOCAB:  # only Wake tokens
                if tid not in token_to_groups:
                    token_to_groups[tid] = []
                token_to_groups[tid].append(gi)

wake_tokens_with_morphemes = len(token_to_groups)
print(f"\n  Wake tokens participating in morpheme groups: {wake_tokens_with_morphemes}")

"""## device Group Builder (stylistic device contrastive loss)"""

with open(DEVICE_JSONL, 'r') as f:
    raw_device_groups = [json.loads(line) for line in f]

# Resolve token IDs for each device group's words
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

# Non-overlapping blocks
blocks = []
for i in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN):
    chunk = ids[i:i + SEQ_LEN]
    if len(chunk) == SEQ_LEN:
        blocks.append(chunk)

print(f"  Total blocks: {len(blocks)}")

# 90/10 split
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

# 4-bit quantization
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
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "13GB", "cpu": "30GB"}
)

model.config.use_cache = False
model.config.attn_implementation = "eager"

# Resize to P1/P2 vocab
print(f"Resizing embeddings: {BASE_VOCAB} -> {TOTAL_VOCAB}...")
model.resize_token_embeddings(TOTAL_VOCAB, mean_resizing=False)

# Load P2 embeddings
wte = model.get_input_embeddings()
with torch.no_grad():
    wte.weight.copy_(embed_weights.to(wte.weight.device))

# Tie input/output embeddings
if hasattr(model, "lm_head"):
    model.lm_head.weight = wte.weight
model.config.tie_word_embeddings = True
if hasattr(model, "tie_weights"):
    model.tie_weights()

# Load P2 LoRA adapters
model = PeftModel.from_pretrained(model, str(P2_SOURCE), is_trainable=True)
model.print_trainable_parameters()

# unfreeze Wake embedding rows via gradient masking

# First freeze everything
for p in model.parameters():
    p.requires_grad = False

# Unfreeze embedding layer
wte = model.get_input_embeddings()
wte.weight.requires_grad = True

# Unfreeze LoRA adapters
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True

# Gradient mask: zero out gradients for base vocab rows
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

E_pre = model.get_input_embeddings().weight.detach().cpu().clone()
torch.save(E_pre, RUN_DIR / "embeddings_pre.pt")
print(f"  Saved: {RUN_DIR / 'embeddings_pre.pt'}")
print(f"  Shape: {E_pre.shape}")

# Compute base vocab norm stats for L_norm
base_norms_pre = torch.norm(E_pre[:BASE_VOCAB], dim=1)
TARGET_NORM = base_norms_pre.mean().item()
NORM_MARGIN = base_norms_pre.std().item()
print(f"  Base vocab norm: mean={TARGET_NORM:.4f}, std={NORM_MARGIN:.4f}")

"""## Morpheme Loss Components"""

import torch.nn.functional as F

# Precompute flat index tensors for fast GPU loss computation.

class MorphemeIndex:

    def __init__(self, groups, tokenizer, device="cuda"):
        self.device = device
        self.group_names = []
        self.group_types = []

        # For each group: list of (word_token_ids, base_token_ids) pairs
        self.groups_data = []
        self.n_groups = 0

        for g in groups:
            word_ids = g["word_ids"]  # list of list of ints
            base_ids = g["base_ids"]

            if len(word_ids) < MIN_GROUP_SIZE:
                continue

            self.groups_data.append({
                "word_ids": word_ids,
                "base_ids": base_ids,
                "n_pairs": len(word_ids),
            })
            self.group_names.append(g["morpheme"])
            self.group_types.append(g["morpheme_type"])
            self.n_groups += 1

        print(f"MorphemeIndex: {self.n_groups} groups on {device}")

    def compute_loss(self, embed_weight):

        if self.n_groups == 0:
            return torch.tensor(0.0, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        total_weight = 0

        for gd in self.groups_data:
            # Compute word embeddings (mean over sub-tokens)
            word_vecs = []
            base_vecs = []
            for w_ids, b_ids in zip(gd["word_ids"], gd["base_ids"]):
                w_t = torch.tensor(w_ids, device=self.device, dtype=torch.long)
                b_t = torch.tensor(b_ids, device=self.device, dtype=torch.long)
                word_vecs.append(embed_weight[w_t].mean(dim=0))
                base_vecs.append(embed_weight[b_t].mean(dim=0))

            word_stack = torch.stack(word_vecs)     # (n_pairs, dim)
            base_stack = torch.stack(base_vecs)     # (n_pairs, dim)

            # Direction vectors: how the morpheme transforms the base
            directions = word_stack - base_stack    # (n_pairs, dim)

            # Group mean direction
            d_mean = directions.mean(dim=0, keepdim=True)  # (1, dim)

            # Variance of directions within group (MSE to mean)
            group_loss = ((directions - d_mean) ** 2).mean()

            n = gd["n_pairs"]
            total_loss = total_loss + group_loss * n
            total_weight += n

        if total_weight == 0:
            return torch.tensor(0.0, device=self.device)

        return total_loss / total_weight


# Build the index
morph_index = MorphemeIndex(morpheme_groups, tok, device="cuda")

# All Wake token indices for repulsion + norm losses
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


# device contrastive loss 
class DeviceIndex:
    """Precomputed index for triplet sampling across device groups."""

    def __init__(self, groups, tokenizer, device="cuda"):
        self.device = device
        self.n_groups = len(groups)
        self.group_names = [g["device"] for g in groups]

        # For each group: list of word token ID lists
        self.groups_word_ids = [g["word_ids"] for g in groups]
        self.group_sizes = [g["n_valid"] for g in groups]

        # Flat array of group labels per word (for negative sampling)
        self.all_word_ids = []    # list of token ID lists
        self.all_group_labels = []  # group index for each word
        for gi, g in enumerate(groups):
            for w_ids in g["word_ids"]:
                self.all_word_ids.append(w_ids)
                self.all_group_labels.append(gi)

        self.all_group_labels_t = torch.tensor(
            self.all_group_labels, dtype=torch.long, device=device
        )
        self.total_words = len(self.all_word_ids)
        print(f"DeviceIndex: {self.n_groups} groups, {self.total_words} words")

    def _embed_word(self, w_ids, embed_weight):
        t = torch.tensor(w_ids, device=self.device, dtype=torch.long)
        return embed_weight[t].mean(dim=0)

    def compute_loss(self, embed_weight, n_triplets=DEVICE_TRIPLETS, margin=DEVICE_MARGIN):
        if self.n_groups < 2 or self.total_words < 3:
            return torch.tensor(0.0, device=self.device)

        # Sample anchor indices
        anchor_idx = torch.randint(0, self.total_words, (n_triplets,), device=self.device)
        anchor_groups = self.all_group_labels_t[anchor_idx]

        # For each anchor, sample a positive (same group) and negative (different group)
        anchor_vecs = []
        pos_vecs = []
        neg_vecs = []
        valid = 0

        for i in range(n_triplets):
            ai = anchor_idx[i].item()
            ag = anchor_groups[i].item()

            # Same-group indices
            same_mask = (self.all_group_labels_t == ag)
            same_indices = same_mask.nonzero(as_tuple=True)[0]
            if len(same_indices) < 2:
                continue

            # Different-group indices
            diff_mask = (self.all_group_labels_t != ag)
            diff_indices = diff_mask.nonzero(as_tuple=True)[0]
            if len(diff_indices) == 0:
                continue

            # Pick random positive (not the anchor itself)
            pi = same_indices[torch.randint(0, len(same_indices), (1,))].item()
            while pi == ai and len(same_indices) > 1:
                pi = same_indices[torch.randint(0, len(same_indices), (1,))].item()

            # Pick random negative
            ni = diff_indices[torch.randint(0, len(diff_indices), (1,))].item()

            anchor_vecs.append(self._embed_word(self.all_word_ids[ai], embed_weight))
            pos_vecs.append(self._embed_word(self.all_word_ids[pi], embed_weight))
            neg_vecs.append(self._embed_word(self.all_word_ids[ni], embed_weight))
            valid += 1

        if valid == 0:
            return torch.tensor(0.0, device=self.device)

        anchors = torch.stack(anchor_vecs)    # (valid, dim)
        positives = torch.stack(pos_vecs)
        negatives = torch.stack(neg_vecs)

        # Cosine similarities
        a_norm = F.normalize(anchors, dim=1)
        p_norm = F.normalize(positives, dim=1)
        n_norm = F.normalize(negatives, dim=1)

        cos_pos = (a_norm * p_norm).sum(dim=1)
        cos_neg = (a_norm * n_norm).sum(dim=1)

        # Triplet margin loss
        loss = torch.clamp(margin + cos_neg - cos_pos, min=0).mean()
        return loss


device_index = DeviceIndex(device_groups, tok, device="cuda")


def compute_device_loss(model):
    E = model.get_input_embeddings().weight
    return device_index.compute_loss(E)


print(f"  L_morpheme: compositional direction consistency ({morph_index.n_groups} groups, {total_pairs} pairs)")
print(f"  L_device: triplet margin loss ({device_index.n_groups} device types, {device_index.total_words} words, margin={DEVICE_MARGIN})")
print(f"  L_repulsion: cosine > {REPULSION_THRESHOLD} penalty ({REPULSION_PAIRS} pairs/step)")
print(f"  L_norm: deviation from base norm mean={TARGET_NORM:.2f} +/- {NORM_MARGIN:.2f}")

"""## trainer"""

from transformers import TrainingArguments, Trainer

class MorphemeTrainer(Trainer):

    def __init__(self, *args, loss_log=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_log = loss_log if loss_log is not None else []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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

        if self.state.global_step % LOG_STEPS == 0:
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
        if state.global_step > 0 and state.global_step % 50 == 0:
            try:
                mdl = kwargs.get("model", model)
                E = mdl.get_input_embeddings().weight.detach().cpu()
                path = EMB_SNAPS / f"emb_step{state.global_step:04d}.pt"
                torch.save(E, path)
                os.sync()
                print(f"[EMB] Step {state.global_step}: saved")
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
                return
            dst = SENTRY / ck.name
            if dst.exists():
                return
            shutil.copytree(ck, dst)
            os.sync()
            print(f"[SENTRY] {ck.name}: mirrored to Drive")
        except Exception as e:
            print(f"[SENTRY] {e}")


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
    def __init__(self, patience=5):
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

# Re-verify trainable params
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True

wte = model.get_input_embeddings()
wte.weight.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}")

if trainable == 0:
    raise RuntimeError("No trainable parameters found")

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
    save_total_limit=10,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    report_to="none",
    dataloader_pin_memory=False,
    gradient_checkpointing=False,
    max_grad_norm=1.0,
    load_best_model_at_end=True,
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
        EmbeddingSnapshot(),
        FullCheckpoint(),
        SentryMirror(),
        LossMonitor(),
        StepTimer(),
        EarlyStoppingCallback(patience=EARLY_STOP_PATIENCE),
    ],
)

print(f"  Train: {len(train_ds)} blocks | Val: {len(val_ds)} blocks")
print(f"  Steps: {MAX_STEPS} (early stop patience={EARLY_STOP_PATIENCE})")
print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  LoRA targets: {LORA_TARGETS}")
print(f"  Trainable: Wake embed rows + LoRA adapters")
print(f"  Loss: LM + {LAMBDA_MORPH}*morph + {LAMBDA_DEVICE}*device + {LAMBDA_REPULSION}*repulsion + {LAMBDA_NORM}*norm")
print(f"  Morpheme groups: {morph_index.n_groups} ({total_pairs} word-base pairs)")
print(f"  Device groups: {device_index.n_groups} ({device_index.total_words} words, {DEVICE_TRIPLETS} triplets/step)")

t0 = time.time()
if RESUME_FROM is not None and RESUME_FROM.exists():
    local_ckpt = LOCAL_RUN / RESUME_FROM.name
    if not local_ckpt.exists():
        shutil.copytree(RESUME_FROM, local_ckpt)
    print(f"[RESUME] Resuming from {RESUME_FROM.name}")
    trainer.train(resume_from_checkpoint=str(local_ckpt))
else:
    trainer.train()
elapsed = (time.time() - t0) / 60

print(f"\nTRAINING COMPLETE ({elapsed:.1f} minutes)")

# Save final model
final_dir = RUN_DIR / "final"
final_dir.mkdir(exist_ok=True)

model.save_pretrained(str(final_dir))
tok.save_pretrained(str(final_dir))

final_emb = model.get_input_embeddings().weight.detach().cpu()
torch.save(final_emb, final_dir / "embeddings.pt")

# Save loss component log
with open(RUN_DIR / "loss_components.json", 'w') as f:
    json.dump(loss_log, f, indent=2)

print(f"Final model saved to {final_dir}")
print(f"Loss component log: {RUN_DIR / 'loss_components.json'}")

"""## Loss Curves (with component breakdown)"""

import matplotlib.pyplot as plt

# standard train/val curve from Trainer state 

state_files = list(LOCAL_RUN.rglob("trainer_state.json"))
train_data, val_data = [], []
if state_files:
    latest = max(state_files, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        state = json.load(f)

    logs = state.get("log_history", [])
    train_data = [(d["step"], d["loss"]) for d in logs if "loss" in d and "eval_loss" not in d]
    val_data = [(d["step"], d["eval_loss"]) for d in logs if "eval_loss" in d]

# component breakdown from our loss log 

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Wake2Vec P3 v2 TinyLlama -- Loss Curves", fontsize=14, fontweight="bold")

# Train vs Val
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

# LM loss component
ax = axes[0, 1]
if loss_log:
    ll_steps = [d["step"] for d in loss_log if d["step"] > 0]
    ll_lm = [d["lm"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_lm, 'b-o', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('LM Loss Component')
ax.grid(True, alpha=0.3)

# Morpheme loss component
ax = axes[0, 2]
if loss_log:
    ll_morph = [d["morph"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_morph, 'g-o', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('L_morpheme (direction consistency)')
ax.grid(True, alpha=0.3)

# Device loss component
ax = axes[1, 0]
if loss_log:
    ll_device = [d["device"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_device, 'c-o', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('L_device (triplet contrastive)')
ax.grid(True, alpha=0.3)

# Repulsion loss component
ax = axes[1, 1]
if loss_log:
    ll_repul = [d["repulsion"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_repul, 'm-s', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('L_repulsion')
ax.grid(True, alpha=0.3)

# Norm loss component
ax = axes[1, 2]
if loss_log:
    ll_norm = [d["norm"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_norm, 'orange', marker='o', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('L_norm')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RUN_DIR / "p3_v2_loss_curves.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plot saved: {plot_path}")
plt.show()

if train_data:
    print(f"\nFinal train loss: {losses[-1]:.4f}")
if val_data:
    print(f"Final val loss: {v_losses[-1]:.4f}")
    print(f"Best val loss: {min(v_losses):.4f}")

"""## embed analysis"""

import numpy as np
from numpy.linalg import norm as l2
from scipy import stats
from sklearn.decomposition import PCA

E_post = final_emb.float().numpy()
vocab_size, dim = E_post.shape
num_new = vocab_size - BASE_VOCAB

E_base = E_post[:BASE_VOCAB]
E_new = E_post[BASE_VOCAB:]

# Load pre-training embeddings for drift analysis
pre_path = RUN_DIR / "embeddings_pre.pt"
has_pre = pre_path.exists()
if has_pre:
    E_pre_all = torch.load(pre_path, map_location="cpu").float().numpy()
    E_pre_base = E_pre_all[:BASE_VOCAB]
    E_pre_new = E_pre_all[BASE_VOCAB:]
    print(f"[PRE] Loaded pre-training embeddings: {E_pre_all.shape}")
else:
    print("[PRE] No pre-training snapshot found (skipping drift analysis)")

# norm analysis w/ statistical tests 

norms = l2(E_post, axis=1)
base_norms = norms[:BASE_VOCAB]
new_norms = norms[BASE_VOCAB:]

t_stat, t_pval = stats.ttest_ind(base_norms, new_norms, equal_var=False)
u_stat, u_pval = stats.mannwhitneyu(base_norms, new_norms, alternative='two-sided')
pooled_std = np.sqrt((base_norms.std()**2 + new_norms.std()**2) / 2)
cohens_d = (base_norms.mean() - new_norms.mean()) / pooled_std

print(f"  Global  -- mean: {norms.mean():.4f}, std: {norms.std():.4f}, "
      f"min: {norms.min():.4f}, max: {norms.max():.4f}")
print(f"  Base    -- mean: {base_norms.mean():.4f}, std: {base_norms.std():.4f} (n={BASE_VOCAB})")
print(f"  New     -- mean: {new_norms.mean():.4f}, std: {new_norms.std():.4f} (n={num_new})")
print(f"  Welch t-test:    t={t_stat:.4f}, p={t_pval:.2e}")
print(f"  Mann-Whitney U:  U={u_stat:.0f}, p={u_pval:.2e}")
print(f"  Cohen's d:       {cohens_d:.4f}")

# eigenvalue-based isotropy (Mu et al. 2018)

def compute_isotropy(embeddings, label=""):
    centered = embeddings - embeddings.mean(axis=0)
    nrm = l2(centered, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-12, 1e-12, nrm)
    unit = centered / nrm

    rng = np.random.default_rng(42)
    n = min(5000, len(unit))
    idx = rng.choice(len(unit), size=n, replace=False)
    sample = unit[idx]

    cos_mat = sample @ sample.T
    np.fill_diagonal(cos_mat, 0)
    Z = np.exp(cos_mat).sum(axis=1)

    isotropy = Z.min() / Z.max()
    mean_cos = (cos_mat.sum()) / (n * (n - 1))
    return isotropy, mean_cos, n

iso_all, cos_all, n_all = compute_isotropy(E_post)
iso_base, cos_base, n_base = compute_isotropy(E_base)
iso_new, cos_new, n_new = compute_isotropy(E_new)

print(f"  All tokens isotropy: {iso_all:.6f}, mean_cos: {cos_all:.4f} (n={n_all})")
print(f"  Base tokens isotropy: {iso_base:.6f}, mean_cos: {cos_base:.4f} (n={n_base})")
print(f"  New tokens isotropy: {iso_new:.6f}, mean_cos: {cos_new:.4f} (n={n_new})")

# embed drift (pre vs post) 

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

all_norms_safe = l2(E_post, axis=1, keepdims=True)
all_norms_safe = np.where(all_norms_safe < 1e-12, 1e-12, all_norms_safe)
E_unit = E_post / all_norms_safe

sample_wake_ids = list(range(BASE_VOCAB, BASE_VOCAB + 10))
if num_new > 100:
    mid = BASE_VOCAB + num_new // 2
    sample_wake_ids += list(range(mid, mid + 5))
if num_new > 1000:
    sample_wake_ids += list(range(BASE_VOCAB + num_new - 5, BASE_VOCAB + num_new))

E_base_unit = E_unit[:BASE_VOCAB]

for wake_id in sample_wake_ids:
    wake_token = tok.convert_ids_to_tokens(wake_id)
    wake_vec = E_unit[wake_id:wake_id+1]
    sims = (wake_vec @ E_base_unit.T).squeeze()
    top5 = np.argsort(sims)[-5:][::-1]
    neighbors = [(tok.convert_ids_to_tokens(int(i)), sims[i]) for i in top5]
    nb_str = ", ".join(f"{t!r}({s:.3f})" for t, s in neighbors)
    print(f"  {wake_token!r:25s} -> {nb_str}")

# intrinsic dimensionality (pca)

n_components = min(100, dim, BASE_VOCAB, num_new)

pca_base = PCA(n_components=n_components).fit(E_base)
pca_new = PCA(n_components=n_components).fit(E_new)

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

# pairwise cosine similarity distributions

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
fig.suptitle("Wake2Vec P3 v2 TinyLlama Morpheme -- Embedding Analysis", fontsize=14, fontweight="bold")

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
    ax.set_title("Embedding Drift")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No pre-training\nsnapshot available",
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
analysis_plot = RUN_DIR / "p3_v2_TinyLlama_analysis.png"
plt.savefig(analysis_plot, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {analysis_plot}")
plt.show()

"""## Morpheme-Specific Analysis"""

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

    # Mean direction (the group's "morpheme vector")
    d_mean = directions.mean(dim=0)

    # Cosine similarity of each direction to the mean direction
    d_norm = F.normalize(directions, dim=1)
    mean_norm = F.normalize(d_mean.unsqueeze(0), dim=1)
    cosines_to_mean = (d_norm @ mean_norm.T).squeeze()

    # Direction magnitudes
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


# Analyze all groups
group_results = []
for g in morpheme_groups:
    result = analyze_morpheme_group(g, E_final, tok)
    if result is not None:
        result["morpheme"] = g["morpheme"]
        result["morpheme_type"] = g["morpheme_type"]
        group_results.append(result)

# Sort by direction consistency 
group_results.sort(key=lambda x: -x["mean_cosine"])

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

# Detailed view for a few interesting groups
interesting = [r for r in group_results if r["morpheme"] in ["-ing", "-ed", "-s", "un-", "re-", "-ly"]]
for r in interesting:
    print(f"\n  === {r['morpheme']} ({r['morpheme_type']}, {r['n_pairs']} pairs) ===")
    print(f"    Direction consistency: {r['mean_cosine']:.4f} +/- {r['std_cosine']:.4f}")
    print(f"    Direction magnitude: {r['mean_magnitude']:.4f} +/- {r['std_magnitude']:.4f}")

    # Show most/least aligned pairs
    sorted_idx = r['cosines_to_mean'].argsort(descending=True)
    print(f"    Most aligned:")
    for i in sorted_idx[:3]:
        w, b = r['pair_labels'][i]
        print(f"      {w} -> {b}  (cos={r['cosines_to_mean'][i].item():.4f})")
    print(f"    Least aligned:")
    for i in sorted_idx[-3:]:
        w, b = r['pair_labels'][i]
        print(f"      {w} -> {b}  (cos={r['cosines_to_mean'][i].item():.4f})")

# direction similarity between morpheme groups, do different morphemes encode different directions?

if len(group_results) >= 5:
    print(f"\n  Cross-group direction similarity (top 20 groups):")
    top_groups = group_results[:20]
    dir_vecs = torch.stack([r["mean_direction"] for r in top_groups])
    dir_normed = F.normalize(dir_vecs, dim=1)
    cross_sim = (dir_normed @ dir_normed.T)

    # Show most similar pairs
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
# For each device group, compute intra-group and inter-group cosine sims
device_centroids = {}
for dg in device_groups:
    vecs = []
    for w_ids in dg["word_ids"]:
        t = torch.tensor(w_ids, dtype=torch.long)
        vecs.append(E_final[t].mean(dim=0))
    if vecs:
        stack = torch.stack(vecs)
        device_centroids[dg["device"]] = stack.mean(dim=0)

        # Intra-group pairwise cosine
        normed = F.normalize(stack, dim=1)
        n = len(vecs)
        if n > 1:
            cos_mat = (normed @ normed.T)
            # Upper triangle mean (excluding diagonal)
            mask_ut = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            intra_cos = cos_mat[mask_ut].mean().item()
        else:
            intra_cos = float('nan')

        print(f"\n  {dg['device']:>15s} ({dg['n_valid']} words)")
        print(f"    Intra-group mean cosine: {intra_cos:.4f}")

# Inter-group centroid similarity
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

"""## summary JSON"""

report = {
    "model": MODEL_NAME,
    "phase": "P3_morpheme_v2",
    "p2_source": str(P2_SOURCE),
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
    "morpheme_stats": {
        "source": "wake_embedding_groups.jsonl",
        "num_groups": len(morpheme_groups),
        "total_pairs": total_pairs,
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

summary_path = RUN_DIR / "p3_v2_TinyLlama_summary.json"
summary_path.write_text(json.dumps(report, indent=2))
print(f"\n[SUMMARY] Saved to {summary_path}")
print(json.dumps(report, indent=2))

"""## wake generation test"""

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


# wake style samples 
generate_wake("riverrun, past Eve and Adam's,")
# generate_wake("riverrun, past Eve and Adam's,", temperature=1.1)
# generate_wake("riverrun, past Eve and Adam's,", num_return_sequences=3, temperature=0.9)
# temperature_sweep("riverrun, past Eve and Adam's,")
