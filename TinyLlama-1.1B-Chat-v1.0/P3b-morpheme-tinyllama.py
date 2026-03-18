# -*- coding: utf-8 -*-
"""Wake2Vec Phase 3b: Geometric Refinement (TinyLlama-1.1B)

# Wake2Vec Phase 3b: Geometric Refinement

**Model:** TinyLlama-1.1B-Chat-v1.0 (4-bit quantized)
**Hardware:** Google Colab T4 GPU
**Source:** P3 step 400 checkpoint (best val 3.4188)

- Lambdas cranked: morph 0.1→50, device 0.05→2.0
  - L_morph is protective (raw value ~0.0002, so 50*0.0002 = 0.01)
  - L_device is offensive (raw value ~0.20, so 2.0*0.20 = 0.40)
- LR halved: 5e-5 → 2e-5 (LM already trained)
- Short run: 1000 steps max (refinement, not retraining)
- Fixed: eval logging spam (guard with model.training)

## Loss

  L_total = L_lm + 50.0 * L_morpheme + 2.0 * L_device
            + 0.05 * L_repulsion + 0.01 * L_norm

─────────────────────────────────────────────────────────

## envi setup
"""

import os, sys, types
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# !pip install --no-cache-dir \
#     torch==2.9.0+cu126 torchvision==0.24.0+cu126 torchaudio==2.9.0+cu126 \
#     --index-url https://download.pytorch.org/whl/cu126

# !pip install -q --no-cache-dir \
#     bitsandbytes==0.45.0 \
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

# P3b source: best checkpoint from P3 run 
P3_SOURCE = Path("/content/drive/MyDrive/wake2vec_tiny_p3_morpheme_v2/full_checkpoints/step_0400")

# P3b output paths 
RUN_DIR = Path("/content/drive/MyDrive/wake2vec_tiny_p3b_refinement")
LOCAL_RUN = Path("/content/runs/wake2vec_tiny_p3b_refinement")
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

# Training hyperparameters (P3b: short refinement) 
MAX_STEPS = 1000
LR = 2e-5                  # halved from P3 (5e-5) — sculpting, not learning
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
GRAD_ACCUM = 2
SEQ_LEN = 256
SAVE_STEPS = 100           
LOG_STEPS = 25              
EVAL_STEPS = 100           

# LoRA (loaded from P3, stays trainable) 
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

# Base vocab (TinyLlama) 
BASE_VOCAB = 32000

# Loss weights (P3b: auxiliary losses now compete)
LAMBDA_MORPH = 50.0         
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
EARLY_STOP_PATIENCE = 3     # tighter for short run

# Resume 
RESUME_FROM = None

print(f"  P3 source: {P3_SOURCE}")
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

"""## load P3 State"""

from transformers import AutoTokenizer

# Verify P3 artifacts
assert P3_SOURCE.exists(), f"P3 source not found: {P3_SOURCE}"
assert (P3_SOURCE / "embeddings.pt").exists(), "P3 embeddings.pt not found"

# Load tokenizer (already has Wake vocab from P1)
tok = AutoTokenizer.from_pretrained(str(P3_SOURCE), use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

TOTAL_VOCAB = len(tok)
NUM_WAKE = TOTAL_VOCAB - BASE_VOCAB

print(f"  Vocab size: {TOTAL_VOCAB}")
print(f"  Base vocab: {BASE_VOCAB}")
print(f"  Wake tokens: {NUM_WAKE}")

# Load P3 trained embeddings (step 400 has best val)
embed_weights = torch.load(P3_SOURCE / "embeddings.pt", map_location="cpu")
print(f"  Shape: {embed_weights.shape}")

"""## Morpheme Group Builder"""

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


# build morpheme groups with resolved token IDs
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

print("Loading base model (4-bit)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "13GB", "cpu": "30GB"}
)

model.config.use_cache = False
model.config.attn_implementation = "eager"

print(f"Resizing embeddings: {BASE_VOCAB} -> {TOTAL_VOCAB}...")
model.resize_token_embeddings(TOTAL_VOCAB, mean_resizing=False)

# Load P3 step 400 embeddings
wte = model.get_input_embeddings()
with torch.no_grad():
    wte.weight.copy_(embed_weights.to(wte.weight.device))

if hasattr(model, "lm_head"):
    model.lm_head.weight = wte.weight
model.config.tie_word_embeddings = True
if hasattr(model, "tie_weights"):
    model.tie_weights()

# Load P3 LoRA adapters
model = PeftModel.from_pretrained(model, str(P3_SOURCE), is_trainable=True)
model.print_trainable_parameters()

# unfreeze Wake embedding rows via gradient masking 

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

"""## Pre-Training embed snapshot"""

E_pre = model.get_input_embeddings().weight.detach().cpu().clone()
torch.save(E_pre, RUN_DIR / "embeddings_pre_p3b.pt")
print(f"  Saved: {RUN_DIR / 'embeddings_pre_p3b.pt'}")
print(f"  Shape: {E_pre.shape}")

base_norms_pre = torch.norm(E_pre[:BASE_VOCAB], dim=1)
TARGET_NORM = base_norms_pre.mean().item()
NORM_MARGIN = base_norms_pre.std().item()
print(f"  Base vocab norm: mean={TARGET_NORM:.4f}, std={NORM_MARGIN:.4f}")

"""## loss components (vectorized)"""

import torch.nn.functional as F


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

        group_sums = torch.zeros(self.n_groups, dim, device=self.device)
        group_sums.scatter_add_(0, expand_groups, directions)
        group_means = group_sums / self.group_sizes.unsqueeze(1)

        pair_means = group_means[self.pair_groups]
        deviations = ((directions - pair_means) ** 2).mean(dim=1)

        return deviations.mean()


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


print(f"  L_morpheme: compositional direction consistency ({morph_index.n_groups} groups, {morph_index.total_pairs} pairs)")
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

        # only log during training (fixes eval spam from P3)
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
        if state.global_step > 0 and state.global_step % 25 == 0:
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
    save_total_limit=5,
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
print(f"  LR: {LR} (halved from P3)")
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

print(f"Final model saved to {final_dir}")

""" Loss Curves & eval """
# loss curve
import matplotlib.pyplot as plt

# train/val curve from Trainer state

state_files = list(LOCAL_RUN.rglob("trainer_state.json"))
train_data, val_data = [], []
if state_files:
    latest = max(state_files, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        state = json.load(f)

    logs = state.get("log_history", [])
    train_data = [(d["step"], d["loss"]) for d in logs if "loss" in d and "eval_loss" not in d]
    val_data = [(d["step"], d["eval_loss"]) for d in logs if "eval_loss" in d]


# component breakdown via loss log

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Wake2Vec P3b Geometric Refinement TinyLlama -- Loss Curves", fontsize=14, fontweight="bold")

# a) train vs val
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

# b) LM loss component
ax = axes[0, 1]
if loss_log:
    ll_steps = [d["step"] for d in loss_log if d["step"] > 0]
    ll_lm = [d["lm"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_lm, 'b-o', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('LM Loss Component')
ax.grid(True, alpha=0.3)

# c) morpheme loss component
ax = axes[0, 2]
if loss_log:
    ll_morph = [d["morph"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_morph, 'g-o', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')

# d) device loss component
ax = axes[1, 0]
if loss_log:
    ll_device = [d["device"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_device, 'c-o', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('L_device (triplet contrastive)')
ax.grid(True, alpha=0.3)

# e) repulsion loss component
ax = axes[1, 1]
if loss_log:
    ll_repul = [d["repulsion"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_repul, 'm-s', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('L_repulsion')
ax.grid(True, alpha=0.3)
ax.set_title('L_morpheme (direction consistency)')
ax.grid(True, alpha=0.3)

# f) norm loss component
ax = axes[1, 2]
if loss_log:
    ll_norm = [d["norm"] for d in loss_log if d["step"] > 0]
    ax.plot(ll_steps, ll_norm, 'orange', marker='o', markersize=3)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('L_norm')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RUN_DIR / "p3b_loss_curves.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plot saved: {plot_path}")
plt.show()

if train_data:
    print(f"\nFinal train loss: {losses[-1]:.4f}")
if val_data:
    print(f"Final val loss: {v_losses[-1]:.4f}")
    print(f"Best val loss: {min(v_losses):.4f}")
