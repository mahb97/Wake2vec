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
# envi p2

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

""" config """
# config

from pathlib import Path

# p1 source (Llama-3.2-1B, embedding-only, best val @ step 1400) 
P1_SOURCE = Path("/content/drive/MyDrive/wake2vec_llama_p1/full_checkpoints/step_1400")

# p2 output paths 
RUN_DIR = Path("/content/drive/MyDrive/wake2vec_llama_p2_lora")
LOCAL_RUN = Path("/content/runs/wake2vec_llama_p2_lora")
SENTRY = RUN_DIR / "sentry_backups"
EMB_SNAPS = RUN_DIR / "emb_snaps"
FULL_CHECKPOINTS = RUN_DIR / "full_checkpoints"

for d in [RUN_DIR, LOCAL_RUN, SENTRY, EMB_SNAPS, FULL_CHECKPOINTS]:
    d.mkdir(parents=True, exist_ok=True)

# model 
MODEL_NAME = "meta-llama/Llama-3.2-1B"
FW_TEXT = "/content/FW_TEXT.txt"

# training hyperparams 
MAX_STEPS = 3000
LR = 2e-5
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
BATCH_SIZE = 4
GRAD_ACCUM = 4
SEQ_LEN = 512
SAVE_STEPS = 100
LOG_STEPS = 50
EVAL_STEPS = 100
EMB_SNAP_STEPS = 50

# uncomment as needed 
# RESUME_FROM SENTRY / "checkpoint - 200"

# call my daughter LoRA
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

# base vocab (Llama-3.2)
BASE_VOCAB = 128256

print("WAKE2VEC PHASE 2: Llama-3.2-1B LoRA CONFIG")
print(f"  P1 source: {P1_SOURCE}")
print(f"  Output: {RUN_DIR}")
print(f"  Steps: {MAX_STEPS}")
print(f"  LR: {LR}")
print(f"  Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  LoRA rank: {LORA_RANK}, targets: {LORA_TARGETS}")
print(f"  Embeddings: FROZEN (loaded from P1)")

""" load p1 state """
# get p1 embeds
from transformers import AutoTokenizer

# verify P1 artifacts
assert P1_SOURCE.exists(), f"P1 source not found: {P1_SOURCE}"
assert (P1_SOURCE / "embeddings.pt").exists(), "P1 embeddings.pt not found"

# load tokenizer (has Wake vocab from P1)
print("Loading P1 tokenizer...")
tok = AutoTokenizer.from_pretrained(str(P1_SOURCE), use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

TOTAL_VOCAB = len(tok)
NUM_WAKE = TOTAL_VOCAB - BASE_VOCAB

print(f"  Vocab size: {TOTAL_VOCAB}")
print(f"  Base vocab: {BASE_VOCAB}")
print(f"  Wake tokens: {NUM_WAKE}")

# load P1 trained embeddings
print("Loading P1 embeddings...")
embed_weights = torch.load(P1_SOURCE / "embeddings.pt", map_location="cpu")
print(f"  Shape: {embed_weights.shape}")

""" Wake datasets (full FW and lexicon) """
# datasets 
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class BlockDataset(Dataset):
    def __init__(self, blocks, seq_len=512):
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

print("Loading Finnegans Wake...")
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

""" Llama model set - up """
# model set up
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, get_peft_model

set_seed(42)

# 4-bit quantization
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

# resize to P1 vocab
print(f"Resizing embeddings: {BASE_VOCAB} -> {TOTAL_VOCAB}...")
model.resize_token_embeddings(TOTAL_VOCAB, mean_resizing=False)

# load P1 embeddings
wte = model.get_input_embeddings()
with torch.no_grad():
    wte.weight.copy_(embed_weights.to(wte.weight.device))

# tie input/output embeddings
if hasattr(model, "lm_head"):
    model.lm_head.weight = wte.weight
model.config.tie_word_embeddings = True
if hasattr(model, "tie_weights"):
    model.tie_weights()

print("P1 embeddings loaded and tied")

# freeze everything
for p in model.parameters():
    p.requires_grad = False

print("All parameters frozen")

# add LoRA adapters
peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# verify embeddings are frozen
wte = model.get_input_embeddings()
print(f"Embedding requires_grad: {wte.weight.requires_grad}")
assert not wte.weight.requires_grad, "Embeddings should be frozen in P2"

""" embed snapshot """
# pre-train snapshot
print("Saving pre-training embedding snapshot...")
E_pre = model.get_input_embeddings().weight.detach().cpu().clone()
torch.save(E_pre, RUN_DIR / "embeddings_pre.pt")
print(f"  Saved: {RUN_DIR / 'embeddings_pre.pt'}")
print(f"  Shape: {E_pre.shape}")

""" sentry """
# callbacks 
import shutil
import time

def has_weights(ck):
    return (ck / "adapter_model.safetensors").exists() or (ck / "pytorch_model.bin").exists()

from transformers import TrainerCallback


class EmbeddingSnapshot(TrainerCallback):
    """Save embedding weights every EMB_SNAP_STEPS steps for geometric analysis."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % EMB_SNAP_STEPS == 0:
            try:
                mdl = kwargs.get("model", model)
                E = mdl.get_input_embeddings().weight.detach().cpu()
                path = EMB_SNAPS / f"emb_step{state.global_step:04d}.pt"
                torch.save(E, path)
                print(f"[EMB] Step {state.global_step}: saved")
            except Exception as e:
                print(f"[EMB] Step {state.global_step}: {e}")


class FullCheckpoint(TrainerCallback):
    """Save complete checkpoint every SAVE_STEPS to Drive."""
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

            print(f"[FULL] Step {step}: saved to Drive")
        except Exception as e:
            print(f"[FULL] Step {state.global_step}: {e}")


class SentryMirror(TrainerCallback):
    """Mirror Trainer checkpoints to Drive."""
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
            print(f"[SENTRY] {ck.name}: mirrored to Drive")
        except Exception as e:
            print(f"[SENTRY] {e}")


class LossMonitor(TrainerCallback):
    """Track and warn about train/eval divergence."""
    def __init__(self):
        self.last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.last_train_loss = logs["loss"]
        if "eval_loss" in logs and self.last_train_loss is not None:
            gap = logs["eval_loss"] - self.last_train_loss
            if gap > 3.0:
                print(f"[WARN] Large train/eval gap: {gap:.2f}")


class StepTimer(TrainerCallback):
    """Track throughput with rolling average."""
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


print("Callbacks defined:")
print(f"  EmbeddingSnapshot (every {EMB_SNAP_STEPS} steps)")
print(f"  FullCheckpoint (every {SAVE_STEPS} steps)")
print("  SentryMirror (mirror to Drive)")
print("  LossMonitor (divergence warning)")
print("  StepTimer (throughput tracking)")

""" trainer & train """
# training args, train and save final 
from transformers import TrainingArguments, Trainer

# Re-verify LoRA gradients
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True

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
    warmup_ratio=WARMUP_RATIO,
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
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    callbacks=[
        EmbeddingSnapshot(),
        FullCheckpoint(),
        SentryMirror(),
        LossMonitor(),
        StepTimer(),
    ],
)

print("WAKE2VEC PHASE 2: Llama-3.2-1B LoRA FINE-TUNE")
print(f"  Train: {len(train_ds)} blocks | Val: {len(val_ds)} blocks")
print(f"  Steps: {MAX_STEPS}")
print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  LoRA targets: {LORA_TARGETS}")
print(f"  Embeddings: FROZEN")

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

""" loss curve """
# loss curve png
import matplotlib.pyplot as plt
import json

state_files = list(LOCAL_RUN.rglob("trainer_state.json"))
if state_files:
    latest = max(state_files, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        state = json.load(f)

    logs = state.get("log_history", [])
    train_data = [(d["step"], d["loss"]) for d in logs if "loss" in d and "eval_loss" not in d]
    val_data = [(d["step"], d["eval_loss"]) for d in logs if "eval_loss" in d]

    if train_data:
        plt.figure(figsize=(12, 6))
        steps, losses = zip(*train_data)
        plt.plot(steps, losses, 'b-o', label='Train', alpha=0.7, markersize=4)

        if val_data:
            v_steps, v_losses = zip(*val_data)
            plt.plot(v_steps, v_losses, 'r-s', label='Val', alpha=0.7, markersize=4)

        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Wake2Vec P2: Llama-3.2-1B LoRA Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = RUN_DIR / "p2_llama_loss_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {plot_path}")
        plt.show()

        print(f"\nFinal train loss: {losses[-1]:.4f}")
        if val_data:
            print(f"Final val loss: {v_losses[-1]:.4f}")
            print(f"Best val loss: {min(v_losses):.4f}")
else:
    print("No trainer_state.json found")

""" full embed analysis """
# embed analysis
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

print("=" * 60)
print("1. NORM ANALYSIS")
print("=" * 60)
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

print(f"\n{'=' * 60}")
print("2. ISOTROPY (Mu et al. 2018 partition function ratio)")
print("=" * 60)
print(f"  All tokens  -- isotropy: {iso_all:.6f}, mean_cos: {cos_all:.4f} (n={n_all})")
print(f"  Base tokens -- isotropy: {iso_base:.6f}, mean_cos: {cos_base:.4f} (n={n_base})")
print(f"  New tokens  -- isotropy: {iso_new:.6f}, mean_cos: {cos_new:.4f} (n={n_new})")

# embed drift (pre vs post) 

if has_pre:
    # Base token drift (should be ~1.0 since embeddings are frozen in P2)
    base_pre_norms = l2(E_pre_base, axis=1, keepdims=True)
    base_post_norms = l2(E_base, axis=1, keepdims=True)
    base_pre_norms = np.where(base_pre_norms < 1e-12, 1e-12, base_pre_norms)
    base_post_norms = np.where(base_post_norms < 1e-12, 1e-12, base_post_norms)

    drift_cos = np.sum(
        (E_pre_base / base_pre_norms) * (E_base / base_post_norms), axis=1
    )
    drift_l2 = l2(E_base - E_pre_base, axis=1)

    # Wake token drift (should also be ~1.0 — embeddings frozen in P2)
    new_pre_norms = l2(E_pre_new, axis=1, keepdims=True)
    new_post_norms = l2(E_new, axis=1, keepdims=True)
    new_pre_norms = np.where(new_pre_norms < 1e-12, 1e-12, new_pre_norms)
    new_post_norms = np.where(new_post_norms < 1e-12, 1e-12, new_post_norms)

    wake_drift_cos = np.sum(
        (E_pre_new / new_pre_norms) * (E_new / new_post_norms), axis=1
    )
    wake_drift_l2 = l2(E_new - E_pre_new, axis=1)

    print(f"\n{'=' * 60}")
    print("3. EMBEDDING DRIFT (pre -> post)")
    print("=" * 60)
    print(f"  Base tokens:")
    print(f"    Cosine sim -- mean: {drift_cos.mean():.6f}, std: {drift_cos.std():.6f}")
    print(f"    L2 dist    -- mean: {drift_l2.mean():.6f}, std: {drift_l2.std():.6f}")
    print(f"  Wake tokens:")
    print(f"    Cosine sim -- mean: {wake_drift_cos.mean():.6f}, std: {wake_drift_cos.std():.6f}")
    print(f"    L2 dist    -- mean: {wake_drift_l2.mean():.6f}, std: {wake_drift_l2.std():.6f}")

    if drift_cos.mean() > 0.999:
        print("  >> Base tokens barely moved (expected: embeddings frozen)")
    elif drift_cos.mean() > 0.99:
        print("  >> Moderate base drift detected")
    else:
        print("  >> Significant base drift detected (unexpected if frozen)")

    # Top 20 most-drifted base tokens
    drift_order = np.argsort(drift_cos)
    print(f"\n  Top 20 most-drifted base tokens:")
    for rank, idx in enumerate(drift_order[:20]):
        token_str = tok.convert_ids_to_tokens(int(idx))
        print(f"    {rank+1:2d}. [{idx:5d}] {token_str!r:20s}  "
              f"cos={drift_cos[idx]:.6f}  L2={drift_l2[idx]:.4f}")

# nearest neighbour sanity checks 
print(f"\n{'=' * 60}")
print("4. NEAREST NEIGHBORS (Wake tokens -> base vocab)")
print("=" * 60)

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

# intrinsic dimensionality pca 
print(f"\n{'=' * 60}")
print("5. INTRINSIC DIMENSIONALITY (PCA explained variance)")
print("=" * 60)

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

print(f"\n{'=' * 60}")
print("6. PAIRWISE COSINE SIMILARITY DISTRIBUTIONS")
print("=" * 60)

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

# plots

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Wake2Vec P2 Llama-3.2-1B LoRA -- Embedding Analysis", fontsize=14, fontweight="bold")

# norm histograms
ax = axes[0, 0]
ax.hist(base_norms, bins=50, alpha=0.5, label=f"Base (u={base_norms.mean():.2f})", density=True)
ax.hist(new_norms, bins=50, alpha=0.5, label=f"New (u={new_norms.mean():.2f})", density=True)
ax.set_xlabel("L2 norm")
ax.set_ylabel("Density")
ax.set_title("Norm Distribution: Base vs New")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# cosine similarity distributions
ax = axes[0, 1]
ax.hist(cos_bb, bins=80, alpha=0.4, label=f"base-base (u={cos_bb.mean():.3f})", density=True)
ax.hist(cos_nn, bins=80, alpha=0.4, label=f"new-new (u={cos_nn.mean():.3f})", density=True)
ax.hist(cos_bn, bins=80, alpha=0.4, label=f"base-new (u={cos_bn.mean():.3f})", density=True)
ax.set_xlabel("Cosine similarity")
ax.set_ylabel("Density")
ax.set_title("Pairwise Cosine Distributions")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# PCA explained variance
ax = axes[0, 2]
ax.plot(range(1, n_components+1), cumvar_base, 'b-', label=f"Base (90%@{dim90_base})")
ax.plot(range(1, n_components+1), cumvar_new, 'r-', label=f"New (90%@{dim90_new})")
ax.axhline(0.90, linestyle='--', color='gray', alpha=0.5, label="90% threshold")
ax.set_xlabel("Principal component")
ax.set_ylabel("Cumulative explained variance")
ax.set_title("Intrinsic Dimensionality")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# drift histogram
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

# norm scatter
ax = axes[1, 1]
ax.scatter(range(BASE_VOCAB), base_norms, s=0.1, alpha=0.3, label="Base", c="blue")
ax.scatter(range(BASE_VOCAB, vocab_size), new_norms, s=0.1, alpha=0.3, label="New", c="red")
ax.set_xlabel("Token index")
ax.set_ylabel("L2 norm")
ax.set_title("Norm by Token Index")
ax.legend(fontsize=8, markerscale=10)
ax.grid(True, alpha=0.3)

# top PCA eigenvalue spectrum
ax = axes[1, 2]
ax.bar(range(1, 21), pca_base.explained_variance_ratio_[:20], alpha=0.5, label="Base")
ax.bar(range(1, 21), pca_new.explained_variance_ratio_[:20], alpha=0.5, label="New")
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance ratio")
ax.set_title("Top-20 PC Eigenspectrum")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
analysis_plot = RUN_DIR / "p2_llama_analysis.png"
plt.savefig(analysis_plot, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {analysis_plot}")
plt.show()

# summary json 
report = {
    "model": MODEL_NAME,
    "phase": "P2_LoRA",
    "p1_source": str(P1_SOURCE),
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
    "loss": {
        "final_train": float(losses[-1]) if train_data else None,
        "final_eval": float(v_losses[-1]) if val_data else None,
        "best_eval": float(min(v_losses)) if val_data else None,
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

summary_path = RUN_DIR / "p2_llama_summary.json"
summary_path.write_text(json.dumps(report, indent=2))
print(f"\n[SUMMARY] Saved to {summary_path}")
print(json.dumps(report, indent=2))

""" Wake generation """
# generation and temp sweep
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
    """Generate the same prompt at multiple temperatures for comparison."""
    print(f"PROMPT: {prompt}\n")
    for t in temps:
        generate_wake(prompt, temperature=t, **kwargs)
        print()


# generation (temp controls)
generate_wake("riverrun, past Eve and Adam's,")
# generate_wake("riverrun, past Eve and Adam's,", temperature=1.1)
# generate_wake("riverrun, past Eve and Adam's,", num_return_sequences=3, temperature=0.9)
# temperature_sweep("riverrun, past Eve and Adam's,")
