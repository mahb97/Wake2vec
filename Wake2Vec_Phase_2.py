# -*- coding: utf-8 -*-
"""
Wake2Vec Phase 2: Full Model Fine-Tune

Notebook: Wake2Vec Phase 2.ipynb
Model: TinyLlama-1.1B
Hardware: Google Colab T4 GPU

Overview
--------
Phase 2 performs full model fine-tuning on Finnegans Wake after P1's embedding-only
warmup. Unlike P1, which trained only the expanded vocabulary embeddings, P2 unfreezes
the entire model and uses LoRA adapters on attention and MLP layers to adapt the
model's behavior to Wake's linguistic patterns.

Prerequisites
-------------
- Phase 1 completed (embeddings trained to step 1300)
- P1 final artifacts saved in /content/drive/MyDrive/wake2vecP1/
- Finnegans Wake text file uploaded (FW_TEXT.txt)

Key Differences from Phase 1
----------------------------
| Aspect            | Phase 1              | Phase 2                      |
|-------------------|----------------------|------------------------------|
| Trainable params  | Embeddings only      | Full model + LoRA            |
| LoRA targets      | q_proj (frozen)      | q_proj, v_proj, MLP layers   |
| Learning rate     | 5e-4                 | 2e-5                         |
| Training duration | 1300 steps           | 2000 steps                   |
| Validation        | None                 | Held-out set + early stop    |
| Batch size        | 1                    | 8                            |
| Grad accumulation | 16                   | 2                            |

Memory Budget (T4: 15GB)
------------------------
Model (4-bit): ~1.5 GB | LoRA: ~0.3 GB | Optimizer: ~2-3 GB | Activations: ~4-5 GB
Total: ~8-10 GB with ~5-7 GB margin

Environment Notes (Nov 2025 Colab)
----------------------------------
Default Colab: torch 2.8.0, CUDA 12.9
This stack: torch 2.5.1+cu121, bitsandbytes 0.43.3, triton 3.1.0

Last updated: 2025-11-27
"""

# Environment Setup
# !pip uninstall -y torch torchvision torchaudio triton bitsandbytes transformers \
#     accelerate peft jax jaxlib flax cupy-cuda12x numba-cuda -y
# !pip cache purge
#
# import os
# os.kill(os.getpid(), 9)  # Force restart

# Install Compatible Versions
# !pip install --no-cache-dir \
#     torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
#     --index-url https://download.pytorch.org/whl/cu121
#
# !pip install -q --no-cache-dir \
#     triton==3.1.0 \
#     bitsandbytes==0.43.3 \
#     transformers==4.45.2 \
#     accelerate==0.34.2 \
#     peft==0.13.2 \
#     scikit-learn

# verify Packages
import os
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import bitsandbytes as bnb
import triton

print("=" * 60)
print("PACKAGE VERSIONS")
print("=" * 60)
print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")
print(f"bitsandbytes: {bnb.__version__}")
print(f"triton: {triton.__version__}")
print("=" * 60)

# bitsandbytes CUDA

print("Verification:")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  CUDA device: {torch.cuda.get_device_name(0)}")

from bitsandbytes.nn import Linear4bit
test_layer = Linear4bit(10, 10, bias=False)
test_layer.cuda()
test_input = torch.randn(1, 10).cuda()
with torch.no_grad():
    output = test_layer(test_input)
print("  bitsandbytes CUDA working")

# HF Login
import shutil

if os.path.exists('/content/drive'):
    shutil.rmtree('/content/drive')

from google.colab import drive
drive.mount('/content/drive')

from getpass import getpass
from huggingface_hub import login

HF_TOKEN = getpass("Paste your HF token (hidden): ")
login(token=HF_TOKEN, add_to_git_credential=True)

# GPU
import gc

torch.cuda.empty_cache()
gc.collect()

print("GPU Check:")
print(f"  Device: {torch.cuda.get_device_name(0)}")
print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# P1 Artifacts
from pathlib import Path

p = Path("/content/drive/MyDrive/wake2vecP1/checkpoint-0")
print(f"Path string: {p}")
print(f"exists(): {p.exists()}")
print(f"is_dir(): {p.is_dir()}")
print(f"resolve(): {p.resolve()}")

# P1 State
from transformers import AutoTokenizer

P1_ROOT = Path("/content/drive/MyDrive/wake2vecP1")
P1_TOKENIZER = P1_ROOT / "checkpoint-0"
P1_EMBEDDINGS = P1_ROOT / "emb_snaps/emb_step1300.pt"

if not P1_TOKENIZER.exists():
    raise FileNotFoundError(f"P1 tokenizer not found: {P1_TOKENIZER}")
if not P1_EMBEDDINGS.exists():
    raise FileNotFoundError(f"P1 embeddings not found: {P1_EMBEDDINGS}")

print("Loading P1 final state (step 1300)...")

tok = AutoTokenizer.from_pretrained(str(P1_TOKENIZER), use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"Tokenizer loaded: {len(tok)} tokens")

embed_weights = torch.load(P1_EMBEDDINGS, map_location="cpu")
print(f"Embeddings loaded: {embed_weights.shape}")

BASE_VOCAB = 32000
WAKE_TOKENS = len(tok) - BASE_VOCAB
print(f"Wake tokens added in P1: {WAKE_TOKENS}")
print(f"Training completed at step 1300")

# Config
DRIVE_RUN = Path("/content/drive/MyDrive/wake2vecP2_new")
LOCAL_RUN = Path("/content/local_run")
SENTRY = DRIVE_RUN / "sentry_backups"
EMB_SNAPS = DRIVE_RUN / "emb_snaps"
FULL_CHECKPOINTS = DRIVE_RUN / "full_checkpoints"

for d in [DRIVE_RUN, LOCAL_RUN, SENTRY, EMB_SNAPS, FULL_CHECKPOINTS]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FW_TEXT = "/content/FW_TEXT.txt"

MAX_STEPS = 2000
LR = 2e-5
WARMUP_RATIO = 0.10
BATCH_SIZE = 8
GRAD_ACCUM = 2
WEIGHT_DECAY = 0.01
SAVE_STEPS = 200
SEQ_LEN = 256
LORA_RANK = 8
EARLY_STOP_PATIENCE = 2

print("=" * 60)
print("WAKE2VEC PHASE 2 CONFIGURATION")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"P1 source: wake2vecP1 (step 1300)")
print(f"P2 output: {DRIVE_RUN}")
print(f"")
print(f"Steps: {MAX_STEPS}")
print(f"Learning rate: {LR}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Gradient accumulation: {GRAD_ACCUM}")
print(f"Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"LoRA rank: {LORA_RANK}")
print(f"Sequence length: {SEQ_LEN}")
print(f"Save every: {SAVE_STEPS} steps")
print("=" * 60)

#Dataset
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class BlockDataset(Dataset):
    def __init__(self, blocks, tokenizer, seq_len=256):
        self.blocks = blocks
        self.tokenizer = tokenizer
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
print(f"Total tokens: {len(ids)}")

blocks = []
stride = SEQ_LEN
for i in range(0, len(ids) - SEQ_LEN + 1, stride):
    chunk = ids[i:i + SEQ_LEN]
    if len(chunk) == SEQ_LEN:
        blocks.append(chunk)

print(f"Total blocks: {len(blocks)}")

train_blocks, val_blocks = train_test_split(
    blocks,
    test_size=0.10,
    random_state=42
)

print(f"Train blocks: {len(train_blocks)}")
print(f"Val blocks: {len(val_blocks)}")

train_ds = BlockDataset(train_blocks, tok, SEQ_LEN)
val_ds = BlockDataset(val_blocks, tok, SEQ_LEN)

print(f"Datasets ready (seq_len={SEQ_LEN})")

# Model with P1 Embeddings
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, get_peft_model

set_seed(42)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "13GB", "cpu": "30GB"}
)

model.config.use_cache = False
model.config.attn_implementation = "eager"
model.config.tie_word_embeddings = True
if hasattr(model, "tie_weights"):
    model.tie_weights()

print("Base model loaded")

print(f"Resizing embeddings to {len(tok)}...")
model.resize_token_embeddings(len(tok))

wte = model.get_input_embeddings()
if hasattr(model, "lm_head"):
    model.lm_head.weight = wte.weight

with torch.no_grad():
    wte.weight.copy_(embed_weights.to(wte.weight.device))

print("P1 embeddings loaded")

print("Adding LoRA adapters...")
peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK * 2,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Enable LoRA Gradients
print("Enabling gradients on trainable parameters...")

trainable_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_params.append(name)

print(f"Trainable parameters: {len(trainable_params)}")
for name in trainable_params[:10]:
    print(f"  {name}")

for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True

trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {trainable_count:,}")

if trainable_count == 0:
    raise RuntimeError("No trainable parameters found!")

# Disable Gradient Checkpointing

if hasattr(model, 'gradient_checkpointing_disable'):
    model.gradient_checkpointing_disable()
    print("Gradient checkpointing disabled on model")

print(f"Model gradient checkpointing: {model.is_gradient_checkpointing}")

# Callbacks

from transformers import TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback

def has_weights(ck):
    return (ck / "adapter_model.safetensors").exists() or (ck / "pytorch_model.bin").exists()

class EmbeddingSnapshot(TrainerCallback):
    """Save embedding weights every 50 steps."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % 50 == 0:
            try:
                m = kwargs['model']
                emb = m.get_input_embeddings().weight.data.clone().cpu()
                path = EMB_SNAPS / f"emb_step{state.global_step:04d}.pt"
                torch.save(emb, path)
                os.sync()
                print(f"[EMB] Step {state.global_step}: saved")
            except Exception as e:
                print(f"[EMB] Step {state.global_step}: {e}")

class FullCheckpoint(TrainerCallback):
    """Save complete resumable checkpoint every SAVE_STEPS."""
    def on_save(self, args, state, control, **kwargs):
        try:
            step = state.global_step
            m = kwargs['model']
            full_ck = FULL_CHECKPOINTS / f"step_{step:04d}"

            if full_ck.exists():
                shutil.rmtree(full_ck)
            full_ck.mkdir(parents=True, exist_ok=True)

            m.save_pretrained(full_ck)
            tok.save_pretrained(full_ck)

            emb = m.get_input_embeddings().weight.data.clone().cpu()
            torch.save(emb, full_ck / "embeddings.pt")

            torch.save({
                'global_step': step,
                'best_metric': state.best_metric,
                'epoch': state.epoch,
            }, full_ck / "training_state.pt")

            os.sync()

            required = ["adapter_config.json", "embeddings.pt", "tokenizer.json"]
            missing = [f for f in required if not (full_ck / f).exists()]

            if missing:
                print(f"[FULL] Step {step}: Missing {missing}")
            else:
                print(f"[FULL] Step {step}: saved")

        except Exception as e:
            print(f"[FULL] Step {state.global_step}: {e}")
            import traceback
            traceback.print_exc()

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
            os.sync()

            if dst.exists() and has_weights(dst):
                print(f"[SENTRY] {ck.name}: mirrored")

        except Exception as e:
            print(f"[SENTRY] {e}")

# trainer Setup

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
    logging_steps=50,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=SAVE_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    dataloader_pin_memory=False,
    gradient_checkpointing=False,
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
        EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)
    ]
)

print("=" * 60)
print("SAVE CONFIGURATION")
print("=" * 60)
print(f"[1] Embedding snapshots  -> {EMB_SNAPS}")
print(f"    Frequency: every 50 steps")
print(f"[2] Full checkpoints     -> {FULL_CHECKPOINTS}")
print(f"    Frequency: every {SAVE_STEPS} steps")
print(f"    Contains: model + tokenizer + embeddings + state")
print(f"[3] Sentry backups       -> {SENTRY}")
print(f"    Frequency: every {SAVE_STEPS} steps")
print("=" * 60)
print(f"MAX_STEPS: {MAX_STEPS} | SAVE_STEPS: {SAVE_STEPS}")
print("=" * 60)

# Train
print("=" * 80)
print("WAKE2VEC PHASE 2: FULL MODEL FINE-TUNE")
print("=" * 80)
print(f"Train samples: {len(train_ds)}")
print(f"Val samples: {len(val_ds)}")
print(f"Total steps: {MAX_STEPS}")
print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print("=" * 80)

print("Re-enabling LoRA gradients...")
for name, param in model.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}")

if trainable == 0:
    raise RuntimeError("No trainable parameters! LoRA not enabled properly.")

trainer.train()

print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

# Save Final Model

final_dir = DRIVE_RUN / "final"
final_dir.mkdir(exist_ok=True)
print(f"Saving final model to {final_dir}...")

model.save_pretrained(str(final_dir))
tok.save_pretrained(str(final_dir))

final_emb = model.get_input_embeddings().weight.data.clone().cpu()
torch.save(final_emb, final_dir / "embeddings.pt")

os.sync()
print("Final model + embeddings saved")

# CELL 18: Evaluation Plot

import matplotlib.pyplot as plt
import json

history_file = LOCAL_RUN / "trainer_state.json"
if history_file.exists():
    with open(history_file) as f:
        state = json.load(f)

    logs = state.get("log_history", [])

    train_loss = [(d["step"], d["loss"]) for d in logs if "loss" in d and "eval_loss" not in d]
    val_loss = [(d["step"], d["eval_loss"]) for d in logs if "eval_loss" in d]

    if train_loss and val_loss:
        plt.figure(figsize=(12, 6))

        train_steps, train_losses = zip(*train_loss)
        val_steps, val_losses = zip(*val_loss)

        plt.plot(train_steps, train_losses, 'o-', label="Training Loss", alpha=0.7)
        plt.plot(val_steps, val_losses, 's-', label="Validation Loss", alpha=0.7)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Wake2Vec P2: Training & Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = DRIVE_RUN / "p2_loss_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {plot_path}")
        plt.show()

        print("P2 Summary:")
        print(f"  Final train loss: {train_losses[-1]:.4f}")
        print(f"  Final val loss: {val_losses[-1]:.4f}")
        print(f"  Best val loss: {min(val_losses):.4f}")
