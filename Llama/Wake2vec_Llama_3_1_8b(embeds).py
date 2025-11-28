# -*- coding: utf-8 -*-
"""
Wake2Vec Phase 1: Llama-3.2-1B Embedding Fine-Tuning

Model: meta-llama/Llama-3.2-1B (4-bit quantized)
Hardware: Google Colab T4 GPU (15GB VRAM)
Training: Embedding-only with gradient masking on base vocabulary

This notebook implements vocabulary extension and embedding-only fine-tuning
for Llama-3.2-1B with Finnegans Wake neologisms. Base model parameters remain
frozen; only new Wake token embeddings receive gradient updates.

Prerequisites:
- Hugging Face account with Llama access approved
- wake_lexicon.txt uploaded to Colab
- FW_TEXT.txt (Finnegans Wake corpus) uploaded to Colab
"""
# Environment Setup

# !pip uninstall -y torch torchvision torchaudio triton bitsandbytes transformers accelerate peft jax jaxlib flax -y
# !pip cache purge
# import os; os.kill(os.getpid(), 9)

# Install Compatible Packages
# !pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# !pip install -q --no-cache-dir triton==3.1.0 bitsandbytes==0.43.3 transformers==4.45.2 accelerate==0.34.2 peft==0.13.2 scikit-learn

# Imports and Environment
import os
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import json
import shutil
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed
)
from peft import LoraConfig, get_peft_model

# Drive and HF Authenticate

from google.colab import drive
drive.mount('/content/drive')

from getpass import getpass
from huggingface_hub import login

HF_TOKEN = getpass("Paste HF token: ")
login(token=HF_TOKEN, add_to_git_credential=True)

# config

SEED = 42
set_seed(SEED)

MODEL_NAME = "meta-llama/Llama-3.2-1B"
WAKE_LEX_PATH = "/content/wake_lexicon.txt"
CORPUS_TXT = "/content/FW_TEXT.txt"

# Paths 
RUN_DIR = Path("/content/drive/MyDrive/wake_llama_P1_v2")
LOCAL_RUN = Path("/content/runs/wake_llama_P1_v2")
SENTRY = RUN_DIR / "sentry_backups"
EMB_SNAPS = RUN_DIR / "emb_snaps"
FULL_CHECKPOINTS = RUN_DIR / "full_checkpoints"

for d in [RUN_DIR, LOCAL_RUN, SENTRY, EMB_SNAPS, FULL_CHECKPOINTS]:
    d.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
SEQ_LEN = 512
STRIDE = 512
MAX_STEPS = 2000
LOG_STEPS = 50
SAVE_STEPS = 100
EMB_SNAP_STEPS = 50
LR = 5e-4
GRAD_ACCUM = 16

print("=" * 60)
print("CONFIGURATION")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"Output: {RUN_DIR}")
print(f"Steps: {MAX_STEPS}")
print(f"Learning rate: {LR}")
print(f"Batch: 1 x {GRAD_ACCUM} = {GRAD_ACCUM}")
print(f"Sequence length: {SEQ_LEN}")
print(f"Save frequency: {SAVE_STEPS} steps")
print(f"Embedding snapshots: {EMB_SNAP_STEPS} steps")
print("=" * 60)

# GPU Verification

torch.cuda.empty_cache()
gc.collect()

print("GPU Status:")
print(f"  Device: {torch.cuda.get_device_name(0)}")
print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# Model with 4-bit Quantization

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True
)

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print("Loading model (4-bit quantized)...")
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

print(f"Base vocabulary: {len(tok)}")

# PEFT Adapter (Frozen)
# Minimal LoRA adapter required for gradient flow in quantized models.
# Adapter weights remain frozen; only embeddings are trained.
peft_cfg = LoraConfig(
    r=1,
    lora_alpha=1,
    lora_dropout=0.0,
    target_modules=["q_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_cfg)

# Freeze all parameters
for n, p in model.named_parameters():
    p.requires_grad = False

# Wake vocab Extension

def read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

wake_tokens = read_lines(WAKE_LEX_PATH)
missing = [t for t in wake_tokens if tok.convert_tokens_to_ids(t) == tok.unk_token_id]
num_added = tok.add_tokens(missing, special_tokens=False)

old_vocab = model.get_input_embeddings().weight.shape[0]
model.resize_token_embeddings(len(tok))

wte = model.get_input_embeddings()
if hasattr(model, "lm_head"):
    model.lm_head.weight = wte.weight

print(f"Vocabulary: {old_vocab} -> {len(tok)} (+{num_added} Wake tokens)")

# Embedding Initialization (Spherical)
# Initialize new embeddings on spherical shell matching base embedding geometry.
# This provides better initial separation than mean or random initialization.
with torch.no_grad():
    base_emb = wte.weight[:old_vocab]
    dim = base_emb.shape[1]
    std = base_emb.std().item()
    base_radius = std * math.sqrt(dim)
    target_radius = 1.5 * base_radius
    
    if num_added > 0:
        new_emb = torch.randn((num_added, dim), device=wte.weight.device)
        new_emb = new_emb / (new_emb.norm(dim=1, keepdim=True) + 1e-8) * target_radius
        wte.weight.data[old_vocab:old_vocab + num_added] = new_emb

print(f"Embedding initialization: spherical shell, radius {target_radius:.2f}")

# Gradient Masking
# Enable gradients only on embedding layer
wte.weight.requires_grad = True

# Create index tensors for gradient masking
new_rows = torch.arange(old_vocab, old_vocab + num_added, device=wte.weight.device) if num_added > 0 else None
base_rows = torch.arange(0, old_vocab, device=wte.weight.device)

def mask_grad(grad):
    """Zero gradients for base vocabulary embeddings."""
    if grad is None or new_rows is None:
        return grad
    grad[base_rows] = 0
    return grad

wte.weight.register_hook(mask_grad)

trainable = num_added * wte.weight.shape[1] if num_added > 0 else 0
print(f"Trainable parameters: {trainable:,} (Wake embeddings only)")

# Dataset

class BlockDataset(Dataset):
    """Fixed-length block dataset for language modeling."""
    
    def __init__(self, path, tokenizer, seq_len=512, stride=512):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Corpus not found: {path}")
        
        text = open(path, "r", encoding="utf-8").read()
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        
        self.blocks = []
        for i in range(0, max(1, len(ids) - seq_len), stride):
            chunk = ids[i:i + seq_len]
            if len(chunk) >= seq_len // 2:
                self.blocks.append(chunk[:seq_len])
    
    def __len__(self):
        return len(self.blocks)
    
    def __getitem__(self, idx):
        ids = torch.tensor(self.blocks[idx], dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids)
        }

train_ds = BlockDataset(CORPUS_TXT, tok, SEQ_LEN, STRIDE)
print(f"Dataset: {len(train_ds)} blocks, {SEQ_LEN} tokens each")

# callbacks

class EmbeddingSnapshot(TrainerCallback):
    """Save embedding weights at regular intervals."""
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % EMB_SNAP_STEPS == 0:
            path = EMB_SNAPS / f"emb_step{state.global_step:04d}.pt"
            torch.save(wte.weight.detach().cpu(), path)
            os.sync()
            print(f"[EMB] Step {state.global_step}: saved")


class FullCheckpoint(TrainerCallback):
    """Save complete resumable checkpoint to Drive."""
    
    def on_save(self, args, state, control, **kwargs):
        try:
            step = state.global_step
            full_ck = FULL_CHECKPOINTS / f"step_{step:04d}"
            
            if full_ck.exists():
                shutil.rmtree(full_ck)
            full_ck.mkdir(parents=True, exist_ok=True)
            
            model.save_pretrained(full_ck)
            tok.save_pretrained(full_ck)
            torch.save(wte.weight.detach().cpu(), full_ck / "embeddings.pt")
            torch.save({
                'global_step': step,
                'best_metric': state.best_metric,
                'epoch': state.epoch,
            }, full_ck / "training_state.pt")
            
            os.sync()
            print(f"[FULL] Step {step}: saved")
        except Exception as e:
            print(f"[FULL] Step {step}: {e}")


def has_weights(ck):
    return (ck / "adapter_model.safetensors").exists() or (ck / "pytorch_model.bin").exists()


class SentryMirror(TrainerCallback):
    """Mirror Trainer checkpoints to Drive for redundancy."""
    
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
            if not dst.exists():
                shutil.copytree(ck, dst)
                os.sync()
                print(f"[SENTRY] {ck.name}: mirrored")
        except Exception as e:
            print(f"[SENTRY] {e}")

# Trainer

class EmbeddingOnlyTrainer(Trainer):
    """Trainer configured for embedding-only optimization."""
    
    def create_optimizer(self):
        from torch.optim import AdamW
        if not hasattr(self, "optimizer") or self.optimizer is None:
            self.optimizer = AdamW(
                [{"params": [wte.weight], "lr": LR, "weight_decay": 0.0}],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        return self.optimizer
    
    def compute_loss(self, model, inputs, return_outputs=False):
        out = model(**inputs, use_cache=False)
        loss = out.loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("NaN/Inf loss detected")
        return (loss, out) if return_outputs else loss

# training args

args = TrainingArguments(
    output_dir=str(LOCAL_RUN),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    max_steps=MAX_STEPS,
    warmup_steps=max(20, MAX_STEPS // 20),
    lr_scheduler_type="cosine",
    weight_decay=0.0,
    fp16=False,
    bf16=True,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=6,
    evaluation_strategy="no",
    report_to="none",
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)

trainer = EmbeddingOnlyTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=None,
    callbacks=[EmbeddingSnapshot(), FullCheckpoint(), SentryMirror()]
)

# Train

print("=" * 60)
print("WAKE2VEC LLAMA P1: EMBEDDING FINE-TUNING")
print("=" * 60)
print(f"Dataset: {len(train_ds)} blocks")
print(f"Steps: {MAX_STEPS}")
print(f"Embedding snapshots: {EMB_SNAPS}")
print(f"Full checkpoints: {FULL_CHECKPOINTS}")
print(f"Sentry backups: {SENTRY}")
print("=" * 60)

trainer.train()

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

# save final 

final_dir = RUN_DIR / "final"
final_dir.mkdir(exist_ok=True)

model.save_pretrained(final_dir)
tok.save_pretrained(final_dir)
torch.save(wte.weight.detach().cpu(), final_dir / "embeddings.pt")
os.sync()

print(f"Final artifacts saved to {final_dir}")

# eval 

import matplotlib.pyplot as plt
import numpy as np

# Load training history
state_files = list(LOCAL_RUN.rglob("trainer_state.json"))
if state_files:
    latest = max(state_files, key=lambda p: p.stat().st_mtime)
    with open(latest) as f:
        state = json.load(f)
    
    logs = state.get("log_history", [])
    train_data = [(d["step"], d["loss"]) for d in logs if "loss" in d]
    
    if train_data:
        steps, losses = zip(*train_data)
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, losses, 'b-o', alpha=0.7, label='Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Wake2Vec Llama P1: Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = RUN_DIR / "p1_loss_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {plot_path}")
        plt.show()
        
        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Final loss: {losses[-1]:.4f}")
        print(f"Reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

# Embedding statistics
E = wte.weight.detach().cpu().numpy()
norms = np.linalg.norm(E, axis=1)

print(f"\nEmbedding Statistics:")
print(f"  Total tokens: {E.shape[0]}")
print(f"  Dimensions: {E.shape[1]}")
print(f"  Mean norm: {norms.mean():.4f}")
print(f"  Std norm: {norms.std():.4f}")
print(f"  Wake token norms: {norms[old_vocab:].mean():.4f} (mean)")
