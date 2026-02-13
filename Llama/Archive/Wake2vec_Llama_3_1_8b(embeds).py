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

import shutil, pathlib

# remove old sentry 
sentry_ck200 = pathlib.Path("/content/drive/MyDrive/wake_llama_P1_fresh/sentry_backups/checkpoint-200")
if sentry_ck200.exists():
    shutil.rmtree(sentry_ck200)
    print("Deleted stale SENTRY checkpoint-200")

# config

SEED = 42
set_seed(SEED)

MODEL_NAME = "meta-llama/Llama-3.2-1B"
WAKE_LEX_PATH = "/content/wake_lexicon.txt"
CORPUS_TXT = "/content/FW_TEXT.txt"

# Paths 
RUN_DIR = Path("/content/drive/MyDrive/wake_llama_P1_fresh")
LOCAL_RUN = Path("/content/runs/wake_llama_P1_fresh")
SENTRY = RUN_DIR / "sentry_backups"
EMB_SNAPS = RUN_DIR / "emb_snaps"
FULL_CHECKPOINTS = RUN_DIR / "full_checkpoints"

RUN_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_RUN.mkdir(parents=True, exist_ok=True)
SENTRY.mkdir(parents=True, exist_ok=True)
EMB_SNAPS.mkdir(parents=True, exist_ok=True)
FULL_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

PHASE1_EMB = FULL_CHECKPOINTS / "step_0700" / "embeddings.pt"
USE_PHASE1 = PHASE1_EMB.exists()
print("[PHASE2] PHASE1_EMB:", PHASE1_EMB, "exists:", USE_PHASE1)

# Training hyperparams
SEQ_LEN = 512
STRIDE = 512          
MAX_STEPS = 6000      # up from 2000
LOG_STEPS = 50
SAVE_STEPS = 200     
EMB_SNAP_STEPS = 200 
LR = 2e-4             # down from 5e-4
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

# load trained embeddings from step_0700 but start fresh optimiser/scheduler
if USE_PHASE1:
    print(f"[PHASE2] Loading embeddings from {PHASE1_EMB}")
    saved_emb = torch.load(PHASE1_EMB, map_location=wte.weight.device)
    if saved_emb.shape != wte.weight.shape:
        raise ValueError(f"[PHASE2] Embedding shape mismatch: saved {saved_emb.shape}, current {wte.weight.shape}")
    with torch.no_grad():
        wte.weight.copy_(saved_emb.to(wte.weight.device))
    print("[PHASE2] Embeddings loaded into model")
else:
    print("[PHASE2] No phase-1 embeddings found; training from spherical init.")
    
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

trainer = EmbOnlyTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=None,
    callbacks=[EmbeddingSnapshot(), FullCheckpoint(), SentryMirror()]
)

print(f"Model: {MODEL_NAME}")
print(f"Max steps: {MAX_STEPS}")
print(f"Embedding snapshots: {EMB_SNAPS} (every {EMB_SNAP_STEPS} steps)")
print(f"Full checkpoints: {FULL_CHECKPOINTS} (every {SAVE_STEPS} steps)")
print(f"Sentry backups: {SENTRY} (every {SAVE_STEPS} steps)")

# resume config 
RESUME_STEP = 200
RESUME_CKPT = SENTRY / f"checkpoint-{RESUME_STEP}"
resume_path = None

if RESUME_CKPT.exists():
    resume_path = str(RESUME_CKPT)
    print(f"[RESUME] Found checkpoint at {resume_path}, resuming from there.")
else:
    print(f"[RESUME] No checkpoint-{RESUME_STEP} found under {SENTRY}, starting from 0.")

print("[RUN] Evolved P1 from PHASE1_EMB (step_0700)")

if resume_path is not None:
    trainer.train(resume_from_checkpoint=resume_path)
else:
    trainer.train()

# save final 

final_dir = RUN_DIR / "final"
final_dir.mkdir(exist_ok=True)

model.save_pretrained(final_dir)
tok.save_pretrained(final_dir)
torch.save(wte.weight.detach().cpu(), final_dir / "embeddings.pt")
os.sync()

print(f"Final artifacts saved to {final_dir}")

"""P1 eval ⬇"""

# P1 Eval 
import json, numpy as np, torch
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
WAKE2VEC_ROOT = Path("/content/drive/MyDrive/wake_llama_P1")
SENTRY = WAKE2VEC_ROOT / "sentry_backups"
LOCAL_RUN = Path("/content/runs/wake_llama_P1")

# latest checkpoint 
local_checkpoints = sorted(LOCAL_RUN.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
sentry_checkpoints = sorted(SENTRY.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))

# Use latest available
if local_checkpoints:
    BASE_CKPT = local_checkpoints[-1]
    print(f"[P1 EVAL] Using LOCAL checkpoint: {BASE_CKPT}")
elif sentry_checkpoints:
    BASE_CKPT = sentry_checkpoints[-1]
    print(f"[P1 EVAL] Using SENTRY checkpoint: {BASE_CKPT}")
else:
    raise FileNotFoundError("No checkpoints found in local or Drive backup!")

print(f"Checkpoint: {BASE_CKPT.name}")

# Load embeds from checkpoint
checkpoint_state = torch.load(BASE_CKPT / "pytorch_model.bin", map_location="cpu")

# Find embed key
embed_key = None
for key in checkpoint_state.keys():
    if "embed_tokens.weight" in key:
        embed_key = key
        break

if embed_key is None:
    raise KeyError("Could not find embeddings in checkpoint")

E_post = checkpoint_state[embed_key].numpy()
print(f"Loaded embeddings: {E_post.shape}")

# Norm statistics
from numpy.linalg import norm
norms = norm(E_post, axis=1)

# Get final step from checkpoint name
final_step = int(BASE_CKPT.name.split("-")[-1])

# Get final loss from trainer_state
state_file = BASE_CKPT / "trainer_state.json"
final_loss = None
if state_file.exists():
    s = json.loads(state_file.read_text())
    logs = [d for d in s.get("log_history", []) if "loss" in d]
    if logs:
        final_loss = float(logs[-1]["loss"])

# best checkpoint info from trainer_state 
best_checkpoint_name = None
if state_file.exists():
    # reuse `s` if still in scope; otherwise reload it
    if "s" not in locals():
        s = json.loads(state_file.read_text())
    best_path = s.get("best_model_checkpoint")
    if best_path:
        best_ckpt_path = Path(best_path)
        best_checkpoint_name = best_ckpt_path.name
        print(f"\n[BEST CKPT] {best_checkpoint_name}")
    else:
        print("\n[BEST CKPT] No best_model_checkpoint recorded in trainer_state.json")

report["best_checkpoint"] = best_checkpoint_name

report = {
    "model": "meta-llama/Llama-3.2-1B",
    "final_step": final_step,
    "final_loss": final_loss,
    "post_mean_norm": float(norms.mean()),
    "post_std_norm": float(norms.std()),
    "post_min_norm": float(norms.min()),
    "post_max_norm": float(norms.max()),
    "n_vocab": int(E_post.shape[0]),
    "base_vocab": 128256,
    "new_tokens": int(E_post.shape[0] - 128256)
}

# Base vs new token norm stats

BASE_VOCAB = 128256  # consistent with report

base_norms = norms[:BASE_VOCAB]
new_norms = norms[BASE_VOCAB:]

report.update({
    "base_mean_norm": float(base_norms.mean()),
    "base_std_norm": float(base_norms.std()),
    "new_mean_norm": float(new_norms.mean()),
    "new_std_norm": float(new_norms.std()),
})

print("\n[BASE VS NEW TOKEN NORMS]")
print(f"Base tokens: mean={base_norms.mean():.3f}, std={base_norms.std():.3f}")
print(f"New  tokens: mean={new_norms.mean():.3f}, std={new_norms.std():.3f}")

# Optional: base vs new norm hist plot
plt.figure(figsize=(10, 6))
plt.hist(base_norms, bins=50, alpha=0.5, label="Base tokens")
plt.hist(new_norms,  bins=50, alpha=0.5, label="New Wake tokens")
plt.axvline(base_norms.mean(), linestyle="--", label=f"Base mean: {base_norms.mean():.2f}")
plt.axvline(new_norms.mean(),  linestyle=":",  label=f"New mean: {new_norms.mean():.2f}")
plt.title("Wake2Vec P1: Embedding Norms – Base vs New Tokens")
plt.xlabel("L2 Norm")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)

base_new_norm_plot = WAKE2VEC_ROOT / "p1_llama_base_vs_new_norms.png"
plt.savefig(base_new_norm_plot, dpi=150, bbox_inches="tight")
print(f"[PLOT] Saved to {base_new_norm_plot}")
plt.show()

# Isotropy estimate (sampled pairwise cosine)
# Unit-normalise embeds along the feature axis
from numpy.linalg import norm as l2

E_unit = E_post / l2(E_post, axis=1, keepdims=True)

# sample subset for isotropy computation
rng = np.random.default_rng(0)
sample_size = min(5000, E_unit.shape[0])
idx = rng.choice(E_unit.shape[0], size=sample_size, replace=False)
E_sample = E_unit[idx]

# pairwise cosine matrix 
cos = E_sample @ E_sample.T

n = cos.shape[0]
# exclude diagonal when averaging
mean_pairwise = (cos.sum() - np.trace(cos)) / (n * n - n)

report["isotropy_mean_pairwise_cosine"] = float(mean_pairwise)
report["isotropy_sample_size"] = int(sample_size)

print(f"\n[ISOTROPY] Sample size={sample_size}, mean pairwise cosine={mean_pairwise:.4f}")

# Save report
(WAKE2VEC_ROOT / "p1_llama_summary.json").write_text(json.dumps(report, indent=2))
print("\n[P1 SUMMARY]")
print(json.dumps(report, indent=2))

# Nearest neighbours helper (embed space sanity check)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(BASE_CKPT)

# Reuse unit-normalised embedds
def nearest(token_str, k=15):
    """Print k nearest neighbours to a given token string."""
    tid = tok.convert_tokens_to_ids(token_str)
    if tid is None or tid < 0:
        print(f"[NN] Token not found in vocab: {token_str!r}")
        return

    v = E_unit[tid]
    sims = E_unit @ v  # cosine similarities because E_unit is normalised
    topk = np.argsort(-sims)[:k+1]  # include self

    print(f"\n[NN] Nearest to {token_str!r} (id={tid})")
    for i in topk:
        tok_str = tok.convert_ids_to_tokens(int(i))
        print(f"  {i:6d}  {tok_str!r}  cos={sims[i]:.3f}")

# probes (tbc)
nearest(" the")
nearest(" and")

# wake tokens
nearest("lipoleums") 
nearest("honuphrius")
nearest("prankquean")
nearest("gracehoper")
nearest("brinabride")

# random sample of new Wake tokens and their nearest neighbours
#new_ids = np.arange(BASE_VOCAB, E_post.shape[0])
#rng = np.random.default_rng(1)
#sample_new = rng.choice(new_ids, size=min(5, len(new_ids)), replace=False)

#for tid in sample_new:
    #token_str = tok.convert_ids_to_tokens(int(tid))
    #print(f"\n[NN – NEW TOKEN] id={tid}, token={token_str!r}")
    #v = E_unit[tid]
    #sims = E_unit @ v
    #topk = np.argsort(-sims)[:10]
    #nn_tokens = [tok.convert_ids_to_tokens(int(i)) for i in topk]
    #print("  nearest:", nn_tokens)

# Loss plot from trainer_state
if state_file.exists():
    s = json.loads(state_file.read_text())
    logs = [d for d in s.get("log_history", []) if "loss" in d]

    if logs:
        steps = [d["step"] for d in logs]
        losses = [float(d["loss"]) for d in logs]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, 'o-', alpha=0.7, label="Training Loss", markersize=4)
        plt.title(f"Wake2Vec P1: Llama-3.2-1B Loss Curve (0→{final_step})")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plot_path = WAKE2VEC_ROOT / "p1_llama_loss.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\n[PLOT] Saved to {plot_path}")
        plt.show()
    else:
        print("\n[WARNING] No loss logs found in trainer_state.json")
else:
    print(f"\n[WARNING] trainer_state.json not found at {state_file}")

# Norm distribution plot
plt.figure(figsize=(10, 6))
plt.hist(norms, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(norms.mean(), color='red', linestyle='--', label=f'Mean: {norms.mean():.2f}')
plt.title("Wake2Vec P1: Embedding Norm Distribution")
plt.xlabel("L2 Norm")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)

norm_plot_path = WAKE2VEC_ROOT / "p1_llama_norms.png"
plt.savefig(norm_plot_path, dpi=150, bbox_inches="tight")
print(f"[PLOT] Saved to {norm_plot_path}")
plt.show()

