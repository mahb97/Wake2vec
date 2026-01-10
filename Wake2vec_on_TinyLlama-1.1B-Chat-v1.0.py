# envi
import os
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import torch
import shutil
import time
from pathlib import Path

torch.cuda.empty_cache()
gc.collect()

print("GPU Check:")
print(f"  Device: {torch.cuda.get_device_name(0)}")
print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"

# drive 
if os.path.exists('/content/drive'):
    shutil.rmtree('/content/drive')

from google.colab import drive
drive.mount('/content/drive')

# config
# Paths
WAKE2VEC_ROOT = Path("/content/drive/MyDrive/wake2vecP1_TinyLlama")
LOCAL_RUN = Path("/content/runs/wake2vecP1_TinyLlama")
SENTRY = WAKE2VEC_ROOT / "sentry_backups"
EMB_SNAPS = WAKE2VEC_ROOT / "emb_snaps"
FULL_CHECKPOINTS = WAKE2VEC_ROOT / "full_checkpoints"

for d in [WAKE2VEC_ROOT, LOCAL_RUN, SENTRY, EMB_SNAPS, FULL_CHECKPOINTS]:
    d.mkdir(parents=True, exist_ok=True)

# Model and data
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FW_TEXT = "/content/FW_TEXT.txt"
WAKE_LEXICON = "/content/wake_lexicon.txt"

# Training hyperparameters
MAX_STEPS = 3000
LR = 1e-4              
WARMUP_RATIO = 0.1     
WEIGHT_DECAY = 0.01    
BATCH_SIZE = 1
GRAD_ACCUM = 16
SEQ_LEN = 256
SAVE_STEPS = 100
LOG_STEPS = 50         
EVAL_STEPS = 100       

# Initialization
INIT_NOISE_SCALE = 0.1  

print("WAKE2VEC P1 TinyLlama config")
print(f"Output: {WAKE2VEC_ROOT}")
print(f"Steps: {MAX_STEPS}")
print(f"LR: {LR}")
print(f"Weight decay: {WEIGHT_DECAY}")
print(f"Warmup: {WARMUP_RATIO}")
print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"Eval: every {EVAL_STEPS} steps")

# tok
from transformers import AutoTokenizer

print("\nLoading base TinyLlama tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

BASE_VOCAB = len(tok)
print(f"Base vocab: {BASE_VOCAB}")

# Add Wake tokens
print(f"Loading Wake lexicon from {WAKE_LEXICON}...")
with open(WAKE_LEXICON, 'r', encoding='utf-8') as f:
    wake_tokens = [line.strip() for line in f if line.strip()]

num_added = tok.add_tokens(wake_tokens)
print(f"Wake tokens added: {num_added}")
print(f"New vocab size: {len(tok)}")

# dataset split
from torch.utils.data import Dataset

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

print("\n get FW")
if not os.path.exists(FW_TEXT):
    raise FileNotFoundError(f"FW not found: {FW_TEXT}")

with open(FW_TEXT, 'r', encoding='utf-8') as f:
    text = f.read()

ids = tok(text, add_special_tokens=False)["input_ids"]
print(f"Total tokens: {len(ids)}")

# Create non-overlapping blocks
blocks = []
for i in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN):
    chunk = ids[i:i + SEQ_LEN]
    if len(chunk) == SEQ_LEN:
        blocks.append(chunk)

print(f"Total blocks: {len(blocks)}")

# val
split_idx = int(len(blocks) * 0.9)
train_blocks = blocks[:split_idx]
val_blocks = blocks[split_idx:]

train_ds = BlockDataset(train_blocks, tok, SEQ_LEN)
val_ds = BlockDataset(val_blocks, tok, SEQ_LEN)

print(f"Train blocks: {len(train_ds)}")
print(f"Val blocks: {len(val_ds)}")

# load tinyLlama 
from transformers import AutoModelForCausalLM

print("\nLoading tiny")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=True,
)

model.to("cuda")
model.config.use_cache = False
model.config.attn_implementation = "eager"

# Resize embeddings
print(f"Resizing embeds: {BASE_VOCAB} -> {len(tok)}...")
model.resize_token_embeddings(len(tok))

# new tokens with unique vectors sampled from base distribution
print("Initializing Wake token embeds (sampled from base distribution)...")
with torch.no_grad():
    emb = model.get_input_embeddings()
    old_embeddings = emb.weight[:BASE_VOCAB]
    
    # statistics of base embeddings
    base_mean = old_embeddings.mean(dim=0)
    base_std = old_embeddings.std(dim=0)
    
    # new tokens 
    num_new = len(tok) - BASE_VOCAB
    noise = torch.randn(num_new, old_embeddings.shape[1], device=emb.weight.device)
    emb.weight[BASE_VOCAB:] = base_mean + (noise * base_std * INIT_NOISE_SCALE)
    
    print(f"  Base mean norm: {base_mean.norm().item():.4f}")
    print(f"  Base std (per-dim avg): {base_std.mean().item():.4f}")
    print(f"  New tokens initialized with noise scale: {INIT_NOISE_SCALE}")

print(f"Embeds shape: {emb.weight.shape}")

# freeze except embeds
for p in model.parameters():
    p.requires_grad = False

emb.weight.requires_grad = True

# Tie input/output embeds
with torch.no_grad():
    model.get_output_embeddings().weight = emb.weight

model.train()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Tied embeddings: {emb.weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()}")
print(f"Trainable params: {trainable:,}")
print("no grad masking") 

# for continuation only, skip to callbacks on first run, uncomment if needed 

# from pathlib import Path

# SENTRY = Path("/content/drive/MyDrive/wake2vec_tiny_p1_fryembeds/sentry_backups")
# print("SENTRY:", SENTRY)
# print("Exists:", SENTRY.exists())

# if SENTRY.exists():
#    ckpts = sorted(SENTRY.glob("checkpoint-*"),
#                   key=lambda p: int(p.name.split("-")[-1]))
#    print("Found checkpoints:", [c.name for c in ckpts])
# else:
#   ckpts = []

# callbacks 
from transformers import TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback

def has_weights(ck):
    return (ck / "model.safetensors").exists() or (ck / "pytorch_model.bin").exists()

class EmbeddingSnapshot(TrainerCallback):
    """Save embedding weights every 50 steps."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % 50 == 0:
            try:
                E = model.get_input_embeddings().weight.detach().cpu()
                path = EMB_SNAPS / f"emb_step{state.global_step:04d}.pt"
                torch.save(E, path)
                os.sync()
                print(f"[EMB] Step {state.global_step}: saved")
            except Exception as e:
                print(f"[EMB] Step {state.global_step}: {e}")

class FullCheckpoint(TrainerCallback):
    """Save complete checkpoint every SAVE_STEPS."""
    def on_save(self, args, state, control, **kwargs):
        try:
            step = state.global_step
            full_ck = FULL_CHECKPOINTS / f"step_{step:04d}"

            if full_ck.exists():
                shutil.rmtree(full_ck)
            full_ck.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(full_ck)
            tok.save_pretrained(full_ck)

            E = model.get_input_embeddings().weight.detach().cpu()
            torch.save(E, full_ck / "embeddings.pt")

            torch.save({
                'global_step': step,
                'best_metric': state.best_metric,
                'epoch': state.epoch,
            }, full_ck / "training_state.pt")

            os.sync()
            print(f"[FULL] Step {step}: saved")
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
            os.sync()
            print(f"[SENTRY] {ck.name}: mirrored")
        except Exception as e:
            print(f"[SENTRY] {e}")

class LossMonitor(TrainerCallback):
    """Track and warn about train/eval divergence."""
    def __init__(self):
        self.last_train_loss = None
        self.last_eval_loss = None
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.last_train_loss = logs["loss"]
        if "eval_loss" in logs:
            self.last_eval_loss = logs["eval_loss"]
            if self.last_train_loss is not None:
                gap = self.last_eval_loss - self.last_train_loss
                if gap > 3.0:
                    print(f"[WARN] Large train/eval gap: {gap:.2f} - potential overfitting")

print(f"Hooks: {emb.weight._backward_hooks}")

# training args 
args = TrainingArguments(
    output_dir=str(LOCAL_RUN),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=MAX_STEPS,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,          
    optim="adafactor",
    logging_steps=LOG_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    load_best_model_at_end=True,        
    metric_for_best_model="eval_loss",  
    greater_is_better=False,            
    gradient_checkpointing=True,
    fp16=False,
    bf16=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    report_to=["none"],
    max_grad_norm=1.0,
)

# trianer 
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
    ],
)

print(f"\nEmbedding snapshots: {EMB_SNAPS} (every 50 steps)")
print(f"Full checkpoints: {FULL_CHECKPOINTS} (every {SAVE_STEPS} steps)")
print(f"Sentry backups: {SENTRY} (every {SAVE_STEPS} steps)")

# train
print("WAKE2VEC P1 TINYLLAMA EMBED-ONLY FINE-TUNE")
print(f"Train: {len(train_ds)} blocks | Val: {len(val_ds)} blocks")
print(f"Steps: {MAX_STEPS}")
print(f"Trainable: {trainable:,} params")

t0 = time.time()
trainer.train()
elapsed = (time.time() - t0) / 60

print("TRAINING COMPLETE")
print(f"Time: {elapsed:.1f} minutes")

# save final model 
final_dir = WAKE2VEC_ROOT / "final"
final_dir.mkdir(exist_ok=True)

model.save_pretrained(str(final_dir))
tok.save_pretrained(str(final_dir))

final_emb = model.get_input_embeddings().weight.detach().cpu()
torch.save(final_emb, final_dir / "embeddings.pt")

os.sync()
print(f"Final model saved to {final_dir}")

# loss curve matplot
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
        plt.title('Wake2Vec P1 V3: Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = WAKE2VEC_ROOT / "p1_v3_loss_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {plot_path}")
        plt.show()

        print(f"\nFinal train loss: {losses[-1]:.4f}")
        if val_data:
            print(f"Final val loss: {v_losses[-1]:.4f}")
            print(f"Best val loss: {min(v_losses):.4f}")

# embeds analysis 
import numpy as np
from numpy.linalg import norm as l2

E_post = final_emb.numpy()
vocab_size, dim = E_post.shape

norms = l2(E_post, axis=1)
base_norms = norms[:BASE_VOCAB]
new_norms = norms[BASE_VOCAB:]

print(f"\n[NORMS - GLOBAL]")
print(f"  Mean: {norms.mean():.4f}, Std: {norms.std():.4f}")
print(f"  Min: {norms.min():.4f}, Max: {norms.max():.4f}")

print(f"\n[NORMS - BASE VS NEW]")
print(f"  Base tokens: mean={base_norms.mean():.4f}, std={base_norms.std():.4f}")
print(f"  New tokens:  mean={new_norms.mean():.4f}, std={new_norms.std():.4f}")

# Isotropy
print(f"\n[ISOTROPY]")
E_unit = E_post / (norms[:, None] + 1e-12)
rng = np.random.default_rng(0)
sample_size = min(5000, vocab_size)
idx = rng.choice(vocab_size, size=sample_size, replace=False)
E_sample = E_unit[idx]
cos = E_sample @ E_sample.T
n = cos.shape[0]
mean_pairwise = (cos.sum() - np.trace(cos)) / (n * n - n)
print(f"  Sample size: {sample_size}")
print(f"  Mean pairwise cosine: {mean_pairwise:.4f}")

# Norm distribution plots
plt.figure(figsize=(10, 6))
plt.hist(norms, bins=50, alpha=0.7, edgecolor="black")
plt.axvline(norms.mean(), linestyle="--", label=f"Mean: {norms.mean():.2f}")
plt.title("Wake2Vec P1 TInyLlama Embedding Norm Distribution (All Tokens)")
plt.xlabel("L2 norm")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.legend()
global_norm_plot = WAKE2VEC_ROOT / "p1_TinyLlama_norms_global.png"
plt.savefig(global_norm_plot, dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {global_norm_plot}")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(base_norms, bins=50, alpha=0.5, label="Base tokens")
plt.hist(new_norms, bins=50, alpha=0.5, label="New Wake tokens")
plt.axvline(base_norms.mean(), linestyle="--", label=f"Base mean: {base_norms.mean():.2f}")
plt.axvline(new_norms.mean(), linestyle=":", label=f"New mean: {new_norms.mean():.2f}")
plt.title("Wake2Vec P1 TinyLlama Embedding Norms - Base vs New Tokens")
plt.xlabel("L2 norm")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.legend()
bv_norm_plot = WAKE2VEC_ROOT / "p1_TinyLlama_norms_base_vs_new.png"
plt.savefig(bv_norm_plot, dpi=150, bbox_inches="tight")
print(f"Plot saved: {bv_norm_plot}")
plt.show()

# sum json 
report = {
    "model": MODEL_NAME,
    "version": "v3",
    "checkpoint": "final",
    "vocab_size": int(vocab_size),
    "dim": int(dim),
    "base_vocab": int(BASE_VOCAB),
    "new_tokens": int(vocab_size - BASE_VOCAB),
    "hyperparameters": {
        "lr": LR,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "max_steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "init_noise_scale": INIT_NOISE_SCALE,
    },
    "post_mean_norm": float(norms.mean()),
    "post_std_norm": float(norms.std()),
    "post_min_norm": float(norms.min()),
    "post_max_norm": float(norms.max()),
    "base_mean_norm": float(base_norms.mean()),
    "base_std_norm": float(base_norms.std()),
    "new_mean_norm": float(new_norms.mean()),
    "new_std_norm": float(new_norms.std()),
    "isotropy_mean_pairwise_cosine": float(mean_pairwise),
    "isotropy_sample_size": int(sample_size),
    "final_train_loss": float(losses[-1]) if train_data else None,
    "final_eval_loss": float(v_losses[-1]) if val_data else None,
    "best_eval_loss": float(min(v_losses)) if val_data else None,
}

summary_path = WAKE2VEC_ROOT / "p1_TinyLlama_summary.json"
summary_path.write_text(json.dumps(report, indent=2))
print(f"\n[SUMMARY] Saved to {summary_path}")
print(json.dumps(report, indent=2))
