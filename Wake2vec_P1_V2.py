# -*- coding: utf-8 -*-
"""
Wake2Vec Phase 1: Embedding-Only Fine-Tuning (v2 - With Validation)

Model: TinyLlama-1.1B
Steps: 2000
Hardware: Google Colab T4 GPU

Changes from v1:
- 90/10 train/val split from start
- 2000 steps (up from 1300)
- Logging every 100 steps
- Full checkpoint saves with embeddings
- New output path: wake2vecP1_v2

Colab setup (run first, then restart runtime):
    pip uninstall -y torch torchvision torchaudio triton bitsandbytes transformers accelerate peft jax jaxlib flax
    pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    pip install triton==3.1.0 bitsandbytes==0.43.3 transformers==4.45.2 accelerate==0.34.2 peft==0.13.2 scikit-learn datasets
"""


# Environment
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
print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Drive
if os.path.exists('/content/drive'):
    shutil.rmtree('/content/drive')

from google.colab import drive
drive.mount('/content/drive')


# Config
WAKE2VEC_ROOT = Path("/content/drive/MyDrive/wake2vecP1_v2")
LOCAL_RUN = Path("/content/runs/wake2vecP1_v2")
SENTRY = WAKE2VEC_ROOT / "sentry_backups"
EMB_SNAPS = WAKE2VEC_ROOT / "emb_snaps"
FULL_CHECKPOINTS = WAKE2VEC_ROOT / "full_checkpoints"
RESUME_FROM = SENTRY / "checkpoint-600"

for d in [WAKE2VEC_ROOT, LOCAL_RUN, SENTRY, EMB_SNAPS, FULL_CHECKPOINTS]:
    d.mkdir(parents=True, exist_ok=True)

# Training config
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FW_TEXT = "/content/FW_TEXT.txt"

MAX_STEPS = 2000
LR = 5e-4
WARMUP_RATIO = 0.05
BATCH_SIZE = 1
GRAD_ACCUM = 16
SEQ_LEN = 256
SAVE_STEPS = 100
LOG_STEPS = 100
EVAL_STEPS = 200

print("=" * 60)
print("WAKE2VEC P1 v2 CONFIGURATION")
print("=" * 60)
print(f"Output: {WAKE2VEC_ROOT}")
print(f"Steps: {MAX_STEPS}")
print(f"LR: {LR}")
print(f"Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"Save: every {SAVE_STEPS} steps")
print(f"Log: every {LOG_STEPS} steps")
print(f"Eval: every {EVAL_STEPS} steps")
print("=" * 60)

# Tokenizer and Extend with Wake Vocab
from transformers import AutoTokenizer

# Load BASE tokenizer 
print("Loading base TinyLlama tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

BASE_VOCAB = len(tok)
print(f"Base vocab: {BASE_VOCAB}")

# Load Wake lexicon and extend tokenizer
WAKE_LEXICON = "/content/wake_lexicon.txt"
print(f"Loading Wake lexicon from {WAKE_LEXICON}...")

with open(WAKE_LEXICON, 'r', encoding='utf-8') as f:
    wake_tokens = [line.strip() for line in f if line.strip()]

num_added = tok.add_tokens(wake_tokens)
print(f"Wake tokens added: {num_added}")
print(f"New vocab size: {len(tok)}")

# Dataset with Train/Val Split
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
    raise FileNotFoundError(f"Corpus not found: {FW_TEXT}")

with open(FW_TEXT, 'r', encoding='utf-8') as f:
    text = f.read()

ids = tok(text, add_special_tokens=False)["input_ids"]
print(f"Total tokens: {len(ids)}")

blocks = []
for i in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN):
    chunk = ids[i:i + SEQ_LEN]
    if len(chunk) == SEQ_LEN:
        blocks.append(chunk)

print(f"Total blocks: {len(blocks)}")

# 90/10 split
train_blocks, val_blocks = train_test_split(blocks, test_size=0.10, random_state=42)
train_ds = BlockDataset(train_blocks, tok, SEQ_LEN)
val_ds = BlockDataset(val_blocks, tok, SEQ_LEN)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# Model and Initialize Wake Embeddings
from transformers import AutoModelForCausalLM

print("Loading fresh base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=None,
    low_cpu_mem_usage=True,
)

model.to("cuda")
model.config.use_cache = False
model.config.attn_implementation = "eager"

# Resize embeds
print(f"Resizing embeddings: {BASE_VOCAB} -> {len(tok)}...")
model.resize_token_embeddings(len(tok))

# Mean initialization for new Wake tokens
print("Initializing Wake token embeddings (mean of base vocab)...")
with torch.no_grad():
    emb = model.get_input_embeddings()
    old_embeddings = emb.weight[:BASE_VOCAB]
    avg_embedding = old_embeddings.mean(dim=0)
    emb.weight[BASE_VOCAB:] = avg_embedding

print(f"Embeddings shape: {emb.weight.shape}")

# Freeze all except embeds
for p in model.parameters():
    p.requires_grad = False

emb.weight.requires_grad = True

# Tie input/output embeddings
with torch.no_grad():
    model.get_output_embeddings().weight = emb.weight

model.train()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Tied embeddings: {emb.weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()}")
print(f"Trainable params: {trainable:,}")

# callbacks
from transformers import TrainingArguments, Trainer, TrainerCallback

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

# trainer 
args = TrainingArguments(
    output_dir=str(LOCAL_RUN),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=MAX_STEPS,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    optim="adafactor",
    logging_steps=LOG_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    load_best_model_at_end=False,
    gradient_checkpointing=True,
    fp16=False,
    bf16=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    report_to=["none"],
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
    ],
)

print(f"Embedding snapshots: {EMB_SNAPS} (every 50 steps)")
print(f"Full checkpoints: {FULL_CHECKPOINTS} (every {SAVE_STEPS} steps)")
print(f"Sentry backups: {SENTRY} (every {SAVE_STEPS} steps)")

# train
print("=" * 80)
print("WAKE2VEC P1 v2: EMBEDDING-ONLY FINE-TUNE WITH VALIDATION")
print("=" * 80)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
print(f"Steps: {MAX_STEPS}")
print(f"Trainable: {trainable:,} params")
print("=" * 80)

t0 = time.time()
trainer.train(resume_from_checkpoint=str(RESUME_FROM))
elapsed = (time.time() - t0) / 60

print("=" * 80)
print("TRAINING COMPLETE")
print(f"Time: {elapsed:.1f} minutes")
print("=" * 80)

# Save Final
final_dir = WAKE2VEC_ROOT / "final"
final_dir.mkdir(exist_ok=True)

model.save_pretrained(str(final_dir))
tok.save_pretrained(str(final_dir))

final_emb = model.get_input_embeddings().weight.detach().cpu()
torch.save(final_emb, final_dir / "embeddings.pt")

os.sync()
print(f"Final model saved to {final_dir}")

# TinyLlama P1 Post-Training Analysis - museum of failures 
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.linalg import norm as l2

from transformers import AutoTokenizer

# Paths
WAKE2VEC_ROOT = Path("/content/drive/MyDrive/wake2vecP1_v2")
FINAL_DIR = WAKE2VEC_ROOT / "final"
SENTRY = WAKE2VEC_ROOT / "sentry_backups"

print("[TINY P1 EVAL] ROOT:", WAKE2VEC_ROOT)
print("[TINY P1 EVAL] FINAL_DIR:", FINAL_DIR)

if not FINAL_DIR.exists():
    raise FileNotFoundError(f"Final dir not found at {FINAL_DIR} – check path / Drive mount.")

# Load embeds from final model

emb_path = FINAL_DIR / "embeddings.pt"
if not emb_path.exists():
    raise FileNotFoundError(f"embeddings.pt not found at {emb_path}")

print(f"[EMB] Loading embeddings from {emb_path}")
E_post = torch.load(emb_path, map_location="cpu").numpy()
vocab_size, dim = E_post.shape
print(f"[EMB] Loaded embeddings: {E_post.shape} (vocab_size, dim)")

#  - base vocab: 32000
#  - resized to 76500 total tokens
BASE_VOCAB = 32000
if BASE_VOCAB > vocab_size:
    raise ValueError(f"BASE_VOCAB ({BASE_VOCAB}) > vocab_size ({vocab_size}) – adjust BASE_VOCAB.")

num_new = vocab_size - BASE_VOCAB
print(f"[EMB] Base vocab: {BASE_VOCAB}, New Wake tokens: {num_new}")

# Norm statistics

norms = l2(E_post, axis=1)

base_norms = norms[:BASE_VOCAB]
new_norms = norms[BASE_VOCAB:]

print("\n[NORMS – GLOBAL]")
print(f"Mean: {norms.mean():.4f}, Std: {norms.std():.4f}, Min: {norms.min():.4f}, Max: {norms.max():.4f}")

print("[NORMS – BASE VS NEW]")
print(f"Base tokens: mean={base_norms.mean():.4f}, std={base_norms.std():.4f}, n={len(base_norms)}")
print(f"New  tokens: mean={new_norms.mean():.4f}, std={new_norms.std():.4f}, n={len(new_norms)}")

# Isotropy estimate (sampled mean pairwise cosine)

print("\n[ISOTROPY] Computing sampled mean pairwise cosine...")

E_unit = E_post / (norms[:, None] + 1e-12)

rng = np.random.default_rng(0)
sample_size = min(5000, vocab_size)
idx = rng.choice(vocab_size, size=sample_size, replace=False)
E_sample = E_unit[idx]

cos = E_sample @ E_sample.T
n = cos.shape[0]
mean_pairwise = (cos.sum() - np.trace(cos)) / (n * n - n)

print(f"[ISOTROPY] Sample size={sample_size}, mean pairwise cosine={mean_pairwise:.4f}")


# Loss curves from Sentry 
train_steps, train_losses = [], []
eval_steps, eval_losses = [], []
final_train_loss = None
final_eval_loss = None

# Prefer checkpoint-2000 if it exists
ck2000 = SENTRY / "checkpoint-2000"
state_file = ck2000 / "trainer_state.json"

if ck2000.exists() and state_file.exists():
    print(f"\n[LOG] Found trainer_state.json at {state_file}")
    s = json.loads(state_file.read_text())
    logs = s.get("log_history", [])
    for d in logs:
        if "loss" in d and "step" in d:
            train_steps.append(d["step"])
            train_losses.append(float(d["loss"]))
            final_train_loss = float(d["loss"])
        if "eval_loss" in d and "step" in d:
            eval_steps.append(d["step"])
            eval_losses.append(float(d["eval_loss"]))
            final_eval_loss = float(d["eval_loss"])
    print(f"[LOG] Train points: {len(train_steps)}, Eval points: {len(eval_steps)}")
else:
    print(f"\n[LOG] No trainer_state.json found at {state_file}; skipping loss curve.")

# Loss plot
if train_steps:
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, "o-", alpha=0.7, label="Train loss", markersize=4)
    if eval_steps:
        plt.plot(eval_steps, eval_losses, "s-", alpha=0.7, label="Eval loss", markersize=4)
    plt.title("TinyLlama P1 v2: Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    loss_plot_path = WAKE2VEC_ROOT / "tiny_p1_loss.png"
    plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Loss curve saved to {loss_plot_path}")
    plt.show()

print("\n[FINAL LOSSES]")
print("Final train loss:", final_train_loss)
print("Final eval loss:", final_eval_loss)


# Norm distribution plots
# Global hist
plt.figure(figsize=(10, 6))
plt.hist(norms, bins=50, alpha=0.7, edgecolor="black")
plt.axvline(norms.mean(), linestyle="--", label=f"Mean: {norms.mean():.2f}")
plt.title("TinyLlama P1 v2: Embedding Norm Distribution (All Tokens)")
plt.xlabel("L2 norm")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.legend()
global_norm_plot = WAKE2VEC_ROOT / "tiny_p1_norms_global.png"
plt.savefig(global_norm_plot, dpi=150, bbox_inches="tight")
print(f"[PLOT] Saved global norm hist to {global_norm_plot}")
plt.show()

# Base vs new hist
plt.figure(figsize=(10, 6))
plt.hist(base_norms, bins=50, alpha=0.5, label="Base tokens")
plt.hist(new_norms, bins=50, alpha=0.5, label="New Wake tokens")
plt.axvline(base_norms.mean(), linestyle="--", label=f"Base mean: {base_norms.mean():.2f}")
plt.axvline(new_norms.mean(), linestyle=":", label=f"New mean: {new_norms.mean():.2f}")
plt.title("TinyLlama P1 v2: Embedding Norms – Base vs New Tokens")
plt.xlabel("L2 norm")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.legend()
bv_norm_plot = WAKE2VEC_ROOT / "tiny_p1_norms_base_vs_new.png"
plt.savefig(bv_norm_plot, dpi=150, bbox_inches="tight")
print(f"[PLOT] Saved base vs new norm hist to {bv_norm_plot}")
plt.show()


# Nearest-neighbour inspection
print("\n[NN] Loading tokenizer from final dir for NN probes...")
tok = AutoTokenizer.from_pretrained(FINAL_DIR)

PROBES_PATH = WAKE2VEC_ROOT / "probes.txt"
if not PROBES_PATH.exists():
    print(f"[NN] No probes.txt found at {PROBES_PATH}, skipping NN probes.")
else:
    probe_words = [w.strip() for w in PROBES_PATH.read_text().splitlines() if w.strip()]
    print(f"[NN] Loaded {len(probe_words)} probe words from {PROBES_PATH}")

    def token_id_for_word(word: str):
        """
        Map a plain word to a single token id, if possible.
        We prepend a space to match Llama-style '▁word' behaviour.
        """
        tok_str = " " + word
        tid = tok.convert_tokens_to_ids(tok_str)
        if tid is None or tid < 0 or tid == tok.unk_token_id:
            return None
        return tid

    def nearest_by_id(tid: int, k: int = 15):
        """Nearest neighbours given a token id."""
        v = E_unit[tid]
        sims = E_unit @ v
        topk = np.argsort(-sims)[:k+1]  # include self
        out = []
        for i in topk:
            tok_str = tok.convert_ids_to_tokens(int(i))
            out.append((int(i), tok_str, float(sims[i])))
        return out

    rng = np.random.default_rng(0)
    probe_words_sample = probe_words  # or rng.choice(probe_words, size=20, replace=False)

    used = 0
    for w in probe_words_sample:
        tid = token_id_for_word(w)
        if tid is None:
            print(f"[NN] Skipping {w!r}: no single-token match")
            continue
        nns = nearest_by_id(tid, k=10)
        print(f"\n[NN] Probes for word {w!r} (token id={tid})")
        for i, tok_str, sim in nns:
            print(f"  {i:6d}  {tok_str!r}  cos={sim:.3f}")
        used += 1

    print(f"\n[NN] Completed probes for {used} / {len(probe_words_sample)} words.")

# random sample of new Wake tokens
if BASE_VOCAB < vocab_size:
    new_ids = np.arange(BASE_VOCAB, vocab_size)
    sample_new = rng.choice(new_ids, size=min(5, len(new_ids)), replace=False)
    for tid in sample_new:
        token_str = tok.convert_ids_to_tokens(int(tid))
        v = E_unit[tid]
        sims = E_unit @ v
        topk = np.argsort(-sims)[:10]
        nn_tokens = [tok.convert_ids_to_tokens(int(i)) for i in topk]
        print(f"\n[NN – NEW TOKEN] id={tid}, token={token_str!r}")
        print("  nearest:", nn_tokens)

# Tiny summary JSON
report = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "checkpoint": "final",
    "vocab_size": int(vocab_size),
    "dim": int(dim),
    "base_vocab": int(BASE_VOCAB),
    "new_tokens": int(num_new),
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
    "final_train_loss": float(final_train_loss) if final_train_loss is not None else None,
    "final_eval_loss": float(final_eval_loss) if final_eval_loss is not None else None,
}

summary_path = WAKE2VEC_ROOT / "tiny_p1_summary.json"
summary_path.write_text(json.dumps(report, indent=2))
print(f"\n[SUMMARY] Saved TinyLlama P1 summary to {summary_path}")
print(json.dumps(report, indent=2))
