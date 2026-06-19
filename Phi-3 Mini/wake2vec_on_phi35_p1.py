# -*- coding: utf-8 -*-
"""Wake2Vec Phase 1: Phi-3.5-mini-instruct Embedding-Only Fine-Tune

# Wake2Vec P1: Phi-3.5-mini-instruct with Gradient Masking

**Model:** microsoft/Phi-3.5-mini-instruct (4-bit quantized)
**Hardware:** Google Colab T4 GPU (2026.06)
**Training data:** Finnegans Wake corpus + Wake lexicon + Wake morpheme groups (JSONL)

## Overview

Phi-3.5-mini-instruct is the fourth datapoint in the smaller-model paradox
investigation. 32K vocab (matches TinyLlama and Mistral cohort), 3.8B
parameters (matches Llama 3.2-3B for the single-variable comparison),
hidden_dim 3072 (identical to 3B), 32 layers.

Two methodological notes:
1. Phi-3.5 is **instruct-tuned**. Microsoft does not publicly release a
   separate base variant so this is a deviation from the base-model
   convention used for every other model in the lineup (Llama, Mistral,
   Qwen, TinyLlama base variants). The textbook-vs-internet comparison
   has an instruct-tuned-vs-base confound built into it.
2. Init: **spherical 1.5x base_radius**. Matches the 32K cohort
   precedent (TinyLlama, Mistral) and the architectural-comparison
   partner (Llama 3.2-3B). Holds init strategy constant so any
   quality difference is attributable to vocab + training-data axes.

## Training data

Three sources concatenated:
- FW_TEXT: full Finnegans Wake corpus
- WAKE_LEXICON: hand-annotated Wake vocabulary
- WAKE_GROUPS_JSONL: morpheme-grouped Wake examples (258 groups).
  Used here as supplementary co-occurrence text. The model sees
  morphologically-related words in close proximity, which gives an
  implicit morpheme-grouping signal under standard LM loss (no
  auxiliary objective in P1).

## Colab 2026.06 compatibility

- Triton shim for bnb / triton 3.x
- eval_strategy (not evaluation_strategy) for transformers 5.0.0
- trust_remote_code=True for Phi-3.5 (harmless on newer transformers)

-------------------------------------------------------------

## envi
"""

import os
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import types, sys
_fake = types.ModuleType("triton.ops")
_fake.matmul_perf_model = types.ModuleType("triton.ops.matmul_perf_model")
sys.modules.setdefault("triton.ops", _fake)
sys.modules.setdefault("triton.ops.matmul_perf_model", _fake.matmul_perf_model)

# Colab 2026.06 ships bnb, scikit-learn, scipy pre-installed.
# subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
#                        "bitsandbytes", "scikit-learn", "scipy"])

import torch, gc
print("envi")
print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")
try:
    import bitsandbytes as bnb_lib
    print(f"bitsandbytes: {bnb_lib.__version__}")
except ImportError:
    print("bitsandbytes: NOT INSTALLED")
import transformers, accelerate, peft
print(f"transformers: {transformers.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"peft: {peft.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("=" * 60)

torch.cuda.empty_cache()
gc.collect()

from google.colab import drive
drive.mount('/content/drive')

from huggingface_hub import login
login()

"""## config"""

import math
from pathlib import Path

# model
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
BASE_VOCAB = 32064  # Phi-3.5-mini vocab size (Llama-derivative SentencePiece)

# data
FW_TEXT = "/content/FW_TEXT.txt"
WAKE_LEXICON = "/content/wake_lexicon.txt"
WAKE_GROUPS_JSONL = "/content/wake_embedding_groups.jsonl"

# outputs
RUN_DIR = Path("/content/drive/MyDrive/wake2vec_phi35_p1")
LOCAL_RUN = Path("/content/runs/wake2vec_phi35_p1")
SENTRY = RUN_DIR / "sentry_backups"
EMB_SNAPS = RUN_DIR / "emb_snaps"
FULL_CHECKPOINTS = RUN_DIR / "full_checkpoints"

for d in [RUN_DIR, LOCAL_RUN, SENTRY, EMB_SNAPS, FULL_CHECKPOINTS]:
    d.mkdir(parents=True, exist_ok=True)

# training hyperparams
MAX_STEPS = 3000
LR = 2e-4
WARMUP_STEPS = max(20, MAX_STEPS // 20)
WEIGHT_DECAY = 0.0
BATCH_SIZE = 1
GRAD_ACCUM = 16
SEQ_LEN = 512                
SAVE_STEPS = 50              
LOG_STEPS = 50
EVAL_STEPS = 100             
EMB_SNAP_STEPS = 50

# resume
RESUME_FROM = None
# RESUME_FROM = SENTRY / "checkpoint-200"

print("wake2vec P1, Phi-3.5-mini-instruct config")
print(f"  Model: {MODEL_NAME}")
print(f"  Base vocab: {BASE_VOCAB} (32K cohort, full Wake injection)")
print(f"  Output: {RUN_DIR}")
print(f"  Steps: {MAX_STEPS}")
print(f"  LR: {LR}")
print(f"  Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Seq len: {SEQ_LEN}")
print(f"  Resume: {RESUME_FROM}")

"""## tokenizer & Wake vocab"""

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"  Base vocab: {len(tok)}")

# Wake lexicon
print(f"getting Wake lexicon from {WAKE_LEXICON}...")
with open(WAKE_LEXICON, 'r', encoding='utf-8') as f:
    wake_tokens = [line.strip() for line in f if line.strip()]

# only add tokens not already in vocab
missing = [t for t in wake_tokens if tok.convert_tokens_to_ids(t) == tok.unk_token_id]
num_added = tok.add_tokens(missing, special_tokens=False)

print(f"  Wake tokens in lexicon: {len(wake_tokens)}")
print(f"  New tokens added: {num_added}")
print(f"  Already in vocab: {len(wake_tokens) - num_added}")
print(f"  Final vocab size: {len(tok)}")

"""## dataset (FW + Lexicon + JSONL morpheme groups)"""

import json
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

# FW text & Wake lexicon
if not os.path.exists(FW_TEXT):
    raise FileNotFoundError(f"FW text not found: {FW_TEXT}")

with open(FW_TEXT, 'r', encoding='utf-8') as f:
    fw_text = f.read()

with open(WAKE_LEXICON, 'r', encoding='utf-8') as f:
    lex_text = f.read()

# JSONL morpheme groups
print(f"getting morpheme groups from {WAKE_GROUPS_JSONL}")
if not os.path.exists(WAKE_GROUPS_JSONL):
    raise FileNotFoundError(f"Wake groups JSONL not found: {WAKE_GROUPS_JSONL}")

groups_text_chunks = []
n_groups = 0
n_examples_total = 0
with open(WAKE_GROUPS_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        group = json.loads(line)
        morpheme = group.get("morpheme", "?")
        examples = group.get("examples", [])
        if not examples:
            continue
        n_groups += 1
        n_examples_total += len(examples)
        # render as header + space-separated examples
        chunk = f"[morpheme={morpheme}] " + " ".join(examples)
        groups_text_chunks.append(chunk)
      
groups_text = "\n".join(groups_text_chunks)
print(f"  Morpheme groups: {n_groups}")
print(f"  Total examples across groups: {n_examples_total}")
print(f"  Groups text: {len(groups_text)} chars")

# merge
combined_text = fw_text + "\n" + lex_text + "\n" + groups_text
print(f"  FW text: {len(fw_text)} chars")
print(f"  Lexicon: {len(lex_text)} chars")
print(f"  Groups:  {len(groups_text)} chars")
print(f"  Combined: {len(combined_text)} chars")

# tokenise
ids = tok(combined_text, add_special_tokens=False)["input_ids"]
print(f"  Total tokens: {len(ids)}")

# non-overlapping blocks 
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
from peft import LoraConfig, get_peft_model

set_seed(42)

# 4-bit quant (NF4)
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
    trust_remote_code=True,
    max_memory={0: "13GB", "cpu": "30GB"}
)

model.config.use_cache = False
model.config.attn_implementation = "eager"
model.config.tie_word_embeddings = True
if hasattr(model, "tie_weights"):
    model.tie_weights()

print(f"  VRAM after load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# frozen LoRA r=1 (PEFT compat shim for quantized training).
peft_cfg = LoraConfig(
    r=1, lora_alpha=1, lora_dropout=0.0,
    target_modules=["qkv_proj"], bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)

for n, p in model.named_parameters():
    p.requires_grad = False

"""## wake embed injection (spherical 1.5x init)"""

old_vocab = model.get_input_embeddings().weight.shape[0]
model.resize_token_embeddings(len(tok))
wte = model.get_input_embeddings()

if hasattr(model, "lm_head"):
    model.lm_head.weight = wte.weight

with torch.no_grad():
    base = wte.weight[:old_vocab]
    dim = base.shape[1]
    std = base.std().item()
    base_radius = std * math.sqrt(dim)
    target_radius = 1.5 * base_radius  # 32K cohort match (TinyLlama, Mistral, 3B)
    if num_added > 0:
        new = torch.randn((num_added, dim), device=wte.weight.device)
        new = new / (new.norm(dim=1, keepdim=True) + 1e-8) * target_radius
        wte.weight.data[old_vocab:old_vocab + num_added] = new

print(f"  Vocab: {old_vocab} -> {len(tok)} (+{num_added} Wake tokens)")
print(f"  Base radius: {base_radius:.4f}")
print(f"  Spherical init radius: {target_radius:.4f} (1.5x base)")

wte.weight.requires_grad = True

new_rows = torch.arange(old_vocab, old_vocab + num_added, device=wte.weight.device) if num_added > 0 else None
base_rows = torch.arange(0, old_vocab, device=wte.weight.device)

def mask_grad(grad):
    if grad is None or new_rows is None:
        return grad
    grad[base_rows] = 0
    return grad

wte.weight.register_hook(mask_grad)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {trainable:,}")
print(f"  VRAM after setup: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

"""## pre-training embed snapshot"""

E_pre = wte.weight.detach().cpu().clone()
torch.save(E_pre, RUN_DIR / "embeddings_pre.pt")
print(f"  Pre-training snapshot saved: {E_pre.shape}")

"""## callbacks"""

import shutil, time
from transformers import TrainerCallback

def has_weights(ck):
    return (ck / "adapter_model.safetensors").exists() or (ck / "pytorch_model.bin").exists()

class EmbeddingSnapshot(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % EMB_SNAP_STEPS == 0:
            try:
                torch.save(wte.weight.detach().cpu(),
                           EMB_SNAPS / f"emb_step{state.global_step:04d}.pt")
                print(f"[EMB] Step {state.global_step}: saved")
            except Exception as e:
                print(f"[EMB] {e}")

class FullCheckpoint(TrainerCallback):
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
            torch.save({'global_step': step, 'best_metric': state.best_metric,
                        'epoch': state.epoch}, full_ck / "training_state.pt")
            print(f"[FULL] Step {step}: saved")
        except Exception as e:
            print(f"[FULL] {e}")

class SentryMirror(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        try:
            cks = sorted(LOCAL_RUN.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1]), reverse=True)
            if not cks:
                return
            ck = cks[0]
            if not has_weights(ck):
                return
            dst = SENTRY / ck.name
            if dst.exists() and has_weights(dst):
                return
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(ck, dst)
            print(f"[SENTRY] {ck.name}: mirrored")
        except Exception as e:
            print(f"[SENTRY] {e}")

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
            if gap > 3.0:
                print(f"[WARN] train/eval gap: {gap:.2f}")

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
                print(f"[{state.global_step:4d}] {avg:.1f}s/step")
        self.last_time = now

"""## trainer + train"""

from transformers import TrainingArguments, Trainer

class EmbOnlyTrainer(Trainer):

    def create_optimizer(self):
        from torch.optim import AdamW
        if not hasattr(self, "optimizer") or self.optimizer is None:
            self.optimizer = AdamW(
                [{"params": [wte.weight], "lr": LR, "weight_decay": 0.0}],
                betas=(0.9, 0.999), eps=1e-8
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = model(**inputs, use_cache=False)
        loss = out.loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("NaN/Inf loss detected")
        return (loss, out) if return_outputs else loss

args = TrainingArguments(
    output_dir=str(LOCAL_RUN),
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="cosine",
    weight_decay=WEIGHT_DECAY,
    fp16=False,
    bf16=True,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=6,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    report_to="none",
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)

trainer = EmbOnlyTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=None,
    callbacks=[EmbeddingSnapshot(), FullCheckpoint(), SentryMirror(),
               LossMonitor(), StepTimer()],
)

print("wake2vec P1 on Phi-3.5-mini-instruct embed only")
print(f"  Train: {len(train_ds)} blocks | Val: {len(val_ds)} blocks")
print(f"  Steps: {MAX_STEPS} | Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  VRAM before training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# train
t0 = time.time()
if RESUME_FROM is not None:
    local_ckpt = LOCAL_RUN / RESUME_FROM.name
    if not local_ckpt.exists():
        shutil.copytree(RESUME_FROM, local_ckpt)
    print(f"[RESUME] {RESUME_FROM.name}")
    trainer.train(resume_from_checkpoint=str(local_ckpt))
else:
    trainer.train()
elapsed = (time.time() - t0) / 60
print(f"\nTRAINING COMPLETE ({elapsed:.1f} minutes)")

# final model save
final_dir = RUN_DIR / "final"
final_dir.mkdir(exist_ok=True)
model.save_pretrained(str(final_dir))
tok.save_pretrained(str(final_dir))
final_emb = wte.weight.detach().cpu()
torch.save(final_emb, final_dir / "embeddings.pt")
print(f"Final model saved to {final_dir}")

"""## loss curve"""

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
        plt.xlabel('Step'); plt.ylabel('Loss')
        plt.title('Wake2Vec P1: Phi-3.5-mini-instruct Loss Curve')
        plt.legend(); plt.grid(True, alpha=0.3)
        plot_path = RUN_DIR / "p1_phi35_loss_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {plot_path}")
        plt.show()
        print(f"\nFinal train loss: {losses[-1]:.4f}")
        if val_data:
            print(f"Final val loss: {v_losses[-1]:.4f}")
            print(f"Best val loss: {min(v_losses):.4f}")

"""## embedding analysis"""

import numpy as np
from numpy.linalg import norm as l2
from scipy import stats
from sklearn.decomposition import PCA

E_post = final_emb.numpy()
vocab_size, dim = E_post.shape
num_new_tokens = vocab_size - BASE_VOCAB
E_base = E_post[:BASE_VOCAB]
E_new = E_post[BASE_VOCAB:]

pre_path = RUN_DIR / "embeddings_pre.pt"
has_pre = pre_path.exists()
if has_pre:
    E_pre_all = torch.load(pre_path, map_location="cpu").numpy()
    E_pre_base = E_pre_all[:BASE_VOCAB]
    E_pre_new = E_pre_all[BASE_VOCAB:]

# norms
norms = l2(E_post, axis=1)
base_norms = norms[:BASE_VOCAB]
new_norms = norms[BASE_VOCAB:]
t_stat, t_pval = stats.ttest_ind(base_norms, new_norms, equal_var=False)
u_stat, u_pval = stats.mannwhitneyu(base_norms, new_norms, alternative='two-sided')
pooled_std = np.sqrt((base_norms.std()**2 + new_norms.std()**2) / 2)
cohens_d = (base_norms.mean() - new_norms.mean()) / pooled_std

print("norm analysis")
print(f"  Global  -- mean: {norms.mean():.4f}, std: {norms.std():.4f}")
print(f"  Base    -- mean: {base_norms.mean():.4f}, std: {base_norms.std():.4f} (n={BASE_VOCAB})")
print(f"  New     -- mean: {new_norms.mean():.4f}, std: {new_norms.std():.4f} (n={num_new_tokens})")
print(f"  Welch t: t={t_stat:.4f}, p={t_pval:.2e} | Cohen's d: {cohens_d:.4f}")

# isotropy
def compute_isotropy(embeddings):
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
    return Z.min() / Z.max(), (cos_mat.sum()) / (n * (n - 1)), n

iso_all, cos_all, n_all = compute_isotropy(E_post)
iso_base, cos_base, n_base = compute_isotropy(E_base)
iso_new, cos_new, n_new = compute_isotropy(E_new)

print("isotropy")
print(f"  All  -- {iso_all:.6f} (mean_cos: {cos_all:.4f})")
print(f"  Base -- {iso_base:.6f} (mean_cos: {cos_base:.4f})")
print(f"  New  -- {iso_new:.6f} (mean_cos: {cos_new:.4f})")

# drift
if has_pre:
    def safe_cos(a, b):
        na = l2(a, axis=1, keepdims=True); na = np.where(na < 1e-12, 1e-12, na)
        nb = l2(b, axis=1, keepdims=True); nb = np.where(nb < 1e-12, 1e-12, nb)
        return np.sum((a / na) * (b / nb), axis=1)

    drift_cos = safe_cos(E_pre_base, E_base)
    wake_drift_cos = safe_cos(E_pre_new, E_new)
    wake_drift_l2 = l2(E_new - E_pre_new, axis=1)

    print("drift (pre to post)")
    print(f"  Base cosine: {drift_cos.mean():.6f} (should be ~1.0)")
    print(f"  Wake cosine: {wake_drift_cos.mean():.6f} +/- {wake_drift_cos.std():.6f}")
    print(f"  Wake L2:     {wake_drift_l2.mean():.4f} +/- {wake_drift_l2.std():.4f}")

    wake_drift_order = np.argsort(wake_drift_cos)
    print(f"\n  Top 20 most-changed Wake tokens:")
    for rank, idx in enumerate(wake_drift_order[:20]):
        global_idx = BASE_VOCAB + idx
        token_str = tok.convert_ids_to_tokens(int(global_idx))
        print(f"    {rank+1:2d}. {token_str!r:25s} cos={wake_drift_cos[idx]:.6f}")

# nearest neighbours 
print("nearest neighbours")

all_norms_safe = l2(E_post, axis=1, keepdims=True)
all_norms_safe = np.where(all_norms_safe < 1e-12, 1e-12, all_norms_safe)
E_unit = E_post / all_norms_safe
E_base_unit = E_unit[:BASE_VOCAB]

sample_ids = list(range(BASE_VOCAB, BASE_VOCAB + 10))
if num_new_tokens > 100:
    sample_ids += list(range(BASE_VOCAB + num_new_tokens // 2, BASE_VOCAB + num_new_tokens // 2 + 5))
if num_new_tokens > 1000:
    sample_ids += list(range(BASE_VOCAB + num_new_tokens - 5, BASE_VOCAB + num_new_tokens))

for wid in sample_ids:
    wt = tok.convert_ids_to_tokens(wid)
    sims = (E_unit[wid:wid+1] @ E_base_unit.T).squeeze()
    top5 = np.argsort(sims)[-5:][::-1]
    nb = ", ".join(f"{tok.convert_ids_to_tokens(int(i))!r}({sims[i]:.3f})" for i in top5)
    print(f"  {wt!r:25s} -> {nb}")

# PCA
print("intrinsic dimensionality")

n_comp = min(100, dim, BASE_VOCAB, num_new_tokens)
pca_base = PCA(n_components=n_comp).fit(E_base)
pca_new = PCA(n_components=n_comp).fit(E_new)
cumvar_base = np.cumsum(pca_base.explained_variance_ratio_)
cumvar_new = np.cumsum(pca_new.explained_variance_ratio_)

print(f"  Base -- 90% in {np.searchsorted(cumvar_base, 0.90)+1} PCs, 95% in {np.searchsorted(cumvar_base, 0.95)+1}")
print(f"  New  -- 90% in {np.searchsorted(cumvar_new, 0.90)+1} PCs, 95% in {np.searchsorted(cumvar_new, 0.95)+1}")

# pairwise cosine 
rng = np.random.default_rng(42)
def sample_cos(E1, E2, n=2000):
    i1 = rng.choice(len(E1), size=min(n, len(E1)), replace=False)
    i2 = rng.choice(len(E2), size=min(n, len(E2)), replace=False)
    s1, s2 = E1[i1], E2[i2]
    n1 = l2(s1, axis=1, keepdims=True); n1 = np.where(n1<1e-12, 1e-12, n1)
    n2 = l2(s2, axis=1, keepdims=True); n2 = np.where(n2<1e-12, 1e-12, n2)
    c = (s1/n1) @ (s2/n2).T
    return c[np.triu_indices_from(c, k=1)] if E1 is E2 else c.ravel()

cos_bb = sample_cos(E_base, E_base)
cos_nn = sample_cos(E_new, E_new)
cos_bn = sample_cos(E_base, E_new)

print("pairwise cosine")
print(f"  (base,base) mean: {cos_bb.mean():.4f}")
print(f"  (new,new)   mean: {cos_nn.mean():.4f}")
print(f"  (base,new)  mean: {cos_bn.mean():.4f}")

# combined plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Wake2Vec P1 Phi-3.5-mini-instruct -- Embedding Analysis", fontsize=14, fontweight="bold")

axes[0,0].hist(base_norms, bins=50, alpha=0.5, label="Base", density=True)
axes[0,0].hist(new_norms, bins=50, alpha=0.5, label="New", density=True)
axes[0,0].set_title("Norm Distribution"); axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.3)

axes[0,1].hist(cos_bb, bins=80, alpha=0.4, label="base-base", density=True)
axes[0,1].hist(cos_nn, bins=80, alpha=0.4, label="new-new", density=True)
axes[0,1].hist(cos_bn, bins=80, alpha=0.4, label="base-new", density=True)
axes[0,1].set_title("Pairwise Cosine"); axes[0,1].legend(fontsize=7); axes[0,1].grid(True, alpha=0.3)

axes[0,2].plot(range(1, n_comp+1), cumvar_base, 'b-', label="Base")
axes[0,2].plot(range(1, n_comp+1), cumvar_new, 'r-', label="New")
axes[0,2].axhline(0.90, ls='--', c='gray', alpha=0.5)
axes[0,2].set_title("Intrinsic Dim (PCA)"); axes[0,2].legend(fontsize=8); axes[0,2].grid(True, alpha=0.3)

if has_pre:
    axes[1,0].hist(drift_cos, bins=80, alpha=0.5, color="coral", label="Base")
    axes[1,0].hist(wake_drift_cos, bins=80, alpha=0.5, color="steelblue", label="Wake")
    axes[1,0].set_title("Embedding Drift"); axes[1,0].legend(fontsize=8)
else:
    axes[1,0].text(0.5, 0.5, "No pre-training\nsnapshot", ha='center', va='center',
                   transform=axes[1,0].transAxes, fontsize=12, color='gray')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].scatter(range(BASE_VOCAB), base_norms, s=0.1, alpha=0.3, label="Base", c="blue")
axes[1,1].scatter(range(BASE_VOCAB, vocab_size), new_norms, s=0.1, alpha=0.3, label="New", c="red")
axes[1,1].set_title("Norm by Index"); axes[1,1].legend(fontsize=8, markerscale=10); axes[1,1].grid(True, alpha=0.3)

axes[1,2].bar(range(1, 21), pca_base.explained_variance_ratio_[:20], alpha=0.5, label="Base")
axes[1,2].bar(range(1, 21), pca_new.explained_variance_ratio_[:20], alpha=0.5, label="New")
axes[1,2].set_title("Top-20 Eigenspectrum"); axes[1,2].legend(fontsize=8); axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RUN_DIR / "p1_phi35_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# sum json
report = {
    "model": MODEL_NAME, "phase": "P1_embedding_only",
    "vocab_size": int(vocab_size), "dim": int(dim),
    "base_vocab": int(BASE_VOCAB), "new_tokens": int(num_new_tokens),
    "training_data": ["FW_TEXT.txt", "wake_lexicon.txt", "wake_embedding_groups.jsonl"],
    "init_strategy": "spherical_1.5x_base_radius",
    "instruct_tuned": True,
    "hyperparameters": {"lr": LR, "max_steps": MAX_STEPS, "batch_size": BATCH_SIZE,
                        "grad_accum": GRAD_ACCUM, "seq_len": SEQ_LEN},
    "norms": {"base_mean": float(base_norms.mean()), "new_mean": float(new_norms.mean()),
              "cohens_d": float(cohens_d)},
    "isotropy": {"all": float(iso_all), "base": float(iso_base), "new": float(iso_new)},
    "loss": {"final_train": float(losses[-1]) if train_data else None,
             "final_eval": float(v_losses[-1]) if val_data else None},
}
if has_pre:
    report["drift"] = {"base_cosine_mean": float(drift_cos.mean()),
                       "wake_cosine_mean": float(wake_drift_cos.mean())}

summary_path = RUN_DIR / "p1_phi35_summary.json"
summary_path.write_text(json.dumps(report, indent=2))
print(f"\n[SUMMARY] {summary_path}")

"""## generation / temperature sweep"""

model.eval()
model.config.use_cache = True

def generate_wake(prompt, max_new_tokens=256, temperature=0.9, top_p=0.92,
                  top_k=50, repetition_penalty=1.15, num_return_sequences=1):
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences, do_sample=True,
            pad_token_id=tok.pad_token_id)
    print(f"-- temp={temperature} | top_p={top_p} | top_k={top_k} --")
    for i, seq in enumerate(outputs):
        gen = tok.decode(seq[prompt_len:], skip_special_tokens=True)
        if num_return_sequences > 1:
            print(f"\n[{i+1}]")
        print(gen)


def temperature_sweep(prompt, temps=[0.5, 0.7, 0.9, 1.0, 1.2], **kwargs):
    print(f"PROMPT: {prompt}\n")
    for t in temps:
        generate_wake(prompt, temperature=t, **kwargs)
        print()


# prompt
generate_wake("riverrun, past Eve and Adam's,")
