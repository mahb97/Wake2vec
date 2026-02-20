# -*- coding: utf-8 -*-
"""wake2vec_on_qwen_2_5_14b.py

# Wake2Vec P1: Qwen2.5-14B Embedding-Only with Gradient Masking

**Model:** Qwen/Qwen2.5-14B (4-bit quantized body, fp32 embeddings)
**Hardware:** Google Colab T4 GPU (15GB VRAM) — living on the edge
**Training data:** Finnegans Wake corpus + Wake lexicon

The absolute unit. 14B params, 48 layers, 5120 hidden dim, 152K base vocab.
This is the biggest model that can theoretically fit on a free T4 at 4-bit.
VRAM budget is EXTREMELY tight (~12-13GB) so every byte counts.

The fun part: Qwen already has a 152K vocab, so a lot of Wake neologisms
might already be in there. We're adding 44,989 Wake tokens but many will
be dupes. Less embedding disruption than TinyLlama's 140% vocab expansion.

If this OOMs, try: seq_len 256->128, or batch 1->1 with grad_accum 16->32.
If it STILL OOMs: skill issue (on Colab's part, not yours).

────────────────────────────────────────────────────────────
reminder: the Wake never ends. it just loops back to the beginning.
riverrun.
"""

# import os
# os.kill(os.getpid(), 9)
# housekeeping 
!pip install bitsandbytes==0.45.0 --force-reinstall --no-cache-dir

# envi
# (pip install is a creative act)
import os
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "bitsandbytes==0.45.0", "scikit-learn", "scipy"])

import torch, gc
print("=" * 60)
print("ENVIRONMENT (pray for VRAM)")
print("=" * 60)
print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")
try:
    import bitsandbytes as bnb_lib
    print(f"bitsandbytes: {bnb_lib.__version__}")
except ImportError:
    print("bitsandbytes: NOT INSTALLED — we're cooked")
import transformers, accelerate, peft
print(f"transformers: {transformers.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"peft: {peft.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM: {vram_total:.2f} GB {'(gaslight GPU energy)' if vram_total < 16 else ''}")
print("=" * 60)

torch.cuda.empty_cache()
gc.collect()

from google.colab import drive
drive.mount('/content/drive')

from huggingface_hub import login
login()


# config
# (do the math and hope)
import math
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-14B"   
BASE_VOCAB = 152064                 # bigger is better

FW_TEXT = "/content/FW_TEXT.txt"
WAKE_LEXICON = "/content/wake_lexicon.txt"

RUN_DIR = Path("/content/drive/MyDrive/wake2vec_qwen14b_p1")
LOCAL_RUN = Path("/content/runs/wake2vec_qwen14b_p1")
SENTRY = RUN_DIR / "sentry_backups"
EMB_SNAPS = RUN_DIR / "emb_snaps"
FULL_CHECKPOINTS = RUN_DIR / "full_checkpoints"

for d in [RUN_DIR, LOCAL_RUN, SENTRY, EMB_SNAPS, FULL_CHECKPOINTS]:
    d.mkdir(parents=True, exist_ok=True)

MAX_STEPS = 3000
LR = 5e-4
WARMUP_STEPS = max(20, MAX_STEPS // 20)  # 5% warmup
WEIGHT_DECAY = 0.0
BATCH_SIZE = 1
GRAD_ACCUM = 16              # effective batch 16, same as everyone else
SEQ_LEN = 256                # might need to drop to 128 if OOM
SAVE_STEPS = 50              # colab disconnect PTSD
LOG_STEPS = 50
EVAL_STEPS = 200
EMB_SNAP_STEPS = 50

RESUME_FROM = None
# RESUME_FROM = SENTRY / "checkpoint-200"   # uncomment when gaslight GPU strikes

# VRAM math:
#   4-bit model body: ~8GB
#   fp32 embeddings (197K x 5120): ~4GB (this is the spicy part)
#   Adafactor states: ~0 (no momentum, bless)
#   gradients + activations: ~1-2GB
#   total: ~13-14GB
#   T4 has 15GB
#   life in the margins

print("WAKE2VEC P1: Qwen2.5-14B CONFIG")
print(f"  Model: {MODEL_NAME}")
print(f"  Base vocab: {BASE_VOCAB} (152K — qwen came prepared)")
print(f"  Output: {RUN_DIR}")
print(f"  Steps: {MAX_STEPS}")
print(f"  LR: {LR}")
print(f"  Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Seq len: {SEQ_LEN}")
print(f"  Resume: {RESUME_FROM}")

# tokenizer + wake vocab 
# (a little game called "how many wake tokens does qwen already know")
from transformers import AutoTokenizer

print("\nLoading Qwen tokenizer (152K vocab, absolute chonker)...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"  Base vocab: {len(tok)}")

print(f"Loading Wake lexicon...")
with open(WAKE_LEXICON, 'r', encoding='utf-8') as f:
    wake_tokens = [line.strip() for line in f if line.strip()]

# check which wake tokens qwen already has in its 152K vocab
# (bigger vocab = fewer new tokens needed)
missing = [t for t in wake_tokens if tok.convert_tokens_to_ids(t) == tok.unk_token_id]
num_added = tok.add_tokens(missing, special_tokens=False)

already_known = len(wake_tokens) - len(missing)
print(f"  Wake tokens in lexicon: {len(wake_tokens)}")
print(f"  Already in Qwen vocab: {already_known} (qwen knows things)")
print(f"  New tokens added: {num_added}")
print(f"  Final vocab size: {len(tok)}")
print(f"  Vocab expansion: {num_added / BASE_VOCAB * 100:.1f}% "
      f"(vs TinyLlama's ~140% — much gentler)")

TOTAL_VOCAB = len(tok)

# rough embedding matrix size check
emb_size_gb = TOTAL_VOCAB * 5120 * 4 / 1e9  # fp32
print(f"\n  Embedding matrix will be: {TOTAL_VOCAB} x 5120 x fp32 = ~{emb_size_gb:.1f}GB")
if emb_size_gb > 4.5:
    print(f"  ...that's a fuck load of embeddings. if OOM, can deal with it")

# dataset
# (same as every other run. FW_TEXT + lexicon, chunked, shuffled. how many people do i piss off if i say "vibes")
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class BlockDataset(Dataset):
    """non-overlapping token blocks. simple. effective. joycean."""
    def __init__(self, blocks, seq_len=256):
        self.blocks = blocks
        self.seq_len = seq_len
    def __len__(self):
        return len(self.blocks)
    def __getitem__(self, idx):
        ids = torch.tensor(self.blocks[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}

print("Loading training data...")
with open(FW_TEXT, 'r', encoding='utf-8') as f:
    fw_text = f.read()
with open(WAKE_LEXICON, 'r', encoding='utf-8') as f:
    lex_text = f.read()

combined_text = fw_text + "\n" + lex_text
print(f"  FW text: {len(fw_text)} chars | Lexicon: {len(lex_text)} chars")

ids = tok(combined_text, add_special_tokens=False)["input_ids"]
print(f"  Total tokens: {len(ids)}")

blocks = [ids[i:i + SEQ_LEN] for i in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN)
          if len(ids[i:i + SEQ_LEN]) == SEQ_LEN]
print(f"  Total blocks: {len(blocks)}")

train_blocks, val_blocks = train_test_split(blocks, test_size=0.10, random_state=42)
train_ds = BlockDataset(train_blocks, SEQ_LEN)
val_ds = BlockDataset(val_blocks, SEQ_LEN)
print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

# model
# (hold my hand through this)
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

print("\nLoading Qwen2.5-14B (4-bit)... this will take a minute. or five.")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "11GB", "cpu": "30GB"},  # tighter than 8B. needs room for the 4GB embedding
    trust_remote_code=True
)

model.config.use_cache = False
model.config.attn_implementation = "eager"
model.config.tie_word_embeddings = True
if hasattr(model, "tie_weights"):
    model.tie_weights()

vram_after_load = torch.cuda.memory_allocated(0) / 1e9
print(f"  VRAM after model load: {vram_after_load:.2f} GB")
print(f"  Remaining: ~{vram_total - vram_after_load:.1f} GB (need ~4GB for embeddings + training)")

# frozen LoRA r=1 (PEFT compat shim for quantized training. contributes nothing)
peft_cfg = LoraConfig(
    r=1, lora_alpha=1, lora_dropout=0.0,
    target_modules=["q_proj"], bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_cfg)

# freeze everything so only the embed layer trainable
for n, p in model.named_parameters():
    p.requires_grad = False

# wake embed injection 
# (spherical init: random directions, radius matched to base vocab norms)
old_vocab = model.get_input_embeddings().weight.shape[0]
model.resize_token_embeddings(TOTAL_VOCAB)
wte = model.get_input_embeddings()

# tie output embeddings
if hasattr(model, "lm_head"):
    model.lm_head.weight = wte.weight

with torch.no_grad():
    base = wte.weight[:old_vocab]
    dim = base.shape[1]
    std = base.std().item()
    base_radius = std * math.sqrt(dim)
    target_radius = 1.5 * base_radius

    if num_added > 0:
        # spherical init: random direction, fixed magnitude. each new token starts at a unique point on a hypersphere
        new = torch.randn((num_added, dim), device=wte.weight.device)
        new = new / (new.norm(dim=1, keepdim=True) + 1e-8) * target_radius
        wte.weight.data[old_vocab:old_vocab + num_added] = new

print(f"  Vocab: {old_vocab} -> {TOTAL_VOCAB} (+{num_added} new Wake tokens)")
print(f"  Spherical init radius: {target_radius:.4f}")
print(f"  Embedding dim: {dim}")

# make embedding trainable, mask base rows
wte.weight.requires_grad = True

new_rows = torch.arange(old_vocab, old_vocab + num_added, device=wte.weight.device) if num_added > 0 else None
base_rows_idx = torch.arange(0, old_vocab, device=wte.weight.device)

def mask_grad(grad):
    """zero gradients for all base vocab rows.
    only the new wake tokens learn. the OGs stay put."""
    if grad is None or new_rows is None:
        return grad
    grad[base_rows_idx] = 0
    return grad

wte.weight.register_hook(mask_grad)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
vram_after_setup = torch.cuda.memory_allocated(0) / 1e9
print(f"  Trainable params: {trainable:,}")
print(f"  VRAM after setup: {vram_after_setup:.2f} GB")
print(f"  Headroom: ~{vram_total - vram_after_setup:.1f} GB (for gradients + activations)")

if vram_total - vram_after_setup < 1.5:
    print("  ⚠ VRAM is VERY tight. if training OOMs, try SEQ_LEN=128")

# pre-training snapshot 
E_pre = wte.weight.detach().cpu().clone()
torch.save(E_pre, RUN_DIR / "embeddings_pre.pt")
print(f"  Pre-training snapshot saved: {E_pre.shape}")

# callbacks 
# (the paranoia suite: save everything everywhere all at once)
import shutil, time
from transformers import TrainerCallback

def has_weights(ck):
    return (ck / "adapter_model.safetensors").exists() or (ck / "pytorch_model.bin").exists()

class EmbeddingSnapshot(TrainerCallback):
    """save embeddings every N steps because colab is not to be trusted"""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % EMB_SNAP_STEPS == 0:
            try:
                torch.save(wte.weight.detach().cpu(),
                           EMB_SNAPS / f"emb_step{state.global_step:04d}.pt")
                os.sync()
            except Exception as e:
                print(f"[EMB] {e}")

class FullCheckpoint(TrainerCallback):
    """full model dump to Drive. belt AND suspenders."""
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
            os.sync()
            print(f"[FULL] Step {step}: saved to the cloud (trust the cloud)")
        except Exception as e:
            print(f"[FULL] {e}")

class SentryMirror(TrainerCallback):
    """mirror trainer checkpoints to Drive because local storage is ephemeral"""
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
            if dst.exists():
                return
            shutil.copytree(ck, dst)
            os.sync()
            print(f"[SENTRY] {ck.name}: safe on Drive")
        except Exception as e:
            print(f"[SENTRY] {e}")

class LossMonitor(TrainerCallback):
    """yell if train and eval diverge too much"""
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
                print(f"[WARN] train/eval gap: {gap:.2f} — might be vibing too hard")

class StepTimer(TrainerCallback):
    """track how slow this absolute unit is"""
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
                remaining = (MAX_STEPS - state.global_step) * avg / 60
                print(f"[{state.global_step:4d}] {avg:.1f}s/step | ~{remaining:.0f}min remaining")
        self.last_time = now

# trainer 
# (Adafactor because we cannot afford momentum states on this budget)
from transformers import TrainingArguments, Trainer

class EmbOnlyTrainer(Trainer):
    """custom trainer: Adafactor on just the embedding weight.
    no momentum = no extra memory = we live another day."""
    def create_optimizer(self):
        # critical for 14B on T4 where every MB counts
        from transformers.optimization import Adafactor
        if not hasattr(self, "optimizer") or self.optimizer is None:
            self.optimizer = Adafactor(
                [{"params": [wte.weight]}],
                lr=LR,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = model(**inputs, use_cache=False)
        loss = out.loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("NaN/Inf loss — the void stares back")
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
    gradient_checkpointing=True,    # a need, a must 
    max_grad_norm=1.0,
)

trainer = EmbOnlyTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=None,
    callbacks=[EmbeddingSnapshot(), FullCheckpoint(), SentryMirror(),
               LossMonitor(), StepTimer()],
)

print("=" * 60)
print("WAKE2VEC P1: Qwen2.5-14B EMBEDDING-ONLY")
print("  (the biggest lad on free colab)")
print("=" * 60)
print(f"  Train: {len(train_ds)} blocks | Val: {len(val_ds)} blocks")
print(f"  Steps: {MAX_STEPS} | Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  New Wake tokens: {num_added} | Already in vocab: {already_known}")
print(f"  VRAM before training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"  Optimizer: Adafactor (no momentum, no mercy)")
print("=" * 60)

# train 
# (to be or not to be innit)
t0 = time.time()
if RESUME_FROM is not None:
    local_ckpt = LOCAL_RUN / RESUME_FROM.name
    if not local_ckpt.exists():
        shutil.copytree(RESUME_FROM, local_ckpt)
    print(f"[RESUME] {RESUME_FROM.name} — welcome back, we missed you")
    trainer.train(resume_from_checkpoint=str(local_ckpt))
else:
    trainer.train()
elapsed = (time.time() - t0) / 60
print(f"\nTRAINING COMPLETE ({elapsed:.1f} minutes)")
print(f"that's {elapsed/60:.1f} hours of T4 time. you're welcome, google.")

# save final 
final_dir = RUN_DIR / "final"
final_dir.mkdir(exist_ok=True)
model.save_pretrained(str(final_dir))
tok.save_pretrained(str(final_dir))
final_emb = wte.weight.detach().cpu()
torch.save(final_emb, final_dir / "embeddings.pt")
os.sync()
print(f"Final model saved to {final_dir}")

# loss curve 
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
        plt.title('Wake2Vec P1: Qwen2.5-14B Loss Curve (the big one)')
        plt.legend(); plt.grid(True, alpha=0.3)
        plot_path = RUN_DIR / "p1_qwen14b_loss_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {plot_path}")
        plt.show()
        print(f"\nFinal train loss: {losses[-1]:.4f}")
        if val_data:
            print(f"Final val loss: {v_losses[-1]:.4f}")
            print(f"Best val loss: {min(v_losses):.4f}")

  # let's see if this shit even runs before i worry about an analysis 
