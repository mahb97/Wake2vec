# -*- coding: utf-8 -*-
"""wake2vec_on_qwen_2_5_14b_EXTENSION.py

# Wake2Vec P1 EXTENDED: Qwen2.5-14B SGDR Continuation Experiment

**Model:** Qwen/Qwen2.5-14B (4-bit quantized body, fp16 frozen embeddings, fp32 Wake overlay)
**Hardware:** Google Colab T4 GPU (15GB VRAM) more steps
**Training data:** Finnegans Wake corpus + Wake lexicon 
**Resume source:** canonical P1 step 3000 sentry

## extension argument

The canonical P1 run hit step 3000 with val still descending. The Qween
was the only model in the Wake2vec lineup that never plateaued, and
investigation revealed that the WakeOverlay + STEP_OFFSET manual-resume
pattern accidentally implements SGDR (Loshchilov and Hutter 2017). At
every session resume, the LR scheduler restarts near peak cosine LR,
producing a train spike followed by val descent. 30+ confirmed cycles
through canonical P1, with the cycles intensifying rather than decaying
toward the end (step 2500 spike 240 produced val drop 0.32, step 2700
spike 224 produced val drop 0.33).

This extender tests whether the SGDR escape-from-local-minima mechanism
keeps working past the canonical 3000-step ceiling, or whether the model
eventually exhausts its effective capacity even under warm restarts.

Three possible outcomes:
  1. SGDR keeps working. Train spikes at session boundaries continue
     producing val descent. Methodological finding about long-horizon
     warm restarts on this architecture.
  2. SGDR exhausts. Train spikes stop producing val descent. Confirms
     the model was at intrinsic capacity ceiling through canonical P1.
  3. Mixed signature. Intermittent escape and plateau cycles.

the extender aims to turns "did Qwen keep descending
because SGDR or because capacity?" into a falsifiable test.

## Architecture recap

Qwen has a 152K vocab, so 767 Wake neologisms are already
in there. The extender continues training the 43,824 added Wake tokens
via WakeOverlay, which is a separate nn.Embedding holding only the Wake rows
trainable in fp32 (0.86 GB) while the 152K base embeddings stay frozen
in fp16 (2.0 GB). This is the only way it will fit on free T4.

Also still required: triton.ops shim (bnb 0.45.0 vs triton 3.x),
mean_resizing=False (OOM on covariance matrix), accelerate monkey-patch
(4-bit + CPU offload training).

────────────────────────────────────────────────────────────
the Qween was always the longest descent and the extender tests just how long the riverrun is.
"""

# housekeeping

# import os
# os.kill(os.getpid(), 9)

# !pip install torch==2.9.0+cu126 torchaudio==2.9.0+cu126 torchvision==0.24.0+cu126 --index-url https://download.pytorch.org/whl/cu126 && pip install bitsandbytes==0.46.1 peft==0.18.1 accelerate==1.12.0 --force-reinstall --no-deps

# !pip show bitsandbytes | grep Version
# !pip show torch | grep Version

# envi
import os
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch, gc
print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")

import sys, types

# Fake triton.ops for bnb 0.45.0 on triton 3.x
fake_perf = types.ModuleType('triton.ops.matmul_perf_model')
fake_perf.early_config_prune = lambda *a, **k: []
fake_perf.estimate_matmul_time = lambda *a, **k: 0

sys.modules['triton.ops'] = types.ModuleType('triton.ops')
sys.modules['triton.ops.matmul_perf_model'] = fake_perf

import bitsandbytes as bnb_lib
print(f"bitsandbytes: {bnb_lib.__version__}")

import transformers, accelerate, peft
print(f"transformers: {transformers.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"peft: {peft.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM: {vram_total:.2f} GB")

torch.cuda.empty_cache()
gc.collect()

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
vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM: {vram_total:.2f} GB {'(gaslight GPU energy)' if vram_total < 16 else ''}")

torch.cuda.empty_cache()
gc.collect()

from google.colab import drive
drive.mount('/content/drive')

from huggingface_hub import login
login()

# config
import math
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-14B"
BASE_VOCAB = 152064               

FW_TEXT = "/content/FW_TEXT.txt"
WAKE_LEXICON = "/content/wake_lexicon.txt"

RUN_DIR = Path("/content/drive/MyDrive/wake2vec_qwen14b_extended")
LOCAL_RUN = Path("/content/runs/wake2vec_qwen14b_extended")
SENTRY = RUN_DIR / "sentry_backups"
CANONICAL_RUN_DIR = Path("/content/drive/MyDrive/wake2vec_qwen14b_p1")  # canonical

for d in [RUN_DIR, LOCAL_RUN, SENTRY]:
    d.mkdir(parents=True, exist_ok=True)

MAX_STEPS = 6000
LR = 5e-4
WARMUP_STEPS = max(20, MAX_STEPS // 20)
WEIGHT_DECAY = 0.0
BATCH_SIZE = 1
GRAD_ACCUM = 16             
SEQ_LEN = 128  
SAVE_STEPS = 20             
LOG_STEPS = 20
EVAL_STEPS = 20
EMB_SNAP_STEPS = 20
STEP_OFFSET = 3000 

# RESUME_FROM = None
RESUME_FROM = CANONICAL_RUN_DIR / "sentry_backups" / "sentry_step_3000.pt"

print(f"  Model: {MODEL_NAME}")
print(f"  Base vocab: {BASE_VOCAB}")
print(f"  Output: {RUN_DIR}")
print(f"  Steps: {MAX_STEPS}")
print(f"  LR: {LR}")
print(f"  Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Seq len: {SEQ_LEN}")
print(f"  Resume: {RESUME_FROM}")

# tokenizer + wake vocab
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"  Base vocab: {len(tok)}")

with open(WAKE_LEXICON, 'r', encoding='utf-8') as f:
    wake_tokens = [line.strip() for line in f if line.strip()]

missing = [t for t in wake_tokens if tok.convert_tokens_to_ids(t) == tok.unk_token_id]
num_added = tok.add_tokens(missing, special_tokens=False)

already_known = len(wake_tokens) - len(missing)
print(f"  Wake tokens in lexicon: {len(wake_tokens)}")
print(f"  Already in Qwen vocab: {already_known}")
print(f"  New tokens added: {num_added}")
print(f"  Final vocab size: {len(tok)}")
print(f"  Vocab expansion: {num_added / BASE_VOCAB * 100:.1f}% "
      f"(vs TinyLlama's ~140% much gentler)")

TOTAL_VOCAB = len(tok)

# rough embedding matrix size check
emb_size_gb = TOTAL_VOCAB * 5120 * 4 / 1e9  # fp32
print(f"\n  Embedding matrix will be: {TOTAL_VOCAB} x 5120 x fp32 = ~{emb_size_gb:.1f}GB")
if emb_size_gb > 4.5:
    print(f"  ...that's a fuck load of embeds. if OOM, can deal with it")

# dataset
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
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}

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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "11GB", "cpu": "30GB"},
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

# Wake overlay
import torch, math
import torch.nn as nn

wake_start = BASE_VOCAB  # 152064
actual_wake_count = TOTAL_VOCAB - wake_start  # 43824

model.resize_token_embeddings(TOTAL_VOCAB, mean_resizing=False)
wte = model.get_input_embeddings()
wte.weight.data = wte.weight.data.half()

# Spherical init for Wake region
with torch.no_grad():
    base = wte.weight[:wake_start]
    dim = base.shape[1]
    std = base.std().item()
    base_radius = std * math.sqrt(dim)
    target_radius = 1.5 * base_radius

    new = torch.randn((actual_wake_count, dim), device=wte.weight.device, dtype=torch.float16)
    new = new / (new.norm(dim=1, keepdim=True) + 1e-8) * target_radius
    wte.weight.data[wake_start:TOTAL_VOCAB] = new

# FREEZE the full embedding so no 1.87 GB gradient
wte.weight.requires_grad = False

# Small trainable overlay for Wake tokens only
class WakeOverlay(nn.Module):
    def __init__(self, full_embed, start, count):
        super().__init__()
        self.full_embed = full_embed
        self.start = start
        self.end = start + count
        # Trainable: just the Wake rows, copied from the initialized values
        self.wake_embed = nn.Embedding(count, full_embed.embedding_dim)
        self.wake_embed.weight.data = full_embed.weight.data[start:start+count].clone().float()

    def forward(self, input_ids):
    # Base lookup (frozen, no gradient)
        base_out = self.full_embed(input_ids)
    # Find Wake tokens in this batch
        mask = (input_ids >= self.start) & (input_ids < self.end)
    # Always compute wake_out to keep grad graph alive
    # (non-wake positions get clamped to index 0 but won't be selected)
        wake_ids = (input_ids - self.start).clamp(min=0, max=self.wake_embed.num_embeddings - 1)
        wake_out = self.wake_embed(wake_ids).to(base_out.dtype)
        mask_expanded = mask.unsqueeze(-1).expand_as(base_out)
        return torch.where(mask_expanded, wake_out, base_out)

overlay = WakeOverlay(wte, wake_start, actual_wake_count)
model.set_input_embeddings(overlay)

# Tie lm_head to the full weight (frozen, that's fine)
if hasattr(model, "lm_head"):
    model.lm_head.weight = wte.weight

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Wake overlay: {actual_wake_count} tokens, fp32, {actual_wake_count*5120*4/1e9:.2f} GB")
print(f"Trainable params: {trainable:,}")
print(f"VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
print(f"Headroom: {vram_total - torch.cuda.memory_allocated(0)/1e9:.1f} GB")

# callbacks
import time
import shutil
from transformers import TrainerCallback

class EmbeddingSnapshot(TrainerCallback):
    def __init__(self, wake_embed, local_run, snap_steps, step_offset=0):
        self.wake_embed = wake_embed
        self.local_run = local_run
        self.snap_steps = snap_steps
        self.step_offset = step_offset
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.snap_steps == 0:
            try:
                global_step = state.global_step + self.step_offset
                emb = self.wake_embed.weight.data.detach().cpu()
                torch.save(emb, self.local_run / f"emb_step{global_step:04d}.pt")
                print(f"[EMB] step {global_step}: local snapshot ({emb.shape[0]} tokens)")
            except Exception as e:
                print(f"[EMB] {e}")

class DriveSentry(TrainerCallback):
    def __init__(self, wake_embed, wake_start, num_wake, sentry_dir, step_offset=0, local_run=None):
        self.wake_embed = wake_embed
        self.wake_start = wake_start
        self.num_wake = num_wake
        self.sentry_dir = sentry_dir
        self.step_offset = step_offset
        self.local_run = local_run or Path("/content/runs")
    def on_save(self, args, state, control, **kw):
        try:
            global_step = state.global_step + self.step_offset
            dst = self.sentry_dir / f"sentry_step_{global_step:04d}.pt"
            if dst.exists():
                return
            emb = self.wake_embed.weight.data.detach().cpu().half()
            payload = {
                'embeddings': emb,
                'step': global_step,
                'best_metric': state.best_metric,
                'epoch': state.epoch,
                'wake_start': self.wake_start,
                'num_wake': self.num_wake,
            }
            local_tmp = self.local_run / "sentry_tmp.pt"
            torch.save(payload, local_tmp)
            shutil.copy2(local_tmp, dst)
            local_tmp.unlink()
            size_mb = dst.stat().st_size / (1024 * 1024)
            print(f"[SENTRY] step {global_step}: {size_mb:.0f}MB to Drive")
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
                remaining = (MAX_STEPS - state.global_step) * avg / 60
                print(f"[{state.global_step:4d}] {avg:.1f}s/step | ~{remaining:.0f}min remaining")
        self.last_time = now

import accelerate.accelerator as _acc
_orig_prepare = _acc.Accelerator.prepare_model
def _hacked_prepare(self, model, device_placement=None, evaluation_mode=False):
    # Skip the 4-bit + CPU offload check
    self._models.append(model)
    return model
_acc.Accelerator.prepare_model = _hacked_prepare

wte = model.get_input_embeddings()
print(f"Full embed dtype: {wte.full_embed.weight.dtype}")
print(f"Full embed shape: {wte.full_embed.weight.shape}")
print(f"Wake overlay dtype: {wte.wake_embed.weight.dtype}")
print(f"Wake overlay shape: {wte.wake_embed.weight.shape}")
print(f"Wake overlay size: {wte.wake_embed.weight.nelement() * wte.wake_embed.weight.element_size() / 1e9:.2f} GB")

# gpu flush
import torch, gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(f"GPU: {torch.cuda.memory_allocated(0)/1e9:.2f} GB allocated")
print(f"     {torch.cuda.memory_reserved(0)/1e9:.2f} GB reserved")

# Extender pre-training snapshot (skip on resume)
import shutil
canonical_3000_path = CANONICAL_RUN_DIR / "sentry_backups" / "sentry_step_3000.pt"
extender_pre_path = RUN_DIR / "embeddings_pre.pt"
if canonical_3000_path.exists() and not extender_pre_path.exists():
    # extract just the embeddings tensor from the canonical sentry payload
    ckpt = torch.load(canonical_3000_path, map_location="cpu")
    torch.save(ckpt['embeddings'].float(), extender_pre_path)
    print(f"Extender pre-snapshot copied from canonical step 3000")

# shell
!mkdir -p "/content/drive/MyDrive/wake2vec_qwen14b_extended/sentry_backups"

# training args and train
import os
# trainer
from transformers import TrainingArguments, Trainer

class EmbOnlyTrainer(Trainer):
    """custom trainer: Adafactor on just the embedding weight."""
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def create_optimizer(self):
        # critical for 14B on T4 where every MB counts
        from transformers.optimization import Adafactor
        if not hasattr(self, "optimizer") or self.optimizer is None:
            self.optimizer = Adafactor(
               [{"params": [wte.wake_embed.weight]}],
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
    gradient_checkpointing=True,   
    max_grad_norm=1.0,
)

trainer = EmbOnlyTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=None,
    callbacks=[
      EmbeddingSnapshot(overlay.wake_embed, LOCAL_RUN, EMB_SNAP_STEPS, STEP_OFFSET),
      DriveSentry(overlay.wake_embed, wake_start, actual_wake_count, SENTRY, STEP_OFFSET, LOCAL_RUN),
      LossMonitor(), StepTimer(),],
)

print(f"  Train: {len(train_ds)} blocks | Val: {len(val_ds)} blocks")
print(f"  Steps: {MAX_STEPS} | Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  New Wake tokens: {num_added} | Already in vocab: {already_known}")
print(f"  VRAM before training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# train
t0 = time.time()
if RESUME_FROM is not None and RESUME_FROM.exists():
    ckpt = torch.load(RESUME_FROM, map_location="cpu")
    with torch.no_grad():
        overlay.wake_embed.weight.data.copy_(ckpt['embeddings'].float())
    STEP_OFFSET = STEP_OFFSET if STEP_OFFSET > 0 else ckpt['step']
    remaining = MAX_STEPS - STEP_OFFSET
    args.max_steps = remaining
    args.warmup_steps = max(20, remaining // 20)
    # rebuild callbacks with correct offset
    trainer.callback_handler.callbacks = [
        cb for cb in trainer.callback_handler.callbacks
        if not isinstance(cb, (EmbeddingSnapshot, DriveSentry))
    ]
    trainer.add_callback(EmbeddingSnapshot(overlay.wake_embed, LOCAL_RUN, EMB_SNAP_STEPS, STEP_OFFSET))
    trainer.add_callback(DriveSentry(overlay.wake_embed, wake_start, actual_wake_count, SENTRY, STEP_OFFSET, LOCAL_RUN))
    print(f"[RESUME] Restored embeddings from step {STEP_OFFSET}")
    print(f"[RESUME] STEP_OFFSET={STEP_OFFSET}, remaining={remaining}, warmup={args.warmup_steps}")
trainer.train()

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

# embed analysis 

import numpy as np
from numpy.linalg import norm as l2
from scipy import stats
from sklearn.decomposition import PCA
import gc

torch.cuda.empty_cache()
gc.collect()

# extract embeddings from WakeOverlay 
E_wake = overlay.wake_embed.weight.detach().cpu().float().numpy()  # (43824, 5120) trained
E_base = overlay.full_embed.weight[:wake_start].detach().cpu().float().numpy()  # (152064, 5120) frozen
n_wake = E_wake.shape[0]  # actual_wake_count
n_base = E_base.shape[0]  # wake_start = BASE_VOCAB
dim = E_wake.shape[1]     # 5120

print(f"Embedding analysis: {n_base} base tokens + {n_wake} wake tokens in {dim}-d space")

# pre-training snapshot (full table saved before training started)
pre_path = RUN_DIR / "embeddings_pre.pt"
has_pre = pre_path.exists()
E_pre_wake = None
if has_pre:
    E_pre_all = torch.load(pre_path, map_location="cpu").float().numpy()
    E_pre_wake = E_pre_all[wake_start:wake_start + n_wake]
    del E_pre_all
    gc.collect()

# norm analysis
base_norms = l2(E_base, axis=1)
wake_norms = l2(E_wake, axis=1)

t_stat, t_pval = stats.ttest_ind(base_norms, wake_norms, equal_var=False)
u_stat, u_pval = stats.mannwhitneyu(base_norms, wake_norms, alternative='two-sided')
pooled_std = np.sqrt((base_norms.std()**2 + wake_norms.std()**2) / 2)
cohens_d = (base_norms.mean() - wake_norms.mean()) / pooled_std

print("norm analysis")
print(f"  Base    -- mean: {base_norms.mean():.4f}, std: {base_norms.std():.4f} (n={n_base})")
print(f"  Wake    -- mean: {wake_norms.mean():.4f}, std: {wake_norms.std():.4f} (n={n_wake})")
print(f"  Welch t:  t={t_stat:.4f}, p={t_pval:.2e}")
print(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_pval:.2e}")
print(f"  Cohen's d: {cohens_d:.4f}")
# check if wake norms are still clustered or have dispersed
print(f"  Wake norm range: [{wake_norms.min():.4f}, {wake_norms.max():.4f}]")
print(f"  Wake norm CV:    {wake_norms.std() / wake_norms.mean():.4f} (lower = still spherical)")

# isotropy
def compute_isotropy(embeddings, seed=42):
    centered = embeddings - embeddings.mean(axis=0)
    nrm = l2(centered, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-12, 1e-12, nrm)
    unit = centered / nrm
    rng = np.random.default_rng(seed)
    n = min(5000, len(unit))
    idx = rng.choice(len(unit), size=n, replace=False)
    sample = unit[idx]
    cos_mat = sample @ sample.T
    np.fill_diagonal(cos_mat, 0)
    Z = np.exp(cos_mat).sum(axis=1)
    isotropy = Z.min() / Z.max()
    mean_cos = cos_mat.sum() / (n * (n - 1))
    return isotropy, mean_cos, n

iso_base, cos_base, n_iso_base = compute_isotropy(E_base)
iso_wake, cos_wake, n_iso_wake = compute_isotropy(E_wake)

print("isotropy (Mu et al. 2018 partition function ratio)")
print(f"  Base -- {iso_base:.6f} (mean_cos: {cos_base:.4f}, n={n_iso_base})")
print(f"  Wake -- {iso_wake:.6f} (mean_cos: {cos_wake:.4f}, n={n_iso_wake})")

# drift (spherical init → trained) 
def safe_cos(a, b):
    na = l2(a, axis=1, keepdims=True); na = np.where(na < 1e-12, 1e-12, na)
    nb = l2(b, axis=1, keepdims=True); nb = np.where(nb < 1e-12, 1e-12, nb)
    return np.sum((a / na) * (b / nb), axis=1)

print("drift of wake tokens from spherical init")
if has_pre and E_pre_wake is not None:
    wake_drift_cos = safe_cos(E_pre_wake, E_wake)
    wake_drift_l2 = l2(E_wake - E_pre_wake, axis=1)

    print(f"  Wake cosine drift: {wake_drift_cos.mean():.6f} +/- {wake_drift_cos.std():.6f}")
    print(f"  Wake L2 drift:     {wake_drift_l2.mean():.4f} +/- {wake_drift_l2.std():.4f}")
    print(f"  (base tokens are frozen via WakeOverlay and no drift by construction)")

    wake_drift_order = np.argsort(wake_drift_cos)
    print(f"\n  Top 20 most-changed Wake tokens (broke free from the sphere):")
    for rank, idx in enumerate(wake_drift_order[:20]):
        global_idx = wake_start + idx
        token_str = tok.convert_ids_to_tokens(int(global_idx))
        print(f"    {rank+1:2d}. {token_str!r:25s} cos={wake_drift_cos[idx]:.6f} L2={wake_drift_l2[idx]:.4f}")

    print(f"\n  Top 20 least-changed (stayed close to init):")
    for rank, idx in enumerate(wake_drift_order[-20:][::-1]):
        global_idx = wake_start + idx
        token_str = tok.convert_ids_to_tokens(int(global_idx))
        print(f"    {rank+1:2d}. {token_str!r:25s} cos={wake_drift_cos[idx]:.6f} L2={wake_drift_l2[idx]:.4f}")
else:
    wake_drift_cos = None
    print("  No pre-training snapshot found, skipping drift analysis")

# nearest neighbour 
print("nearest neighbour")

# Normalize both sets
base_norms_safe = l2(E_base, axis=1, keepdims=True)
base_norms_safe = np.where(base_norms_safe < 1e-12, 1e-12, base_norms_safe)
E_base_unit = E_base / base_norms_safe

wake_norms_safe = l2(E_wake, axis=1, keepdims=True)
wake_norms_safe = np.where(wake_norms_safe < 1e-12, 1e-12, wake_norms_safe)
E_wake_unit = E_wake / wake_norms_safe

# Sample 20 Wake tokens: first 10, middle 5, last 5
sample_local = list(range(10))
if n_wake > 100:
    sample_local += list(range(n_wake // 2, n_wake // 2 + 5))
if n_wake > 1000:
    sample_local += list(range(n_wake - 5, n_wake))

for local_idx in sample_local:
    global_idx = wake_start + local_idx
    wt = tok.convert_ids_to_tokens(global_idx)
    # cosine similarity against all base tokens
    sims = (E_wake_unit[local_idx:local_idx+1] @ E_base_unit.T).squeeze()
    top5 = np.argsort(sims)[-5:][::-1]
    nb = ", ".join(f"{tok.convert_ids_to_tokens(int(i))!r}({sims[i]:.3f})" for i in top5)
    print(f"  {wt!r:25s} -> {nb}")

del E_base_unit, E_wake_unit, base_norms_safe, wake_norms_safe
gc.collect()

# PCA / intrinsic dimensionality
print("intrinsic dimensionality (PCA)")

n_comp = min(100, dim, n_base, n_wake)
pca_base = PCA(n_components=n_comp).fit(E_base)
pca_wake = PCA(n_components=n_comp).fit(E_wake)
cumvar_base = np.cumsum(pca_base.explained_variance_ratio_)
cumvar_wake = np.cumsum(pca_wake.explained_variance_ratio_)

dim90_base = np.searchsorted(cumvar_base, 0.90) + 1
dim95_base = np.searchsorted(cumvar_base, 0.95) + 1
dim90_wake = np.searchsorted(cumvar_wake, 0.90) + 1
dim95_wake = np.searchsorted(cumvar_wake, 0.95) + 1

print(f"  Base -- 90% in {dim90_base} PCs, 95% in {dim95_base} PCs")
print(f"  Wake -- 90% in {dim90_wake} PCs, 95% in {dim95_wake} PCs")
print(f"  Base top-1 PC explains {pca_base.explained_variance_ratio_[0]*100:.1f}%")
print(f"  Wake top-1 PC explains {pca_wake.explained_variance_ratio_[0]*100:.1f}%")

# pairwise cosine distribution 
rng = np.random.default_rng(42)

def sample_cos(E1, E2, n=2000):
    i1 = rng.choice(len(E1), size=min(n, len(E1)), replace=False)
    i2 = rng.choice(len(E2), size=min(n, len(E2)), replace=False)
    s1, s2 = E1[i1], E2[i2]
    n1 = l2(s1, axis=1, keepdims=True); n1 = np.where(n1 < 1e-12, 1e-12, n1)
    n2 = l2(s2, axis=1, keepdims=True); n2 = np.where(n2 < 1e-12, 1e-12, n2)
    c = (s1 / n1) @ (s2 / n2).T
    return c[np.triu_indices_from(c, k=1)] if E1 is E2 else c.ravel()

cos_bb = sample_cos(E_base, E_base)
cos_ww = sample_cos(E_wake, E_wake)
cos_bw = sample_cos(E_base, E_wake)

ks_stat, ks_pval = stats.ks_2samp(cos_bb, cos_ww)

print("pairwise cosine similarity distribution")
print(f"  (base,base) -- mean: {cos_bb.mean():.4f}, std: {cos_bb.std():.4f}")
print(f"  (wake,wake) -- mean: {cos_ww.mean():.4f}, std: {cos_ww.std():.4f}")
print(f"  (base,wake) -- mean: {cos_bw.mean():.4f}, std: {cos_bw.std():.4f}")
print(f"  KS test (base-base vs wake-wake): D={ks_stat:.4f}, p={ks_pval:.2e}")

# 6 panel analysis fig 
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Wake2Vec P1 Qwen2.5-14B -- Embedding Analysis\n(the absolute unit)", fontsize=14, fontweight="bold")

# 1: Norm distributions
ax = axes[0, 0]
ax.hist(base_norms, bins=50, alpha=0.5, label=f"Base (\u03bc={base_norms.mean():.1f})", density=True)
ax.hist(wake_norms, bins=50, alpha=0.5, label=f"Wake (\u03bc={wake_norms.mean():.1f})", density=True)
ax.set_xlabel("L2 norm"); ax.set_ylabel("Density")
ax.set_title("Norm Distribution: Base vs Wake"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 2: Pairwise cosine distributions
ax = axes[0, 1]
ax.hist(cos_bb, bins=80, alpha=0.4, label=f"base-base (\u03bc={cos_bb.mean():.3f})", density=True)
ax.hist(cos_ww, bins=80, alpha=0.4, label=f"wake-wake (\u03bc={cos_ww.mean():.3f})", density=True)
ax.hist(cos_bw, bins=80, alpha=0.4, label=f"base-wake (\u03bc={cos_bw.mean():.3f})", density=True)
ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Density")
ax.set_title("Pairwise Cosine Distributions"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# 3: PCA cumulative variance
ax = axes[0, 2]
ax.plot(range(1, n_comp+1), cumvar_base, 'b-', label=f"Base (90%@{dim90_base})")
ax.plot(range(1, n_comp+1), cumvar_wake, 'r-', label=f"Wake (90%@{dim90_wake})")
ax.axhline(0.90, ls='--', c='gray', alpha=0.5, label="90% threshold")
ax.set_xlabel("Principal component"); ax.set_ylabel("Cumulative explained variance")
ax.set_title("Intrinsic Dimensionality"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 4: Drift histogram
ax = axes[1, 0]
if has_pre and wake_drift_cos is not None:
    ax.hist(wake_drift_cos, bins=80, alpha=0.7, color="steelblue",
            label=f"Wake (\u03bc={wake_drift_cos.mean():.4f})")
    ax.axvline(1.0, ls='--', c='coral', alpha=0.8, label="No drift (cos=1)")
    ax.set_xlabel("Cosine similarity (init \u2192 trained)"); ax.set_ylabel("Frequency")
    ax.set_title("Wake Token Drift from Spherical Init"); ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No pre-training\nsnapshot", ha='center', va='center',
            transform=ax.transAxes, fontsize=12, color='gray')
ax.grid(True, alpha=0.3)

# 5: Norm by token index
ax = axes[1, 1]
# subsample base for scatter (152K points is too many)
base_sample_idx = rng.choice(n_base, size=min(10000, n_base), replace=False)
ax.scatter(base_sample_idx, base_norms[base_sample_idx], s=0.1, alpha=0.3, label="Base", c="blue")
ax.scatter(np.arange(n_wake) + wake_start, wake_norms, s=0.1, alpha=0.3, label="Wake", c="red")
ax.set_xlabel("Token index"); ax.set_ylabel("L2 norm")
ax.set_title("Norm by Token Index"); ax.legend(fontsize=8, markerscale=10); ax.grid(True, alpha=0.3)

# 6: Top-20 eigenspectrum
ax = axes[1, 2]
x = np.arange(1, 21)
w = 0.35
ax.bar(x - w/2, pca_base.explained_variance_ratio_[:20], w, alpha=0.7, label="Base")
ax.bar(x + w/2, pca_wake.explained_variance_ratio_[:20], w, alpha=0.7, label="Wake")
ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance ratio")
ax.set_title("Top-20 PC Eigenspectrum"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RUN_DIR / "p1_qwen14b_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Analysis plot saved: {RUN_DIR / 'p1_qwen14b_analysis.png'}")

# wake generation
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

def temperature_sweep(prompt, temps=[0.5, 0.7, 0.9, 1.0, 1.2], **kwargs):
    print(f"PROMPT: {prompt}\n")
    for t in temps:
        generate_wake(prompt, temperature=t, **kwargs)
        print()
        
# gen samples 

print("single generation at temp=0.9")
generate_wake("riverrun, past Eve and Adam's,")

print("temperature sweep")
temperature_sweep("riverrun, past Eve and Adam's,",
                  temps=[0.5, 0.7, 0.9, 1.0, 1.2])

print("multi-sequence (3 samples at temp=0.9)")
generate_wake("riverrun, past Eve and Adam's,",
              num_return_sequences=3,
              temperature=0.9)

print("extended at temp=1.1 (max_new_tokens=512)")
generate_wake("riverrun, past Eve and Adam's,",
              num_return_sequences=3,
              temperature=1.1,
              max_new_tokens=512)

# summary JSON

import json

report = {
    "model": MODEL_NAME,
    "phase": "P1_embedding_only",
    "checkpoint": "final",
    "architecture": "WakeOverlay (custom nn.Embedding holding Wake rows only)",
    "vocab_size": int(TOTAL_VOCAB),
    "base_vocab": int(BASE_VOCAB),
    "wake_tokens": int(n_wake),
    "wake_already_in_vocab": int(already_known),
    "wake_share_of_total": float(n_wake / TOTAL_VOCAB),
    "dim": int(dim),
    "training_data": ["FW_TEXT.txt", "wake_lexicon.txt"],
    "hyperparameters": {
        "lr": LR,
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "max_steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "seq_len": SEQ_LEN,
        "optimizer": "Adafactor",
        "lr_scheduler": "cosine",
        "embedding_init": "spherical_1.5x",
        "embedding_strategy": "WakeOverlay (Wake rows trainable fp32, base frozen fp16)",
        "quantization": "4-bit NF4 (base body)",
    },
    "norms": {
        "base": {"mean": float(base_norms.mean()), "std": float(base_norms.std())},
        "wake": {"mean": float(wake_norms.mean()), "std": float(wake_norms.std())},
        "wake_range": [float(wake_norms.min()), float(wake_norms.max())],
        "welch_t": {"t": float(t_stat), "p": float(t_pval)},
        "mann_whitney_u": {"U": float(u_stat), "p": float(u_pval)},
        "cohens_d": float(cohens_d),
    },
    "isotropy": {
        "base": {"score": float(iso_base), "mean_cos": float(cos_base), "n": int(n_iso_base)},
        "wake": {"score": float(iso_wake), "mean_cos": float(cos_wake), "n": int(n_iso_wake)},
    },
    "pairwise_cosine": {
        "base_base": {"mean": float(cos_bb.mean()), "std": float(cos_bb.std())},
        "wake_wake": {"mean": float(cos_ww.mean()), "std": float(cos_ww.std())},
        "base_wake": {"mean": float(cos_bw.mean()), "std": float(cos_bw.std())},
        "ks_test_bb_vs_ww": {"D": float(ks_stat), "p": float(ks_pval)},
    },
    "intrinsic_dim": {
        "base_90pct": int(dim90_base), "base_95pct": int(dim95_base),
        "wake_90pct": int(dim90_wake), "wake_95pct": int(dim95_wake),
        "base_top1_var": float(pca_base.explained_variance_ratio_[0]),
        "wake_top1_var": float(pca_wake.explained_variance_ratio_[0]),
    },
    "loss": {
        "final_train": float(losses[-1]) if train_data else None,
        "final_eval": float(v_losses[-1]) if val_data else None,
        "best_eval": float(min(v_losses)) if val_data else None,
    },
}

if has_pre and wake_drift_cos is not None:
    report["drift"] = {
        "wake_cosine_mean": float(wake_drift_cos.mean()),
        "wake_cosine_std": float(wake_drift_cos.std()),
        "wake_l2_mean": float(wake_drift_l2.mean()),
        "wake_l2_std": float(wake_drift_l2.std()),
        "note": "Base tokens frozen via WakeOverlay (no drift by construction)",
    }

summary_path = RUN_DIR / "p1_qwen14b_summary.json"
summary_path.write_text(json.dumps(report, indent=2))
print(f"\n[SUMMARY] Saved to {summary_path}")
print(json.dumps(report, indent=2))

