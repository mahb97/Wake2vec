# -*- coding: utf-8 -*-
"""Wake2Vec Phase 2: LoRA Fine-Tune (Mistral 7B v0.3)

# Wake2Vec Phase 2: LoRA Fine-Tune with Frozen P1 Embeddings

**Model:** mistralai/Mistral-7B-v0.3 (4-bit quantized, sliding-window attention)
**Hardware:** Google Colab T4 GPU + CPU offload
**P1 Source:** wake2vec_mistral7b_p1/full_checkpoints/step_1200 (best val ~10.92, the global minimum / first 11.0 break)

## Overview

Phase 2 loads the embedding weights from the Mistral 7B v0.3 P1 run
(3000 steps, gradient-masked, spherical 1.5x init, val descended to 11.0936
and was still falling at the canonical endpoint) and freezes them entirely.
LoRA adapters are applied to attention (q, k, v) and MLP (gate, up, down)
projections. The model learns to use the Wake-adapted embeddings through
attention/MLP adaptation rather than further embedding modification.

This is the arrow phase of mu p -> UP: P1 laid down the micro-units (the
deepest embedding reorganization in the lineup, Wake drift 0.485) while P2's LoRA
routes them. The question P2 answers for Mistral is the suspension test: does
the voice rise coherently (TinyLlama-style, holding the fine line) when the
arrow routes Mistral's richly-learned 58%-share micro-units, or does it
over-deform as the P1 generation did. Given the matched 58% share with
TinyLlama and the deepest P1 learning in the lineup, the prior on a strong
suspension result is the most favourable in the lineup.

## Why step 1200 as the P1 source

Mistral's P1 val had TWO local minima. It broke 11.0 at step 1150-1200 (val ~10.92,
the global minimum), drifted back up through the long survey-phase plateau,
then descended again to 11.0936 at step 3000. The endpoint (11.09) is a second
minimum that never reclaimed the first (10.92). The global minimum is step 1200,
so by the best-val-source convention (lowest val = least overfit = best
generalising embeddings) the P1 source is step 1150, not the endpoint.
Consistent with the 8B, whose P1 U-curved and whose P2 sourced from the global
minimum (step 1200) rather than the more-trained final (step 3000).

## Colab 2026.06 compatibility
- Triton shim for bnb / triton 3.x
- SentryMirror with has_weights(dst) verification
- may need to use pip install -U bitsandbytes>=0.46.1

## envi
"""

import os, sys, types
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Triton shim
fake_perf = types.ModuleType('triton.ops.matmul_perf_model')
fake_perf.early_config_prune = lambda *a, **k: []
fake_perf.estimate_matmul_time = lambda *a, **k: 0
sys.modules['triton.ops'] = types.ModuleType('triton.ops')
sys.modules['triton.ops.matmul_perf_model'] = fake_perf

import torch, gc
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

torch.cuda.empty_cache()
gc.collect()

from google.colab import drive
drive.mount('/content/drive')

from huggingface_hub import login
login()

# find best checkpoint, only uncomment if needed

# import json
# from pathlib import Path
# sf = list(Path("/content/drive/MyDrive/wake2vec_mistral7b_p1").rglob("trainer_state.json"))
# logs = json.load(open(max(sf, key=lambda p: p.stat().st_mtime))).get("log_history", [])
# vals = [(d["step"], d["eval_loss"]) for d in logs if "eval_loss" in d]
# best_step, best_val = min(vals, key=lambda x: x[1])
# print(f"Global min: {best_val:.4f} at step {best_step}")

"""## config"""

from pathlib import Path

P1_SOURCE = Path("/content/drive/MyDrive/wake2vec_mistral7b_p1/full_checkpoints/step_1200")

# P2 outputs 
RUN_DIR = Path("/content/drive/MyDrive/wake2vec_mistral7b_p2_lora")
LOCAL_RUN = Path("/content/runs/wake2vec_mistral7b_p2_lora")
SENTRY = RUN_DIR / "sentry_backups"
FULL_CHECKPOINTS = RUN_DIR / "full_checkpoints"
TRAINER_STATES = RUN_DIR / "trainer_states"

for d in [RUN_DIR, LOCAL_RUN, SENTRY, FULL_CHECKPOINTS, TRAINER_STATES]:
    d.mkdir(parents=True, exist_ok=True)

# model
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
FW_TEXT = "/content/FW_TEXT.txt"

# training hyperparams 
MAX_STEPS = 3000
LR = 2e-5
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
BATCH_SIZE = 1
GRAD_ACCUM = 16            # effective batch 16
SEQ_LEN = 512              # use 256 if vram oom 
SAVE_STEPS = 50          
LOG_STEPS = 50
EVAL_STEPS = 100

# resume
RESUME_FROM = None
# RESUME_FROM = SENTRY / "checkpoint-200"

# LoRA
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

# base vocab (Mistral v0.3) 
BASE_VOCAB = 32768

print("Mistral 7B v0.3 LoRA config")
print(f"  P1 source: {P1_SOURCE}")
print(f"  Output: {RUN_DIR}")
print(f"  Steps: {MAX_STEPS}")
print(f"  LR: {LR}")
print(f"  Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  SEQ_LEN: {SEQ_LEN}")
print(f"  Eval every: {EVAL_STEPS} | Save every: {SAVE_STEPS}")
print(f"  LoRA rank: {LORA_RANK}, targets: {LORA_TARGETS}")

"""## get P1 state"""

from transformers import AutoTokenizer

assert P1_SOURCE.exists(), f"P1 source not found: {P1_SOURCE}"
assert (P1_SOURCE / "embeddings.pt").exists(), "P1 embeddings.pt not found"

tok = AutoTokenizer.from_pretrained(str(P1_SOURCE), use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

TOTAL_VOCAB = len(tok)
NUM_WAKE = TOTAL_VOCAB - BASE_VOCAB
print(f"  Vocab size: {TOTAL_VOCAB}")
print(f"  Base vocab: {BASE_VOCAB}")
print(f"  Wake tokens: {NUM_WAKE}")

embed_weights = torch.load(P1_SOURCE / "embeddings.pt", map_location="cpu")
print(f"  Shape: {embed_weights.shape}")
assert embed_weights.shape[0] == TOTAL_VOCAB, \
    f"Embedding/vocab mismatch: {embed_weights.shape[0]} vs {TOTAL_VOCAB}"

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
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}

if not os.path.exists(FW_TEXT):
    raise FileNotFoundError(f"FW text not found: {FW_TEXT}")
with open(FW_TEXT, 'r', encoding='utf-8') as f:
    text = f.read()

ids = tok(text, add_special_tokens=False)["input_ids"]
print(f"  Total tokens: {len(ids)}")

blocks = [ids[i:i + SEQ_LEN] for i in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN)
          if len(ids[i:i + SEQ_LEN]) == SEQ_LEN]
print(f"  Total blocks: {len(blocks)}")

train_blocks, val_blocks = train_test_split(blocks, test_size=0.10, random_state=42)
train_ds = BlockDataset(train_blocks, SEQ_LEN)
val_ds = BlockDataset(val_blocks, SEQ_LEN)
print(f"  Train: {len(train_ds)} blocks | Val: {len(val_ds)} blocks")

"""## model setup"""

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
    dtype=torch.bfloat16,           # `dtype` not `torch_dtype` 
    device_map="auto",
    max_memory={0: "13GB", "cpu": "30GB"}
)

model.config.use_cache = False
model.config.attn_implementation = "eager"
model.config.tie_word_embeddings = False
print(f"  VRAM after load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# resize to P1 vocab 
print(f"Resizing embeddings: {BASE_VOCAB} -> {TOTAL_VOCAB}...")
model.resize_token_embeddings(TOTAL_VOCAB, mean_resizing=False)

# load P1 embeds into the input embedding layer
wte = model.get_input_embeddings()
with torch.no_grad():
    wte.weight.copy_(embed_weights.to(wte.weight.device, dtype=wte.weight.dtype))

if hasattr(model, "lm_head"):
    model.lm_head.weight = wte.weight
print("P1 embeddings loaded and manually tied (no tie_weights() call)")

del embed_weights
gc.collect()

# freeze everything
for p in model.parameters():
    p.requires_grad = False
print("All parameters frozen")

# add LoRA adapters
peft_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

wte = model.get_input_embeddings()
print(f"Embedding requires_grad: {wte.weight.requires_grad}")
assert not wte.weight.requires_grad, "Embeddings should be frozen in P2"

"""## save frozen Embeds"""

import shutil
E_frozen = model.get_input_embeddings().weight.detach().cpu().clone()
torch.save(E_frozen, RUN_DIR / "embeddings_frozen.pt")
torch.save(E_frozen, RUN_DIR / "embeddings_pre.pt")   # drift baseline (~0, frozen)
print(f"  Saved: {RUN_DIR / 'embeddings_frozen.pt'}  shape {E_frozen.shape}")

"""## callbacks"""

import time, json

def has_weights(ck):
    return (ck / "adapter_model.safetensors").exists() or (ck / "pytorch_model.bin").exists()

from transformers import TrainerCallback

class FullCheckpoint(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        try:
            step = state.global_step
            mdl = kwargs.get("model", model)
            full_ck = FULL_CHECKPOINTS / f"step_{step:04d}"
            if full_ck.exists():
                shutil.rmtree(full_ck)
            full_ck.mkdir(parents=True, exist_ok=True)
            mdl.save_pretrained(full_ck)       # LoRA adapter only
            tok.save_pretrained(full_ck)
            torch.save({'global_step': step, 'best_metric': state.best_metric,
                        'epoch': state.epoch}, full_ck / "training_state.pt")
            print(f"[FULL] Step {step}: adapter saved to Drive")
        except Exception as e:
            print(f"[FULL] Step {state.global_step}: {e}")

class SentryMirror(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        try:
            cks = sorted(LOCAL_RUN.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1]), reverse=True)
            if not cks:
                return
            ck = cks[0]
            if not has_weights(ck):
                print(f"[SENTRY] {ck.name}: local has no weights yet, skipping")
                return
            dst = SENTRY / ck.name
            if dst.exists():
                if has_weights(dst):
                    return
                print(f"[SENTRY] {ck.name}: incomplete mirror, retrying")
                shutil.rmtree(dst)
            shutil.copytree(ck, dst)
            print(f"[SENTRY] {ck.name}: mirrored" if has_weights(dst)
                  else f"[SENTRY] {ck.name}: WARNING weights missing after copy")
        except Exception as e:
            print(f"[SENTRY] {e}")

class TrainerStateMirror(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        try:
            cks = sorted(LOCAL_RUN.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1]), reverse=True)
            if not cks:
                return
            local_ts = cks[0] / "trainer_state.json"
            if local_ts.exists():
                shutil.copy(local_ts, TRAINER_STATES / f"trainer_state_step_{state.global_step:04d}.json")
                shutil.copy(local_ts, TRAINER_STATES / "trainer_state_latest.json")
        except Exception as e:
            print(f"[TS] {e}")

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

"""## trainer & train"""

from transformers import TrainingArguments, Trainer

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
    warmup_steps=int(MAX_STEPS * WARMUP_RATIO),
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
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=val_ds,
    callbacks=[FullCheckpoint(), SentryMirror(), TrainerStateMirror(),
               LossMonitor(), StepTimer()],
)

print(f"  Train: {len(train_ds)} blocks | Val: {len(val_ds)} blocks")
print(f"  Steps: {MAX_STEPS} | Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  VRAM before training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

t0 = time.time()
if RESUME_FROM is not None and RESUME_FROM.exists():
    local_ckpt = LOCAL_RUN / RESUME_FROM.name
    if not local_ckpt.exists():
        shutil.copytree(RESUME_FROM, local_ckpt)
    print(f"[RESUME] {RESUME_FROM.name}")
    trainer.train(resume_from_checkpoint=str(local_ckpt))
else:
    trainer.train()
elapsed = (time.time() - t0) / 60
print(f"\nTRAINING COMPLETE ({elapsed:.1f} minutes)")

final_dir = RUN_DIR / "final"
final_dir.mkdir(exist_ok=True)
model.save_pretrained(str(final_dir))
tok.save_pretrained(str(final_dir))
shutil.copy(RUN_DIR / "embeddings_frozen.pt", final_dir / "embeddings.pt")
print(f"Final LoRA + tokenizer saved to {final_dir}")

"""## loss curve"""

import matplotlib.pyplot as plt

candidates = [TRAINER_STATES / "trainer_state_latest.json"]
candidates += sorted(TRAINER_STATES.glob("trainer_state_step_*.json"), reverse=True)
candidates += list(LOCAL_RUN.rglob("trainer_state.json"))

state_to_use = next((c for c in candidates if c.exists()), None)
if state_to_use is not None:
    print(f"Using trainer state: {state_to_use}")
    with open(state_to_use) as f:
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
        plt.title('Wake2Vec P2: Mistral 7B v0.3 LoRA Loss Curve')
        plt.legend(); plt.grid(True, alpha=0.3)
        plot_path = RUN_DIR / "p2_mistral7b_loss_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {plot_path}")
        plt.show()
        print(f"\nFinal train loss: {losses[-1]:.4f}")
        if val_data:
            print(f"Final val loss: {v_losses[-1]:.4f}")
            print(f"Best val loss: {min(v_losses):.4f}")
else:
    print("No trainer_state.json found anywhere, reconstruct from devlog tables")

"""## wake generation / temperature sweep"""

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
    print(f"-- temp={temperature} | top_p={top_p} | top_k={top_k} | rep={repetition_penalty} --")
    for i, seq in enumerate(outputs):
        gen = tok.decode(seq[prompt_len:], skip_special_tokens=True)
        if num_return_sequences > 1:
            print(f"\n[{i+1}]")
        print(gen)
    print("-" * 60)

def temperature_sweep(prompt, temps=[0.5, 0.7, 0.9, 1.0, 1.2], **kwargs):
    print(f"PROMPT: {prompt}\n")
    for t in temps:
        generate_wake(prompt, temperature=t, **kwargs)
        print()

generate_wake("riverrun, past Eve and Adam's,")
