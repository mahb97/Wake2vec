# -*- coding: utf-8 -*-
"""Wake2Vec Phase 2: Qwen 2.5-14B LoRA Fine-Tune (post-WakeOverlay)

# Wake2Vec P2: Qwen 2.5-14B LoRA from canonical P1

**Model:** Qwen/Qwen2.5-14B (4-bit NF4, WakeOverlay → LoRA)
**Hardware:** Google Colab T4 GPU + CPU offload
**P1 source:** sentry_step_2700.pt (best val 15.05 @ step 2700, 6th SGDR low)
**Training data:** FW corpus + wake_lexicon.txt

## Architecture transition: WakeOverlay & standard LoRA

P1 used WakeOverlay (base frozen, separate Wake-row matrix trained, no LoRA).
P2 reverts to the standard project P2 protocol:
1. Reconstruct full embedding matrix (base ⊕ canonical Wake rows from sentry)
2. Freeze ALL parameters including embeddings (P2 is behavioural adaptation, not embedding adaptation)
3. Apply LoRA (r=8, α=16) to attention + MLP projections
4. Train ONLY the LoRA adapters

This is the same protocol Llama 3B P2 ran. Direct cross-architecture comparability
with the LoRA-ceiling wall finding (val 5.33 across 6 evals on 3B P2).

## Lessons from Qwen P1 baked in

1. **`llm_int8_enable_fp32_cpu_offload=True`** in BitsAndBytesConfig
   (without this, accelerate dispatches layers to CPU and bnb errors out)
2. **`max_memory={0: "13GB", "cpu": "30GB"}`** in from_pretrained
   (explicit budget; auto-mapping is too conservative)
3. **`mean_resizing=False`** in resize_token_embeddings
   (the default mean_resizing OOM's by allocating fp32 base copies for covariance)
4. **TrainerStateMirror callback** copies trainer_state.json to Drive at every save
   (P1 lost trainer_state.json in the 9 June Colab cut; devlog reconstruction worked
   but the JSON copy is cheap insurance)
5. **SentryMirror for LoRA-only** is small — LoRA adapter is ~10-50MB regardless
   of base model size. Save aggressively (every 25 steps).

## Colab 2026.06 compatibility

- Triton shim for bnb / triton 3.x
- eval_strategy (not evaluation_strategy) for transformers 5.0.0
- trust_remote_code=True for Qwen tokenizer

-------------------------------------------------------------

## envi
"""

import os
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# triton shim for bnb / triton 3.x
import types, sys
_fake = types.ModuleType("triton.ops")
_fake.matmul_perf_model = types.ModuleType("triton.ops.matmul_perf_model")
sys.modules.setdefault("triton.ops", _fake)
sys.modules.setdefault("triton.ops.matmul_perf_model", _fake.matmul_perf_model)

import torch, gc
print("ENVI")
print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")
try:
    import bitsandbytes as bnb_lib
    print(f"bitsandbytes: {bnb_lib.__version__}")
except ImportError:
    print("bitsandbytes is NOT INSTALLED")
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

"""## config"""

import math
from pathlib import Path

# hi qween 
MODEL_NAME = "Qwen/Qwen2.5-14B"
BASE_VOCAB = 152064  # base vocab

# P1 source for canonical sentry, 
P1_RUN_DIR = Path("/content/drive/MyDrive/wake2vec_qwen14b_p1")
P1_SENTRY = P1_RUN_DIR / "sentry_backups" / "sentry_step_2700.pt"  # best val 15.05
P1_EMB_PRE = P1_RUN_DIR / "embeddings_pre.pt"  # for cross-check, not training-time use

# data
FW_TEXT = "/content/FW_TEXT.txt"
WAKE_LEXICON = "/content/wake_lexicon.txt"

# outputs
RUN_DIR = Path("/content/drive/MyDrive/wake2vec_qwen14b_p2_lora")
LOCAL_RUN = Path("/content/runs/wake2vec_qwen14b_p2_lora")
SENTRY = RUN_DIR / "sentry_backups"
FULL_CHECKPOINTS = RUN_DIR / "full_checkpoints"
TRAINER_STATES = RUN_DIR / "trainer_states"  # trainer_state.json mirror

for d in [RUN_DIR, LOCAL_RUN, SENTRY, FULL_CHECKPOINTS, TRAINER_STATES]:
    d.mkdir(parents=True, exist_ok=True)

# training hyperparams 
MAX_STEPS = 3000           # early-stop if wall hit
LR = 5e-5                  
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
BATCH_SIZE = 1
GRAD_ACCUM = 16            # effective batch 16
SEQ_LEN = 128              
SAVE_STEPS = 25            # aggressive 
LOG_STEPS = 25
EVAL_STEPS = 50            
EARLY_STOP_PATIENCE = 4    

# LoRA config 
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

# resume 
RESUME_FROM = None
# RESUME_FROM = SENTRY / "checkpoint-200"

print("Wake2Vec P2 on Qwen 2.5-14B with LoRA config")
print(f"  Model: {MODEL_NAME}")
print(f"  P1 source: {P1_SENTRY}")
print(f"  Output: {RUN_DIR}")
print(f"  Steps: {MAX_STEPS} (early stop patience {EARLY_STOP_PATIENCE})")
print(f"  LR: {LR}")
print(f"  Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  SEQ_LEN: {SEQ_LEN}")
print(f"  Eval cadence: every {EVAL_STEPS} steps")
print(f"  Save cadence: every {SAVE_STEPS} steps (aggressive — Colab cut insurance)")
print(f"  LoRA: r={LORA_RANK}, α={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"  LoRA targets: {LORA_TARGETS}")
print(f"  Resume: {RESUME_FROM}")

"""## tokenizer (deterministic rebuild from wake_lexicon.txt)"""

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"  Base vocab: {len(tok)}")

# get  Wake lexicon, add tokens deterministically 
print(f"getting Wake lexicon from {WAKE_LEXICON}...")
with open(WAKE_LEXICON, 'r', encoding='utf-8') as f:
    wake_tokens = [line.strip() for line in f if line.strip()]

missing = [t for t in wake_tokens if tok.convert_tokens_to_ids(t) == tok.unk_token_id]
num_added = tok.add_tokens(missing, special_tokens=False)

print(f"  Wake tokens in lexicon: {len(wake_tokens)}")
print(f"  New tokens added: {num_added}")
print(f"  Final vocab size: {len(tok)}")

# cross-check against canonical sentry expected count
import torch
state = torch.load(P1_SENTRY, map_location="cpu", weights_only=False)
canonical_num_wake = state["num_wake"]  # 43824
canonical_wake_start = state["wake_start"]  # 152064
drift_rows = num_added - canonical_num_wake
if drift_rows != 0:
    print(f"\n  -- tok drift note --")
    print(f"  Canonical P1 had {canonical_num_wake} Wake tokens at indices [{canonical_wake_start}, {canonical_wake_start + canonical_num_wake})")
    print(f"  Today's tokenizer adds {num_added} ({drift_rows:+d} from canonical)")
    print(f"  ~{100*abs(drift_rows)/num_added:.2f}% of Wake-row positions may have different lexical meaning")

"""## dataset"""

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class BlockDataset(Dataset):
    def __init__(self, blocks, seq_len=128):
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

"""## model setup and Lora)"""

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, get_peft_model

set_seed(42)

# 4-bit quant with CPU offload allowed 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "13GB", "cpu": "30GB"},
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.attn_implementation = "eager"
model.config.tie_word_embeddings = True
if hasattr(model, "tie_weights"):
    model.tie_weights()

print(f"  VRAM after load: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# resize embeddings without mean_resizing init 
model.resize_token_embeddings(len(tok), mean_resizing=False)
wte = model.get_input_embeddings()
if hasattr(model, "lm_head"):
    model.lm_head.weight = wte.weight
if hasattr(model, "tie_weights"):
    model.tie_weights()
print(f"  Embedding layer: {wte.weight.shape}")

# inject canonical Wake embeds
canonical_wake = state["embeddings"]  # [43824, 5120] from P1 
wake_start = state["wake_start"]
num_wake = state["num_wake"]

with torch.no_grad():
    target_device = wte.weight.device
    target_dtype = wte.weight.dtype
    wte.weight.data[wake_start:wake_start + num_wake] = (
        canonical_wake.to(device=target_device, dtype=target_dtype)
    )
    if drift_rows > 0:
        print(f"  Note: {drift_rows} extra Wake-row positions retain default resize init")

print(f"  Canonical Wake injection: {num_wake} rows at indices [{wake_start}, {wake_start + num_wake})")
print(f"  VRAM after injection: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

del canonical_wake
gc.collect()
torch.cuda.empty_cache()

# apply LoRA to attention + MLP 
print(f"\nApplying LoRA: r={LORA_RANK}, α={LORA_ALPHA}, targets={LORA_TARGETS}")
peft_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)

# freeze everything except LoRA adapters 
trainable_count = 0
for n, p in model.named_parameters():
    if "lora_" in n:
        p.requires_grad = True
        trainable_count += p.numel()
    else:
        p.requires_grad = False

# freeze the embedding layer (double-check)
model.get_input_embeddings().weight.requires_grad = False

print(f"  Trainable LoRA params: {trainable_count:,} ({trainable_count/1e6:.2f}M)")
print(f"  All base + embedding params frozen")
print(f"  VRAM after LoRA: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

# free the canonical sentry dict
del state
gc.collect()

"""## pre-train LoRA snap"""

# save the initial LoRA state (all zeros) as a sanity-check baseline
import shutil
initial_lora_dir = RUN_DIR / "initial_lora"
if not initial_lora_dir.exists():
    initial_lora_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(initial_lora_dir))
    print(f"  Initial LoRA state saved: {initial_lora_dir}")
else:
    print(f"  Initial LoRA state already exists at {initial_lora_dir}")

"""## callbacks"""

import time, json
from transformers import TrainerCallback

def has_weights(ck):
    return (ck / "adapter_model.safetensors").exists() or (ck / "pytorch_model.bin").exists()

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
            torch.save({
                'global_step': step,
                'best_metric': state.best_metric,
                'epoch': state.epoch,
            }, full_ck / "training_state.pt")
            print(f"[FULL] Step {step}: saved to {full_ck}")
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
            print(f"[SENTRY] {ck.name}: mirrored to Drive")
        except Exception as e:
            print(f"[SENTRY] {e}")

class TrainerStateMirror(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        try:
            step = state.global_step 
            # get the latest checkpoint's trainer_state.json in LOCAL_RUN
            cks = sorted(LOCAL_RUN.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[-1]), reverse=True)
            if not cks:
                return
            local_ts = cks[0] / "trainer_state.json"
            if local_ts.exists():
                drive_ts = TRAINER_STATES / f"trainer_state_step_{step:04d}.json"
                shutil.copy(local_ts, drive_ts)
                # maintain a "latest" pointer for easy access
                latest = TRAINER_STATES / "trainer_state_latest.json"
                shutil.copy(local_ts, latest)
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
                step_logical = state.global_step + STEP_OFFSET
                print(f"[{step_logical:4d}] {avg:.1f}s/step")
        self.last_time = now

"""## trainer + train"""

from transformers import TrainingArguments, Trainer
from transformers.optimization import Adafactor

class LoRATrainer(Trainer):

    def create_optimizer(self):
        if self.optimizer is None:
            lora_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
            print(f"  Optimizer over {len(lora_params)} LoRA tensor params, "
                  f"{sum(p.numel() for p in lora_params):,} total scalar params")
            self.optimizer = Adafactor(
                lora_params,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
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
    warmup_steps=int(MAX_STEPS * WARMUP_RATIO),
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
    optim="adafactor",             
)

trainer = LoRATrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=None,
    callbacks=[
        FullCheckpoint(),
        SentryMirror(),
        TrainerStateMirror(),   
        LossMonitor(),
        StepTimer(),
    ],
)

print("training for Wake2vec P2: Qwen 2.5-14B LoRA")
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

# save final 
final_dir = RUN_DIR / "final"
final_dir.mkdir(exist_ok=True)
model.save_pretrained(str(final_dir))
tok.save_pretrained(str(final_dir))
print(f"Final LoRA + tokenizer saved to {final_dir}")

"""## loss curve"""

import matplotlib.pyplot as plt

# use the Drive-mirrored trainer_state.json 
candidates = [
    TRAINER_STATES / "trainer_state_latest.json",
    LOCAL_RUN / "trainer_state.json",
]
state_files = list(LOCAL_RUN.rglob("trainer_state.json"))
for f in state_files:
    candidates.append(f)
# plus any mirrored ones on Drive
candidates.extend(sorted(TRAINER_STATES.glob("trainer_state_step_*.json"), reverse=True))

state_to_use = None
for c in candidates:
    if c.exists():
        state_to_use = c
        break

if state_to_use is not None:
    print(f"Using trainer state from: {state_to_use}")
    with open(state_to_use) as f:
        state_data = json.load(f)
    logs = state_data.get("log_history", [])
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
        plt.title('Wake2Vec P2: Qwen 2.5-14B LoRA Loss Curve')
        plt.legend(); plt.grid(True, alpha=0.3)
        plot_path = RUN_DIR / "p2_qwen14b_loss_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {plot_path}")
        plt.show()
        print(f"\nFinal train loss: {losses[-1]:.4f}")
        if val_data:
            print(f"Final val loss: {v_losses[-1]:.4f}")
            print(f"Best val loss: {min(v_losses):.4f}")
else:
    print("No trainer_state.json found so use from devlog tables.")

"""## summary JSON"""

report = {
    "model": MODEL_NAME,
    "phase": "P2_LoRA_on_WakeOverlay_canonical",
    "p1_source": str(P1_SENTRY),
    "hyperparameters": {
        "max_steps": MAX_STEPS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "seq_len": SEQ_LEN,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_targets": LORA_TARGETS,
        "optimizer": "adafactor",
    },
    "embedding_strategy": "best-val Wake from sentry_step_2700.pt (P1 6th SGDR low, val 15.05), frozen during P2",
    "wake_start": canonical_wake_start,
    "num_wake_canonical": canonical_num_wake,
    "tokenizer_drift_rows": drift_rows,
    "vocab_size": int(len(tok)),
    "loss": {
        "final_train": float(losses[-1]) if train_data else None,
        "final_eval": float(v_losses[-1]) if val_data else None,
        "best_val": float(min(v_losses)) if val_data else None,
    },
}

summary_path = RUN_DIR / "p2_qwen14b_summary.json"
summary_path.write_text(json.dumps(report, indent=2))
print(f"\n[SUMMARY] {summary_path}")

"""## generation / temperature sweep (P2 LoRA output)"""

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
