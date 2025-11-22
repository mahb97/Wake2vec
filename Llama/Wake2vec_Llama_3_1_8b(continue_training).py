This notebook trains Llama-3.2-1B on Finnegans Wake vocabulary.
It includes TWO training sections:
  1. Fresh training: 0 → MAX_STEPS (or until T4 dies)
  2. Resume training: checkpoint-600 → 1100

IMPORTANT: This is a Colab notebook. Shell commands (lines starting with !) 
must be run in Colab cells. The runtime restarts are necessary for bitsandbytes.
"""


# ============================================================================
# ## Wake2Vec “*F* the Embeddings” (T4 Edn)
# This notebook is a colab-friendly, embedding-only finetune pipeline for large decoder LMs (Mistral-7B / Llama-2-13B / Llama-3.1-8B) using a Wake lexicon injection. It adds Joyce-specific tokens, initializes them on a sphere, and trains only the input embedding rows (optionally with a minimal LoRA r=1 on q_proj to satisfy quantized-training rules). The goal is to bend local geometry (neighbors, isotropy) while keeping the rest of the model frozen.
# 
# 
# 
# 
# 
# 
# 
# ============================================================================


# ============================================================================
# # p2
# max_steps: 1500-2500
# 
# lr: 5e-4 → 1e-4 (cosine decay)
# 
# batch_size: 1
# 
# grad_accum: 16
# 
# Custom loss = LM_loss + λ₁·attraction + λ₂·repulsion + λ₃·morphological + λ₄·adversarial
# ============================================================================


# ============================================================================
# guardrail
# ============================================================================


#==============================================================================
# SECTION 0: ENVIRONMENT SETUP (CRITICAL - RUN FIRST)
# Uninstalls conflicting packages and restarts runtime.
# This is NECESSARY to avoid bitsandbytes import errors.
#==============================================================================

# ===== CELL 1 =====
# NUCLEAR OPT
!pip uninstall -y torch torchvision torchaudio triton bitsandbytes transformers accelerate peft fastai timm

# rr
import os
os.kill(os.getpid(), 9)


#==============================================================================
# SECTION 1: INSTALL COMPATIBLE VERSIONS
# Installs exact package versions that work together on T4.
#==============================================================================

# ===== CELL 2 =====
# Stop TorchAO
import os
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"

# compat versions
!pip install -q --no-cache-dir \
    torch==2.5.1 \
    triton==3.1.0 \
    bitsandbytes==0.43.3 \
    transformers==4.45.2 \
    accelerate==0.34.2 \
    peft==0.13.2

# Verify
import torch, bitsandbytes as bnb, triton
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("bnb:", bnb.__version__, "| triton:", triton.__version__)

# ===== CELL 3 =====
# stop TorchAO
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from getpass import getpass
from huggingface_hub import login

HF_TOKEN = getpass("Paste your HF token (hidden): ")
login(token=HF_TOKEN, add_to_git_credential=True)

# ===== CELL 4 =====
# Val dataset
from torch.utils.data import Dataset
import torch

class BlockDataset(Dataset):
    """Sliding window dataset for causal LM training."""
    def __init__(self, txt_path, tokenizer, seq_len=512, stride=512):
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokens = tokenizer(text, add_special_tokens=False)['input_ids']
        self.seq_len = seq_len
        self.stride = stride
        self.starts = list(range(0, len(self.tokens) - seq_len + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        chunk = self.tokens[start:start + self.seq_len]
        return {'input_ids': torch.tensor(chunk, dtype=torch.long)}

from transformers import AutoTokenizer

print("Testing dataset creation...")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_fast=True)

ds = BlockDataset("/content/FW_TEXT.txt", tok, seq_len=512, stride=512)
print(f"  Dataset size: {len(ds)} blocks")
print(f"  First block shape: {ds[0]['input_ids'].shape}")

print("\nDataset validated")

# ===== CELL 5 =====
# Helper functions
def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

class BlockDataset(Dataset):
    """Sliding window dataset for causal LM training."""
    def __init__(self, txt_path, tokenizer, seq_len=512, stride=512):
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokens = tokenizer(text, add_special_tokens=False)['input_ids']
        self.seq_len = seq_len
        self.stride = stride
        self.starts = list(range(0, len(self.tokens) - seq_len + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        chunk = self.tokens[start:start + self.seq_len]
        return {'input_ids': torch.tensor(chunk, dtype=torch.long)}

# ===== CELL 6 =====
# Pre-train
import os

print("GPU Check:")
print(f"  Device: {torch.cuda.get_device_name(0)}")
print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

print("\nFile Check:")
files = {
    "Wake Lexicon": "/content/wake_lexicon.txt",
    "FW Text": "/content/FW_TEXT.txt"
}
for name, path in files.items():
    exists = os.path.exists(path)
    status = "Found" if exists else "MISSING"
    print(f"  {name}: {status}")
    if exists:
        size = os.path.getsize(path) / 1024
        print(f"    Size: {size:.1f} KB")

# ===== CELL 7 =====
import gc
import torch

# Verify clean slate
torch.cuda.empty_cache()
gc.collect()

print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# ===== CELL 8 =====
# memory footprint
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

MODEL_NAME = "meta-llama/Llama-3.2-3B"

print(f"Testing {MODEL_NAME} load on T4")

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("\nLoading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"  Vocab size: {len(tok)}")

print("\nLoading model with 4-bit quantization...")
torch.cuda.empty_cache()
initial_mem = torch.cuda.memory_allocated(0) / 1e9

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb,
    torch_dtype=torch.float16,
    device_map="auto"
)

loaded_mem = torch.cuda.memory_allocated(0) / 1e9
print(f"  Model loaded: {loaded_mem:.2f} GB")
print(f"  Delta: {loaded_mem - initial_mem:.2f} GB")

# Validate forward pass
print("\nTesting forward pass...")
test_ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
with torch.no_grad():
    out = model(test_ids)
    print(f"  Output shape: {out.logits.shape}")
    print(f"  Memory after forward: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

print(f"\nPeak memory: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
print("Model validated successfully")

# Cleanup
del model, tok
torch.cuda.empty_cache()
print("Memory cleared for main run")


#==============================================================================
# SECTION 2: GPU MEMORY CHECKS (T4 CRITICAL)
# Verifies clean GPU state and tests model loading.
#==============================================================================

# ===== CELL 9 =====
import os, math, json, random, torch, shutil
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, TrainerCallback, set_seed)
from peft import LoraConfig, get_peft_model

SEED=42; set_seed(SEED)
MODEL_NAME = "meta-llama/Llama-3.2-1B"
WAKE_LEX_PATH = "/content/wake_lexicon.txt"
CORPUS_TXT = "/content/finnegans_wake.txt"

# CRITICAL: Save to Drive, not /content
RUN_DIR = Path("/content/drive/MyDrive/wake_llama_P1")
LOCAL_RUN = Path("/content/runs/wake_llama_P1")
SENTRY = RUN_DIR / "sentry_backups"

RUN_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_RUN.mkdir(parents=True, exist_ok=True)
SENTRY.mkdir(parents=True, exist_ok=True)

SEQ_LEN=512; STRIDE=512
MAX_STEPS=1100; LOG_STEPS=20; SAVE_STEPS=200
LR=5e-5
GRAD_ACCUM=8
REPULSION_W=0.0
TARGET_NORM=None
MAX_ROW_NORM=None
REPORT_SAMPLE=1500

# 4-bit quantization
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True
)

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "13GB", "cpu": "30GB"}
)
model.config.use_cache = False
model.config.attn_implementation = "eager"
model.config.tie_word_embeddings = True
if hasattr(model, "tie_weights"): model.tie_weights()

# Frozen PEFT adapter
peft_cfg = LoraConfig(r=1, lora_alpha=1, lora_dropout=0.0,
                      target_modules=["q_proj"], bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, peft_cfg)
for n,p in model.named_parameters(): p.requires_grad=False

# Wake vocab injection
def read_lines(p):
    return [x.strip() for x in open(p, encoding="utf-8") if x.strip()] if os.path.exists(p) else []

wake = read_lines(WAKE_LEX_PATH)
missing = [t for t in wake if tok.convert_tokens_to_ids(t)==tok.unk_token_id]
num_added = tok.add_tokens(missing, special_tokens=False)

old_vocab = model.get_input_embeddings().weight.shape[0]
model.resize_token_embeddings(len(tok))
wte = model.get_input_embeddings()
if hasattr(model, "lm_head"): model.lm_head.weight = wte.weight

# Spherical init
with torch.no_grad():
    base = wte.weight[:old_vocab]; dim = base.shape[1]
    std = base.std().item(); base_radius = std * math.sqrt(dim)
    target_radius = TARGET_NORM or (1.5 * base_radius)
    if num_added>0:
        new = torch.randn((num_added, dim), device=wte.weight.device)
        new = new/(new.norm(dim=1, keepdim=True)+1e-8)*target_radius
        wte.weight.data[old_vocab:old_vocab+num_added] = new

# Only embeddings trainable
wte.weight.requires_grad=True
new_rows = torch.arange(old_vocab, old_vocab+num_added, device=wte.weight.device) if num_added>0 else None
base_rows = torch.arange(0, old_vocab, device=wte.weight.device)

def mask_grad(grad):
    if grad is None or new_rows is None: return grad
    grad[base_rows]=0; return grad
wte.weight.register_hook(mask_grad)

# Dataset
class BlockDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len=512, stride=512):
        if not os.path.exists(path):
            stub = ("riverrun, past Eve and Adam's, from swerve of shore to bend of bay, "
                    "brings us by a commodius vicus of recirculation to Howth Castle and Environs. ")*2000
            text = stub
        else:
            text = open(path, "r", encoding="utf-8").read()
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        blocks=[]
        for i in range(0, max(1, len(ids)-seq_len), stride):
            chunk = ids[i:i+seq_len]
            if len(chunk) >= seq_len//2:
                blocks.append(chunk[:seq_len])
        self.blocks = blocks
    def __len__(self): return len(self.blocks)
    def __getitem__(self, idx):
        ids = torch.tensor(self.blocks[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}

train_ds = BlockDataset(CORPUS_TXT, tok, SEQ_LEN, STRIDE)
print(f"[Data] chunks={len(train_ds)}; tokens/step={SEQ_LEN}")

# Sentry callback
def has_weights(ck):
    return (ck/"adapter_model.safetensors").exists() or (ck/"pytorch_model.bin").exists()

class SentryMirror(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        try:
            cks = sorted(LOCAL_RUN.glob("checkpoint-*"),
                        key=lambda p: int(p.name.split("-")[-1]),
                        reverse=True)
            if not cks:
                return
            ck = cks[0]
            if not has_weights(ck):
                print(f"[SENTRY] {ck.name} no weights, skip")
                return
            dst = SENTRY / ck.name
            if not dst.exists():
                print(f"[SENTRY] Mirroring {ck.name}...")
                shutil.copytree(ck, dst)
                print(f"[SENTRY] {ck.name} backed up to Drive")
            os.sync()
        except Exception as e:
            print(f"[SENTRY] ERROR: {e}")

# Custom trainer
class EmbOnlyTrainer(Trainer):
    def create_optimizer(self):
        from torch.optim import AdamW
        if not hasattr(self, "optimizer") or self.optimizer is None:
            self.optimizer = AdamW([{"params": [wte.weight], "lr": LR, "weight_decay": 0.0}],
                                   betas=(0.9, 0.999), eps=1e-8)
        return self.optimizer
    def compute_loss(self, model, inputs, return_outputs=False):
        out = model(**inputs, use_cache=False)
        loss = out.loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("NaN/Inf loss detected")
        return (loss, out) if return_outputs else loss

args = TrainingArguments(
    output_dir=str(LOCAL_RUN),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    max_steps=MAX_STEPS,
    warmup_steps=max(20, MAX_STEPS//20),
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
    callbacks=[SentryMirror()]
)


#==============================================================================
# SECTION 3: FRESH TRAINING (0 → MAX_STEPS)
# Run this section to train from scratch.
# T4 will likely die around step 600-800 due to memory.
#==============================================================================

print(f"[Run] {MODEL_NAME} | steps={MAX_STEPS} | seq_len={SEQ_LEN}")
trainer.train()

# Save final artifacts to Drive
save_dir = RUN_DIR / "final"
save_dir.mkdir(exist_ok=True)
torch.save(wte.weight.detach().cpu(), save_dir / "embed_tokens.pt")
tok.save_pretrained(str(save_dir))
print(f"[SAVED] Final artifacts to {save_dir}")


# ============================================================================
# continue from 600 ⬇
# ============================================================================

# ===== CELL 10 =====
# Force reinstall bitsandbytes with CUDA support
!pip uninstall -y bitsandbytes
!pip install bitsandbytes==0.43.3 --no-cache-dir

# CRITICAL: Restart the runtime
import os
os.kill(os.getpid(), 9)

# ===== CELL 11 =====
# Better verification for bitsandbytes 0.43.3
import torch
import bitsandbytes as bnb

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"bitsandbytes version: {bnb.__version__}")

# Test if bitsandbytes CUDA functions work
try:
    test_tensor = torch.randn(10, 10).cuda()
    quantized = bnb.functional.quantize_4bit(test_tensor)
    print("✓ bitsandbytes CUDA quantization working!")
except Exception as e:
    print(f"✗ bitsandbytes CUDA test failed: {e}")

# ===== CELL 12 =====
# Stop TorchAO
import os
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"

# Install compatible versions
!pip install -q --no-cache-dir \
    torch==2.5.1 \
    triton==3.1.0 \
    bitsandbytes==0.43.3 \
    transformers==4.45.2 \
    accelerate==0.34.2 \
    peft==0.13.2

import torch, bitsandbytes as bnb, triton
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("bnb:", bnb.__version__, "| triton:", triton.__version__)
print("✓ All packages installed successfully")

# ===== CELL 13 =====
from google.colab import drive
drive.mount('/content/drive')

# ===== CELL 14 =====
# HF login
from huggingface_hub import login

login()


#==============================================================================
# SECTION 4: RESUME FROM CHECKPOINT (600 → 1100)
# Run these cells INSTEAD of Section 3 to resume from checkpoint-600.
# This section reconstructs the model and loads saved embeddings.
#==============================================================================

# ===== CELL 15 =====
from pathlib import Path

CHECKPOINT_NUM = 600

# paths
RUN_DIR = Path("/content/drive/MyDrive/wake_llama_P1")
SENTRY = RUN_DIR / "sentry_backups"
CHECKPOINT_DIR = SENTRY / f"checkpoint-{CHECKPOINT_NUM}"

# file paths
WAKE_LEX_PATH = "/content/wake_lexicon.txt"
FW_TEXT = "/content/FW_TEXT.txt"

# checkpoint
print(f"Checkpoint path: {CHECKPOINT_DIR}")
print(f"Exists: {CHECKPOINT_DIR.exists()}")

if not CHECKPOINT_DIR.exists():
    print("\nAvailable checkpoints in Drive:")
    for ck in sorted(SENTRY.glob("checkpoint-*")):
        print(f"  - {ck.name}")

# ===== CELL 16 =====
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, get_peft_model

SEED = 42
set_seed(SEED)
MODEL_NAME = "meta-llama/Llama-3.2-1B"

# 4-bit quantization config
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "13GB", "cpu": "30GB"}
)

# ===== CELL 17 =====
# Check what's in the checkpoint directory
from pathlib import Path

print(f"Checking contents of: {CHECKPOINT_DIR}\n")
print("Files in checkpoint:")
for f in sorted(CHECKPOINT_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")

# ===== CELL 18 =====
from pathlib import Path

# Possible tokenizer locations
possible_locations = [
    RUN_DIR / "final",  # Final save location
    RUN_DIR,  # Root directory
    Path("/content/runs/wake_llama_P1"),  # Local run dir
]

tok = None
for loc in possible_locations:
    if (loc / "tokenizer_config.json").exists():
        print(f"Found tokenizer at: {loc}")
        tok = AutoTokenizer.from_pretrained(str(loc), use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        print(f"✓ Tokenizer loaded, vocab_size: {len(tok)}")
        break

if tok is None:
    print("Tokenizer not found in expected locations. Checking what's in Drive...")
    print("\nContents of RUN_DIR:")
    for item in RUN_DIR.iterdir():
        print(f"  {item.name}")

    # If tokenizer isn't saved, we'll need to reconstruct it
    print("\n Need to reconstruct tokenizer from base + wake lexicon")

# ===== CELL 19 =====
# Reconstruct tokenizer with Wake vocabulary
import os

# Load base tokenizer
print("Loading base tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"Base vocab size: {len(tok)}")

# Load Wake lexicon
WAKE_LEX_PATH = "/content/wake_lexicon.txt"

def read_lines(p):
    return [x.strip() for x in open(p, encoding="utf-8") if x.strip()] if os.path.exists(p) else []

wake = read_lines(WAKE_LEX_PATH)
print(f"Wake lexicon tokens: {len(wake)}")

# Find missing tokens and add them
missing = [t for t in wake if tok.convert_tokens_to_ids(t) == tok.unk_token_id]
print(f"Missing tokens to add: {len(missing)}")

num_added = tok.add_tokens(missing, special_tokens=False)
print(f"✓ Added {num_added} new tokens")
print(f"✓ New vocab size: {len(tok)}")

# Now resize model embeddings
print(f"Resizing model embeddings to {len(tok)}...")
old_vocab = model.get_input_embeddings().weight.shape[0]
model.resize_token_embeddings(len(tok))
wte = model.get_input_embeddings()
if hasattr(model, "lm_head"):
    model.lm_head.weight = wte.weight
print(f"✓ Embeddings resized from {old_vocab} to {len(tok)}")

# Load checkpoint embeddings
from safetensors.torch import load_file

print(f"Loading embeddings from checkpoint...")
checkpoint_state = load_file(CHECKPOINT_DIR / "adapter_model.safetensors")
embed_key = "base_model.model.model.embed_tokens.weight"

with torch.no_grad():
    wte.weight.copy_(checkpoint_state[embed_key])
print(f"✓ Loaded embeddings from checkpoint")

# Only embeddings trainable
wte.weight.requires_grad = True

# Gradient masking (only train new tokens)
base_vocab = 128256  # Original Llama vocab
new_rows = torch.arange(base_vocab, len(tok), device=wte.weight.device) if len(tok) > base_vocab else None
base_rows = torch.arange(0, base_vocab, device=wte.weight.device)

def mask_grad(grad):
    if grad is None or new_rows is None:
        return grad
    grad[base_rows] = 0
    return grad

wte.weight.register_hook(mask_grad)
print(f"✓ Model ready. Training {len(tok) - base_vocab} new tokens only")

# ===== CELL 20 =====
# dataset set up
import os
from torch.utils.data import Dataset

SEQ_LEN = 512
STRIDE = 512

class BlockDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len=512, stride=512):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Text file not found: {path}")
        text = open(path, "r", encoding="utf-8").read()
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        blocks = []
        for i in range(0, max(1, len(ids) - seq_len), stride):
            chunk = ids[i:i + seq_len]
            if len(chunk) >= seq_len // 2:
                blocks.append(chunk[:seq_len])
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        ids = torch.tensor(self.blocks[idx], dtype=torch.long)
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones_like(ids)
        }

train_ds = BlockDataset(FW_TEXT, tok, SEQ_LEN, STRIDE)
print(f"Dataset ready: {len(train_ds)} chunks, {SEQ_LEN} tokens/step")

# ===== CELL 21 =====
# Trainer setup and resume training
import shutil
from transformers import TrainingArguments, Trainer, TrainerCallback

LOCAL_RUN = Path("/content/runs/wake_llama_P1")
LOCAL_RUN.mkdir(parents=True, exist_ok=True)

# Training config
MAX_STEPS = 1100
LOG_STEPS = 20
SAVE_STEPS = 200
LR = 5e-5
GRAD_ACCUM = 8

# Sentry callback
def has_weights(ck):
    return (ck / "adapter_model.safetensors").exists() or (ck / "pytorch_model.bin").exists()

class SentryMirror(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        try:
            cks = sorted(LOCAL_RUN.glob("checkpoint-*"),
                        key=lambda p: int(p.name.split("-")[-1]),
                        reverse=True)
            if not cks:
                return
            ck = cks[0]
            if not has_weights(ck):
                print(f"[SENTRY] {ck.name} no weights, skip")
                return
            dst = SENTRY / ck.name
            if not dst.exists():
                print(f"[SENTRY] Mirroring {ck.name}...")
                shutil.copytree(ck, dst)
                print(f"[SENTRY] {ck.name} backed up to Drive")
            os.sync()
        except Exception as e:
            print(f"[SENTRY] ERROR: {e}")

# Custom trainer
class EmbOnlyTrainer(Trainer):
    def create_optimizer(self):
        from torch.optim import AdamW
        if not hasattr(self, "optimizer") or self.optimizer is None:
            # Ensure embeddings are trainable
            wte.weight.requires_grad = True
            self.optimizer = AdamW(
                [{"params": [wte.weight], "lr": LR, "weight_decay": 0.0}],
                betas=(0.9, 0.999), eps=1e-8
            )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        out = model(**inputs, use_cache=False)
        loss = out.loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError("NaN/Inf loss detected")
        return (loss, out) if return_outputs else loss

# CRITICAL: Re-enable gradients on embeddings
print("Re-enabling gradients on embeddings...")
wte.weight.requires_grad = True

# Verify trainable parameters
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable parameters: {len(trainable_params)}")
print(f"Embedding requires_grad: {wte.weight.requires_grad}")

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
    callbacks=[SentryMirror()]
)

# Load trainer state
import json
trainer_state_data = json.load(open(CHECKPOINT_DIR / "trainer_state.json"))

# Set the trainer state to resume from step 600
trainer.state.global_step = trainer_state_data["global_step"]
trainer.state.epoch = trainer_state_data.get("epoch", 0)

print(f"Resuming from step {trainer.state.global_step} → {MAX_STEPS}")

# Create optimizer first (needed before loading state)
trainer.create_optimizer()
trainer.create_scheduler(num_training_steps=MAX_STEPS, optimizer=trainer.optimizer)

# Now load optimizer and scheduler states
print("Loading optimizer and scheduler state...")
optimizer_state = torch.load(CHECKPOINT_DIR / "optimizer.pt", map_location="cpu", weights_only=False)
scheduler_state = torch.load(CHECKPOINT_DIR / "scheduler.pt", map_location="cpu", weights_only=False)

trainer.optimizer.load_state_dict(optimizer_state)
trainer.lr_scheduler.load_state_dict(scheduler_state)

print(f"✓ Optimizer and scheduler restored from checkpoint-{CHECKPOINT_NUM}")
print(f"Final check - embedding requires_grad: {wte.weight.requires_grad}")

# Train normally - will continue from step 600
trainer.train()

# Save final artifacts
save_dir = RUN_DIR / "final"
save_dir.mkdir(exist_ok=True)
torch.save(wte.weight.detach().cpu(), save_dir / "embed_tokens.pt")
tok.save_pretrained(str(save_dir))
print(f"✓ Saved final artifacts to {save_dir}")


#==============================================================================
# SECTION 5: EVALUATION
# Analyzes final embeddings and creates loss plots.
#==============================================================================

# ===== CELL 22 =====
# P1 Eval (Llama-3.2-1B Complete)
import json, numpy as np, torch
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
WAKE2VEC_ROOT = Path("/content/drive/MyDrive/wake_llama_P1")
SENTRY = WAKE2VEC_ROOT / "sentry_backups"
LOCAL_RUN = Path("/content/runs/wake_llama_P1")

# Find latest checkpoint (check both local and Drive backup)
local_checkpoints = sorted(LOCAL_RUN.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
sentry_checkpoints = sorted(SENTRY.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))

# Use the latest one available
if local_checkpoints:
    BASE_CKPT = local_checkpoints[-1]
    print(f"[P1 EVAL] Using LOCAL checkpoint: {BASE_CKPT}")
elif sentry_checkpoints:
    BASE_CKPT = sentry_checkpoints[-1]
    print(f"[P1 EVAL] Using SENTRY checkpoint: {BASE_CKPT}")
else:
    raise FileNotFoundError("No checkpoints found in local or Drive backup!")

print(f"Checkpoint: {BASE_CKPT.name}")

# Load embeddings from checkpoint
checkpoint_state = torch.load(BASE_CKPT / "pytorch_model.bin", map_location="cpu")

# Find embedding key
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

# Save report
(WAKE2VEC_ROOT / "p1_llama_summary.json").write_text(json.dumps(report, indent=2))
print("\n[P1 SUMMARY]")
print(json.dumps(report, indent=2))

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
