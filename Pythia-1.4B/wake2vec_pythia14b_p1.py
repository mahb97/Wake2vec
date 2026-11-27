# -*- coding: utf-8 -*-
"""Wake2Vec Phase 1: Pythia-1.4B Embedding-Only Fine-Tune

# Wake2Vec P1: EleutherAI/pythia-1.4b, embedding-only with gradient masking

**Model:** EleutherAI/pythia-1.4b (4-bit quantised, GPTNeoXForCausalLM)
**Hardware:** Google Colab free T4
**Training data:** Finnegans Wake corpus + Wake lexicon

Pythia-1.4B and SmolLM2-1.7B sit at 50,304 and 49,152 base vocabulary, that is
within 2.3% of each other, and at 1.4B and 1.7B parameters, within 1.2x. They
are the only near-matched pair in the fourteen-model lineup. Their near-identity 
on both the vocabulary and the scale axis makes them a controlled contrast in which 
vocabulary and scale are held approximately constant and **pretraining corpus** is the
variable. Pythia is trained on the Pile, which is heavily weighted toward
academic text, code and scraped web data. SmolLM2 is trained on FineWeb-Edu and
the SmolLM corpus, which are aggressively filtered for educational prose and
contain very little literary fiction.

## Architecture differences 

| | Llama / Mistral / Gemma / Phi | Pythia (GPT-NeoX) |
|---|---|---|
| class | `*ForCausalLM`, Llama-style | `GPTNeoXForCausalLM` |
| input embedding | `model.model.embed_tokens` | `model.gpt_neox.embed_in` |
| output head | `model.lm_head` | `model.embed_out` |
| tying | tied (or manually tied) | **untied** |
| attention proj | `q_proj`, `k_proj`, `v_proj` | **fused** `query_key_value` |
| attention out | `o_proj` | `dense` |
| MLP | `gate_proj`, `up_proj`, `down_proj` | `dense_h_to_4h`, `dense_4h_to_h` |
| matrix rows vs tokenizer | equal | **50,304 rows for 50,254 tokens** |

## envi
"""

import os
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Triton shim: bnb 0.45.x imports triton.ops.matmul_perf_model, removed in
# triton 3.x. Harmless if bnb >= 0.46.1 already handles it.
import types, sys
_fake = types.ModuleType("triton.ops")
_fake.matmul_perf_model = types.ModuleType("triton.ops.matmul_perf_model")
sys.modules.setdefault("triton.ops", _fake)
sys.modules.setdefault("triton.ops.matmul_perf_model", _fake.matmul_perf_model)

import torch, gc

print(f"torch: {torch.__version__} | cuda: {torch.version.cuda}")
try:
    import bitsandbytes as _bnb
    print(f"bitsandbytes: {_bnb.__version__}")
except ImportError:
    print("bitsandbytes NOT INSTALLED. run: pip install -U bitsandbytes")
import transformers, accelerate, peft
print(f"transformers: {transformers.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"peft: {peft.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

_BF16_OK = torch.cuda.is_bf16_supported()
print(f"bf16 supported natively: {_BF16_OK}")
if not _BF16_OK:
    print("  >> NOTE: bf16 is emulated on this GPU. Expect slower steps.")

torch.cuda.empty_cache()
gc.collect()

from google.colab import drive
drive.mount('/content/drive')

"""## config"""

import math
from pathlib import Path

# model 
MODEL_NAME = "EleutherAI/pythia-1.4b"

EXPECTED_MATRIX_ROWS = 50304   
EXPECTED_TOKENIZER_LEN = 50254  # len(tok) before injection
EXPECTED_HIDDEN = 2048

# data 
FW_TEXT = "/content/FW_TEXT.txt"
WAKE_LEXICON = "/content/wake_lexicon.txt"
MORPHEME_JSONL = "/content/wake_embedding_groups.jsonl"

# outputs
RUN_DIR = Path("/content/drive/MyDrive/wake2vec_pythia14b_p1")
LOCAL_RUN = Path("/content/runs/wake2vec_pythia14b_p1")
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
EMB_SNAP_STEPS = 100       
USE_BF16 = True            
GRAD_CKPT = True          

# architecture handling
TIE_OUTPUT_HEAD = True      
LORA_TARGET = "query_key_value"

# retain 
KEEP_EVERY = 500
KEEP_RECENT = 4

# resume 
RESUME_FROM = None
# RESUME_FROM = SENTRY / "checkpoint-200"

print("WAKE2VEC p1 on PYTHIA-1.4B CONFIG")
print(f"  Model:            {MODEL_NAME}")
print(f"  Output:           {RUN_DIR}")
print(f"  Steps:            {MAX_STEPS} | LR {LR} | warmup {WARMUP_STEPS}")
print(f"  Batch:            {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Seq len:          {SEQ_LEN}")
print(f"  bf16:             {USE_BF16} | grad ckpt: {GRAD_CKPT}")
print(f"  Tie output head:  {TIE_OUTPUT_HEAD}")
print(f"  Gradient masking: ENABLED (base rows frozen)")
print(f"  Snapshots:        Wake rows only, fp32, every {EMB_SNAP_STEPS} steps")
print(f"  Resume:           {RESUME_FROM}")

"""## tokenizer + Wake vocab injection"""

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token

VOCAB_BEFORE = len(tok)
print(f"  Tokenizer length before injection: {VOCAB_BEFORE}")
print(f"  unk_token: {tok.unk_token!r} | unk_token_id: {tok.unk_token_id}")
print(f"  eos_token: {tok.eos_token!r} | pad_token: {tok.pad_token!r}")

if VOCAB_BEFORE != EXPECTED_TOKENIZER_LEN:
    print(f"  >> NOTE: expected {EXPECTED_TOKENIZER_LEN}, got {VOCAB_BEFORE}.")
    print("     Not fatal. Record the actual value in the devlog,")
    print("     because Wake share depends on it.")

# GPT-NeoX is byte-level BPE with a leading-space marker; added tokens are matched literally, before pre-tokenisation, which interacts with
# that marker differently from SentencePiece. Worth seeing on three lines of actual Wake before committing 3000 steps.
_PROBE = [
    "riverrun, past Eve and Adam's, from swerve of shore to bend of bay",
    "The fall (bababadalgharaghtakamminarronnkonnbronntonnerronntuonn)",
    "a waalworth of a skyerscape of most eyeful hoyth entowerly",
]
_pre_probe = [tok(s, add_special_tokens=False)["input_ids"] for s in _PROBE]

# missing-token detection 
print(f"Loading Wake lexicon from {WAKE_LEXICON}...")
with open(WAKE_LEXICON, "r", encoding="utf-8") as f:
    wake_tokens = [line.strip() for line in f if line.strip()]

# a word is missing iff it is not a key of the vocabulary
_vocab = tok.get_vocab()
_missing_raw = [w for w in wake_tokens if w not in _vocab]

# deduplicate preserving order 
_seen = set()
missing = []
for w in _missing_raw:
    if w not in _seen:
        _seen.add(w)
        missing.append(w)

_dupes = len(_missing_raw) - len(missing)
num_added = tok.add_tokens(missing, special_tokens=False)
TOTAL_VOCAB = len(tok)

print(f"  Lexicon entries:       {len(wake_tokens)}")
print(f"  Already in vocab:      {len(wake_tokens) - len(_missing_raw)}")
print(f"  Missing (raw):         {len(_missing_raw)}")
print(f"  Duplicates dropped:    {_dupes}")
print(f"  Tokens added:          {num_added}")
print(f"  Tokenizer length now:  {TOTAL_VOCAB}")

if num_added == 0:
    raise RuntimeError(
        "num_added is 0. Check tok.get_vocab() directly before "
        "concluding anything."
    )

assert num_added == len(missing), (
    f"add_tokens added {num_added} of {len(missing)} requested. The row "
    "alignment assumed by the compositional init no longer holds."
)
assert TOTAL_VOCAB == VOCAB_BEFORE + num_added, (
    f"{VOCAB_BEFORE} + {num_added} != {TOTAL_VOCAB}. Something was added to "
    "the tokenizer, most likely a special token."
)

# verify the mapping word -> row for every injected token 
_bad = []
for i, w in enumerate(missing):
    if tok.convert_tokens_to_ids(w) != VOCAB_BEFORE + i:
        _bad.append((i, w, tok.convert_tokens_to_ids(w)))
        if len(_bad) >= 5:
            break
if _bad:
    raise RuntimeError(
        f"Token-to-row alignment is broken. First failures: {_bad}. "
        "The compositional init and the gradient mask both depend on "
        "missing[i] occupying row VOCAB_BEFORE + i."
    )
print(f"  Alignment verified for all {num_added} injected tokens")

WAKE_SHARE = num_added / TOTAL_VOCAB
print(f"  Wake share of vocabulary: {WAKE_SHARE * 100:.1f}%")
print(f"  (LINEUP.md planning figure for Pythia-1.4B: 46.9%)")

# byte-level BPE diagnostic, after injection 
print()
print("  Tokenisation before and after injection:")
_post_probe = []
for s, pre_ids in zip(_PROBE, _pre_probe):
    post_ids = tok(s, add_special_tokens=False)["input_ids"]
    _post_probe.append(post_ids)
    print(f"    {len(pre_ids):4d} -> {len(post_ids):4d} tokens  {s[:52]!r}")
print()
print("  First probe line, post-injection tokens (watch for a bare space")
print("  token before each injected word; that is byte-level BPE meeting an")
print("  added token, and it is expected, not a fault):")
print("   ", tok.convert_ids_to_tokens(_post_probe[0])[:24])

"""## dataset"""

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
            "attention_mask": torch.ones_like(ids),
        }


for _p in (FW_TEXT, WAKE_LEXICON):
    if not os.path.exists(_p):
        raise FileNotFoundError(f"missing training file: {_p}")

with open(FW_TEXT, "r", encoding="utf-8") as f:
    fw_text = f.read()
with open(WAKE_LEXICON, "r", encoding="utf-8") as f:
    lex_text = f.read()

combined_text = fw_text + "\n" + lex_text
print(f"  FW text:  {len(fw_text)} chars")
print(f"  Lexicon:  {len(lex_text)} chars")
print(f"  Combined: {len(combined_text)} chars")

ids = tok(combined_text, add_special_tokens=False)["input_ids"]
print(f"  Total tokens: {len(ids)}")

blocks = [
    ids[i:i + SEQ_LEN]
    for i in range(0, len(ids) - SEQ_LEN + 1, SEQ_LEN)
]
blocks = [b for b in blocks if len(b) == SEQ_LEN]
print(f"  Blocks: {len(blocks)}")

# random_state=42 and test_size=0.10 are fixed across the lineup 
train_blocks, val_blocks = train_test_split(blocks, test_size=0.10, random_state=42)
train_ds = BlockDataset(train_blocks, SEQ_LEN)
val_ds = BlockDataset(val_blocks, SEQ_LEN)
print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

"""## model setup"""

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, get_peft_model

set_seed(42)

_compute_dtype = torch.bfloat16 if USE_BF16 else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=_compute_dtype,
    llm_int8_enable_fp32_cpu_offload=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    dtype=_compute_dtype,        # `dtype`, not `torch_dtype` (transformers 5.x)
    device_map="auto",
    max_memory={0: "13GB", "cpu": "30GB"},
    attn_implementation="eager",
)
model.config.use_cache = False

# verify the architecture 
print()
print(f"  class:        {type(model).__name__}")
print(f"  hidden_size:  {model.config.hidden_size}")
print(f"  layers:       {model.config.num_hidden_layers}")
print(f"  config vocab: {model.config.vocab_size}")
print(f"  tie_word_embeddings (config): {model.config.tie_word_embeddings}")

_in_emb = model.get_input_embeddings()
_out_emb = model.get_output_embeddings()
MATRIX_ROWS_BEFORE = _in_emb.weight.shape[0]
HIDDEN = _in_emb.weight.shape[1]

print(f"  input embedding:  {type(_in_emb).__name__} {tuple(_in_emb.weight.shape)}")
print(f"  output embedding: "
      f"{type(_out_emb).__name__ if _out_emb is not None else None} "
      f"{tuple(_out_emb.weight.shape) if _out_emb is not None else '-'}")
print(f"  tied in memory:   "
      f"{_out_emb is not None and _out_emb.weight is _in_emb.weight}")

assert HIDDEN == EXPECTED_HIDDEN, f"hidden {HIDDEN} != {EXPECTED_HIDDEN}"
if _out_emb is None:
    raise RuntimeError(
        "get_output_embeddings() returned None. This script assumes an "
        "explicit output head (Pythia's embed_out) and the tie logic below "
        "has nothing to attach to."
    )

# padding
PAD_ROWS = MATRIX_ROWS_BEFORE - VOCAB_BEFORE
print()
print(f"  tokenizer length before injection : {VOCAB_BEFORE}")
print(f"  embedding matrix rows before      : {MATRIX_ROWS_BEFORE}")
print(f"  padding rows (unreachable ids)    : {PAD_ROWS}")
if PAD_ROWS > 0:
    print(f"  >> This model pads its embedding matrix. Rows "
          f"{VOCAB_BEFORE}..{MATRIX_ROWS_BEFORE - 1} are not reachable by any")
    print(f"     token id and will be REUSED by the first {PAD_ROWS} Wake")
    print(f"     tokens. resize_token_embeddings does not initialise them,")
    print(f"     so this script writes them explicitly. Every boundary below")
    print(f"     is VOCAB_BEFORE ({VOCAB_BEFORE}), never "
          f"{MATRIX_ROWS_BEFORE}.")
assert PAD_ROWS >= 0, (
    f"tokenizer ({VOCAB_BEFORE}) is longer than the embedding matrix "
    f"({MATRIX_ROWS_BEFORE}); resize would truncate real tokens."
)
if MATRIX_ROWS_BEFORE != EXPECTED_MATRIX_ROWS:
    print(f"  >> NOTE: expected {EXPECTED_MATRIX_ROWS} rows, got "
          f"{MATRIX_ROWS_BEFORE}. Derived values are used; record this.")

# verify the LoRA target exists 
_mod_names = {n.split(".")[-1] for n, _ in model.named_modules()}
if LORA_TARGET not in _mod_names:
    _cands = sorted(n for n in _mod_names if any(
        k in n for k in ("query", "key", "value", "proj", "dense")))
    raise RuntimeError(
        f"LoRA target {LORA_TARGET!r} is not a module name on this model. "
        f"Candidates: {_cands}"
    )
print(f"  LoRA target {LORA_TARGET!r}: present")

# frozen r=1 adapter
peft_cfg = LoraConfig(
    r=1, lora_alpha=1, lora_dropout=0.0,
    target_modules=[LORA_TARGET], bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)

for _n, _p in model.named_parameters():
    _p.requires_grad = False

# resize
model.config.tie_word_embeddings = False
model.resize_token_embeddings(TOTAL_VOCAB, mean_resizing=False)

wte = model.get_input_embeddings()
out_head = model.get_output_embeddings()
assert wte.weight.shape[0] == TOTAL_VOCAB, \
    f"resize gave {wte.weight.shape[0]} rows, expected {TOTAL_VOCAB}"
print(f"  resized: {MATRIX_ROWS_BEFORE} -> {wte.weight.shape[0]} rows")

# manual tie 
if TIE_OUTPUT_HEAD:
    out_head.weight = wte.weight
    assert out_head.weight is wte.weight, "manual tie did not take"
    print("  output head MANUALLY TIED to input embedding (one tensor)")
    print("  >> P2 and P3 must repeat this tie. It is not in the checkpoint.")
else:
    print("  output head LEFT UNTIED. The Wake rows of embed_out will not")
    print("  train, so the model cannot learn to emit Wake tokens. This is a")
    print("  different experiment from the rest of the lineup.")

# initialisation: compositional, from the pretrained embeddings of the morphological  base word, for Wake words with a known decomposition.
# AND spherical, random direction at 1.0x the base radius, for the rest.

with torch.no_grad():
    # frozen fp32 copy of the PRETRAINED block, for composition 
    base_block = wte.weight[:VOCAB_BEFORE].detach().float().cpu().clone()

    _nz = base_block.norm(dim=1) > 1e-8
    print(f"  pretrained rows: {VOCAB_BEFORE} "
          f"({int((~_nz).sum())} with zero norm)")

    # statistics from non-zero pretrained rows only, so the padding rows do not drag the target radius down.
    std = base_block[_nz].std().item()
    base_radius = std * math.sqrt(HIDDEN)
    observed_norm = base_block[_nz].norm(dim=1).mean().item()
    target_radius = 1.0 * base_radius
    print(f"  base std {std:.6f} (non-zero rows only)")
    print(f"  base radius std*sqrt(d) {base_radius:.4f} | "
          f"observed mean norm {observed_norm:.4f} "
          f"(ratio {observed_norm / base_radius:.3f})")
    print(f"  target radius {target_radius:.4f} (1.0x, lineup convention)")

    g = torch.Generator(device="cpu").manual_seed(42)
    new = torch.randn((num_added, HIDDEN), generator=g, dtype=torch.float32)
    new = new / (new.norm(dim=1, keepdim=True) + 1e-8) * target_radius
    wte.weight.data[VOCAB_BEFORE:TOTAL_VOCAB] = new.to(
        wte.weight.device, dtype=wte.weight.dtype)
    print(f"  spherical init written to rows "
          f"{VOCAB_BEFORE}..{TOTAL_VOCAB - 1}")

    # overriding where a base word is available.
    comp_count = 0
    skipped_all_new = 0
    skipped_no_base = 0

    if os.path.exists(MORPHEME_JSONL):
        import json as _json
        with open(MORPHEME_JSONL, "r") as f:
            morph_groups = [_json.loads(line) for line in f]

        word_to_base = {}
        for grp in morph_groups:
            for w, b in zip(grp["examples"], grp["bases"]):
                word_to_base.setdefault(w, b)
        print(f"  morpheme groups: {len(morph_groups)} | "
              f"words with a base: {len(word_to_base)}")

        for i, wake_word in enumerate(missing):
            base_word = word_to_base.get(wake_word)
            if base_word is None:
                skipped_no_base += 1
                continue

            base_ids = tok(base_word, add_special_tokens=False)["input_ids"]
            base_ids = [j for j in base_ids if j < VOCAB_BEFORE]
            if not base_ids:
                skipped_all_new += 1
                continue

            composed = base_block[base_ids].mean(dim=0)
            composed = composed + torch.randn(
                HIDDEN, generator=g, dtype=torch.float32) * std * 0.1
            composed = composed / (composed.norm() + 1e-8) * target_radius
            wte.weight.data[VOCAB_BEFORE + i] = composed.to(
                wte.weight.device, dtype=wte.weight.dtype)
            comp_count += 1
    else:
        print(f"  morpheme file not found: {MORPHEME_JSONL}")
        print("  all Wake tokens keep spherical init")
        skipped_no_base = num_added

    del base_block
    gc.collect()

spherical_count = num_added - comp_count
print(f"  compositional: {comp_count}")
print(f"  spherical:     {spherical_count}")
print(f"    of which, no base word in the morpheme file: {skipped_no_base}")
print(f"    of which, base word is itself injected:      {skipped_all_new}")
print("  (the second figure is the one the source scripts composed from "
      "random\n   vectors instead of skipping)")

# gradient masking 
wte.weight.requires_grad = True

def mask_grad(grad):
    if grad is None:
        return grad
    grad[:VOCAB_BEFORE] = 0
    return grad

_mask_handle = wte.weight.register_hook(mask_grad)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print()
print(f"  trainable params: {trainable:,}")
print(f"  of which Wake rows: {num_added * HIDDEN:,} "
      f"({num_added} x {HIDDEN})")
print(f"  masked rows: 0..{VOCAB_BEFORE - 1}")

# optimizer memory
_bytes = 2 if USE_BF16 else 2
_emb_mb = TOTAL_VOCAB * HIDDEN * _bytes / 1e6
print(f"  embedding tensor: {_emb_mb:.0f} MB | "
      f"with grad + Adam states: {_emb_mb * 4:.0f} MB")

# verify mask 
_probe_ids = torch.tensor([blocks[0]], device=wte.weight.device)
model.train()
_out = model(input_ids=_probe_ids, labels=_probe_ids, use_cache=False)
_out.loss.backward()
_g = wte.weight.grad
if _g is None:
    raise RuntimeError("no gradient reached the embedding; masking untested")
_base_gn = _g[:VOCAB_BEFORE].abs().max().item()
_wake_gn = _g[VOCAB_BEFORE:].abs().max().item()
print(f"  loss: {_out.loss.item():.4f}")
print(f"  max |grad| on pretrained rows: {_base_gn:.3e}  (must be exactly 0)")
print(f"  max |grad| on Wake rows:       {_wake_gn:.3e}  (must be > 0)")
if _base_gn != 0.0:
    raise RuntimeError(
        f"gradient mask leaked: max |grad| on base rows is {_base_gn:.3e}. "
        "Do not train; the base-drift control is void."
    )
if _wake_gn == 0.0:
    raise RuntimeError(
        "no gradient on the Wake rows. Nothing would train. Check the tie and "
        "that the probe block actually contains injected tokens."
    )
model.zero_grad(set_to_none=True)
del _g, _out, _probe_ids
gc.collect(); torch.cuda.empty_cache()
print("  mask verified")

if GRAD_CKPT:
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    print("  gradient checkpointing: enabled (use_reentrant=False)")

"""## pre-training snapshot, the only full matrix written other than final/."""

E_pre = wte.weight.detach().float().cpu().clone()
_pre_path = RUN_DIR / "embeddings_pre.pt"
torch.save(E_pre, _pre_path)
print(f"  {_pre_path}")
print(f"  shape {tuple(E_pre.shape)} dtype {E_pre.dtype} "
      f"({E_pre.numel() * 4 / 1e6:.0f} MB)")

# boundary metadata
import json
_meta = {
    "model": MODEL_NAME,
    "vocab_before": int(VOCAB_BEFORE),
    "matrix_rows_before": int(MATRIX_ROWS_BEFORE),
    "padding_rows_reused": int(PAD_ROWS),
    "num_added": int(num_added),
    "total_vocab": int(TOTAL_VOCAB),
    "hidden": int(HIDDEN),
    "wake_share": float(WAKE_SHARE),
    "tie_output_head": bool(TIE_OUTPUT_HEAD),
    "lora_target": LORA_TARGET,
    "init": {
        "compositional": int(comp_count),
        "spherical": int(spherical_count),
        "skipped_no_base": int(skipped_no_base),
        "skipped_base_is_injected": int(skipped_all_new),
        "target_radius": float(target_radius),
        "radius_formula": "std*sqrt(dim) over non-zero pretrained rows",
        "observed_mean_base_norm": float(observed_norm),
        "radius_multiple": 1.0,
    },
}
(RUN_DIR / "vocab_boundary.json").write_text(json.dumps(_meta, indent=2))
print(f"  {RUN_DIR / 'vocab_boundary.json'}")
print(json.dumps(_meta, indent=2))

"""## callbacks"""

import shutil, time, re
from transformers import TrainerCallback


def wake_rows_fp32():
    return wte.weight[VOCAB_BEFORE:TOTAL_VOCAB].detach().float().cpu().clone()


def save_wake_rows(path, step):
    torch.save({
        "wake_rows": wake_rows_fp32(),
        "start": int(VOCAB_BEFORE),
        "end": int(TOTAL_VOCAB),
        "step": int(step),
        "dtype": "float32",
    }, path)


def load_wake_rows(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "wake_rows" in obj:
        rows = obj["wake_rows"]
        start, end = obj.get("start", VOCAB_BEFORE), obj.get("end", TOTAL_VOCAB)
    else:
        rows = obj
        start, end = VOCAB_BEFORE, TOTAL_VOCAB
        if rows.shape[0] == TOTAL_VOCAB:       # a full matrix was handed in
            rows = rows[VOCAB_BEFORE:TOTAL_VOCAB]
    if rows.shape != (num_added, HIDDEN):
        raise RuntimeError(
            f"{path}: wake rows are {tuple(rows.shape)}, expected "
            f"{(num_added, HIDDEN)}"
        )
    return rows.float(), int(start), int(end)


def full_from_wake_rows(rows, pre=None):
    E = (E_pre if pre is None else pre).clone()
    E[VOCAB_BEFORE:TOTAL_VOCAB] = rows.float()
    return E


def _step_of(p):
    m = re.search(r"(\d+)", p.name)
    return int(m.group(1)) if m else -1


def prune(directory, pattern, keep_every=KEEP_EVERY, keep_recent=KEEP_RECENT):
    items = sorted(
        [p for p in directory.glob(pattern) if _step_of(p) >= 0],
        key=_step_of,
    )
    if len(items) <= keep_recent:
        return
    protect = set(items[-keep_recent:])
    for p in items:
        if p in protect or _step_of(p) % keep_every == 0:
            continue
        try:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        except Exception as e:
            print(f"[PRUNE] {p.name}: {e}")


class EmbeddingSnapshot(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        s = state.global_step
        if s > 0 and s % EMB_SNAP_STEPS == 0:
            try:
                save_wake_rows(EMB_SNAPS / f"wake_step{s:05d}.pt", s)
                prune(EMB_SNAPS, "wake_step*.pt")
                print(f"[EMB] step {s}: wake rows saved")
            except Exception as e:
                print(f"[EMB] step {s}: {e}")


class FullCheckpoint(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        try:
            s = state.global_step
            d = FULL_CHECKPOINTS / f"step_{s:05d}"
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(d), save_embedding_layers=False)
            tok.save_pretrained(str(d))
            save_wake_rows(d / "wake_rows.pt", s)
            torch.save({
                "global_step": s,
                "best_metric": state.best_metric,
                "epoch": state.epoch,
                "tie_output_head": bool(TIE_OUTPUT_HEAD),
            }, d / "training_state.pt")
            prune(FULL_CHECKPOINTS, "step_*")
            print(f"[FULL] step {s}: saved")
        except Exception as e:
            print(f"[FULL] step {state.global_step}: {e}")


def has_weights(ck):
    return (ck / "adapter_model.safetensors").exists() or \
           (ck / "pytorch_model.bin").exists()


class SentryMirror(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        try:
            cks = sorted(LOCAL_RUN.glob("checkpoint-*"),
                         key=_step_of, reverse=True)
            if not cks or not has_weights(cks[0]):
                return
            ck = cks[0]
            dst = SENTRY / ck.name
            if dst.exists():
                return
            shutil.copytree(ck, dst)
            # adapter carries no embedding (save_embedding_layers=False), so a resume from this directory needs the Wake rows alongside it.
            save_wake_rows(dst / "wake_rows.pt", state.global_step)
            prune(SENTRY, "checkpoint-*")
            print(f"[SENTRY] {ck.name}: mirrored (+ wake rows)")
        except Exception as e:
            print(f"[SENTRY] {e}")


class LossMonitor(TrainerCallback):
    def __init__(self):
        self.last_train = None

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        if "loss" in logs:
            self.last_train = logs["loss"]
        if "eval_loss" in logs and self.last_train is not None:
            gap = logs["eval_loss"] - self.last_train
            if gap > 3.0:
                print(f"[WARN] train/eval gap {gap:.2f}")


class StepTimer(TrainerCallback):
    def __init__(self):
        self.times, self.last = [], None

    def on_step_end(self, args, state, control, **kw):
        now = time.time()
        if self.last is not None:
            self.times.append(now - self.last)
            if state.global_step % 10 == 0:
                recent = self.times[-10:]
                print(f"[{state.global_step:5d}] "
                      f"{sum(recent) / len(recent):.1f}s/step")
        self.last = now


print("Callbacks:")
print(f"  EmbeddingSnapshot  every {EMB_SNAP_STEPS} steps, Wake rows fp32")
print(f"  FullCheckpoint     every {SAVE_STEPS} steps, adapter + Wake rows")
print(f"  SentryMirror       mirror to Drive, pruned")
print(f"  retention          keep every {KEEP_EVERY}th + last {KEEP_RECENT}")
print("  LossMonitor, StepTimer")
