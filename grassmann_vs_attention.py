"""
Grassmann Flows vs. Self-Attention: A Wake2Vec Experiment
=========================================================

Side-by-side comparison of two small (~15M param) language models
trained from scratch on Finnegans Wake:

  Model A: Standard causal self-attention transformer
  Model B: Causal Grassmann mixing (Zhang Chong, arXiv:2512.19428)

Both share identical embeddings, FFN, LayerNorm, and output head.
The ONLY difference is the sequence mixing mechanism.

Usage:
    python grassmann_vs_attention.py --mode train
    python grassmann_vs_attention.py --mode generate
    python grassmann_vs_attention.py --mode eval

Requires: torch, tiktoken (or a simple char/BPE tokenizer)
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    # paths
    fw_text_path = "/Users/maymaybiehle/Desktop/FW_TEXT.txt"
    wake_lexicon_path = "/Users/maymaybiehle/Desktop/wake2vec/wake_lexicon.txt"
    output_dir = "/Users/maymaybiehle/Desktop/wake2vec/grassmann_experiment"

    # model architecture (matched to Zhang Chong's setup)
    vocab_size = 8192          # character-level BPE — will be set after tokenizer init
    d_model = 256              # hidden dimension
    n_layers = 6               # number of layers
    n_heads = 4                # attention heads (transformer only)
    d_ff = 1024                # feed-forward dimension
    max_seq_len = 256          # context window
    dropout = 0.1

    # grassmann-specific
    grassmann_r = 32           # reduced dimension for Plucker encoding
    grassmann_windows = [1, 2, 4, 8, 12, 16]  # multi-scale offsets

    # training
    batch_size = 32
    lr = 3e-4
    weight_decay = 0.01
    epochs = 30
    warmup_steps = 500
    grad_clip = 1.0
    eval_interval = 500        # steps between evals
    save_interval = 2000       # steps between checkpoint saves
    log_interval = 100         # steps between log prints

    # generation
    gen_length = 512
    gen_temperature = 0.8
    gen_top_k = 50
    num_samples = 5

    # device
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ---------------------------------------------------------------------------
# Simple byte-pair-ish tokenizer (character-level with merges)
# ---------------------------------------------------------------------------

class CharTokenizer:
    """Minimal character-level tokenizer with a few Wake-aware merges."""

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

    def fit(self, text):
        chars = sorted(set(text))
        # reserve 0 for <pad>, 1 for <eos>
        self.char_to_id = {"<pad>": 0, "<eos>": 1}
        for i, c in enumerate(chars, start=2):
            self.char_to_id[c] = i
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        return self

    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.id_to_char.get(i, "") for i in ids)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"char_to_id": self.char_to_id}, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        return self


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Sliding-window dataset over tokenized text."""

    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.token_ids[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.gelu(self.w1(x))))


# ---------------------------------------------------------------------------
# Model A: Standard Causal Self-Attention Transformer
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, dh)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) * (self.d_head ** -0.5)
        att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        out = (att @ v).transpose(1, 2).reshape(B, L, D)
        return self.resid_drop(self.out_proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.max_seq_len, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, L = x.shape
        pos = torch.arange(0, L, device=x.device).unsqueeze(0)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Model B: Causal Grassmann Mixing LM
# ---------------------------------------------------------------------------

class GrassmannMixingBlock(nn.Module):
    """
    Causal Grassmann mixing layer (Zhang Chong, Section 3.2).

    Steps:
      1. Linear reduction: h_t (d) -> z_t (r)
      2. Multi-scale pairing: for offsets in windows, pair (z_t, z_{t+delta})
      3. Plucker encoding: p_ij = z_t[i]*z_{t+d}[j] - z_t[j]*z_{t+d}[i]
      4. Project back: g_t = W_plu @ normalize(p) + b_plu
      5. Gated fusion: alpha * h_t + (1-alpha) * g_t
    """

    def __init__(self, d_model, r=32, windows=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.r = r
        self.windows = windows or [1, 2, 4, 8, 12, 16]
        self.plucker_dim = r * (r - 1) // 2  # C(r, 2)

        # step 1: reduce d -> r
        self.W_red = nn.Linear(d_model, r)

        # step 4: project plucker coords back to d_model
        self.W_plu = nn.Linear(self.plucker_dim, d_model)

        # step 5: gated fusion
        self.W_gate = nn.Linear(2 * d_model, d_model)

        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # precompute index pairs for plucker coordinates (vectorized)
        idx_i, idx_j = [], []
        for i in range(r):
            for j in range(i + 1, r):
                idx_i.append(i)
                idx_j.append(j)
        self.register_buffer("idx_i", torch.tensor(idx_i, dtype=torch.long))
        self.register_buffer("idx_j", torch.tensor(idx_j, dtype=torch.long))

    def plucker_coords(self, z1, z2):
        """
        Vectorized Plucker coordinate computation.
        z1, z2: (B, L, r)
        returns: (B, L, r*(r-1)//2)
        """
        # gather the i-th and j-th components
        z1_i = z1[..., self.idx_i]  # (B, L, P)
        z1_j = z1[..., self.idx_j]
        z2_i = z2[..., self.idx_i]
        z2_j = z2[..., self.idx_j]
        # p_ij = z1[i]*z2[j] - z1[j]*z2[i]
        return z1_i * z2_j - z1_j * z2_i

    def forward(self, h):
        """
        h: (B, L, d_model)
        returns: (B, L, d_model)
        """
        B, L, D = h.shape

        # Step 1: linear reduction
        z = self.W_red(h)  # (B, L, r)

        # Steps 2-3: multi-scale pairing + Plucker encoding
        # accumulate grassmann features per position
        g_accum = torch.zeros(B, L, D, device=h.device)
        count = torch.zeros(B, L, 1, device=h.device)

        for delta in self.windows:
            if delta >= L:
                continue
            # causal: pair position t with t+delta (look-ahead for LM)
            # Actually for causal LM we pair t with t-delta (look back)
            # so position t sees positions t-delta (past context)
            if delta > 0:
                z_t = z[:, delta:, :]       # positions delta..L-1
                z_past = z[:, :-delta, :]   # positions 0..L-1-delta

                # Plucker coords
                p = self.plucker_coords(z_t, z_past)  # (B, L-delta, P)

                # normalize for stability
                p_norm = torch.clamp(p.norm(dim=-1, keepdim=True), min=1e-8)
                p = p / p_norm

                # project back to d_model
                g_delta = self.W_plu(p)  # (B, L-delta, D)

                # accumulate at the correct positions
                g_accum[:, delta:, :] += g_delta
                count[:, delta:, :] += 1.0

        # average over valid offsets (avoid div by zero for early positions)
        count = torch.clamp(count, min=1.0)
        g = g_accum / count  # (B, L, D)

        # Step 5: gated fusion
        u = torch.cat([h, g], dim=-1)  # (B, L, 2D)
        alpha = torch.sigmoid(self.W_gate(u))  # (B, L, D)
        h_mix = alpha * h + (1.0 - alpha) * g

        # LayerNorm + dropout
        h_mix = self.dropout(self.ln(h_mix))
        return h_mix


class GrassmannTransformerBlock(nn.Module):
    """Grassmann mixing + FFN, mirroring TransformerBlock structure."""

    def __init__(self, d_model, d_ff, r=32, windows=None, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.grass = GrassmannMixingBlock(d_model, r, windows, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.grass(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GrassmannLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            GrassmannTransformerBlock(
                cfg.d_model, cfg.d_ff,
                r=cfg.grassmann_r,
                windows=cfg.grassmann_windows,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, L = x.shape
        pos = torch.arange(0, L, device=x.device).unsqueeze(0)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def get_lr(step, warmup_steps, lr, total_steps):
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr * max(coeff, 0.1)  # floor at 10% of base lr


@torch.no_grad()
def estimate_loss(model, dataloader, device, max_batches=50):
    """Estimate average loss over a number of batches."""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.8, top_k=50, device="cpu"):
    """Autoregressive generation."""
    model.eval()
    ids = tokenizer.encode(prompt)
    ids = ids[-(model.cfg.max_seq_len - 1):]  # truncate to fit
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.cfg.max_seq_len:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    model.train()
    return tokenizer.decode(x[0].tolist())


# ---------------------------------------------------------------------------
# Stylometry: simple character n-gram analysis
# ---------------------------------------------------------------------------

def char_ngram_profile(text, n=3, top_k=200):
    """Return a frequency-ranked dict of character n-grams."""
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    counts = Counter(ngrams)
    total = sum(counts.values())
    return {ng: c / total for ng, c in counts.most_common(top_k)}


def burrows_delta(profile_a, profile_b, top_k=200):
    """
    Simplified Burrows' Delta between two n-gram profiles.
    Lower = more similar.
    """
    all_ngrams = set(list(profile_a.keys())[:top_k]) | set(list(profile_b.keys())[:top_k])
    delta = 0.0
    for ng in all_ngrams:
        fa = profile_a.get(ng, 0.0)
        fb = profile_b.get(ng, 0.0)
        delta += abs(fa - fb)
    return delta / len(all_ngrams) if all_ngrams else 0.0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_model(model, model_name, train_loader, val_loader, cfg, tokenizer):
    """Train a model and return training history."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Parameters: {model.count_params():,}")
    print(f"Device: {cfg.device}")
    print(f"{'='*60}\n")

    model = model.to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    total_steps = cfg.epochs * len(train_loader)
    history = {"train_loss": [], "val_loss": [], "val_ppl": [], "step": []}

    step = 0
    best_val_loss = float("inf")
    save_dir = os.path.join(cfg.output_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    t0 = time.time()

    for epoch in range(cfg.epochs):
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)

            # LR schedule
            lr = get_lr(step, cfg.warmup_steps, cfg.lr, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # forward + backward
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            # logging
            if step % cfg.log_interval == 0:
                elapsed = time.time() - t0
                print(
                    f"[{model_name}] epoch {epoch+1}/{cfg.epochs} | "
                    f"step {step}/{total_steps} | "
                    f"loss {loss.item():.4f} | lr {lr:.2e} | "
                    f"{elapsed:.0f}s"
                )

            # evaluation
            if step % cfg.eval_interval == 0 and step > 0:
                val_loss = estimate_loss(model, val_loader, cfg.device)
                val_ppl = math.exp(min(val_loss, 20))  # cap for numerical safety
                history["train_loss"].append(loss.item())
                history["val_loss"].append(val_loss)
                history["val_ppl"].append(val_ppl)
                history["step"].append(step)
                print(
                    f"  >> val_loss={val_loss:.4f} | val_ppl={val_ppl:.1f} | "
                    f"best={best_val_loss:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
                    print(f"  >> saved best model (val_loss={val_loss:.4f})")

            # periodic save
            if step % cfg.save_interval == 0 and step > 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f"step_{step}.pt"))

            step += 1

    # final save
    torch.save(model.state_dict(), os.path.join(save_dir, "final.pt"))

    # save history
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def run_experiment(mode="train"):
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- Load and tokenize text ----
    print("Loading Finnegans Wake text...")
    with open(cfg.fw_text_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"  Text length: {len(raw_text):,} characters")

    # tokenizer
    tok_path = os.path.join(cfg.output_dir, "tokenizer.json")
    tokenizer = CharTokenizer()
    if os.path.exists(tok_path) and mode != "train":
        tokenizer.load(tok_path)
    else:
        tokenizer.fit(raw_text)
        os.makedirs(cfg.output_dir, exist_ok=True)
        tokenizer.save(tok_path)
    cfg.vocab_size = tokenizer.vocab_size
    print(f"  Vocabulary size: {cfg.vocab_size}")

    # tokenize
    all_ids = tokenizer.encode(raw_text)
    print(f"  Total tokens: {len(all_ids):,}")

    # train/val split (90/10)
    split = int(0.9 * len(all_ids))
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]

    train_ds = TextDataset(train_ids, cfg.max_seq_len)
    val_ds = TextDataset(val_ids, cfg.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)

    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val samples: {len(val_ds):,}")
    print(f"  Steps per epoch: {len(train_loader)}")

    # ---- Build models ----
    transformer_model = TransformerLM(cfg)
    grassmann_model = GrassmannLM(cfg)

    print(f"\nTransformerLM params: {transformer_model.count_params():,}")
    print(f"GrassmannLM params:   {grassmann_model.count_params():,}")

    if mode == "train":
        # ---- Train both models ----
        print("\n" + "=" * 60)
        print("PHASE 1: Training TransformerLM (attention baseline)")
        print("=" * 60)
        t_hist = train_model(
            transformer_model, "transformer", train_loader, val_loader, cfg, tokenizer
        )

        print("\n" + "=" * 60)
        print("PHASE 2: Training GrassmannLM (Grassmann flows)")
        print("=" * 60)
        g_hist = train_model(
            grassmann_model, "grassmann", train_loader, val_loader, cfg, tokenizer
        )

        # ---- Summary ----
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        if t_hist["val_ppl"]:
            print(f"TransformerLM best val PPL: {min(t_hist['val_ppl']):.1f}")
        if g_hist["val_ppl"]:
            print(f"GrassmannLM  best val PPL: {min(g_hist['val_ppl']):.1f}")

    elif mode == "generate":
        # ---- Generate samples from both models ----
        prompts = [
            "riverrun, past Eve and Adam's",
            "the fall of a once wallstrait oldparr",
            "Bygmester Finnegan, of the",
            "What clashes here of wills gen wonts",
            "Sir Tristram, violer d'amores",
        ]

        for name, model in [("transformer", transformer_model), ("grassmann", grassmann_model)]:
            ckpt = os.path.join(cfg.output_dir, name, "best.pt")
            if not os.path.exists(ckpt):
                ckpt = os.path.join(cfg.output_dir, name, "final.pt")
            if not os.path.exists(ckpt):
                print(f"No checkpoint found for {name}, skipping.")
                continue

            model.load_state_dict(torch.load(ckpt, map_location=cfg.device, weights_only=True))
            model = model.to(cfg.device)

            print(f"\n{'='*60}")
            print(f"GENERATION: {name.upper()}")
            print(f"{'='*60}")

            all_generated = []
            for prompt in prompts:
                text = generate(
                    model, tokenizer, prompt,
                    max_new_tokens=cfg.gen_length,
                    temperature=cfg.gen_temperature,
                    top_k=cfg.gen_top_k,
                    device=cfg.device,
                )
                all_generated.append(text)
                print(f"\nPrompt: {prompt[:40]}...")
                print(f"Output: {text[:300]}...")
                print("-" * 40)

            # save all generations
            gen_path = os.path.join(cfg.output_dir, name, "generations.txt")
            with open(gen_path, "w") as f:
                for i, (p, g) in enumerate(zip(prompts, all_generated)):
                    f.write(f"=== Sample {i+1} ===\n")
                    f.write(f"Prompt: {p}\n")
                    f.write(f"Generated:\n{g}\n\n")
            print(f"\nSaved generations to {gen_path}")

    elif mode == "eval":
        # ---- Evaluate: perplexity + stylometry ----
        print("\n" + "=" * 60)
        print("EVALUATION: Perplexity + Stylometric Analysis")
        print("=" * 60)

        # Reference profile from actual FW
        fw_profile = char_ngram_profile(raw_text, n=3, top_k=300)

        results = {}
        for name, model in [("transformer", transformer_model), ("grassmann", grassmann_model)]:
            ckpt = os.path.join(cfg.output_dir, name, "best.pt")
            if not os.path.exists(ckpt):
                ckpt = os.path.join(cfg.output_dir, name, "final.pt")
            if not os.path.exists(ckpt):
                print(f"No checkpoint found for {name}, skipping.")
                continue

            model.load_state_dict(torch.load(ckpt, map_location=cfg.device, weights_only=True))
            model = model.to(cfg.device)

            # Perplexity
            val_loss = estimate_loss(model, val_loader, cfg.device, max_batches=200)
            val_ppl = math.exp(min(val_loss, 20))

            # Generate a large sample for stylometry
            generated_text = ""
            seed_prompts = [
                "riverrun, past Eve",
                "the commodius vicus",
                "Bygmester Finnegan",
                "What clashes here",
                "Sir Tristram violer",
            ]
            for sp in seed_prompts:
                generated_text += generate(
                    model, tokenizer, sp,
                    max_new_tokens=1024,
                    temperature=0.8,
                    top_k=50,
                    device=cfg.device,
                )
                generated_text += "\n"

            # Stylometric comparison
            gen_profile = char_ngram_profile(generated_text, n=3, top_k=300)
            delta = burrows_delta(fw_profile, gen_profile, top_k=300)

            results[name] = {
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "burrows_delta": delta,
                "generated_chars": len(generated_text),
            }

            print(f"\n{name.upper()}")
            print(f"  Val loss:       {val_loss:.4f}")
            print(f"  Val perplexity: {val_ppl:.1f}")
            print(f"  Burrows' Delta: {delta:.6f} (lower = closer to FW)")
            print(f"  Generated:      {len(generated_text):,} chars")

        # Comparison
        if len(results) == 2:
            print(f"\n{'='*60}")
            print("HEAD-TO-HEAD COMPARISON")
            print(f"{'='*60}")
            t = results["transformer"]
            g = results["grassmann"]
            print(f"  {'Metric':<25} {'Transformer':>12} {'Grassmann':>12} {'Winner':>12}")
            print(f"  {'-'*61}")

            ppl_winner = "Transformer" if t["val_ppl"] < g["val_ppl"] else "Grassmann"
            delta_winner = "Transformer" if t["burrows_delta"] < g["burrows_delta"] else "Grassmann"

            print(f"  {'Val Perplexity':<25} {t['val_ppl']:>12.1f} {g['val_ppl']:>12.1f} {ppl_winner:>12}")
            print(f"  {'Burrows Delta (FW)':<25} {t['burrows_delta']:>12.6f} {g['burrows_delta']:>12.6f} {delta_winner:>12}")

        # Save results
        res_path = os.path.join(cfg.output_dir, "eval_results.json")
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {res_path}")

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: --mode train|generate|eval")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grassmann vs Attention Wake2Vec Experiment")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "generate", "eval"],
                        help="train: train both models | generate: sample text | eval: compare")
    args = parser.parse_args()
    run_experiment(args.mode)
