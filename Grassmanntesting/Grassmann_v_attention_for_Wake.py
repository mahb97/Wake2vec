"""
========================================================
DRAFT STAGE 

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

# imports 
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

# Configuration
class Config:
    # paths
    fw_text_path = ""
    wake_lexicon_path = ""
    output_dir = ""

    # model architecture (paper insights)
    vocab_size = 8192          # character-level BPE 
    d_model = 256              # hidden dimension
    n_layers = 6               
    n_heads = 4                # attention heads (transformer only)
    d_ff = 1024                # feed-forward dimension
    max_seq_len = 256         
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
    eval_interval = 500        
    save_interval = 2000       
    log_interval = 100         

    # generation
    gen_length = 512
    gen_temperature = 0.8
    gen_top_k = 50
    num_samples = 5

    # device
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

# to be continued 
