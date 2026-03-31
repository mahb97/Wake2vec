# -*- coding: utf-8 -*-
# ╔══════════════════════════════════════════════════════════════════╗
# ║  wake2vec Embedding Animation — Llama 3.2-1B P1                  ║
# ║  Watch 50 Wake tokens find their place in embedding space        ║
# ║  Colour-coded by word-formation device                           ║
# ║  "riverrun, past Eve and Adam's, from swerve of shore..."        ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Loads P1 embedding snapshots (every 50 training steps),
# selects ~50 Wake tokens across morpheme families,
# colour-codes by device type (portmanteau, foreign, malapropism, nonce, pun),
# projects to 2D via PCA, and outputs:
#   - Static PNG preview (first vs last frame)
#   - MP4 animation with motion trails
#   - Interactive HTML with step slider (plotly)
#   - Displacement convergence curve

import torch
import numpy as np
import json
import glob
import re
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from pathlib import Path
from collections import defaultdict

# paths
WAKE2VEC_ROOT = Path("/Users/maymaybiehle/Desktop/wake2vec")
SNAP_DIR      = WAKE2VEC_ROOT / "Embeds Llama 1b for P1"
LEXICON_PATH  = WAKE2VEC_ROOT / "wake_lexicon.txt"
OGDEN_PATH    = WAKE2VEC_ROOT / "odgen_basic_full.txt"
DEVICE_PATH   = WAKE2VEC_ROOT / "Devices" / "device_classification.jsonl"
MORPHEME_PATH = WAKE2VEC_ROOT / "FW morphology" / "wake_embedding_groups.jsonl"
OUTPUT_DIR    = WAKE2VEC_ROOT / "Dataviz" / "llama1b_p1_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# config
BASE_VOCAB    = 128256    # Llama 3.2-1B base vocab
N_WAKE        = 50        # Wake tokens to visualise
N_LANDMARKS   = 15        # Ogden's Basic English reference points
SEED          = 42
TRAIL_LEN     = 5         # motion trail length (frames)
FPS           = 10        # animation framerate

# device type colours (ignore faust)
DEVICE_COLOURS = {
    "portmanteau":  "#FF6B6B",  # coral red
    "foreign":      "#4ECDC4",  # teal
    "malapropism":  "#FFE66D",  # yellow
    "nonce":        "#A78BFA",  # purple
    "pun":          "#F97316",  # orange
}
DEVICE_ORDER = ["portmanteau", "foreign", "malapropism", "nonce", "pun"]
LANDMARK_COLOUR = "#666666"
BACKGROUND = "#0a0a0a"

print("wake2vec Embedding Animation — Llama 3.2-1B P1")

# load lexicon & build positional mapping 
print("\n── Loading Wake lexicon ──")
with open(LEXICON_PATH, "r", encoding="utf-8") as f:
    wake_words = [line.strip() for line in f if line.strip()]
print(f"  Lexicon: {len(wake_words)} words")

# positional mapping: each Wake word → BASE_VOCAB + index
# matches how resize_token_embeddings adds them
wake_word_to_id = {}
for i, word in enumerate(wake_words):
    wake_word_to_id[word] = BASE_VOCAB + i
print(f"  Mapped {len(wake_word_to_id)} Wake tokens (positional from {BASE_VOCAB})")

# load device classifications
print("\n── Loading device classifications ──")
word_to_device = {}
device_counts = defaultdict(int)
with open(DEVICE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        label = entry["label"]
        if label == "faust":
            continue 
        word = entry["text"]
        word_to_device[word] = label
        device_counts[label] += 1

print(f"  Device types:")
for dev in DEVICE_ORDER:
    print(f"    {dev}: {device_counts[dev]}")
print(f"  Total classified: {sum(device_counts.values())}")

# select 50 Wake tokens stratified by device type
print("\n── Selecting Wake tokens ──")
rng = np.random.RandomState(SEED)

# group Wake words that have both a device label AND a token mapping
device_pools = defaultdict(list)
for word, device in word_to_device.items():
    if word in wake_word_to_id:
        device_pools[device].append(word)

# stratified sample: ~10 per device type
tokens_per_device = max(N_WAKE // len(DEVICE_ORDER), 5)
selected_words = []
selected_devices = []

for device in DEVICE_ORDER:
    pool = device_pools[device]
    if len(pool) == 0:
        print(f"  Warning: no tokens for device '{device}'")
        continue
    n = min(tokens_per_device, len(pool))
    chosen = rng.choice(pool, size=n, replace=False)
    for w in chosen:
        selected_words.append(w)
        selected_devices.append(device)
    print(f"  {device}: selected {n} from {len(pool)} available")

# if below N_WAKE, top up from portmanteau (largest pool)
while len(selected_words) < N_WAKE and len(device_pools["portmanteau"]) > 0:
    remaining = [w for w in device_pools["portmanteau"] if w not in selected_words]
    if not remaining:
        break
    w = rng.choice(remaining)
    selected_words.append(w)
    selected_devices.append("portmanteau")

N_WAKE = len(selected_words)
selected_ids = [wake_word_to_id[w] for w in selected_words]
print(f"\n  Total selected: {N_WAKE} Wake tokens")

# load Ogden's landmarks
print("\n── Loading Ogden's landmarks ──")
with open(OGDEN_PATH, "r", encoding="utf-8") as f:
    ogden_raw = f.read()
ogden_words_all = [w.strip() for w in ogden_raw.replace('\n', '\r').split('\r') if w.strip()]

# Use a curated set of very common single-token English words 
LANDMARK_WORDS_MANUAL = [
    "the", "and", "for", "not", "you", "all", "can", "her", "was", "one",
    "our", "out", "day", "had", "hot", "old", "red", "sit", "top", "boy"
]
# place them using the first snapshot, find them in Ogden's and use sequential low IDs
landmark_words = LANDMARK_WORDS_MANUAL[:N_LANDMARKS]
# typically in the first few thousand tokens of any LLM vocab
landmark_ids = list(range(1000, 1000 + N_LANDMARKS))
print(f"  Selected {N_LANDMARKS} landmark words (base vocab reference points)")

# load snapshots 
print("\n── Loading embedding snapshots ──")

# pre-training snapshot (step 0, spherical init, before any training)
PRE_TRAIN_PATH = SNAP_DIR / "embeddings_pre.pt"

snap_files = sorted(glob.glob(str(SNAP_DIR / "emb_step*.pt")))

def get_step(path):
    m = re.search(r'emb_step(\d+)\.pt', path)
    return int(m.group(1)) if m else 0

snap_files = sorted(snap_files, key=get_step)

# prepend pre-training snapshot as step 0
if PRE_TRAIN_PATH.exists():
    snap_files = [str(PRE_TRAIN_PATH)] + snap_files
    step_numbers = [0] + [get_step(f) for f in snap_files[1:]]
    print(f"  Pre-training snapshot found: embeddings_pre.pt (step 0)")
else:
    step_numbers = [get_step(f) for f in snap_files]
    print(f"  No pre-training snapshot found")

print(f"  Found {len(snap_files)} snapshots")
print(f"  Steps: {step_numbers[0]} to {step_numbers[-1]}")

# check first snapshot for dimensions and validate token IDs
E_first = torch.load(snap_files[0], map_location="cpu")
embed_dim = E_first.shape[1]
max_idx = E_first.shape[0] - 1
print(f"  First snapshot shape: {E_first.shape} (max index {max_idx})")

# filter selected tokens to those within bounds
valid_mask = [tid <= max_idx for tid in selected_ids]
selected_words = [w for w, v in zip(selected_words, valid_mask) if v]
selected_devices = [d for d, v in zip(selected_devices, valid_mask) if v]
selected_ids = [tid for tid, v in zip(selected_ids, valid_mask) if v]
dropped = sum(1 for v in valid_mask if not v)
if dropped > 0:
    print(f"  Warning: dropped {dropped} tokens (IDs out of bounds)")
N_WAKE = len(selected_ids)
print(f"  Valid Wake tokens: {N_WAKE}")

# filter landmarks too
landmark_ids = [lid for lid in landmark_ids if lid <= max_idx]
landmark_words = landmark_words[:len(landmark_ids)]
N_LANDMARKS = len(landmark_ids)

del E_first  # free memory

# extract rows for all snapshots
print("\n  Loading snapshots...")
wake_positions = []      # [n_steps, N_WAKE, embed_dim]
landmark_positions = []  # [n_steps, N_LANDMARKS, embed_dim]

for i, fpath in enumerate(snap_files):
    E = torch.load(fpath, map_location="cpu")

    # handle different snapshot sizes (full matrix vs Wake-only)
    if E.shape[0] > max(selected_ids):
        # full embed matrix
        wake_rows = E[selected_ids].float().numpy()
        land_rows = E[landmark_ids].float().numpy()
    else:
        # smaller snapshot, might be Wake rows only
        valid_wake = [tid for tid in selected_ids if tid < E.shape[0]]
        valid_land = [lid for lid in landmark_ids if lid < E.shape[0]]
        if len(valid_wake) < N_WAKE * 0.5:
            print(f"  Skipping step {step_numbers[i]} (shape {E.shape[0]}, too small)")
            continue
        wake_rows = E[valid_wake].float().numpy()
        land_rows = E[valid_land].float().numpy() if valid_land else np.zeros((N_LANDMARKS, embed_dim))
        # pad if needed
        if wake_rows.shape[0] < N_WAKE:
            pad = np.zeros((N_WAKE - wake_rows.shape[0], embed_dim))
            wake_rows = np.vstack([wake_rows, pad])

    wake_positions.append(wake_rows)
    landmark_positions.append(land_rows)
    if (i + 1) % 10 == 0 or i == 0:
        print(f"    Loaded {i+1}/{len(snap_files)} (step {step_numbers[i]}, shape {E.shape})")

    del E

wake_positions = np.array(wake_positions)
landmark_positions = np.array(landmark_positions)
n_steps = wake_positions.shape[0]
# update step_numbers to match loaded steps
step_numbers = step_numbers[:n_steps]
print(f"\n  Wake positions: {wake_positions.shape}")
print(f"  Landmark positions: {landmark_positions.shape}")
print(f"  Frames: {n_steps}")

# pca projection 
print("\n── PCA projection ──")
pca = PCA(n_components=2, random_state=SEED)
pca.fit(wake_positions[-1])  # fit on final frame for stable axes
print(f"  Explained variance: {pca.explained_variance_ratio_}")
print(f"  Total: {sum(pca.explained_variance_ratio_):.3f}")

wake_2d = np.zeros((n_steps, N_WAKE, 2))
landmark_2d = np.zeros((n_steps, N_LANDMARKS, 2))
for i in range(n_steps):
    wake_2d[i] = pca.transform(wake_positions[i])
    landmark_2d[i] = pca.transform(landmark_positions[i])

# drift stats
landmark_drift = np.linalg.norm(landmark_2d[-1] - landmark_2d[0], axis=1)
wake_drift = np.linalg.norm(wake_2d[-1] - wake_2d[0], axis=1)
print(f"\n  Landmark drift (mean): {landmark_drift.mean():.4f}")
print(f"  Wake drift (mean): {wake_drift.mean():.4f}")
print(f"  Wake drift (max): {wake_drift.max():.4f}")

# colour arrays 
wake_colours = [DEVICE_COLOURS.get(d, "#ffffff") for d in selected_devices]

# static preview 
print("\n── Static preview: first vs last frame ──")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(BACKGROUND)

for ax, frame_idx, title in [(axes[0], 0, f"Step {step_numbers[0]} (initial)"),
                               (axes[1], -1, f"Step {step_numbers[-1]} (final)")]:
    ax.set_facecolor(BACKGROUND)

    # landmarks (grey squares, labelled)
    ax.scatter(landmark_2d[frame_idx, :, 0], landmark_2d[frame_idx, :, 1],
               c=LANDMARK_COLOUR, s=50, alpha=0.8, zorder=4, marker='s')
    for j, word in enumerate(landmark_words):
        ax.annotate(word, (landmark_2d[frame_idx, j, 0], landmark_2d[frame_idx, j, 1]),
                    fontsize=7, color='#999999', ha='left', va='bottom',
                    xytext=(4, 4), textcoords='offset points')

    # Wake tokens by device type
    for device in DEVICE_ORDER:
        mask = [d == device for d in selected_devices]
        if not any(mask):
            continue
        idx = [k for k, m in enumerate(mask) if m]
        ax.scatter(wake_2d[frame_idx, idx, 0], wake_2d[frame_idx, idx, 1],
                   c=DEVICE_COLOURS[device], s=30, alpha=0.85, zorder=5,
                   label=device, edgecolors='white', linewidths=0.3)
        # label each token
        for k in idx:
            ax.annotate(selected_words[k],
                        (wake_2d[frame_idx, k, 0], wake_2d[frame_idx, k, 1]),
                        fontsize=5, color=DEVICE_COLOURS[device],
                        ha='left', va='bottom', alpha=0.9,
                        xytext=(3, 3), textcoords='offset points')

    ax.set_title(title, color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='#444444', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#333333')

# legend on the right panel
axes[1].legend(loc='upper right', fontsize=9, facecolor='#1a1a1a',
               edgecolor='#333333', labelcolor='white', markerscale=1.5)

fig.suptitle("wake2vec P1 — Llama 3.2-1B — Embedding Settlement",
             color='white', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_DIR / "llama1b_embedding_preview.png", dpi=200, facecolor=BACKGROUND)
plt.show()
print(f"  Saved: {OUTPUT_DIR / 'llama1b_embedding_preview.png'}")

# MP4 Animation
print("\n── Building MP4 animation ──")
matplotlib.use('Agg')

fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor(BACKGROUND)
ax.set_facecolor(BACKGROUND)

# global axis limits
all_x = np.concatenate([wake_2d[:, :, 0].flatten(), landmark_2d[:, :, 0].flatten()])
all_y = np.concatenate([wake_2d[:, :, 1].flatten(), landmark_2d[:, :, 1].flatten()])
pad_x = (all_x.max() - all_x.min()) * 0.1
pad_y = (all_y.max() - all_y.min()) * 0.1
xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
ylim = (all_y.min() - pad_y, all_y.max() + pad_y)

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.tick_params(colors='#333333', labelsize=7)
for spine in ax.spines.values():
    spine.set_color('#222222')

# title
title_text = ax.set_title("", color='white', fontsize=14, fontweight='bold', pad=12)

# landmark scatter + labels
land_scatter = ax.scatter([], [], c=LANDMARK_COLOUR, s=50, alpha=0.8, zorder=4, marker='s')
land_labels = []
for j in range(N_LANDMARKS):
    txt = ax.annotate(landmark_words[j], (0, 0), fontsize=6, color='#777777',
                      ha='left', va='bottom', xytext=(4, 4),
                      textcoords='offset points', zorder=5)
    land_labels.append(txt)

# wake scatters with one per device type for legend
wake_scatters = {}
for device in DEVICE_ORDER:
    mask = [d == device for d in selected_devices]
    if any(mask):
        sc = ax.scatter([], [], c=DEVICE_COLOURS[device], s=25, alpha=0.85,
                        zorder=6, label=device, edgecolors='white', linewidths=0.3)
        wake_scatters[device] = (sc, [k for k, m in enumerate(mask) if m])

# wake labels
wake_labels = []
for k in range(N_WAKE):
    txt = ax.annotate(selected_words[k], (0, 0), fontsize=5,
                      color=wake_colours[k], alpha=0.8,
                      ha='left', va='bottom', xytext=(3, 3),
                      textcoords='offset points', zorder=7)
    wake_labels.append(txt)

# trails, one line per token
trail_lines = []
for k in range(N_WAKE):
    line, = ax.plot([], [], color=wake_colours[k], alpha=0.2, linewidth=0.6, zorder=2)
    trail_lines.append(line)

# step counter
step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='#555555',
                    fontsize=11, verticalalignment='top', fontfamily='monospace')

# legend
ax.legend(loc='upper right', fontsize=8, facecolor='#1a1a1a',
          edgecolor='#333333', labelcolor='white', markerscale=1.5)

# subtitle
ax.text(0.5, -0.02, 'wake2vec P1 — Llama 3.2-1B — PCA projection — colour = word-formation device',
        transform=ax.transAxes, color='#444444', fontsize=8,
        ha='center', va='top', fontfamily='monospace')

def init():
    land_scatter.set_offsets(np.empty((0, 2)))
    for device, (sc, idx) in wake_scatters.items():
        sc.set_offsets(np.empty((0, 2)))
    for line in trail_lines:
        line.set_data([], [])
    return []

def update(frame):
    step = step_numbers[frame]

    # landmarks
    land_scatter.set_offsets(landmark_2d[frame])
    for j, txt in enumerate(land_labels):
        txt.set_position((landmark_2d[frame, j, 0], landmark_2d[frame, j, 1]))

    # wake tokens by device
    for device, (sc, idx) in wake_scatters.items():
        sc.set_offsets(wake_2d[frame, idx])

    # Wake labels
    for k, txt in enumerate(wake_labels):
        txt.set_position((wake_2d[frame, k, 0], wake_2d[frame, k, 1]))

    # trails
    trail_start = max(0, frame - TRAIL_LEN)
    for k in range(N_WAKE):
        if frame > 0:
            xs = wake_2d[trail_start:frame+1, k, 0]
            ys = wake_2d[trail_start:frame+1, k, 1]
            trail_lines[k].set_data(xs, ys)
        else:
            trail_lines[k].set_data([], [])

    title_text.set_text(f"Wake Tokens in Embedding Space — Step {step}")
    step_text.set_text(f"step {step:>4d} / {step_numbers[-1]}")

    return []

anim = FuncAnimation(fig, update, init_func=init, frames=n_steps,
                     interval=1000 // FPS, blit=True)

try:
    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=FPS, metadata={'title': 'wake2vec Llama 1B embedding animation'})
    anim.save(str(OUTPUT_DIR / "llama1b_embedding_animation.mp4"), writer=writer, dpi=150)
    print(f"  Saved MP4: {OUTPUT_DIR / 'llama1b_embedding_animation.mp4'}")
except Exception as e:
    print(f"  FFmpeg failed ({e}), trying GIF...")
    try:
        anim.save(str(OUTPUT_DIR / "llama1b_embedding_animation.gif"), writer='pillow', fps=FPS, dpi=120)
        print(f"  Saved GIF: {OUTPUT_DIR / 'llama1b_embedding_animation.gif'}")
    except Exception as e2:
        print(f"  GIF also failed ({e2}). Install ffmpeg: brew install ffmpeg")

plt.close(fig)

# Interactive HTML 
print("\n── Building interactive HTML ──")

try:
    import plotly.graph_objects as go

    frames = []
    for i in range(n_steps):
        step = step_numbers[i]
        frame_data = []

        # landmarks
        frame_data.append(go.Scatter(
            x=landmark_2d[i, :, 0].tolist(),
            y=landmark_2d[i, :, 1].tolist(),
            mode='markers+text',
            marker=dict(size=10, color=LANDMARK_COLOUR, symbol='square', opacity=0.8),
            text=landmark_words,
            textposition='top right',
            textfont=dict(size=8, color='#888888'),
            name='Ogden\'s landmarks',
            hovertemplate='%{text}<extra>landmark</extra>',
            showlegend=(i == 0),
        ))

        # wake tokens by device type
        for device in DEVICE_ORDER:
            mask = [d == device for d in selected_devices]
            if not any(mask):
                continue
            idx = [k for k, m in enumerate(mask) if m]
            words = [selected_words[k] for k in idx]
            frame_data.append(go.Scatter(
                x=wake_2d[i, idx, 0].tolist(),
                y=wake_2d[i, idx, 1].tolist(),
                mode='markers+text',
                marker=dict(size=8, color=DEVICE_COLOURS[device], opacity=0.9,
                            line=dict(width=0.5, color='white')),
                text=words,
                textposition='top right',
                textfont=dict(size=7, color=DEVICE_COLOURS[device]),
                name=device,
                hovertemplate='%{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra>' + device + '</extra>',
                showlegend=(i == 0),
            ))

        frames.append(go.Frame(data=frame_data, name=str(step)))

    fig_plotly = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=dict(
                text='wake2vec P1 — Llama 3.2-1B — Wake Tokens Settling Into Embedding Space',
                font=dict(color='white', size=16),
            ),
            paper_bgcolor=BACKGROUND,
            plot_bgcolor=BACKGROUND,
            xaxis=dict(range=[xlim[0], xlim[1]], gridcolor='#1a1a1a',
                       zerolinecolor='#333333', tickfont=dict(color='#444444')),
            yaxis=dict(range=[ylim[0], ylim[1]], gridcolor='#1a1a1a',
                       zerolinecolor='#333333', tickfont=dict(color='#444444')),
            legend=dict(font=dict(color='white'), bgcolor='#1a1a1a',
                        bordercolor='#333333'),
            updatemenus=[dict(
                type='buttons', showactive=False, y=0, x=0.5, xanchor='center',
                buttons=[
                    dict(label='▶ Play', method='animate',
                         args=[None, dict(frame=dict(duration=150, redraw=True),
                                          fromcurrent=True,
                                          transition=dict(duration=50))]),
                    dict(label='⏸ Pause', method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode='immediate',
                                            transition=dict(duration=0))]),
                ],
                font=dict(color='white'), bgcolor='#222222', bordercolor='#444444',
            )],
            sliders=[dict(
                active=0,
                steps=[
                    dict(args=[[str(step_numbers[i])],
                               dict(frame=dict(duration=0, redraw=True),
                                    mode='immediate',
                                    transition=dict(duration=0))],
                         label=str(step_numbers[i]),
                         method='animate')
                    for i in range(n_steps)
                ],
                x=0.05, len=0.9, xanchor='left', y=-0.05,
                currentvalue=dict(prefix='Step: ', visible=True,
                                  xanchor='center', font=dict(color='white')),
                font=dict(color='#888888'), bgcolor='#222222',
                activebgcolor='#FF6B6B', bordercolor='#444444', tickcolor='#444444',
            )],
            width=1100, height=750,
        )
    )

    fig_plotly.write_html(str(OUTPUT_DIR / "llama1b_embedding_interactive.html"))
    print(f"  Saved HTML: {OUTPUT_DIR / 'llama1b_embedding_interactive.html'}")

except ImportError:
    print("  plotly not installed. Run: pip install plotly")
except Exception as e:
    print(f"  plotly failed: {e}")

# displacement curve 
print("\n── Displacement convergence curve ──")

matplotlib.use('TkAgg')  # switch back to interactive backend

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
fig2.patch.set_facecolor(BACKGROUND)

# Mean displacement over time
mean_disp = [np.linalg.norm(wake_2d[i] - wake_2d[0], axis=1).mean() for i in range(n_steps)]

ax1.set_facecolor(BACKGROUND)
ax1.plot(step_numbers, mean_disp, color='#FF6B6B', linewidth=2.5, zorder=3)
ax1.set_xlabel('Training Step', color='#aaaaaa', fontsize=10)
ax1.set_ylabel('Mean Displacement from Init (2D)', color='#aaaaaa', fontsize=10)
ax1.set_title('Wake Token Migration Over Training', color='white', fontweight='bold', fontsize=12)
ax1.tick_params(colors='#555555')
for spine in ax1.spines.values():
    spine.set_color('#333333')
ax1.grid(True, alpha=0.15, color='#444444')

# Per-device displacement
ax2.set_facecolor(BACKGROUND)
for device in DEVICE_ORDER:
    mask = [d == device for d in selected_devices]
    if not any(mask):
        continue
    idx = [k for k, m in enumerate(mask) if m]
    dev_disp = [np.linalg.norm(wake_2d[i, idx] - wake_2d[0, idx], axis=1).mean()
                for i in range(n_steps)]
    ax2.plot(step_numbers, dev_disp, color=DEVICE_COLOURS[device],
             linewidth=2, label=device, alpha=0.9)

ax2.set_xlabel('Training Step', color='#aaaaaa', fontsize=10)
ax2.set_ylabel('Mean Displacement by Device Type', color='#aaaaaa', fontsize=10)
ax2.set_title('Migration by Word-Formation Device', color='white', fontweight='bold', fontsize=12)
ax2.tick_params(colors='#555555')
for spine in ax2.spines.values():
    spine.set_color('#333333')
ax2.grid(True, alpha=0.15, color='#444444')
ax2.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#333333', labelcolor='white')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "llama1b_displacement_curves.png", dpi=200, facecolor=BACKGROUND)
plt.show()
print(f"  Saved: {OUTPUT_DIR / 'llama1b_displacement_curves.png'}")

# summary stats & JSON
print("\n── Summary Statistics ──")

displacement_2d = np.linalg.norm(wake_2d[-1] - wake_2d[0], axis=1)
displacement_full = np.linalg.norm(wake_positions[-1] - wake_positions[0], axis=1)

print(f"\n  Displacement (2D PCA):")
print(f"    Mean: {displacement_2d.mean():.4f}")
print(f"    Max:  {displacement_2d.max():.4f}")
print(f"    Min:  {displacement_2d.min():.4f}")

print(f"\n  Displacement (full {embed_dim}D):")
print(f"    Mean: {displacement_full.mean():.4f}")
print(f"    Max:  {displacement_full.max():.4f}")
print(f"    Min:  {displacement_full.min():.4f}")

# top 10 most mobile
top_idx = np.argsort(displacement_2d)[::-1][:10]
print(f"\n  Top 10 most mobile:")
for rank, idx in enumerate(top_idx):
    print(f"    {rank+1}. '{selected_words[idx]}' ({selected_devices[idx]}) — {displacement_2d[idx]:.4f}")

# top 10 most stationary
bot_idx = np.argsort(displacement_2d)[:10]
print(f"\n  Top 10 most stationary:")
for rank, idx in enumerate(bot_idx):
    print(f"    {rank+1}. '{selected_words[idx]}' ({selected_devices[idx]}) — {displacement_2d[idx]:.4f}")

# per-device stats
print(f"\n  Displacement by device type:")
for device in DEVICE_ORDER:
    mask = [d == device for d in selected_devices]
    if not any(mask):
        continue
    idx = [k for k, m in enumerate(mask) if m]
    dev_disp = displacement_2d[idx]
    print(f"    {device:15s}: mean={dev_disp.mean():.4f}, std={dev_disp.std():.4f}")

# save summary
summary = {
    "model": "Llama-3.2-1B",
    "phase": "P1",
    "n_snapshots": n_steps,
    "steps": step_numbers,
    "n_wake_tokens": N_WAKE,
    "n_landmarks": N_LANDMARKS,
    "embed_dim": embed_dim,
    "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
    "selected_tokens": [{"word": w, "device": d, "token_id": tid}
                        for w, d, tid in zip(selected_words, selected_devices, selected_ids)],
    "displacement_2d": {
        "mean": float(displacement_2d.mean()),
        "std": float(displacement_2d.std()),
        "max": float(displacement_2d.max()),
        "min": float(displacement_2d.min()),
    },
    "displacement_full": {
        "mean": float(displacement_full.mean()),
        "std": float(displacement_full.std()),
        "max": float(displacement_full.max()),
        "min": float(displacement_full.min()),
    },
    "displacement_by_device": {
        device: {
            "mean": float(displacement_2d[[k for k, d in enumerate(selected_devices) if d == device]].mean()),
            "std": float(displacement_2d[[k for k, d in enumerate(selected_devices) if d == device]].std()),
        }
        for device in DEVICE_ORDER
        if any(d == device for d in selected_devices)
    },
    "top10_mobile": [(selected_words[i], selected_devices[i], float(displacement_2d[i])) for i in top_idx],
    "top10_stationary": [(selected_words[i], selected_devices[i], float(displacement_2d[i])) for i in bot_idx],
    "landmark_drift_mean": float(landmark_drift.mean()),
}

with open(OUTPUT_DIR / "llama1b_embedding_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Saved summary: {OUTPUT_DIR / 'llama1b_embedding_summary.json'}")

print("\n" + "=" * 60)
