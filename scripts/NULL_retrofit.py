"""Corrected-loss permutation null, retrofitted onto a finished run.

Self-contained: it does NOT need the model, a GPU, or the run's own script. It
rebuilds the morpheme groups from the tokenizer and the JSONL, then runs the
cosine-mode null on two embedding matrices and reports the chance-corrected
movement between them.

WHY THIS EXISTS. TinyLlama, Llama 3.2-1B and Llama 3.2-3B ran P3 with the
legacy squared-deviation morpheme loss. At lambda=50 on a quantity worth ~1e-6
the morpheme term was ~0.001% of the objective, so those runs were effectively
UNTREATED on the morpheme axis: LM + device + repulsion + norm, no morpheme
pressure. Their endpoints therefore answer a question Mistral alone cannot:
what happens to morphological direction consistency when everything EXCEPT the
morpheme loss is trained. That is the no-treatment control for Mistral's
+0.0708, and it costs no training.

Settings are pinned to Mistral's so the numbers are directly comparable. Do not
change them without recording it.
"""

# ══ EDIT PER MODEL ═══════════════════════════════════════════════════
MODEL_TAG     = "tinyllama"
TOKENIZER_SRC = "/content/drive/MyDrive/wake2vec_tinyllama_p3/final"
BASE_VOCAB    = 32000          # vocab size BEFORE Wake injection
PRE_EMB       = "/content/drive/MyDrive/wake2vec_tinyllama_p2/embeddings_frozen.pt"
FINAL_EMB     = "/content/drive/MyDrive/wake2vec_tinyllama_p3/final/embeddings.pt"
MORPHEME_JSONL = "/content/wake_embedding_groups.jsonl"
OUT_JSON      = f"/content/drive/MyDrive/null_retrofit_{MODEL_TAG}.json"
# ═════════════════════════════════════════════════════════════════════

# Pinned to Mistral. Changing any of these breaks comparability.
MIN_GROUP_SIZE      = 3
MORPH_MOVABLE_ONLY  = True
MORPH_EXCLUDE_STRIP = True
STRIP_MORPHEMES     = {"-s", "-ed", "-y", "-ing", "-er", "-ly", "-on", "-'s", "-a", "co-"}
N_PERM              = 500
PERM_SEED           = 42

import json, torch, torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(TOKENIZER_SRC, use_fast=True)
TOTAL_VOCAB = len(tok)
print(f"[{MODEL_TAG}] vocab {TOTAL_VOCAB}, base {BASE_VOCAB}, "
      f"Wake {TOTAL_VOCAB - BASE_VOCAB}")

# ── groups, built from the tokenizer alone ──────────────────────────
raw = [json.loads(l) for l in open(MORPHEME_JSONL)]
morpheme_groups, dropped_frozen, skipped = [], 0, 0
for g in raw:
    m = g["morpheme"]
    if MORPH_EXCLUDE_STRIP and m in STRIP_MORPHEMES:
        continue
    wid, bid, pw = [], [], []
    for word, base in zip(g["examples"], g["bases"]):
        wi = tok(word, add_special_tokens=False)["input_ids"]
        bi = tok(base, add_special_tokens=False)["input_ids"]
        if not wi or not bi:
            continue
        if MORPH_MOVABLE_ONLY and not any(i >= BASE_VOCAB for i in list(wi) + list(bi)):
            dropped_frozen += 1
            continue
        wid.append(wi); bid.append(bi); pw.append((word, base))
    if len(wid) >= MIN_GROUP_SIZE:
        morpheme_groups.append({"morpheme": m, "morpheme_type": g["morpheme_type"],
                                "word_ids": wid, "base_ids": bid, "pair_words": pw})
    else:
        skipped += 1
n_pairs_total = sum(len(g["word_ids"]) for g in morpheme_groups)
print(f"[{MODEL_TAG}] groups {len(morpheme_groups)} (skipped {skipped}), "
      f"pairs {n_pairs_total}, frozen pairs dropped {dropped_frozen}")
assert morpheme_groups, "no groups survived; check BASE_VOCAB and the tokenizer"

# ── the null, identical to the Mistral script ───────────────────────
def _morph_directions(E):
    dirs, labels = [], []
    for gi, g in enumerate(morpheme_groups):
        w = torch.stack([E[torch.tensor(i, dtype=torch.long)].mean(0) for i in g["word_ids"]])
        b = torch.stack([E[torch.tensor(i, dtype=torch.long)].mean(0) for i in g["base_ids"]])
        dirs.append(w - b)
        labels.append(torch.full((len(g["word_ids"]),), gi, dtype=torch.long))
    return torch.cat(dirs), torch.cat(labels)

def _pairwise_cos(D, D_unit, lbl, n_groups):
    sums = torch.zeros(n_groups, D.shape[1], dtype=D.dtype)
    sums.index_add_(0, lbl, D)
    cnt = torch.bincount(lbl, minlength=n_groups).clamp(min=1).unsqueeze(1).to(D.dtype)
    means = F.normalize(sums / cnt, dim=1, eps=1e-8)
    return (D_unit * means[lbl]).sum(dim=1)

def morph_null_report(E, label=""):
    E = E.detach().float().cpu()
    D, L_ = _morph_directions(E)
    n_pairs, n_groups = D.shape[0], len(morpheme_groups)
    D_unit = F.normalize(D, dim=1, eps=1e-8)
    cnt = torch.bincount(L_, minlength=n_groups).clamp(min=1).float()
    obs_cos = _pairwise_cos(D, D_unit, L_, n_groups)
    obs_global = float((1.0 - obs_cos).mean())
    obs_group = torch.zeros(n_groups).index_add_(0, L_, 1.0 - obs_cos) / cnt
    g = torch.Generator().manual_seed(PERM_SEED)
    null_global = torch.empty(N_PERM); null_group = torch.empty(N_PERM, n_groups)
    for k in range(N_PERM):
        perm = L_[torch.randperm(n_pairs, generator=g)]
        c = _pairwise_cos(D, D_unit, perm, n_groups)
        null_global[k] = (1.0 - c).mean()
        null_group[k] = torch.zeros(n_groups).index_add_(0, perm, 1.0 - c) / cnt
    mu, sd = float(null_global.mean()), float(null_global.std())
    gmu, gsd = null_group.mean(0), null_group.std(0).clamp(min=1e-9)
    gz = (obs_group - gmu) / gsd
    per_group = sorted(({"morpheme": morpheme_groups[i]["morpheme"],
                         "n_pairs": int(cnt[i]),
                         "observed_consistency": float(1 - obs_group[i]),
                         "excess": float(gmu[i] - obs_group[i]),
                         "z": float(gz[i])} for i in range(n_groups)),
                       key=lambda r: r["z"])
    sig = [r for r in per_group if r["z"] < -3.0]
    print(f"\n{'='*66}\nNULL -- {label}\n{'='*66}")
    print(f"  pairs {n_pairs:,} | groups {n_groups} | perms {N_PERM}")
    print(f"  observed L_morph  : {obs_global:.6f}  (consistency {1-obs_global:.6f})")
    print(f"  null mean         : {mu:.6f}  (consistency {1-mu:.6f})  sd {sd:.6f}")
    print(f"  EXCESS OVER CHANCE: {mu-obs_global:+.6f}   z = {(obs_global-mu)/sd:+.2f}")
    print(f"  groups below z=-3 : {len(sig)}/{n_groups}")
    print(f"  strongest: " + ", ".join(f"{r['morpheme']}({r['z']:+.1f})" for r in per_group[:6]))
    return {"label": label, "n_pairs": n_pairs, "n_groups": n_groups,
            "observed_L_morph": obs_global, "observed_consistency": 1-obs_global,
            "null_mean": mu, "null_sd": sd, "excess_over_chance": mu-obs_global,
            "z": (obs_global-mu)/sd, "n_groups_below_z3": len(sig),
            "per_group": per_group}

def _load(p):
    E = torch.load(p, map_location="cpu")
    if isinstance(E, dict):
        E = E.get("wake_rows", E.get("embeddings", next(iter(E.values()))))
    assert E.shape[0] == TOTAL_VOCAB, f"{p}: {tuple(E.shape)} vs vocab {TOTAL_VOCAB}"
    return E.float()

PRE  = morph_null_report(_load(PRE_EMB),   label=f"{MODEL_TAG} PRE  (P3 start)")
POST = morph_null_report(_load(FINAL_EMB), label=f"{MODEL_TAG} POST (P3 end)")

d = POST["excess_over_chance"] - PRE["excess_over_chance"]
print(f"\n{'='*66}\n{MODEL_TAG.upper()} -- CONTROL ARM RESULT\n{'='*66}")
print(f"  {'quantity':>24} {'pre':>12} {'post':>12} {'delta':>12}")
for k, n in (("observed_consistency","consistency"), ("null_mean","null mean"),
             ("excess_over_chance","EXCESS")):
    print(f"  {n:>24} {PRE[k]:>12.6f} {POST[k]:>12.6f} {POST[k]-PRE[k]:>+12.6f}")
print(f"  {'groups z<-3':>24} {PRE['n_groups_below_z3']:>12d} "
      f"{POST['n_groups_below_z3']:>12d} "
      f"{POST['n_groups_below_z3']-PRE['n_groups_below_z3']:>+12d}")
print(f"\n  >> EXCESS DELTA WITHOUT MORPHEME SUPERVISION: {d:+.6f}")
print(f"     Mistral, WITH supervision, moved +0.070773.")
if abs(d) < 0.01:
    print("  >> Reads as NULL. The untreated arm does not move, so Mistral's")
    print("     movement is attributable to the morpheme loss and not to the")
    print("     LM objective or the other auxiliaries.")
else:
    print("  >> MOVED. The LM objective or another auxiliary shifts morpheme")
    print("     geometry on its own. Mistral's +0.0708 is then an upper bound")
    print("     on the morpheme loss's contribution, not a measurement of it.")

Path(OUT_JSON).write_text(json.dumps(
    {"model": MODEL_TAG, "pre": PRE, "post": POST, "excess_delta": d,
     "mistral_treated_delta": 0.070773}, indent=2))
print(f"\n[SAVED] {OUT_JSON}")
