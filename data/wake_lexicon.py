# -*- coding: utf-8 -*-
"""
# Wake2vec - Build a Joyce-Heavy Lexicon

---

## What this notebook does
- **Load & clean** a *Finnegans Wake* text file (strip Gutenberg/DP banners; anchor at `riverrun`).
- **Tokenize** with a diacritic-aware regex; normalize to NFC.
- **Score** each type by:
  - frequency in FW
  - Zipf frequency in general English (`wordfreq`)
  - a simple ovelty score `7 − Zipf`
- **Filter** out stop-ish function words and high-Zipf English.
- **Select** a Joyce-heavy list (target ≥ 1,000 unique tokens), prioritising rare, recurrent, longer, and non-ASCII items.
- **Export**:
  - `wake_lexicon.txt` one token per line (to inject)
  - `wake_lexicon_scored.csv` full table for audit/plots

---

## Inputs
- `finnegans_wake.txt` (UTF-8).  
  > I upload it in the notebook or place it under `data/`.

## Outputs
- `/content/wake2vec_lexicon_clean/wake_lexicon.txt`
- `/content/wake2vec_lexicon_clean/wake_lexicon_scored.csv`

---

## Controls (tweak if needed)
- **Target size:** `TARGET_MIN = 1000`
- **English filter:** `ZIPF_MAX_START = 3.0` (lower → spikier Wake list)
- **Novelty floor:** `NOVELTY_MIN = 3.0`
- **Recurrence:** `MIN_FW_FREQ = 2`
- **Stop-ish list:** see the hard-drop set in the notebook

> If the head of the list reads too plain-English, I set `ZIPF_MAX_START = 2.75` and/or `NOVELTY_MIN = 3.5`.

---

## Quality checks
- No exact duplicates  
- 0 stop-ish offenders in the selection (reports if any slip)  
- Zipf distribution skewed low (Joyce-heavy)  
- Summary stats printed for frequency/novelty/length

---

## How this feeds Notebook 02
I use `wake_lexicon.txt` to augment the tokenizer with both the bare tokens and **`▁token`** start-of-word variants (SentencePiece-style). Then:
1) Smart-init new embeddings as subword means + small noise  
2) Tie `lm_head` to input embeddings  
3) Train embeddings (plus optional tiny LoRA) on Wake-dense windows

---

## Repro notes
- Unicode is normalized to NFC; diacritics preserved (e.g., `fainéants`, `ténèbres`).  
- Regex captures letters with accents + internal apostrophes/hyphens.  
- Zipf frequencies via `wordfreq` (English).  
- Deterministic seed: `42`.

---

## License & credit
- Text for token mining: James Joyce, *Finnegans Wake*.  
- This notebook only builds a lexicon; it does **not** redistribute book text.  
- Techniques inspired by embedding surgery / lexical retrofitting literature.

"""

!pip -q install wordfreq==3.1.1 unidecode==1.3.8

import re, math, json, unicodedata, random
from collections import Counter
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
from wordfreq import zipf_frequency
from unidecode import unidecode

random.seed(42)

try:
    from google.colab import files
    uploaded = files.upload()
    fw_path = list(uploaded.keys())[0]
    raw_text = Path(fw_path).read_text(encoding="utf-8", errors="ignore")
except Exception as e:
    print("Upload failed or not on Colab. Falling back to pasted text.")
    raw_text = """
    [PASTE THE FULL TEXT OF FINNEGANS WAKE HERE IF NOT UPLOADING]
    """

import re, unicodedata

def strip_prologue_epilogue(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    header_patterns = [
        r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"PROJECT GUTENBERG(?:[^\n]*\n){0,50}",
        r"A\s+Distributed\s+Proofreaders\s+Canada\s+eBook(?:[^\n]*\n){0,80}",
        r"Produced\s+by(?:[^\n]*\n){0,40}",
        r"Faded\s+Page(?:[^\n]*\n){0,40}",
    ]
    footer_patterns = [
        r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*",
        r"End\s+of\s+(?:the\s+)?Project\s+Gutenberg.*",
        r"(?:Distributed\s+Proofreaders\s+Canada|Faded\s+Page).*",
        r"Transcriber(?:'s)?\s+Notes?:?(?:[^\n]*\n)*$",
        r"^.*?This\s+eBook\s+is\s+made\s+available.*$",
    ]

    for pat in header_patterns:
        s = re.sub(pat, "", s, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)

    incipit = re.search(r"\briverrun\b", s, flags=re.IGNORECASE)
    if incipit:
        s = s[incipit.start():]

    for pat in footer_patterns:
        s = re.sub(pat, "", s, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)

    return s.strip()

text = strip_prologue_epilogue(raw_text)

print("Characters after strip:", len(text))
print("First 300 chars:")
print(text[:300])

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’-][A-Za-zÀ-ÖØ-öø-ÿ]+)*")

raw_tokens = WORD_RE.findall(text)
tokens = [unicodedata.normalize("NFC", t).lower() for t in raw_tokens]

freq = Counter(tokens)
print("Unique types:", len(freq), "| Total tokens:", sum(freq.values()))
print("Sample 10:", list(freq.items())[:10])

def novelty_score(word: str) -> float:
    z = zipf_frequency(word, "en")
    if math.isinf(z) or math.isnan(z):
        z = 0.0
    return max(0.0, 7.0 - z)

rows = []
for w, f in freq.items():
    if len(w) < 3:
        continue
    w2 = w.strip("'-–—")
    if len(w2) < 3:
        continue
    z = zipf_frequency(w2, "en")
    z = 0.0 if math.isinf(z) or math.isnan(z) else z
    rows.append({
        "token": w2,
        "freq": f,
        "zipf": z,
        "novelty": novelty_score(w2),
        "length": len(w2),
        "nonascii": int(any(ord(ch) > 127 for ch in w2)),
        "ascii": unidecode(w2)
    })

df = (
    pd.DataFrame(rows)
      .drop_duplicates(subset=["token"])
      .reset_index(drop=True)
)
df.head(10)

STOPISH = set("""
the of and to a in that is it for on as with was his her he she they be by
this at from or an are i you we their which not had have but all were so
one out up into over more when there been if may my your our any who than
him like what how old now will after would where them off has two about till
time while then down through man see first well before let too say some here
under ever can good did could come way never only upon its little three every
know still back tell such those it's why made himself just most yet shall again
other though round make he's long being hear last take night i'll own same very
i'm four even always whole day must since sure said poor love that's might life
between look world great you're think nor these name big much place dear hand
there's once give don't having many house right thee mind full behind went thing
away white times king should true best around among high put call thou without eyes
because making light please end get whose woman got used whom word tis going feel
home half left far another saint holy free gone face done fall side show course sweet
she's heard young water years fair hold says mean sir i'd thought against soon came
each better heart black seen hat lay men father lost next both son grand hands along
fine took stop days keep want mine sea nothing fire does ear bad green leave hour gave
yes new number we'll wall pass past knows quite part ere boy bit set myself peace words
much very many any every none each either neither both few several most some all any
""".split())

TARGET_MIN = 1000
MIN_LEN = 3
MIN_FW_FREQ = 1
ZIPF_MAX_START = 3.0     # Zipf >= 3.0
ZIPF_RELAX_STEP = 0.25
NOVELTY_MIN = 3.0        # 7 - zipf >= 3

def wake_heavy_select(df: pd.DataFrame, zipf_max: float) -> List[str]:
    cand = df[
        (df["length"] >= MIN_LEN) &
        (df["freq"] >= MIN_FW_FREQ) &
        (df["zipf"] < zipf_max) &
        (df["novelty"] >= NOVELTY_MIN)
    ].copy()

    # Hard drop stop-ish
    cand = cand[~cand["token"].isin(STOPISH)]

    # Scoring: rarity * recurrence, with bonuses for diacritics and length
    cand["score"] = (
        cand["novelty"] * (1 + 0.10 * (cand["freq"] >= 5))
        * (1 + 0.20 * cand["nonascii"])
        * (1 + 0.15 * (cand["length"] >= 8))
    )

    cand = cand.sort_values(["score","freq","length"], ascending=[False, False, False])

    sel = cand["token"].tolist()
    return sel

zipf_max = ZIPF_MAX_START
picked = wake_heavy_select(df, zipf_max)

# Relax Zipf threshold if unable to reach 1000
while len(picked) < TARGET_MIN and zipf_max < 5.0:
    zipf_max += ZIPF_RELAX_STEP
    picked = wake_heavy_select(df, zipf_max)

# If STILL short (unlikely), top up from rarest remaining (still non-stopish)
if len(picked) < TARGET_MIN:
    fallback = (
        df[(df["length"] >= MIN_LEN) & (df["freq"] >= 1) & (~df["token"].isin(STOPISH))]
        .sort_values(["novelty","freq"], ascending=[False, False])["token"].tolist()
    )
    seen = set(picked)
    for w in fallback:
        if len(picked) >= TARGET_MIN: break
        if w not in seen:
            seen.add(w); picked.append(w)

print(f"Selected {len(picked)} tokens | zipf_max≈{zipf_max:.2f}")
print("Head 30:", picked[:30])

out_dir = Path("/content/wake2vec_lexicon_clean")
out_dir.mkdir(parents=True, exist_ok=True)

(out_dir / "wake_lexicon.txt").write_text("\n".join(picked), encoding="utf-8")

df.to_csv(out_dir / "wake_lexicon_scored.csv", index=False, encoding="utf-8")

print("Saved:", out_dir / "wake_lexicon.txt")
print("Saved:", out_dir / "wake_lexicon_scored.csv")

sel = set(picked)
sub = df[df["token"].isin(sel)].copy()

offenders = [w for w in picked if w in STOPISH]
print(f"Stop-ish offenders in selection: {len(offenders)}")
print(offenders[:30])

high_zipf = sub[sub["zipf"] >= 3.0]["token"].tolist()
print(f"High-Zipf tokens in selection: {len(high_zipf)}")
print(high_zipf[:30])

print("\nSelection stats:")
print(sub[["freq","zipf","novelty","length","nonascii"]].describe())

try:
    from google.colab import files
    files.download("/content/wake2vec_lexicon_clean/wake_lexicon.txt")
    files.download("/content/wake2vec_lexicon_clean/wake_lexicon_scored.csv")
except Exception:
    print("Files saved under /content/wake2vec_lexicon_clean")
