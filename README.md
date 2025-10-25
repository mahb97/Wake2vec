# Wake2vec
TL;DR

I extend a tokenizer with 1,000+ Finnegans Wake tokens and fine-tune only the input embeddings (plus an optional tiny LoRA) to study controlled semantic drift under tight compute. I report geometry shifts (PIP loss, isotropy, top-k neighbor overlap), language behavior (perplexity on FW vs modern prose), and qualitative intrusion. All results are reproducible on a Colab T4.

# Why this project

Style transfer is usually prompt magic or full fine-tuning. Wake2vec shows a third path: embedding surgery. By adding Wake-specific tokens and training just the embedding table (optionally a tiny attention LoRA), we bend local semantic neighborhoods with clear, measurable effects and minimal compute.

This is designed as a compact research artifact: clean hypotheses, careful controls, and readable receipts.

# Method

- Lexicon: Builds a Joyce-heavy list (≥1,000 tokens) from Finnegans Wake with Zipf rarity and stopword filtering.
- Tokenizer augmentation: Adds bare tokens and SentencePiece start-of-word variants (▁token), then ties output head to input embeddings.
- Smart init: Initialises each new vector as the mean of its base subwords + small noise.
- Training: Freezes almost everything; trains embeddings only (+ optional LoRA r=8 on Q/K/V/O). Curriculum = Wake-dense paragraph windows.

# Evaluation

- Geometry: PIP loss, isotropy, top-k neighbor overlap (before/after).
- Language: perplexity on FW vs a small modern-prose slice.
- Behavior: Joyce-style “intrusion” in fixed prompts; neighbor maps.

# Quickstart (T4 or CPU)

- pip install -r requirements.txt
- Run lexicon: regenerate wake_lexicon.txt.
- Run token injection → smart init → training → metrics → hero.
- GPU (T4): keep BASE_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0".
- CPU fallback: set BASE_MODEL="distilgpt2" to smoke-test the full pipeline.

# Artifacts saved automatically

- results/metrics_summary.csv (PIP, isotropy, top-k)
- results/hero.png (UMAP + completions)
- Wake2vec_adapter/minipack/* (embedding rows for new tokens)  

# Citation / Credit

- James Joyce, Finnegans Wake (text for token mining).
- Base model: TinyLlama-1.1B-Chat.
- Techniques inspired by embedding surgery, retrofitting, and LoRA literature.
