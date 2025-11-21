# This is just a draft: Wake2Vec Phase 1 (Comparative Analysis of Lexicon-Augmented Embedding Training)

because everything is better when you have two google accounts. bee that means double the t4 allocation, keep the projects coming. 

## Abstract

This draft report presents a comparative analysis of embedding-only fine-tuning for LLMs augmented with 44,990 morphologically complex neologisms from Joyce's *Finnegans Wake*. Joycean neologisms are synthetic portmanteaux combining multiple languages and morphemes into single lexical units that challenge standard tokenization and compositional semantics, I have a whole dissertation on this from 2021 but who cares. Because one wasn't enough, this uses two base models (TinyLlama-1.1B and Llama-3.2-1B) with lexicon and train using embed-only optimization while keeping transformer parameters frozen.

## Introduction

You probably know what FW is else you woudn't be here. 

Examples from the Wake lexicon include:
- Multilingual portmanteaux: "misunderstord" (mis- + understood + stored)
- Cross-linguistic blends: "allocution" (all + allocution/elocution)  
- Morpheme stacking: "miserendissimest" (mis- + -issimest superlative)

These tokens test whether embedding-only training can learn compositional structure when:
1. Components span multiple languages
2. Morphological boundaries are intentionally ambiguous
3. Semantic meaning emerges from phonetic + morphemic interaction
4. No parallel corpus exists for supervision
## Methodology

### Vocabulary Extension

Both models were extended from their base vocabularies (32,000 tokens) with 44,990 Wake-specific tokens extracted from the *Wake*, yielding a final vocabulary of 76,500 tokens. New token embeddings were initialized using mean initialization from existing embeddings.

### Training Config

**shared params:**
- Training corpus: *Finnegans Wake* (full text)
- Optimization: Embedding-only (transformer layers frozen)
- Optimizer: AdamW / Adafactor
- Gradient masking: Base vocabulary gradients zeroed
- Embedding tying: Input and output embeddings tied

**TinyLlama-1.1B:**
- Precision: fp32
- Sequence length: 256 tokens
- Learning rate: 5e-4
- Warmup: 5% (65 steps)
- Batch size: 1, gradient accumulation: 16
- Training steps: 1,100

**Llama-3.2-1B:**
- Precision: 4-bit (NF4 quantization, bfloat16 compute)
- Sequence length: 512 tokens
- Learning rate: 5e-5
- Warmup: ~5% 
- Batch size: 1, gradient accumulation: 8
- Training steps: 1,100 (in progress)

### Hardware

Everything run on Google Colab T4 GPU (15GB VRAM) with automatic checkpoint backup system for training continuity across session timeouts.

## Results

### Loss Convergence

**TinyLlama-1.1B (1,100 steps):**
- Initial loss: 8.46
- Final loss: 0.134
- Reduction: 98.4%

**Llama-3.2-1B (400 steps, in progress):**
- Initial loss: 3.16
- Current loss: 0.529 (step 400)
- Reduction: 83.3%

At equivalent step counts:
- Step 200: TinyLlama 3.48 vs Llama 0.89 (3.9x difference)
- Step 400: TinyLlama 1.82 vs Llama 0.53 (3.4x difference)

### Convergence Rate Analysis

Llama-3.2-1B demonstrates significantly faster vocabulary integration:
- TinyLlama requires ~600 steps to reach loss 1.0
- Llama-3.2-1B reaches loss 1.0 at step 180 (3.3x faster)

Despite 4-bit quantization and CPU offloading, Llama maintains superior convergence throughout training, suggesting base model quality dominates precision effects for embedding-only optimization.

## Discussion

### Base Model Effects

The performance gap likely stems from:
1. **Multilingual pretraining**: Llama's stronger multilingual base better handles Joyce's polyglot constructions
2. **Architecture quality**: Llama's improved attention mechanisms may facilitate semantic composition
3. **Tokenization**: Llama's tokenizer may preserve more morphological structure

### Quantization Viability

4-bit quantization with embedding-only training proves surprisingly effective, maintaining learning capacity while enabling larger models on constrained hardware. This suggests embedding optimization is robust to reduced precision in frozen transformer layers.

### Training Efficiency

Sequence length differences (256 vs 512 tokens) may contribute to Llama's advantage, providing richer contextual signals for Wake vocabulary learning. However, this also increases computational cost per step (~19s vs ~9.3s per optimizer update).

## Limitations

- Validation loss increases in both models indicate overfitting on small corpus (910 samples)
- No held-out test set for *Finnegans Wake* specifically
- Comparison confounds multiple variables (architecture, precision, sequence length)

## Conclusion

Embedding-only fine-tuning successfully integrates specialized vocabulary into pretrained language models, with base model quality showing greater impact than precision constraints. Llama-3.2-1B achieves 4x faster convergence than TinyLlama-1.1B despite 4-bit quantization, demonstrating that stronger multilingual pretraining facilitates domain-specific lexicon acquisition. These results support the viability of compute-constrained approaches for specialized NLP applications requiring vocabulary extension.

## Acknowledgments

- Training infrastructure: Google Colab T4 GPU. 
- Joyce, as always. 

---

*Note: This analysis represents work in progress. Llama-3.2-1B training ongoing; final results pending.*
