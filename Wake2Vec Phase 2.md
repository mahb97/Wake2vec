# Wake2Vec Phase 2: Full Model Fine-Tune

**status:** Readyyyy
**prerequisites:** A complete Phase 1 (embeddings trained to step 1100)

## Overview

Phase 2 unfreezes the full model and fine-tunes on Finnegans Wake with conservative hyperparameters. Unlike P1's embedding-only warmup, P2 adapts the entire model's behavior to Wake's linguistic patterns while preserving the embedding geometry established in P1.

### key differences 

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| Trainable params | Embeddings only | Full model |
| LoRA targets | `q_proj` (frozen) | `q_proj`, `v_proj`, MLP layers |
| Learning rate | 5e-5 | 2e-5 |
| Training duration | 1100 steps | 2 epochs |
| Validation | None | Held-out set with early stopping |
| Batch size | 1 | 8 |
| Gradient accumulation | 16 | 2 |

## Hyperparameters
```python
# P2 Configuration
EPOCHS = 2
LR = 2e-5
WARMUP_RATIO = 0.10
BATCH_SIZE = 8
GRAD_ACCUM = 2
WEIGHT_DECAY = 0.01
SAVE_STEPS = 200
SEQ_LEN = 512              # or 384 
EARLY_STOP_PATIENCE = 2
LORA_RANK = 8              # or 16 
```

## Data Requirements

### Dataset Split
- **Train blocks:** 1,566
- **Valid blocks:** 174  
- **Total sequences:** 1,740

### Vocabulary
- **Base tokenizer:** ~32,000 tokens
- **Wake additions:** 447-534 tokens (varies by lexicon)
- **Final vocab size:** ~33,098 tokens

### Input Format
- **Sequence length:** 512 tokens (384 on memory-constrained T4)
- **Corpus:** Finnegans Wake plain text, blockified
- **Validation strategy:** Held-out blocks for early stopping

## Architecture Changes

### LoRA Configuration
```python
LoraConfig(
    r=8,                    
    lora_alpha=16,          
    lora_dropout=0.1,       
    target_modules=[
        "q_proj",           
        "v_proj",           
        "gate_proj",       
        "up_proj",          
        "down_proj"         
    ],
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Training Strategy
1. Load P1 final embeddings from `wake_llama_P1/final/embed_tokens.pt`
2. Initialize model with P1 vocabulary (expanded tokenizer)
3. Attach LoRA adapters to attention + MLP layers
4. Unfreeze all parameters (model + embeddings + LoRA)
5. Train on Wake corpus with validation monitoring
6. Apply early stopping when validation loss plateaus

## taking a guess

### Performance Metrics
- **Validation loss:** Convergence within 2 epochs
- **Perplexity:** Lower than P1 baseline on Wake text
- **Training time:** ~6 to infinity 

### KPIs (happy monday)
- Model generates coherent Wake-style text
- Wake tokens used appropriately in context
- Embedding geometry from P1 preserved
- No catastrophic forgetting of base vocabulary

## Memory Management (T4)

### Expected VRAM Usage
- **Model (4-bit):** ~2.5 GB
- **LoRA adapters:** ~0.5 GB
- **Optimizer states:** ~3-4 GB
- **Activations (batch=8):** ~6-7 GB
- **Total:** ~12-14 GB

## File Structure
```
wake_llama_P2/
├── checkpoints/
│   ├── checkpoint-200/
│   ├── checkpoint-400/
│   └── checkpoint-best/
├── final/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── tokenizer files
├── logs/
│   └── training_log.json
└── eval/
    ├── val_loss_curve.png
    ├── perplexity_report.txt
    └── sample_generations.txt
```
