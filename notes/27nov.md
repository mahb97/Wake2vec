# Wake2Vec Phase 2 Devlog
  
**P2:** Full model fine-tune with LoRA

---

## Session Summary

Resolved Google Drive sync issues blocking P1 artifact access...

## P1 Artifacts Loaded

- **Tokenizer:** `wake2vecP1/checkpoint-0/` (32k base + Wake tokens)
- **Embeddings:** `emb_step1300.pt` 

## P2 Training Config

| Parameter | Value |
|-----------|-------|
| Max steps | 2000 |
| Learning rate | 2e-5 |
| Batch size | 8 |
| Grad accum | 2 |
| Effective batch | 16 |
| LoRA rank | 8 |
| Sequence length | 256 |
| Save steps | 200 |
| Early stopping | 2 evals |

