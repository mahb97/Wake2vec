# Wake2Vec P1 Training Log

## 2025-11-20: Resume from checkpoint-500

**Session: 500→900 (estimated T4 timeout)**

### Training progress
```
[ 510] 9.4s/step (avg last 10)
[ 520] 9.2s/step (avg last 10)
[ 530] 9.3s/step (avg last 10)
[ 540] 9.3s/step (avg last 10)
[SNAP] 550
[ 550] 9.6s/step (avg last 10)
[ 560] 9.3s/step (avg last 10)
[ 570] 9.2s/step (avg last 10)
[ 580] 9.3s/step (avg last 10)
[ 590] 9.3s/step (avg last 10)
[SNAP] 600 (pending...)
```

### Loss trajectory

| Step | Training Loss |
|------|---------------|
| 500  | 1.30 (resume) |
| 550  | 1.08          |
| 600  | [pending]     |

### Infrastructure

- **GPU:** T4, stable at 12.3 GB VRAM
- **Speed:** ~50 steps per 16 minutes
- **Backups:** Sentry system active, checkpoints every 100 steps
- **Expected completion:** Step ~900 before timeout, then one final session 900→1300

### Notes

- Resumed successfully from checkpoint-500 using checkpoint-0 tokenizer
- Optimizer state loaded correctly (~5GB overhead from resume)
- Step timer consistent at 9.3s/step
- Bo (cat) contributed to debugging process

### Status

not crying today. training running smoothly. 
