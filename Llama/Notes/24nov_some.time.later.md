# Wake2Vec Llama P1 Fresh Training 

## Training Configuration

**Model:** meta-llama/Llama-3.2-1B  
**Vocabulary:** 172,451 tokens (128,256 base + 44,195 Wake)  
**Dataset:** 720 chunks × 512 tokens  

**(re)training...:** Training in progress (step 77/1300)

### Hyperparameters
- **Max steps:** 1300
- **Learning rate:** 5e-4
- **Batch size:** 1
- **Gradient accumulation:** 16 (effective batch: 16)
- **Sequence length:** 512 tokens
- **Warmup steps:** 65

### Loss Trajectory
- Step 20: 7.310
- Step 40: 7.256
- Step 60: 6.723

## bUt WhY THe FresH TrAInInG?

because i don't have any embeds saved...

## Technical Notes

- Training only Wake token embeddings (44,195 tokens), base Llama vocab frozen
- Gradient masking ensures base embeddings unchanged
- 4-bit quantization (NF4) for memory efficiency on T4
- Spherical initialization for new Wake tokens
- Sentry backup mirrors all checkpoints to Drive

**more vibes:** Beto’s Horns (fred remix) / Superrich (Alok, me n ü, Ten Fé)

- note to self, the Llama still needs a manifesto (just poetry, more and more html waste) 
