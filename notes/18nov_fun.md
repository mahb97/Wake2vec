# Wake2Vec Phase 1: Embedding-Only Fine-Tuning (Steps 750→1300) - some notes 

**Status:** Active Training  
**Current Step:** 752/1300  
**Model:** TinyLlama-1.1B  
**Task:** Embedding injection for 44K Wake tokens from *Finnegans Wake* (**coughs* the gospel) 

---

## Training Config

**Hardware & Runtime:**
- GPU: T4 (Colab)
- Expected duration: ~12 hours
- Throughput: ~80s/step (with gradient checkpointing + Drive mirroring)

**Hyperparams:**
- Batch size: 1
- Gradient accumulation: 16
- Learning rate: 5e-4
- Optimizer: Adafactor
- Max sequence length: 256 tokens
- Gradient checkpointing: Enabled

---

## Checkpoints

**Save frequency:** Every 100 steps (800, 900, 1000, 1100, 1200, 1300)

**Critical checkpoints:**
- checkpoint-750-rebuilt (starting point)
- checkpoint-800 
- checkpoint-1000 (safety net for resume)
- checkpoint-1300 (target completion)

---

## Monitoring

**Callbacks active:**
- `EvalEveryNSteps(200)` 
- `SentryMirror()`
- `EmbeddingSnap(50)` 
- `StepTimer()` 

---

## Notes

**On T4 reliability:**  
I didn't think a GPU would gaslight me but here we are (*swears in those nice words that Billie used to describe Elon last week*). "At your current usage level," is a lie. Lie better pls. Should probably make a note about Fred Again's *Studio Live (London, April 2021)*. Correlation between Fred and successful checkpoint mirroring remains scientifically unverified but hey at least he gets me going. 

**On persistence:**  
This is day 8+ of attempting to complete Phase 1. I'm not crying, you are. 

---

## Recovery Plan

**If/when T4 times out before 1300:**
1. cry
2. Resume from highest numbered checkpoint with weights
3. Repeat until 1300

**Expected loss curve:**
- Training loss at 750: ~0.32
- Validation loss at 800: ~6.31 (target)
- Final metrics TBD at 1300

---

*Last updated: 2025-11-18*
