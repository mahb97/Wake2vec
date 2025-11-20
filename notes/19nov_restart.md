## 2025-11-19 (Day 9 i mean who is counting at this point...)

**Mission:** Complete Wake2Vec Phase 1 (0→1300)

**Weights** None saved so restarted yet again. fun. 

**actions taken:**
- Verified checkpoint corruption via SafeTensors inspection
- Rebuilt from ground zero with proper vocab extension
- Created clean `wake2vecP1` directory structure
- Extended TinyLlama vocab: 32K → 76.5K tokens (44,990 Wake tokens)
- Initialized checkpoint-0 with mean embedding strategy

**Kpi log**
- Started: 0→1300 at ~10s/step
- Current: Step 280/1300
- Loss trajectory: 8.46 → 2.89 (66% reduction in 250 steps)
- Validation loss: 5.67 at step 200
- ETA: ~3 hours total runtime

**Status:** T4 disconnected at step 510. Resume tomorrow from checkpoint-500. 800 steps remaining (~2-3 more sessions).

**here is a free reading suggestion:** 
*Blockchain Chicken Farm: And Other Stories of Tech in China's Countryside, Xiaowei Wang (FSG Originals, 2020)*
