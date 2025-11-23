# Colab Updates (November 2025)

**Date:** 2025-11-13

## key changes for Wake2Vec training

### CUDA & PyTorch Updates
- **cuda-python**: `12.6.2` → `12.9.4`
- **cupy-cuda12x**: `13.3.0` → `13.6.0`
- **jax/jaxlib**: `0.5.3` → `0.7.2`

### impact on pipeline
The default Colab environment now ships with torch 2.8.0, which conflicts with our tested stack (`torch 2.5.1` + `bitsandbytes 0.43.3`).

### Updated Installation 

#### Cell 1 - Nuclear Uninstall
```python
# uninstall for Nov 2025 Colab
!pip uninstall -y torch torchvision torchaudio triton bitsandbytes transformers accelerate peft jax jaxlib flax -y
!pip cache purge

import os
os.kill(os.getpid(), 9)
```

#### Cell 2 - Install Compatible Versions
```python
import os
os.environ["TRANSFORMERS_NO_TORCHAO"] = "1"

# avoid pulling torch 2.8.0
!pip install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

!pip install -q --no-cache-dir \
    triton==3.1.0 \
    bitsandbytes==0.43.3 \
    transformers==4.45.2 \
    accelerate==0.34.2 \
    peft==0.13.2
```

**Expected output:**
```
torch: 2.5.1+cu121 | cuda: 12.1
bnb: 0.43.3 | triton: 3.1.0
```
