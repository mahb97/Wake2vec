# Frying the Embedding Layer

> The train/validation gap exists **on purpose**.

---

## 1. Reframing P1

Since the risk register has gone out the window and timelines are now **flexible**, this is the new P1 run:

- **P1a – step 400**  
  Val loss still improving. Embeddings drifting, generalisation OK.

- **P1b – step 800**  
  Val loss flattens or ticks up slightly, train loss keeps dropping. Start of real overfit.

- **P1c – step 1300**  
  Continue training long after val has stopped caring. 

Plan:

- log train/val loss throughout,  
- snapshot embedding tables at key steps,  
- treat the validation gap as a *signal* about where the manifold melts, not a bug.
- *for the vibes:* Superrich, Alok, me n ü, Ten Fé (https://soundcloud.com/livealok/superrich)


---

## 2. Instrumentation (next rerun)

Probably need one more rerun with the extra scaffolding:

```python
import os, torch

CHECK_STEPS = [400, 800, 1300]

def save_emb_snapshot(step, model, tokenizer, run_dir):
    os.makedirs(os.path.join(run_dir, "snapshots"), exist_ok=True)
    emb = model.get_input_embeddings().weight.detach().cpu()
    payload = {
        "step": step,
        "emb": emb,
        "vocab": tokenizer.get_vocab(),
    }
    path = os.path.join(run_dir, "snapshots", f"emb_step_{step}.pt")
    torch.save(payload, path)
    print(f"[SNAP] Saved embeddings at step {step} -> {path}")
