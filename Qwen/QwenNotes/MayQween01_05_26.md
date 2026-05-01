# wake2vec devlog 2026-05-01

## Qwen 2.5-14B P1 session 26 (resuming from step 1880)

it's may now, some joke about "It's just a spring clean for the May qween". this one should finish P1 by mid-May, two and a half months after starting.

Resuming from `sentry_step_1880.pt` with `STEP_OFFSET=1880`.

### P1 loss table (recent)

| Step | Train | Val | Session | Notes |
|------|-------|-----|---------|-------|
| 1700 | 232.25 | 15.89 | 22 | broke 16.0 |
| 1750 | 189.83 | 15.81 | 23 | continued descent |
| 1800 | 185.09 | 15.79 | 25 | |
| 1850 | 220.09 | 15.72 | 25 | second train spike |
| 1900 | 184.87 | 15.67 | 26 | gift that keeps on giving |

---

## P2 strategy note (planning ahead)

at some point in the next ~14 sessions the Qween will finish P1, and can then finally launch into P2. this is non-trivial because P1 uses the **WakeOverlay** architecture, which is a custom `nn.Embedding` subclass that holds only the 43,824 Wake embeddings as a separate tensor, scattered onto the frozen base at forward pass.

PEFT/LoRA expects a standard `nn.Embedding`. so the P1 to P2 transition needs a merge step:

### The merge strategy propsal 

```python
# at end of P1:
base_emb = base_model.get_input_embeddings().weight.detach().clone()  # 152,064 × 5,120
wake_emb = wake_overlay.weight.detach().clone()                       #  43,824 × 5,120
merged = torch.cat([base_emb, wake_emb], dim=0)                       # 195,888 × 5,120
torch.save(merged, 'qwen_p1_merged_embeddings.pt')
```

### P2 loads the merged matrix into a standard model

```python
# fresh Qwen 2.5-14B with default nn.Embedding (no overlay)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B", ...)
model.resize_token_embeddings(195888, mean_resizing=False)
merged = torch.load('qwen_p1_merged_embeddings.pt')
model.get_input_embeddings().weight.data.copy_(merged)
model.lm_head.weight = model.get_input_embeddings().weight  # tie

# freeze ALL embeddings (same as Llama P2)
for p in model.get_input_embeddings().parameters():
    p.requires_grad = False

# LoRA on attention and MLP, identical config to Llama P2
peft_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_cfg)
```

WakeOverlay was a P1 architectural solution to the gradient masking problem on a 152K-vocab model. once P1 finishes, all embedding rows are frozen for P2 while the Wake rows just need to exist in the matrix at the right indices. The overlay's selective-row-update mechanism is no longer needed, and since LoRA only touches attention/MLP, the embedding storage format becomes irrelevant

---

fred uploaded a 100 hour long video to youtube. 

So here is some Fred Again... [Facilita](https://soundcloud.com/fredagain/facilita?in=fredagain/sets/usb002&si=2fb5ffcaa6dc4b1199081c1b9792eaaf&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
