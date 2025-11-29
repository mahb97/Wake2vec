## Training Snapshots (2025-11-29)

### LLaMA P1 (embedding-only warm-up)

| Step | Training loss |
|------|---------------|
| 20   | 7.3083        |
| 40   | 7.2757        |
| 60   | 6.9490        |
| 80   | 6.6012        |
| 100  | 6.1583        |
| 120  | 5.8085        |
| 140  | 5.6777        |
| 160  | 5.2230        |
| 180  | 5.2684        |
| 200  | 4.7678        |

The curve fine for this stage.

---

### TinyLlama P1 v2 (embedding-only, with validation)

TinyLlama is starting much closer to the “random baseline” entropy, but the train loss is moving. Validation hovers in the 11.x range.

Header:

- Train blocks: 1299  
- Val blocks: 145  
- Steps: 2000  
- Trainable params: 156,672,000 (embedding table; base rows masked)

Current snapshot:

| Step | Training loss | Validation loss |
|------|---------------|-----------------|
| 200  | 10.7318       | 11.2044         |
| 400  | 9.7591        | 11.0242         |

So far: gentle improvement in val, heavier drop in train; the planned P1a/P1b/P1c split will map exactly where that gap starts to open.

---

Joyce x ROSALÍA: it's like proven that embeddings fry better with...  
- [LA YUGULAR – ROSALÍA](https://soundcloud.com/rosaliaofficial/la-yugular?in=houseof_kyri/sets/quantum-leaped)  
- [RELIQUIA – ROSALÍA](https://soundcloud.com/rosaliaofficial/reliquia?in=rosaliaofficial/sets/lux-226901126)
