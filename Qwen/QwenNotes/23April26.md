# wake2vec devlog 2026-04-23

## Qwen 2.5-14B P1 session 22 (resuming from step 1600)

one unplanned rest day for the Qween, blame UCC. 1,400 steps to go.

Resuming from `sentry_step_1600.pt` with `STEP_OFFSET=1600`.

### P1 loss table (recent)

| Step | Train | Val | Session | Notes |
|------|-------|-----|---------|-------|
| 1400 | 197.10 | 16.23 | 19 | |
| 1450 | 186.83 | 16.25 | 19 | |
| 1500 | 195.01 | 16.14 | 20 | |
| 1550 | 192.06 | 16.05 | 21 | |
| 1600 | 182.28 | 16.11 | 21 | |
| 1650 | 189.35 | 16.01 | 22 | |
| 1700 | 232.25 | 15.89 | 22 | **BROKE 16.0, that geom space is shaking and i'm crying lol (not actually though)** |

why the emotions? well, fourteen billion parameters trying to fit on a consumer GPU, stateless resumes, Adafactor instead of AdamW because there's no VRAM for momentum buffers, SEQ_LEN 128 because anything more OOMs. Everything about training this is a compromise with reality, and it just kept working.

Also, this has the Wake overlay architecture, so the only model in the lineup with a custom class that holds the Wake embeddings separately and scatters them over the frozen base at forward pass. The others are just 4-bit + gradient masking, but Qween got her own engineering.

---

listened to this on the bus from Cork to Dublin last night, fucking bangs: [Fred Again, may 8th 2021](https://soundcloud.com/rinsefm/fredagain080521?si=9a97571ce53f4ea1a5ab950d7f2f7ef4&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing)
