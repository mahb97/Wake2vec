# wake2vec devlog 2026-03-18

## TinyLlama 1.1B P3b session 2 (resuming from step 300)

resumed from checkpoint-300. early stop was 2/3 when T4 cut last session, but callback has reset on resume, so at this point I'll just let it do the full 1000 steps. 

L_device hasn't moved across 1,700+ combined P3/P3b steps. 

### P3b loss table (continued)

| Step | L_total | L_lm | L_morph | L_device | Val | Early stop |
|------|---------|------|---------|----------|-----|------------|
| 0 | 3.3345 | 2.9223 | 0.0002 | 0.1993 | — | — |
| 25 | 3.1146 | 2.6596 | 0.0002 | 0.2208 | — | — |
| 50 | 2.8106 | 2.4213 | 0.0002 | 0.1879 | — | — |
| 75 | 3.5489 | 3.1264 | 0.0002 | 0.2045 | — | — |
| 100 | 3.0486 | 2.6129 | 0.0002 | 0.2111 | **3.8411** | best ✓ |
| 125 | 2.6515 | 2.2122 | 0.0002 | 0.2129 | — | — |
| 150 | 3.1455 | 2.7470 | 0.0002 | 0.1925 | — | — |
| 200 | 2.5526 | 2.1544 | 0.0002 | 0.1923 | **3.8930** | 1/3 |
| 225 | 2.4912 | 2.0348 | 0.0002 | 0.2215 | — | — |
| 250 | 3.1178 | 2.6897 | 0.0002 | 0.2073 | — | — |
| 275 | 2.6355 | 2.1859 | 0.0002 | 0.2180 | — | — |
| 300 | 2.7273 | 2.2882 | 0.0002 | 0.2128 | **3.9284** | 2/3 |
| 400 | 2.4965 | 2.1011 | 0.0002 | 0.1909 | **3.9905** | |
| 500 | — | — | — | — | **3.9901** | best ✓ |


---

it's a [dawn chorus](https://soundcloud.com/thomyorkeofficial/dawn-chorus?si=3090780a2c9a48b08964d8a8f7410964&utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing) and I've been missing my abuser every single day. 
