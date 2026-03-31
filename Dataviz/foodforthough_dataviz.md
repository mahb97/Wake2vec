# Wake2Vec Visualisation Ideas

## Inspired by Chenglou's Pretext demos (chenglou.me/pretext)

**Date:** 31 March 2026

---

## 1. Riverrun Variable Typographic ASCII

**Source demo:** Variable Typographic ASCII

A literal `riverrun` on screen with Wake2Vec generated text.

- Top-k neighbour adjustment shifts the path of the river
- River path = token probability landscape
- Adjust k and watch the text reroute through different semantic channels
- Opening word of Finnegans Wake becomes the interface metaphor
- Potential for multi-model comparison: TinyLlama river vs Llama 1B river vs Qwen river, different currents from the same source

---

## 2. Temperature Orbs via Editorial Engine

**Source demo:** Editorial Engine

Animated orbs representing temperature or device options, with live text reflow.

- Drag temperature orb: text fragments at high temp, tightens at low temp
- User physically controls how Wakean the output is
- Could also map orbs to:
  - λ_morph (morphological pressure)
  - λ_device (stylistic device weight)
  - Model scale (orb size = parameter count)
- Multi-column flow: TinyLlama output in one column, 1B in another, both responding to same prompt, reflowing in real time
- Wake text routing around images of actual Joyce locations (Avenue Charles Floquet, Rue Edmond Valentin, etc.)

---

## 3. Embedding Settlement Animation

**Current state:** Static PCA scatter plot (Step 50 vs Step 3000, TinyLlama P1)

**Upgrade ideas:**
- Animate the training process: slider from step 0 to step 3000
- Wake tokens as particles drifting through embedding space
- Ogden's Basic English landmarks as fixed attractors
- Particle field approach from Pretext's typographic ASCII demo
- Cyan cluster breathing outward, tokens finding semantic homes
- Could layer P1 → P2 → P3 as continuous animation showing the full pipeline
- Colour shift as tokens move through phases (P1 = cyan, P2 = purple, P3 = gold?)

---

## 4. Multi-Architecture Comparison

- Side-by-side rivers: same prompt, different models
- Visual diff of how model scale changes generation character
- TinyLlama = dense Wakean pastiche (narrow, fast river)
- Llama 1B = Wake-inflected Victorian narrative (wider, slower river)
- Llama 3B / Qwen 14B = TBD
- Temperature sweep as animation: 0.5 → 1.2, watch text transform

---

## Tech Notes

- Pretext = text height without DOM measurement, manual line routing, width-tight multiline UI
- Pure JS layout, zero DOM reads in the hot path
- Knuth-Plass justification relevant for Joyce (obsessive about text appearance)
- Could build on existing `dataviz_p1_embeds.py` for data pipeline, Pretext for frontend

See [Pretext](https://github.com/chenglou/pretext)
---
