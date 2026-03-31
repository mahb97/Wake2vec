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
## 5. The Dissolving Book Token Puddle (kinda cute)

**Concept:** An open book on screen displaying generated Wake paragraphs. Click anywhere on the page and the words fall from the book (gravity, tumbling, scattering) into a wet puddle of language pooling at the bottom of the screen.

**The key mechanic:** The sentences are strings. Each word remains connected to its neighbours by a visible thread. the probabilistic chain that generated them. Click a word in the puddle and pull the string upward: the connected words follow, dragging behind like fish on a line. Drop them back onto the page and they snap into new positions. The reader recomposes Joyce from the model's raw output.

**What it means:**
- The generated text appears as coherent prose, invoking the illusion of authorship
- Click and the illusion dissolves and tokens become visible as tokens, individual units on probability strings
- The puddle is the latent space made physical: a pool of disconnected potential
- Pulling words back up = recomposition, the reader becomes author
- The strings = attention patterns, token dependencies, the invisible architecture that holds language together
- Rearranging on the page = what Joyce did with language, what Wake2Vec does with embeddings: taking apart and reassembling

**Technical approach:**
- Matter.js or similar for 2D physics (gravity, rigid body words, fluid-like pooling)
- Pretext for text measurement and page layout
- Words as physics objects connected by spring constraints (visible strings)
- Click-and-drag with string tension, connected words follow with slight delay
- Snap-to-grid on the page for recomposition
- Could save reader-composed arrangements for collaborative Wake rewriting

---

## Tech Notes

- Pretext = text height without DOM measurement, manual line routing, width-tight multiline UI
- Pure JS layout, zero DOM reads in the hot path
- Knuth-Plass justification relevant for Joyce (obsessive about text appearance)
- Could build on existing `dataviz_p1_embeds.py` for data pipeline, Pretext for frontend

See [Pretext](https://github.com/chenglou/pretext)

---
