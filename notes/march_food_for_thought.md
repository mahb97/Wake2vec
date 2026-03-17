# some alternative models for Wake2vec if I ever get better GPU access 

Current pipeline runs across TinyLlama, Qwen 2.5-14B, Llama 3.2-3B, and Llama 3.1-8B on free-tier Colab

## list and thoughts

models for TU:

- Nemotron 3 Super 
- Qwen3-235B-A22B
- Qwen3-32B (Dense)
- Llama 4 Maverick
- NVIDIA Nemotron 3 Super (120B-A12B)
- DeepSeek-V3 (671B-A37B)
- NVIDIA Nemotron 3 Ultra (unreleased)


## paper Qs 
- Does SSM architecture (Nemotron Super's Mamba layers) affect morpheme-level embedding geometry in ways that matter for Phase 1 surgery? 
- Do the assumptions behind μp → UP still hold in state-space models where there are no explicit attention heads to interrogate?
- At what parameter scale does the Wake's idiolect actually "break through" pre-training? Is there a threshold effect, or is it smooth?
- MoE routing introduces token-level selectivity so which experts activate on nonce formations vs. standard lexis?

## RL for Wake2vec

- train stylometric discriminator which is then used as the reward model.
- or some kind of small-scale RLHF where outputs on Joycean authenticity are scored (would be nice to colab with other Joyceans or some FW reading group).

(needs to reinforce specific behaviours that LoRA alone doesn't reliably produce)

- can literary style/a linguistic fingerprint be reinforced through adversarial stylometric feedback? 

## Wake2Vec Generative Loop & Memorization Analysis (Paper 2 Concept)

Design notes for a post-Phase 3 generative pipeline using the trained models (TinyLlama/Llamas/Qwen) to extend the morpheme dataset, generate synthetic training data, and empirically measure memorization vs. novel generalization. 

# Overview

After Phase 3 (morpheme-compositional alignment), the models have internalised Joycean morphological patterning. Rather than treating this as the end of the data pipeline, the next step is to use the trained models as generators, curate their outputs, verify them against the source corpus, and feed the results back into future training. Simultaneously, the verification layer produces an empirical memorization/novelty analysis which is necessary for judging if the model can produce Wake-like style or if it's just memorizing wake tokens. 

Phase 3 model
    │
    ├─► Word-level generation
    │       │
    │       ├─► AntConc check ──► memorized / compositional / novel taxonomy
    │       │
    │       └─► Human curation ──► extended morpheme dataset ──► Phase 4+ training
    │
    └─► Sentence-level generation
            │
            ├─► AntConc check (same taxonomy, sentence level)
            │
            └─► Filtered synthetic corpus ──► Phase 4+ training data


# Branch 1: Word-Level Generation → Morpheme Dataset Extension

# Rationale

Prompting Phase 3 models to generate candidate nonce lexis is a way of probing what the embedding geometry has actually learned. If the morpheme-compositional alignment (Phase 3) did what it was supposed to do, the models should produce forms that exhibit Joycean compounding logic and not just surface mimicry of the training corpus.

