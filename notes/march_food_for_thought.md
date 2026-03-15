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