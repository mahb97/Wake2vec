**Self attenion vs Grassmann**

*Two matched ~13-15M parameter models, identical except for the mixing mechanism:*

| | TransformerLM | GrassmannLM |
|---|---|---|
| **Mixing** | Causal self-attention (4 heads) | Grassmann flows (Plücker on Gr(2,32)) |
| **d_model** | 256 | 256 |
| **Layers** | 6 | 6 |
| **FFN** | 1024 | 1024 |


**Food for thought**

- Perplexity Q: which model fits held-out Wake text better (quantitative)
- Burrows' Delta Q: which model's output is stylometrically closer to the actual FW 

Character-level tokenization is deliberate. Joyce's morphological inventions in the Wake (portmanteaus, neologisms, nonce words) happen at the character level. A BPE tokenizer trained on standard English would tokenize thunderwords into nonsense. Character-level lets both models learn Wake's actual letter patterns.

- Linear reduction: d=256 -> r=32
- Multi-scale causal pairing at offsets {1, 2, 4, 8, 12, 16} (looks backward, not forward — causal)
- Vectorized Plucker coordinates: p_ij = z_t[i]*z_past[j] - z_t[j]*z_past[i] producing C(32,2)=496 features
- Project back to d=256
- Sigmoid-gated fusion: alpha*h + (1-alpha)*gLinear reduction: d=256 -> r=32

@misc{chong2025attentionneed,
      title={Attention Is Not What You Need}, 
      author={Zhang Chong},
      year={2025},
      eprint={2512.19428},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.19428}, 
}
