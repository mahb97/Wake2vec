# Wake2Vec P1 Llama Restart, Nov 26, 2025

and the plot thickens...discovered the original Llama P1 training (0-600) never saved embedding snapshots lol - only LoRA adapter weights. 

TinyLlama P1 succeeded because it saved `emb_snaps/` every 50 steps. Llama P1 did not. Without the trained embeddings, the checkpoint is useless for resume. 

Will retrain Llama P1 from step 0 with proper embedding snapshot saving. 

**a vibe:** USB002
