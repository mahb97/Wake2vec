### 2025-12-02 Paid Storage & TinyLlama Weirdness

- **Google Drive upgrade:**  
  Had to upgrade Google storage to accommodate all the Wake checkpoints / emb snapshots.  
  Cost: £2.47 for 1 TB. Today’s run is officially not free and Wake2vec now has a direct monetary footprint. 

- **Context:**  
  - Heavy checkpointing setup (HF checkpoints + custom full checkpoints + Sentry mirrors + embedding snapshots) pushed Drive usage over the free limit.
  - The I/O load from this setup is likely also responsible for the apparent “freeze” around step ~900 in the TinyLlama run, even though checkpoints later showed progress up to at least step 1300.

