## Wake2Vec on a Llama 

**date**: November 14, 2025  
**status**: 2:57am and OOM previals    
**runtime restarts**: 5 (and counting)  
**T4 hours wasted**: 3.5 / "yOu CuRrENtly hAve ZErO CoMPuTe UnITs aVaiLaBLe."

## setup (maybe just myself)

was trying to inject 44,990 Joycean neologisms into Llama-3.1-8B on a free Colab T4 between 9.00pm and 3.00am. why? going to quote Fred Again (Danielle) here: "sometimes I want to feel the pain" and I also don't have A100 money. 

*hashtag hot girls use free Colab.*

## best to avoid...

1. **The FP16 incident** - "Attempting to unscale FP16 gradients" feel like crying 
2. **OOM speed run** - 2 steps into training CUDA said *nope*
3. **Model downgrade** - Llama-3.1-8B → Llama-3.2-3B (at least my HF account looks better)
4. **The Loss that never was** - Step 20: loss = 50,462,697. Step 40: loss = inf. Models really said "fuck this"
5. **NaN embeddings** - yh that was a fun one 
6. **Runtime restart speed run** - 5 restarts in one session i don't have much to say for myself lol. someone gimme gpu pls 

## lessons learned 

- T4s have 15GB VRAM until you actually wanna use it (if you're quiet enough you can hear it laugh).
- `seq_len=1024` is a suggestion, (512 gang rise up in London town)
- spherical initialization sounds cool until your gradients explode. you know what else sounds cool? Berghain
- dont talk to me about loss unless i have beatiful screenshots 
- embeddings need emotional support via the no longer opt. gradient clipping 
- 1 runtime restart = 20 minutes of model loading. so she cries because there aren't enough hours in the day for machine learning. 
- i *had* T4 allocation and now i have nothing 

## cost/benefit

**cost: total debugging time**: 4 hours  
**benefit: actual training time**: 4, maybe 6 minutes (yo it's another day tomorrow)  
**cost: models loaded and discarded**: 5  
**benefit: Rosalia**: sat through that a couple of times (minus Divinize, only the bees know)

## current log

copied base embeddings + noise instead of doing math but runtime has finished soo...

**embeddings injected**: 44,195 Joycean tokens  
**GPU memory**: 7.2GB / 14.7GB (dude i had hope whatever)   
**T4 allocation left**: none 

things that will have to wait until the above is resolved: stage 2 (morphological amplification) and all the funnn
