#!/usr/bin/env python3
"""per-token and per-word surprisal for wake2vec lineup.

Surprisal (Hale 2001) is -log P(t_i | t_0 ... t_{i-1}). A causal LM produces it
at every position under teacher forcing, and mean per-token cross-entropy is its
expectation. 

aim of this script:
1. loads the trained embedding matrix. PEFT writes a resized
   embedding into adapter_model.safetensors and then ignores it on load, so
   PeftModel.from_pretrained() on a P1 or P3 checkpoint supplies the
   BASE embedding geometry with the adapter on top.
2. reproduces the project's exact validation split
   (sklearn train_test_split, test_size=0.10, random_state=42, over
   non-overlapping SEQ_LEN blocks), so "the mean should equal the logged
   validation loss" is a claim that can be verified.
3. fingerprints the embedding it measured, so the devlog records which 
   matrix produced the number.

Examples
--------
Reproduce a P2 logged validation loss:
    python surprisal_extract.py \
      --model mistralai/Mistral-7B-v0.3 \
      --tokenizer  /drive/wake2vec_mistral7b_p2_lora/full_checkpoints/step_0600 \
      --adapter    /drive/wake2vec_mistral7b_p2_lora/full_checkpoints/step_0600 \
      --embeddings /drive/wake2vec_mistral7b_p2_lora/embeddings_frozen.pt \
      --base-vocab 32768 --seq-len 256 --split val --expect 4.705

Corpus-wide profile from a P3 endpoint, with per-word output:
    python surprisal_extract.py \
      --model mistralai/Mistral-7B-v0.3 \
      --tokenizer  /drive/wake2vec_mistral7b_p3_morpheme_strong/final \
      --adapter    /drive/wake2vec_mistral7b_p3_morpheme_strong/final \
      --embeddings /drive/wake2vec_mistral7b_p3_morpheme_strong/final/embeddings.pt \
      --base-vocab 32768 --seq-len 256 --split all --dump-words words.jsonl
"""

import argparse, hashlib, json, math, sys
from pathlib import Path

import torch
import torch.nn.functional as F

# parse args
def parse_args():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--model", required=True, help="base model id or path")
    a.add_argument("--tokenizer", required=True,
                   help="checkpoint holding the EXTENDED tokenizer (not the base model)")
    a.add_argument("--adapter", default=None, help="LoRA adapter directory (P2/P3)")
    a.add_argument("--embeddings", default=None,
                   help="trained embedding matrix: a bare tensor, or a dict with "
                        "'wake_rows' (overlay format) or 'embeddings'")
    a.add_argument("--base-vocab", type=int, required=True,
                   help="embedding rows BEFORE Wake injection")
    a.add_argument("--seq-len", type=int, required=True,
                   help="must match the SEQ_LEN the checkpoint was evaluated with")
    a.add_argument("--text", default="FW_TEXT.txt")
    a.add_argument("--split", choices=["val", "train", "all"], default="val")
    a.add_argument("--expect", type=float, default=None,
                   help="logged value to compare against")
    a.add_argument("--expect-is-composite", action="store_true",
                   help="set when --expect came from a P3 run: its eval_loss is "
                        "LM PLUS weighted auxiliaries, so it will NOT match")
    a.add_argument("--load-4bit", action="store_true")
    a.add_argument("--no-tie", action="store_true",
                   help="do not tie lm_head to the input embedding after loading")
    a.add_argument("--pos-chunk", type=int, default=256,
                   help="positions per log_softmax chunk (memory)")
    a.add_argument("--dump-tokens", default=None)
    a.add_argument("--dump-words", default=None)
    a.add_argument("--out", default=None, help="summary JSON")
    a.add_argument("--limit-blocks", type=int, default=None, help="smoke test")
    return a.parse_args()


# get embeds
def load_embeddings(path, total_vocab, base_vocab, dim):
    """Accept every format this project has written."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "wake_rows" in obj:          # EmbeddingSnapshot 
            rows = obj["wake_rows"]
            if tuple(rows.shape) != (total_vocab - base_vocab, dim):
                raise ValueError(f"{path}: wake_rows {tuple(rows.shape)}, expected "
                                 f"{(total_vocab - base_vocab, dim)}")
            return ("wake_only", rows)
        if "embeddings" in obj:         # P1 sentry format
            obj = obj["embeddings"]
        else:
            raise ValueError(f"{path}: dict without 'wake_rows' or 'embeddings', "
                             f"keys={list(obj)[:6]}")
    if tuple(obj.shape) == (total_vocab, dim):
        return ("full", obj)
    if tuple(obj.shape) == (total_vocab - base_vocab, dim):
        return ("wake_only", obj)
    raise ValueError(f"{path}: shape {tuple(obj.shape)} matches neither the full "
                     f"matrix {(total_vocab, dim)} nor the Wake block "
                     f"{(total_vocab - base_vocab, dim)}")


def fingerprint(W, base_vocab):
    with torch.no_grad():
        wake = W[base_vocab:].float()
        base = W[:base_vocab].float()
        h = hashlib.sha256(W.detach().cpu().to(torch.float32).numpy().tobytes()).hexdigest()[:16]
        return {
            "sha256_16": h,
            "shape": list(W.shape),
            "dtype": str(W.dtype),
            "base_norm_mean": round(base.norm(dim=1).mean().item(), 6),
            "wake_norm_mean": round(wake.norm(dim=1).mean().item(), 6),
            "wake_norm_std": round(wake.norm(dim=1).std().item(), 6),
        }


# extracting surprisal
def main():
    args = parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not tok.is_fast:
        raise RuntimeError("a fast tokenizer is required for per-word alignment")
    total_vocab = len(tok)
    print(f"  tokenizer   {args.tokenizer}")
    print(f"  vocab       {total_vocab:,}  (base {args.base_vocab:,}, "
          f"Wake {total_vocab - args.base_vocab:,})")

    # model 
    kw = dict(dtype=torch.bfloat16, device_map="auto")
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.config.use_cache = False

    if model.get_input_embeddings().weight.shape[0] != total_vocab:
        model.config.tie_word_embeddings = False
        model.resize_token_embeddings(total_vocab, mean_resizing=False)

    wte = model.get_input_embeddings()
    dim = wte.weight.shape[1]

    # the embedding
    if args.embeddings:
        kind, W = load_embeddings(args.embeddings, total_vocab, args.base_vocab, dim)
        with torch.no_grad():
            if kind == "full":
                wte.weight.copy_(W.to(wte.weight.device, dtype=wte.weight.dtype))
            else:
                wte.weight[args.base_vocab:].copy_(
                    W.to(wte.weight.device, dtype=wte.weight.dtype))
        print(f"  embeddings  {args.embeddings}  [{kind}]")
    elif args.adapter:
        raise SystemExit(
            "\nREFUSING TO RUN.\n"
            "An adapter was given with no --embeddings. PEFT writes the resized\n"
            "embedding into adapter_model.safetensors and ignores it on load, so\n"
            "this would measure the BASE embedding geometry with the adapter on\n"
            "top and report it as the checkpoint's. Pass --embeddings explicitly.\n"
            "Use --embeddings none to override deliberately (base model probe).")
    if args.embeddings == "none":
        print("  embeddings  BASE (explicitly overridden)")

    if not args.no_tie and hasattr(model, "lm_head"):
        model.lm_head.weight = wte.weight
        print("  lm_head     tied to the input embedding")

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        print(f"  adapter     {args.adapter}")

    model.eval()
    fp = fingerprint(model.get_input_embeddings().weight, args.base_vocab)
    print(f"  fingerprint {fp['sha256_16']}  wake_norm {fp['wake_norm_mean']:.4f} "
          f"+/- {fp['wake_norm_std']:.4f}  base_norm {fp['base_norm_mean']:.4f}")

    # corpus and the exact split 
    text = Path(args.text).read_text(encoding="utf-8")
    enc = tok(text, add_special_tokens=False)
    ids, word_ids = enc["input_ids"], enc.word_ids()
    L = args.seq_len
    starts = [i for i in range(0, len(ids) - L + 1, L)]
    print(f"\n  corpus      {len(ids):,} tokens -> {len(starts):,} blocks of {L}")

    if args.split == "all":
        sel = starts
    else:
        from sklearn.model_selection import train_test_split
        tr, va = train_test_split(starts, test_size=0.10, random_state=42)
        sel = va if args.split == "val" else tr
        print(f"  split       {args.split}: {len(sel):,} blocks "
              f"(sklearn test_size=0.10, random_state=42, matching Cell 5)")
    if args.limit_blocks:
        sel = sel[:args.limit_blocks]
        print(f"  LIMITED to {len(sel)} blocks (smoke test, not a real measurement)")

    # forward pass
    dev = next(model.parameters()).device
    tot_nll, tot_tok = 0.0, 0
    tok_rows, prev_H = [], None
    want_rows = bool(args.dump_tokens or args.dump_words)

    with torch.no_grad():
        for bi, s in enumerate(sel):
            block = torch.tensor(ids[s:s + L], dtype=torch.long, device=dev).unsqueeze(0)
            logits = model(input_ids=block).logits[0]        # [L, V]
            # teacher forcing: position t predicts token t+1, so the first token of every block is undefined and is dropped. This is exactly HF's shift, which is what makes the mean comparable to a logged block-wise validation loss.
            tgt = block[0, 1:]                                # [L-1]
            src = logits[:-1]                                 # [L-1, V]
            for c0 in range(0, src.shape[0], args.pos_chunk):
                c1 = min(c0 + args.pos_chunk, src.shape[0])
                lp = F.log_softmax(src[c0:c1].float(), dim=-1)
                sur = -lp.gather(1, tgt[c0:c1].unsqueeze(1)).squeeze(1)   # nats
                tot_nll += sur.sum().item()
                tot_tok += sur.numel()
                if want_rows:
                    H = -(lp.exp() * lp).sum(-1)              # predictive entropy
                    for k in range(sur.shape[0]):
                        gpos = s + 1 + c0 + k                 # global index of the TARGET
                        h = H[k].item()
                        tok_rows.append({
                            "block": bi, "pos": gpos,
                            "token_id": int(tgt[c0 + k]),
                            "token": tok.convert_ids_to_tokens(int(tgt[c0 + k])),
                            "word_id": word_ids[gpos],
                            "is_wake": int(tgt[c0 + k]) >= args.base_vocab,
                            "surprisal_nats": round(sur[k].item(), 6),
                            "entropy_nats": round(h, 6),
                            "entropy_reduction": None if prev_H is None else round(prev_H - h, 6),
                        })
                        prev_H = h
                del lp
            prev_H = None        # entropy reduction does not cross a block boundary
            if (bi + 1) % 50 == 0:
                print(f"    {bi+1}/{len(sel)} blocks, running mean "
                      f"{tot_nll/tot_tok:.4f} nats", flush=True)

    mean_nats = tot_nll / tot_tok
    print(f"  tokens scored          {tot_tok:,}")
    print(f"  MEAN per-token surprisal   {mean_nats:.6f} nats")
    print(f"                             {mean_nats/math.log(2):.6f} bits")
    print(f"  perplexity                 {math.exp(mean_nats):.4f}")

    if args.expect is not None:
        d = mean_nats - args.expect
        print(f"\n  logged value           {args.expect:.6f}")
        print(f"  difference             {d:+.6f}")
        if args.expect_is_composite:
            print("  NOTE: --expect-is-composite. A P3 run's logged eval_loss is")
            print("  LM PLUS the weighted auxiliaries, because no P3 script overrides")
            print("  prediction_step. The difference above should be approximately the")
            print("  auxiliary offset, NOT zero. Subtract the offset before comparing.")
        elif abs(d) < 0.01:
            print("  MATCHES. The loss-is-surprisal identity reproduces.")
        else:
            print("  DOES NOT MATCH. Check, in this order: --seq-len against the")
            print("  checkpoint's SEQ_LEN; --split val; whether --embeddings is the")
            print("  matrix that checkpoint was evaluated with; and whether the logged")
            print("  value is a composite (see --expect-is-composite).")

    # per-word aggregation 
    word_rows = []
    if args.dump_words:
        agg = {}
        for r in tok_rows:
            w = r["word_id"]
            if w is None:
                continue
            e = agg.setdefault(w, {"word_id": w, "surprisal_nats": 0.0,
                                   "n_subwords": 0, "any_wake": False, "blocks": set()})
            e["surprisal_nats"] += r["surprisal_nats"]
            e["n_subwords"] += 1
            e["any_wake"] |= r["is_wake"]
            e["blocks"].add(r["block"])
        # -log P(word) is the sum of its subword surprisals, Hale's unit. Words split across a block boundary are scored incompletely; so must be dropped.
        for e in agg.values():
            if len(e["blocks"]) > 1:
                continue
            e.pop("blocks")
            e["surprisal_bits"] = round(e["surprisal_nats"] / math.log(2), 6)
            e["surprisal_nats"] = round(e["surprisal_nats"], 6)
            word_rows.append(e)
        with open(args.dump_words, "w") as f:
            for e in word_rows:
                f.write(json.dumps(e) + "\n")
        print(f"\n  per-word rows          {len(word_rows):,} -> {args.dump_words}")
        print("  (words spanning a block boundary are dropped as incompletely scored)")

    if args.dump_tokens:
        with open(args.dump_tokens, "w") as f:
            for r in tok_rows:
                f.write(json.dumps(r) + "\n")
        print(f"  per-token rows         {len(tok_rows):,} -> {args.dump_tokens}")

    if tok_rows:
        wk = [r["surprisal_nats"] for r in tok_rows if r["is_wake"]]
        bs = [r["surprisal_nats"] for r in tok_rows if not r["is_wake"]]
        if wk and bs:
            print(f"\n  Wake-token surprisal   {sum(wk)/len(wk):.4f} nats  (n={len(wk):,})")
            print(f"  base-token surprisal   {sum(bs)/len(bs):.4f} nats  (n={len(bs):,})")
            print(f"  difference             {sum(wk)/len(wk) - sum(bs)/len(bs):+.4f}")

    summary = {
        "model": args.model, "tokenizer": args.tokenizer,
        "adapter": args.adapter, "embeddings": args.embeddings,
        "embedding_fingerprint": fp,
        "base_vocab": args.base_vocab, "total_vocab": total_vocab,
        "seq_len": L, "split": args.split, "blocks": len(sel),
        "tokens_scored": tot_tok,
        "mean_surprisal_nats": mean_nats,
        "mean_surprisal_bits": mean_nats / math.log(2),
        "perplexity": math.exp(mean_nats),
        "expect": args.expect, "expect_is_composite": args.expect_is_composite,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\n[SAVED] {args.out}")
    return summary


if __name__ == "__main__":
    main()
