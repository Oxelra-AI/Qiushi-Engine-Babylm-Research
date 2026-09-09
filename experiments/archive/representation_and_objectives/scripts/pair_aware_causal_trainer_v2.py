#!/usr/bin/env python3
"""research v2: Pair-aware causal GPT2 trainer with epoch-level reciprocal direction.

Design:
  Oneway  — source→side every epoch.
  Reciprocal — source→side even epochs, side→source odd epochs.
  Same total dose; only causal-prediction direction differs.

Per-pair packing: each pair occupies one seq_len sequence, padding masked from loss.
Filler: tokenized, concatenated, chunked to seq_len.
Copied-vs-noncopied loss breakdown on pair segment-B tokens.
GPT2 8×480, 16k BPE neutral tokenizer.
"""
import argparse, json, math, os, pathlib, random, sys, time, hashlib
import torch, torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

def build_model(vocab_size, seq_len, n_layer=8, n_embd=480, n_head=8, seed=43):
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=vocab_size, n_positions=seq_len,
        n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        activation_function="gelu_new", resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        bos_token_id=0, eos_token_id=0)
    return GPT2LMHeadModel(cfg)

def pre_tokenize_pairs(pair_jsonl, tokenizer):
    """Tokenize source and side separately; return list of dicts with token lists."""
    rows = []
    for line in open(pair_jsonl):
        r = json.loads(line)
        st = tokenizer.encode(r["source_text"], add_special_tokens=False)
        bt = tokenizer.encode(r["side_text"], add_special_tokens=False)
        rows.append(dict(pair_id=r["pair_id"], src_tok=st, side_tok=bt,
                         src_wc=r["source_words"], side_wc=r["side_words"],
                         topology=r["topology"]))
    return rows

def pre_tokenize_filler(filler_jsonl, tokenizer, seq_len, max_words=0):
    """Tokenize filler, concatenate, chunk to seq_len."""
    all_tok = []; tw = 0
    for line in open(filler_jsonl):
        r = json.loads(line)
        all_tok.extend(tokenizer.encode(r["text"], add_special_tokens=False))
        tw += r.get("words", len(r["text"].split()))
        if max_words > 0 and tw >= max_words:
            break
    chunks = [all_tok[i:i+seq_len] for i in range(0, len(all_tok) - seq_len, seq_len)]
    return chunks, tw

def make_pair_seq(pair, epoch, seq_len, pad_id):
    """Build one seq_len-padded sequence from a pair row for the given epoch."""
    reverse = (pair["topology"] == "reciprocal" and epoch % 2 == 1)
    if reverse:
        first, second = pair["side_tok"], pair["src_tok"]
    else:
        first, second = pair["src_tok"], pair["side_tok"]
    combined = first + second
    boundary = len(first)
    if len(combined) > seq_len:
        combined = combined[:seq_len]
        boundary = min(boundary, seq_len)
    real = len(combined)
    tokens = combined + [pad_id] * (seq_len - real)
    # Copied mask: segment-B tokens that also appear in segment-A
    a_set = set(first)
    copied = [0] * seq_len
    for i in range(boundary, real):
        copied[i] = 1 if tokens[i] in a_set else 0
    return dict(tokens=tokens, real=real, boundary=boundary, copied=copied, is_pair=True)

def make_filler_seq(chunk, seq_len):
    return dict(tokens=chunk, real=len(chunk), boundary=-1, copied=[0]*seq_len, is_pair=False)

def loss_with_breakdown(logits, tok_t, reals, copieds, bounds, device):
    B, T, V = logits.shape
    pred = logits[:, :-1].contiguous().view(-1, V)
    targ = tok_t[:, 1:].contiguous().view(-1)
    per = F.cross_entropy(pred, targ, reduction='none').view(B, T-1)
    T2 = T - 1
    mask = torch.zeros(B, T2, device=device)
    for i in range(B):
        mask[i, :min(reals[i]-1, T2)] = 1.0
    ml = per * mask
    total = ml.sum() / mask.sum().clamp(min=1)
    # segment-B mask
    cm = torch.tensor([c[1:T2+1] for c in copieds], device=device, dtype=torch.float)
    sb = torch.zeros(B, T2, device=device)
    for i in range(B):
        if bounds[i] >= 0:
            s = max(bounds[i]-1, 0); e = min(reals[i]-1, T2)
            sb[i, s:e] = 1.0
    cl = (ml * sb * cm).sum(); cc = (mask * sb * cm).sum().clamp(min=1)
    nl = (ml * sb * (1-cm)).sum(); nc = (mask * sb * (1-cm)).sum().clamp(min=1)
    return total, dict(loss=total.item(),
        copied_loss=(cl/cc).item(), noncopied_loss=(nl/nc).item(),
        n_real=mask.sum().item(), n_copied=cc.item(), n_noncopied=nc.item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_pair_jsonl", required=True)
    ap.add_argument("--filler_jsonl", required=True)
    ap.add_argument("--tokenizer_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_epochs", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--model_seed", type=int, default=43)
    ap.add_argument("--train_seed", type=int, default=43022)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max_filler_words", type=int, default=0)
    ap.add_argument("--hf_save_epochs", type=str, default="1,2",
                    help="Comma-separated epoch numbers at which to save HF model for eval")
    args = ap.parse_args()

    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    print(f"Tokenizer: vocab={tok.vocab_size} pad={pad_id}", flush=True)

    t0 = time.time()
    model = build_model(tok.vocab_size, args.seq_len, seed=args.model_seed).to(args.device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Model: GPT2 8×480 {n_par:,} params on {args.device}", flush=True)

    print("Tokenizing pairs...", flush=True)
    pairs = pre_tokenize_pairs(args.arm_pair_jsonl, tok)
    print(f"  {len(pairs)} pairs", flush=True)

    print("Tokenizing filler...", flush=True)
    filler_chunks, fw = pre_tokenize_filler(
        args.filler_jsonl, tok, args.seq_len, args.max_filler_words)
    print(f"  {len(filler_chunks)} filler seqs ({fw:,} words) in {time.time()-t0:.1f}s", flush=True)

    eff = args.batch_size * args.grad_accum
    seqs_per_epoch = len(pairs) + len(filler_chunks)
    total_steps = args.n_epochs * math.ceil(seqs_per_epoch / eff)
    warmup = int(total_steps * args.warmup_frac)
    hf_epochs = set(int(x) for x in args.hf_save_epochs.split(",") if x.strip())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def lr_fn(step):
        if step < warmup: return step / max(warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total_steps - warmup, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    cfg = dict(arm=args.arm_pair_jsonl, filler=args.filler_jsonl, tokenizer=args.tokenizer_dir,
               epochs=args.n_epochs, total_steps=total_steps, warmup=warmup, eff_batch=eff,
               pairs=len(pairs), filler_seqs=len(filler_chunks), model_seed=args.model_seed,
               train_seed=args.train_seed, n_params=n_par, device=args.device)
    (out / "train_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Training: {total_steps} steps, warmup={warmup}, eff_batch={eff}", flush=True)

    log_f = open(out / "training_log.jsonl", "w")
    rng = random.Random(args.train_seed)
    gstep = 0; acc = 0; acc_loss = 0.0

    model.train()
    for epoch in range(args.n_epochs):
        # Build epoch sequences
        ep_seqs = [make_pair_seq(p, epoch, args.seq_len, pad_id) for p in pairs]
        ep_seqs += [make_filler_seq(c, args.seq_len) for c in filler_chunks]
        rng.shuffle(ep_seqs)
        ep_words = sum(s["real"]/1.3 for s in ep_seqs)
        print(f"\nEpoch {epoch}: {len(ep_seqs)} seqs, ~{ep_words/1e6:.1f}M words", flush=True)

        for bi in range(0, len(ep_seqs), args.batch_size):
            batch = ep_seqs[bi:bi+args.batch_size]
            tt = torch.tensor([s["tokens"] for s in batch], device=args.device)
            reals = [s["real"] for s in batch]
            copieds = [s["copied"] for s in batch]
            bounds = [s["boundary"] for s in batch]

            loss, bd = loss_with_breakdown(model(tt).logits, tt, reals, copieds, bounds, args.device)
            (loss / args.grad_accum).backward()
            acc_loss += loss.item(); acc += 1

            if acc % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                gstep += 1
                if gstep % 100 == 0 or gstep <= 3:
                    entry = dict(step=gstep, epoch=epoch, lr=sched.get_last_lr()[0],
                                 loss=acc_loss/args.grad_accum, **bd)
                    log_f.write(json.dumps(entry)+"\n"); log_f.flush()
                    if gstep % 500 == 0:
                        print(f"  step {gstep}/{total_steps} loss={entry['loss']:.4f} "
                              f"cp={bd['copied_loss']:.4f} nc={bd['noncopied_loss']:.4f} "
                              f"lr={entry['lr']:.6f} {time.time()-t0:.0f}s", flush=True)
                    acc_loss = 0.0

        # End-of-epoch save
        ep_num = epoch + 1
        if ep_num in hf_epochs:
            hf = out / "hf_model" / f"epoch_{ep_num}"
            hf.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(hf); tok.save_pretrained(hf)
            print(f"  HF model saved: {hf}", flush=True)

    # Final save
    fin = out / "hf_model" / "final"
    fin.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(fin); tok.save_pretrained(fin)
    elapsed = time.time() - t0
    summary = dict(status="CAUSAL_TRAINING_COMPLETE", steps=gstep,
                   epochs=args.n_epochs, elapsed_sec=elapsed, final=str(fin))
    (out / "training_manifest.json").write_text(json.dumps(summary, indent=2))
    log_f.close()
    print(f"\nDone: {gstep} steps in {elapsed:.1f}s\n{json.dumps(summary, indent=2)}", flush=True)

if __name__ == "__main__":
    main()
