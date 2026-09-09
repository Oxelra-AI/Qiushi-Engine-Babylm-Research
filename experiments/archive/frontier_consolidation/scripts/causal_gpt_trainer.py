#!/usr/bin/env python3
"""research: Causal GPT2 trainer for architecture-transfer test.

Trains GPT2LMHeadModel on order-balanced causal pools. Dense 2M-word checkpoints
for trajectory comparison with DeBERTa scale1.75.

Usage:
  python causal_gpt_trainer.py \
    --pool data/causal_transfer_scaffold/causal_compact_10M.jsonl \
    --tokenizer data/causal_transfer_scaffold/neutral_tokenizer \
    --run-dir training/runs/causal_compact_seed43022 \
    --gpu 0 --seed 43022 --total-words 100000000
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, hashlib, math, os, pathlib, random, sys, time
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = _public_path('experiments/archive/frontier_consolidation')  # Study root.
# fallback if run from user root
if not (_public_path('experiments/archive/frontier_consolidation')).exists():
    ROOT = pathlib.Path("experiments/archive/frontier_consolidation")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def wc_text(text):
    return len(text.split())


class ChunkedCausalDataset(Dataset):
    """Tokenize all docs, concatenate with EOS, split into seq_len chunks."""

    def __init__(self, jsonl_path, tokenizer, seq_len=256, seed=42):
        self.seq_len = seq_len
        self.seed = seed
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            eos_id = tokenizer.convert_tokens_to_ids("</s>")

        self.doc_ids = []
        self.total_words = 0
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                ids = tokenizer.encode(r["text"])
                self.doc_ids.append(ids + [eos_id])
                self.total_words += r.get("words", wc_text(r["text"]))

        self.total_raw_tokens = sum(len(d) for d in self.doc_ids)
        self.words_per_token = self.total_words / self.total_raw_tokens
        print(f"  Loaded {len(self.doc_ids)} docs, {self.total_words:,} words, "
              f"{self.total_raw_tokens:,} raw tokens, ratio {self.words_per_token:.4f}",
              flush=True)
        self._build_chunks(list(range(len(self.doc_ids))))

    def _build_chunks(self, order):
        all_ids = []
        for i in order:
            all_ids.extend(self.doc_ids[i])
        n = (len(all_ids) // self.seq_len) * self.seq_len
        self.chunks = torch.tensor(all_ids[:n], dtype=torch.long).view(-1, self.seq_len)
        self.active_tokens = n

    def reshuffle(self, epoch):
        rng = random.Random(self.seed + epoch)
        order = list(range(len(self.doc_ids)))
        rng.shuffle(order)
        self._build_chunks(order)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def save_checkpoint(model, tokenizer, config, run_dir, name, words, step, loss):
    ckdir = run_dir / "hf_model" / name
    ckdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckdir))
    tokenizer.save_pretrained(str(ckdir))
    meta = {"name": name, "words": int(words), "step": step, "loss": round(loss, 6),
            "params": sum(p.numel() for p in model.parameters())}
    sf = ckdir / "model.safetensors"
    if sf.exists():
        meta["model_sha256"] = sha256_file(sf)
    (ckdir / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  [CHECKPOINT] {name}: words={int(words):,} step={step} loss={loss:.4f}"
          f" sha={meta.get('model_sha256','?')[:12]}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="JSONL pool path")
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer directory")
    ap.add_argument("--run-dir", required=True, help="Output run directory")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=43022)
    ap.add_argument("--total-words", type=int, default=100_000_000)
    ap.add_argument("--checkpoint-interval", type=int, default=2_000_000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=480)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true", help="CPU smoke: 2 steps only")
    args = ap.parse_args()

    t0 = time.time()
    run_dir = pathlib.Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ──────────────────────────────────────────────────────────
    if args.smoke:
        device = torch.device("cpu")
        print("SMOKE MODE: CPU, 2 steps", flush=True)
    else:
        assert torch.cuda.is_available(), "CUDA required for real training"
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
        print(f"Device: {device} ({torch.cuda.get_device_name(args.gpu)})", flush=True)

    # ── Tokenizer ───────────────────────────────────────────────────────
    os.environ["HF_HOME"] = str(run_dir / ".hf_cache")
    os.environ["TRANSFORMERS_CACHE"] = str(run_dir / ".hf_cache")
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
    from transformers import get_cosine_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tokenizer)
    print(f"Tokenizer: vocab={vocab_size}, bos={tokenizer.bos_token_id}, "
          f"eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}", flush=True)

    # ── Data ────────────────────────────────────────────────────────────
    print("Loading data...", flush=True)
    dataset = ChunkedCausalDataset(args.pool, tokenizer, args.seq_len, args.seed)

    words_per_epoch = dataset.total_words  # exactly 10M legal charged words per epoch
    total_epochs = args.total_words // words_per_epoch
    # Charge the legal 10M words across active chunks in the epoch.  The
    # concatenated token stream drops only the final <seq_len token tail; using
    # raw-token words_per_token would make the endpoint slightly below 100M and
    # omit chck_100M.  This counter is the legal exposure coordinate; the
    # manifest also records raw tokens so the small effective-token difference
    # between arms remains explicit.
    charged_words_per_active_token = words_per_epoch / dataset.active_tokens
    raw_words_per_token = dataset.words_per_token

    if args.smoke:
        total_epochs = 1
        args.total_words = words_per_epoch
        args.checkpoint_interval = words_per_epoch  # one checkpoint

    # Estimate total steps after any smoke adjustment.
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    total_steps = total_epochs * steps_per_epoch
    print(f"Training plan: {total_epochs} epochs, {steps_per_epoch} steps/epoch, "
          f"{total_steps} total steps, {args.total_words/1e6:.0f}M legal charged words", flush=True)

    # ── Model ───────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=args.hidden,
        n_layer=args.layers,
        n_head=args.heads,
        n_positions=args.seq_len,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: GPT2 {args.layers}×{args.hidden}, {n_params:,} params ({n_train:,} trainable)",
          flush=True)

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.999))
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                 num_warmup_steps=warmup_steps,
                                                 num_training_steps=total_steps)

    # ── Training loop ───────────────────────────────────────────────────
    cumulative_words = 0.0
    next_checkpoint = float(args.checkpoint_interval)
    global_step = 0
    all_checkpoints = []
    losses_log = []

    for epoch in range(total_epochs):
        if epoch > 0:
            dataset.reshuffle(epoch)

        g = torch.Generator()
        g.manual_seed(args.seed + epoch + 1000)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            generator=g, drop_last=False, num_workers=0)

        model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(loader):
            if args.smoke and batch_idx >= 2:
                break

            input_ids = batch.to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_steps += 1
            step_loss = loss.item()
            epoch_loss_sum += step_loss

            # Track legal charged words over active chunks.  Effective token
            # exposure remains separately recorded in the manifest/audit.
            batch_tokens = input_ids.numel()
            batch_words = batch_tokens * charged_words_per_active_token
            cumulative_words += batch_words

            # Periodic logging
            if global_step % 50 == 0 or (args.smoke and batch_idx < 3):
                avg = epoch_loss_sum / epoch_steps
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {global_step:>5}/{total_steps} | epoch {epoch+1}/{total_epochs} | "
                      f"loss {step_loss:.4f} (avg {avg:.4f}) | lr {lr_now:.2e} | "
                      f"words {cumulative_words/1e6:.2f}M", flush=True)

            losses_log.append({"step": global_step, "loss": round(step_loss, 6),
                               "words": round(cumulative_words), "epoch": epoch + 1})

            # Checkpoint
            while cumulative_words >= next_checkpoint and next_checkpoint <= args.total_words:
                avg = epoch_loss_sum / max(epoch_steps, 1)
                meta = save_checkpoint(model, tokenizer, config, run_dir,
                                        f"chck_{int(next_checkpoint/1e6)}M",
                                        cumulative_words, global_step, avg)
                all_checkpoints.append(meta)
                next_checkpoint += args.checkpoint_interval

        # Snap legal exposure at epoch boundary to the exact word budget and
        # catch any checkpoint missed by floating-point accumulation at the
        # final batch (especially chck_10M, ..., chck_100M).
        cumulative_words = float((epoch + 1) * words_per_epoch)
        while cumulative_words >= next_checkpoint and next_checkpoint <= args.total_words:
            avg = epoch_loss_sum / max(epoch_steps, 1)
            meta = save_checkpoint(model, tokenizer, config, run_dir,
                                    f"chck_{int(next_checkpoint/1e6)}M",
                                    cumulative_words, global_step, avg)
            all_checkpoints.append(meta)
            next_checkpoint += args.checkpoint_interval

        epoch_avg = epoch_loss_sum / max(epoch_steps, 1)
        print(f"Epoch {epoch+1}/{total_epochs} done: {epoch_steps} steps, "
              f"avg loss {epoch_avg:.4f}, legal words {cumulative_words/1e6:.2f}M", flush=True)

    # ── Final checkpoint ────────────────────────────────────────────────
    final_avg = epoch_loss_sum / max(epoch_steps, 1) if epoch_steps > 0 else 0
    meta = save_checkpoint(model, tokenizer, config, run_dir, "final",
                           cumulative_words, global_step, final_avg)
    all_checkpoints.append(meta)

    # ── Manifest ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    manifest = {
        "status": "CAUSAL_GPT_TRAINING_COMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(elapsed, 1),
        "args": vars(args),
        "model": {
            "type": "GPT2LMHeadModel",
            "layers": args.layers,
            "hidden": args.hidden,
            "heads": args.heads,
            "vocab_size": vocab_size,
            "params": n_params,
            "trainable": n_train,
        },
        "training": {
            "total_words": round(cumulative_words),
            "legal_charged_words": round(cumulative_words),
            "total_steps": global_step,
            "total_epochs": total_epochs,
            "raw_words_per_token_including_eos_tail": round(raw_words_per_token, 9),
            "charged_words_per_active_token": round(charged_words_per_active_token, 9),
            "raw_tokens_per_epoch_with_eos": dataset.total_raw_tokens,
            "active_tokens_per_epoch": dataset.active_tokens,
            "dropped_tail_tokens_per_epoch": dataset.total_raw_tokens - dataset.active_tokens,
            "final_loss": round(final_avg, 6),
        },
        "pool_sha256": sha256_file(args.pool),
        "tokenizer_sha256": sha256_file(pathlib.Path(args.tokenizer) / "tokenizer.json"),
        "checkpoints": all_checkpoints,
    }

    manifest_path = run_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    losses_path = run_dir / "losses.jsonl"
    with open(losses_path, "w") as f:
        for rec in losses_log:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Words: {cumulative_words/1e6:.2f}M, Steps: {global_step}, "
          f"Final loss: {final_avg:.4f}")
    print(f"Checkpoints: {len(all_checkpoints)}")
    print(f"Manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
