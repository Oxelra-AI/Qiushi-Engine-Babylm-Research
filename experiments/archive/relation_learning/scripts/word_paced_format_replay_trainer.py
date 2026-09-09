#!/usr/bin/env python3
"""research: word-paced format-alignment replay trainer.

Matches coherent86 private-phase parameters:
  - Same 3,992,800 words from same BabyLM Strict-Small suffix
  - Same ~101 macro-updates of ~39,533 words each (gradient accumulation)
  - Same LR schedule (455 total steps, cosine decay, warmup)
  - Fresh private adapter from chck_82M, frozen slow path
  - Same objective: WWM CE + private-on-vs-off neutral KL

Key changes from research:
  - Dynamic padding (not fixed 256 tokens) → efficient for short isolated rows
  - Word-paced gradient accumulation → matched words/update despite variable row count
  - Held-coherent neutral readout → cross-arm drift comparison
  - add_special_tokens=True so isolated sentences match BLiMP eval format
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, os, sys, time, math, argparse, pathlib, random, hashlib
from collections import defaultdict

_SCRIPT = _public_path('experiments/archive/relation_learning/scripts/word_paced_format_replay_trainer.py')
_ROOT = _public_path('.')

# Cache dirs set dynamically in train() after out_dir is known
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(_public_path('experiments/archive/frontier_consolidation/scripts')))
from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CHCK82 = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')

# Coherent86 private-phase reference parameters
COHERENT86_WORDS = 3_992_800
COHERENT86_UPDATES = 101
WORDS_PER_UPDATE = COHERENT86_WORDS // COHERENT86_UPDATES  # ~39,533
LR_TOTAL_STEPS = 455
WARMUP_FRACTION = 0.06


def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(_ROOT))
    except: return str(p)


# ── Model loading ──────────────────────────────────────────────────────────────
def load_frozen_model(chck_path, private_scale, device):
    from transformers import DebertaV2Config, AutoTokenizer
    from safetensors.torch import load_file

    tokenizer = AutoTokenizer.from_pretrained(str(chck_path), use_fast=True, local_files_only=True)
    config = DebertaV2Config.from_pretrained(str(chck_path))
    config.private_adapter_bottleneck = getattr(config, "adapter_bottleneck", 128)
    config.private_adapter_scale = private_scale
    config.private_adapter_enabled = True
    config.private_adapter_activation = getattr(config, "adapter_activation", "gelu")

    model = FrozenSlowPrivateDebertaV2ForMaskedLM(config)
    sf = chck_path / "model.safetensors"
    state_dict = load_file(str(sf), device="cpu")
    model.load_state_dict(state_dict, strict=False)

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = 0
    for name, p in model.named_parameters():
        if "private_adapter" in name:
            p.requires_grad_(True)
            trainable += p.numel()
    model.to(device)
    total = sum(p.numel() for p in model.parameters())
    return model, tokenizer, total, trainable


def set_private_enabled(model, enabled):
    if hasattr(model, "config"):
        model.config.private_adapter_enabled = enabled


# ── Data ───────────────────────────────────────────────────────────────────────
def load_jsonl_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({"text": obj["text"], "words": int(obj["words"]),
                         "source": obj.get("source", "")})
    return rows


def tokenize_row(text, tokenizer, max_length=256):
    enc = tokenizer(text, truncation=True, max_length=max_length,
                    add_special_tokens=True, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    # Build word groups: consecutive tokens sharing same whitespace word get same group ID
    word_group = [-1] * len(ids)
    g = 0
    prev_end = -1
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:  # special token
            word_group[i] = -1
            continue
        if s > prev_end and prev_end >= 0:  # new word boundary
            g += 1
        word_group[i] = g
        prev_end = e
    return ids, word_group


def collate_dynamic(batch, pad_id):
    mx = max(len(b["ids"]) for b in batch)
    n = len(batch)
    ids = torch.full((n, mx), pad_id, dtype=torch.long)
    att = torch.zeros((n, mx), dtype=torch.long)
    wg = torch.full((n, mx), -1, dtype=torch.long)
    words = torch.zeros(n, dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["ids"])
        ids[i, :L] = torch.tensor(b["ids"], dtype=torch.long)
        att[i, :L] = 1
        wg[i, :L] = torch.tensor(b["wg"], dtype=torch.long)
        words[i] = b["words"]
    return {"input_ids": ids, "attention_mask": att, "word_group": wg, "words": words}


# ── WWM masking ────────────────────────────────────────────────────────────────
def apply_wwm(input_ids, attention_mask, word_group, mask_id, mask_prob, vocab_size):
    """Whole-word masking: select ~mask_prob of word groups, mask all their tokens."""
    bsz, seq = input_ids.shape
    masked = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    for i in range(bsz):
        groups = set()
        for j in range(seq):
            g = int(word_group[i, j].item())
            if g >= 0 and int(attention_mask[i, j].item()) == 1:
                groups.add(g)
        if not groups:
            continue
        groups = sorted(groups)
        n_mask = max(1, int(len(groups) * mask_prob))
        selected = set(random.sample(groups, min(n_mask, len(groups))))
        for j in range(seq):
            g = int(word_group[i, j].item())
            if g in selected:
                labels[i, j] = input_ids[i, j]
                r = random.random()
                if r < 0.8:
                    masked[i, j] = mask_id
                elif r < 0.9:
                    masked[i, j] = random.randint(0, vocab_size - 1)
                # else: keep original
    return masked, labels


# ── LR schedule ────────────────────────────────────────────────────────────────
def lr_at_update(step, total_steps, warmup_steps, peak_lr):
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Neutral loss ───────────────────────────────────────────────────────────────
def compute_neutral_kl(model, input_ids, attention_mask, device, subsample=64):
    """KL(private_off || private_on) on given batch."""
    n = min(subsample, input_ids.size(0))
    ids = input_ids[:n].to(device)
    att = attention_mask[:n].to(device)
    set_private_enabled(model, False)
    with torch.no_grad():
        off_logits = model(input_ids=ids, attention_mask=att).logits
    set_private_enabled(model, True)
    on_logits = model(input_ids=ids, attention_mask=att).logits
    # Compute KL on valid positions
    valid = att[:n].bool()
    off_lp = F.log_softmax(off_logits[valid], dim=-1).detach()
    on_lp = F.log_softmax(on_logits[valid], dim=-1)
    kl = F.kl_div(on_lp, off_lp, log_target=True, reduction="batchmean")
    return kl, float(kl.detach().cpu())


# ── Main training loop ─────────────────────────────────────────────────────────
def train(args):
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = out_dir / "hf_cache"
    (hf_cache / "modules").mkdir(parents=True, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_MODULES_CACHE"] = str(hf_cache / "modules")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(json.dumps({"event": "start", "stream": rel(args.stream),
                       "arm": args.arm_label, "device": str(device)}), flush=True)

    # Load model with controlled seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, tokenizer, total_params, trainable_params = load_frozen_model(
        pathlib.Path(args.endpoint), args.private_scale, device)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    print(json.dumps({"event": "model_loaded", "total_params": total_params,
                       "trainable_params": trainable_params,
                       "private_scale": args.private_scale}), flush=True)

    # Load and tokenize training rows
    rows = load_jsonl_rows(args.stream)
    total_words = sum(r["words"] for r in rows)
    print(json.dumps({"event": "data_loaded", "rows": len(rows), "total_words": total_words}), flush=True)

    tokenized = []
    for r in rows:
        ids, wg = tokenize_row(r["text"], tokenizer, max_length=args.max_length)
        tokenized.append({"ids": ids, "wg": wg, "words": r["words"]})

    # Load held-coherent neutral batch (fixed 256 coherent rows from tail)
    held_coherent = None
    if args.held_coherent_stream:
        held_rows = load_jsonl_rows(args.held_coherent_stream)[:args.held_neutral_rows]
        hc_tok = [tokenize_row(r["text"], tokenizer, max_length=256) for r in held_rows]
        # Pad to 256
        hc_ids = torch.full((len(hc_tok), 256), tokenizer.pad_token_id, dtype=torch.long)
        hc_att = torch.zeros((len(hc_tok), 256), dtype=torch.long)
        for i, (ids, _) in enumerate(hc_tok):
            L = min(len(ids), 256)
            hc_ids[i, :L] = torch.tensor(ids[:L], dtype=torch.long)
            hc_att[i, :L] = 1
        held_coherent = (hc_ids, hc_att)
        print(json.dumps({"event": "held_coherent_loaded", "rows": len(hc_tok)}), flush=True)

    # Optimizer
    private_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(private_params, lr=args.lr, betas=(0.9, 0.98),
                                   eps=1e-6, weight_decay=args.weight_decay)

    # Schedule
    warmup_steps = max(1, int(LR_TOTAL_STEPS * WARMUP_FRACTION))
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    # Word-paced macro-update batching
    # Organize rows into macro-updates of ~WORDS_PER_UPDATE words each
    macro_batches = []
    current_batch = []
    current_words = 0
    for t in tokenized:
        current_batch.append(t)
        current_words += t["words"]
        if current_words >= WORDS_PER_UPDATE:
            macro_batches.append(current_batch)
            current_batch = []
            current_words = 0
    if current_batch:
        macro_batches.append(current_batch)

    n_updates = len(macro_batches)
    print(json.dumps({"event": "batching", "macro_updates": n_updates,
                       "target_words_per_update": WORDS_PER_UPDATE,
                       "actual_mean_words_per_update": total_words / max(1, n_updates),
                       "lr_total_steps": LR_TOTAL_STEPS, "warmup": warmup_steps}), flush=True)

    config = {
        "status": "FORMAT_REPLAY_CONFIG",
        "arm_label": args.arm_label,
        "stream": rel(args.stream),
        "endpoint": rel(args.endpoint),
        "total_words": total_words,
        "macro_updates": n_updates,
        "words_per_update_target": WORDS_PER_UPDATE,
        "lr": args.lr,
        "lr_total_steps": LR_TOTAL_STEPS,
        "warmup_steps": warmup_steps,
        "private_scale": args.private_scale,
        "alpha_endpoint": args.alpha_endpoint,
        "neutral_lambda": args.neutral_lambda,
        "mask_prob": args.mask_prob,
        "micro_batch_size": args.micro_batch_size,
        "max_length": args.max_length,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Training
    random.seed(args.seed)
    model.train()
    log_path = out_dir / "training_log.jsonl"
    cum_words = 0

    with log_path.open("w", encoding="utf-8") as logf:
        for update_idx, macro_batch in enumerate(macro_batches):
            lr = lr_at_update(update_idx, LR_TOTAL_STEPS, warmup_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

            macro_words = sum(t["words"] for t in macro_batch)
            n_micro = math.ceil(len(macro_batch) / args.micro_batch_size)
            total_targets = 0
            total_ce_sum = 0.0

            # Micro-batch gradient accumulation
            for mi in range(n_micro):
                start = mi * args.micro_batch_size
                end = min(start + args.micro_batch_size, len(macro_batch))
                micro = macro_batch[start:end]
                batch = collate_dynamic(micro, pad_id)
                ids = batch["input_ids"].to(device, non_blocking=True)
                att = batch["attention_mask"].to(device, non_blocking=True)
                wg = batch["word_group"].to(device, non_blocking=True)

                masked, labels = apply_wwm(ids, att, wg, mask_id, args.mask_prob, vocab_size)
                n_targets = int((labels != -100).sum().item())
                if n_targets == 0:
                    continue

                set_private_enabled(model, True)
                model.train()
                out = model(input_ids=masked, attention_mask=att)
                vocab = out.logits.shape[-1]
                ce_sum = F.cross_entropy(out.logits.reshape(-1, vocab),
                                         labels.reshape(-1),
                                         ignore_index=-100, reduction="sum")
                # Scale: divide by total expected targets in macro-update to match research per-token avg
                scaled_ce = (args.main_lambda * ce_sum) / max(1, n_targets * n_micro)
                scaled_ce.backward()
                total_targets += n_targets
                total_ce_sum += float(ce_sum.detach().cpu())
                del out, ce_sum, scaled_ce

            # Neutral loss on held-coherent batch (once per macro-update)
            neutral_val = 0.0
            if held_coherent is not None and args.neutral_lambda > 0:
                neutral_kl, neutral_val = compute_neutral_kl(
                    model, held_coherent[0], held_coherent[1], device,
                    subsample=min(64, held_coherent[0].size(0)))
                (args.neutral_lambda * neutral_kl).backward()
                del neutral_kl

            # Clip and step
            torch.nn.utils.clip_grad_norm_(private_params, 1.0)
            optimizer.step()

            cum_words += macro_words
            mean_ce = total_ce_sum / max(1, total_targets)

            rec = {
                "update": update_idx + 1,
                "lr": lr,
                "macro_words": macro_words,
                "cum_words": cum_words,
                "micro_batches": n_micro,
                "macro_rows": len(macro_batch),
                "targets": total_targets,
                "main_ce": round(mean_ce, 6),
                "held_neutral_kl": round(neutral_val, 6),
                "elapsed_sec": round(time.time() - t0, 1),
            }
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

            if (update_idx + 1) % 10 == 0 or update_idx == 0:
                print(json.dumps({"event": "progress", "update": update_idx + 1,
                                   "of": n_updates, "cum_words": cum_words,
                                   "main_ce": round(mean_ce, 6),
                                   "held_neutral_kl": round(neutral_val, 6),
                                   "lr": round(lr, 8),
                                   "elapsed_sec": round(time.time() - t0, 1)}), flush=True)

    # Save checkpoint
    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    print(json.dumps({"event": "checkpoint_saved", "path": rel(ckpt_dir),
                       "updates": n_updates, "cum_words": cum_words}), flush=True)

    # Materialize alpha endpoint
    if args.alpha_endpoint != args.private_scale:
        alpha_dir = out_dir / f"alpha_{args.alpha_endpoint:.2f}"
        alpha_dir.mkdir(parents=True, exist_ok=True)
        model.config.private_adapter_scale = args.alpha_endpoint
        model.save_pretrained(str(alpha_dir))
        tokenizer.save_pretrained(str(alpha_dir))
        print(json.dumps({"event": "alpha_endpoint", "alpha": args.alpha_endpoint,
                           "path": rel(alpha_dir)}), flush=True)

    elapsed = round(time.time() - t0, 1)
    summary = {
        "status": "FORMAT_REPLAY_DONE",
        "arm_label": args.arm_label,
        "updates": n_updates,
        "total_words": cum_words,
        "final_main_ce": round(mean_ce, 6) if 'mean_ce' in dir() else None,
        "final_held_neutral_kl": round(neutral_val, 6),
        "elapsed_sec": elapsed,
        "checkpoint": rel(ckpt_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True, help="JSONL stream (isolated or half)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm-label", default="isolated_all")
    ap.add_argument("--endpoint", default=str(CHCK82))
    ap.add_argument("--held-coherent-stream", default="",
                    help="JSONL of coherent rows for fixed neutral readout")
    ap.add_argument("--held-neutral-rows", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--private-scale", type=float, default=0.2)
    ap.add_argument("--alpha-endpoint", type=float, default=0.75)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--main-lambda", type=float, default=1.0)
    ap.add_argument("--neutral-lambda", type=float, default=0.15)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--micro-batch-size", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=97097)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
