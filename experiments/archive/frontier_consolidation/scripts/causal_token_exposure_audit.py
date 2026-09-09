#!/usr/bin/env python3
"""research: exact token/effective-exposure audit for causal GPT transfer arms.

The legal BabyLM budget is word exposure: both causal arms are exactly 10M words
per epoch and at most 10 epochs.  A decoder-only implementation nevertheless
sees a slightly different number of subword tokens/sequences because the compact
semantic view and first-N repeat strings tokenize differently.  This script
computes the exact tokenizer-side quantities used by the research trainer so the
compact-vs-repeat result can be interpreted with that small effective-token
exposure difference explicit rather than hidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import time
from typing import Any


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len(text.split())


def quantiles(xs: list[float], qs: list[float]) -> dict[str, float]:
    if not xs:
        return {str(q): float("nan") for q in qs}
    ys = sorted(xs)
    out: dict[str, float] = {}
    for q in qs:
        pos = q * (len(ys) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            val = ys[lo]
        else:
            val = ys[lo] * (hi - pos) + ys[hi] * (pos - lo)
        out[str(q)] = float(val)
    return out


def audit_pool(path: pathlib.Path, tokenizer, seq_len: int, batch_sizes: list[int], epochs: int, checkpoint_interval: int) -> dict[str, Any]:
    rows = 0
    total_words = 0
    raw_tokens = 0
    per_source: dict[str, dict[str, int]] = {}
    row_tok_per_word: list[float] = []
    row_words: list[int] = []
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.convert_tokens_to_ids("</s>")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = r["text"]
            words = int(r.get("words", wc(text)))
            toks = len(tokenizer.encode(text)) + 1  # research trainer appends EOS per row.
            src = str(r.get("source", ""))
            rows += 1
            total_words += words
            raw_tokens += toks
            per_source.setdefault(src, {"rows": 0, "words": 0, "tokens_with_eos": 0})
            per_source[src]["rows"] += 1
            per_source[src]["words"] += words
            per_source[src]["tokens_with_eos"] += toks
            if words > 0:
                row_tok_per_word.append(toks / words)
            row_words.append(words)

    active_tokens = (raw_tokens // seq_len) * seq_len
    dropped_tokens = raw_tokens - active_tokens
    chunks_per_epoch = active_tokens // seq_len
    words_per_token = total_words / raw_tokens
    token_per_word = raw_tokens / total_words
    effective_active_words_per_epoch = active_tokens * words_per_token

    batch_plans: dict[str, Any] = {}
    for bsz in batch_sizes:
        full_batches = chunks_per_epoch // bsz
        last_chunks = chunks_per_epoch % bsz
        steps_per_epoch = full_batches + (1 if last_chunks else 0)
        tokens_per_full_batch = bsz * seq_len
        tokens_last_batch = last_chunks * seq_len if last_chunks else tokens_per_full_batch

        # Deterministic checkpoint-crossing schedule matching the trainer's
        # proportional active-token word counter.
        checkpoints = []
        cumulative = 0.0
        next_ckpt = float(checkpoint_interval)
        step = 0
        for epoch in range(epochs):
            for bi in range(steps_per_epoch):
                step += 1
                if bi < full_batches:
                    batch_tokens = tokens_per_full_batch
                else:
                    batch_tokens = tokens_last_batch
                cumulative += batch_tokens * words_per_token
                while cumulative >= next_ckpt and next_ckpt <= epochs * total_words:
                    checkpoints.append({
                        "name": f"chck_{int(next_ckpt / 1_000_000)}M",
                        "target_words": int(next_ckpt),
                        "saved_after_step": step,
                        "trainer_effective_words_at_save": round(cumulative),
                        "epoch": epoch + 1,
                    })
                    next_ckpt += checkpoint_interval
        batch_plans[str(bsz)] = {
            "batch_size": bsz,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": steps_per_epoch * epochs,
            "full_batches_per_epoch": full_batches,
            "last_batch_chunks": last_chunks,
            "last_batch_tokens": tokens_last_batch,
            "trainer_effective_words_after_epochs": round(cumulative),
            "legal_charged_words_after_epochs": epochs * total_words,
            "checkpoint_count": len(checkpoints),
            "first_five_checkpoints": checkpoints[:5],
            "last_five_checkpoints": checkpoints[-5:],
        }

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "words": total_words,
        "raw_tokens_with_eos": raw_tokens,
        "token_per_word": token_per_word,
        "words_per_token": words_per_token,
        "active_tokens_per_epoch": active_tokens,
        "dropped_tail_tokens_per_epoch": dropped_tokens,
        "chunks_per_epoch": chunks_per_epoch,
        "effective_active_words_per_epoch": effective_active_words_per_epoch,
        "legal_charged_words_per_epoch": total_words,
        "active_word_fraction_by_token_tail": effective_active_words_per_epoch / total_words,
        "sources": per_source,
        "row_words_summary": {
            "mean": statistics.fmean(row_words) if row_words else None,
            "min": min(row_words) if row_words else None,
            "max": max(row_words) if row_words else None,
        },
        "row_tokens_per_word_summary": {
            "mean": statistics.fmean(row_tok_per_word) if row_tok_per_word else None,
            "quantiles": quantiles(row_tok_per_word, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]),
        },
        "batch_plans": batch_plans,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", default="experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/causal_compact_10M.jsonl")
    ap.add_argument("--repeat", default="experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/causal_repeat_10M.jsonl")
    ap.add_argument("--tokenizer", default="experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/neutral_tokenizer")
    ap.add_argument("--out-dir", default="experiments/archive/frontier_consolidation/data/causal_token_exposure_audit")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--checkpoint-interval", type=int, default=2_000_000)
    ap.add_argument("--batch-sizes", default="128,256")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    compact = audit_pool(pathlib.Path(args.compact), tokenizer, args.seq_len, batch_sizes, args.epochs, args.checkpoint_interval)
    repeat = audit_pool(pathlib.Path(args.repeat), tokenizer, args.seq_len, batch_sizes, args.epochs, args.checkpoint_interval)

    def diff(a: float, b: float) -> dict[str, float]:
        return {"compact_minus_repeat": a - b, "relative_to_repeat": (a - b) / b if b else float("nan")}

    comparison = {
        "raw_tokens_with_eos": diff(compact["raw_tokens_with_eos"], repeat["raw_tokens_with_eos"]),
        "token_per_word": diff(compact["token_per_word"], repeat["token_per_word"]),
        "active_tokens_per_epoch": diff(compact["active_tokens_per_epoch"], repeat["active_tokens_per_epoch"]),
        "chunks_per_epoch": diff(compact["chunks_per_epoch"], repeat["chunks_per_epoch"]),
        "effective_active_words_per_epoch": diff(compact["effective_active_words_per_epoch"], repeat["effective_active_words_per_epoch"]),
    }
    for bsz in batch_sizes:
        k = str(bsz)
        comparison[f"total_steps_batch{bsz}"] = diff(compact["batch_plans"][k]["total_steps"], repeat["batch_plans"][k]["total_steps"])
        comparison[f"trainer_effective_words_after_epochs_batch{bsz}"] = diff(
            compact["batch_plans"][k]["trainer_effective_words_after_epochs"],
            repeat["batch_plans"][k]["trainer_effective_words_after_epochs"],
        )

    summary = {
        "status": "CAUSAL_TOKEN_EXPOSURE_AUDIT_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 1),
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "tokenizer_dir": args.tokenizer,
        "tokenizer_json_sha256": sha256_file(pathlib.Path(args.tokenizer) / "tokenizer.json"),
        "compact": compact,
        "repeat": repeat,
        "comparison": comparison,
        "interpretation": {
            "legal_budget": "Both arms are exactly 10,000,000 words per epoch and 100,000,000 charged words over 10 epochs.",
            "effective_token_exposure": "Decoder-only tokenization/chunking differs slightly between compact and repeat; report deltas with all trajectory comparisons.",
        },
    }

    json_path = out_dir / "causal_token_exposure_audit.json"
    md_path = out_dir / "causal_token_exposure_audit.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    c = comparison
    md = [
        "# research causal GPT token/effective-exposure audit",
        "",
        f"Tokenizer SHA: `{summary['tokenizer_json_sha256']}`",
        "",
        "Both arms are legal-word matched: 10,000,000 words per epoch, 10 epochs = 100,000,000 charged words. The quantities below describe the tokenizer/chunk stream actually seen by the GPT trainer.",
        "",
        "| quantity | compact | repeat | compact-repeat | rel. |",
        "|---|---:|---:|---:|---:|",
        f"| raw tokens incl EOS / epoch | {compact['raw_tokens_with_eos']:,} | {repeat['raw_tokens_with_eos']:,} | {c['raw_tokens_with_eos']['compact_minus_repeat']:,} | {100*c['raw_tokens_with_eos']['relative_to_repeat']:.4f}% |",
        f"| active 256-token positions / epoch | {compact['active_tokens_per_epoch']:,} | {repeat['active_tokens_per_epoch']:,} | {c['active_tokens_per_epoch']['compact_minus_repeat']:,} | {100*c['active_tokens_per_epoch']['relative_to_repeat']:.4f}% |",
        f"| chunks / epoch | {compact['chunks_per_epoch']:,} | {repeat['chunks_per_epoch']:,} | {c['chunks_per_epoch']['compact_minus_repeat']:,} | {100*c['chunks_per_epoch']['relative_to_repeat']:.4f}% |",
        f"| token/word | {compact['token_per_word']:.6f} | {repeat['token_per_word']:.6f} | {c['token_per_word']['compact_minus_repeat']:.6f} | {100*c['token_per_word']['relative_to_repeat']:.4f}% |",
    ]
    for bsz in batch_sizes:
        k = str(bsz)
        md.append(f"| total updates at batch {bsz} | {compact['batch_plans'][k]['total_steps']:,} | {repeat['batch_plans'][k]['total_steps']:,} | {c[f'total_steps_batch{bsz}']['compact_minus_repeat']:,} | {100*c[f'total_steps_batch{bsz}']['relative_to_repeat']:.4f}% |")
    md.extend([
        "",
        "## Interpretation",
        "The causal replication should be interpreted as a word-budget-matched experiment with a small tokenizer/chunk exposure difference intrinsic to the different texts. A compact advantage must be read across checkpoints and families, not reduced to this token-count delta or to one column spike.",
        "",
        f"JSON: `{json_path}`",
    ])
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(json_path), "md": str(md_path), "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
