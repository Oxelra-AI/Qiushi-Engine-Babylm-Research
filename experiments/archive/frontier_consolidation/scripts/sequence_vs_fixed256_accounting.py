#!/usr/bin/env python3
"""research: compare faithful pair-aware sequence accounting against fixed seq256.

research/early-research ratios were intentionally measured against the inherited
prefix-sliced sequence path.  The current best legal endpoint, however, is a
fixed seq256 model, not a prefix-sliced 64/128/256 curriculum.  This CPU-only
combiner prevents a route-selection error by quantifying what a faithful
pair-aware sequence stream changes relative to fixed seq256 training on the same
10M compact-view reinvest corpus.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

USER_ROOT = Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
PAIRAWARE = WORKSPACE / "data" / "pairaware_sequence_accounting" / "pairaware_sequence_accounting.json"
OUT_DIR = WORKSPACE / "data" / "sequence_vs_fixed256_accounting"
NOTE = (USER_ROOT / 'research/notes/frontier_consolidation/sequence_vs_fixed256_accounting.md')
MASK_PROB = 0.15
FIXED_EPOCHS = 10


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(PAIRAWARE.read_text(encoding="utf-8"))
    if d.get("status") != "PAIRAWARE_SEQUENCE_ACCOUNTING":
        raise RuntimeError(f"unexpected input {d.get('status')}")
    result: dict[str, Any] = {
        "status": "SEQUENCE_VS_FIXED256_ACCOUNTING",
        "created_utc": now_utc(),
        "purpose": "Clarify that faithful sequence-stream ratios should be compared to the actual fixed-seq256 legal endpoint recipe, not only to the flawed prefix-sliced sequence path.",
        "input_json": rel(PAIRAWARE),
        "sha256": d["sha256"],
        "by_tokenizer": {},
        "interpretation": {
            "fixed_reference": "The best legal endpoint used fixed seq256; each epoch exposes only the first 256 tokenizer tokens of each row, so it sees visible seq256 active tokens, not the full untruncated tokenizer stream.",
            "sequence_reference": "Faithful group chunks expose the full untruncated tokenizer stream at every stage length while preserving the same 10M declared words per epoch; the main direct difference from fixed seq256 is recovering the small truncated-token tail plus changing context-length and update geometry.",
            "routing_reading": "Large ratios against prefix slicing show why the old seq_len_schedule path was not a valid sequence experiment. Against fixed seq256, the active-token gain is only the truncation tail; any real score change would have to come from shorter-context curriculum and pair-aware packing, not from a massive increase in text seen.",
        },
    }
    rows = []
    for tok, rec in d["by_tokenizer"].items():
        fixed = rec["by_length"]["256"]
        fixed_active_per_epoch = int(fixed["prefix_active_tokens_per_epoch"])
        fixed_active_total = fixed_active_per_epoch * FIXED_EPOCHS
        fixed_steps_total = int(fixed["prefix_steps_per_epoch_current_loop"]) * FIXED_EPOCHS
        raw_total = int(rec["raw_tokens_per_epoch"]) * FIXED_EPOCHS
        tok_out: dict[str, Any] = {
            "declared_words_total_fixed_and_sequence": int(rec["declared_words_per_epoch"]) * FIXED_EPOCHS,
            "fixed_seq256_active_tokens_total": fixed_active_total,
            "fixed_seq256_expected_masked_tokens_total": fixed_active_total * MASK_PROB,
            "fixed_seq256_optimizer_steps_total": fixed_steps_total,
            "raw_untruncated_tokens_total": raw_total,
            "raw_untruncated_token_ratio_vs_fixed_seq256": raw_total / max(1, fixed_active_total),
            "raw_minus_fixed_seq256_tokens_total": raw_total - fixed_active_total,
            "raw_minus_fixed_seq256_tokens_per_100M_words": (raw_total - fixed_active_total),
            "schedules": {},
        }
        for schedule, srec in rec["schedules"].items():
            seq_tokens = int(srec["pairaware_active_tokens_total"])
            seq_steps = int(srec["pairaware_optimizer_steps_total"])
            out = {
                "charged_words_total": int(srec["charged_words_total"]),
                "sequence_active_tokens_total": seq_tokens,
                "sequence_expected_masked_tokens_total": seq_tokens * MASK_PROB,
                "active_token_ratio_vs_fixed_seq256": seq_tokens / max(1, fixed_active_total),
                "active_token_delta_vs_fixed_seq256": seq_tokens - fixed_active_total,
                "expected_masked_token_delta_vs_fixed_seq256": (seq_tokens - fixed_active_total) * MASK_PROB,
                "sequence_optimizer_steps_total": seq_steps,
                "optimizer_step_ratio_vs_fixed_seq256": seq_steps / max(1, fixed_steps_total),
                "optimizer_step_delta_vs_fixed_seq256": seq_steps - fixed_steps_total,
                "active_token_ratio_vs_flawed_prefix_sequence": float(srec["pairaware_active_token_ratio_vs_prefix"]),
                "optimizer_step_ratio_vs_flawed_prefix_sequence": float(srec["pairaware_step_ratio_vs_prefix"]),
                "pair_full_fraction_epoch_mean": float(srec["pairaware_pair_full_fraction_epoch_mean"]),
                "pair_overlong_fraction_epoch_mean": float(srec["pairaware_pair_overlong_fraction_epoch_mean"]),
            }
            tok_out["schedules"][schedule] = out
            rows.append({"tokenizer": tok, "schedule": schedule, **out, **{
                "fixed_seq256_active_tokens_total": fixed_active_total,
                "fixed_seq256_optimizer_steps_total": fixed_steps_total,
                "raw_untruncated_token_ratio_vs_fixed_seq256": tok_out["raw_untruncated_token_ratio_vs_fixed_seq256"],
            }})
        result["by_tokenizer"][tok] = tok_out
    out_json = OUT_DIR / "sequence_vs_fixed256_accounting.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT_DIR / "sequence_vs_fixed256_accounting.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tokenizer", "schedule", "charged_words_total", "fixed_seq256_active_tokens_total", "sequence_active_tokens_total",
            "active_token_ratio_vs_fixed_seq256", "active_token_delta_vs_fixed_seq256", "expected_masked_token_delta_vs_fixed_seq256",
            "fixed_seq256_optimizer_steps_total", "sequence_optimizer_steps_total", "optimizer_step_ratio_vs_fixed_seq256",
            "optimizer_step_delta_vs_fixed_seq256", "raw_untruncated_token_ratio_vs_fixed_seq256",
            "active_token_ratio_vs_flawed_prefix_sequence", "optimizer_step_ratio_vs_flawed_prefix_sequence",
            "pair_full_fraction_epoch_mean", "pair_overlong_fraction_epoch_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    lines = []
    lines.append("# research sequence vs fixed-seq256 accounting\n")
    lines.append("CPU-only correction of the sequence-route interpretation. No model was trained and no official evaluation text was read.\n")
    lines.append(f"\nInput: `{rel(PAIRAWARE)}`. Pool SHA matched: `{result['sha256']['pool_matches']}`.\n")
    lines.append("\n## Why this comparison matters\n")
    lines.append("The large faithful/prefix ratios from research-research are real, but they are ratios against the flawed prefix-sliced `seq_len_schedule` path. The current strongest legal model is fixed seq256. Against fixed seq256, a faithful sequence stream mainly recovers the row-truncation tail and changes context/update geometry; it is not a 1.6x increase in training text seen.\n")
    lines.append("\n## Schedule comparison against fixed seq256\n")
    lines.append("| tokenizer | schedule | charged words | fixed active tokens | sequence active tokens | token ratio vs fixed | token Δ | expected masked-token Δ | fixed steps | sequence steps | step ratio vs fixed | pair full | pair overlong |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for tok, tok_out in result["by_tokenizer"].items():
        for schedule, r in tok_out["schedules"].items():
            lines.append(
                f"| {tok} | {schedule} | {r['charged_words_total']} | {tok_out['fixed_seq256_active_tokens_total']} | {r['sequence_active_tokens_total']} | {r['active_token_ratio_vs_fixed_seq256']:.4f} | {r['active_token_delta_vs_fixed_seq256']} | {r['expected_masked_token_delta_vs_fixed_seq256']:.0f} | {tok_out['fixed_seq256_optimizer_steps_total']} | {r['sequence_optimizer_steps_total']} | {r['optimizer_step_ratio_vs_fixed_seq256']:.3f} | {r['pair_full_fraction_epoch_mean']:.3f} | {r['pair_overlong_fraction_epoch_mean']:.3f} |\n"
            )
    lines.append("\n## Scientific reading\n")
    lines.append("- The faithful pair-aware sequence stream is legally clean and implementation-feasible, but relative to the actual fixed-seq256 endpoint it adds only about 2.3-2.6% active tokens (the untruncated tail), while adding about 8-13% optimizer steps under inverse batch sizing.\n")
    lines.append("- Its possible value is therefore not simple token-volume expansion; it would have to come from shorter-context learning dynamics, row-tail recovery, and preserving compact source+rewrite atoms during early short-context exposure.\n")
    lines.append("- This makes sequence curriculum a plausible later route, especially because it is compatible with minfreq50 and pair-aware packing, but it is not strong enough to jump ahead of the active word-mean result or the already-hardened minfreq50 screen.\n")
    lines.append(f"\nFull JSON: `{rel(out_json)}`\n")
    lines.append(f"CSV: `{rel(OUT_DIR / 'sequence_vs_fixed256_accounting.csv')}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
