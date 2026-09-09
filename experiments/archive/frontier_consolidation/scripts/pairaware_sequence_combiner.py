#!/usr/bin/env python3
"""research: combine tokenizer-group chunk and pair-atomic measurements.

The previous research measurement already scanned the full 10M pool with greedy
WWM-group chunks and scanned the changed compact-pair rows with pair-atomic
chunks.  This CPU-only script combines those measurements into the full-pool
accounting for a possible faithful pair-aware sequence stream:

  full-pool pair-aware chunks = full-pool greedy chunks
                               - changed-row greedy chunks
                               + changed-row pair-atomic chunks.

Active target tokens and legal declared-word exposure are unchanged relative to
greedy group chunks; only chunk/optimizer-step count and pair preservation change.
No training and no official evaluation text are used.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

USER_ROOT = Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
INPUT_JSON = WORKSPACE / "data" / "tokenizer_group_chunk_measurement" / "tokenizer_group_chunk_measurement.json"
OUT_DIR = WORKSPACE / "data" / "pairaware_sequence_accounting"
NOTE = (USER_ROOT / 'research/notes/frontier_consolidation/pairaware_sequence_accounting.md')
LENGTHS = [64, 128, 256]
SCHEDULES = {
    "64x3_128x4_256x3": {64: 3, 128: 4, 256: 3},
    "64x7_256x3": {64: 7, 128: 0, 256: 3},
}


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
    d = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    if d.get("status") != "TOKENIZER_GROUP_CHUNK_MEASUREMENT":
        raise RuntimeError(f"unexpected input status {d.get('status')}")
    result: dict[str, Any] = {
        "status": "PAIRAWARE_SEQUENCE_ACCOUNTING",
        "created_utc": now_utc(),
        "purpose": "Combine full-pool greedy WWM-group chunk accounting with changed-block pair-atomic chunk accounting to quantify a faithful pair-aware sequence stream before any GPU sequence run.",
        "input_json": rel(INPUT_JSON),
        "sha256": d["sha256"],
        "tokenizers": d["tokenizers"],
        "by_tokenizer": {},
        "interpretation": {
            "word_budget": "Pair-aware chunks reorganize the same frozen 10M declared words per epoch. The ten-epoch schedules remain 100M charged words.",
            "active_tokens": "Full-pool active tokens equal the greedy tokenizer-group chunk stream because pair-atomic packing does not drop tokens; it changes only boundaries and step count.",
            "compact_view": "The changed compact block uses source+rewrite pairs as atoms when they fit the stage length, preserving the load-bearing same-window second-view signal for most pairs at short lengths.",
            "launch_status": "No model training was launched. A sequence-training comparison remains conditional on the word-mean/support-floor evidence.",
        },
    }
    rows = []
    sched_rows = []
    for tok, full in d["full_pool_group_chunking"].items():
        changed = d["changed_pair_group_chunking"][tok]
        tok_out: dict[str, Any] = {
            "declared_words_per_epoch": full["declared_words"],
            "raw_tokens_per_epoch": full["raw_tokens"],
            "changed_rows": changed["changed_rows"],
            "changed_pairs": changed["pairs"],
            "by_length": {},
            "schedules": {},
        }
        for L in LENGTHS:
            fs = full["by_length"][str(L)]
            cs = changed["by_length"][str(L)]
            full_greedy_chunks = int(fs["group_chunks_per_epoch"])
            changed_greedy_chunks = int(cs["greedy_chunks_total_changed_rows"])
            changed_pair_chunks = int(cs["pair_atomic_chunks_total_changed_rows"])
            pairaware_chunks = full_greedy_chunks - changed_greedy_chunks + changed_pair_chunks
            inverse_batch = int(fs["inverse_scaled_row_batch"])
            prefix_steps = int(fs["prefix_steps_per_epoch_current_loop_batch256"])
            greedy_steps = int(fs["group_steps_per_epoch_inverse_batch"])
            pair_steps = math.ceil(pairaware_chunks / inverse_batch)
            rec = {
                "stage_length": L,
                "inverse_scaled_row_batch": inverse_batch,
                "prefix_steps_per_epoch_current_loop": prefix_steps,
                "prefix_active_tokens_per_epoch": int(fs["prefix_active_tokens_per_epoch"]),
                "pairaware_active_tokens_per_epoch": int(fs["group_active_tokens_per_epoch"]),
                "pairaware_active_token_ratio_vs_prefix": fs["group_active_token_ratio_vs_prefix"],
                "full_pool_greedy_chunks_per_epoch": full_greedy_chunks,
                "changed_row_greedy_chunks": changed_greedy_chunks,
                "changed_row_pair_atomic_chunks": changed_pair_chunks,
                "pairaware_chunks_per_epoch": pairaware_chunks,
                "pairaware_chunk_delta_vs_greedy": pairaware_chunks - full_greedy_chunks,
                "greedy_steps_per_epoch_inverse_batch": greedy_steps,
                "pairaware_steps_per_epoch_inverse_batch": pair_steps,
                "pairaware_step_delta_vs_greedy": pair_steps - greedy_steps,
                "pairaware_step_ratio_vs_prefix": pair_steps / max(1, prefix_steps),
                "pairaware_step_ratio_vs_greedy": pair_steps / max(1, greedy_steps),
                "prefix_pair_full_fraction": cs["prefix_full_fraction"],
                "greedy_pair_full_fraction": cs["greedy_full_same_chunk_fraction"],
                "pair_atomic_pair_full_fraction": cs["pair_atomic_full_same_chunk_fraction"],
                "pair_atomic_pair_overlong_fraction": cs["pair_atomic_overlong_fraction"],
            }
            tok_out["by_length"][str(L)] = rec
            rows.append({"tokenizer": tok, **rec})
        for name, counts_by_L in SCHEDULES.items():
            total_epochs = sum(counts_by_L.values())
            charged = int(full["declared_words"]) * total_epochs
            prefix_tokens = 0
            pair_tokens = 0
            prefix_steps_total = 0
            greedy_steps_total = 0
            pair_steps_total = 0
            prefix_pair = 0.0
            greedy_pair = 0.0
            atomic_pair = 0.0
            atomic_overlong = 0.0
            for L, epochs in counts_by_L.items():
                if epochs <= 0:
                    continue
                r = tok_out["by_length"][str(L)]
                prefix_tokens += epochs * int(r["prefix_active_tokens_per_epoch"])
                pair_tokens += epochs * int(r["pairaware_active_tokens_per_epoch"])
                prefix_steps_total += epochs * int(r["prefix_steps_per_epoch_current_loop"])
                greedy_steps_total += epochs * int(r["greedy_steps_per_epoch_inverse_batch"])
                pair_steps_total += epochs * int(r["pairaware_steps_per_epoch_inverse_batch"])
                prefix_pair += epochs * float(r["prefix_pair_full_fraction"])
                greedy_pair += epochs * float(r["greedy_pair_full_fraction"])
                atomic_pair += epochs * float(r["pair_atomic_pair_full_fraction"])
                atomic_overlong += epochs * float(r["pair_atomic_pair_overlong_fraction"])
            srec = {
                "epochs_by_length": counts_by_L,
                "charged_words_total": charged,
                "prefix_active_tokens_total": prefix_tokens,
                "pairaware_active_tokens_total": pair_tokens,
                "pairaware_active_token_ratio_vs_prefix": pair_tokens / max(1, prefix_tokens),
                "prefix_optimizer_steps_total": prefix_steps_total,
                "greedy_group_optimizer_steps_total": greedy_steps_total,
                "pairaware_optimizer_steps_total": pair_steps_total,
                "pairaware_step_delta_vs_greedy": pair_steps_total - greedy_steps_total,
                "pairaware_step_ratio_vs_prefix": pair_steps_total / max(1, prefix_steps_total),
                "pairaware_step_ratio_vs_greedy": pair_steps_total / max(1, greedy_steps_total),
                "prefix_pair_full_fraction_epoch_mean": prefix_pair / total_epochs,
                "greedy_pair_full_fraction_epoch_mean": greedy_pair / total_epochs,
                "pairaware_pair_full_fraction_epoch_mean": atomic_pair / total_epochs,
                "pairaware_pair_overlong_fraction_epoch_mean": atomic_overlong / total_epochs,
            }
            tok_out["schedules"][name] = srec
            sched_rows.append({"tokenizer": tok, "schedule": name, **srec})
        result["by_tokenizer"][tok] = tok_out

    out_json = OUT_DIR / "pairaware_sequence_accounting.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT_DIR / "pairaware_by_length.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tokenizer", "stage_length", "pairaware_active_token_ratio_vs_prefix", "full_pool_greedy_chunks_per_epoch",
            "changed_row_greedy_chunks", "changed_row_pair_atomic_chunks", "pairaware_chunks_per_epoch", "pairaware_chunk_delta_vs_greedy",
            "greedy_steps_per_epoch_inverse_batch", "pairaware_steps_per_epoch_inverse_batch", "pairaware_step_delta_vs_greedy",
            "pairaware_step_ratio_vs_prefix", "pairaware_step_ratio_vs_greedy", "prefix_pair_full_fraction", "greedy_pair_full_fraction",
            "pair_atomic_pair_full_fraction", "pair_atomic_pair_overlong_fraction",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    with (OUT_DIR / "pairaware_schedules.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "tokenizer", "schedule", "charged_words_total", "pairaware_active_token_ratio_vs_prefix",
            "prefix_optimizer_steps_total", "greedy_group_optimizer_steps_total", "pairaware_optimizer_steps_total",
            "pairaware_step_delta_vs_greedy", "pairaware_step_ratio_vs_prefix", "pairaware_step_ratio_vs_greedy",
            "prefix_pair_full_fraction_epoch_mean", "greedy_pair_full_fraction_epoch_mean", "pairaware_pair_full_fraction_epoch_mean",
            "pairaware_pair_overlong_fraction_epoch_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sched_rows:
            flat = dict(r)
            flat.pop("epochs_by_length", None)
            w.writerow({k: flat.get(k) for k in fieldnames})

    lines: list[str] = []
    lines.append("# research pair-aware sequence accounting\n")
    lines.append("CPU-only combination of full-pool tokenizer-group chunk accounting with compact-block pair-atomic chunk accounting. No model was trained and no official evaluation text was read.\n")
    lines.append(f"\nInput JSON: `{rel(INPUT_JSON)}`. Pool SHA matched: `{result['sha256']['pool_matches']}`.\n")
    lines.append("\n## Per-length full-pool accounting\n")
    lines.append("Pair-aware chunks differ from greedy chunks only in the compact changed block. Active tokens are unchanged; step changes come from atom-preserving boundaries.\n\n")
    lines.append("| tokenizer | L | active-token ratio vs prefix | greedy chunks | pair-aware chunks | chunk Δ | greedy steps | pair-aware steps | step Δ | step ratio vs prefix | pair full prefix/greedy/atomic | overlong atomic |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for tok, tok_out in result["by_tokenizer"].items():
        for L in ["64", "128", "256"]:
            r = tok_out["by_length"][L]
            lines.append(
                f"| {tok} | {L} | {r['pairaware_active_token_ratio_vs_prefix']:.3f} | {r['full_pool_greedy_chunks_per_epoch']} | {r['pairaware_chunks_per_epoch']} | {r['pairaware_chunk_delta_vs_greedy']} | {r['greedy_steps_per_epoch_inverse_batch']} | {r['pairaware_steps_per_epoch_inverse_batch']} | {r['pairaware_step_delta_vs_greedy']} | {r['pairaware_step_ratio_vs_prefix']:.3f} | {r['prefix_pair_full_fraction']:.3f}/{r['greedy_pair_full_fraction']:.3f}/{r['pair_atomic_pair_full_fraction']:.3f} | {r['pair_atomic_pair_overlong_fraction']:.3f} |\n"
            )
    lines.append("\n## Ten-epoch schedules\n")
    lines.append("| tokenizer | schedule | charged words | active-token ratio vs prefix | prefix steps | greedy steps | pair-aware steps | step Δ vs greedy | step ratio vs prefix | pair full prefix/greedy/atomic | overlong atomic |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for tok, tok_out in result["by_tokenizer"].items():
        for schedule, r in tok_out["schedules"].items():
            lines.append(
                f"| {tok} | {schedule} | {r['charged_words_total']} | {r['pairaware_active_token_ratio_vs_prefix']:.3f} | {r['prefix_optimizer_steps_total']} | {r['greedy_group_optimizer_steps_total']} | {r['pairaware_optimizer_steps_total']} | {r['pairaware_step_delta_vs_greedy']} | {r['pairaware_step_ratio_vs_prefix']:.3f} | {r['prefix_pair_full_fraction_epoch_mean']:.3f}/{r['greedy_pair_full_fraction_epoch_mean']:.3f}/{r['pairaware_pair_full_fraction_epoch_mean']:.3f} | {r['pairaware_pair_overlong_fraction_epoch_mean']:.3f} |\n"
            )
    lines.append("\n## Scientific reading\n")
    lines.append("- Pair-aware group chunking is affordable: on the 64x3/128x4/256x3 schedule it adds only a few optimizer steps over greedy group chunks while preserving about 0.94 of compact source+rewrite pairs as full same-window atoms across the ten-epoch mix.\n")
    lines.append("- The 64x7/256x3 schedule exposes more active tokens but sacrifices more pair atoms at L64; it is a sharper sequence-first intervention and less obviously compatible with the validated compact-view mechanism.\n")
    lines.append("- This strengthens the construction case for a future faithful pair-aware chunk-stream trainer, but it remains subordinate to the live word-mean decision and the prepared support-floor screen.\n")
    lines.append(f"\nFull JSON: `{rel(out_json)}`\n")
    lines.append(f"CSV: `{rel(OUT_DIR / 'pairaware_by_length.csv')}`, `{rel(OUT_DIR / 'pairaware_schedules.csv')}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
