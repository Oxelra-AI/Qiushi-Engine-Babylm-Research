#!/usr/bin/env python3
"""research: corrected segmented schedule for exact 82M-anchor replay.

The known research coherent tail stopped after a partial 101st batch because its
max_tail_charged_words cap was 3,992,918 and the next row would exceed that cap.
A truthful 86M->100M completion must therefore replay the exact partial 101-batch
prefix first, compare it against the known research states, then continue from the
next unread row.  A naive single DataLoader over the full remaining suffix would
make update 101 a full 256-row batch and cannot reproduce known 86M bit-for-bit.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
A01 = _public_path('experiments/archive/representation_and_objectives')
A02 = _public_path('experiments/archive/frontier_consolidation')
STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
LOG = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/training_log.jsonl')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/alpha075_exact_replay_plan')
INITIAL = 82_012_495
FULL_CAP = 100_000_000
SKIP_ROWS = 530_944
BATCH_SIZE = 256
KNOWN_CAP = 3_992_918
SCHEDULE_TOTAL = 455
KNOWN_THRESHOLDS = [1_000_000, 2_000_000, 3_000_000]
OFFICIAL_TOTALS = [90_000_000, 100_000_000]


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def iter_rows(start_row: int, max_words: int):
    selected = 0
    with STREAM.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_row:
                continue
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch row={idx}: {words} vs {len(text.split())}")
            if selected + words > max_words:
                return
            selected += words
            yield {"row_index": idx, "words": words}


def make_batches(start_row: int, max_words: int, starting_update: int, starting_tail: int) -> tuple[list[dict[str, Any]], int, int, int]:
    rows = list(iter_rows(start_row, max_words))
    batches = []
    tail = starting_tail
    for bi in range(0, len(rows), BATCH_SIZE):
        chunk = rows[bi:bi + BATCH_SIZE]
        words = sum(r["words"] for r in chunk)
        tail += words
        rec = {
            "update": starting_update + len(batches) + 1,
            "source_row_start": chunk[0]["row_index"],
            "source_row_end": chunk[-1]["row_index"],
            "batch_words": words,
            "tail_main_words": tail,
            "total_consumed_words": INITIAL + tail,
            "rows_in_batch": len(chunk),
        }
        batches.append(rec)
    return batches, tail, rows[0]["row_index"] if rows else start_row, rows[-1]["row_index"] if rows else start_row - 1


def read_log() -> list[dict[str, Any]]:
    out = []
    with LOG.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def compare_prefix(log_rows, phase1_batches):
    mismatches = []
    fields = ["update", "source_row_start", "source_row_end", "batch_words", "tail_main_words", "total_consumed_words"]
    for a, b in zip(log_rows, phase1_batches):
        mm = {k: [a.get(k), b.get(k)] for k in fields if a.get(k) != b.get(k)}
        if mm:
            mismatches.append({"update": a.get("update"), "mismatches": mm})
    if len(log_rows) != len(phase1_batches):
        mismatches.append({"length_mismatch": [len(log_rows), len(phase1_batches)]})
    return mismatches


def first_crossing(batches: list[dict[str, Any]], target_tail: int):
    for b in batches:
        if b["tail_main_words"] >= target_tail:
            return b
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_rows = read_log()
    phase1, tail_after_phase1, first1, last1 = make_batches(SKIP_ROWS, KNOWN_CAP, 0, 0)
    continuation_start = last1 + 1
    remaining_after_phase1 = FULL_CAP - INITIAL - tail_after_phase1
    phase2, tail_final, first2, last2 = make_batches(continuation_start, remaining_after_phase1, len(phase1), tail_after_phase1)
    all_batches = phase1 + phase2
    prefix_mismatches = compare_prefix(log_rows, phase1)
    known_crossings = {f"tail_{t}": first_crossing(phase1, t) for t in KNOWN_THRESHOLDS}
    official_crossings = {f"chck_{total // 1_000_000}M": first_crossing(all_batches, total - INITIAL) for total in OFFICIAL_TOTALS}
    out = {
        "status": "ALPHA075_SEGMENTED_REPLAY_SCHEDULE",
        "stream": rel(STREAM),
        "initial_consumed_words": INITIAL,
        "full_cap_words": FULL_CAP,
        "skip_rows": SKIP_ROWS,
        "known_phase_cap_words": KNOWN_CAP,
        "known_phase_batches": len(phase1),
        "known_phase_tail_words": tail_after_phase1,
        "known_phase_total_words": INITIAL + tail_after_phase1,
        "known_phase_first_row": first1,
        "known_phase_last_row": last1,
        "continuation_start_row": continuation_start,
        "continuation_remaining_word_cap": remaining_after_phase1,
        "continuation_batches": len(phase2),
        "continuation_first_row": first2,
        "continuation_last_row": last2,
        "final_tail_words": tail_final,
        "final_total_words": INITIAL + tail_final,
        "total_batches": len(all_batches),
        "schedule_total_original": SCHEDULE_TOTAL,
        "schedule_total_matches_segmented_replay": len(all_batches) == SCHEDULE_TOTAL,
        "log_rows": len(log_rows),
        "prefix_matches_phase1": not prefix_mismatches,
        "prefix_mismatches": prefix_mismatches[:20],
        "known_tail_crossings": known_crossings,
        "official_crossings": official_crossings,
        "selected_batch_records": {
            "phase1_last": phase1[-1] if phase1 else None,
            "first_continuation": phase2[0] if phase2 else None,
            "chck_90M": official_crossings.get("chck_90M"),
            "chck_100M": official_crossings.get("chck_100M"),
        },
        "interpretation": "Use this segmented schedule for guarded replay. The first 101 updates reproduce research's partial-last-batch run; updates 102-455 continue from the next unread row under the same 455-step LR schedule.",
    }
    out_json = _public_path('experiments/archive/representation_and_objectives/data/alpha075_exact_replay_plan/alpha075_segmented_replay_schedule.json')
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# research corrected segmented replay schedule",
        "",
        f"Status: `{out['status']}`.",
        f"Known research prefix: {len(phase1)} batches, {tail_after_phase1:,} tail words, total {INITIAL + tail_after_phase1:,}, rows {first1}-{last1}.",
        f"Prefix matches research log exactly: {not prefix_mismatches}.",
        f"Continuation starts at row {continuation_start} and has {len(phase2)} batches, ending row {last2}; full segmented replay total batches {len(all_batches)} equals original LR schedule_total {SCHEDULE_TOTAL}: {len(all_batches) == SCHEDULE_TOTAL}.",
        f"Official chck_90M should be saved after update {official_crossings['chck_90M']['update']} at actual total {official_crossings['chck_90M']['total_consumed_words']:,}.",
        f"Official chck_100M should be saved after update {official_crossings['chck_100M']['update']} at actual total {official_crossings['chck_100M']['total_consumed_words']:,}.",
        "",
        "This repairs the naive full-suffix audit: a single DataLoader would make update 101 a full batch and fail to reproduce the known final86 state.",
        f"JSON: `{rel(out_json)}`",
    ]
    (_public_path('research/documents/representation_and_objectives/data/alpha075_exact_replay_plan/alpha075_segmented_replay_schedule.md')).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "out_json": rel(out_json),
        "prefix_matches": not prefix_mismatches,
        "known_phase_batches": len(phase1),
        "continuation_batches": len(phase2),
        "total_batches": len(all_batches),
        "schedule_matches": len(all_batches) == SCHEDULE_TOTAL,
        "official_crossings": official_crossings,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
