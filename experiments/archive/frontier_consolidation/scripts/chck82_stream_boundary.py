#!/usr/bin/env python3
"""Extract the exact stream/update boundary for the verified scale1.75 chck_82M.

The frozen-82M private-tail design must spend only the remaining legal exposure after
the batch that created chck_82M.  This script reads the original training log and
scientific_metrics checkpoint record to locate that update and the next legal row.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import time
from pathlib import Path

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
RUN_DIR = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder')
LOG = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/training_log.jsonl')
METRICS = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/scientific_metrics.json')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/chck82_stream_boundary')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/chck82_stream_boundary/chck82_stream_boundary.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/chck82_stream_boundary/chck82_stream_boundary.md')

TARGET = 82_000_000
FULL_CAP = 100_000_000


def rel(p: Path | str):
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    rows = []
    with LOG.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    boundary = None
    prev = None
    for r in rows:
        cw = int(r.get("cumulative_words", r.get("cumulative_word_exposure", r.get("cumulative_main_words", -1))))
        if cw >= TARGET:
            boundary = r
            break
        prev = r
    if boundary is None:
        raise RuntimeError("No row crosses 82M")
    update = int(boundary.get("step", boundary.get("update")))
    loader_step = int(boundary.get("loader_step", update))
    actual = int(boundary.get("cumulative_words", boundary.get("cumulative_word_exposure", boundary.get("cumulative_main_words"))))
    prev_actual = int(prev.get("cumulative_words", prev.get("cumulative_word_exposure", prev.get("cumulative_main_words", 0)))) if prev else 0
    batch_words = int(boundary.get("batch_words", actual - prev_actual))
    # Original trainer consumed rows sequentially one batch per update. The next tail batch
    # should start at loader_step+1 (0-index skip=loader_step*batch_size if no partial batch).
    batch_size = 256
    row_skip = loader_step * batch_size
    remaining = FULL_CAP - actual
    ckpt_records = {x.get("name"): x for x in metrics.get("saved_checkpoints", [])}
    chck82_record = ckpt_records.get("chck_82M")
    checks = {
        "boundary_matches_metrics_chck82_words": chck82_record is not None and actual == int(chck82_record.get("actual_cumulative_word_exposure")),
        "boundary_crosses_target_from_below": prev_actual < TARGET <= actual,
        "remaining_positive": remaining > 0,
        "remaining_within_cap": actual + remaining == FULL_CAP,
        "training_log_has_expected_updates": len(rows) == int(metrics.get("actual_training_steps")),
    }
    checks["all_checks_passed"] = all(checks.values())
    result = {
        "status": "PASS" if checks["all_checks_passed"] else "CHECKS_FAILED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": rel(RUN_DIR),
        "training_log": rel(LOG),
        "target_checkpoint": "chck_82M",
        "target_word_exposure": TARGET,
        "full_cap": FULL_CAP,
        "boundary_update_record": boundary,
        "previous_update_record": prev,
        "boundary_update": update,
        "boundary_loader_step": loader_step,
        "boundary_batch_words": batch_words,
        "actual_cumulative_words_after_boundary_batch": actual,
        "previous_cumulative_words": prev_actual,
        "overshoot_words": actual - TARGET,
        "remaining_legal_charged_words_after_chck82": remaining,
        "tail_start_loader_step_1indexed": loader_step + 1,
        "tail_start_row_index_0based_if_batch256": row_skip,
        "tail_skip_rows_if_reusing_same_stream_batching": row_skip,
        "batch_size_assumed": batch_size,
        "metrics_chck82_record": chck82_record,
        "checks": checks,
        "scientific_reading": "For a frozen-82M tail, the slow function already consumed the full batch ending at the boundary update. Any remaining-exposure private training should start from the next stream batch/row and cap total new charged words at 17,987,505 so the full model remains at or below 100M counted-word exposure.",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# research chck_82M stream boundary",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- Boundary update: `{update}`; loader step: `{loader_step}`.",
        f"- Previous cumulative words: `{prev_actual}`; boundary cumulative words: `{actual}`; target overshoot: `{actual - TARGET}`.",
        f"- Remaining legal charged words after chck_82M: `{remaining}`.",
        f"- Tail should start at next loader step `{loader_step + 1}`; if using batch256 sequential rows, skip `{row_skip}` rows.",
        "",
        "## Checks",
    ]
    for k, v in checks.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append(result["scientific_reading"])
    lines.append(f"\nJSON: `{rel(OUT_JSON)}`")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "boundary_update": update, "actual": actual, "remaining": remaining, "tail_skip_rows": row_skip, "out_json": rel(OUT_JSON)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
