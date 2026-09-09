#!/usr/bin/env python3
"""research: audit exact row/batch schedule for a truthful alpha0.75 tail.

This is CPU-only.  It reads the original compact-view 100M stream used by the
research coherent private-tail run and computes the row-boundary batches from the
verified 82,012,495-word anchor through the remaining <=100M cap.  It also checks
where the known research 83/84/85M checkpoint states and 86.005M final state sit
inside a full deterministic replay, and where genuine official 90M/100M states
would be saved.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import hashlib
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
A02 = _public_path('experiments/archive/frontier_consolidation')
A01 = _public_path('experiments/archive/representation_and_objectives')
STREAM = _public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
RUN = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/alpha075_exact_replay_plan')

INITIAL = 82_012_495
FULL_CAP = 100_000_000
SKIP_ROWS = 530_944
BATCH_SIZE = 256
KNOWN_TAIL_CAP = 3_992_918
KNOWN_TAIL_FINAL = 3_992_800
SCHEDULE_TOTAL = 455
# Official fast/AoA names expected for Strict-Small submission tail.
OFFICIAL_TARGET_TOTALS = [90_000_000, 100_000_000]
# research's saved known checkpoints are tail-threshold names, not official names.
KNOWN_TAIL_THRESHOLDS = [1_000_000, 2_000_000, 3_000_000]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_step150_log() -> list[dict[str, Any]]:
    p = _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/training_log.jsonl')
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    remaining = FULL_CAP - INITIAL
    batches: list[dict[str, Any]] = []
    selected = 0
    current_words = 0
    current_count = 0
    batch_start_row = None
    first_row = None
    last_row = None
    n_rows = 0
    next_over = None
    known_crossings: dict[str, Any] = {}
    official_crossings: dict[str, Any] = {}

    known_targets = {t: f"tail_{t}" for t in KNOWN_TAIL_THRESHOLDS}
    official_targets = {t - INITIAL: f"chck_{t // 1_000_000}M" for t in OFFICIAL_TARGET_TOTALS}
    all_tail_targets = sorted(set(known_targets) | set(official_targets))

    with STREAM.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < SKIP_ROWS:
                continue
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch row={idx} words={words} split={len(text.split())}")
            if selected + words > remaining:
                next_over = {"row_index": idx, "words": words, "would_tail_words": selected + words, "would_total": INITIAL + selected + words}
                break
            if first_row is None:
                first_row = idx
            last_row = idx
            if current_count == 0:
                batch_start_row = idx
            n_rows += 1
            selected += words
            current_words += words
            current_count += 1
            if current_count == BATCH_SIZE:
                update = len(batches) + 1
                tail_words = selected
                total_words = INITIAL + tail_words
                rec = {
                    "update": update,
                    "source_row_start": batch_start_row,
                    "source_row_end": idx,
                    "batch_words": current_words,
                    "tail_main_words": tail_words,
                    "total_consumed_words": total_words,
                }
                batches.append(rec)
                for target in all_tail_targets:
                    if tail_words >= target:
                        key = (known_targets.get(target) or official_targets.get(target))
                        store = known_crossings if target in known_targets else official_crossings
                        store.setdefault(key, {"target_tail_words": target, "crossing": rec})
                current_count = 0
                current_words = 0
                batch_start_row = None

    if current_count:
        update = len(batches) + 1
        tail_words = selected
        total_words = INITIAL + tail_words
        rec = {
            "update": update,
            "source_row_start": batch_start_row,
            "source_row_end": last_row,
            "batch_words": current_words,
            "tail_main_words": tail_words,
            "total_consumed_words": total_words,
            "partial_batch_rows": current_count,
        }
        batches.append(rec)
        for target in all_tail_targets:
            if tail_words >= target:
                key = (known_targets.get(target) or official_targets.get(target))
                store = known_crossings if target in known_targets else official_crossings
                store.setdefault(key, {"target_tail_words": target, "crossing": rec})

    log_rows = read_step150_log()
    log_matches = []
    for a, b in zip(log_rows, batches):
        fields = ["update", "source_row_start", "source_row_end", "batch_words", "tail_main_words", "total_consumed_words"]
        mism = {k: (a.get(k), b.get(k)) for k in fields if a.get(k) != b.get(k)}
        if mism:
            log_matches.append({"update": a.get("update"), "mismatches": mism})
    known_final_batch = None
    for b in batches:
        if b["tail_main_words"] == KNOWN_TAIL_FINAL:
            known_final_batch = b
            break
    if known_final_batch is None:
        # research stopped before adding the next row/batch that would exceed its tail cap;
        # because it uses full batches, its final tail appears as update 101.
        known_final_batch = batches[len(log_rows) - 1] if log_rows else None

    known_sources = {
        "tail_1M": _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/chck_total_83012495w'),
        "tail_2M": _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/chck_total_84012495w'),
        "tail_3M": _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/chck_total_85012495w'),
        "tail_3992800_final86": _public_path('experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final'),
    }
    source_hashes = {k: {"path": rel(v), "model_safetensors_sha256": sha256_file(v / "model.safetensors"), "config_sha256": sha256_file(v / "config.json")} for k, v in known_sources.items()}

    out = {
        "status": "ALPHA075_TAIL_SCHEDULE_AUDIT",
        "purpose": "Plan exact 82M-anchor replay; do not resume optimizer from alpha0.75 final. Known 83-86 states must be reproduced before extending to 90/100.",
        "stream": rel(STREAM),
        "initial_consumed_words": INITIAL,
        "full_cap_words": FULL_CAP,
        "remaining_tail_word_cap": remaining,
        "skip_rows": SKIP_ROWS,
        "batch_size": BATCH_SIZE,
        "row_level_selected_tail_words": selected,
        "row_level_total_after_selected": INITIAL + selected,
        "row_level_rows": n_rows,
        "first_tail_row": first_row,
        "last_in_cap_row": last_row,
        "next_row_would_exceed": next_over,
        "num_batches_if_full_replay": len(batches),
        "schedule_total_original": SCHEDULE_TOTAL,
        "schedule_total_matches_full_batches": len(batches) == SCHEDULE_TOTAL,
        "known_step150_log_rows": len(log_rows),
        "known_step150_final_batch_from_audit": known_final_batch,
        "known_step150_prefix_batch_schedule_matches": not log_matches,
        "known_step150_prefix_mismatches": log_matches[:10],
        "known_tail_crossings": known_crossings,
        "official_target_crossings": official_crossings,
        "known_source_hashes": source_hashes,
        "tail_replay_policy": {
            "phase1_decision_test": "Run deterministic 82M->known86 replay in a fresh A01 dir, comparing tail_1M/2M/3M and final86 weights to these hashes. If any mismatch, stop and do not extend.",
            "phase2_extension": "Use the same deterministic algorithm from 82M through the full remaining tail with the original 455-step LR schedule; save official chck_90M at first row/batch crossing 90M and chck_100M at the final <=100M row boundary; materialize alpha0.75 only after training for evaluation.",
        },
    }
    out_path = _public_path('experiments/archive/representation_and_objectives/data/alpha075_exact_replay_plan/alpha075_tail_schedule_audit.json')
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = [
        "# research alpha0.75 exact tail replay schedule audit",
        "",
        f"Status: `{out['status']}`.",
        f"From the verified 82,012,495-word anchor, the whole-row remaining suffix contains {selected:,} words and ends at total {INITIAL + selected:,} before row {next_over['row_index'] if next_over else 'EOF'} would exceed 100M.",
        f"Full deterministic replay has {len(batches)} batches; original research schedule_total is {SCHEDULE_TOTAL}; match = {len(batches) == SCHEDULE_TOTAL}.",
        f"The first {len(log_rows)} batches match the known research training log exactly on row ranges, batch words, tail words, and total words: {not log_matches}.",
        f"Known final86 is update {known_final_batch['update'] if known_final_batch else None}, tail {known_final_batch['tail_main_words'] if known_final_batch else None}, total {known_final_batch['total_consumed_words'] if known_final_batch else None}.",
        f"Genuine chck_90M crossing: {official_crossings.get('chck_90M')}.",
        f"Genuine chck_100M final/crossing: {official_crossings.get('chck_100M')}.",
        "",
        "The next GPU action should be a guarded replay from the 82M anchor, not optimizer restart from alpha0.75 final. It must abort before extension if known 83-86 hashes are not reproduced.",
        f"JSON: `{rel(out_path)}`",
    ]
    (_public_path('research/documents/representation_and_objectives/data/alpha075_exact_replay_plan/alpha075_tail_schedule_audit.md')).write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(out_path), "num_batches": len(batches), "row_level_total": INITIAL + selected, "prefix_matches_step150": not log_matches, "official_crossings": official_crossings}, indent=2), flush=True)


if __name__ == "__main__":
    main()
