#!/usr/bin/env python3
"""research: audit paired-alignment pool and exposure accounting.

Purpose: audit exact word-budget / epoch accounting before a Wave 2 comparison.
The audit distinguishes
mechanism-screen evidence from official-compatible training exposure.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

ROOT = Path("experiments/archive/compact_experience")
DATA = ROOT / "data/paired_alignment"
RUNS = ROOT / "training/runs"
OUT = ROOT / "data/paired_alignment_accounting_audit.json"
NOTE = (ROOT.parents[2] / 'research/notes/compact_experience/paired_alignment_accounting_audit.md')
SUMMARY = DATA / "screen_summary.json"

ARMS = ["aligned", "mismatched", "single_repeat", "single_orig"]


def count_jsonl_words(path: Path) -> Dict[str, Any]:
    rows = 0
    words = 0
    sources: Dict[str, int] = {}
    if not path.exists():
        return {"exists": False}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            w = int(obj.get("words", len(str(obj.get("text", "")).split())))
            rows += 1
            words += w
            src = str(obj.get("source", ""))
            sources[src] = sources.get(src, 0) + w
    return {"exists": True, "rows": rows, "words": words, "sources": sources}


def read_metrics(arm: str) -> Dict[str, Any]:
    p = RUNS / f"step017_{arm}_100M_seed43" / "scientific_metrics.json"
    if not p.exists():
        return {"exists": False, "path": str(p)}
    m = json.loads(p.read_text(encoding="utf-8"))
    keep = {
        "exists": True,
        "path": str(p),
        "word_exposure": m.get("word_exposure"),
        "actual_training_steps": m.get("actual_training_steps"),
        "loss_first": m.get("loss_first"),
        "loss_last": m.get("loss_last"),
        "example_jsonl_total_words": m.get("example_jsonl_total_words"),
        "source_words_consumed": m.get("source_words_consumed"),
        "saved_checkpoint_names": [x.get("name") for x in m.get("saved_checkpoints", [])],
    }
    return keep


def main() -> None:
    screen = json.loads(SUMMARY.read_text(encoding="utf-8"))
    arms = {}
    for arm in ARMS:
        base_info = screen.get("arms", {}).get(arm, {})
        base_words = int(base_info.get("total_words", 0) or 0)
        base_rows = int(base_info.get("rows", 0) or 0)
        expanded_path = DATA / "training_expanded" / f"{arm}_100M.jsonl"
        expanded = count_jsonl_words(expanded_path)
        metrics = read_metrics(arm)
        exposure = metrics.get("word_exposure") or (expanded.get("words") if expanded.get("exists") else None)
        exact_10_epoch_words = base_words * 10 if base_words else None
        pass_count = (float(exposure) / float(base_words)) if base_words and exposure is not None else None
        extra_beyond_10_passes = (int(exposure) - exact_10_epoch_words) if exposure is not None and exact_10_epoch_words is not None else None
        missing_to_10m_pool = 10_000_000 - base_words if base_words else None
        arms[arm] = {
            "base_pool_words": base_words,
            "base_pool_rows": base_rows,
            "missing_to_10m_pool_words": missing_to_10m_pool,
            "expanded_training_file": str(expanded_path),
            "expanded": expanded,
            "metrics": metrics,
            "exposure_words": exposure,
            "exact_10_epoch_words_for_this_pool": exact_10_epoch_words,
            "effective_passes_over_base_pool": pass_count,
            "extra_words_beyond_10_passes": extra_beyond_10_passes,
            "literal_epoch_constraint_status": (
                "within_10_passes" if extra_beyond_10_passes is not None and extra_beyond_10_passes <= 0 else
                "exceeds_10_passes" if extra_beyond_10_passes is not None else
                "unknown"
            ),
        }
    critical_ok = (
        arms["aligned"]["base_pool_words"] == arms["mismatched"]["base_pool_words"] == 9_999_840
        and arms["aligned"]["exposure_words"] == arms["mismatched"]["exposure_words"] == 100_000_000
    )
    payload = {
        "status": "PAIRED_ALIGNMENT_ACCOUNTING_AUDIT_DONE",
        "interpretation": {
            "critical_aligned_mismatched_mechanism_screen_matched": critical_ok,
            "do_not_launch_existing_wave2_reason": "single_orig base pool is 9,551,200 words, so the existing 100M expanded file corresponds to about 10.47 passes, not <=10 passes; aligned/mismatched/single_repeat are also 1600 words over 10 exact passes because their base pools are 9,999,840 words rather than exactly 10,000,000.",
            "needed_repair": "Create exactly 10,000,000-word legal base pools before any official-candidate 100M training, or set max exposure to 10*base_pool_words and handle final checkpoint naming explicitly. For mechanism follow-up, add a coherent same-topic non-synonymous adjacency control before interpreting ALIGNED-MISMATCHED as paraphrase-specific rewrite correspondence.",
        },
        "arms": arms,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — Paired-alignment accounting audit",
        "",
        f"JSON: `{OUT}`",
        "",
        "The accounting audit identifies why the existing script does not support a valid Wave 2 comparison.",
        "",
        "## Arm accounting",
        "",
        "| arm | base words | missing to 10M | exposure words | effective passes | extra beyond 10 passes | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        a = arms[arm]
        lines.append(
            f"| {arm} | {a['base_pool_words']} | {a['missing_to_10m_pool_words']} | {a['exposure_words']} | {a['effective_passes_over_base_pool']:.6f} | {a['extra_words_beyond_10_passes']} | {a['literal_epoch_constraint_status']} |"
        )
    lines += [
        "",
        "## Consequences",
        "",
        "- The Wave-1 ALIGNED/MISMATCHED comparison is still a matched mechanism screen: both use 9,999,840-word pools and 100,000,000-word exposure, so both are 10 full passes plus 1,600 words.",
        "- The same runs should not be treated as official-candidate compliant under a literal ≤10 epoch interpretation until rebuilt with exactly 10,000,000-word base pools or rerun at exactly 99,998,400 exposure with explicit final-checkpoint handling.",
        "- Existing Wave 2 must not be launched: SINGLE_ORIG has only 9,551,200 base words, so 100,000,000 exposure would be about 10.47 passes and would confound the control as well as violate the literal epoch limit.",
        "- The ALIGNED–MISMATCHED contrast tests meaning-related/coherent adjacency versus scrambled local adjacency. It does not by itself separate paraphrase/rewrite correspondence from topical coherence; a same-topic non-synonymous adjacency control is needed next.",
        "",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
