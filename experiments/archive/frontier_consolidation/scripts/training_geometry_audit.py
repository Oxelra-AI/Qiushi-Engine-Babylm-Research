#!/usr/bin/env python3
"""research: static training-geometry audit for dose and RoBERTa transfer arms.

This CPU/file-only audit records a small but important consequence of the repaired
MAX row-holdout pools: the MAX 100M streams contain more rows than the old 1x
streams at the same word exposure. With batch_size=256 this changes the number
of optimizer updates from 2529 to 2552. The DeBERTa MAX jobs already launched
with the research/253 lr_total_steps=2529 recipe; this is shared by MAX view and
MAX repeat, so it does not contaminate the semantic view-minus-repeat leg, but
comparisons to 0x/1x should remember that repeat-clean/source-admission includes
row/update geometry as part of the fixed-budget substitution coordinate.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import pathlib
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WS = USER_ROOT / "experiments/archive/frontier_consolidation"
OUT_DIR = WS / "data/training_geometry_audit"
BATCH = 256
LR_TOTAL_STEPS_USED_FOR_MAX = 2529

RUNS = {
    "clean0_80M": WS / "training/runs/complianttok_cleanqwen_seed43022_80M",
    "dose1_view_100M": WS / "training/runs/complianttok_reinvest_seed43022_r2",
    "dose1_repeat_100M": WS / "training/runs/full_p2c_c2p_abs_repeat_deberta100M_seed43022",
}
MAX_META = WS / "data/dose_2p64x_rowholdout_pools/dose2p64x_rowholdout_metadata.json"
ROBERTA_SCAFFOLD = WS / "data/roberta_maxdose_transfer_preflight/roberta_maxdose_transfer_scaffold.json"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_row(run_dir: pathlib.Path) -> dict[str, Any]:
    man = read_json(run_dir / "example_order_manifest.json")
    metrics = read_json(run_dir / "scientific_metrics.json")
    rows = int(man.get("num_consumed_examples", 0))
    words = int(man.get("selected_for_training_words", metrics.get("word_exposure", 0)))
    actual_steps = int(metrics.get("actual_training_steps", math.ceil(rows / BATCH)))
    return {
        "run_dir": str(run_dir),
        "rows": rows,
        "words": words,
        "mean_words_per_row": words / rows if rows else None,
        "batch_size": int(metrics.get("batch_size", BATCH) or BATCH),
        "ceil_rows_over_batch": math.ceil(rows / BATCH) if rows else None,
        "actual_training_steps_metrics": actual_steps,
        "lr_total_steps_metrics": metrics.get("lr_total_steps"),
        "checkpoint_words": (metrics.get("saved_checkpoints") or [{}])[0].get("target_word_exposure") if metrics.get("saved_checkpoints") else None,
        "loss_first": metrics.get("loss_first"),
        "loss_last": metrics.get("loss_last"),
    }


def max_rows(meta: dict[str, Any]) -> dict[str, Any]:
    audit = meta.get("audit", {})
    rows10 = int(audit.get("row_counts", {}).get("compact_view_dose2p64x", 0))
    words10 = int(audit.get("word_totals", {}).get("compact_view_dose2p64x", 0))
    rows100 = rows10 * 10
    words100 = words10 * 10
    return {
        "one_pass_rows": rows10,
        "one_pass_words": words10,
        "training_rows_100M": rows100,
        "training_words_100M": words100,
        "mean_words_per_row": words100 / rows100 if rows100 else None,
        "ceil_rows_over_batch": math.ceil(rows100 / BATCH) if rows100 else None,
        "lr_total_steps_used_by_step256_wrapper": LR_TOTAL_STEPS_USED_FOR_MAX,
        "extra_updates_beyond_lr_total_steps_if_trained_all_rows": math.ceil(rows100 / BATCH) - LR_TOTAL_STEPS_USED_FOR_MAX if rows100 else None,
    }


def lr_lambda(step: int, total: int, warmup_fraction: float = 0.06, cycles: float = 0.5) -> float:
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return float(step) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total - warmup))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * cycles * progress)))


def lr_tail_summary(total: int, actual_steps: int) -> dict[str, Any]:
    # Approximate scheduler multipliers at optimizer step indices 1..actual_steps.
    vals = [lr_lambda(i, total) for i in range(1, actual_steps + 1)]
    beyond = vals[total:] if actual_steps > total else []
    return {
        "configured_total": total,
        "actual_steps": actual_steps,
        "warmup_steps": max(1, int(total * 0.06)),
        "sum_lr_multipliers_all_steps": sum(vals),
        "mean_lr_multiplier_all_steps": sum(vals) / len(vals) if vals else None,
        "extra_steps_after_configured_total": max(0, actual_steps - total),
        "extra_tail_sum_lr_multipliers": sum(beyond),
        "extra_tail_max_lr_multiplier": max(beyond) if beyond else 0.0,
        "last_five_multipliers": vals[-5:] if vals else [],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {name: run_row(path) for name, path in RUNS.items()}
    meta = read_json(MAX_META)
    max_geom = max_rows(meta)
    roberta = read_json(ROBERTA_SCAFFOLD) if ROBERTA_SCAFFOLD.exists() else None
    payload = {
        "status": "TRAINING_GEOMETRY_AUDIT",
        "created_utc": now(),
        "runs": runs,
        "max_static_geometry": max_geom,
        "roberta_scaffold_geometry": roberta.get("stream_rows_words") if isinstance(roberta, dict) else None,
        "lr_horizon_summary_for_max_if_lr_total_steps_2529": lr_tail_summary(LR_TOTAL_STEPS_USED_FOR_MAX, int(max_geom["ceil_rows_over_batch"] or 0)),
        "scientific_interpretation": {
            "semantic_leg": "MAX view and MAX repeat have identical rows, words, batch count, and LR horizon; view-minus-repeat remains the cleanest estimate of semantic re-expression at MAX dose.",
            "source_leg": "MAX repeat-minus-clean0 and MAX view-minus-clean0 compare different row/update geometries as well as different text. Interpret this as the value of admitting the matched MAX source-packet stream under the fixed training recipe, not a pure source-content scalar.",
            "verification_update": "Do not reject the launched MAX trainings merely because actual_training_steps is 2552 rather than the 2529 old-stream value. The run wrapper passed lr_total_steps=2529 by design, but the exact 100M MAX streams contain 653130 rows, so batch256 training should execute 2552 optimizer updates.",
            "roberta_policy": "The RoBERTa MAX-dose scaffold currently mirrors the DeBERTa MAX schedule with lr_total_steps=2529; its dry-run records 2552 batch updates for all three candidate arms, so any later RoBERTa pair/trio shares this geometry internally.",
        },
    }
    out_json = OUT_DIR / "training_geometry_audit.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research training-geometry audit",
        "",
        "Static CPU/file-only audit; no task status was queried and no training/evaluation was started.",
        "",
        "## Main correction",
        "The repaired MAX 2.64x streams have 653,130 rows at 100M words. With batch256 this implies 2,552 optimizer updates, while the old 1x 100M streams have 647,400 rows and 2,529 updates.",
        "",
        "Therefore the MAX training verification should expect 2,552 actual updates if all rows are consumed, while checking that `lr_total_steps` was the intended 2,529 recipe value.",
        "",
        "## Interpretation",
        payload["scientific_interpretation"]["semantic_leg"],
        payload["scientific_interpretation"]["source_leg"],
        "",
        f"JSON: `{out_json}`",
    ]
    (OUT_DIR / "training_geometry_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": str(out_json), "max_updates": max_geom["ceil_rows_over_batch"], "extra_updates_after_2529": max_geom["extra_updates_beyond_lr_total_steps_if_trained_all_rows"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
