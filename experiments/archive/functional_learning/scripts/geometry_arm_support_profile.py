#!/usr/bin/env python3
"""research: full-prefix support profile for the preservation-geometry arm.

Counts, without model forwards, how many preservation target positions would be
used by clean ordinary-WWM parent KL and by the proposed dense-corrupted
non-label parent KL across the same 80-update unchanged-Qwen prefix. This makes a
future full arm interpretable as a joint input-rendering and target-support
change when appropriate.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import statistics
import sys
import time
from collections import Counter
from typing import Any, Dict, List

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import real_stream_train_weighted as s64  # noqa: E402
import corrected_bridge_trainer as bridge  # noqa: E402
import preservation_geometry_matched_diagnostic as geom  # noqa: E402

DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/geometry_arm_support_profile')


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def summarise(vals: List[int]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "sum": 0, "mean": 0.0, "median": 0.0, "min": None, "max": None}
    return {
        "n": len(vals),
        "sum": int(sum(vals)),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "min": int(min(vals)),
        "max": int(max(vals)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tail-jsonl", type=pathlib.Path, default=s64.DEFAULT_TAIL)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--max-updates", type=int, default=80)
    ap.add_argument("--words-per-update", type=int, default=39533)
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--train-seed", type=int, default=62064)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--focus-prob", type=float, default=0.35)
    ap.add_argument("--max-focus-groups-per-row", type=int, default=16)
    ap.add_argument("--max-dense-mask-groups", type=int, default=128)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, prefix_info = s64.load_prefix(pathlib.Path(args.tail_jsonl), int(args.max_updates), int(args.words_per_update))
    macros = geom.split_macros(rows, int(args.max_updates), int(args.words_per_update))
    tokenizer = bridge.AutoTokenizer.from_pretrained(str(bridge.PARENT_PATH), local_files_only=True, use_fast=True)
    wgb = bridge.WordGroupBuilder(tokenizer)

    total = Counter()
    macro_records: List[Dict[str, Any]] = []
    row_common_vals: List[int] = []
    row_ord_vals: List[int] = []
    row_dense_vals: List[int] = []
    for macro in macros:
        # Reuse research local constructor. It needs only namespace fields below.
        ns = argparse.Namespace(
            seq_length=int(args.seq_length), train_seed=int(args.train_seed), mask_prob=float(args.mask_prob),
            focus_prob=float(args.focus_prob), max_focus_groups_per_row=int(args.max_focus_groups_per_row),
            max_dense_mask_groups=int(args.max_dense_mask_groups),
        )
        _rends, st = geom.build_preservation_renderings(macro["rows"], tokenizer, wgb, ns)
        counts = Counter(st["counts"])
        total.update(counts)
        total["selected_rows"] += len(macro["rows"])
        total["selected_words"] += int(macro["words"])
        rec = {"update_index0": int(macro["update_index0"]), "rows": len(macro["rows"]), "words": int(macro["words"]), "qwen_rows": int(macro["qwen_rows"]), **dict(counts)}
        macro_records.append(rec)
        for rr in st.get("row_records_head", []):
            # Only first 12 per macro are retained by the imported helper, so row summaries below are approximate samples.
            row_common_vals.append(int(rr.get("common_targets", 0)))
            row_ord_vals.append(int(rr.get("ordinary_targets", 0)))
            row_dense_vals.append(int(rr.get("dense_nonlabel_targets", 0)))

    total_d = dict(total)
    ratios = {
        "dense_nonlabel_over_ordinary_full_targets": float(total_d.get("dense_nonlabel_targets", 0) / max(1, total_d.get("ordinary_full_targets", 0))),
        "common_over_ordinary_full_targets": float(total_d.get("common_targets", 0) / max(1, total_d.get("ordinary_full_targets", 0))),
        "common_over_dense_nonlabel_targets": float(total_d.get("common_targets", 0) / max(1, total_d.get("dense_nonlabel_targets", 0))),
        "dense_sparse_label_over_dense_masked_tokens": float(total_d.get("dense_sparse_label_targets", 0) / max(1, total_d.get("dense_masked_tokens", 0))),
        "dense_nonlabel_over_dense_masked_tokens": float(total_d.get("dense_nonlabel_targets", 0) / max(1, total_d.get("dense_masked_tokens", 0))),
    }
    result = {
        "status": "GEOMETRY_ARM_SUPPORT_PROFILE_DONE",
        "created_utc": now(),
        "script": rel(_public_path('experiments/archive/functional_learning/scripts/geometry_arm_support_profile.py')),
        "purpose": "Count target support for ordinary full-row KL and dense-corrupted non-label KL before any full preservation-geometry training run.",
        "prefix_info": prefix_info,
        "settings": {
            "max_updates": int(args.max_updates),
            "words_per_update": int(args.words_per_update),
            "seq_length": int(args.seq_length),
            "train_seed": int(args.train_seed),
            "mask_prob": float(args.mask_prob),
            "focus_prob": float(args.focus_prob),
            "max_focus_groups_per_row": int(args.max_focus_groups_per_row),
            "max_dense_mask_groups": int(args.max_dense_mask_groups),
        },
        "support_totals": total_d,
        "support_ratios": ratios,
        "macro_summaries": {
            "ordinary_full_targets_by_macro": summarise([int(r.get("ordinary_full_targets", 0)) for r in macro_records]),
            "dense_nonlabel_targets_by_macro": summarise([int(r.get("dense_nonlabel_targets", 0)) for r in macro_records]),
            "common_targets_by_macro": summarise([int(r.get("common_targets", 0)) for r in macro_records]),
            "qwen_rows_by_macro": summarise([int(r.get("qwen_rows", 0)) for r in macro_records]),
        },
        "sampled_row_summaries": {
            "note": "Only the first 12 row records per macro are retained by the imported builder; macro totals above are complete.",
            "ordinary_targets_per_sampled_row": summarise(row_ord_vals),
            "dense_nonlabel_targets_per_sampled_row": summarise(row_dense_vals),
            "common_targets_per_sampled_row": summarise(row_common_vals),
        },
        "macro_records": macro_records,
        "interpretation": "The proposed dense-corrupted non-label KL changes both prediction-state rendering and target support relative to clean ordinary-WWM KL if run as a full arm. Common-support gradient measurements are needed to understand rendering geometry, while these totals describe the resource/support side of the intervention.",
    }
    out_json = args.out_dir / "geometry_arm_support_profile.json"
    out_md = args.out_dir / "geometry_arm_support_profile.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research geometry-arm support profile", "",
        f"Created: `{result['created_utc']}`", "",
        "## Full 80-update support totals", "",
        "| quantity | value |", "|---|---:|",
    ]
    for k in ["qwen_rows", "ordinary_full_targets", "dense_masked_tokens", "dense_sparse_label_targets", "dense_nonlabel_targets", "common_targets", "ordinary_full_examples", "dense_nonlabel_full_examples", "ordinary_common_examples"]:
        lines.append(f"| {k} | {total_d.get(k, '')} |")
    lines += ["", "## Ratios", "", "| ratio | value |", "|---|---:|"]
    for k, v in ratios.items():
        lines.append(f"| {k} | {v:.6f} |")
    lines += ["", "## Macro summaries", "", json.dumps(result["macro_summaries"], indent=2, ensure_ascii=False), "", result["interpretation"]]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md), "support_ratios": ratios}, indent=2), flush=True)


if __name__ == "__main__":
    main()
