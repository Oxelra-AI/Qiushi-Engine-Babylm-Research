#!/usr/bin/env python3
"""Inspect official prediction payload shapes from prior complete cheap-column evals.

This is CPU/file-only and prepares robust item-level comparison code for the
running causal compact-vs-repeat selected evaluations.
"""
from __future__ import annotations
import json
import pathlib

PATHS = {
    "BLiMP": "experiments/archive/frontier_consolidation/data/scale1p75_split_eval/chck_80M/BLiMP/eval/official_outputs/scale1p75_chck_80M_BLiMP/BLiMP/chck_80M/full_scale1p75_chck_80M_BLiMP_BLiMP/zero_shot/mlm/blimp/blimp_filtered/predictions.json",
    "Supplement": "experiments/archive/frontier_consolidation/data/scale1p75_split_eval/chck_80M/Supplement/eval/official_outputs/scale1p75_chck_80M_Supplement/Supplement/chck_80M/full_scale1p75_chck_80M_Supplement_Supplement/zero_shot/mlm/blimp/supplement_filtered/predictions.json",
    "EWoK": "experiments/archive/frontier_consolidation/data/scale1p75_split_eval/chck_80M/EWoK/eval/official_outputs/scale1p75_chck_80M_EWoK/EWoK/chck_80M/full_scale1p75_chck_80M_EWoK_EWoK/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "Entity": "experiments/archive/frontier_consolidation/data/scale1p75_split_eval/chck_80M/Entity/eval/official_outputs/scale1p75_chck_80M_Entity/Entity/chck_80M/full_scale1p75_chck_80M_Entity_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json",
    "COMPS": "experiments/archive/frontier_consolidation/data/scale1p75_split_eval/chck_80M/COMPS/eval/official_outputs/scale1p75_chck_80M_COMPS/COMPS/chck_80M/full_scale1p75_chck_80M_COMPS_COMPS/zero_shot/mlm/comps/comps/predictions.json",
    "GlobalPIQA_parallel": "experiments/archive/frontier_consolidation/data/scale1p75_split_eval/chck_80M/GlobalPIQA_parallel/eval/official_outputs/scale1p75_chck_80M_GP_parallel/GlobalPIQA_parallel/chck_80M/full_scale1p75_chck_80M_GP_parallel_GlobalPIQA_parallel/zero_shot/mlm/global_piqa_parallel/global_piqa_parallel/predictions.json",
    "GlobalPIQA_nonparallel": "experiments/archive/frontier_consolidation/data/scale1p75_split_eval/chck_80M/GlobalPIQA_nonparallel/eval/official_outputs/scale1p75_chck_80M_GP_nonparallel/GlobalPIQA_nonparallel/chck_80M/full_scale1p75_chck_80M_GP_nonparallel_GlobalPIQA_nonparallel/zero_shot/mlm/global_piqa_nonparallel/global_piqa_nonparallel/predictions.json",
    "Reading": "experiments/archive/frontier_consolidation/data/scale1p75_split_eval/chck_80M/Reading/eval/official_outputs/scale1p75_chck_80M_Reading/Reading/chck_80M/full_scale1p75_chck_80M_Reading_Reading/zero_shot/mlm/reading/predictions.json",
}
OUT_DIR = pathlib.Path("experiments/archive/frontier_consolidation/data/prediction_shape_inspection")

def preview(obj, max_chars=2000):
    return json.dumps(obj, ensure_ascii=False)[:max_chars]

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    lines = ["# research prediction shape inspection\n\n"]
    for name, rel in PATHS.items():
        path = pathlib.Path(rel)
        rec = {"column": name, "path": rel, "exists": path.exists(), "size": path.stat().st_size if path.exists() else None}
        if not path.exists():
            rows.append(rec)
            lines.append(f"## {name}\nMissing: `{rel}`\n\n")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rec["top_type"] = type(data).__name__
        if isinstance(data, list):
            items = data
            rec["n_items"] = len(items)
            rec["container"] = "list"
        elif isinstance(data, dict):
            rec["top_keys"] = sorted(list(data.keys()))[:50]
            items = []
            for k, v in data.items():
                if isinstance(v, list):
                    rec["container"] = f"dict.{k}"
                    rec["n_items"] = len(v)
                    items = v
                    break
            if not items:
                rec["container"] = "dict-self"
                rec["n_items"] = 1
                items = [data]
        else:
            items = [data]
            rec["container"] = "scalar"
            rec["n_items"] = 1
        sample = items[:3]
        rec["sample_keys"] = [sorted(list(x.keys())) if isinstance(x, dict) else type(x).__name__ for x in sample]
        rec["sample_preview"] = [preview(x, 1200) for x in sample]
        rows.append(rec)
        lines.append(f"## {name}\n\n")
        lines.append(f"Path: `{rel}`\n\nType: {rec['top_type']}; container: {rec.get('container')}; n_items: {rec.get('n_items')}.\n\n")
        lines.append("Sample keys:\n\n")
        for sk in rec["sample_keys"]:
            lines.append(f"- `{sk}`\n")
        lines.append("\nFirst sample preview:\n\n```json\n" + rec["sample_preview"][0] + "\n```\n\n")
    out_json = OUT_DIR / "prediction_shape_inspection.json"
    out_md = (OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/prediction_shape_inspection/prediction_shape_inspection.md')
    out_json.write_text(json.dumps({"status":"PREDICTION_SHAPE_INSPECTION", "columns": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text("".join(lines) + f"\nJSON: `{out_json}`\n", encoding="utf-8")
    print(json.dumps({"status":"PREDICTION_SHAPE_INSPECTION", "out_json":str(out_json), "out_md":str(out_md)}, indent=2), flush=True)

if __name__ == "__main__":
    main()
