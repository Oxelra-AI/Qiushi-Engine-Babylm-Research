#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
BASE = ROOT / "data/wwm100m_8of9_coordinate.json"
BERT34 = ROOT / "data/bert8x512_available_coordinate.json"
DEB = ROOT / "data/debertav2_b256_available_coordinate.json"
OUT = ROOT / "data/34m_coordinate_comparison.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/34m_coordinate_comparison.md')

COLUMNS = [
    ("blimp", "BLiMP"),
    ("supplement", "Supplement"),
    ("entity_tracking", "Entity"),
    ("comps", "COMPS"),
    ("global_piqa_parallel", "GlobalPIQA parallel"),
    ("global_piqa_nonparallel", "GlobalPIQA nonparallel"),
    ("global_piqa_mean", "GlobalPIQA mean"),
    ("reading_eye_tracking", "Reading eye"),
    ("reading_self_paced", "Reading self-paced"),
]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract(d):
    s = dict(d.get("scores", {}))
    der = d.get("derived_columns", {})
    if "GlobalPIQA_mean_parallel_nonparallel" in der:
        s["global_piqa_mean"] = der["GlobalPIQA_mean_parallel_nonparallel"]
    elif "global_piqa_parallel" in s and "global_piqa_nonparallel" in s:
        s["global_piqa_mean"] = (s["global_piqa_parallel"] + s["global_piqa_nonparallel"]) / 2.0
    return s


def main():
    base = extract(load(BASE)); bert = extract(load(BERT34)); deb = extract(load(DEB))
    rows = []
    for key, label in COLUMNS:
        if key not in base or key not in bert or key not in deb:
            continue
        rows.append({
            "key": key,
            "label": label,
            "bert10p7m_wwm": base[key],
            "bert34m_batch512": bert[key],
            "debertav2_34m_batch256": deb[key],
            "delta_capacity_bert34_minus_bert10p7": bert[key] - base[key],
            "delta_deberta_b256_minus_bert34_b512": deb[key] - bert[key],
            "delta_deberta_b256_minus_bert10p7": deb[key] - base[key],
        })
    payload = {
        "status": "comparison_available_columns_only",
        "baseline_file": str(BASE),
        "bert34_file": str(BERT34),
        "debertav2_file": str(DEB),
        "interpretation_warning": "DeBERTa-v2 used batch_size=256 and lr_total_steps=2442 after batch-512 OOM; BERT34 used batch_size=512 and 1221 steps. DeBERTa-v2 minus BERT34 is not a clean pure architecture effect.",
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — 34M coordinate comparison", "",
        f"Evidence JSON: `{OUT}`", "",
        "DeBERTa-v2 is a batch-256 rescue run after batch-512 OOM; differences against BERT34 mix architecture/package with update geometry.", "",
        "| column | 10.7M BERT | 34M BERT b512 | DeBERTa-v2 b256 | cap Δ | DeBERTa-b256 Δ vs BERT34 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['label']} | {r['bert10p7m_wwm']:.2f} | {r['bert34m_batch512']:.2f} | {r['debertav2_34m_batch256']:.2f} | {r['delta_capacity_bert34_minus_bert10p7']:+.2f} | {r['delta_deberta_b256_minus_bert34_b512']:+.2f} |")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status":"COMPARISON_DONE","out":str(OUT),"rows":rows}, indent=2))

if __name__ == "__main__":
    main()
