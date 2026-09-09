#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
BASE = ROOT / "data/debertav2_b256_available_coordinate.json"
OFF40 = ROOT / "data/official40k_available_coordinate.json"
TRAIN_BASE = ROOT / "data/official40k_accum_training_validation.json"
OUT = ROOT / "data/official40k_vs_baseline16k_available_comparison.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/official40k_vs_baseline16k_available_comparison.md')
LEADER = {
    "blimp": 67.20,
    "supplement": 56.01,
    "entity_tracking": 28.45,
    "comps": 53.57,
    "global_piqa_mean": 39.67,
    "reading_mean": 5.42,
}
COLS = [
    ("blimp", "BLiMP"),
    ("supplement", "Supplement"),
    ("entity_tracking", "Entity"),
    ("comps", "COMPS"),
    ("global_piqa_parallel", "GlobalPIQA parallel"),
    ("global_piqa_nonparallel", "GlobalPIQA nonparallel"),
    ("global_piqa_mean", "GlobalPIQA mean"),
    ("reading_eye_tracking", "Reading eye"),
    ("reading_self_paced", "Reading self-paced"),
    ("reading_mean", "Reading mean"),
]

def load(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))

def extract(d):
    s = dict(d["scores"])
    der = d.get("derived_columns", {})
    if "GlobalPIQA_mean_parallel_nonparallel" in der:
        s["global_piqa_mean"] = der["GlobalPIQA_mean_parallel_nonparallel"]
    else:
        s["global_piqa_mean"] = (s["global_piqa_parallel"] + s["global_piqa_nonparallel"]) / 2.0
    if "Reading_mean_eye_self_paced" in der:
        s["reading_mean"] = der["Reading_mean_eye_self_paced"]
    else:
        s["reading_mean"] = (s["reading_eye_tracking"] + s["reading_self_paced"]) / 2.0
    return s

def main():
    base = extract(load(BASE))
    off = extract(load(OFF40))
    train = load(TRAIN_BASE)
    rows = []
    for key, label in COLS:
        if key not in base or key not in off:
            continue
        row = {
            "key": key,
            "label": label,
            "baseline16k": base[key],
            "official40k": off[key],
            "delta_40k_minus_16k": off[key] - base[key],
        }
        if key in LEADER:
            row["leader"] = LEADER[key]
            row["official40k_minus_leader"] = off[key] - LEADER[key]
        rows.append(row)
    payload = {
        "status": "official40k_available_columns_comparison",
        "baseline16k_file": str(BASE),
        "official40k_file": str(OFF40),
        "training_validation_file": str(TRAIN_BASE),
        "interpretation_warning": "official40k run preserves DeBERTa 8x480 non-embedding capacity but increases total parameters via embeddings and uses microbatch128 accum2 to approximate effective batch256.",
        "training_summary": {
            "official40k_params": train["parameter_count"],
            "official40k_embedding_params": train["embedding_parameter_count"],
            "official40k_non_embedding_params": train["non_embedding_parameter_count"],
            "official40k_kept_tokens_per_word": train["tokenization_coupling_summary"]["kept_tokens_per_whitespace_word"],
            "official40k_truncated_fraction": train["tokenization_coupling_summary"]["truncated_example_fraction"],
            "loss_first": train["loss_first"],
            "loss_last": train["loss_last"],
        },
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — official40k vs baseline16k DeBERTa available-column comparison", "",
        f"Evidence JSON: `{OUT}`", "",
        "official40k preserves the 8x480 non-embedding backbone but increases total parameters through embeddings and was trained with microbatch128 accumulation2.", "",
        "| column | baseline16k | official40k | delta | leader | 40k-leader |", "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        leader = r.get("leader")
        gap = r.get("official40k_minus_leader")
        lines.append(f"| {r['label']} | {r['baseline16k']:.2f} | {r['official40k']:.2f} | {r['delta_40k_minus_16k']:+.2f} | {leader:.2f} | {gap:+.2f} |" if leader is not None else f"| {r['label']} | {r['baseline16k']:.2f} | {r['official40k']:.2f} | {r['delta_40k_minus_16k']:+.2f} | — | — |")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "OFFICIAL40K_COMPARISON_DONE", "out": str(OUT), "rows": rows}, indent=2))

if __name__ == "__main__":
    main()
