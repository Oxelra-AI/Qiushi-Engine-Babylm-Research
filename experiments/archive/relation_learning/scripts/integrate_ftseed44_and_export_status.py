#!/usr/bin/env python3
"""research integrate delivered clean-preservation ftseed44 and export status.

This is a research-facing consolidation, not final writing.  It reads the canonical
same-coordinate repaired-loading comparison and the completed downstream
fine-tuning seed44 SuperGLUE summary for clean-preservation seed62064, recomputes
an alternate Overall holding all zero-shot/Reading/AoA components fixed, and checks
that the v5 export package exists with key evidence files.
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

ROOT = _public_path('.')
OUT = _public_path('experiments/archive/relation_learning/data/ftseed44_export_status')
CANON = _public_path('experiments/archive/functional_learning/data/same_coordinate_all_candidates_after_clean65_superglue/same_coordinate_all_candidates.json')
SEED44_CLEAN64 = _public_path('experiments/archive/relation_learning/data/clean_pres62064_ftseed44_superglue/clean_pres62064_u0080_ftseed44/faithful_superglue_seeded_summary.json')
EXPORT_DIR = _public_path('experiments/archive/functional_learning/data/v5_candidate_export')
A01_VERIFIER = _public_path('experiments/archive/functional_learning/data/v5_export_evidence_verifier/v5_export_evidence_verifier.json')
A01_PROVENANCE = _public_path('research/documents/functional_learning/data/v5_candidate_provenance/v5_candidate_provenance.md')
COMPONENTS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "SuperGLUE", "GlobalPIQA", "Reading", "AoA"]


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_file(path: pathlib.Path, block: int = 1 << 20) -> str | None:
    if not path.is_file():
        return None
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_map(task_records: list[dict[str, Any]]) -> dict[str, float]:
    return {str(r["task"]): float(r["primary_score"]) for r in task_records}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    canon = read_json(CANON)
    seed44 = read_json(SEED44_CLEAN64)
    models = canon["models"]
    clean64 = models["clean_pres_lambda1_eval_seed62064"]
    clean65 = models["clean_pres_lambda1_eval_seed62065"]
    coherent = models["coherent86"]
    exact_ms = models.get("densemask_sparselabel_seed62064")
    comp_clean64 = dict(clean64["overall_computation"]["components"])
    comp_seed44 = dict(comp_clean64)
    comp_seed44["SuperGLUE"] = float(seed44["superglue_primary_metric_mean"])
    overall_seed44 = sum(float(comp_seed44[c]) for c in COMPONENTS) / len(COMPONENTS)
    overall_clean64_seed42 = float(clean64["overall_computation"]["Overall"])
    overall_coherent = float(coherent["overall_computation"]["Overall"])
    overall_clean65 = float(clean65["overall_computation"]["Overall"])
    sg_clean64_seed42 = float(clean64["superglue"]["score"])
    sg_seed44 = float(seed44["superglue_primary_metric_mean"])
    task_seed44 = metric_map(seed44["task_records"])
    task_clean64_seed42 = {str(r["task"]): float(r["score"]) for r in clean64["superglue"]["primary_metric_details"]}
    task_deltas = {t: task_seed44[t] - task_clean64_seed42[t] for t in sorted(set(task_seed44) & set(task_clean64_seed42))}
    record_file_checks = []
    for rec in seed44["task_records"]:
        for key in ["results_txt", "predictions", "log"]:
            p = ROOT / rec[key]
            record_file_checks.append({"task": rec["task"], "kind": key, "path": rec[key], "exists": p.is_file(), "size": p.stat().st_size if p.is_file() else None, "sha256": sha256_file(p) if p.is_file() and key == "results_txt" else None})
    export_entries = []
    if EXPORT_DIR.exists():
        for p in sorted(EXPORT_DIR.rglob("*")):
            if p.is_file():
                export_entries.append({"path": rel(p), "size": p.stat().st_size, "sha256": sha256_file(p) if p.stat().st_size < 200_000_000 else None})
            elif p.is_dir():
                export_entries.append({"path": rel(p), "type": "dir"})
    payload = {
        "status": "FTSEED44_EXPORT_STATUS_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canonical_coordinate": canon.get("coordinate"),
        "inputs": {
            "canonical_same_coordinate_json": rel(CANON),
            "clean64_seed44_superglue_summary": rel(SEED44_CLEAN64),
            "a01_export_dir": rel(EXPORT_DIR),
            "a01_verifier": rel(A01_VERIFIER),
            "a01_provenance_md": rel(A01_PROVENANCE),
        },
        "clean64_seed44": {
            "superglue": sg_seed44,
            "superglue_seed44_minus_seed42": sg_seed44 - sg_clean64_seed42,
            "overall_using_seed44_superglue_and_fixed_other_components": overall_seed44,
            "overall_seed44_minus_clean64_seed42_overall": overall_seed44 - overall_clean64_seed42,
            "overall_seed44_minus_faithful_coherent86_seed42": overall_seed44 - overall_coherent,
            "overall_seed44_minus_clean65_seed42": overall_seed44 - overall_clean65,
            "components_used": comp_seed44,
            "task_primary_scores_seed44": task_seed44,
            "task_primary_scores_clean64_seed42": task_clean64_seed42,
            "task_deltas_seed44_minus_clean64_seed42": task_deltas,
        },
        "canonical_reference": {
            "coherent86_overall_seed42": overall_coherent,
            "clean64_overall_seed42": overall_clean64_seed42,
            "clean65_overall_seed42": overall_clean65,
            "exact_ms_present": exact_ms is not None,
            "exact_ms_complete": bool((exact_ms or {}).get("complete_same_coordinate", False)),
            "historical_stripped_coherent_overall": (canon.get("historical_platform_records") or {}).get("coherent86_historical_platform_style_overall", {}).get("Overall"),
        },
        "seed44_output_file_checks": record_file_checks,
        "all_seed44_records_complete": all(bool(x["exists"]) and int(x.get("size") or 0) > 0 for x in record_file_checks),
        "export_package": {
            "exists": EXPORT_DIR.is_dir(),
            "file_or_dir_count_listed": len(export_entries),
            "entries": export_entries,
        },
        "interpretation": "Seed44 SuperGLUE for clean-preservation seed62064 is higher than the seed42 repaired coordinate; holding zero-shot/Reading/AoA fixed gives a larger alternate Overall, so this downstream-seed repeat does not weaken the seed62064 export endpoint. It is not a new submitted coordinate because the official comparison uses the fixed seed42 repaired table; it is stability evidence.",
    }
    (_public_path('experiments/archive/relation_learning/data/ftseed44_export_status/summary.json')).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = ["# research ftseed44 and export status", "", "This note consolidates a completed downstream SuperGLUE repeat and the export-package location.", "", "## Clean-preservation seed62064 downstream seed44", ""]
    lines.append(f"- SuperGLUE seed44: `{sg_seed44}`; seed42 canonical: `{sg_clean64_seed42}`; delta `{sg_seed44 - sg_clean64_seed42}`.")
    lines.append(f"- Overall with only SuperGLUE swapped to seed44 and all other canonical clean64 components fixed: `{overall_seed44}`.")
    lines.append(f"- Delta vs faithful coherent86 seed42 coordinate: `{overall_seed44 - overall_coherent}`; delta vs clean64 seed42 coordinate: `{overall_seed44 - overall_clean64_seed42}`.")
    lines.append("- Per-task primary deltas seed44 minus clean64 seed42: `" + json.dumps(task_deltas, sort_keys=True) + "`.")
    lines.append("")
    lines.append("## Export package")
    lines.append("")
    lines.append(f"- Export package: `{rel(EXPORT_DIR)}`; exists `{EXPORT_DIR.is_dir()}`; listed entries `{len(export_entries)}`.")
    lines.append(f"- Source-level verifier: `{rel(A01_VERIFIER)}`.")
    lines.append("")
    lines.append(f"Full JSON: `{rel(_public_path('experiments/archive/relation_learning/data/ftseed44_export_status/summary.json'))}`")
    (_public_path('research/documents/relation_learning/data/ftseed44_export_status/summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary_json": rel(_public_path('experiments/archive/relation_learning/data/ftseed44_export_status/summary.json')), "summary_md": rel(_public_path('research/documents/relation_learning/data/ftseed44_export_status/summary.md')), "clean64_seed44_overall": overall_seed44, "delta_vs_coherent86": overall_seed44 - overall_coherent, "export_exists": EXPORT_DIR.is_dir()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
