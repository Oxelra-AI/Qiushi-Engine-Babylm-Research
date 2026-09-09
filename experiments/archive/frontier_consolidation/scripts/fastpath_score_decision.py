#!/usr/bin/env python3
"""Score and transition synthesis for research frozen-anchor fast-path replay."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import time
from statistics import mean
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
OUT_DIR = _public_path('experiments/archive/frontier_consolidation/data/fastpath_score_decision')
OUT_JSON = _public_path('experiments/archive/frontier_consolidation/data/fastpath_score_decision/fastpath_score_decision.json')
OUT_MD = _public_path('research/documents/frontier_consolidation/data/fastpath_score_decision/fastpath_score_decision.md')

CHCK82_VERIFY = _public_path('experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json')
SHUFFLED86_CHEAP = _public_path('experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_summary/frozen82_tail4M_shuffled_summary.json')
SHUFFLED86_SG = _public_path('experiments/archive/frontier_consolidation/data/repeat_shuffled_tail_superglue_summary/repeat_frozen82_tail4M_shuffled_superglue_superglue_summary.json')
TRUTHFUL86 = _public_path('experiments/archive/frontier_consolidation/data/truthful_shuffled86_carrier/truthful_shuffled86_carrier_manifest.json')
FULL_MODEL_VALIDATION = _public_path('experiments/archive/frontier_consolidation/data/fastpath_full_model_validation/fastpath_full_model_validation.json')
ITEM_ANALYSIS = _public_path('experiments/archive/frontier_consolidation/data/fastpath_item_family_analysis/fastpath_item_family_analysis.json')
ARM_PATHS = {
    "coherent": {
        "cheap": _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_summary/fastpath4M_coherent_summary.json'),
        "sg": _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_coherent_superglue_summary/fastpath4M_coherent_superglue_summary.json'),
    },
    "spanbreak": {
        "cheap": _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_spanbreak_summary/fastpath4M_spanbreak_summary.json'),
        "sg": _public_path('experiments/archive/frontier_consolidation/data/fastpath4M_spanbreak_superglue_summary/fastpath4M_spanbreak_superglue_summary.json'),
    },
}
CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ALL_COLUMNS = CHEAP_COLUMNS + ["SuperGLUE", "AoA"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def chck82_record() -> dict[str, Any]:
    j = read_json(CHCK82_VERIFY)
    scores = {k: float(v) for k, v in j["score_arithmetic"]["scores"].items() if v is not None}
    return {
        "status": "available",
        "scores": scores,
        "cheap7": mean(scores[c] for c in CHEAP_COLUMNS),
        "superglue": scores["SuperGLUE"],
        "aoa": scores["AoA"],
        "overall": float(j["score_arithmetic"]["overall_reported"]),
        "source": rel(CHCK82_VERIFY),
    }


def from_cheap_sg(name: str, cheap_path: pathlib.Path, sg_path: pathlib.Path | None) -> dict[str, Any]:
    cheap = read_json(cheap_path)
    if cheap is None:
        return {"status": "pending_cheap", "cheap_path": rel(cheap_path)}
    scores = {k: float(v) for k, v in cheap["scores"].items() if v is not None}
    rec = {
        "status": "cheap_available",
        "scores": scores,
        "cheap7": float(cheap["cheap7"]),
        "cheap_path": rel(cheap_path),
        "training_metrics": cheap.get("training_metrics"),
    }
    sg = read_json(sg_path) if sg_path is not None else None
    if sg is not None:
        rec["status"] = "cheap_and_superglue_available"
        rec["superglue"] = float(sg["superglue"])
        rec["scores"]["SuperGLUE"] = float(sg["superglue"])
        rec["scores"]["AoA"] = 0.0
        rec["aoa"] = 0.0
        rec["overall_aoa0"] = float(sg["projected_overall_with_aoa0"])
        rec["sg_path"] = rel(sg_path)
    else:
        rec["sg_path"] = None if sg_path is None else rel(sg_path)
    return rec


def shuffled86_record() -> dict[str, Any]:
    rec = from_cheap_sg("shuffled86", SHUFFLED86_CHEAP, SHUFFLED86_SG)
    manifest = read_json(TRUTHFUL86)
    if manifest is not None:
        rec["truthful_carrier"] = {
            "path": rel(TRUTHFUL86),
            "carrier_path": manifest.get("carrier_path"),
            "carrier_sha256": manifest.get("carrier_sha256"),
            "candidate_native_overall_aoa0": manifest.get("candidate_native_overall_aoa0"),
            "status": manifest.get("status"),
        }
    return rec


def delta_record(rec: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    if not rec.get("scores"):
        return None
    out: dict[str, Any] = {}
    for c in CHEAP_COLUMNS:
        if rec["scores"].get(c) is not None and ref["scores"].get(c) is not None:
            out[c] = float(rec["scores"][c] - ref["scores"][c])
    if rec.get("cheap7") is not None:
        out["cheap7"] = float(rec["cheap7"] - ref["cheap7"])
    if rec.get("superglue") is not None:
        out["SuperGLUE"] = float(rec["superglue"] - ref["superglue"])
    if rec.get("overall_aoa0") is not None:
        out["Overall_AoA0"] = float(rec["overall_aoa0"] - ref["overall"])
    return out


def route_read(records: dict[str, dict[str, Any]], item: dict[str, Any] | None) -> str:
    coherent = records.get("coherent", {})
    span = records.get("spanbreak", {})
    if coherent.get("status") == "pending_cheap" or span.get("status") == "pending_cheap":
        return "pending_cheap7"
    if coherent.get("cheap7") is None or span.get("cheap7") is None:
        return "missing_scores"
    c = float(coherent["cheap7"])
    s = float(span["cheap7"])
    ref = float(records["chck82"]["cheap7"])
    sh = records.get("shuffled86", {}).get("cheap7")
    diff_cs = c - s
    diff_ref = c - ref
    diff_sh = None if sh is None else c - float(sh)
    read = []
    if diff_cs <= 0.05:
        read.append("coherent_not_clear_over_spanbreak")
    else:
        read.append("coherent_has_cheap7_edge_over_spanbreak")
    if diff_ref <= 0:
        read.append("coherent_below_anchor_cheap7")
    else:
        read.append("coherent_above_anchor_cheap7")
    if diff_sh is not None:
        if diff_sh <= 0.05:
            read.append("coherent_not_clear_over_shuffled86")
        else:
            read.append("coherent_above_shuffled86_cheap7")
    if item and item.get("status") == "COMPLETE":
        rr = item.get("route_read")
        if rr:
            read.append(f"item_read={rr}")
    if coherent.get("superglue") is None or span.get("superglue") is None:
        read.append("superglue_pending_or_not_run")
    return ";".join(read)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = chck82_record()
    records: dict[str, dict[str, Any]] = {
        "chck82": ref,
        "shuffled86": shuffled86_record(),
    }
    for name, paths in ARM_PATHS.items():
        records[name] = from_cheap_sg(name, paths["cheap"], paths["sg"])
    deltas = {name: delta_record(rec, ref) for name, rec in records.items() if name != "chck82"}
    validation = read_json(FULL_MODEL_VALIDATION)
    item = read_json(ITEM_ANALYSIS)
    read = route_read(records, item)
    out = {
        "status": "PENDING" if "pending" in read or records["coherent"].get("status") == "pending_cheap" or records["spanbreak"].get("status") == "pending_cheap" else "COMPLETE_OR_PARTIAL_WITH_SCORES",
        "created_utc": now(),
        "records": records,
        "deltas_vs_chck82": deltas,
        "full_model_validation": None if validation is None else {"path": rel(FULL_MODEL_VALIDATION), "status": validation.get("status")},
        "item_family_analysis": None if item is None else {"path": rel(ITEM_ANALYSIS), "status": item.get("status"), "route_read": item.get("route_read")},
        "route_read": read,
        "scientific_reading": "The frozen-anchor fast-path route should continue only if coherent private-ON replay retains anchor-correct fragile relation/state decisions and adds new correct decisions reproducibly, not merely because it has private-OFF reversibility or a small aggregate score edge over a disrupted input.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research fast-path score decision", "", f"Status: **{out['status']}**", f"Route read: `{read}`", "", "## Score table", "", "| arm | status | cheap7 | Δcheap7 vs chck82 | SuperGLUE | Overall(AoA0) | ΔOverall vs chck82 |", "|---|---|---:|---:|---:|---:|---:|"]
    for name in ["chck82", "shuffled86", "coherent", "spanbreak"]:
        rec = records[name]
        d = deltas.get(name) or {}
        lines.append(f"| {name} | {rec.get('status')} | {rec.get('cheap7')} | {d.get('cheap7')} | {rec.get('superglue')} | {rec.get('overall', rec.get('overall_aoa0'))} | {d.get('Overall_AoA0')} |")
    lines += ["", "## Sources", ""]
    for name, rec in records.items():
        lines.append(f"- {name}: cheap `{rec.get('cheap_path', rec.get('source'))}`, SuperGLUE `{rec.get('sg_path')}`")
    lines += ["", out["scientific_reading"], "", f"JSON: `{rel(OUT_JSON)}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "route_read": read, "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
