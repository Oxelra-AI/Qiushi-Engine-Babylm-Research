#!/usr/bin/env python3
"""Collate research coherent86 continuation training, geometry, and common-screen evals."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
PARENT_COMMON = _public_path('experiments/archive/functional_learning/data/common_eval/collated_common_eval.json')
STD_EVAL = _public_path('experiments/archive/functional_learning/data/standard_ladder_eval/selected_ladder_summary.json')
CR_EVAL = _public_path('experiments/archive/functional_learning/data/carrier_ladder_eval/selected_ladder_summary.json')
STD_GEOM = _public_path('experiments/archive/functional_learning/data/adapter_geometry/adapter_geometry.json')
CR_GEOM = _public_path('experiments/archive/functional_learning/data/adapter_geometry_carrier/adapter_geometry.json')
STD_METRICS = _public_path('experiments/archive/functional_learning/training/runs/coherent86_continue_standard_seed43023/scientific_metrics.json')
CR_METRICS = _public_path('experiments/archive/functional_learning/training/runs/coherent86_continue_carrier_residual_seed43023/scientific_metrics.json')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/continuation_collated')

COLS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_mean", "Reading"]


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def parent_scores() -> dict[str, Any] | None:
    rec = load_json(PARENT_COMMON)
    if not rec:
        return None
    # research collate format may be dict with rows or list; handle both.
    if isinstance(rec, dict):
        for key in ["records", "rows", "results"]:
            if isinstance(rec.get(key), list):
                for r in rec[key]:
                    tag = r.get("tag") or r.get("model") or r.get("name")
                    if tag == "coherent86_alpha075":
                        return r
        if "coherent86_alpha075" in rec:
            return rec["coherent86_alpha075"]
    return None


def records_from_eval(path: Path, arm: str) -> list[dict[str, Any]]:
    s = load_json(path)
    if not s:
        return []
    out = []
    for r in s.get("records", []):
        scores = r.get("scores") or {}
        if not scores:
            continue
        out.append({
            "arm": arm,
            "tag": r.get("tag"),
            "model_path": r.get("model_path"),
            "status": r.get("status"),
            "equal7": scores.get("equal_valid_mean"),
            **{c: scores.get(c) for c in COLS},
        })
    return out


def geom_map(path: Path) -> dict[str, Any]:
    g = load_json(path) or {}
    return g.get("models", {})


def attach_geometry(rows: list[dict[str, Any]], std_g: dict[str, Any], cr_g: dict[str, Any]) -> None:
    for r in rows:
        tag = str(r.get("tag"))
        candidates = []
        if r.get("arm") == "standard":
            # geometry tags use std_chck_total_... or std_final
            if tag.startswith("std_"):
                suffix = tag[len("std_"):]
                candidates.append(f"std_{suffix}")
                if suffix == "100M_final":
                    candidates.append("std_final")
            candidates.extend(std_g.keys())
            gsrc = std_g
        else:
            if tag.startswith("cr_"):
                suffix = tag[len("cr_"):]
                candidates.append(f"cr_{suffix}")
                if suffix == "100M_final":
                    candidates.append("cr_final")
            candidates.extend(cr_g.keys())
            gsrc = cr_g
        match = None
        for k in candidates:
            m = gsrc.get(k)
            if not m or m.get("status") != "ok":
                continue
            mp = str(m.get("path", ""))
            if tag in k or (r.get("model_path") and r["model_path"] == mp) or ("final" in tag and "final" in k):
                match = m
                break
        if match:
            r["exposure_words"] = match.get("exposure_words")
            r["relative_delta_to_parent"] = match.get("all", {}).get("relative_delta_to_parent")
            r["delta_norm"] = match.get("all", {}).get("delta_norm")
            r["cos_delta_parent"] = match.get("all", {}).get("cos_delta_parent")
            r["cos_child_parent"] = match.get("all", {}).get("cos_child_parent")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = records_from_eval(STD_EVAL, "standard") + records_from_eval(CR_EVAL, "carrier_residual")
    attach_geometry(rows, geom_map(STD_GEOM), geom_map(CR_GEOM))
    parent = parent_scores()
    parent_eq = None
    if parent:
        parent_eq = parent.get("eq_mean") or parent.get("equal7") or parent.get("equal_valid_mean") or parent.get("equal7_or_valid_mean")
        if parent_eq is None and isinstance(parent.get("scores"), dict):
            parent_eq = parent["scores"].get("equal_valid_mean") or parent["scores"].get("equal7")
    for r in rows:
        if parent_eq is not None and r.get("equal7") is not None:
            r["delta_equal7_vs_step026_coherent86"] = r["equal7"] - float(parent_eq)
    result = {
        "status": "CONTINUATION_COLLATION_DONE",
        "parent_common_record": parent,
        "parent_common_source": rel(PARENT_COMMON),
        "standard_eval_source": rel(STD_EVAL),
        "carrier_eval_source": rel(CR_EVAL),
        "standard_metrics": load_json(STD_METRICS),
        "carrier_metrics": load_json(CR_METRICS),
        "records": rows,
    }
    out_json = out_dir / "continuation_collated.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# research coherent86 continuation collated result", "", f"Parent common-screen source: `{rel(PARENT_COMMON)}`", "", "| arm | tag | exposure | equal7 | Δ vs parent | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading | rel adapter Δ | cos(child,parent) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: (x.get("arm", ""), x.get("exposure_words") or 0, x.get("tag") or "")):
        lines.append("| {arm} | {tag} | {exp} | {eq} | {deq} | {BLiMP} | {Supplement} | {EWoK} | {Entity} | {COMPS} | {GPIQA} | {Reading} | {reld} | {cpar} |".format(
            arm=r.get("arm", ""), tag=r.get("tag", ""), exp="" if r.get("exposure_words") is None else r.get("exposure_words"),
            eq="" if r.get("equal7") is None else f"{r['equal7']:.4f}",
            deq="" if r.get("delta_equal7_vs_step026_coherent86") is None else f"{r['delta_equal7_vs_step026_coherent86']:+.4f}",
            BLiMP="" if r.get("BLiMP") is None else f"{r['BLiMP']:.2f}",
            Supplement="" if r.get("Supplement") is None else f"{r['Supplement']:.2f}",
            EWoK="" if r.get("EWoK") is None else f"{r['EWoK']:.2f}",
            Entity="" if r.get("Entity") is None else f"{r['Entity']:.2f}",
            COMPS="" if r.get("COMPS") is None else f"{r['COMPS']:.2f}",
            GPIQA="" if r.get("GlobalPIQA_mean") is None else f"{r['GlobalPIQA_mean']:.3f}",
            Reading="" if r.get("Reading") is None else f"{r['Reading']:.3f}",
            reld="" if r.get("relative_delta_to_parent") is None else f"{r['relative_delta_to_parent']:.4f}",
            cpar="" if r.get("cos_child_parent") is None else f"{r['cos_child_parent']:.4f}",
        ))
    out_md = out_dir / "continuation_collated.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "n_records": len(rows), "out_json": rel(out_json), "out_md": rel(out_md)}), flush=True)


if __name__ == "__main__":
    main()
