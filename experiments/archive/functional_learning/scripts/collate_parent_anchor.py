#!/usr/bin/env python3
"""Collate research parent-anchored continuation against research carrier-anchored continuation.

Inputs are produced by:
  * research common-screen coherent86 parent evaluation;
  * research standard/carrier-anchor ordinary continuation ladder;
  * research std87 executed-scale alpha checks;
  * research parent-anchor ordinary continuation ladder and optional drift probes.

The scientific comparison is intentionally narrow: changing only the neutral KL target
from private-off carrier to frozen private-on parent while keeping the legal suffix and
ordinary WWM adaptation fixed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
PARENT_COMMON = _public_path('experiments/archive/functional_learning/data/common_eval/collated_common_eval.json')
STD_LADDER = _public_path('experiments/archive/functional_learning/data/standard_ladder_eval/selected_ladder_summary.json')
CR_LADDER = _public_path('experiments/archive/functional_learning/data/carrier_ladder_eval/selected_ladder_summary.json')
PA_LADDER = _public_path('experiments/archive/functional_learning/data/parent_anchor_ladder_eval/selected_ladder_summary.json')
NO_KL_LADDER = _public_path('experiments/archive/functional_learning/data/no_kl_ladder_eval/selected_ladder_summary.json')
STD87_ALPHA = _public_path('experiments/archive/functional_learning/data/std87_alpha_eval')
PA_METRICS = _public_path('experiments/archive/functional_learning/training/runs/coherent86_continue_parent_anchor_seed43023/scientific_metrics.json')
PA_DRIFT = _public_path('experiments/archive/functional_learning/data/parent_anchor_drift_probe/model_drift_probe.json')
OUT_DIR = _public_path('experiments/archive/functional_learning/data/parent_anchor_collated')
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


def coherent86_parent_row() -> dict[str, Any] | None:
    obj = load_json(PARENT_COMMON)
    if not isinstance(obj, dict):
        return None
    for row in obj.get("rows", []):
        if row.get("tag") == "coherent86_alpha075":
            scores = row.get("scores", {})
            return {"arm": "parent", "tag": "coherent86_alpha075", "scores": scores, "path": row.get("source")}
    return None


def exposure_from_tag(tag: str) -> int | None:
    if "87M" in tag:
        return 87_005_295
    if "90M" in tag:
        return 90_005_295
    if "94M" in tag:
        return 94_005_295
    if "98M" in tag:
        return 98_005_295
    if "100M" in tag or tag.endswith("final"):
        return 100_000_000
    return None


def rows_from_ladder(path: Path, arm: str) -> list[dict[str, Any]]:
    obj = load_json(path)
    rows = []
    if not isinstance(obj, dict):
        return rows
    for r in obj.get("records", []):
        scores = r.get("scores") or {}
        if not scores:
            continue
        tag = r.get("tag")
        rows.append({
            "arm": arm,
            "tag": tag,
            "path": r.get("model_path"),
            "status": r.get("status"),
            "exposure_words": exposure_from_tag(str(tag)),
            "scores": scores,
        })
    return rows


def rows_from_alpha() -> list[dict[str, Any]]:
    rows = []
    for p in sorted(STD87_ALPHA.glob("std_87M_alpha*_eval.json")):
        obj = load_json(p) or {}
        rows.append({
            "arm": "standard_scale_check",
            "tag": obj.get("tag", p.stem),
            "path": obj.get("executed_model_path"),
            "status": "ok" if obj.get("scores") else "missing_scores",
            "exposure_words": 87_005_295,
            "alpha": obj.get("alpha"),
            "scores": obj.get("scores") or {},
        })
    return rows


def score_delta(scores: dict[str, Any], ref_scores: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in COLS:
        if scores.get(c) is not None and ref_scores.get(c) is not None:
            out[c] = float(scores[c]) - float(ref_scores[c])
    if scores.get("equal_valid_mean") is not None and ref_scores.get("equal_valid_mean") is not None:
        out["equal7"] = float(scores["equal_valid_mean"]) - float(ref_scores["equal_valid_mean"])
    return out


def best_by_equal(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [r for r in rows if r.get("scores", {}).get("equal_valid_mean") is not None]
    if not ok:
        return None
    return max(ok, key=lambda r: float(r["scores"]["equal_valid_mean"]))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parent = coherent86_parent_row()
    rows = []
    if parent:
        rows.append(parent)
    rows.extend(rows_from_ladder(STD_LADDER, "standard_carrier_anchor"))
    rows.extend(rows_from_ladder(CR_LADDER, "carrier_residual"))
    rows.extend(rows_from_ladder(PA_LADDER, "standard_parent_anchor"))
    rows.extend(rows_from_ladder(NO_KL_LADDER, "standard_no_kl"))
    rows.extend(rows_from_alpha())
    parent_scores = (parent or {}).get("scores", {})
    for r in rows:
        scores = r.get("scores") or {}
        r["equal7"] = scores.get("equal_valid_mean")
        if parent_scores:
            r["delta_vs_parent"] = score_delta(scores, parent_scores)
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_arm.setdefault(str(r.get("arm")), []).append(r)
    best = {arm: best_by_equal(rs) for arm, rs in by_arm.items()}
    result = {
        "status": "PARENT_ANCHOR_COLLATION",
        "sources": {
            "parent_common": rel(PARENT_COMMON),
            "standard_ladder": rel(STD_LADDER),
            "carrier_ladder": rel(CR_LADDER),
            "parent_anchor_ladder": rel(PA_LADDER),
            "no_kl_ladder": rel(NO_KL_LADDER),
            "std87_alpha_dir": rel(STD87_ALPHA),
            "parent_anchor_metrics": rel(PA_METRICS),
            "parent_anchor_drift": rel(PA_DRIFT),
        },
        "parent_anchor_training_metrics": load_json(PA_METRICS),
        "parent_anchor_drift_probe": load_json(PA_DRIFT),
        "rows": rows,
        "best_by_arm": best,
    }
    out_json = _public_path('experiments/archive/functional_learning/data/parent_anchor_collated/parent_anchor_collated.json')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# research parent-anchor continuation collation",
        "",
        f"Parent common-screen source: `{rel(PARENT_COMMON)}`",
        "",
        "| arm | tag | exposure | alpha | equal7 | Δ vs parent | BLiMP | Supp | EWoK | Entity | COMPS | GPIQA | Reading |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (str(x.get("arm")), x.get("exposure_words") or 0, str(x.get("tag")))):
        s = r.get("scores") or {}
        d = r.get("delta_vs_parent") or {}
        lines.append("| {arm} | {tag} | {exp} | {alpha} | {eq} | {deq} | {BLiMP} | {Supplement} | {EWoK} | {Entity} | {COMPS} | {GPIQA} | {Reading} |".format(
            arm=r.get("arm", ""), tag=r.get("tag", ""), exp="" if r.get("exposure_words") is None else r.get("exposure_words"),
            alpha="" if r.get("alpha") is None else r.get("alpha"),
            eq="" if s.get("equal_valid_mean") is None else f"{s['equal_valid_mean']:.4f}",
            deq="" if d.get("equal7") is None else f"{d['equal7']:+.4f}",
            BLiMP="" if s.get("BLiMP") is None else f"{s['BLiMP']:.2f}",
            Supplement="" if s.get("Supplement") is None else f"{s['Supplement']:.2f}",
            EWoK="" if s.get("EWoK") is None else f"{s['EWoK']:.2f}",
            Entity="" if s.get("Entity") is None else f"{s['Entity']:.2f}",
            COMPS="" if s.get("COMPS") is None else f"{s['COMPS']:.2f}",
            GPIQA="" if s.get("GlobalPIQA_mean") is None else f"{s['GlobalPIQA_mean']:.3f}",
            Reading="" if s.get("Reading") is None else f"{s['Reading']:.3f}",
        ))
    lines += ["", "## Best equal7 by arm", ""]
    for arm, r in sorted(best.items()):
        if not r:
            lines.append(f"- {arm}: no evaluated row")
        else:
            s = r.get("scores") or {}
            d = r.get("delta_vs_parent") or {}
            lines.append(f"- {arm}: `{r.get('tag')}` equal7 {s.get('equal_valid_mean'):.4f}, Δ vs parent {d.get('equal7', 0.0):+.4f}.")
    out_md = _public_path('research/documents/functional_learning/data/parent_anchor_collated/parent_anchor_collated.md')
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "n_rows": len(rows), "out_json": rel(out_json), "out_md": rel(out_md)}), flush=True)


if __name__ == "__main__":
    main()
