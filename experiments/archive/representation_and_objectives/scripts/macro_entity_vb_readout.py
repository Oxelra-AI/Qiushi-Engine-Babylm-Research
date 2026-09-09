#!/usr/bin/env python3
"""research: integrate macro Entity V/B/R readouts with binding affected/unaffected evidence.

This is file-only analysis. It reads existing official Entity split tables,
the binding affected/unaffected CPU probe, and broad family-vector
summaries. It does not run a model, scorer, GPU job, upload, or leaderboard code.

Scientific question: does first-basin DeBERTa MAX view-minus-breadth (V-B) look
like a source-correspondence/state-record benefit, or is it mostly another finite-
budget decision offset / family-vector redistribution pattern?
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import pathlib
import re
import statistics
import time
from collections import defaultdict
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
A01_WS = ROOT / "experiments/archive" / 'representation_and_objectives'
A02_WS = ROOT / "experiments/archive" / 'frontier_consolidation'
OUT = A01_WS / "data" / "macro_entity_vb_readout"

ENTITY_CONTRASTS = A02_WS / "data" / "entity_balanced_and_transfer_readout_full" / "entity_contrast_rows.csv"
ENTITY_BALANCED = A02_WS / "data" / "entity_balanced_and_transfer_readout_full" / "entity_balanced_rows.csv"
BINDING_CONTRASTS = A01_WS / "data" / "breadth_binding_affunaff_probe" / "binding_contrast_deltas.csv"
BINDING_WINDOW = A01_WS / "data" / "breadth_binding_affunaff_probe" / "binding_window_summary.csv"
BROAD_VECTOR = A02_WS / "data" / "conservation_vector_readout" / "visible_reference_net_norm.csv"
CONSERVATION_SUMMARY = A02_WS / "data" / "conservation_vector_readout" / "conservation_vector_summary.json"

CONTRAST_LABELS = {
    "deberta_basin1": "V-R",
    "deberta_view_minus_breadth": "V-B",
    "deberta_breadth_minus_repeat": "B-R",
    "deberta_basin2": "D2 V-R",
    "roberta_max": "RoBERTa V-R",
}
KEY_CONTRASTS = ["deberta_basin1", "deberta_view_minus_breadth", "deberta_breadth_minus_repeat"]
KEY_GROUPS = [
    "all_18_subtasks", "zero_ops", "nonzero_ops",
    "numops_0", "numops_1", "numops_2", "numops_3", "numops_4", "numops_5",
    "split_ambiref", "split_move_contents", "split_regular",
    "ambiref_zero_ops", "ambiref_nonzero_ops", "move_contents_zero_ops", "move_contents_nonzero_ops", "regular_zero_ops", "regular_nonzero_ops",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    pp = pathlib.Path(p)
    try:
        return str(pp.relative_to(ROOT))
    except Exception:
        return str(pp)


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fval(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except Exception:
        return None
    return x if x == x else None


def mean(xs: list[float]) -> float | None:
    return None if not xs else sum(xs) / len(xs)


def summarize_group(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    vals = [fval(r, metric) for r in rows]
    vals = [v for v in vals if v is not None]
    cks = sorted({r.get("checkpoint", "") for r in rows if r.get("checkpoint")})
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": statistics.median(vals) if vals else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
        "positive_count": sum(1 for v in vals if v > 0),
        "negative_count": sum(1 for v in vals if v < 0),
        "checkpoints": ";".join(cks),
    }


def entity_summary(entity_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    late_rows = [r for r in entity_rows if r.get("checkpoint") in {"chck_80M", "chck_90M", "chck_100M"}]
    summary_rows: list[dict[str, Any]] = []
    for contrast in KEY_CONTRASTS:
        for group in KEY_GROUPS:
            rs = [r for r in late_rows if r.get("contrast_name") == contrast and r.get("group") == group]
            if not rs:
                continue
            s = summarize_group(rs, "delta_macro_pp")
            # Prediction option0 fraction is the selected-option-surface shift, not the answer margin.
            os = summarize_group(rs, "delta_pred_option0_frac")
            summary_rows.append({
                "contrast_name": contrast,
                "contrast_short": CONTRAST_LABELS.get(contrast, contrast),
                "group": group,
                "n": s["n"],
                "mean_delta_pp": s["mean"],
                "median_delta_pp": s["median"],
                "min_delta_pp": s["min"],
                "max_delta_pp": s["max"],
                "positive_count": s["positive_count"],
                "negative_count": s["negative_count"],
                "mean_delta_pred_option0_frac": os["mean"],
                "checkpoints": s["checkpoints"],
            })
    # Identity residuals: V-R=(V-B)+(B-R) inside Entity tables.
    residual_rows: list[dict[str, Any]] = []
    by_ck_group: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in late_rows:
        c = r.get("contrast_name")
        if c in KEY_CONTRASTS:
            v = fval(r, "delta_macro_pp")
            if v is not None:
                by_ck_group[(r.get("checkpoint", ""), r.get("group", ""))][c] = v
    for (ck, group), d in sorted(by_ck_group.items()):
        if all(k in d for k in KEY_CONTRASTS):
            residual_rows.append({
                "checkpoint": ck,
                "group": group,
                "VminusR_pp": d["deberta_basin1"],
                "VminusB_plus_BminusR_pp": d["deberta_view_minus_breadth"] + d["deberta_breadth_minus_repeat"],
                "residual_pp": d["deberta_basin1"] - d["deberta_view_minus_breadth"] - d["deberta_breadth_minus_repeat"],
            })
    return summary_rows, residual_rows


def read_binding(binding_rows: list[dict[str, str]], binding_window: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cks = {"chck_80M", "chck_90M", "chck_100M"}
    rows = [r for r in binding_rows if r.get("checkpoint") in cks]
    summary: list[dict[str, Any]] = []
    for c in ["VminusR", "VminusB", "BminusR"]:
        rs = [r for r in rows if r.get("contrast") == c]
        for q in ["affected_delta", "unaffected_delta", "EEBF_delta", "affected_margin_delta", "unaffected_margin_delta", "signed_bias_margin_delta"]:
            vals = [fval(r, q) for r in rs]
            vals = [v for v in vals if v is not None]
            summary.append({
                "contrast": c,
                "quantity": q,
                "n": len(vals),
                "mean": mean(vals),
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
            })
        summary.append({
            "contrast": c,
            "quantity": "offset_like_count",
            "n": len(rs),
            "mean": sum(1 for r in rs if str(r.get("offset_like_signature", "")).lower() == "true"),
            "min": None,
            "max": None,
        })
    # Keep the window file rows too, because they are produced by the original inference script.
    window_rows: list[dict[str, Any]] = []
    for r in binding_window:
        window_rows.append({k: (fval(r, k) if k in {"mean", "min", "max"} else r.get(k)) for k in r})
    return summary, window_rows


def extract_visible_vectors(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    wanted = {
        ("D1_VminusB", "chck_80M", "stable6"),
        ("D1_VminusB", "chck_80M", "stable5_exEntity"),
        ("D1_BminusCold", "chck_80M", "stable5_exEntity"),
        ("D1_VminusCold", "chck_80M", "stable5_exEntity"),
        ("D1_BminusR", "chck_80M", "stable5_exEntity"),
        ("D1_VminusR", "chck_80M", "stable5_exEntity"),
    }
    for r in rows:
        key = (r.get("contrast"), r.get("checkpoint"), r.get("family_set"))
        if key in wanted:
            vec = {}
            try:
                vec = json.loads(r.get("vector_json", "{}"))
            except Exception:
                pass
            out.append({
                "contrast": r.get("contrast"),
                "checkpoint": r.get("checkpoint"),
                "family_set": r.get("family_set"),
                "mean_net": fval(r, "mean_net"),
                "rms_family_delta": fval(r, "rms_family_delta"),
                "abs_net_over_rms": fval(r, "abs_net_over_rms"),
                "vector": vec,
            })
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fmt(x: Any, scale: float = 1.0) -> str:
    if x is None or x == "":
        return "NA"
    try:
        return f"{scale * float(x):+.3f}"
    except Exception:
        return str(x)


def row_lookup(rows: list[dict[str, Any]], contrast: str, group: str) -> dict[str, Any] | None:
    for r in rows:
        if r.get("contrast_name") == contrast and r.get("group") == group:
            return r
    return None


def bind_lookup(rows: list[dict[str, Any]], contrast: str, quantity: str) -> dict[str, Any] | None:
    for r in rows:
        if r.get("contrast") == contrast and r.get("quantity") == quantity:
            return r
    return None


def write_md(payload: dict[str, Any], path: pathlib.Path) -> None:
    es = payload["entity_group_summary"]
    bs = payload["binding_late_summary"]
    lines: list[str] = []
    lines.append("# research macro Entity V/B/R readout\n\n")
    lines.append("This file-only readout integrates A02 official Entity split tables with the A01 CPU binding affected/unaffected probe. It separates the official Entity surface from the counterbalanced binding substrate before deciding whether compact source views support correspondence or merely redistribute finite-budget decisions.\n\n")

    lines.append("## Coarse Entity operation rows (late 80/90/100M)\n\n")
    lines.append("| contrast | official all18 Δ pp | zero-op Δ pp | nonzero Δ pp | balanced zero/nonzero Δ pp | pred-option0 shift on all18 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for c in KEY_CONTRASTS:
        allr = row_lookup(es, c, "all_18_subtasks")
        zr = row_lookup(es, c, "zero_ops")
        nr = row_lookup(es, c, "nonzero_ops")
        z = zr.get("mean_delta_pp") if zr else None
        n = nr.get("mean_delta_pp") if nr else None
        balanced = None if z is None or n is None else 0.5 * (float(z) + float(n))
        lines.append(f"| {CONTRAST_LABELS[c]} | {fmt(allr.get('mean_delta_pp') if allr else None)} | {fmt(z)} | {fmt(n)} | {fmt(balanced)} | {fmt(allr.get('mean_delta_pred_option0_frac') if allr else None)} |\n")

    lines.append("\n## Entity by number of operations\n\n")
    lines.append("| group | V-R Δ pp | V-B Δ pp | B-R Δ pp |\n")
    lines.append("|---|---:|---:|---:|\n")
    for group in [f"numops_{i}" for i in range(6)]:
        vr = row_lookup(es, "deberta_basin1", group)
        vb = row_lookup(es, "deberta_view_minus_breadth", group)
        br = row_lookup(es, "deberta_breadth_minus_repeat", group)
        lines.append(f"| {group} | {fmt(vr.get('mean_delta_pp') if vr else None)} | {fmt(vb.get('mean_delta_pp') if vb else None)} | {fmt(br.get('mean_delta_pp') if br else None)} |\n")

    lines.append("\n## Entity split families\n\n")
    lines.append("| split group | V-R Δ pp | V-B Δ pp | B-R Δ pp |\n")
    lines.append("|---|---:|---:|---:|\n")
    for group in ["split_ambiref", "split_move_contents", "split_regular", "ambiref_zero_ops", "ambiref_nonzero_ops", "move_contents_zero_ops", "move_contents_nonzero_ops", "regular_zero_ops", "regular_nonzero_ops"]:
        vr = row_lookup(es, "deberta_basin1", group)
        vb = row_lookup(es, "deberta_view_minus_breadth", group)
        br = row_lookup(es, "deberta_breadth_minus_repeat", group)
        lines.append(f"| {group} | {fmt(vr.get('mean_delta_pp') if vr else None)} | {fmt(vb.get('mean_delta_pp') if vb else None)} | {fmt(br.get('mean_delta_pp') if br else None)} |\n")

    lines.append("\n## Counterbalanced binding substrate\n\n")
    lines.append("| contrast | affected Δ pp | unaffected Δ pp | EEBF Δ pp | affected margin Δ | unaffected margin Δ | signed margin Δ |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for c in ["VminusR", "VminusB", "BminusR"]:
        lines.append(f"| {c.replace('minus','-')} | {fmt((bind_lookup(bs,c,'affected_delta') or {}).get('mean'),100)} | {fmt((bind_lookup(bs,c,'unaffected_delta') or {}).get('mean'),100)} | {fmt((bind_lookup(bs,c,'EEBF_delta') or {}).get('mean'),100)} | {fmt((bind_lookup(bs,c,'affected_margin_delta') or {}).get('mean'))} | {fmt((bind_lookup(bs,c,'unaffected_margin_delta') or {}).get('mean'))} | {fmt((bind_lookup(bs,c,'signed_bias_margin_delta') or {}).get('mean'))} |\n")

    lines.append("\n## Broad-family visible vector rows\n\n")
    lines.append("| contrast | set | net | RMS | |net|/RMS | vector |\n")
    lines.append("|---|---|---:|---:|---:|---|\n")
    for r in payload["visible_broad_vectors"]:
        vec = ", ".join(f"{k}:{fmt(v)}" for k, v in sorted(r.get("vector", {}).items()))
        lines.append(f"| {r['contrast']} | {r['family_set']} | {fmt(r['mean_net'])} | {fmt(r['rms_family_delta'])} | {fmt(r['abs_net_over_rms'])} | {vec} |\n")

    lines.append("\n## Scientific reading\n\n")
    for item in payload["scientific_reading"]:
        lines.append(f"- {item}\n")

    lines.append("\n## Files\n\n")
    for k, v in payload["files"].items():
        lines.append(f"- {k}: `{v}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entity_rows = read_csv(ENTITY_CONTRASTS)
    balanced_rows = read_csv(ENTITY_BALANCED)
    binding_rows = read_csv(BINDING_CONTRASTS)
    binding_window_rows = read_csv(BINDING_WINDOW)
    broad_rows = read_csv(BROAD_VECTOR)
    conservation = json.loads(CONSERVATION_SUMMARY.read_text(encoding="utf-8")) if CONSERVATION_SUMMARY.exists() else {}

    entity_group_summary, residual_rows = entity_summary(entity_rows)
    binding_late_summary, binding_window = read_binding(binding_rows, binding_window_rows)
    visible = extract_visible_vectors(broad_rows)

    # Pull central quantities for concise scientific statements.
    vr_z = (row_lookup(entity_group_summary, "deberta_basin1", "zero_ops") or {}).get("mean_delta_pp")
    vr_nz = (row_lookup(entity_group_summary, "deberta_basin1", "nonzero_ops") or {}).get("mean_delta_pp")
    vb_z = (row_lookup(entity_group_summary, "deberta_view_minus_breadth", "zero_ops") or {}).get("mean_delta_pp")
    vb_nz = (row_lookup(entity_group_summary, "deberta_view_minus_breadth", "nonzero_ops") or {}).get("mean_delta_pp")
    br_z = (row_lookup(entity_group_summary, "deberta_breadth_minus_repeat", "zero_ops") or {}).get("mean_delta_pp")
    br_nz = (row_lookup(entity_group_summary, "deberta_breadth_minus_repeat", "nonzero_ops") or {}).get("mean_delta_pp")
    vb_aff = (bind_lookup(binding_late_summary, "VminusB", "affected_delta") or {}).get("mean")
    vb_unaff = (bind_lookup(binding_late_summary, "VminusB", "unaffected_delta") or {}).get("mean")
    vb_eebf = (bind_lookup(binding_late_summary, "VminusB", "EEBF_delta") or {}).get("mean")
    dose_ex = ((conservation.get("dose_summary") or {}).get("dose_vr_stable5_exEntity_net_trend") or {}).get("values")

    reading = [
        f"Official Entity V-R is strongly operation-skewed in the first basin: zero-op {fmt(vr_z)} pp versus nonzero {fmt(vr_nz)} pp. B-R has the same broad shape: zero-op {fmt(br_z)} pp versus nonzero {fmt(br_nz)} pp.",
        f"Official Entity V-B is positive on both coarse strata: zero-op {fmt(vb_z)} pp and nonzero {fmt(vb_nz)} pp. This means V-B is not simply the same zero/nonzero arithmetic as V-R/B-R on the official Entity surface.",
        f"The counterbalanced binding substrate changes that reading: late V-B affected gain is {fmt(vb_aff,100)} pp, unaffected change is {fmt(vb_unaff,100)} pp, and balanced EEBF movement is only {fmt(vb_eebf,100)} pp. Thus the source-view advantage on binding rows is small and offset-shaped, not a clean affected-with-retained-unaffected record-selection result.",
        "Visible broad V-B at 80M is not a broad ex-Entity gain: stable5 ex-Entity net is -0.062 pp while RMS is about 1.048 pp. B-C_old is positive on ex-Entity, so broad total V-C can arise from admitting different experience or geometry without showing own-source companion correspondence.",
        f"A02's conservation-vector file reports ex-Entity V-R net across doses {dose_ex}; the non-Entity mean remains small relative to family-vector motion. This supports a finite-budget redistribution view: many interventions move capabilities between families or item decisions more than they raise all families together.",
        "A permuted compact companion remains the cleanest future asymmetry only if updated broad V-B survives after missing breadth/clean scoring, or if a stronger binding/state panel shows V-B without an unaffected cost. Current A01 binding evidence alone does not warrant an H100 permuted run.",
    ]

    files = {
        "entity_group_summary_csv": rel(OUT / "entity_group_late_summary.csv"),
        "entity_identity_residuals_csv": rel(OUT / "entity_identity_residuals.csv"),
        "binding_late_summary_csv": rel(OUT / "binding_late_summary.csv"),
        "visible_broad_vectors_csv": rel(OUT / "visible_broad_vectors.csv"),
        "summary_json": rel(OUT / "macro_entity_vb_readout_summary.json"),
        "summary_md": rel(OUT / "macro_entity_vb_readout_summary.md"),
        "source_entity_contrasts": rel(ENTITY_CONTRASTS),
        "source_binding_contrasts": rel(BINDING_CONTRASTS),
        "source_broad_vectors": rel(BROAD_VECTOR),
    }
    payload = {
        "status": "MACRO_ENTITY_VB_READOUT_COMPLETE",
        "created_utc": now(),
        "created_from_existing_files_only": True,
        "no_model_loading_training_evaluation_gpu_upload_or_leaderboard": True,
        "sources": {
            "entity_contrasts": rel(ENTITY_CONTRASTS),
            "entity_balanced": rel(ENTITY_BALANCED),
            "binding_contrasts": rel(BINDING_CONTRASTS),
            "binding_window": rel(BINDING_WINDOW),
            "broad_vector": rel(BROAD_VECTOR),
            "conservation_summary": rel(CONSERVATION_SUMMARY),
        },
        "entity_group_summary": entity_group_summary,
        "entity_identity_residuals": residual_rows,
        "binding_late_summary": binding_late_summary,
        "binding_window_rows_from_probe": binding_window,
        "visible_broad_vectors": visible,
        "scientific_reading": reading,
        "files": files,
    }
    write_csv(OUT / "entity_group_late_summary.csv", entity_group_summary)
    write_csv(OUT / "entity_identity_residuals.csv", residual_rows)
    write_csv(OUT / "binding_late_summary.csv", binding_late_summary)
    # Flatten vector rows for CSV readability.
    flat_vec = []
    for r in visible:
        rr = {k: v for k, v in r.items() if k != "vector"}
        for fam, val in r.get("vector", {}).items():
            rr[f"delta_{fam}"] = val
        flat_vec.append(rr)
    write_csv(OUT / "visible_broad_vectors.csv", flat_vec)
    (OUT / "macro_entity_vb_readout_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload, OUT / "macro_entity_vb_readout_summary.md")
    print(json.dumps({
        "status": payload["status"],
        "summary_md": files["summary_md"],
        "official_entity_VB_zero_pp": vb_z,
        "official_entity_VB_nonzero_pp": vb_nz,
        "binding_VB_affected_delta": vb_aff,
        "binding_VB_unaffected_delta": vb_unaff,
        "binding_VB_EEBF_delta": vb_eebf,
        "no_model_loading_training_evaluation_gpu_upload_or_leaderboard": True,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
